import torch
from ecg_pytorch.data_reader.ecg_dataset_lstm import ToTensor, EcgHearBeatsDataset, EcgHearBeatsDatasetTest
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from ecg_pytorch.classifiers.models import lstm
import time


def init_weights(m):
    if type(m) == nn.LSTM:
       torch.nn.init.xavier_uniform(m.weight_hh_l0.data)
       # torch.nn.init.xavier_uniform(m.weight_hh_l1.data)
       torch.nn.init.xavier_uniform(m.weight_ih_l0.data)
       # torch.nn.init.xavier_uniform(m.weight_ih_l1.data)
       # torch.nn.init.xavier_uniform(m.bias_hh_l0.data)
       # torch.nn.init.xavier_uniform(m.bias_hh_l1.data)
       # torch.nn.init.xavier_uniform(m.bias_ih_l0.data)
       # torch.nn.init.xavier_uniform(m.bias_ih_l1.data)


def plt_roc_curve(y_true, y_pred, classes, writer, total_iters):
    """

    :param y_true:[[1,0,0,0,0], [0,1,0,0], [1,0,0,0,0],...]
    :param y_pred: [0.34,0.2,0.1] , 0.2,...]
    :param classes:5
    :return:
    """
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        fig = plt.figure()
        lw = 2
        plt.plot(fpr[classes[i]], tpr[classes[i]], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[classes[i]])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic beat {}'.format(classes[i]))
        plt.legend(loc="lower right")
        writer.add_figure('test/roc_curve_beat_{}'.format(classes[i]), fig, total_iters)
        plt.close()
        fig.clf()
        fig.clear()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


def train_classifier(net, batch_size, model_dir):
    """

    :param network:
    :return:
    """

    composed = transforms.Compose([ToTensor()])
    dataset = EcgHearBeatsDataset(transform=composed)
    weights_for_balance = dataset.make_weights_for_balanced_classes()
    weights_for_balance = torch.DoubleTensor(weights_for_balance)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights_for_balance,
        num_samples=len(weights_for_balance),
        replacement=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=1, sampler=sampler)
    # weights = dataset.weights_per_class()
    # # weights[4] = 0
    # weights = torch.Tensor(weights)
    # print(weights)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                           num_workers=1, shuffle=True)

    testset = EcgHearBeatsDatasetTest(transform=composed)
    testdataloader = torch.utils.data.DataLoader(testset, batch_size=300,
                                             shuffle=True, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.01)
    writer = SummaryWriter(model_dir)
    total_iters = 0
    for epoch in range(2):  # loop over the dataset multiple times

        for i, data in enumerate(dataloader):
            total_iters += 1
            # get the inputs
            ecg_batch = data['cardiac_cycle'].permute(1, 0, 2).float()
            b_size = ecg_batch.shape[1]
            labels = data['label']
            labels_class = torch.max(labels, 1)[1]
            # zero the parameter gradients
            net.zero_grad()

            # forward + backward + optimize
            outputs = net(ecg_batch)
            # print("output from network: {}".format(outputs.data.numpy()))
            outputs_class = torch.max(outputs, 1)[1]
            # print("output class from network: {}".format(outputs_class.data.numpy()))
            accuracy = (outputs_class == labels_class).sum().float() / b_size
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            writer.add_scalars('Cross entropy loss', {'Train batches loss': loss.item()}, total_iters)
            writer.add_scalars('Accuracy', {'Train batches accuracy': accuracy.item()}, total_iters)
            # print statistics
            print("Epoch {}. Iteration {}.\t Batch loss = {:.2f}. Accuracy = {:.2f}".format(epoch + 1, i, loss.item(),
                                                                                            accuracy.item()))
            if i % 50 == 0:
                fig, _ = plot_confusion_matrix(labels_class.numpy(), outputs_class.numpy(), np.array(['N', 'S', 'V', 'F', 'Q']))
                # fig, _ = plot_confusion_matrix(labels_class.numpy(), outputs_class.numpy(),
                #                                np.array(['N', 'Others']))
                writer.add_figure('train/confusion_matrix', fig, total_iters)

                grad_norm = get_gradient_norm_l2(net)
                writer.add_scalar('gradients_norm', grad_norm, total_iters)
                print("Norm of gradients = {}.".format(grad_norm))

            if i % 1000 == 0:
                with torch.no_grad():
                    labels_total_one_hot = np.array([]).reshape((0, 5))
                    outputs_preds = np.array([]).reshape((0, 5))
                    # labels_total_one_hot = np.array([]).reshape((0, 2))
                    # outputs_preds = np.array([]).reshape((0, 2))
                    labels_total = np.array([])
                    outputs_total = np.array([])
                    loss_hist = []
                    for _, test_data in enumerate(testdataloader):
                        start = time.time()
                        ecg_batch = test_data['cardiac_cycle'].permute(1, 0, 2).float()
                        labels = test_data['label']

                        labels_class = torch.max(labels, 1)[1]
                        outputs = net(ecg_batch)
                        loss = criterion(outputs, torch.max(labels, 1)[1])
                        loss_hist.append(loss.item())
                        outputs_class = torch.max(outputs, 1)[1]
                        end = time.time()
                        print("time took {}".format(end - start))

                        labels_total_one_hot = np.concatenate((labels_total_one_hot, labels.numpy()))
                        labels_total = np.concatenate((labels_total, labels_class.numpy()))
                        outputs_total = np.concatenate((outputs_total, outputs_class.numpy()))
                        outputs_preds = np.concatenate((outputs_preds, outputs.numpy()))

                    outputs_total = outputs_total.astype(int)
                    labels_total = labels_total.astype(int)
                    fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                                                   np.array(['N', 'S', 'V', 'F', 'Q']))
                    # fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                    #                                np.array(['N', 'Other']))
                    # Accuracy and Loss:
                    accuracy = sum((outputs_total == labels_total)) / len(outputs_total)
                    writer.add_scalars('Accuracy', {'Test set accuracy': accuracy}, global_step=total_iters)
                    writer.add_figure('test/confusion_matrix', fig, total_iters)
                    loss = sum(loss_hist) / len(loss_hist)
                    writer.add_scalars('Cross entropy loss', {'Test set loss': loss}, total_iters)
                    plt_roc_curve(labels_total_one_hot, outputs_preds, np.array(['N', 'S', 'V', 'F', 'Q']), writer, total_iters)
                    # plt_roc_curve(labels_total_one_hot, outputs_preds, np.array(['N', 'Others']), writer,
                    #               total_iters)
    writer.close()


def get_gradient_norm_l2(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


if __name__ == "__main__":
    net = lstm.ECGLSTM(5, 512,  5, 2)
    # net.apply(init_weights)
    train_classifier(net, 50, model_dir='tensorboard/lstm')