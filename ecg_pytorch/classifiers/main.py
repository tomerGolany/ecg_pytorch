import torchvision.transforms as transforms
from ecg_pytorch.data_reader import ecg_dataset
import torch.optim as optim
import torch.nn as nn
import torch
from ecg_pytorch.classifiers.models import cnn
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import numpy as np
from torch.utils.data.sampler import Sampler
from sklearn.metrics import roc_curve, auc
from ecg_pytorch.classifiers.models import fully_connected
from ecg_pytorch.gan_models.models import dcgan
from ecg_pytorch.gan_models.models import ode_gan_aaai
import shutil
import os
import logging
from ecg_pytorch.gan_models.models.old_ode_combined import CombinedGenerator as CG
from ecg_pytorch.gan_models import checkpoint_paths

AUC_VALUE = 0


def plt_roc_curve(y_true, y_pred, classes, writer, total_iters):
    """

    :param y_true:[[1,0,0,0,0], [0,1,0,0], [1,0,0,0,0],...]
    :param y_pred: [0.34,0.2,0.1] , 0.2,...]
    :param classes:5
    :return:
    """
    global AUC_VALUE
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        if i == 0:
            curr_auc = roc_auc[classes[i]]
            if curr_auc > AUC_VALUE:
                logging.info("New maxumal AUC: {}".format(AUC_VALUE))
                AUC_VALUE = curr_auc
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


def train_classifier(net, batch_size, model_dir, n=0, ch_path=None, beat_type=None):
    """

    :param network:
    :return:
    """
    global AUC_VALUE
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    composed = transforms.Compose([ecg_dataset.ToTensor()])
    dataset = ecg_dataset.EcgHearBeatsDataset(transform=composed)
    # gNet = CG(0, 'cpu')
    if n != 0:
        gNet = ode_gan_aaai.DCGenerator(0)

        dataset.add_beats_from_generator(gNet, n, ch_path, beat_type)
    # gNet = dcgan.DCGenerator(0)
    # checkpoint_path = '/Users/tomer.golany/PycharmProjects/ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_V_beat/' \
    #                   'checkpoint_epoch_22_iters_1563'

    # dataset.add_beats_from_generator(gNet, 15000,
    #                                      checkpoint_path,
    #                                      'V')
    # weights_for_balance = dataset.make_weights_for_balanced_classes()
    # weights_for_balance = torch.DoubleTensor(weights_for_balance)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(
    #     weights=weights_for_balance,
    #     num_samples=len(weights_for_balance),
    #     replacement=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=1, shuffle=True)

    testset = ecg_dataset.EcgHearBeatsDatasetTest(transform=composed)
    testdataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    writer = SummaryWriter(model_dir)
    total_iters = 0
    for epoch in range(4):  # loop over the dataset multiple times

        for i, data in enumerate(dataloader):
            total_iters += 1
            # get the inputs
            ecg_batch = data['cardiac_cycle'].view(-1, 1, 216).float()
            # ecg_batch = data['cardiac_cycle'].float()
            b_size = ecg_batch.shape[0]
            labels = data['label']
            labels_class = torch.max(labels, 1)[1]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(ecg_batch)
            outputs_class = torch.max(outputs, 1)[1]

            accuracy = (outputs_class == labels_class).sum().float() / b_size
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            writer.add_scalars('Cross_entropy_loss', {'Train batches loss': loss.item()}, total_iters)
            writer.add_scalars('Accuracy', {'Train batches accuracy': accuracy.item()}, total_iters)
            # print statistics
            print("Epoch {}. Iteration {}.\t Batch loss = {:.2f}. Accuracy = {:.2f}".format(epoch + 1, i, loss.item(),
                                                                                            accuracy.item()))
            if total_iters % 50 == 0:
                fig, _ = plot_confusion_matrix(labels_class.numpy(), outputs_class.numpy(), np.array(['N', 'S', 'V', 'F', 'Q']))
                writer.add_figure('train/confusion_matrix', fig, total_iters)

            if total_iters % 200 == 0:
                with torch.no_grad():
                    labels_total_one_hot = np.array([]).reshape((0, 5))
                    outputs_preds = np.array([]).reshape((0, 5))
                    labels_total = np.array([])
                    outputs_total = np.array([])
                    loss_hist = []
                    for _, test_data in enumerate(testdataloader):
                        ecg_batch = test_data['cardiac_cycle'].view(-1, 1, 216).float()
                        # ecg_batch = test_data['cardiac_cycle'].float()
                        labels = test_data['label']
                        labels_class = torch.max(labels, 1)[1]
                        outputs = net(ecg_batch)
                        loss = criterion(outputs, torch.max(labels, 1)[1])
                        loss_hist.append(loss.item())
                        outputs_class = torch.max(outputs, 1)[1]

                        labels_total_one_hot = np.concatenate((labels_total_one_hot, labels.numpy()))
                        labels_total = np.concatenate((labels_total, labels_class.numpy()))
                        outputs_total = np.concatenate((outputs_total, outputs_class.numpy()))
                        outputs_preds = np.concatenate((outputs_preds, outputs.numpy()))

                    outputs_total = outputs_total.astype(int)
                    labels_total = labels_total.astype(int)
                    fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                                                   np.array(['N', 'S', 'V', 'F', 'Q']))
                    # Accuracy and Loss:
                    accuracy = sum((outputs_total == labels_total)) / len(outputs_total)
                    writer.add_scalars('Accuracy', {'Test set accuracy': accuracy}, global_step=total_iters)
                    writer.add_figure('test/confusion_matrix', fig, total_iters)
                    loss = sum(loss_hist) / len(loss_hist)
                    writer.add_scalars('Cross_entropy_loss', {'Test set loss': loss}, total_iters)
                    plt_roc_curve(labels_total_one_hot, outputs_preds, np.array(['N', 'S', 'V', 'F', 'Q']), writer, total_iters)

    torch.save({
        'net': net.state_dict()
    }, model_dir + '/checkpoint_epoch_iters_{}'.format(total_iters))
    writer.close()


def train_mult():
    # model_dir = 'tensorboard/N/add_{}_fc_ode_gan/'
    ch_ppath = checkpoint_paths.ODE_GAN_N_CHK
    beat_type = 'N'

    with open('N_fc_ode_res.txt', 'w') as fd:
        for n in [500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]:
            num_runs = 0
            AUC_VALUE = 0
            while num_runs < 10:
                net = fully_connected.FF()
                logging.info("AUC VALUE: {}".format(AUC_VALUE))
                model_dir = 'tensorboard/N/add_{}_fc_ode_gan/'.format(str(n))
                train_classifier(net, 50, model_dir=model_dir, n=n, ch_path=ch_ppath, beat_type=beat_type)
                num_runs += 1
            logging.info("Best AUC VALUE: {}".format(AUC_VALUE))
            w = "{} : {}".format(str(n), str(AUC_VALUE))
            fd.write(w)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    net = cnn.Net()
    # net = fully_connected.FF()
    train_classifier(net, 50, model_dir='tensorboard/cnn_with_chk')
    # train_mult()