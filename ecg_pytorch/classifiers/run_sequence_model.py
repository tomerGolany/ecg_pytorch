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
from ecg_pytorch.train_configs import ECGTrainConfig, GeneratorAdditionalDataConfig
from ecg_pytorch.gan_models.models import dcgan
import logging
import time
import os
import shutil
from ecg_pytorch.gan_models import checkpoint_paths

BEST_AUC_N = 0
BEST_AUC_S = 0
BEST_AUC_V = 0
BEST_AUC_F = 0
BEST_AUC_Q = 0

base_local = '/Users/tomer.golany/PycharmProjects/'
base_remote = '/home/tomer.golany@st.technion.ac.il/'


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
    global BEST_AUC_N
    global BEST_AUC_S
    global BEST_AUC_V
    global BEST_AUC_F
    global BEST_AUC_Q
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])

        if i == 0 and roc_auc[classes[i]] > BEST_AUC_N:
            BEST_AUC_N = roc_auc[classes[i]]
        if i == 1 and roc_auc[classes[i]] > BEST_AUC_S:
            BEST_AUC_S = roc_auc[classes[i]]
        if i == 2 and roc_auc[classes[i]] > BEST_AUC_V:
            BEST_AUC_V = roc_auc[classes[i]]
        if i == 3 and roc_auc[classes[i]] > BEST_AUC_F:
            BEST_AUC_F = roc_auc[classes[i]]
        if i == 4 and roc_auc[classes[i]] > BEST_AUC_Q:
            BEST_AUC_Q = roc_auc[classes[i]]

        logging.info("Best AUC:\n N: {}\tS: {}\tV: {}\tF: {}\tQ: {}".format(BEST_AUC_N, BEST_AUC_S,
                                                                            BEST_AUC_V, BEST_AUC_F,
                                                                            BEST_AUC_Q))

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


def train_classifier(net, model_dir, train_config=None):
    """

    :param network:
    :return:
    """
    batch_size = train_config.batch_size
    num_of_epochs = train_config.num_epochs
    lr = train_config.lr
    device = train_config.device
    add_from_gan = train_config.add_data_from_gan

    composed = transforms.Compose([ToTensor()])
    dataset = EcgHearBeatsDataset(transform=composed)
    num_examples_to_add = train_config.generator_details.num_examples_to_add
    generator_beat_type = train_config.generator_details.beat_type
    dataset.add_noise(num_examples_to_add, generator_beat_type)
    #
    # Check if to add data from GAN #
    #
    if add_from_gan:
        num_examples_to_add = train_config.generator_details.num_examples_to_add
        generator_checkpoint_path = train_config.generator_details.checkpoint_path
        generator_beat_type = train_config.generator_details.beat_type
        gNet = dcgan.DCGenerator(0)
        dataset.add_beats_from_generator(gNet, num_examples_to_add,
                                         generator_checkpoint_path,
                                         generator_beat_type)

    if train_config.weighted_sampling:
        weights_for_balance = dataset.make_weights_for_balanced_classes()
        weights_for_balance = torch.DoubleTensor(weights_for_balance)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=weights_for_balance,
            num_samples=len(weights_for_balance),
            replacement=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 num_workers=1, sampler=sampler)
    else:
        # weights = dataset.weights_per_class()
        # # weights[4] = 0
        # weights = torch.Tensor(weights)
        # print(weights)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 num_workers=1, shuffle=True)

    testset = EcgHearBeatsDatasetTest(transform=composed)
    testdataloader = torch.utils.data.DataLoader(testset, batch_size=300,
                                                 shuffle=True, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter(model_dir, max_queue=1000)
    total_iters = 0
    num_iters_per_epoch = int(np.floor(len(dataset) / batch_size) + batch_size)
    for epoch in range(num_of_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            total_iters += 1
            # get the inputs
            ecg_batch = data['cardiac_cycle'].float().to(device)
            b_size = ecg_batch.shape[0]
            labels = data['label'].to(device)
            labels_class = torch.max(labels, 1)[1]
            labels_str = data['beat_type']
            logging.debug("batch labels: {}".format(labels_str))
            # zero the parameter gradients
            net.zero_grad()

            # forward + backward + optimize
            outputs = net(ecg_batch)
            outputs_class = torch.max(outputs, 1)[1]
            accuracy = (outputs_class == labels_class).sum().float() / b_size
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            writer.add_scalars('cross_entropy_loss', {'Train batches loss': loss.item()}, total_iters)
            writer.add_scalars('accuracy', {'Train batches accuracy': accuracy.item()}, total_iters)
            # print statistics
            logging.info("Epoch {}. Iteration {}/{}.\t Batch loss = {:.2f}. Accuracy = {:.2f}".format(epoch + 1, i,
                                                                                                      num_iters_per_epoch,
                                                                                                      loss.item(),
                                                                                                      accuracy.item()))
            if total_iters % 50 == 0:
                fig, _ = plot_confusion_matrix(labels_class.cpu().numpy(), outputs_class.cpu().numpy(),
                                               np.array(['N', 'S', 'V', 'F', 'Q']))
                # fig, _ = plot_confusion_matrix(labels_class.numpy(), outputs_class.numpy(),
                #                                np.array(['N', 'Others']))
                writer.add_figure('train/confusion_matrix', fig, total_iters)

                grad_norm = get_gradient_norm_l2(net)
                writer.add_scalar('gradients_norm', grad_norm, total_iters)
                logging.info("Norm of gradients = {}.".format(grad_norm))

            if total_iters % 200 == 0:
                with torch.no_grad():
                    labels_total_one_hot = np.array([]).reshape((0, 5))
                    outputs_preds = np.array([]).reshape((0, 5))
                    # labels_total_one_hot = np.array([]).reshape((0, 2))
                    # outputs_preds = np.array([]).reshape((0, 2))
                    labels_total = np.array([])
                    outputs_total = np.array([])
                    loss_hist = []
                    start = time.time()
                    for _, test_data in enumerate(testdataloader):
                        ecg_batch = test_data['cardiac_cycle'].float().to(device)
                        labels = test_data['label'].to(device)

                        labels_class = torch.max(labels, 1)[1]
                        outputs = net(ecg_batch)
                        loss = criterion(outputs, torch.max(labels, 1)[1])
                        loss_hist.append(loss.item())
                        outputs_class = torch.max(outputs, 1)[1]

                        labels_total_one_hot = np.concatenate((labels_total_one_hot, labels.cpu().numpy()))
                        labels_total = np.concatenate((labels_total, labels_class.cpu().numpy()))
                        outputs_total = np.concatenate((outputs_total, outputs_class.cpu().numpy()))
                        outputs_preds = np.concatenate((outputs_preds, outputs.cpu().numpy()))
                    end = time.time()
                    print("Test evaluation took {:.2f} seconds".format(end - start))
                    outputs_total = outputs_total.astype(int)
                    labels_total = labels_total.astype(int)
                    fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                                                   np.array(['N', 'S', 'V', 'F', 'Q']))
                    # fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                    #                                np.array(['N', 'Other']))
                    # Accuracy and Loss:
                    accuracy = sum((outputs_total == labels_total)) / len(outputs_total)
                    writer.add_scalars('accuracy', {'Test set accuracy': accuracy}, global_step=total_iters)
                    writer.add_figure('test/confusion_matrix', fig, total_iters)
                    loss = sum(loss_hist) / len(loss_hist)
                    writer.add_scalars('cross_entropy_loss', {'Test set loss': loss}, total_iters)
                    plt_roc_curve(labels_total_one_hot, outputs_preds, np.array(['N', 'S', 'V', 'F', 'Q']), writer,
                                  total_iters)
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


def train_mult():
    beat_type = 'N'
    global BEST_AUC_N
    global BEST_AUC_S
    global BEST_AUC_V
    global BEST_AUC_F
    global BEST_AUC_Q
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Device: {}".format(device))
    if beat_type == 'N':
        ck_path = checkpoint_paths.DCGAN_N_CHK
    elif beat_type == 'S':
        ck_path = checkpoint_paths.DCGAN_S_CHK
    elif beat_type == 'V':
        ck_path = checkpoint_paths.DCGAN_V_CHK
    elif beat_type == 'F':
        ck_path = checkpoint_paths.DCGAN_F_CHK
    else:
        raise ValueError("Bad beat type")
    with open('res_{}.text'.format(beat_type), 'w') as fd:
        for n in [500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]:
        # for n in [1500, 3000, 5000, 7000, 10000, 15000]:
            model_dir = base_local + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/{}/lstm_add_{}/'.format(beat_type, str(n))
            gen_details = GeneratorAdditionalDataConfig(beat_type=beat_type, checkpoint_path=ck_path, num_examples_to_add=n)
            train_config = ECGTrainConfig(num_epochs=5, batch_size=20, lr=0.0002, weighted_loss=False,
                                          weighted_sampling=True,
                                          device=device, add_data_from_gan=True, generator_details=gen_details)

            total_runs = 0
            BEST_AUC_N = 0
            BEST_AUC_S = 0
            BEST_AUC_V = 0
            BEST_AUC_F = 0
            BEST_AUC_Q = 0
            # while BEST_AUC_S <= 0.876:
            while total_runs < 10:
                if os.path.isdir(model_dir):
                    logging.info("Removing model dir")
                    shutil.rmtree(model_dir)
                net = lstm.ECGLSTM(5, 512, 5, 2).to(device)
                train_classifier(net, model_dir=model_dir, train_config=train_config)
                w_write = "AUC on RUN {}: \n N: {}\tS: {}\tV: {}\tF: {}\tQ: {}".format(total_runs, BEST_AUC_N, BEST_AUC_S,
                                                                                BEST_AUC_V, BEST_AUC_F,
                                                                                BEST_AUC_Q)
                fd.write(w_write)
                total_runs += 1
            logging.info("Done after {} runs.".format(total_runs))
            logging.info("Best AUC:\n N: {}\tS: {}\tV: {}\tF: {}\tQ: {}".format(BEST_AUC_N, BEST_AUC_S,
                                                                                BEST_AUC_V, BEST_AUC_F,
                                                                                BEST_AUC_Q))
            w = "#n: {} .Best AUC:\n N: {}\tS: {}\tV: {}\tF: {}\tQ: {}\n".format(n, BEST_AUC_N, BEST_AUC_S,
                                                                                BEST_AUC_V, BEST_AUC_F,
                                                                                BEST_AUC_Q)
            fd.write(w)


def train_with_noise():
    beat_type = 'N'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('res_noise_{}.text'.format(beat_type), 'w') as fd:
        for n in [500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]:
            model_dir = '/home/tomer.golany@st.technion.ac.il/ecg_pytorch/ecg_pytorch/classifiers/tensorboard/noise_{}/lstm_add_{}/'.format(
                str(n), beat_type)

            total_runs = 0
            BEST_AUC_N = 0
            BEST_AUC_S = 0
            BEST_AUC_V = 0
            BEST_AUC_F = 0
            BEST_AUC_Q = 0
            # while BEST_AUC_S <= 0.876:
            while total_runs < 10:
                if os.path.isdir(model_dir):
                    logging.info("Removing model dir")
                    shutil.rmtree(model_dir)
                net = lstm.ECGLSTM(5, 512, 5, 2).to(device)
                gen_details = GeneratorAdditionalDataConfig(beat_type=beat_type, checkpoint_path='',
                                                            num_examples_to_add=n)
                train_config = ECGTrainConfig(num_epochs=4, batch_size=16, lr=0.002, weighted_loss=False,
                                              weighted_sampling=True,
                                              device=device, add_data_from_gan=False, generator_details=gen_details)
                train_classifier(net, model_dir=model_dir, train_config=train_config)
                total_runs += 1
            logging.info("Done after {} runs.".format(total_runs))
            logging.info("Best AUC:\n N: {}\tS: {}\tV: {}\tF: {}\tQ: {}".format(BEST_AUC_N, BEST_AUC_S,
                                                                                BEST_AUC_V, BEST_AUC_F,
                                                                                BEST_AUC_Q))
            w = "#n: {} .Best AUC:\n N: {}\tS: {}\tV: {}\tF: {}\tQ: {}\n".format(n, BEST_AUC_N, BEST_AUC_S,
                                                                                 BEST_AUC_V, BEST_AUC_F,
                                                                                 BEST_AUC_Q)
            fd.write(w)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # logging.info("Device: {}".format(device))
    #
    # net = lstm.ECGLSTM(5, 512, 5, 2).to(device)
    # train_config = ECGTrainConfig(num_epochs=4, batch_size=16, lr=0.002, weighted_loss=False, weighted_sampling=True,
    #                               device=device, add_data_from_gan=False, generator_details=None)
    # train_classifier(net, model_dir='tensorboard/lstm_adam_weighted', train_config=train_config)
    train_mult()
    # train_with_noise()
