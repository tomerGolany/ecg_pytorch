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
from ecg_pytorch.gan_models.models import ode_gan_aaai
import shutil
from ecg_pytorch.gan_models import checkpoint_paths
import pickle
from ecg_pytorch import train_configs

base_path = train_configs.base

BEAT_TO_INDEX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}


def init_weights(m):
    if type(m) == nn.LSTM:
        torch.nn.init.xavier_uniform(m.weight_hh_l0.data)

        torch.nn.init.xavier_uniform(m.weight_ih_l0.data)


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
    roc_auc_res = []
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        roc_auc_res.append(roc_auc[classes[i]])
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
    return roc_auc_res


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
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

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
    #
    # Get configs:
    #
    batch_size = train_config.batch_size
    num_of_epochs = train_config.num_epochs
    lr = train_config.lr
    device = train_config.device
    add_from_gan = train_config.add_data_from_gan
    best_auc_scores = [0, 0, 0, 0]

    composed = transforms.Compose([ToTensor()])
    if train_config.train_one_vs_all:
        dataset = EcgHearBeatsDataset(transform=composed, beat_type=train_config.generator_details.beat_type,
                                      one_vs_all=True)
        testset = EcgHearBeatsDatasetTest(transform=composed, beat_type=train_config.generator_details.beat_type,
                                          one_vs_all=True)
    else:
        dataset = EcgHearBeatsDataset(transform=composed)
        testset = EcgHearBeatsDatasetTest(transform=composed)

    testdataloader = torch.utils.data.DataLoader(testset, batch_size=300,
                                                 shuffle=True, num_workers=1)

    num_examples_to_add = train_config.generator_details.num_examples_to_add
    generator_beat_type = train_config.generator_details.beat_type
    dataset.add_noise(num_examples_to_add, generator_beat_type)

    #
    # Check if to add data from GAN:
    #
    if add_from_gan:

        num_examples_to_add = train_config.generator_details.num_examples_to_add
        generator_checkpoint_path = train_config.generator_details.checkpoint_path
        generator_beat_type = train_config.generator_details.beat_type
        gan_type = train_config.generator_details.gan_type
        logging.info("Adding {} samples of type {} from GAN {}".format(num_examples_to_add, generator_beat_type,
                                                                       gan_type))
        logging.info("Size of training data before additional data from GAN: {}".format(len(dataset)))
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(dataset.len_beat('N'), dataset.len_beat('S'),
                                                                  dataset.len_beat('V'), dataset.len_beat('F')))
        if num_examples_to_add > 0:
            if gan_type == 'DCGAN':
                gNet = dcgan.DCGenerator(0)
                dataset.add_beats_from_generator(gNet, num_examples_to_add,
                                                 generator_checkpoint_path,
                                                 generator_beat_type)
            elif gan_type == 'ODE_GAN':
                gNet = ode_gan_aaai.DCGenerator(0)
                dataset.add_beats_from_generator(gNet, num_examples_to_add,
                                                 generator_checkpoint_path,
                                                 generator_beat_type)
            elif gan_type == 'SIMULATOR':
                dataset.add_beats_from_simulator(num_examples_to_add, generator_beat_type)
            else:
                raise ValueError("Unknown gan type {}".format(gan_type))

        logging.info("Size of training data after additional data from GAN: {}".format(len(dataset)))
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(dataset.len_beat('N'), dataset.len_beat('S'),
                                                                  dataset.len_beat('V'), dataset.len_beat('F')))

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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 num_workers=1, shuffle=True)

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

            #
            # print statistics
            #
            if total_iters % 1000 == 0:
                logging.info("Epoch {}. Iteration {}/{}.\t Batch loss = {:.2f}. Accuracy = {:.2f}".format(epoch + 1, i,
                                                                                                          num_iters_per_epoch,
                                                                                                          loss.item(),
                                                                                                          accuracy.item()))
            if total_iters % 3000 == 0:
                if train_config.train_one_vs_all:
                    fig, _ = plot_confusion_matrix(labels_class.numpy(), outputs_class.numpy(),
                                                   np.array([generator_beat_type, 'Others']))
                else:
                    fig, _ = plot_confusion_matrix(labels_class.cpu().numpy(), outputs_class.cpu().numpy(),
                                                   np.array(['N', 'S', 'V', 'F', 'Q']))

                writer.add_figure('train/confusion_matrix', fig, total_iters)

                grad_norm = get_gradient_norm_l2(net)
                writer.add_scalar('gradients_norm', grad_norm, total_iters)
                logging.info("Norm of gradients = {}.".format(grad_norm))

            if total_iters % 200 == 0:
                with torch.no_grad():

                    if train_config.train_one_vs_all:
                        labels_total_one_hot = np.array([]).reshape((0, 2))
                        outputs_preds = np.array([]).reshape((0, 2))
                    else:
                        labels_total_one_hot = np.array([]).reshape((0, 5))
                        outputs_preds = np.array([]).reshape((0, 5))

                    labels_total = np.array([])
                    outputs_total = np.array([])
                    loss_hist = []
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

                    outputs_total = outputs_total.astype(int)
                    labels_total = labels_total.astype(int)

                    if train_config.train_one_vs_all:
                        fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                                                       np.array([generator_beat_type, 'Other']))
                    else:
                        fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                                                       np.array(['N', 'S', 'V', 'F', 'Q']))

                    # Accuracy and Loss:
                    accuracy = sum((outputs_total == labels_total)) / len(outputs_total)
                    writer.add_scalars('accuracy', {'Test set accuracy': accuracy}, global_step=total_iters)
                    writer.add_figure('test/confusion_matrix', fig, total_iters)
                    loss = sum(loss_hist) / len(loss_hist)
                    writer.add_scalars('cross_entropy_loss', {'Test set loss': loss}, total_iters)

                    #
                    # Check AUC values:
                    #
                    if train_config.train_one_vs_all:
                        auc_roc = plt_roc_curve(labels_total_one_hot, outputs_preds,
                                                np.array([generator_beat_type, 'Other']),
                                                writer,
                                                total_iters)
                        for i_auc in range(2):
                            if auc_roc[i_auc] > best_auc_scores[i_auc]:
                                best_auc_scores[i_auc] = auc_roc[i_auc]

                    else:
                        auc_roc = plt_roc_curve(labels_total_one_hot, outputs_preds,
                                                np.array(['N', 'S', 'V', 'F', 'Q']),
                                                writer,
                                                total_iters)
                        for i_auc in range(4):
                            if auc_roc[i_auc] > best_auc_scores[i_auc]:
                                best_auc_scores[i_auc] = auc_roc[i_auc]
    writer.close()
    return best_auc_scores


def get_gradient_norm_l2(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def train_mult(beat_type, gan_type, device):
    summary_model_dir = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/{}/lstm_{}_summary/'.format(
        beat_type, gan_type)
    writer = SummaryWriter(summary_model_dir)
    #
    # Retrieve Checkpoint path:
    #
    if gan_type in ['DCGAN', 'ODE_GAN']:
        ck_path = checkpoint_paths.BEAT_AND_MODEL_TO_CHECKPOINT_PATH[beat_type][gan_type]
    else:
        ck_path = None

    #
    # Define summary values:
    #
    mean_auc_values = []
    var_auc_values = []
    best_auc_values = []

    best_auc_for_each_n = {}
    #
    # Run with different number of additional data from trained generator:
    #
    for n in [500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]:
        # for n in [5000]:
        #
        # Train configurations:
        #
        model_dir = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/{}/lstm_{}_{}/'.format(
            beat_type, str(n), gan_type)
        gen_details = GeneratorAdditionalDataConfig(beat_type=beat_type, checkpoint_path=ck_path,
                                                    num_examples_to_add=n, gan_type=gan_type)
        train_config = ECGTrainConfig(num_epochs=5, batch_size=20, lr=0.0002, weighted_loss=False,
                                      weighted_sampling=True,
                                      device=device, add_data_from_gan=True, generator_details=gen_details,
                                      train_one_vs_all=False)
        #
        # Run 10 times each configuration:
        #
        total_runs = 0
        best_auc_per_run = []
        while total_runs < 10:
            if os.path.isdir(model_dir):
                logging.info("Removing model dir")
                shutil.rmtree(model_dir)
            #
            # Initialize the network each run:
            #
            net = lstm.ECGLSTM(5, 512, 5, 2).to(device)

            #
            # Train the classifier:
            #
            best_auc_scores = train_classifier(net, model_dir=model_dir, train_config=train_config)
            best_auc_per_run.append(best_auc_scores[BEAT_TO_INDEX[beat_type]])
            writer.add_scalar('auc_with_additional_{}_beats'.format(n), best_auc_scores[BEAT_TO_INDEX[beat_type]],
                              total_runs)
            if best_auc_scores[BEAT_TO_INDEX[beat_type]] >= 0.88:
                logging.info("Found desired AUC: {}".format(best_auc_scores))
                break
            total_runs += 1
        best_auc_for_each_n[n] = best_auc_per_run
        mean_auc_values.append(np.mean(best_auc_per_run))
        var_auc_values.append(np.var(best_auc_per_run))
        best_auc_values.append(max(best_auc_per_run))
        writer.add_scalar('mean_auc', np.mean(best_auc_per_run), n)
        writer.add_scalar('max_auc', max(best_auc_per_run), n)
    writer.close()
    #
    # Save data in pickle:
    #
    all_results = {'best_auc_for_each_n': best_auc_for_each_n, 'mean': mean_auc_values, 'var': var_auc_values,
                   'best': best_auc_values}
    pickle_file_path = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/pickles_results/{}_{}_lstm.pkl'.format(
        beat_type, gan_type)
    with open(pickle_file_path, 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_optimal_checkpoint(chk_dir, beat_type, gan_type, device, num_samples_to_add):
    model_dir = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/{}/find_optimal_chk_{}_{}_agg/' \
        .format(beat_type, str(num_samples_to_add), gan_type)

    writer = SummaryWriter(model_dir)
    if not os.path.isdir(chk_dir):
        raise ValueError("{} not a directory".format(chk_dir))

    #
    # Define summary values:
    #
    mean_auc_values = []
    best_auc_values = []
    final_dict = {}
    for i, chk_name in enumerate(os.listdir(chk_dir)):
        if chk_name.startswith('checkpoint'):
            chk_path = os.path.join(chk_dir, chk_name)

            #
            # Train configurations:
            #
            model_dir = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/{}/lstm_{}_{}_{}/'.format(
                beat_type, str(num_samples_to_add), gan_type, chk_name)
            gen_details = GeneratorAdditionalDataConfig(beat_type=beat_type, checkpoint_path=chk_path,
                                                        num_examples_to_add=num_samples_to_add, gan_type=gan_type)
            train_config = ECGTrainConfig(num_epochs=5, batch_size=20, lr=0.0002, weighted_loss=False,
                                          weighted_sampling=True,
                                          device=device, add_data_from_gan=True, generator_details=gen_details,
                                          train_one_vs_all=False)
            #
            # Run 10 times each configuration:
            #
            total_runs = 0
            best_auc_per_run = []
            while total_runs < 10:
                if os.path.isdir(model_dir):
                    logging.info("Removing model dir")
                    shutil.rmtree(model_dir)
                #
                # Initialize the network each run:
                #
                net = lstm.ECGLSTM(5, 512, 5, 2).to(device)

                #
                # Train the classifier:
                #
                best_auc_scores = train_classifier(net, model_dir=model_dir, train_config=train_config)
                best_auc_per_run.append(best_auc_scores[BEAT_TO_INDEX[beat_type]])
                writer.add_scalar('best_auc_{}'.format(chk_name), best_auc_scores[BEAT_TO_INDEX[beat_type]], total_runs)
                total_runs += 1
            mean_auc = np.mean(best_auc_per_run)
            max_auc = max(best_auc_per_run)
            logging.info("Checkpoint {}: Mean AUC {}. Max AUC: {}".format(chk_name, mean_auc, max_auc))
            mean_auc_values.append(mean_auc)
            best_auc_values.append(max_auc)
            final_dict[chk_name] = {}
            final_dict[chk_name]['MEAN'] = mean_auc
            final_dict[chk_name]['MAX'] = max_auc
            writer.add_scalar('mean_auc_per_chk', mean_auc, i)
            writer.add_scalar('max_auc_per_chk', max_auc, i)

    writer.close()
    #
    # Save data in pickle:
    #
    pickle_file_path = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/pickles_results/{}_{}_lstm_different_ckps_500.pkl'.format(
        beat_type, gan_type)
    with open(pickle_file_path, 'wb') as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_with_noise():
    beat_type = 'N'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    with open('res_noise_{}.text'.format(beat_type), 'w') as fd:
        # for n in [500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]:
        for n in [0]:
            base_tomer_remote = '/home/nivgiladi/tomer/'
            model_dir = base_tomer_remote + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/noise_{}/lstm_add_{}/'.format(
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
                                              device=device, add_data_from_gan=False, generator_details=gen_details,
                                              train_one_vs_all=False)
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
    logging.basicConfig(level=logging.INFO)
    beat_type = 'N'
    # gan_type = 'ODE_GAN'
    gan_type = 'SIMULATOR'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_mult(beat_type, gan_type, device)

    # chk_dir = base_tomer_remote + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_S_beat'
    # find_optimal_checkpoint(chk_dir, beat_type, gan_type, device, 500)
