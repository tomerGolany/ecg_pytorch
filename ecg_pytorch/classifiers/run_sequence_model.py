import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
from ecg_pytorch.classifiers.models import lstm, deep_residual_conv
from ecg_pytorch.train_configs import ECGTrainConfig, GeneratorAdditionalDataConfig
import logging
import os
import shutil
from ecg_pytorch.gan_models import checkpoint_paths
import pickle
from ecg_pytorch import train_configs
from ecg_pytorch.data_reader import dataset_builder
from ecg_pytorch.classifiers import metrics


base_path = train_configs.base

BEAT_TO_INDEX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}


def init_weights(m):
    if type(m) == nn.LSTM:
        torch.nn.init.xavier_uniform(m.weight_hh_l0.data)

        torch.nn.init.xavier_uniform(m.weight_ih_l0.data)


def predict_fn(data_loader, train_config, criterion, writer, total_iters, best_auc_scores):
    device = train_config.device
    logging.info("Performing evaluation:")
    softmax_layer = nn.Softmax(dim=1)
    with torch.no_grad():

        if train_config.train_one_vs_all:
            labels_total_one_hot = np.array([]).reshape((0, 2))
            outputs_preds = np.array([]).reshape((0, 2))
        else:
            labels_total_one_hot = np.array([]).reshape((0, 5))
            outputs_preds = np.array([]).reshape((0, 5))

        labels_ind_total = np.array([])
        outputs_ind_total = np.array([])
        loss_hist = []
        for _, test_data in enumerate(data_loader):
            ecg_batch = test_data['cardiac_cycle'].float().to(device)
            labels = test_data['label'].to(device)
            # logging.info("labels shape: {}. first labels in batch: {}".format(labels.shape, labels[0]))
            labels_ind = torch.max(labels, 1)[1]
            preds_before_softmax = net(ecg_batch)
            # logging.info(
            #     "output before softmax shape: {}. first in batch: {}".format(preds_before_softmax.shape,
            #                                                                  preds_before_softmax[0]))
            probs_after_softmax = softmax_layer(preds_before_softmax)
            # logging.info(
            #     "output softmax shape: {}. first in batch: {}".format(probs_after_softmax.shape,
            #                                                           probs_after_softmax[0]))

            loss = criterion(preds_before_softmax, torch.max(labels, 1)[1])
            loss_hist.append(loss.item())
            outputs_class = torch.max(preds_before_softmax, 1)[1]

            labels_total_one_hot = np.concatenate((labels_total_one_hot, labels.cpu().numpy()))
            labels_ind_total = np.concatenate((labels_ind_total, labels_ind.cpu().numpy()))
            outputs_ind_total = np.concatenate((outputs_ind_total, outputs_class.cpu().numpy()))
            outputs_preds = np.concatenate((outputs_preds, probs_after_softmax.cpu().numpy()))

        outputs_ind_total = outputs_ind_total.astype(int)
        labels_ind_total = labels_ind_total.astype(int)

        loss = sum(loss_hist) / len(loss_hist)
        writer.add_scalars('cross_entropy_loss', {'Test set loss': loss}, total_iters)
        return labels_ind_total, outputs_ind_total, labels_total_one_hot, outputs_preds


def write_summaries(writer, train_config, labels_class, outputs_class, total_iters, data_set_type, best_auc_scores,
                    output_probabilites=None, labels_one_hot=None):
    if train_config.train_one_vs_all:
        generator_beat_type = train_config.generator_details.beat_type
        fig, _ = metrics.plot_confusion_matrix(labels_class, outputs_class,
                                               np.array([generator_beat_type, 'Others']))
    else:
        fig, _ = metrics.plot_confusion_matrix(labels_class, outputs_class,
                                               np.array(['N', 'S', 'V', 'F', 'Q']))

    writer.add_figure('{}/confusion_matrix'.format(data_set_type), fig, total_iters)

    if data_set_type == 'test':
        #
        # Check AUC values:
        #
        if train_config.train_one_vs_all:
            generator_beat_type = train_config.generator_details.beat_type
            auc_roc = metrics.plt_roc_curve(labels_one_hot, output_probabilites,
                                            np.array([generator_beat_type, 'Other']),
                                            writer,
                                            total_iters)

            metrics.plt_precision_recall_curve(labels_one_hot, output_probabilites,
                                               np.array([generator_beat_type, 'Other']),
                                               writer,
                                               total_iters)

            for i_auc in range(2):
                if auc_roc[i_auc] > best_auc_scores[i_auc]:
                    best_auc_scores[i_auc] = auc_roc[i_auc]

        else:
            auc_roc = metrics.plt_roc_curve(labels_one_hot, output_probabilites,
                                            np.array(['N', 'S', 'V', 'F', 'Q']),
                                            writer,
                                            total_iters)

            metrics.plt_precision_recall_curve(labels_one_hot, output_probabilites,
                                            np.array(['N', 'S', 'V', 'F', 'Q']),
                                            writer,
                                            total_iters)

            for i_auc in range(4):
                if auc_roc[i_auc] > best_auc_scores[i_auc]:
                    best_auc_scores[i_auc] = auc_roc[i_auc]


def model_fn(network_object, model_dir, train_config=None):
    """

    :param network_object: Network type - Type of recurrent network.
    :return:
    """
    logging.info("Train configurations: {}".format(train_config))
    # with open(os.path.join(model_dir, 'pipeline.config'), 'wb') as fd:
    #     fd.write(str.encode("{}".format(train_config)))

    num_of_epochs = train_config.num_epochs
    lr = train_config.lr
    device = train_config.device
    batch_size = train_config.batch_size

    train_data_loader, test_data_loader, best_auc_scores = dataset_builder.build(train_config)

    if train_config.train_one_vs_all:
        label_names = np.array([train_config.generator_details.beat_type, 'Others'])
    else:
        label_names = np.array(['N', 'S', 'V', 'F', 'Q'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network_object.parameters(), lr=lr)
    writer = SummaryWriter(model_dir, max_queue=1000)
    total_iters = 0
    num_iters_per_epoch = int(np.floor(len(train_data_loader) / batch_size) + batch_size)

    for epoch in range(num_of_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(train_data_loader):
            total_iters += 1
            ecg_batch = data['cardiac_cycle'].float().to(device)
            b_size = ecg_batch.shape[0]
            labels = data['label'].to(device)
            labels_class = torch.max(labels, 1)[1]
            labels_str = data['beat_type']

            logging.debug("batch labels: {}".format(labels_str))

            # zero the parameter gradients
            network_object.zero_grad()

            # forward + backward + optimize
            outputs = network_object(ecg_batch)

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
            if total_iters % 10 == 0:
                logging.info(
                    "Epoch {}. Iteration {}/{}.\t Batch train loss = {:.2f}. Accuracy batch train = {:.2f}".format(
                        epoch + 1, i,
                        num_iters_per_epoch,
                        loss.item(),
                        accuracy.item()))
            if total_iters % 300 == 0:
                write_summaries(writer, train_config, labels_class.cpu().numpy(), outputs_class.cpu().numpy(),
                                total_iters, 'train', best_auc_scores)

                grad_norm = get_gradient_norm_l2(network_object)
                writer.add_scalar('gradients_norm', grad_norm, total_iters)
                logging.info("Norm of gradients = {}.".format(grad_norm))

            if total_iters % 200 == 0:
                labels_ind_total, outputs_ind_total, labels_total_one_hot, outputs_preds = predict_fn(test_data_loader,
                                                                                                      train_config,
                                                                                                      criterion, writer,
                                                                                                      total_iters,
                                                                                                      best_auc_scores)

                accuracy = sum((outputs_ind_total == labels_ind_total)) / len(outputs_ind_total)
                writer.add_scalars('accuracy', {'Test set accuracy': accuracy}, global_step=total_iters)

                logging.info("total output preds shape: {}. labels_total_one_hot shape: {}".format(outputs_preds.shape,
                                                                                                   labels_total_one_hot.shape))
                # write_summaries(writer, train_config, labels_ind_total, outputs_ind_total, total_iters, 'test',
                #                 best_auc_scores,
                #                 outputs_preds, labels_total_one_hot)

        labels_ind_total, outputs_ind_total, labels_total_one_hot, outputs_preds = predict_fn(test_data_loader,
                                                                                              train_config, criterion,
                                                                                              writer, total_iters,
                                                                                              best_auc_scores)

        write_summaries(writer, train_config, labels_ind_total, outputs_ind_total, epoch, 'test',
                        best_auc_scores,
                        outputs_preds, labels_total_one_hot)

        metrics.add_roc_curve_bokeh(labels_total_one_hot, outputs_preds,
                                                label_names, model_dir, epoch)

        metrics.plt_precision_recall_bokeh(labels_total_one_hot, outputs_preds, label_names, model_dir, epoch)

    writer.close()

    torch.save({
        'net': net.state_dict()
    }, model_dir + '/checkpoint_epoch_iters_{}'.format(total_iters))
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
    model_dir = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/s_resnet_raw_v2/vgan_1000/'
    if os.path.exists(model_dir):
        print("Model dir {} already exists! exiting...".format(model_dir))
        exit()
    else:
        os.makedirs(model_dir)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("{}/logs.log".format(model_dir)),
                                                      logging.StreamHandler()])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Devices: {}".format(device))



    # net = lstm.ECGLSTM(5, 512, 2, 2).to(device)

    net = deep_residual_conv.Net(2).to(device)

    gen_details = GeneratorAdditionalDataConfig(beat_type='S', checkpoint_path=checkpoint_paths.VGAN_S_CHK,
                                                num_examples_to_add=1000, gan_type=dataset_builder.GanType.VANILA_GAN)

    train_config_1 = ECGTrainConfig(num_epochs=15, batch_size=300, lr=0.0001, weighted_loss=False,
                                    weighted_sampling=False,
                                    device=device, add_data_from_gan=True, generator_details=gen_details,
                                    train_one_vs_all=True)
    model_fn(net, model_dir, train_config=train_config_1)
