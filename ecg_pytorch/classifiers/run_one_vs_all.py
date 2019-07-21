import pickle
from ecg_pytorch import train_configs
from tensorboardX import SummaryWriter
from ecg_pytorch.gan_models import checkpoint_paths
from ecg_pytorch.train_configs import ECGTrainConfig, GeneratorAdditionalDataConfig
import os
import logging
import shutil
from ecg_pytorch.classifiers.models import lstm
from ecg_pytorch.classifiers import run_sequence_model
import numpy as np
import torch

base_path = train_configs.base
BEAT_TO_INDEX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}


def train_multiple_one_vs_all(beat_type, gan_type, device):

    summary_model_dir = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/{}/lstm_one_vs_all_{}_summary/'.format(
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
    for n in [0, 500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]:
        #
        # Train configurations:
        #
        logging.info("Train with additional {} beats".format(n))
        model_dir = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/tensorboard/{}/lstm_one_vs_all_{}_{}/'.format(
            beat_type, str(n), gan_type)
        gen_details = GeneratorAdditionalDataConfig(beat_type=beat_type, checkpoint_path=ck_path,
                                                    num_examples_to_add=n, gan_type=gan_type)
        train_config = ECGTrainConfig(num_epochs=5, batch_size=20, lr=0.0002, weighted_loss=False,
                                      weighted_sampling=False,
                                      device=device, add_data_from_gan=True, generator_details=gen_details,
                                      train_one_vs_all=True)
        #
        # Run 10 times each configuration:
        #
        total_runs = 0
        best_auc_per_run = []
        while total_runs < 1:
            if os.path.isdir(model_dir):
                logging.info("Removing model dir")
                shutil.rmtree(model_dir)
            #
            # Initialize the network each run:
            #
            net = lstm.ECGLSTM(5, 512, 2, 2).to(device)

            #
            # Train the classifier:
            #
            best_auc_scores = run_sequence_model.train_classifier(net, model_dir=model_dir, train_config=train_config)
            best_auc_per_run.append(best_auc_scores[BEAT_TO_INDEX[beat_type]])
            writer.add_scalar('auc_with_additional_{}_beats'.format(n), best_auc_scores[BEAT_TO_INDEX[beat_type]],
                              total_runs)
            # if best_auc_scores[BEAT_TO_INDEX[beat_type]] >= 0.88:
            #     logging.info("Found desired AUC: {}".format(best_auc_scores))
            #     break
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
    pickle_file_path = base_path + 'ecg_pytorch/ecg_pytorch/classifiers/pickles_results/{}_{}_lstm_one_vs_all.pkl'.format(
        beat_type, gan_type)
    with open(pickle_file_path, 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    beat_type = 'N'
    # gan_type = 'ODE_GAN'
    gan_type = 'DCGAN'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_multiple_one_vs_all(beat_type, gan_type, device)