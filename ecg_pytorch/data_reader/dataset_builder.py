"""Function that support building dataset for ECG heartbeats."""
import torchvision.transforms as transforms
from ecg_pytorch.data_reader.ecg_dataset_lstm import ToTensor, EcgHearBeatsDataset, EcgHearBeatsDatasetTest
import torch
import logging
from enum import Enum
from ecg_pytorch.gan_models.models import dcgan
from ecg_pytorch.gan_models.models import vanila_gan
from ecg_pytorch.gan_models.models import ode_gan_aaai


class GanType(Enum):
    DCGAN = 0
    ODE_GAN = 1
    SIMULATOR = 2
    VANILA_GAN = 3
    VANILA_GAN_ODE = 4
    NOISE = 5


def build(train_config,):
    """Build PyTorch train and test data-loaders.

    :param train_config:
    :return:
    """
    add_from_gan = train_config.add_data_from_gan
    batch_size = train_config.batch_size
    composed = transforms.Compose([ToTensor()])
    if train_config.train_one_vs_all:
        dataset = EcgHearBeatsDataset(transform=composed, beat_type=train_config.generator_details.beat_type,
                                      one_vs_all=True, lstm_setting=False)
        testset = EcgHearBeatsDatasetTest(transform=composed, beat_type=train_config.generator_details.beat_type,
                                          one_vs_all=True, lstm_setting=False)
        init_auc_scores = [0, 0]
    else:
        init_auc_scores = [0, 0, 0, 0]
        dataset = EcgHearBeatsDataset(transform=composed, lstm_setting=False)
        testset = EcgHearBeatsDatasetTest(transform=composed, lstm_setting=False)

    testdataloader = torch.utils.data.DataLoader(testset, batch_size=300,
                                                 shuffle=True, num_workers=1)

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
            if gan_type == GanType.DCGAN:
                gNet = dcgan.DCGenerator(0)
                dataset.add_beats_from_generator(gNet, num_examples_to_add,
                                                 generator_checkpoint_path,
                                                 generator_beat_type)
            elif gan_type == GanType.ODE_GAN:
                gNet = ode_gan_aaai.DCGenerator(0)
                dataset.add_beats_from_generator(gNet, num_examples_to_add,
                                                 generator_checkpoint_path,
                                                 generator_beat_type)
            elif gan_type == GanType.SIMULATOR:
                dataset.add_beats_from_simulator(num_examples_to_add, generator_beat_type)

            elif gan_type == GanType.VANILA_GAN:
                gNet = vanila_gan.VGenerator(0)
                dataset.add_beats_from_generator(gNet, num_examples_to_add,
                                                 generator_checkpoint_path,
                                                 generator_beat_type)

            elif gan_type == GanType.VANILA_GAN_ODE:
                gNet = vanila_gan.VGenerator(0)
                dataset.add_beats_from_generator(gNet, num_examples_to_add,
                                                 generator_checkpoint_path,
                                                 generator_beat_type)

            elif gan_type == GanType.NOISE:
                dataset.add_noise(num_examples_to_add, generator_beat_type)

            else:
                raise ValueError("Unknown gan type {}".format(gan_type))

        logging.info("Size of training data after additional data from GAN: {}".format(len(dataset)))
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(dataset.len_beat('N'), dataset.len_beat('S'),
                                                                dataset.len_beat('V'), dataset.len_beat('F')))
    
    else:
        logging.info("No data is added. Train set size: ")
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(dataset.len_beat('N'), dataset.len_beat('S'),
                                                                  dataset.len_beat('V'), dataset.len_beat('F')))
        logging.info("test set size: ")
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(testset.len_beat('N'), testset.len_beat('S'),
                                                                  testset.len_beat('V'), testset.len_beat('F')))

    if train_config.weighted_sampling:
        weights_for_balance = dataset.make_weights_for_balanced_classes()
        weights_for_balance = torch.DoubleTensor(weights_for_balance)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=weights_for_balance,
            num_samples=len(weights_for_balance),
            replacement=True)
        train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 num_workers=1, sampler=sampler)
    else:
        train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 num_workers=1, shuffle=True)

    return train_data_loader, testdataloader, init_auc_scores