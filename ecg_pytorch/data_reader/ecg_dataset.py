from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from ecg_pytorch.data_reader import pickle_data
from ecg_pytorch.gan_models.models import dcgan
import random
from ecg_pytorch.gan_models.models.old_ode_combined import CombinedGenerator as CG
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, LabelSet


class EcgHearBeatsDataset(Dataset):
    """ECG heart beats dataset."""

    def __init__(self, transform=None, beat_type=None):
        """
        [45443, 884, 3536, 414, 8]
        :param transform:
        :param beat_type:
        """
        self.train, self.val, _ = pickle_data.load_ecg_input_from_pickle()

        self.train = np.concatenate((self.train, self.val), axis=0)
        if beat_type is not None:
            self.train = np.array([sample for sample in self.train if sample['beat_type'] == beat_type])
        self.transform = transform

        # Consts:
        self.num_of_classes = 5
        self.beat_types = ['N', 'S', 'V', 'F', 'Q']
        self.beat_type_to_one_hot_label = {'N': [1, 0, 0, 0, 0],
                                           'S': [0, 1, 0, 0, 0],
                                           'V': [0, 0, 1, 0, 0],
                                           'F': [0, 0, 0, 1, 0],
                                           'Q': [0, 0, 0, 0, 1]}
        self.additional_data_from_gan = None

    def __len__(self):
        return len(self.train)

    def len_beat(self, beat_Type):
        return len(np.array([sample for sample in self.train if sample['beat_type'] == beat_Type]))

    def make_weights_for_balanced_classes(self):
        count = [self.len_beat('N'), self.len_beat('S'), self.len_beat('V'),
                 self.len_beat('F'), self.len_beat('Q')]
        weight_per_class = [0.] * self.num_of_classes
        N = float(sum(count))
        for i in range(self.num_of_classes):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.train)
        for idx, val in enumerate(self.train):
            label_ind = int(np.argmax(val['label']))
            weight[idx] = weight_per_class[label_ind]
        return weight

    def add_beats_from_generator(self, generator_model, num_beats_to_add, checkpoint_path, beat_type):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        generator_model.load_state_dict(checkpoint['generator_state_dict'])
        # discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
        with torch.no_grad():
            input_noise = torch.Tensor(np.random.normal(0, 1, (num_beats_to_add, 100)))
            output_g = generator_model(input_noise)
            output_g = output_g.numpy()
            output_g = np.array(
                [{'cardiac_cycle': x, 'beat_type': beat_type, 'label': self.beat_type_to_one_hot_label[beat_type]} for x
                 in output_g])
            # plt.plot(output_g[0]['cardiac_cycle'])
            # plt.show()
            self.additional_data_from_gan = output_g
            self.train = np.concatenate((self.train, output_g))
            print("Length of train samples after adding from generator is {}".format(len(self.train)))

    def add_beats_from_combined_generator(self, generator_model, num_beats_to_add, checkpoint_path, beat_type):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        generator_model.load_state_dict(checkpoint['generator_state_dict'])
        # discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
        with torch.no_grad():
            input_noise = torch.Tensor(np.random.normal(0, 1, (num_beats_to_add, 50)))
            inds = np.random.choice(len(self.train), 500, replace=False)
            input_v0 = self.train[inds]

            input_v0 = torch.Tensor([v['cardiac_cycle'][0] for v in input_v0])
            output_g = generator_model(input_noise, input_v0)
            output_g = output_g.numpy()
            output_g = np.array(
                [{'cardiac_cycle': x, 'beat_type': beat_type, 'label': self.beat_type_to_one_hot_label[beat_type]} for x
                 in output_g])
            plt.plot(output_g[0]['cardiac_cycle'])
            plt.show()
            self.additional_data_from_gan = output_g
            self.train = np.concatenate((self.train, output_g))
            print("Length of train samples after adding from generator is {}".format(len(self.train)))


    def vizualize_data_from_gan_and_real(self, beat_type):
        specific_beat = np.array([sample for sample in self.train if sample['beat_type'] == beat_type])
        # Show 4 beats:
        for _ in range(4):
            plt.figure()
            ind = random.randint(0, len(self.additional_data_from_gan))
            fake_beat = self.additional_data_from_gan[ind]
            plt.plot(fake_beat['cardiac_cycle'], label='Fake')
            ind = random.randint(0, len(specific_beat))
            plt.plot(specific_beat[ind]['cardiac_cycle'], label='Real')
            plt.title("Compare beats of type {}".format(beat_type))
            plt.legend()

    def __getitem__(self, idx):
        sample = self.train[idx]
        beat = sample['cardiac_cycle']
        tag = sample['beat_type']
        sample = {'cardiac_cycle': beat, 'beat_type': tag, 'label': np.array(sample['label'])}
        if self.transform:
            sample = self.transform(sample)
        return sample


def scale_signal(signal, min_val=-0.01563, max_val=0.042557):
    """

    :param min:
    :param max:
    :return:
    """
    # Scale signal to lie between -0.4 and 1.2 mV :
    scaled = np.interp(signal, (signal.min(), signal.max()), (min_val, max_val))

    # zmin = min(signal)
    # zmax = max(signal)
    # zrange = zmax - zmin
    # # for (i=1; i <= Nts; i++)
    # scaled = [(z - zmin) * max_val / zrange + min_val for z in signal]
    return scaled


class EcgHearBeatsDatasetTest(Dataset):
    """ECG heart beats dataset."""

    def __init__(self, transform=None, beat_type=None):
        _, _, self.test = pickle_data.load_ecg_input_from_pickle()

        if beat_type is not None:
            self.test = np.array([sample for sample in self.test if sample['beat_type'] == beat_type])
        self.transform = transform

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        sample = self.test[idx]
        beat = sample['cardiac_cycle']
        tag = sample['beat_type']
        sample = {'cardiac_cycle': beat, 'beat_type': tag, 'label': np.array(sample['label'])}
        if self.transform:
            sample = self.transform(sample)
        return sample


def TestEcgDataset():
    ecg_dataset = EcgHearBeatsDataset()

    fig = plt.figure()

    for i in range(len(ecg_dataset)):
        sample = ecg_dataset[i]

        print(i, sample['cardiac_cycle'].shape, sample['label'].shape, sample['beat_type'])

        ax = plt.subplot(2, 2, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.plot(sample['cardiac_cycle'])

        if i == 3:
            plt.show()
            break


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        heartbeat, label = sample['cardiac_cycle'], sample['label']
        return {'cardiac_cycle': (torch.from_numpy(heartbeat)).double(),
                'label': torch.from_numpy(label),
                'beat_type': sample['beat_type']}


class Scale(object):
    def __call__(self, sample):
        heartbeat, label = sample['cardiac_cycle'], sample['label']
        heartbeat = scale_signal(heartbeat)
        return {'cardiac_cycle': heartbeat,
                'label': label,
                'beat_type': sample['beat_type']}


def iterate_data_example():
    composed = transforms.Compose([ToTensor()])
    ecg_dataset = EcgHearBeatsDataset(transform=composed)
    for i in range(len(ecg_dataset)):
        sample = ecg_dataset[i]

        print(i, sample['cardiac_cycle'].size(), sample['label'].size())

        if i == 3:
            break


def iterate_with_dataloader(transformed_dataset):
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    # Helper function to show a batch
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        ecg_batch, label_batch = \
            sample_batched['cardiac_cycle'], sample_batched['label']
        batch_size = len(ecg_batch)
        # im_size = images_batch.size(2)

        # grid = utils.make_grid(ecg_batch)
        # plt.imshow(grid.numpy().transpose((1, 2, 0)))

        for i in range(batch_size):
            ax = plt.subplot(2, 2, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            # ax.axis('off')
            plt.plot(ecg_batch[i].numpy())
            print(label_batch[i])
            # plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
            #             landmarks_batch[i, :, 1].numpy(),
            #             s=10, marker='.', c='r')
            #
            # plt.title('Batch from dataloader')

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['cardiac_cycle'].size(),
              sample_batched['label'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            # plt.axis('off')
            plt.ioff()
            plt.show()
            break


def test_balanced_iterations():
    composed = transforms.Compose([ToTensor()])
    dataset = EcgHearBeatsDataset(transform=composed)
    weights_for_balance = dataset.make_weights_for_balanced_classes()
    weights_for_balance = torch.DoubleTensor(weights_for_balance)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights_for_balance,
        num_samples=len(weights_for_balance),
        replacement=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=1, sampler=sampler)
    for epoch in range(2):
        for i_batch, sample_batched in enumerate(dataloader):
            ecg_batch = sample_batched['cardiac_cycle']
            b_size = ecg_batch.shape[0]
            if b_size == 4:
                print("epcoh {}, Batch number {}\t Label #1 = {}, Label #2 = {}, Label #3 = {}, Label #4 = {}".format(
                    epoch, i_batch,
                    sample_batched[
                        'beat_type'][0],
                    sample_batched[
                        'beat_type'][1],
                    sample_batched[
                        'beat_type'][2],
                    sample_batched[
                        'beat_type'][
                        3]))


def vizualize_heart_beat_with_pqrst(beat_type):
    dataset = EcgHearBeatsDataset(beat_type=beat_type)
    index = np.random.choice(len(dataset), 10, replace=False)
    # index = [19155 42001 27081  7074 41453 32289 28426 34762 35361  9642]
    index = 32289
    print(index)
    # for i in index:
    p = figure(x_axis_label='Sample number (360 Hz)', y_axis_label='Voltage[mV]')
    time = np.arange(0, 216)
    b = dataset.train[index]
    p_loc = b['P_wave_location']
    q_loc = b['Q_wave_location']
    r_loc = b['R_peak_location']
    s_loc = 80
    t_loc = b['T_wave_location']

    # add a line renderer with legend and line thickness
    real_sample = b['cardiac_cycle']

    p.line(time, real_sample, line_width=2, line_color='blue')
    p.circle(p_loc, real_sample[p_loc], size=8, fill_color="purple")
    p.circle(q_loc, real_sample[q_loc], size=8, fill_color="black")
    p.circle(r_loc, real_sample[r_loc], size=8, fill_color="green")
    p.circle(s_loc, real_sample[s_loc], size=8, fill_color="orange")
    p.circle(t_loc, real_sample[t_loc], size=8, fill_color="red")

    source = ColumnDataSource(data=dict(time=[p_loc - 4, q_loc - 4, r_loc - 4, s_loc - 4, t_loc - 4],
                                        voltage=[real_sample[p_loc] + 0.03, real_sample[q_loc] - 0.1, real_sample[r_loc] + 0.01,
                                                 real_sample[s_loc] - 0.1, real_sample[t_loc] + 0.03],
                                        waves=['P', 'Q', 'R', 'S', 'T']))
    labels = LabelSet(x='time', y='voltage', text='waves', level='glyph',
                      x_offset=5, y_offset=5, source=source, render_mode='canvas')
    p.add_layout(labels)

    # show the results
    show(p)


if __name__ == "__main__":
    # TestEcgDataset()
    # iterate_data_example()
    # composed = transforms.Compose([Scale(), ToTensor()])
    # transformed_dataset = EcgHearBeatsDataset(transform=composed, beat_type='S')
    # print(len(transformed_dataset))
    # iterate_with_dataloader(transformed_dataset)
    # ecg_dataset = EcgHearBeatsDataset()
    # hb = ecg_dataset[152]['cardiac_cycle']
    # smooth_signal(hb)
    # plt.plot(hb)
    # plt.xlabel('Time (1/360 ms)')
    # plt.ylabel('Voltage')
    # plt.show()
    # ecg_dataset = EcgHearBeatsDataset()
    # ecg_dataset.make_weights_for_balanced_classes()
    # test_balanced_iterations()
    # ecg_dataset = EcgHearBeatsDataset(beat_type='N')
    # gNet = dcgan.DCGenerator(0)
    # checkpoint_path = '/Users/tomer.golany/PycharmProjects/ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_N_beat/' \
    #                   'checkpoint_epoch_0_iters_801'
    # ecg_dataset.add_beats_from_generator(gNet, 500,
    #                                      checkpoint_path,
    #                                      'N')

    # gNet = CG(0, 'cpu')
    # ch_path = '/Users/tomer.golany/PycharmProjects/py_torch_runge_kutta/gan_models/tensorboard/combined_generator_N/checkpoint_epoch_0_iters_200'
    # ecg_dataset.add_beats_from_combined_generator(gNet, 500, ch_path, 'N')

    # beats = ecg_dataset.train
    # max_value = max([max(b['cardiac_cycle']) for b in beats])
    # min_value = min([min(b['cardiac_cycle']) for b in beats])
    # print(max_value)
    # print(min_value)
    vizualize_heart_beat_with_pqrst('N')
