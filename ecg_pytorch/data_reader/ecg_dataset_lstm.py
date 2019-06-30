from torch.utils.data import Dataset
import numpy as np
import torch
from ecg_pytorch.data_reader import pickle_data


class EcgHearBeatsDataset(Dataset):
    """ECG heart beats dataset."""

    def __init__(self, transform=None, beat_type=None, one_vs_all=None):
        """
        [45443, 884, 3536, 414, 8]
        :param transform:
        :param beat_type:
        """
        self.train, self.val, _ = pickle_data.load_ecg_input_from_pickle()
        self.one_vs_all = False
        self.train = np.concatenate((self.train, self.val), axis=0)
        if beat_type is not None and one_vs_all is None:
            self.train = np.array([sample for sample in self.train if sample['beat_type'] == beat_type])

        if one_vs_all is not None:
            self.beat_type = beat_type
            self.one_vs_all = True
            self.num_of_classes = 2
        else:
            self.num_of_classes = 5

        # consts:
        self.transform = transform
        self.beat_types = ['N', 'S', 'V', 'F', 'Q']
        self.beat_type_to_one_hot_label = {'N': [1, 0, 0, 0, 0],
                                           'S': [0, 1, 0, 0, 0],
                                           'V': [0, 0, 1, 0, 0],
                                           'F': [0, 0, 0, 1, 0],
                                           'Q': [0, 0, 0, 0, 1]}

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

    def weights_per_class(self):
        count = np.array([self.len_beat('N'), self.len_beat('S'), self.len_beat('V'),
                 self.len_beat('F'), self.len_beat('Q')])
        print("Beat N: #{}\t Beat S: #{}\t Beat V: #{}\n Beat F: #{}\t Beat Q: #{}".format(count[0], count[1], count[2],
                                                                                           count[3], count[4]))
        N = float(sum(count))
        print("Total num of beats: #{}".format(N))
        weights = N / count
        print(weights)
        return weights

    def __len__(self):
        return len(self.train)

    def len_beat(self, beat_Type):
        return len(np.array([sample for sample in self.train if sample['beat_type'] == beat_Type]))

    def __getitem__(self, idx):
        sample = self.train[idx]
        lstm_beat = np.array([sample['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])
        tag = sample['beat_type']
        if not self.one_vs_all:
            sample = {'cardiac_cycle': lstm_beat, 'beat_type': tag, 'label': np.array(sample['label'])}
        else:
            if tag == self.beat_type:
                sample = {'cardiac_cycle': lstm_beat, 'beat_type': tag, 'label': np.array([1, 0])}
            else:
                sample = {'cardiac_cycle': lstm_beat, 'beat_type': tag, 'label': np.array([0, 1])}
        if self.transform:
            sample = self.transform(sample)
        return sample


class EcgHearBeatsDatasetTest(Dataset):
    """ECG heart beats dataset."""

    def __init__(self, transform=None, beat_type=None, one_vs_all=None):
        _, _, self.test = pickle_data.load_ecg_input_from_pickle()
        self.one_vs_all = False
        if beat_type is not None and one_vs_all is None:
            self.test = np.array([sample for sample in self.test if sample['beat_type'] == beat_type])

        if one_vs_all is not None:
            self.beat_type = beat_type
            self.one_vs_all = True
            self.num_of_classes = 2
        else:
            self.num_of_classes = 5
        self.transform = transform

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        sample = self.test[idx]
        lstm_beat = np.array([sample['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])  # [43, 5]
        tag = sample['beat_type']
        # sample = {'cardiac_cycle': lstm_beat, 'beat_type': tag, 'label': np.array(sample['label'])}
        if not self.one_vs_all:
            sample = {'cardiac_cycle': lstm_beat, 'beat_type': tag, 'label': np.array(sample['label'])}
        else:
            if tag == self.beat_type:
                sample = {'cardiac_cycle': lstm_beat, 'beat_type': tag, 'label': np.array([1, 0])}
            else:
                sample = {'cardiac_cycle': lstm_beat, 'beat_type': tag, 'label': np.array([0, 1])}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        heartbeat, label = sample['cardiac_cycle'], sample['label']
        return {'cardiac_cycle': (torch.from_numpy(heartbeat)).double(),
                'label': torch.from_numpy(label),
                'beat_type': sample['beat_type']}