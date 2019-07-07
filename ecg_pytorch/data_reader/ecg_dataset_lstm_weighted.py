from torch.utils.data import Dataset
import numpy as np
import torch
from ecg_pytorch.data_reader import pickle_data


class EcgHearBeatsDataset(Dataset):
    """ECG heart beats dataset."""

    def __init__(self, transform=None):
        """
        [45443, 884, 3536, 414, 8]
        :param transform:
        """
        self.train, self.val, _ = pickle_data.load_ecg_input_from_pickle()
        self.train = np.concatenate((self.train, self.val), axis=0)

        self.n_beats = np.array([sample for sample in self.train if sample['beat_type'] == 'N'])
        self.s_beats = np.array([sample for sample in self.train if sample['beat_type'] == 'S'])
        self.v_beats = np.array([sample for sample in self.train if sample['beat_type'] == 'V'])
        self.f_beats = np.array([sample for sample in self.train if sample['beat_type'] == 'F'])
        self.q_beats = np.array([sample for sample in self.train if sample['beat_type'] == 'Q'])
        # consts:
        self.transform = transform
        self.beat_types = ['N', 'S', 'V', 'F', 'Q']
        self.beat_type_to_one_hot_label = {'N': [1, 0, 0, 0, 0],
                                           'S': [0, 1, 0, 0, 0],
                                           'V': [0, 0, 1, 0, 0],
                                           'F': [0, 0, 0, 1, 0],
                                           'Q': [0, 0, 0, 0, 1]}
        self.len_n = len(self.n_beats)
        self.len_s = len(self.s_beats)
        self.len_v = len(self.v_beats)
        self.len_f = len(self.f_beats)
        self.len_q = len(self.q_beats)

    def __len__(self):
        return len(self.train)

    def len_beat(self, beat_Type):
        return len(np.array([sample for sample in self.train if sample['beat_type'] == beat_Type]))

    def __getitem__(self, idx):
        # returns one element from each label:
        if idx == len(self.train):
            pass
            # TODO: shuffle data.
        i_n = idx % self.len_n
        i_s = idx % self.len_s
        i_v = idx % self.len_v
        i_f = idx % self.len_f
        i_q = idx % self.len_q

        sample_n = self.n_beats[i_n]
        sample_s = self.s_beats[i_s]
        sample_v = self.v_beats[i_v]
        sample_f = self.f_beats[i_f]
        sample_q = self.q_beats[i_q]

        lstm_beat_n = np.array([sample_n['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])
        lstm_beat_s = np.array([sample_s['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])
        lstm_beat_v = np.array([sample_v['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])
        lstm_beat_f = np.array([sample_f['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])
        lstm_beat_q = np.array([sample_q['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])
        sample = {'cardiac_cycle_n': lstm_beat_n,
                  'cardiac_cycle_s': lstm_beat_s,
                  'cardiac_cycle_v': lstm_beat_v,
                  'cardiac_cycle_f': lstm_beat_f,
                  'cardiac_cycle_q': lstm_beat_q}

        if self.transform:
            sample = self.transform(sample)
        return sample

