"""Preprocessing the MIT-BIH dataset"""
from ecg_pytorch import train_configs
from ecg_pytorch.data_reader import patient
import logging

DATA_DIR = train_configs.base + 'ecg_pytorch/ecg_pytorch/data_reader/text_files/'

train_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                 223, 230]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
            233, 234]  # DS2
test_set = [str(x) for x in test_set]


def build_dataset():
    train_beats = []
    for p in train_set:
        heart_beats_single_patient = patient.beats_from_patient(p)
        train_beats += heart_beats_single_patient

    test_beats = []
    for p in test_set:
        heart_beats_single_patient = patient.beats_from_patient(p)
        test_beats += heart_beats_single_patient

    return train_beats, test_beats


def counts():
    train, test = build_dataset()

    # train count:
    n_train = len([x for x in train if x['beat_type'] == 'N'])
    s_train = len([x for x in train if x['beat_type'] == 'S'])
    v_train = len([x for x in train if x['beat_type'] == 'V'])
    f_train = len([x for x in train if x['beat_type'] == 'F'])
    q_train = len([x for x in train if x['beat_type'] == 'Q'])

    n_test = len([x for x in test if x['beat_type'] == 'N'])
    s_test = len([x for x in test if x['beat_type'] == 'S'])
    v_test = len([x for x in test if x['beat_type'] == 'V'])
    f_test = len([x for x in test if x['beat_type'] == 'F'])
    q_test = len([x for x in test if x['beat_type'] == 'Q'])

    print("In train set: #N: {}, #S: {}, #V: {}, #F: {}, #Q: {}".format(n_train, s_train, v_train, f_train, q_train))
    print("In test set: #N: {}, #S: {}, #V: {}, #F: {}, #Q: {}".format(n_test, s_test, v_test, f_test, q_test))

    print(train[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    counts()