import pickle
from ecg_pytorch import train_configs
from ecg_pytorch.data_reader import ecg_mit_bih

full_path = train_configs.base + 'ecg_pytorch/ecg_pytorch/data_reader'


def load_ecg_input_from_pickle():
    with open(full_path + '/train_beats.pickle', 'rb') as handle:
        train_beats = pickle.load(handle)
    with open(full_path + '/val_beats.pickle', 'rb') as handle:
        validation_beats = pickle.load(handle)
    with open(full_path + '/test_beats.pickle', 'rb') as handle:
        test_beats = pickle.load(handle)
    return train_beats, validation_beats, test_beats


def save_ecg_mit_bih_to_pickle():
    with open(full_path + '/ecg_mit_bih.pickle', 'wb') as output:
        ecg_ds = ecg_mit_bih.ECGMitBihDataset()
        pickle.dump(ecg_ds, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    save_ecg_mit_bih_to_pickle()
