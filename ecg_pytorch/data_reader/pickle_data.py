import pickle

niv_remote = '/home/nivgiladi/tomer/'
local_base = '/Users/tomer.golany/PycharmProjects/'
nlp_base = '/home/tomer.golany@st.technion.ac.il/'
tomer_remote = '/home/tomer/tomer/'
full_path = local_base + 'ecg_pytorch/ecg_pytorch/data_reader'
# full_path = '/home/tomer.golany@st.technion.ac.il/ecg_pytorch/ecg_pytorch/data_reader'


def load_ecg_input_from_pickle():
    with open(full_path + '/train_beats.pickle', 'rb') as handle:
        train_beats = pickle.load(handle)
    with open(full_path + '/val_beats.pickle', 'rb') as handle:
        validation_beats = pickle.load(handle)
    with open(full_path + '/test_beats.pickle', 'rb') as handle:
        test_beats = pickle.load(handle)
    return train_beats, validation_beats, test_beats


if __name__ == "__main__":
    pass
    # create_ecg_input_pickle()
