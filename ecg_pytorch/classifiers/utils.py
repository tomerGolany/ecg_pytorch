import pickle


def load_and_print_pickle_contents(pickle_path):
    with open(pickle_path, 'rb') as handle:
        b = pickle.load(handle)
        # print(b)
    return b


if __name__ == "__main__":
    pickle_path = "/home/tomer/tomer/ecg_pytorch/ecg_pytorch/classifiers/pickles_results/" \
                  "S_ODE_GAN_lstm_different_ckps_500.pkl"
    d = load_and_print_pickle_contents(pickle_path)
    for chk in d:
        print("{}: mean: {}, max: {}".format(chk, d[chk]['MEAN'], d[chk]['MAX']))
