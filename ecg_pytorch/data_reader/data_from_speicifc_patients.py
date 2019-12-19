import numpy as np
import os
import logging
from matplotlib import pyplot as plt
from bokeh.io import output_file, show
from bokeh.layouts import row
from bokeh.plotting import figure
from ecg_pytorch import train_configs

DATA_DIR = train_configs.base + 'ecg_pytorch/ecg_pytorch/data_reader/text_files/'

train_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                 223, 230]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
            233, 234]  # DS2
test_set = [str(x) for x in test_set]


def convert_tag_to_aami_class_str(x):
    """

    :param x: label str.
    :return:
    """
    if x == 'N' or x == 'L' or x == 'R' or x == 'e' or x == 'j':
        beat_type_str = 'N'
    elif x == 'A' or x == 'a' or x == 'J' or x == 'S':
        beat_type_str = 'S'
    elif x == "V" or x == 'E':
        beat_type_str = 'V'
    elif x == "F":
        beat_type_str = 'F'
    else:
        beat_type_str = 'Q'
    return beat_type_str


def convert_tag_to_aami_class_ind(tag):
    """
    convert to 5 different classes:
    0 - N
    1 - S
    2 - V
    3 - F
    4 - Q
    :param tags:
    :return: ret.
    """

    x = tag
    if x == 'N' or x == 'L' or x == 'R' or x == 'e' or x == 'j':
        ret = 0
    elif x == 'A' or x == 'a' or x == 'J' or x == 'S':
        ret = 1
    elif x == "V" or x == 'E':
        ret = 2
    elif x == "F":
        ret = 3
    else:
        ret = 4
    return ret


def read_data_from_single_patient(patient_number):
    """Read text file belonging the a patient.

    :param patient_number: string which represents the patient number.
    :return:
    """
    dat_file = os.path.join(DATA_DIR, patient_number + '.txt')
    time = []
    voltage1 = []
    voltage2 = []
    with open(dat_file, 'r') as fd:
        for line in fd:
            line = line.split()
            time.append(line[0])
            voltage1.append(float(line[1]))
            voltage2.append(float(line[2]))

    tags_file = os.path.join(DATA_DIR, patient_number + '_tag.txt')
    tags_time = []
    tags = []
    r_peaks_indexes = []
    with open(tags_file, 'r') as fd:
        for line in fd:
            line = line.split()
            tags_time.append(line[0])
            tags.append(line[2])
            r_peaks_indexes.append(int(line[1]))
    return time, voltage1, voltage2, tags_time, tags, r_peaks_indexes


def beats_from_patient(patient_number):
    """Get data from a patient.

    :param patient_number: str patient.
    :return:
    """
    sampling_rate = 360  # 360 samples per second
    before = 0.2  # 0.2 seconds == 0.2 * 10^3 miliseconds == 200 ms
    after = 0.4  # --> 400 ms
    time, voltage1, voltage2, tags_time, tags, r_peaks_indexes = read_data_from_single_patient(patient_number)

    ecg_signal = np.array(voltage1)
    r_peak_locations = np.array(r_peaks_indexes)

    # convert seconds to samples
    before = int(before * sampling_rate)  # Number of samples per 200 ms.
    after = int(after * sampling_rate)  # number of samples per 400 ms.

    len_of_signal = len(ecg_signal)

    heart_beats = []

    for ind, r_peak in enumerate(r_peak_locations):
        start = r_peak - before
        if start < 0:
            logging.info("Skipping beat {}".format(ind))
            continue
        end = r_peak + after
        if end > len_of_signal - 1:
            logging.info("Skipping beat {}".format(ind))
            break
        heart_beats_dict = {}
        heart_beat = np.array(ecg_signal[start:end])
        heart_beats_dict['patient_number'] = patient_number
        heart_beats_dict['cardiac_cycle'] = heart_beat
        heart_beats_dict['label'] = convert_tag_to_aami_class_ind(tags[ind])  # index label.
        heart_beats_dict['label'] = convert_to_one_hot(heart_beats_dict['label'])
        heart_beats_dict['beat_type'] = convert_tag_to_aami_class_str(tags[ind])  # str label
        heart_beats_dict['beat_ind'] = ind
        heart_beats.append(heart_beats_dict)
    return heart_beats


def convert_to_one_hot(ind):
    res = [0 for _ in range(5)]
    res[ind] = 1
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p_100 = beats_from_patient('101')
    print(len(p_100))

    time = list(range(216))

    N_b = [x['cardiac_cycle'] for x in p_100 if x['beat_type'] == 'N']
    for i in range(100):
        p = figure(x_axis_label='Sample number (360 Hz)', y_axis_label='Voltage[mV]')
        p.line(time, N_b[i], line_width=2, line_color="green")
        output_file("N_{}_real.html".format(i))
        show(p)
        # plt.plot(N_b[i])
        # plt.show()
