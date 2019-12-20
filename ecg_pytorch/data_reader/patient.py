import numpy as np
import os
import logging
# from matplotlib import pyplot as plt
# from bokeh.io import output_file, show
# from bokeh.layouts import row
# from bokeh.plotting import figure
from ecg_pytorch import train_configs
from ecg_pytorch.data_reader import heartbeat_types

DATA_DIR = train_configs.base + 'ecg_pytorch/ecg_pytorch/data_reader/text_files/'

train_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                 223, 230]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
            233, 234]  # DS2
test_set = [str(x) for x in test_set]


class Patient(object):
    """Patient object represents a patient from the MIT-BIH AR database.

    Attributes:


    """
    def __init__(self, patient_number):
        """Init Patient object from corresponding text file.

        :param patient_number: string which represents the patient number.
        """
        self.patient_number = patient_number
        self.time, self.voltage1, self.voltage2, self.tags_time, self.tags, self.r_peaks_indexes = \
            self.read_raw_data()
        self.heartbeats = self.slice_heartbeats()

    def read_raw_data(self):
        """Read patient's data file.

        :return:
        """
        dat_file = os.path.join(DATA_DIR, self.patient_number + '.txt')
        if not os.path.exists(dat_file):
            raise AssertionError("{} doesn't exist.".format(dat_file))
        time = []
        voltage1 = []
        voltage2 = []
        with open(dat_file, 'r') as fd:
            for line in fd:
                line = line.split()
                time.append(line[0])
                voltage1.append(float(line[1]))
                voltage2.append(float(line[2]))

        tags_file = os.path.join(DATA_DIR, self.patient_number + '_tag.txt')
        if not os.path.exists(dat_file):
            raise AssertionError("{} doesn't exist.".format(tags_file))
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

    def slice_heartbeats(self):
        """Slice heartbeats from the raw signal.

        :return:
        """
        sampling_rate = 360  # 360 samples per second
        before = 0.2  # 0.2 seconds == 0.2 * 10^3 miliseconds == 200 ms
        after = 0.4  # --> 400 ms
        ecg_signal = np.array(self.voltage1)
        r_peak_locations = np.array(self.r_peaks_indexes)

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
            heart_beats_dict['patient_number'] = self.patient_number
            heart_beats_dict['cardiac_cycle'] = heart_beat
            aami_label_str = heartbeat_types.convert_heartbeat_mit_bih_to_aami(self.tags[ind])
            aami_label_ind = heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class(self.tags[ind])
            heart_beats_dict['mit_bih_label_str'] = self.tags[ind]
            heart_beats_dict['aami_label_str'] = aami_label_str
            heart_beats_dict['aami_label_ind'] = aami_label_ind
            heart_beats_dict['aami_label_one_hot'] = heartbeat_types.convert_to_one_hot(aami_label_ind)
            heart_beats_dict['beat_ind'] = ind
            heart_beats.append(heart_beats_dict)
        return heart_beats

    def get_heartbeats_of_type(self, aami_label_str):
        """

        :param aami_label_str:
        :return:
        """
        return [hb for hb in self.heartbeats if hb['aami_label_str'] == aami_label_str]

    def num_heartbeats(self, aami_label_str):
        """

        :param aami_label_str:
        :return:
        """
        return len(self.get_heartbeats_of_type(aami_label_str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # p_100 = Patient('101')
    # heartbeats = p_100.heartbeats
    #
    # logging.info("Total number of heartbeats: {}\t #N: {}\t #S: {}\t, #V: {}, #F: {}\t #Q: {}"
    #              .format(len(heartbeats), p_100.num_heartbeats('N'), p_100.num_heartbeats('S'), p_100.num_heartbeats('V'),
    #                      p_100.num_heartbeats('F'), p_100.num_heartbeats('Q')))

    # import wfdb


    # time = list(range(216))
    # for i in range(100):
    #     p = figure(x_axis_label='Sample number (360 Hz)', y_axis_label='Voltage[mV]')
    #     p.line(time, N_b[i], line_width=2, line_color="green")
    #     output_file("N_{}_real.html".format(i))
    #     show(p)
        # plt.plot(N_b[i])
        # plt.show()
