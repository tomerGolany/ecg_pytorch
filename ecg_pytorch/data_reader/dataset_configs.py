"""Module to define available dataset configurations."""
from enum import Enum
from ecg_pytorch.data_reader import heartbeat_types


class PartitionNames(Enum):
    train = 0
    test = 1


class DatasetConfigs(object):
    def __init__(self, partition, classified_heartbeat, one_vs_all, lstm_setting, over_sample_minority_class,
                 under_sample_majority_class, only_take_heartbeat_of_type):
        """Create a new DatasetConfigs object which has configurations on how to create the ecg dataset.

        :param partition: To which dataset partition it belongs. can be train or test.
        :param classified_heartbeat: Which heartbeat is classified. Only used for One vs. All settings.
        :param one_vs_all: boolean that indicates weather we are using multi-class classification or one vs. all.
        :param lstm_setting: boolean that indicates weather we are using lstm setting in the model or not.
        :param over_sample_minority_class: boolean that indicates if the dataset should over sample the minority class.
        Only used for one Vs. all setting.
        :param under_sample_majority_class: boolean that indicates if the dataset should under sample the majority class.
        Only used for ons vs. all setting.
        :param only_take_heartbeat_of_type: if not None, the dataset should contain only heartbeats from the specified
        type.
        """
        if partition not in PartitionNames.__members__:
            raise ValueError("Invalid partition name: {}".format(partition))

        if classified_heartbeat not in heartbeat_types.AAMIHeartBeatTypes.__members__:
            raise ValueError("Invalid target heartbeat: {}".format(classified_heartbeat))

        if one_vs_all not in [True, False]:
            raise ValueError("Expected boolean type for argument one vs. all. Got instead: {}".format(one_vs_all))

        if lstm_setting not in [True, False]:
            raise ValueError("Expected boolean type for argument lstm setting. Got instead: {}".format(lstm_setting))

        if over_sample_minority_class not in [True, False]:
            raise ValueError("Expected boolean type for argument over sample minority class. "
                             "Got instead: {}".format(over_sample_minority_class))

        if under_sample_majority_class not in [True, False]:
            raise ValueError("Expected boolean type for argument under sample majority class. "
                             "Got instead: {}".format(under_sample_majority_class))

        if only_take_heartbeat_of_type is None:
            self.only_take_heartbeat_of_type = only_take_heartbeat_of_type

        elif only_take_heartbeat_of_type in heartbeat_types.AAMIHeartBeatTypes.__members__:
            self.only_take_heartbeat_of_type = only_take_heartbeat_of_type

        elif only_take_heartbeat_of_type == heartbeat_types.OTHER_HEART_BEATS:
            self.only_take_heartbeat_of_type = only_take_heartbeat_of_type
        else:
            raise ValueError("Invalid argument for parameter only take heartbeat of type: {}"
                             .format(only_take_heartbeat_of_type))

        self.partition = partition

        self.classified_heartbeat = classified_heartbeat

        self.one_vs_all = one_vs_all

        self.lstm_setting = lstm_setting

        self.over_sample_minority_class = over_sample_minority_class

        self.under_sample_majority_class = under_sample_majority_class



