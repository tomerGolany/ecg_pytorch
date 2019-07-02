from collections import namedtuple

ECGTrainConfig = namedtuple('ECGTrainConfig', 'num_epochs batch_size lr weighted_loss weighted_sampling')
