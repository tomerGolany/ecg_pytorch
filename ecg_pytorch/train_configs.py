from collections import namedtuple

ECGTrainConfig = namedtuple('ECGTrainConfig',
                            'num_epochs batch_size lr weighted_loss weighted_sampling device add_data_'
                            'from_gan generator_details')

GeneratorAdditionalDataConfig = namedtuple('GeneratorAdditionalDataConfig', 'beat_type checkpoint_path num_examples_to_'
                                                                            'add gan_type')
