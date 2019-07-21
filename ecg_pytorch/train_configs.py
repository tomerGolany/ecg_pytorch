from collections import namedtuple

niv_remote = '/home/nivgiladi/tomer/'
local_base = '/Users/tomer.golany/PycharmProjects/'
nlp_base = '/home/tomer.golany@st.technion.ac.il/'
tomer_remote = '/home/tomer/tomer/'
niv2_remote = '/home/niv/tomer/'
yochai_remote = '/home/yochaiz/tomergolany'
base = yochai_remote

ECGTrainConfig = namedtuple('ECGTrainConfig',
                            'num_epochs batch_size lr weighted_loss weighted_sampling device add_data_'
                            'from_gan generator_details train_one_vs_all')

GeneratorAdditionalDataConfig = namedtuple('GeneratorAdditionalDataConfig', 'beat_type checkpoint_path num_examples_to_'
                                                                            'add gan_type')
