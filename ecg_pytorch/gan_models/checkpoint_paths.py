base_local = '/Users/tomer.golany/PycharmProjects/'
base_remote = '/home/tomer.golany@st.technion.ac.il/'
base_niv_remote = '/home/nivgiladi/tomer/'
base_tomer_remote = '/home/tomer/tomer/'

use_type = 'TOMER'
if use_type == 'LOCAL':
    base_path = base_local
elif use_type == 'NLP':
    base_path = base_remote
elif use_type == "NIV":
    base_path = base_niv_remote
elif use_type == "TOMER":
    base_path = base_tomer_remote
#
# ODE GAN N:
#
ODE_GAN_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_N_beat' \
                '/checkpoint_epoch_1_iters_600'

# ODE_GAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_S_beat' \
#                 '/checkpoint_epoch_633_iters_3800'

ODE_GAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_S_beat' \
                '/checkpoint_epoch_166_iters_1000'

ODE_GAN_F_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_F_beat/' \
                            'checkpoint_epoch_933_iters_2800'

#
# DCGAN:
#
DCGAN_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_N_beat/' \
                          'checkpoint_epoch_0_iters_401'

DCGAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_S_beat/' \
                          'checkpoint_epoch_46_iters_829'

DCGAN_V_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_V_beat/' \
                          'checkpoint_epoch_15_iters_1066'

DCGAN_F_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/' \
                          'ecg_dcgan_F_beat/checkpoint_epoch_50_iters_451'

#
# Helper dict:
#
BEAT_AND_MODEL_TO_CHECKPOINT_PATH = {'N': {'DCGAN': DCGAN_N_CHK, 'ODE_GAN': ODE_GAN_N_CHK},
                                     'S': {'DCGAN': DCGAN_S_CHK, 'ODE_GAN': ODE_GAN_S_CHK},
                                     'V': {'DCGAN': DCGAN_V_CHK},
                                     'F': {'DCGAN': DCGAN_F_CHK, 'ODE_GAN': ODE_GAN_F_CHK}}


