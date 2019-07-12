base_local = '/Users/tomer.golany/PycharmProjects/'
base_remote = '/home/tomer.golany@st.technion.ac.il/'

use_type = 'LOCAL'
if use_type == 'LOCAL':
    base_path = base_local
else:
    base_path = base_remote

#
# ODE GAN N:
#
ODE_GAN_N_CHK = '/home/tomer.golany@st.technion.ac.il/ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ode_gan_aaai/' \
                'ecg_ode_gan_N_beat/checkpoint_epoch_9_iters_4200'

#
# DCGAN:
#
DCGAN_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_N_beat/' \
                          'checkpoint_epoch_0_iters_401'

DCGAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_S_beat/' \
                          'checkpoint_epoch_46_iters_829'

DCGAN_V_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_V_beat/' \
                          'checkpoint_epoch_15_iters_1066'

DCGAN_F_CHK = base_path + '/home/tomer.golany@st.technion.ac.il/ecg_pytorch/ecg_pytorch/gan_models/tensorboard/' \
                          'ecg_dcgan_F_beat/checkpoint_epoch_50_iters_451'
