from ecg_pytorch import train_configs

base_path = train_configs.base
#
# ODE GAN N:
#
# ODE_GAN_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_N_beat' \
#                 '/checkpoint_epoch_1_iters_600'

ODE_GAN_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_N_beat' \
                '/checkpoint_epoch_8_iters_2500'

# ODE_GAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_S_beat' \
#                 '/checkpoint_epoch_633_iters_3800'

ODE_GAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_S_beat' \
                '/checkpoint_epoch_166_iters_1000'

ODE_GAN_F_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_F_beat/' \
                            'checkpoint_epoch_933_iters_2800'

ODE_GAN_V_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_ode_gan_V_beat/' \
    'checkpoint_epoch_51_iters_1350'

#
# DCGAN:
#
DCGAN_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_N_beat/' \
                          'checkpoint_epoch_0_iters_401'

DCGAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan_s_beat/' \
                          'checkpoint_epoch_94_iters_1800'

DCGAN_V_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/dcgan_v_beat/' \
                          'checkpoint_epoch_21_iters_1600'

DCGAN_F_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/' \
                          'ecg_dcgan_F_beat/checkpoint_epoch_50_iters_451'

#
# Vanilla GAN:
#
VGAN_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanilla_gan_N_beat/checkpoint_epoch_1_iters_1400'

VGAN_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanilla_gan_s_beat/checkpoint_epoch_94_iters_1800'

VGAN_V_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanilla_gan_V_beat/checkpoint_epoch_21_iters_1600'

VGAN_F_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanilla_gan_F_beat/checkpoint_epoch_177_iters_1600'

#
# Vanilla GAN ODE:
#
VGAN_ODE_N_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanila_ode_gan_N_beat/checkpoint_epoch_15_iters_4800'

VGAN_ODE_S_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanila_ode_gan_S_beat/checkpoint_epoch_800_iters_4800'

VGAN_ODE_F_CHK = base_path +'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanila_ode_gan_F_beat/checkpoint_epoch_1600_iters_4800'

VGAN_ODE_V_CHK = base_path + 'ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_vanila_ode_gan_V_beat/checkpoint_epoch_200_iters_4800'

#
# Helper dict:
#
BEAT_AND_MODEL_TO_CHECKPOINT_PATH = {'N': {'DCGAN': DCGAN_N_CHK, 'ODE_GAN': ODE_GAN_N_CHK, 'VANILA_GAN': VGAN_N_CHK,
                                           'VANILA_GAN_ODE': VGAN_ODE_N_CHK},
                                     'S': {'DCGAN': DCGAN_S_CHK, 'ODE_GAN': ODE_GAN_S_CHK, 'VANILA_GAN': VGAN_S_CHK,
                                           'VANILA_GAN_ODE': VGAN_ODE_S_CHK},
                                     'V': {'DCGAN': DCGAN_V_CHK, 'ODE_GAN': ODE_GAN_V_CHK, 'VANILA_GAN': VGAN_V_CHK,
                                           'VANILA_GAN_ODE': VGAN_ODE_V_CHK},
                                     'F': {'DCGAN': DCGAN_F_CHK, 'ODE_GAN': ODE_GAN_F_CHK, 'VANILA_GAN': VGAN_F_CHK,
                                           'VANILA_GAN_ODE': VGAN_ODE_F_CHK}}


