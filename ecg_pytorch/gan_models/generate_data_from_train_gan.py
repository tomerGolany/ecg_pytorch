import torch
import numpy as np
from ecg_pytorch.gan_models.models import dcgan
from ecg_pytorch.gan_models.models import ode_gan_aaai
from ecg_pytorch.gan_models import checkpoint_paths
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show


def generate_data_from_trained_gan(generator_model, num_of_samples_to_generate, checkpoint_path):
    """

    :param generator_model:
    :param num_of_samples_to_generate:
    :param checkpoint_path:
    :return:
    """
    checkpoint = torch.load(checkpoint_path,  map_location='cpu')
    generator_model.load_state_dict(checkpoint['generator_state_dict'])
    generator_model.eval()
    # discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
    with torch.no_grad():
        input_noise = torch.Tensor(np.random.normal(0, 1, (num_of_samples_to_generate, 100)))
        output_g = generator_model(input_noise)
        output_g = output_g.numpy()
        return output_g

"""
checkpoint = torch.load(checkpoint_path)
        generator_model.load_state_dict(checkpoint['generator_state_dict'])
        # discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
        with torch.no_grad():
            input_noise = torch.Tensor(np.random.normal(0, 1, (num_beats_to_add, 100)))
            output_g = generator_model(input_noise)
            output_g = output_g.numpy()
"""


def generate_N_beat_from_DCSimGAN():
    chk_path = checkpoint_paths.BEAT_AND_MODEL_TO_CHECKPOINT_PATH['N']['ODE_GAN']
    generator_model = ode_gan_aaai.DCGenerator(0)
    fake_beat = generate_data_from_trained_gan(generator_model, 1, chk_path)
    return fake_beat

def generate_S_beat_from_DCSimGAN():
    chk_path = checkpoint_paths.BEAT_AND_MODEL_TO_CHECKPOINT_PATH['S']['ODE_GAN']
    generator_model = ode_gan_aaai.DCGenerator(0)
    fake_beat = generate_data_from_trained_gan(generator_model, 1, chk_path)
    return fake_beat

def generate_V_beat_from_DCSimGAN():
    chk_path = checkpoint_paths.BEAT_AND_MODEL_TO_CHECKPOINT_PATH['V']['ODE_GAN']
    generator_model = ode_gan_aaai.DCGenerator(0)
    fake_beat = generate_data_from_trained_gan(generator_model, 1, chk_path)
    return fake_beat

def generate_F_beat_from_DCSimGAN():
    chk_path = checkpoint_paths.BEAT_AND_MODEL_TO_CHECKPOINT_PATH['F']['ODE_GAN']
    generator_model = ode_gan_aaai.DCGenerator(0)
    fake_beat = generate_data_from_trained_gan(generator_model, 1, chk_path)
    return fake_beat

def generate_N_beat_from_DCGAN():
    chk_path = checkpoint_paths.BEAT_AND_MODEL_TO_CHECKPOINT_PATH['N']['DCGAN']
    generator_model = dcgan.DCGenerator(0)
    fake_beat = generate_data_from_trained_gan(generator_model, 1, chk_path)
    return fake_beat

def plot_beat(b):
    p = figure(x_axis_label='Sample number (360 Hz)', y_axis_label='Voltage[mV]')
    time = np.arange(0, 216)
    p.line(time, b[0], line_width=2, line_color="black")
    output_file("N_DCGAN.html")
    # p.legend.location = "bottom_right"
    show(p)



if __name__ == "__main__":
    # generator_model = dcgan.DCGenerator(0)
    # one_beat = generate_data_from_trained_gan(generator_model, 1, '/Users/tomer.golany/PycharmProjects/ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan/checkpoint_epoch_0_iters_201')
    # print(one_beat.shape)
    # print(type(one_beat))
    # plt.plot(one_beat[0].numpy())
    # plt.show()

    fake_beat = generate_N_beat_from_DCGAN()
    plot_beat(fake_beat)