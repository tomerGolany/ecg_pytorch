import torch
import numpy as np
from ecg_pytorch.gan_models.models import dcgan
import matplotlib.pyplot as plt


def generate_data_from_trained_gan(generator_model, num_of_samples_to_generate, checkpoint_path):
    """

    :param generator_model:
    :param num_of_samples_to_generate:
    :param checkpoint_path:
    :return:
    """
    checkpoint = torch.load(checkpoint_path)
    generator_model.load_state_dict(checkpoint['generator_state_dict'])
    # discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
    with torch.no_grad():
        input_noise = torch.Tensor(np.random.normal(0, 1, (num_of_samples_to_generate, 100)))
        output_g = generator_model(input_noise)
        return output_g


if __name__ == "__main__":
    generator_model = dcgan.DCGenerator(0)
    one_beat = generate_data_from_trained_gan(generator_model, 1, '/Users/tomer.golany/PycharmProjects/ecg_pytorch/ecg_pytorch/gan_models/tensorboard/ecg_dcgan/checkpoint_epoch_0_iters_201')
    print(one_beat.shape)
    print(type(one_beat))
    plt.plot(one_beat[0].numpy())
    plt.show()