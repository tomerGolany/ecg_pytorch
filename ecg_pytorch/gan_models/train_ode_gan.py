import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from ecg_pytorch.data_reader import ecg_dataset
from tensorboardX import SummaryWriter
from ecg_pytorch.gan_models.models import ode_gan
from ecg_pytorch.gan_models.models import dcgan
from ecg_pytorch.dynamical_model import equations
from ecg_pytorch.dynamical_model.ode_params import ODEParams
import math
import logging


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def euler_loss(hb_batch, params_batch, x_batch, y_batch, ode_params):
    """

    :param hb_batch: Nx216
    :param params_batch: Nx15
    :return:
    """
    logging.debug('hb batch shape: {}'.format(hb_batch.shape))
    logging.debug('params batch shape: {}'.format(params_batch.shape))
    logging.debug('x batch shape: {}'.format(x_batch.shape))
    logging.debug('y batch shape: {}'.format(y_batch.shape))

    delta_t = ode_params.h
    t = torch.tensor(0.0)
    f_ode_z_signal = None
    delta_hb_signal = None
    for i in range(215):
        delta_hb = (hb_batch[:, i + 1] - hb_batch[:, i]) / delta_t
        delta_hb = delta_hb.view(-1, 1)
        logging.debug("Delta heart-beat shape: {}".format(delta_hb.shape))
        y_t = y_batch[:, i].view(-1, 1)
        x_t = x_batch[:, i].view(-1, 1)
        z_t = hb_batch[:, i].view(-1, 1)
        f_ode_x = equations.d_x_d_t(y_t, x_t, t,  ode_params.rrpc, ode_params.h)
        f_ode_y = equations.d_y_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_z = equations.d_z_d_t(x_t, y_t, z_t, t, params_batch, ode_params)
        t += 1 / 512

        logging.debug("f ode z shape {}".format(f_ode_z.shape))  # Nx1
        logging.debug("f ode x shape {}".format(f_ode_x.shape))
        logging.debug("f ode y shape {}".format(f_ode_y.shape))
        if f_ode_z_signal is None:
            f_ode_z_signal = f_ode_z
            delta_hb_signal = delta_hb
        else:
            f_ode_z_signal = torch.cat((f_ode_z_signal, f_ode_z), 1)
            delta_hb_signal = torch.cat((delta_hb_signal, delta_hb), 1)

    logging.debug("f signal shape: {}".format(f_ode_z_signal.shape))
    logging.debug("delta hb signal shape: {}".format(delta_hb_signal.shape))

    return delta_hb_signal, f_ode_z_signal


def test_euler_loss():
    ode_params = ODEParams('cpu')

    # input = np.array((
    #     ([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
    #       30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
    #       90.0 * math.pi / 180.0]))).reshape((1, 15))
    # a1 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    # a2 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    # input_params = torch.cat((a1, a2), 0)
    # print("Input params shape: {}".format(input_params.shape))
    # x = torch.tensor([-0.417750770388669, -0.417750770388669]).view(2, 1)
    # y = torch.tensor([-0.9085616622823985, -0.9085616622823985]).view(2, 1)
    # z = torch.tensor([-0.004551233843726818, 0.03]).view(2, 1)
    # t = torch.tensor(0.0)
    pass


def train(batch_size, num_train_steps, model_dir):
    ode_params = ODEParams('cpu')
    # Support for tensorboard:
    writer = SummaryWriter(model_dir)
    # 1. create the ECG dataset:
    composed = transforms.Compose([ecg_dataset.ToTensor()])
    dataset = ecg_dataset.EcgHearBeatsDataset(transform=composed, beat_type='N')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)
    print("Size of real dataset is {}".format(len(dataset)))

    # 2. Create the models:
    netG = ode_gan.ODEGenerator()
    netD = dcgan.DCDiscriminator(0)
    netD.apply(weights_init)
    ode_params_g = netG.ode_params_generator
    convG = netG.conv_generator
    convG.apply(weights_init)

    # Loss functions:
    cross_entropy_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # Optimizers:
    lr = 0.0002
    beta1 = 0.5
    writer.add_scalar('Learning_Rate', lr)
    optimizer_d = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Noise for validation:
    val_noise = torch.Tensor(np.random.normal(0, 1, (4, 100)))
    epoch = 0
    iters = 0
    while True:
        num_of_beats_seen = 0
        if iters == num_train_steps:
            break
        for i, data in enumerate(dataloader):
            if iters == num_train_steps:
                break

            netD.zero_grad()
            ecg_batch = data['cardiac_cycle'].float()
            b_size = ecg_batch.shape[0]

            num_of_beats_seen += ecg_batch.shape[0]
            output = netD(ecg_batch)
            labels = torch.full((b_size,), 1, device='cpu')

            ce_loss_d_real = cross_entropy_loss(output, labels)
            writer.add_scalar('Discriminator/cross_entropy_on_real_batch', ce_loss_d_real.item(), global_step=iters)
            writer.add_scalars('Merged/losses', {'d_cross_entropy_on_real_batch': ce_loss_d_real.item()},
                               global_step=iters)
            ce_loss_d_real.backward()
            mean_d_real_output = output.mean().item()
            noise_input = torch.Tensor(np.random.normal(0, 1, (b_size, 100)))

            output_ode_params, output_g_fake, x_signal, y_signal = netG(noise_input)
            output = netD(output_g_fake.detach())
            labels.fill_(0)

            ce_loss_d_fake = cross_entropy_loss(output, labels)
            writer.add_scalar('Discriminator/cross_entropy_on_fake_batch', ce_loss_d_fake.item(), iters)
            writer.add_scalars('Merged/losses', {'d_cross_entropy_on_fake_batch': ce_loss_d_fake.item()},
                               global_step=iters)
            ce_loss_d_fake.backward()

            mean_d_fake_output = output.mean().item()
            total_loss_d = ce_loss_d_fake + ce_loss_d_real
            writer.add_scalar(tag='Discriminator/total_loss', scalar_value=total_loss_d.item(),
                              global_step=iters)
            optimizer_d.step()

            netG.zero_grad()
            labels.fill_(1)
            output = netD(output_g_fake)

            ##
            # Add euler loss:
            ##
            delta_hb_signal, f_ode_z_signal = euler_loss(output_g_fake, output_ode_params, x_signal, y_signal, ode_params)
            mse_loss_euler = mse_loss(delta_hb_signal, f_ode_z_signal)

            ce_loss_g_fake = cross_entropy_loss(output, labels)
            total_g_loss = mse_loss_euler + ce_loss_g_fake
            total_g_loss.backward()

            writer.add_scalar(tag='Generator/mse_ode', scalar_value=mse_loss_euler.item(), global_step=iters)
            writer.add_scalar(tag='Generator/cross_entropy_on_fake_batch', scalar_value=ce_loss_g_fake.item(),
                              global_step=iters)
            writer.add_scalars('Merged/losses', {'g_cross_entropy_on_fake_batch': ce_loss_g_fake.item()},
                               global_step=iters)
            mean_d_fake_output_2 = output.mean().item()

            optimizer_g.step()

            print("{}/{}: Epoch #{}: Iteration #{}: Mean D(real_hb_batch) = {}, mean D(G(z)) = {}."
                  .format(num_of_beats_seen, len(dataset), epoch, iters, mean_d_real_output, mean_d_fake_output),
                  end=" ")
            print("mean D(G(z)) = {} After backprop of D".format(mean_d_fake_output_2))

            print("Loss D from real beats = {}. Loss D from Fake beats = {}. Total Loss D = {}".
                  format(ce_loss_d_real, ce_loss_d_fake, total_loss_d), end=" ")
            print("Loss G = {}".format(ce_loss_g_fake))


            # Norma of gradients:
            gNormGrad = get_gradient_norm_l2(netG)
            dNormGrad = get_gradient_norm_l2(netD)
            writer.add_scalar('Generator/gradients_norm', gNormGrad, iters)
            writer.add_scalar('Discriminator/gradients_norm', dNormGrad, iters)
            print(
                "Generator Norm of gradients = {}. Discriminator Norm of gradients = {}.".format(gNormGrad, dNormGrad))

            if iters % 25 == 0:
                with torch.no_grad():
                    with torch.no_grad():
                        _ , output_g, _, _ = netG(val_noise)
                        fig = plt.figure()
                        plt.title("Fake beats from Generator. iteration {}".format(i))
                        for p in range(4):
                            plt.subplot(2, 2, p + 1)
                            plt.plot(output_g[p].detach().numpy(), label="fake beat")
                            plt.plot(ecg_batch[p].detach().numpy(), label="real beat")
                            plt.legend()
                        writer.add_figure('Generator/output_example', fig, iters)
                        plt.close()

            if iters % 200 == 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': netG.state_dict(),
                    'discriminator_state_dict': netD.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'loss': cross_entropy_loss,

                }, model_dir + '/checkpoint_epoch_{}_iters_{}'.format(epoch, iters))
            iters += 1
        epoch += 1
    writer.close()


def get_gradient_norm_l2(model):
    total_norm = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'ode_params_generator' not in name:
            # print(name)
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    # for p in model.parameters():
    #
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** (1. / 2)
    return total_norm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_dir = 'tensorboard/ode_gan/ecg_ode_gan_N_beat_temp_temp'
    train(50, 2000, model_dir)