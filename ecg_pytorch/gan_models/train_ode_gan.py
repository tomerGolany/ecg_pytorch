import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from ecg_pytorch.data_reader import ecg_dataset_lstm
from tensorboardX import SummaryWriter
from ecg_pytorch.gan_models.models import ode_gan
from ecg_pytorch.gan_models.models import ode_gan_aaai
from ecg_pytorch.dynamical_model import equations
from ecg_pytorch.dynamical_model.ode_params import ODEParams
import math
import logging
import pickle
from ecg_pytorch.dynamical_model import typical_beat_params
from ecg_pytorch.gan_models.models import vanila_gan
from bokeh.plotting import figure, output_file, show, save
from ecg_pytorch.data_reader import smooth_signal

TYPICAL_ODE_N_PARAMS = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                      160.0 * math.pi / 180.0]


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_typical_N_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(TYPICAL_ODE_N_PARAMS).to(device)
    return params


def generate_typical_S_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_S_PARAMS).to(device)
    return params


def generate_typical_F_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_F_PARAMS).to(device)
    return params

def generate_typical_V_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_V_PARAMS).to(device)
    return params


def ode_loss(hb_batch, ode_params, device, beat_type):
    """

    :param hb_batch:
    :return:
    """
    # print(hb_batch[:, 0])
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    delta_t = ode_params.h
    # params_one = [1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
    #                   30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
    #                   90.0 * math.pi / 180.0]
    # params_one = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
    #                   30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
    #                   160.0 * math.pi / 180.0]
    batch_size = hb_batch.size()[0]
    # params_batch = np.array([params_one for _ in range(batch_size)]).reshape((batch_size, 15))
    # params_batch = torch.Tensor(params_batch)

    if beat_type == "N":
        params_batch = generate_typical_N_ode_params(batch_size, device)
    elif beat_type == "S":
        params_batch = generate_typical_S_ode_params(batch_size, device)
    elif beat_type == 'F':
        params_batch = generate_typical_F_ode_params(batch_size, device)
    elif beat_type == 'V':
        params_batch = generate_typical_V_ode_params(batch_size, device)
    else:
        raise NotImplementedError()

    logging.debug("params batch shape: {}".format(params_batch.size()))
    x_t = torch.tensor(-0.417750770388669).to(device)
    y_t = torch.tensor(-0.9085616622823985).to(device)
    t = torch.tensor(0.0).to(device)
    f_ode_z_signal = None
    delta_hb_signal = None
    for i in range(215):
        # if i == 0:
        #     init_z = -0.004551233843726818
        #     init_z =torch.Tensor(np.array([init_z for _ in range(batch_size)])).to(device)
        #     delta_hb = (hb_batch[:, i + 1] - init_z) / delta_t
        # else:
        delta_hb = (hb_batch[:, i + 1] - hb_batch[:, i]) / delta_t
        delta_hb = delta_hb.view(-1, 1)
        z_t = hb_batch[:, i].view(-1, 1)

        f_ode_x = equations.d_x_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_y = equations.d_y_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_z = equations.d_z_d_t(x_t, y_t, z_t, t, params_batch, ode_params)

        logging.debug("f ode z shape {}".format(f_ode_z.shape))  # Nx1
        logging.debug("f ode x shape {}".format(f_ode_x.shape))
        logging.debug("f ode y shape {}".format(f_ode_y.shape))

        y_t = y_t + delta_t * f_ode_y
        x_t = x_t + delta_t * f_ode_x
        t += 1 / 360

        if f_ode_z_signal is None:
            f_ode_z_signal = f_ode_z
            delta_hb_signal = delta_hb
        else:
            f_ode_z_signal = torch.cat((f_ode_z_signal, f_ode_z), 1)
            delta_hb_signal = torch.cat((delta_hb_signal, delta_hb), 1)
    logging.debug("f signal shape: {}".format(f_ode_z_signal.shape))
    logging.debug("delta hb signal shape: {}".format(delta_hb_signal.shape))
    return delta_hb_signal, f_ode_z_signal


def test_ode_loss():
    with open('/Users/tomer.golany/PycharmProjects/ecg_pytorch/ecg_pytorch/dynamical_model/ode_normal_sample.pkl', 'rb') as handle:
        good_sample = pickle.load(handle)
    good_sample = np.array(good_sample).reshape((1, 216))
    good_sample = torch.Tensor(good_sample)
    ode_params = ODEParams('cpu')
    delta_hb_signal, f_ode_z_signal = ode_loss(good_sample, ode_params)
    print(delta_hb_signal - f_ode_z_signal)
    mse_loss = nn.MSELoss()
    loss = mse_loss(delta_hb_signal, f_ode_z_signal)
    print("LOSS: ", loss)


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
    f_ode_x_signal = None
    f_ode_y_signal = None
    delta_hb_signal = None
    delta_x_signal = None
    delta_y_signal = None
    for i in range(215):
        delta_hb = (hb_batch[:, i + 1] - hb_batch[:, i]) / delta_t
        delta_y = (y_batch[:, i + 1] - y_batch[:, i]) / delta_t
        delta_x = (x_batch[:, i + 1] - x_batch[:, i]) / delta_t
        delta_hb = delta_hb.view(-1, 1)
        delta_x = delta_x.view(-1, 1)
        delta_y = delta_y.view(-1, 1)
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
            f_ode_x_signal = f_ode_x
            f_ode_y_signal = f_ode_y
            delta_hb_signal = delta_hb
            delta_x_signal = delta_x
            delta_y_signal = delta_y
        else:
            f_ode_z_signal = torch.cat((f_ode_z_signal, f_ode_z), 1)
            f_ode_x_signal = torch.cat((f_ode_x_signal, f_ode_x), 1)
            f_ode_y_signal = torch.cat((f_ode_y_signal, f_ode_y), 1)
            delta_hb_signal = torch.cat((delta_hb_signal, delta_hb), 1)
            delta_x_signal = torch.cat((delta_x_signal, delta_x), 1)
            delta_y_signal = torch.cat((delta_y_signal, delta_y), 1)


    logging.debug("f signal shape: {}".format(f_ode_z_signal.shape))
    logging.debug("delta hb signal shape: {}".format(delta_hb_signal.shape))

    return delta_hb_signal, f_ode_z_signal, f_ode_x_signal, f_ode_y_signal, delta_x_signal, delta_y_signal


def train(batch_size, num_train_steps, model_dir, beat_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ode_params = ODEParams(device)

    #
    # Support for tensorboard:
    #
    writer = SummaryWriter(model_dir)

    #
    # 1. create the ECG dataset:
    #
    composed = transforms.Compose([ecg_dataset_lstm.Scale(), ecg_dataset_lstm.ToTensor()])
    dataset = ecg_dataset_lstm.EcgHearBeatsDataset(transform=composed, beat_type=beat_type, lstm_setting=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)
    print("Size of real dataset is {}".format(len(dataset)))

    #
    # 2. Create the models:

    # netG = ode_gan_aaai.DCGenerator(0).to(device)
    # netD = ode_gan_aaai.DCDiscriminator(0).to(device)
    # netD.apply(weights_init)
    # netG.apply(weights_init)
    netG = vanila_gan.VGenerator(0).to(device)
    netD = vanila_gan.VDiscriminator(0).to(device)

    #
    # Define loss functions:
    #
    cross_entropy_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    #
    # Optimizers:
    #
    lr = 0.0002
    beta1 = 0.5
    writer.add_scalar('Learning_Rate', lr)
    optimizer_d = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    #
    # Noise for validation:
    #
    val_noise = torch.Tensor(np.random.uniform(0, 1, (4, 100))).to(device)
    # val_noise = torch.Tensor(np.random.normal(0, 1, (4, 100))).to(device)

    #
    # Training loop"
    #
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

            #
            # Discriminator from real beats:
            #
            ecg_batch = data['cardiac_cycle'].float().to(device)
            b_size = ecg_batch.shape[0]

            num_of_beats_seen += ecg_batch.shape[0]
            output = netD(ecg_batch)
            labels = torch.full((b_size,), 1, device=device)

            ce_loss_d_real = cross_entropy_loss(output, labels)
            writer.add_scalar('Discriminator/cross_entropy_on_real_batch', ce_loss_d_real.item(), global_step=iters)
            writer.add_scalars('Merged/losses', {'d_cross_entropy_on_real_batch': ce_loss_d_real.item()},
                               global_step=iters)
            ce_loss_d_real.backward()
            mean_d_real_output = output.mean().item()

            #
            # Discriminator from fake beats:
            #
            noise_input = torch.Tensor(np.random.uniform(0, 1, (b_size, 100))).to(device)
            # noise_input = torch.Tensor(np.random.normal(0, 1, (b_size, 100))).to(device)

            output_g_fake = netG(noise_input)
            output = netD(output_g_fake.detach()).to(device)
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

            #
            # Add euler loss:
            #
            delta_hb_signal, f_ode_z_signal = ode_loss(output_g_fake, ode_params, device, beat_type)
            mse_loss_euler = mse_loss(delta_hb_signal, f_ode_z_signal)
            logging.info("MSE ODE loss: {}".format(mse_loss_euler.item()))
            ce_loss_g_fake = cross_entropy_loss(output, labels)
            total_g_loss = mse_loss_euler + ce_loss_g_fake
            # total_g_loss = mse_loss_euler
            total_g_loss.backward()

            writer.add_scalar(tag='Generator/mse_ode', scalar_value=mse_loss_euler.item(), global_step=iters)
            writer.add_scalar(tag='Generator/cross_entropy_on_fake_batch', scalar_value=ce_loss_g_fake.item(),
                              global_step=iters)
            writer.add_scalars('Merged/losses', {'g_cross_entropy_on_fake_batch': ce_loss_g_fake.item()},
                               global_step=iters)
            mean_d_fake_output_2 = output.mean().item()

            optimizer_g.step()

            if iters % 50 == 0:
                print("{}/{}: Epoch #{}: Iteration #{}: Mean D(real_hb_batch) = {}, mean D(G(z)) = {}."
                      .format(num_of_beats_seen, len(dataset), epoch, iters, mean_d_real_output, mean_d_fake_output),
                      end=" ")
                print("mean D(G(z)) = {} After backprop of D".format(mean_d_fake_output_2))

                print("Loss D from real beats = {}. Loss D from Fake beats = {}. Total Loss D = {}".
                      format(ce_loss_d_real, ce_loss_d_fake, total_loss_d), end=" ")
                print("Loss G = {}".format(ce_loss_g_fake))

            #
            # Norma of gradients:
            #
            gNormGrad = get_gradient_norm_l2(netG)
            dNormGrad = get_gradient_norm_l2(netD)
            writer.add_scalar('Generator/gradients_norm', gNormGrad, iters)
            writer.add_scalar('Discriminator/gradients_norm', dNormGrad, iters)
            print(
                "Generator Norm of gradients = {}. Discriminator Norm of gradients = {}.".format(gNormGrad, dNormGrad))

            if iters % 25 == 0:
                with torch.no_grad():
                    with torch.no_grad():
                        output_g = netG(val_noise)
                        fig = plt.figure()
                        plt.title("Fake beats from Generator. iteration {}".format(i))
                        for p in range(4):
                            plt.subplot(2, 2, p + 1)
                            plt.plot(output_g[p].cpu().detach().numpy(), label="fake beat")
                            plt.plot(ecg_batch[p].cpu().detach().numpy(), label="real beat")
                            plt.legend()
                        writer.add_figure('Generator/output_example', fig, iters)
                        plt.close()

                    #
                    # Add bokeh plot:
                    #
                    p = figure(x_axis_label='Sample number (360 Hz)', y_axis_label='Voltage[mV]')
                    time = np.arange(0, 216)
                    fake_beat = output_g[0].cpu().detach().numpy()
                    w = 'hanning'
                    smoothed_beat = smooth_signal.smooth(fake_beat, 10, w)
                    p.line(time, smoothed_beat, line_width=2, line_color="blue")
                    output_file("N_{}_ODE.html".format(iters))
                    save(p)

            if iters % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': netG.state_dict(),
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
    model_dir = 'tensorboard/ecg_vanila_ode_gan_S_beat/'
    beat_type = 'S'
    train(150, 5000, model_dir, beat_type)
