import torch.nn as nn
from ecg_pytorch.dynamical_model.Euler.euler import Euler
import torch
import numpy as np
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngpu, device):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.euler = Euler(device).to(device)

        self.fc1 = nn.Linear(50, 25)
        self.bn1 = nn.BatchNorm1d(num_features=25)
        self.fc2 = nn.Linear(25, 15)

        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, )

    def forward(self, noise_input, v0):

        x = F.leaky_relu(self.bn1(self.fc1(noise_input)), inplace=True)
        x = F.sigmoid(self.fc2(x))
        x = 0.1 * x + torch.Tensor([1.2, 0.25, - (1 / 3) * np.pi,
                                    -5.0, 0.1, - (1 / 12) * np.pi,
                                    20, 0.1, 0.0,
                                    -7.5, 0.1, (1 / 12) * np.pi,
                                    0.3, 0.4, (1 / 2) * np.pi])

        # input_params = x
        # input_params[:, 0] = input_params[:, 0] * 0.1 + 1.2
        # input_params[:, 3] = input_params[:, 3] * 0.1 - 5.0
        # input_params[:, 6] = input_params[:, 6] * 0.1 + 20
        # input_params[:, 9] = input_params[:, 9] * 0.1 - 7.5
        # input_params[:, 12] = input_params[:, 12] * 0.2 + 0.3
        #
        # input_params[:, 1] = input_params[:, 1] * 0.1 + 0.25
        # input_params[:, 4] = input_params[:, 4] * 0.1 + 0.1
        # input_params[:, 7] = input_params[:, 7] * 0.1 + 0.1
        # input_params[:, 10] = input_params[:, 10] * 0.1 + 0.1
        # input_params[:, 13] = input_params[:, 13] * 0.1 + 0.4
        #
        # input_params[:, 2] = input_params[:, 2] * 0.1 - (1 / 3) * np.pi
        # input_params[:, 5] = input_params[:, 5] * 0.1 - (1 / 12) * np.pi
        # input_params[:, 8] = input_params[:, 8] * 0.1
        # input_params[:, 11] = input_params[:, 11] * 0.1 + (1 / 12) * np.pi
        # input_params[:, 14] = input_params[:, 14] * 0.1 + (1 / 2) * np.pi
        # input_params = input_params.float()
        z_t = self.euler(x.float(), v0)
        # output shape: NX216
        # Try to add learnable noise layer:
        # z_t = z_t.view(-1, 1, 216)
        # z_t =
        return z_t


def scale_signal(ecg_signal, min_val=-0.01563, max_val=0.042557):
    """

    :param min:
    :param max:
    :return:
    """
    res = []
    for beat in ecg_signal:
        # Scale signal to lie between -0.4 and 1.2 mV :
        zmin = min(beat)
        zmax = max(beat)
        zrange = zmax - zmin
        scaled = [(z - zmin) * max_val / zrange + min_val for z in beat]
        scaled = torch.stack(scaled)
        res.append(scaled)
    res = torch.stack(res)
    return res


# class DeltaGenerator(nn.Module):
#     def __init__(self, ngpu, device):
#         super(DeltaGenerator, self).__init__()
#         self.ngpu = ngpu
#
#         self.fc1_z_delta = nn.Linear(in_features=50, out_features=1024)
#         self.bn1 = nn.BatchNorm1d(num_features=1024)
#         self.fc2_z_delta = nn.Linear(in_features=1024, out_features=54 * 128)
#         self.bn2 = nn.BatchNorm1d(num_features=54 * 128)
#         self.conv1_z_delta = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.conv2_z_delta = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
#
#     def forward(self, x):
#         x = F.elu(self.bn1(self.fc1_z_delta(x)))
#         x = F.elu(self.bn2(self.fc2_z_delta(x)))
#         x = x.view(-1, 128, 54)
#         x = F.relu(self.bn3(self.conv1_z_delta(x)))
#         x = (self.conv2_z_delta(x))
#         return x.view(-1, 216)


class DeltaGenerator(nn.Module):
    def __init__(self, ngpu):
        super(DeltaGenerator, self).__init__()
        self.ngpu = ngpu
        ngf = 64
        self.main = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(50, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 32),
            nn.ReLU(True),
            # shape in = [N, 64*4, 4]
            nn.ConvTranspose1d(ngf * 32, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            # shape in = [N, 64*2, 7]
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        input = input.view(-1, 50, 1)
        x = self.main(input)
        x = x.view(-1, 216)
        x = scale_signal(x)
        return x


class CombinedGenerator(nn.Module):
    def __init__(self, ngpu, device):
        super(CombinedGenerator, self).__init__()
        self.ngpu = ngpu
        self.ode_generator = Generator(ngpu, device)
        self.z_delta_generator = DeltaGenerator(ngpu)

    def forward(self, x, v0):
        z = self.ode_generator(x, v0)
        z_delta = self.z_delta_generator(x)

        total = z + z_delta
        return total


def test_generator():
    netG = Generator(0, "cpu")
    noise_input = torch.Tensor(np.random.normal(0, 1, (2, 50)))
    print("Noise shape: {}".format(noise_input.size()))

    out = netG(noise_input)
    print(out.shape)
    print(list(netG.parameters()))



if __name__ == "__main__":
    test_generator()