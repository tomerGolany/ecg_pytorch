import torch.nn as nn
from ecg_pytorch.dynamical_model.Euler.euler import Euler
import torch
import numpy as np
import torch.nn.functional as F


class ODEGenerator(nn.Module):
    def __init__(self, ngpu, device):
        super(ODEGenerator, self).__init__()
        self.ngpu = ngpu
        self.euler = Euler(device).to(device)
        self.fc1 = nn.Linear(50, 25)
        self.bn1 = nn.BatchNorm1d(num_features=25)
        self.fc2 = nn.Linear(25, 15)

    def forward(self, noise_input, v0):

        x = F.leaky_relu(self.bn1(self.fc1(noise_input)), inplace=True)
        x = F.sigmoid(self.fc2(x))
        x = 0.1 * x + torch.Tensor([1.2, 0.25, - (1 / 3) * np.pi,
                                    -5.0, 0.1, - (1 / 12) * np.pi,
                                    20, 0.1, 0.0,
                                    -7.5, 0.1, (1 / 12) * np.pi,
                                    0.3, 0.4, (1 / 2) * np.pi])

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
        self.ode_generator = ODEGenerator(ngpu, device)
        self.z_delta_generator = DeltaGenerator(ngpu)

    def forward(self, x, v0):
        z = self.ode_generator(x, v0)
        z_delta = self.z_delta_generator(x)

        total = z + z_delta
        return total


def test_generator():
    netG = ODEGenerator(0, "cpu")
    noise_input = torch.Tensor(np.random.normal(0, 1, (2, 50)))
    print("Noise shape: {}".format(noise_input.size()))

    out = netG(noise_input)
    print(out.shape)
    print(list(netG.parameters()))


if __name__ == "__main__":
    test_generator()