import torch.nn as nn
import torch.nn.functional as F


class DCGenerator(nn.Module):
    def __init__(self):
        super(DCGenerator, self).__init__()
        ngf = 64
        ##
        # Conv Transpose 1d:
        # Input: (N, Cin, Lin) --> Output: (N, Cout, Lout)
        # Lout = (Lin -1) * s -2 * p + k
        ##
        self.main = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(100, ngf * 32, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, x):
        x = x.view(-1, 100, 1)
        x = self.main(x)
        x = x.view(-1, 216)
        return x


class DCXGenerator(nn.Module):
    def __init__(self):
        super(DCXGenerator, self).__init__()
        ngf = 64
        ##
        # Conv Transpose 1d:
        # Input: (N, Cin, Lin) --> Output: (N, Cout, Lout)
        # Lout = (Lin -1) * s -2 * p + k
        ##
        self.main = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(100, ngf * 32, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, x):
        x = x.view(-1, 100, 1)
        x = self.main(x)
        x = x.view(-1, 216)
        return x


class DCYGenerator(nn.Module):
    def __init__(self):
        super(DCYGenerator, self).__init__()
        ngf = 64
        ##
        # Conv Transpose 1d:
        # Input: (N, Cin, Lin) --> Output: (N, Cout, Lout)
        # Lout = (Lin -1) * s -2 * p + k
        ##
        self.main = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(100, ngf * 32, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, x):
        x = x.view(-1, 100, 1)
        x = self.main(x)
        x = x.view(-1, 216)
        return x


class ODEParamsGenerator(nn.Module):
    def __init__(self):
        super(ODEParamsGenerator, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(50, 25)
        self.bn2 = nn.BatchNorm1d(num_features=25)
        self.fc3 = nn.Linear(25, 15)

    def forward(self, noise_input):

        x = F.leaky_relu(self.bn1(self.fc1(noise_input)), inplace=True)
        x = F.sigmoid(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class ODEGenerator(nn.Module):
    def __init__(self):
        super(ODEGenerator, self).__init__()

        self.ode_params_generator = ODEParamsGenerator()
        self.conv_generator = DCGenerator()

        self.x_signal_generator = DCXGenerator()
        self.y_signal_generator = DCYGenerator()

    def forward(self, noise_input):
        fake_ecg = self.conv_generator(noise_input)
        ode_params = self.ode_params_generator(noise_input)

        x_signal =self.x_signal_generator(noise_input)
        y_signal = self.y_signal_generator(noise_input)

        return ode_params, fake_ecg, x_signal, y_signal
