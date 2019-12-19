"""Residual 1d convolution network

Based on: "ECG Heartbeat Classification: A Deep Transferable
Representation" by : Mohammad Kachuee and Shayan Fazeli.

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8419425

"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class ResidualLayer(nn.Module):
    def __init__(self):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, bias=True)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
        x = self.conv2(F.relu(self.bn1(self.conv1(x)))) + x

        x = F.relu(self.pool1(self.bn2(x)))

        return x


class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2, bias=True)
        self.bn1 = nn.BatchNorm1d(32)
        self.res1 = ResidualLayer()
        self.res2 = ResidualLayer()
        self.res3 = ResidualLayer()
        self.res4 = ResidualLayer()
        self.res5 = ResidualLayer()

        self.fc1 = nn.Linear(32 * 3, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.view(-1, 1, 216)
        # print(x.shape)
        x = self.conv1(x)

        x = self.bn1(x)
        # print(x.shape)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = x.view(-1, 32 * 3)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # res_layer = ResidualLayer()
    # inp = np.array([[[0.0 for _ in range(216)] for _ in range(32)]])
    # print(inp.shape)
    #
    # print(res_layer)
    #
    # output = res_layer(torch.from_numpy(inp).float())
    # print(output.shape)

    inp = np.array([[[0.0 for _ in range(216)] for _ in range(1)]])
    res_net = Net(5)
    print(inp.shape)

    print(res_net)

    output = res_net(torch.from_numpy(inp).float())
    print(output.shape)
