import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.conv2 = nn.Conv1d(in_channels=18, out_channels=36, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=72, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv4 = nn.Conv1d(in_channels=72, out_channels=144, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(144 * 13, 1024)
        self.fc2 = nn.Linear(1024, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 144 * 13)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x