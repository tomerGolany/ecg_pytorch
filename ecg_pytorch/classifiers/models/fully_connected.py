import torch.nn as nn
import torch.nn.functional as F


class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.fc1 = nn.Linear(216, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x