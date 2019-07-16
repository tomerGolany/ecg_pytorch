import torch
from torch import nn
from ecg_pytorch.dynamical_model.Euler.euler import euler
from ecg_pytorch.data_reader import ecg_dataset
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt


class LearnParams(nn.Module):
    def __init__(self, device_name):
        super(LearnParams, self).__init__()
        self.device_name = device_name
        self.params = nn.Linear(15, 15)

    def forward(self, x, v0):
        x = self.params(x)
        x = euler(x, self.device_name, v0)
        return x


def train(device_name, num_steps, batch_size):
    #
    # Get desired real beat:
    #
    composed = transforms.Compose([ecg_dataset.Scale()])
    dataset = ecg_dataset.EcgHearBeatsDatasetTest(beat_type='V', transform=composed)
    beats = dataset.test
    n = 1  # for 2 random indices
    # index = np.random.choice(len(beats), n, replace=False)
    index = [2439]
    print(index)
    random_v_beats = beats[index][0]['cardiac_cycle']
    v0 = random_v_beats[0]
    random_v_beats = torch.Tensor([random_v_beats for _ in range(batch_size)])

    #
    # Create model:
    #
    net = LearnParams(device_name)

    #
    # Loss:
    #
    mse_loss = nn.MSELoss()
    opt = optim.Adam(net.parameters(), lr=0.1)

    #
    # train:
    #
    const_input = torch.Tensor([[1] * 15 for _ in range(batch_size)])
    v0_batch = torch.Tensor([v0 for _ in range(batch_size)])

    #
    # to gpu:
    #
    net = net.to(device_name)
    const_input = const_input.to(device_name)
    v0_batch = v0_batch.to(device_name)
    random_v_beats = random_v_beats.to(device_name)

    for i in range(num_steps):

        net.zero_grad()

        output = net(const_input, v0_batch)

        loss = mse_loss(output, random_v_beats)
        print("Loss: {}".format(loss))
        loss.backward()
        opt.step()

        if i % 100 == 0:
            with torch.no_grad():
                ode_beat = output.detach().numpy()[0]
                plt.figure()
                plt.plot(ode_beat, label='ode')
                plt.plot(random_v_beats.numpy()[0], label="real")
                plt.legend()
                plt.show()
                plt.clf()
                plt.close()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(device, 2000, 5)