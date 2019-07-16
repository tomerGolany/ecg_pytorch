from ecg_pytorch.dynamical_model.Euler.single_step import single_step_euler
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from ecg_pytorch.dynamical_model.ode_params import ODEParams
# from ecg_pytorch.data_reader import ecg_dataset
import torchvision.transforms as transforms


class Euler(nn.Module):
    def __init__(self, device_name):
        super(Euler, self).__init__()
        self.device_name = device_name

    def forward(self, x, v0):
        # x = x.view(x.size(0), -1)
        x = euler(x, self.device_name, v0)
        # x = down_sample(x)
        # x = x.view(-1, 1, 216)
        x = scale_signal(x)
        return x


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


def down_sample(ecg_signal):
    res = []
    for beat in ecg_signal:
        i = 0
        down_sampled_ecg = []
        q = int(514 / 216)
        while i < 514:
            # j += 1
            if len(down_sampled_ecg) == 216:
                break
            down_sampled_ecg.append(beat[i])
            i += q  # q = int(np.rint(self.ecg_params.getSf() / self.ecg_params.getSfEcg()))
        down_sampled_ecg = torch.stack(down_sampled_ecg)
        res.append(down_sampled_ecg)
    res = torch.stack(res)
    return res


def euler(params_batch, device_name, v0):
    ode_params = ODEParams(device_name)
    # params = params_batch
    x = torch.tensor(-0.417750770388669).to(device_name)
    y = torch.tensor(-0.9085616622823985).to(device_name)
    # z = torch.tensor(-0.004551233843726818).to(device_name)

    res = []
    # print(len(params_batch))
    for j, params in enumerate(params_batch):
        z = torch.tensor(v0[j]).to(device_name)
        t = torch.tensor(0.0).to(device_name)
        x_next, y_next, z_next = single_step_euler(ode_params, x, y, z, t, params, device_name)
        x_t = [x_next]
        y_t = [y_next]
        z_t = [z_next]
        for i in range(215):
            t += 1 / 216
            x_next, y_next, z_next = single_step_euler(ode_params, x_next, y_next, z_next, t, params, device_name)
            x_t.append(x_next)
            y_t.append(y_next)
            z_t.append(z_next)
        # z_res = []
        z_t = torch.stack(z_t)
        # z_t += torch.Tensor(np.random.normal(0, 0.0001, 216))

        res.append(z_t)
    res = torch.stack(res)
    return res


# def test_euler_with_different_inits():
#     composed = transforms.Compose([ecg_dataset.Scale(), ecg_dataset.ToTensor()])
#     dataset = ecg_dataset.EcgHearBeatsDataset(transform=composed, beat_type='N')
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
#                                              shuffle=False, num_workers=1)
#     for i, e in enumerate(dataloader):
#         # e = dataset[i]
#         if i == 5:
#             break
#         v0 = e['cardiac_cycle'].numpy()[:, 0]
#         print("v0 = {}".format(v0))
#         input = np.array((
#             ([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
#               30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
#               90.0 * math.pi / 180.0]))).reshape((1, 15))
#
#         a1 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
#         a2 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
#         third_tensor = torch.cat((a1, a2), 0)
#         res = euler(third_tensor, "cpu", v0)
#         print(res.shape)
#         res0 = res.detach().numpy()[0]
#         plt.figure()
#         plt.plot(res0, label='ode beat')
#         plt.plot(e['cardiac_cycle'][0].numpy(), label="real")
#         plt.legend()
#         plt.show()
#
#         res1 = res.detach().numpy()[1]
#         plt.figure()
#         plt.plot(res1, label='ode beat')
#         plt.plot(e['cardiac_cycle'][1].numpy(), label="real")
#         plt.legend()
#         plt.title("verf")
#         plt.show()


if __name__ == "__main__":
    # params = torch.nn.Parameter(
    #     torch.tensor([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
    #                   30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
    #                   90.0 * math.pi / 180.0]))
    #
    # input = np.array((
    #          ([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
    #                        30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
    #                        90.0 * math.pi / 180.0]))).reshape((1,15))
    #
    # a1 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    # a2 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    # print(a1.shape)
    # # input = torch.nn.Parameter(torch.Tensor([input]), requires_grad=True)
    # third_tensor = torch.cat((a1, a2), 0)
    # # print(third_tensor)
    # ngpu = 0
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # res = euler(third_tensor, device, 0.0)
    #
    # # print(res[1][-1].backward())
    # # print(a2.grad)
    # # print(res.shape)
    #
    # res = [x.detach().numpy() for x in res[1]]
    # plt.plot(res)
    # plt.show()
    # test_euler_with_different_inits()
    pass