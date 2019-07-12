from ecg_pytorch.dynamical_model import equations
from ecg_pytorch.dynamical_model.ode_params import ODEParams
import torch
import math
from matplotlib import pyplot as plt
import pickle
from ecg_pytorch.data_reader.ecg_dataset import EcgHearBeatsDataset
import numpy as np
from bokeh.plotting import figure, output_file, show


def create_good_sample():
    ode_params = ODEParams('cpu')

    input_params = torch.nn.Parameter(
        torch.tensor([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
                      90.0 * math.pi / 180.0])).view(1, 15)
    x = torch.tensor(-0.417750770388669)
    y = torch.tensor(-0.9085616622823985)
    z = torch.tensor(-0.004551233843726818)
    t = torch.tensor(0.0)

    x_signal = [x]
    y_signal = [y]
    z_signal = [z]
    for i in range(215):
        f_x = equations.d_x_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_y = equations.d_y_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_z = equations.d_z_d_t(x, y, z, t, input_params, ode_params)

        t += 1 / 512
        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z
    res = [v.detach().item() for v in z_signal]

    # print(res)
    print(len(res))
    plt.plot(res)
    plt.show()
    # with open('ode_normal_sample.pkl', 'wb') as f:
    #     pickle.dump(res, f)


def create_sample():
    ode_params = ODEParams('cpu')
    ode_params.h = torch.tensor(1 / 216).to('cpu')
    params = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                      160.0 * math.pi / 180.0]

    input_params = torch.tensor(params).view(1, 15)
    x = torch.tensor(-0.417750770388669)
    y = torch.tensor(-0.9085616622823985)
    z = torch.tensor(-0.004551233843726818)
    t = torch.tensor(0.0)

    x_signal = [x]
    y_signal = [y]
    z_signal = [z]
    for i in range(215):
        f_x = equations.d_x_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_y = equations.d_y_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_z = equations.d_z_d_t(x, y, z, t, input_params, ode_params)

        t += 1 / 360
        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z
    res = [v.detach().item() for v in z_signal]

    ecg_data = EcgHearBeatsDataset(beat_type='N')

    rael_sample = ecg_data[55]['cardiac_cycle']
    rael_sample = np.interp(rael_sample, (rael_sample.min(), rael_sample.max()), (min(res), max(res)))

    # print(res)
    print(len(res))
    plt.plot(res, label='ode')
    plt.plot(rael_sample, label='N_beat')
    plt.legend()
    plt.title('sample output from ode')
    plt.show()

    # create a new plot with a title and axis labels
    p = figure(title="ode",
               x_axis_label='#samples', y_axis_label='V')
    time = np.arange(0, 216)
    # add a line renderer with legend and line thickness
    p.line(time, rael_sample, legend="real", line_width=2)

    p.line(time, res, legend="fake", line_width=2, line_color="black")


    p.legend.location = "bottom_right"
    # show the results
    show(p)


def create_S_sample():
    ecg_data = EcgHearBeatsDataset(beat_type='S')
    beats = ecg_data.train
    n = 1  # for 2 random indices
    index = np.random.choice(len(beats), n, replace=False)
    print(index)
    random_s_beats = beats[index]
    p = figure(title="S beats",
               x_axis_label='#samples', y_axis_label='V')
    for b in random_s_beats:
        # create a new plot with a title and axis labels

        time = np.arange(0, 216)
        # add a line renderer with legend and line thickness
        p.line(time, b['cardiac_cycle'], line_width=2)
    # show the results
    show(p)




if __name__ == "__main__":
    # create_good_sample()
    # create_sample()
    create_S_sample()