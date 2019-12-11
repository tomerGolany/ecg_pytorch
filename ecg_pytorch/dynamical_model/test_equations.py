from ecg_pytorch.dynamical_model import equations
from ecg_pytorch.dynamical_model.ode_params import ODEParams
import torch
import math
from matplotlib import pyplot as plt
import pickle
from ecg_pytorch.data_reader.ecg_dataset import EcgHearBeatsDataset, EcgHearBeatsDatasetTest
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, LabelSet

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

        t += 1 / 360
        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z
    res = [v.detach().item() for v in z_signal]

    #
    # Plot:
    #
    wave_values = [0.0075, -0.011, 0.042, -0.016, 0.0139]
    wave_locations = [33, 59, 69, 79, 123]
    source = ColumnDataSource(data=dict(time=[30, 55, 65, 75, 119],
                                        voltage=[0.0075, -0.015, 0.0424, -0.019, 0.0139],
                                        waves=['P', 'Q', 'R', 'S', 'T']))
    p = figure(x_axis_label='sample # (360HZ)', y_axis_label='voltage [mV]', x_range=(0, 216))
    time = np.arange(0, 216)
    p.line(time, res, line_width=2, line_color='green')
    p.scatter(x=wave_locations, y=wave_values, size=8, legend='P, Q, R, S, T wave events')
    labels = LabelSet(x='time', y='voltage', text='waves', level='glyph',
                      x_offset=5, y_offset=5, source=source, render_mode='canvas')
    p.add_layout(labels)
    p.legend.location = "bottom_right"
    show(p)


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
    #index = np.random.choice(len(beats), n, replace=False)
    index = [331]
    print(index)
    random_s_beats = beats[index]

    #
    # Generate S beat from simulator:
    #
    ode_params = ODEParams('cpu')
    ode_params.h = torch.tensor(1 / 216).to('cpu')
    params = [0.2, 0.25, -0.5 * math.pi, -1.0, 0.1, -15.0 * math.pi / 180.0,
              30.0, 0.1, 0.0 * math.pi / 180.0, -10.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
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

    p = figure(title="S beats",
               x_axis_label='#samples', y_axis_label='V')
    for b in random_s_beats:
        # create a new plot with a title and axis labels

        time = np.arange(0, 216)
        # add a line renderer with legend and line thickness
        real_sample = b['cardiac_cycle']
        real_sample = np.interp(real_sample, (real_sample.min(), real_sample.max()), (min(res), max(res)))
        p.line(time, real_sample, legend='real', line_width=2, line_color='green')
        p.line(time, res, legend="fake", line_width=2, line_color="black")
        p.legend.location = "bottom_right"
    # show the results
    show(p)


def create_F_sample():
    ecg_data = EcgHearBeatsDatasetTest(beat_type='F')
    beats = ecg_data.test
    n = 1  # for 2 random indices
    index = np.random.choice(len(beats), n, replace=False)
    index = [186]
    print(index)
    random_s_beats = beats[index]

    #
    # Generate F beat from simulator:
    #
    ode_params = ODEParams('cpu')
    ode_params.h = torch.tensor(1 / 216).to('cpu')
    params = [0.8, 0.25, -0.5 * math.pi, -10.0, 0.1, -15.0 * math.pi / 180.0,
              30.0, 0.1, 0.03 * math.pi / 180.0, -10.0, 0.1, 15.0 * math.pi / 180.0, 0.5, 0.2,
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

    p = figure(title="F beats",
               x_axis_label='#samples', y_axis_label='Voltage')
    for b in random_s_beats:
        # create a new plot with a title and axis labels

        time = np.arange(0, 216)
        # add a line renderer with legend and line thickness
        real_sample = b['cardiac_cycle']
        real_sample = np.interp(real_sample, (real_sample.min(), real_sample.max()), (min(res), max(res)))
        p.line(time, real_sample, legend='real', line_width=2, line_color='green')
        p.line(time, res, legend="fake", line_width=2, line_color="black")
        p.legend.location = "bottom_right"
    # show the results
    show(p)


def create_V_sample():
    ecg_data = EcgHearBeatsDatasetTest(beat_type='V')
    beats = ecg_data.test
    n = 1  # for 2 random indices
    index = np.random.choice(len(beats), n, replace=False)
    # index = [2439]
    print(index)
    random_s_beats = beats[index]

    #
    # Generate V beat from simulator:
    #
    ode_params = ODEParams('cpu')
    ode_params.h = torch.tensor(1 / 216).to('cpu')
    #params = [1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
    #                  30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
    #                  90.0 * math.pi / 180.0]
    params = [0.1, 0.6, -0.5 * math.pi,
              0.0, 0.1, -15.0 * math.pi / 180.0,
               30.0, 0.1, 0.00 * math.pi / 180.0,
              -10.0, 0.1, 15.0 * math.pi / 180.0,
              0.5, 0.2, 160.0 * math.pi / 180.0]

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

    p = figure(title="V beats",
               x_axis_label='#samples', y_axis_label='Voltage')
    for b in random_s_beats:
        # create a new plot with a title and axis labels

        time = np.arange(0, 216)
        # add a line renderer with legend and line thickness
        real_sample = b['cardiac_cycle']
        real_sample = np.interp(real_sample, (real_sample.min(), real_sample.max()), (min(res), max(res)))
        p.line(time, real_sample, legend='real', line_width=2, line_color='green')
        p.line(time, res, legend="fake", line_width=2, line_color="black")
        p.legend.location = "bottom_right"
    # show the results
    show(p)


if __name__ == "__main__":
    # create_good_sample()
    # create_sample()
    # create_S_sample()
    # create_F_sample()
    create_V_sample()