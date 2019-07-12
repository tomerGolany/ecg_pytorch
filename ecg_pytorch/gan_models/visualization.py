"""Visualize output from trained generators."""
from ecg_pytorch.gan_models import checkpoint_paths
from bokeh.io import output_file, show
from bokeh.layouts import row
from bokeh.plotting import figure
from ecg_pytorch.gan_models import generate_data_from_train_gan
from ecg_pytorch.gan_models.models import dcgan
from ecg_pytorch.data_reader import ecg_dataset


def compare_real_vs_fake(beat_type, real_beat, fake_beat):
    output_file("{}_fake_vs_real_dcgan.html".format(beat_type))

    time = list(range(216))
    # create a new plot
    s1 = figure(title=None)
    s1.line(time, real_beat, legend="Real beat", line_width=2, color="navy", alpha=0.5)

    # create another one
    s2 = figure(title=None)
    s2.line(time, fake_beat, legend="Fake beat", line_width=2, color="firebrick", alpha=0.5)
    # put the results in a row
    show(row(s1, s2))

#
# DCGAN:
#

#
# Visualize fake N beat against real N beat:
#
gNET = dcgan.DCGenerator(0)
fake_n_beat = generate_data_from_train_gan.generate_data_from_trained_gan(gNET, 1, checkpoint_paths.DCGAN_N_CHK)
fake_n_beat = fake_n_beat.numpy()[0]
dataset = ecg_dataset.EcgHearBeatsDataset(beat_type='N')
real_n_beat = dataset.train[200]['cardiac_cycle']
compare_real_vs_fake('N', real_n_beat, fake_n_beat)




