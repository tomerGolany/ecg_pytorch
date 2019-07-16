from bokeh.plotting import figure, output_file, show

# Run tensorboard with --samples_per_plugin images=100

# Prepare data:
num_samples_added_from_gan = [0, 500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]

##
# DCGAN results #
##

##
# FC network #
##
fc_auc_N = [0.85, 0.86, 0.86, 0.86, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]
fc_auc_S = [0.81, 0.82, 0.84, 0.87, 0.88, 0.89, 0.89, 0.89, 0.9, 0.9]
fc_auc_V = [0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98]
fc_auc_F = [0.5, 0.73, 0.78, 0.82, 0.84, 0.86, 0.86, 0.91, 0.92, 0.93]
fc_auc_Q = [0.86]

##
# LSTM network #
##
lstm_auc_N = [0.87, 0.88, 0.9, 0.9, 0.89, 0.89, 0.91, 0.91, 0.91, 0.91]
lstm_auc_S = [0.82, 0.87, 0.89, 0.87, 0.89, 0.89, 0.89, 0.92, 0.91, 0.91]
lstm_auc_V = [0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.98]
lstm_auc_F = [0.95, 0.96, 0.95, 0.96, 0.96, 0.96, 0.95, 0.95, 0.96, 0.96]
lstm_auc_Q = [0.83]

##
# Adding random noise #
##
noise_N = []

#
# ODE GAN - Best scores:
#
ODE_N = []

# create a new plot with a title and axis labels
p = figure(title="Average AUC comparison on classifying heartbeat of type N - Normal beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')

# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_N, legend="Fully Connected", line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_N, size=8)
p.line(num_samples_added_from_gan, lstm_auc_N, legend="LSTM", line_width=2, line_color="orange")
p.circle(num_samples_added_from_gan, lstm_auc_N, size=8, fill_color="orange")
# p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")
output_file("N.html")
p.legend.location = "bottom_right"
# show the results
show(p)

p = figure(title="Average AUC comparison on classifying heartbeat of type S - Normal beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("S.html")
# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_S, legend="Fully Connected", line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_S, size=8)
p.line(num_samples_added_from_gan, lstm_auc_S, legend="LSTM", line_width=2, line_color="orange")
p.circle(num_samples_added_from_gan, lstm_auc_S, size=8, fill_color="orange")
# p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")
# show the results
p.legend.location = "bottom_right"

show(p)

p = figure(title="Average AUC comparison on classifying heartbeat of type V - Normal beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("V.html")
# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_V, legend="Fully Connected", line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_V, size=8)
p.line(num_samples_added_from_gan, lstm_auc_V, legend="LSTM", line_width=2, line_color="orange")
p.circle(num_samples_added_from_gan, lstm_auc_V, size=8, fill_color="orange")
# p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")

p.legend.location = "bottom_right"
# show the results
show(p)

p = figure(title="Average AUC comparison on classifying heartbeat of type F - Normal beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("F.html")
# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_F, legend="Fully Connected", line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_F, size=8)
p.line(num_samples_added_from_gan, lstm_auc_F, legend="LSTM", line_width=2, line_color="orange")
p.circle(num_samples_added_from_gan, lstm_auc_F, size=8, fill_color="orange")
# p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")

p.legend.location = "bottom_right"
# show the results
show(p)

