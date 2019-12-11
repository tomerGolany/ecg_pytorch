from bokeh.plotting import figure, output_file, show

# Run tensorboard with --samples_per_plugin images=100

# Prepare data:
num_samples_added_from_gan = [0, 500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]

##
# FC network #
##
fc_auc_N = [0.85, 0.86, 0.86, 0.86, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]
fc_auc_S = [0.81, 0.82, 0.84, 0.87, 0.88, 0.89, 0.89, 0.89, 0.9, 0.9]
fc_auc_V = [0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98]
fc_auc_F = [0.5, 0.73, 0.78, 0.82, 0.84, 0.86, 0.86, 0.91, 0.92, 0.93]

##
# LSTM network #
##
lstm_auc_N = [0.87, 0.88, 0.9, 0.9, 0.89, 0.89, 0.91, 0.91, 0.91, 0.91]
lstm_auc_S = [0.82, 0.87, 0.89, 0.87, 0.89, 0.89, 0.89, 0.92, 0.91, 0.91]
lstm_auc_V = [0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.98]
lstm_auc_F = [0.95, 0.96, 0.95, 0.96, 0.96, 0.96, 0.95, 0.95, 0.96, 0.96]


#
# Beat N Comparision:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type N - Normal beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')

# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_N, line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_N, size=8, legend="DCGAN + FF")
p.line(num_samples_added_from_gan, lstm_auc_N, line_width=2, line_color="orange")
p.triangle(num_samples_added_from_gan, lstm_auc_N, size=8, fill_color="orange", legend="DCGAN + LSTM")

output_file("N.html")
p.legend.location = "bottom_right"
# show the results
show(p)

#
# Beat S Comparision:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type S - Supraventricular ectopic beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("S.html")
# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_S, line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_S,legend="DCGAN + FF",  size=8)
p.line(num_samples_added_from_gan, lstm_auc_S, line_width=2, line_color="orange")
p.triangle(num_samples_added_from_gan, lstm_auc_S, legend="DCGAN + LSTM", size=8, fill_color="orange")
p.legend.location = "bottom_right"

show(p)

#
# Beat V Comparision:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type V - Normal beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("V.html")
# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_V, line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_V, size=8, legend="DCGAN + FF")
p.line(num_samples_added_from_gan, lstm_auc_V, line_width=2, line_color="orange")
p.triangle(num_samples_added_from_gan, lstm_auc_V, size=8, fill_color="orange",  legend="DCGAN + LSTM")

p.legend.location = "bottom_right"
# show the results
show(p)

#
# Beat F Comparision:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type F - Fusion beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("F.html")
# add a line renderer with legend and line thickness
p.line(num_samples_added_from_gan, fc_auc_F, line_width=2)
p.circle(num_samples_added_from_gan, fc_auc_F, size=8, legend="DCGAN + FF")
p.line(num_samples_added_from_gan, lstm_auc_F, line_width=2, line_color="orange")
p.triangle(num_samples_added_from_gan, lstm_auc_F, size=8, fill_color="orange", legend="DCGAN + LSTM")

p.legend.location = "bottom_right"
# show the results
show(p)