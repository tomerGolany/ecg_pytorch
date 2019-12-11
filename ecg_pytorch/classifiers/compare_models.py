from bokeh.plotting import figure, output_file, show

# Run tensorboard with --samples_per_plugin images=100

# Prepare data:
num_samples_added_from_gan = [0, 500, 800, 1000, 1500, 3000, 5000, 7000, 10000, 15000]

#
# One Vs. ALL settings:
#

#
# DCGAN:
#
lstm_auc_N_DCGAN = [0.84, 0.87, 0.88, 0.88, 0.63, 0.87, 0.91, 0.81, 0.88, 0.67]
lstm_auc_S_DCGAN = [0.65, 0.7831, 0.72, 0.80, 0.82, 0.8, 0.78, 0.83, 0.74, 0.8]
lstm_auc_F_DCGAN = [0.93, 0.96, 0.94, 0.95, 0.95,0.95, 0.95, 0.96, 0.95, 0.95]
lstm_auc_V_DCGAN = [0.95, 0.97, 0.96, 0.94, 0.7, 0.95, 0.956, 0.957, 0.96, 0.956]

#
# ODEGAN:
#
lstm_auc_N_ODEGAN = [0.84, 0.61, 0.87, 0.89, 0.80, 0.71, 0.90, 0.90, 0.915, 0.88]
lstm_auc_S_ODEGAN = [0.65, 0.88, 0.80, 0.86, 0.8, 0.83, 0.91, 0.88, 0.85, 0.88]
lstm_auc_F_ODEGAN = [0.93, 0.95, 0.9, 0.955, 0.97, 0.93, 0.95, 0.95, 0.947, 0.96]
lstm_auc_V_ODEGAN = [0.95, 0.96, 0.945, 0.75, 0.75, 0.966, 0.97, 0.975, 0.91, 0.96]

#
# Simulator:
#
lstm_auc_S_SIMULATOR = [0.65, 0.82, 0.86, 0.7, 0.78, 0.88, 0.83, 0.89, 0.86, 0.9]
lstm_auc_N_SIMULATOR = [0.84, 0.65, 0.762, 0.51, 0.89, 0.55, 0.88, 0.88, 0.52, 0.77]
lstm_auc_F_SIMULATOR = [0.93, 0.935, 0.89, 0.96, 0.86,  0.901, 0.935, 0.92, 0.94, 0.95]
lstm_auc_V_SIMULATOR = [0.95, 0.95, 0.97, 0.965, 0.97, 0.87, 0.94, 0.97, 0.8, 0.955]

#
# Vanila GAN:
#
lstm_auc_N_VANILA_GAN = [0.84, 0.86, 0.64, 0.89, 0.86, 0.89, 0.64, 0.87, 0.85, 0.89]
lstm_auc_S_VANILA_GAN = [0.65, 0.78, 0.73, 0.7, 0.83, 0.80, 0.83, 0.816, 0.811, 0.81]
lstm_auc_F_VANILA_GAN = [0.93, 0.91, 0.931, 0.925, 0.935, 0.948, 0.945, 0.945, 0.945, 0.95]
lstm_auc_V_VANILA_GAN = [0.95, 0.955, 0.93, 0.95, 0.7, 0.85, 0.97, 0.95, 0.96, 0.96]

#
# Vanila ODEGAN:
#
lstm_auc_N_VANILA_ODE_GAN = [0.84, 0.88, 0.9, 0.65, 0.895, 0.89, 0.85, 0.84, 0.885, 0.85]
lstm_auc_S_VANILA_ODE_GAN = [0.65, 0.61, 0.82, 0.72, 0.85, 0.77, 0.70, 0.80, 0.82, 0.8]
lstm_auc_F_VANILA_ODE_GAN = [0.93, 0.933, 0.90, 0.966, 0.945, 0.97, 0.943, 0.93, 0.943, 0.943]
lstm_auc_V_VANILA_ODE_GAN = [0.95, 0.956, 0.943, 0.945, 0.963, 0.953, 0.961, 0.95, 0.957, 0.956]
#
# Figures:
#

#
# N beat:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type N - Normal beats",
           x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')

# add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_N, line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_N, size=8, legend="DCGAN + FF")
p.line(num_samples_added_from_gan, lstm_auc_N_DCGAN, line_width=2, line_color="orange")
p.circle(num_samples_added_from_gan, lstm_auc_N_DCGAN, size=8, fill_color="orange", legend="DCGAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_N_ODEGAN, line_width=2, line_color="purple")
p.square(num_samples_added_from_gan, lstm_auc_N_ODEGAN, size=8, fill_color="purple", legend="ODEGAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_N_SIMULATOR, line_width=2, line_color="green")
p.asterisk(num_samples_added_from_gan, lstm_auc_N_SIMULATOR, line_width=2, size=15, line_color="green", fill_color="green", legend="Simulator + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_N_VANILA_GAN, line_width=2, line_color="blue")
p.diamond(num_samples_added_from_gan, lstm_auc_N_VANILA_GAN, line_width=2, size=15, fill_color="blue", legend="Vanilla GAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_N_VANILA_ODE_GAN, line_width=2, line_color="pink")
p.triangle(num_samples_added_from_gan, lstm_auc_N_VANILA_ODE_GAN, legend="Vanilla ODE GAN + LSTM", size=8, fill_color="pink")

output_file("N.html")
p.legend.location = "bottom_right"
# show the results
show(p)

#
# S Beat:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type S - Supraventricular ectopic beats",
            x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("S.html")
# add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_S, line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_S,legend="DCGAN + FF",  size=8)
# p.line(num_samples_added_from_gan, lstm_auc_S_DCGAN, line_width=2, line_color="orange")
# p.triangle(num_samples_added_from_gan, lstm_auc_S_DCGAN, legend="DCGAN + LSTM", size=8, fill_color="orange")

p.line(num_samples_added_from_gan, lstm_auc_S_ODEGAN, line_width=2, line_color="green")
p.square(num_samples_added_from_gan, lstm_auc_S_ODEGAN, size=8, fill_color="green", legend="SimDCGAN + LSTM")

# p.line(num_samples_added_from_gan, lstm_auc_S_SIMULATOR, line_width=2, line_color="green")
# p.asterisk(num_samples_added_from_gan, lstm_auc_S_SIMULATOR, line_width=2, size=15, line_color="green", fill_color="green", legend="Simulator + LSTM")

# p.line(num_samples_added_from_gan, lstm_auc_S_VANILA_GAN, line_width=2, line_color="blue")
# p.diamond(num_samples_added_from_gan, lstm_auc_S_VANILA_GAN, line_width=2, size=15, fill_color="blue", legend="Vanilla GAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_S_VANILA_ODE_GAN, line_width=2, line_color="blue")
p.circle(num_samples_added_from_gan, lstm_auc_S_VANILA_ODE_GAN, line_width=2, size=10, fill_color="blue", legend="SimVGAN + LSTM")
# # show the results
p.legend.location = "bottom_right"
#
show(p)

#
# F Beat:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type F - Fusion beats",
            x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("F.html")
# add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_S, line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_S,legend="DCGAN + FF",  size=8)
p.line(num_samples_added_from_gan, lstm_auc_F_DCGAN, line_width=2, line_color="orange")
p.triangle(num_samples_added_from_gan, lstm_auc_F_DCGAN, legend="DCGAN + LSTM", size=8, fill_color="orange")

p.line(num_samples_added_from_gan, lstm_auc_F_ODEGAN, line_width=2, line_color="purple")
p.square(num_samples_added_from_gan, lstm_auc_F_ODEGAN, size=8, fill_color="purple", legend="ODEGAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_F_SIMULATOR, line_width=2, line_color="green")
p.asterisk(num_samples_added_from_gan, lstm_auc_F_SIMULATOR, line_width=2, size=15, line_color="green", fill_color="green", legend="Simulator + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_F_VANILA_GAN, line_width=2, line_color="blue")
p.diamond(num_samples_added_from_gan, lstm_auc_F_VANILA_GAN, line_width=2, size=15, fill_color="blue", legend="Vanilla GAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_F_VANILA_ODE_GAN, line_width=2, line_color="pink")
p.circle(num_samples_added_from_gan, lstm_auc_F_VANILA_ODE_GAN, line_width=2, size=10, fill_color="pink", legend="Vanilla ODE GAN + LSTM")
# # show the results
p.legend.location = "bottom_right"
#
show(p)


#
# V Beat:
#
p = figure(# title="Average AUC comparison on classifying heartbeat of type F - Fusion beats",
            x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
output_file("V.html")
# add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_S, line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_S,legend="DCGAN + FF",  size=8)
p.line(num_samples_added_from_gan, lstm_auc_V_DCGAN, line_width=2, line_color="orange")
p.triangle(num_samples_added_from_gan, lstm_auc_V_DCGAN, legend="DCGAN + LSTM", size=8, fill_color="orange")

p.line(num_samples_added_from_gan, lstm_auc_V_ODEGAN, line_width=2, line_color="purple")
p.square(num_samples_added_from_gan, lstm_auc_V_ODEGAN, size=8, fill_color="purple", legend="ODEGAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_V_SIMULATOR, line_width=2, line_color="green")
p.asterisk(num_samples_added_from_gan, lstm_auc_V_SIMULATOR, line_width=2, size=15, line_color="green", fill_color="green", legend="Simulator + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_V_VANILA_GAN, line_width=2, line_color="blue")
p.diamond(num_samples_added_from_gan, lstm_auc_V_VANILA_GAN, line_width=2, size=15, fill_color="blue", legend="Vanilla GAN + LSTM")

p.line(num_samples_added_from_gan, lstm_auc_V_VANILA_ODE_GAN, line_width=2, line_color="pink")
p.circle(num_samples_added_from_gan, lstm_auc_V_VANILA_ODE_GAN, line_width=2, size=10, fill_color="pink", legend="Vanilla ODE GAN + LSTM")
# # show the results
p.legend.location = "bottom_right"
#
show(p)








# ##
# # DCGAN results #
# ##
#
# ##
# # FC network #
# ##
# fc_auc_N = [0.85, 0.86, 0.86, 0.86, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]
# fc_auc_S = [0.81, 0.82, 0.84, 0.87, 0.88, 0.89, 0.89, 0.89, 0.9, 0.9]
# fc_auc_V = [0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98]
# fc_auc_F = [0.5, 0.73, 0.78, 0.82, 0.84, 0.86, 0.86, 0.91, 0.92, 0.93]
# fc_auc_Q = [0.86]
#
# ##
# # LSTM network #
# ##
# lstm_auc_N = [0.87, 0.88, 0.9, 0.9, 0.89, 0.89, 0.91, 0.91, 0.91, 0.91]
# lstm_auc_S = [0.82, 0.87, 0.89, 0.87, 0.89, 0.89, 0.89, 0.92, 0.91, 0.91]
# lstm_auc_V = [0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.98]
# lstm_auc_F = [0.95, 0.96, 0.95, 0.96, 0.96, 0.96, 0.95, 0.95, 0.96, 0.96]
# lstm_auc_Q = [0.83]
#
# #
# # ODE GAN - Best scores:
# #
# ODE_N_lstm = [0.89, 0.9, 0.91, 0.915, 0.921,   0.92, 0.91, 0.91, 0.91, 0.9]
# ODE_S_lstm = [0.82, 0.87, 0.88, 0.9, 0.93, 0.87,     0.91, 0.9, 0.91, 0.92]
#
# ODE_F_lstm = [0.95, 0.96, 0.96, 0.965, 0.96, 0.96, 0.97, 0.96, 0.96, 0.96]
#
#
# #
# # Pure data from simulator:
# #
# SIM_N_lstm = []
# SIM_S_lstm = [0.82, 0.87, 0.82, 0.87, 0.86, 0.87, 0.87, 0.86, 0.84, 0.85]
# SIM_V_lstm = []
# SIM_F_lstm = [0.95, 0.96, 0.95, 0.96, 0.95, 0.95, 0.96, 0.95, 0.96, 0.95]
#
# #
# # ODE Gan Combined Version:
# #
#
#
#
#
#
#
# #
# # Beat N Comparision:
# #
# p = figure(title="Average AUC comparison on classifying heartbeat of type N - Normal beats",
#            x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
#
# # add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_N, line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_N, size=8, legend="DCGAN + FF")
# p.line(num_samples_added_from_gan, lstm_auc_N, line_width=2, line_color="orange")
# p.circle(num_samples_added_from_gan, lstm_auc_N, size=8, fill_color="orange", legend="DCGAN + LSTM")
# p.line(num_samples_added_from_gan, ODE_N_lstm, line_width=2, line_color="purple")
# p.square(num_samples_added_from_gan, ODE_N_lstm, size=8, fill_color="purple", legend="ODEGAN + LSTM")
#
# # p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# # p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")
# output_file("N.html")
# p.legend.location = "bottom_right"
# # show the results
# show(p)
#
# #
# # Beat S Comparision:
# #
# p = figure(title="Average AUC comparison on classifying heartbeat of type S - Supraventricular ectopic beats",
#            x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
# output_file("S.html")
# # add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_S, line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_S,legend="DCGAN + FF",  size=8)
# p.line(num_samples_added_from_gan, lstm_auc_S, line_width=2, line_color="orange")
# p.triangle(num_samples_added_from_gan, lstm_auc_S, legend="DCGAN + LSTM", size=8, fill_color="orange")
# p.line(num_samples_added_from_gan, ODE_S_lstm, line_width=2, line_color="purple")
# p.square(num_samples_added_from_gan, ODE_S_lstm, size=8, fill_color="purple", legend="ODEGAN + LSTM")
# p.line(num_samples_added_from_gan, SIM_S_lstm, line_width=2, line_color="green")
# p.asterisk(num_samples_added_from_gan, SIM_S_lstm, line_width=2, size=15, line_color="green", fill_color="green", legend="Simulator + LSTM")
#
# # p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# # p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")
# # show the results
# p.legend.location = "bottom_right"
#
# show(p)
#
# #
# # Beat V Comparision:
# #
# p = figure(title="Average AUC comparison on classifying heartbeat of type V - Normal beats",
#            x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
# output_file("V.html")
# # add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_V, legend="Fully Connected", line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_V, size=8)
# p.line(num_samples_added_from_gan, lstm_auc_V, legend="LSTM", line_width=2, line_color="orange")
# p.circle(num_samples_added_from_gan, lstm_auc_V, size=8, fill_color="orange")
# # p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# # p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")
#
# p.legend.location = "bottom_right"
# # show the results
# show(p)
#
# #
# # Beat F Comparision:
# #
# p = figure(title="Average AUC comparison on classifying heartbeat of type F - Fusion beats",
#            x_axis_label='# Synthetic examples added', y_axis_label='AUC of ROC')
# output_file("F.html")
# # add a line renderer with legend and line thickness
# p.line(num_samples_added_from_gan, fc_auc_F, line_width=2)
# p.circle(num_samples_added_from_gan, fc_auc_F, size=8, legend="DCGAN + FF")
# p.line(num_samples_added_from_gan, lstm_auc_F, line_width=2, line_color="orange")
# p.triangle(num_samples_added_from_gan, lstm_auc_F, size=8, fill_color="orange", legend="DCGAN + LSTM")
# p.line(num_samples_added_from_gan, ODE_F_lstm, line_width=2, line_color="purple")
# p.square(num_samples_added_from_gan, ODE_F_lstm, size=8, fill_color="purple", legend="ODEGAN + LSTM")
# # p.line(num_samples_added_from_gan, noise_N, legend="Noise", line_width=2, line_color="black")
# # p.circle(num_samples_added_from_gan, noise_N, size=8, fill_color="black")
#
# p.legend.location = "bottom_right"
# # show the results
# show(p)

