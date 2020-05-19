import os
import numpy as np
import net_filter.directories as dirs

# load
img_dir = dirs.monte_carlo_dir
trials_npz = os.path.join(img_dir, 'trials.npz')
dat = np.load(trials_npz)
p_err_meas_mean = dat['p_err_meas_mean']
p_err_filt_mean = dat['p_err_filt_mean']
R_err_meas_mean = dat['R_err_meas_mean']
R_err_filt_mean = dat['R_err_filt_mean']
n_trials = len(p_err_meas_mean)

# print
print('p err meas', p_err_meas_mean)
print('p err filt', p_err_filt_mean)
print('R err meas', R_err_meas_mean)
print('R err filt', R_err_filt_mean)

p_filter_is_better = (p_err_filt_mean < p_err_meas_mean)
R_filter_is_better = (R_err_filt_mean < R_err_meas_mean)
print(p_filter_is_better)
print(R_filter_is_better)
print('p filter better:', np.sum(p_filter_is_better), '/', n_trials)
print('R filter better:', np.sum(R_filter_is_better), '/', n_trials)
