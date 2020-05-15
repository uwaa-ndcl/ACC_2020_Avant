import os
import numpy as np
import net_filter.directories as dirs

# load
img_dir = dirs.trials_dir
trials_npz = os.path.join(img_dir, 'trials.npz')
dat = np.load(trials_npz)
xyz_err_mean = dat['xyz_err_mean']
xyz_err_mean_meas = dat['xyz_err_mean_meas']
R_err_mean = dat['R_err_mean']
R_err_mean_meas = dat['R_err_mean_meas']
n_trials = len(xyz_err_mean)

# print
print('xyz err filt', xyz_err_mean)
print('xyz err meas', xyz_err_mean_meas)
print('R err filt', R_err_mean)
print('R err meas', R_err_mean_meas)

xyz_filter_is_better = (xyz_err_mean < xyz_err_mean_meas)
R_filter_is_better = (R_err_mean < R_err_mean_meas)
print(xyz_filter_is_better)
print(R_filter_is_better)
print('xyz filter better:', np.sum(xyz_filter_is_better), '/', n_trials)
print('R filter better:', np.sum(R_filter_is_better), '/', n_trials)
