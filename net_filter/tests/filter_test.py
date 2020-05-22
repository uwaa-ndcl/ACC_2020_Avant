import os
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as pp

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.dope.dope_to_blender as db
import net_filter.dynamics.rigid_body as rb
import net_filter.dynamics.unscented_filter as uf
import net_filter.sim.dynamic_functions as df

# times
n_ims = 30
t = np.linspace(0, .5, n_ims)
inds = np.arange(n_ims)
dt = t[1] - t[0]

# intial condition
p0 = np.array([-.1, .7, -.1])
q0 = t3d.euler.euler2quat(-.70, -1.25, 2.69, 'sxyz')
R0 = t3d.quaternions.quat2mat(q0)
pdot0 = np.array([0.9, 1.1, 2.3])
om0 = np.array([5, 8, 4])

# rigid body dynamics
p, R, pdot, om  = rb.integrate(p0, R0, pdot0, om0, t)
save_dir = os.path.join(dirs.results_dir, 'filter_test/')

# simulate measurements
noise_p = .04
noise_R = .17
p_meas = p + noise_p*np.random.uniform(-1,1,(3,n_ims))
R_meas = np.full((3,3,n_ims), np.nan)
for i in range(n_ims):
    s_noise = noise_R*np.random.uniform(.5,-.5,3)
    R_noise = so3.exp(so3.cross(s_noise))
    R_meas[:,:,i] = R_noise @ R[:,:,i]

# filter initial estimates
p0_hat = p_meas[:,0] # use the true value to make it a fair comparison
R0_hat = R_meas[:,:,0] # use the true value to make it a fair comparison
pdot0_hat = pdot0 + .2*np.array([-1.1, 1.2, 1.1])
om0_hat = om0 + .6*np.array([-1.0, 1.1, 1.0])
cov_xx_0_hat = np.ones(12)
COV_xx_0_hat = np.diag(cov_xx_0_hat)

# run filter
U = np.full((1, n_ims), 0.0) # there is no control in this problem
COV_ww, COV_vv = df.get_noise_covariances()
p_filt, R_filt, pdot_filt, om_filt, COV_XX_ALL = uf.filter(p0_hat, R0_hat, pdot0_hat, om0_hat, COV_xx_0_hat, COV_ww, COV_vv, U, p_meas, R_meas, dt)

# convert outputs and calculate errors
df.conversion_and_error(t, p, R, pdot, om, p_filt, R_filt, pdot_filt, om_filt, p_meas, R_meas, COV_XX_ALL, save_dir)
filter_results_npz = os.path.join(save_dir, 'filter_results.npz')
data = np.load(filter_results_npz)
p = data['p']
p_err_meas = data['p_err_meas']
p_err_filt = data['p_err_filt']
R_err_meas = data['R_err_meas']
R_err_filt = data['R_err_filt']

# totals
p_meas_err_mean = np.mean(p_err_meas)
p_filt_err_mean = np.mean(p_err_filt)
p_meas_err_norm = np.linalg.norm(p_err_meas)
p_filt_err_norm = np.linalg.norm(p_err_filt)

R_meas_err_mean = np.mean(R_err_meas)
R_filt_err_mean = np.mean(R_err_filt)
R_meas_err_norm = np.linalg.norm(R_err_meas)
R_filt_err_norm = np.linalg.norm(R_err_filt)

# print
print('average p bias: ', np.mean(p - p_meas, axis=1))
print('p measur error norm: ', p_meas_err_norm)
print('p filter error norm: ', p_filt_err_norm)
print('R measur error norm: ', R_meas_err_norm)
print('R filter error norm: ', R_filt_err_norm)

# plot
fig, ax = pp.subplots(2)
ax[0].set_ylabel('position error')
ax[0].plot(t, p_err_meas, label='measure')
ax[0].plot(t, p_err_filt, label='filter')
ax[0].legend()
ax[1].set_ylabel('rotation error')
ax[1].plot(t, R_err_meas, label='measure')
ax[1].plot(t, R_err_filt, label='filter')
ax[1].legend()
pp.show()
