import os
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.dope.dope_to_blender as db
import net_filter.dynamics.rigid_body as rb
import net_filter.sim.dynamic_gen as dg
import net_filter.sim.dynamic_filter as df
import net_filter.sim.dynamic_filter_plots as fp

# image directory
img_dir = dirs.simulation_dir

# times
n_ims = 30
t = np.linspace(0, .5, n_ims)
inds = np.arange(n_ims)
dt = t[1] - t[0]

# good bad
p0 = np.array([-.1, .7, -.1])
q0 = t3d.euler.euler2quat(-.70, -1.25, 2.69, 'sxyz')
R0 = t3d.quaternions.quat2mat(q0)
pdot0 = np.array([0.9, 1.1, 2.3]) # paper
om0 = np.array([5, 8, 4]) # paper

# rigid body dynamics
p, R, pdot, om  = rb.integrate(t, p0, R0, pdot0, om0)

# regenerate and re-evaluate images?
regen = 1
if regen:
    # regenerate images
    dg.generate_images(n_ims, dt, p, R, pdot, om, img_dir)
    dg.generate_snapshots(n_ims, inds, p, R)

    # re-evaluate images
    p, R, p_est, R_est = db.get_predictions(img_dir)

# load dope pose estimates
npz_file = os.path.join(img_dir, 'dope_pR.npz')
data = np.load(npz_file)
p_meas = data['p']
R_meas = data['R']

# filter initial estimates
p0_hat = p_meas[:,0] # use  the true value to make it a fair comparison
R0_hat = R_meas[:,:,0] # use  the true value to make it a fair comparison
pdot0_hat = pdot0 + .2*np.array([-1.1, 1.2, 1.1])
om0_hat = om0 + .6*np.array([-1.0, 1.1, 1.0])
cov_xx_0_hat = np.ones(12)
COV_xx_0_hat = np.diag(cov_xx_0_hat)

# run filter
p_hat, R_hat, pdot_hat, om_hat, COV_XX_ALL = df.apply_filter(p0_hat, R0_hat, pdot0_hat, om0_hat, COV_xx_0_hat, t, dt, p_meas, R_meas)

# convert outputs and calculate errors
p, R, pdot, om, p_hat, R_hat, pdot_hat, om_hat, p_meas, R_meas, p_err, p_err_meas, R_err, R_err_meas, COV_XX_ALL = df.conversion_and_error(t, p, R, pdot, om, p_hat, R_hat, pdot_hat, om_hat, p_meas, R_meas, COV_XX_ALL, img_dir)

# totals
p_filt_err_mean = np.mean(p_err)
p_meas_err_mean = np.mean(p_err_meas)
p_filt_err_norm = np.linalg.norm(p_err)
p_meas_err_norm = np.linalg.norm(p_err_meas)

R_meas_err_mean = np.mean(R_err_meas)
R_filt_err_mean = np.mean(R_err)
R_meas_err_norm = np.linalg.norm(R_err_meas)
R_filt_err_norm = np.linalg.norm(R_err)

# print
print('average p bias: ', np.mean(p - p_meas, axis=1))
print('p measur error norm: ', p_meas_err_norm)
print('p filter error norm: ', p_filt_err_norm)
print('R measur error norm: ', R_meas_err_norm)
print('R filter error norm: ', R_filt_err_norm)
print('R measur error mean: ', R_meas_err_mean)
print('R filter error mean: ', R_filt_err_mean)

# plot
fp.plot_errors()
fp.plot_3sigma()
fp.plot_velocities()
