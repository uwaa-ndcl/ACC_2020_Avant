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
xyz0 = np.array([-.1, .7, -.1])
q0 = t3d.euler.euler2quat(-.70, -1.25, 2.69, 'sxyz')
R0 = t3d.quaternions.quat2mat(q0)
xyzdot0 = np.array([0.9, 1.1, 2.3]) # paper
om0 = np.array([5, 8, 4]) # paper

# rigid body dynamics
xyz, R, v, om  = rb.integrate(t, xyz0, R0, xyzdot0, om0)

# regenerate and re-evaluate images?
regen = 1
if regen:
    # regenerate images
    dg.generate_images(n_ims, dt, xyz, R, v, om, img_dir)
    dg.generate_snapshots(n_ims, inds, xyz, R)

    # re-evaluate images
    xyz, R, xyz_est, R_est = db.get_predictions(img_dir)

# load dope pose estimates
npz_file = os.path.join(img_dir, 'dope_xyzR.npz')
data = np.load(npz_file)
xyz_meas = data['xyz']
R_meas = data['R']

# filter initial estimates
xyz0_hat = xyz_meas[:,0] # use  the true value to make it a fair comparison
R0_hat = R_meas[:,:,0] # use  the true value to make it a fair comparison
xyzdot0_hat = xyzdot0 + .2*np.array([-1.1, 1.2, 1.1])
om0_hat = om0 + .6*np.array([-1.0, 1.1, 1.0])
p0_hat = np.ones(12)
P0_hat = np.diag(p0_hat)

# run filter
xyz_hat, R_hat, v_hat, om_hat, P_ALL = df.apply_filter(xyz0_hat, R0_hat, xyzdot0_hat, om0_hat, P0_hat, t, dt, xyz_meas, R_meas)

# convert outputs and calculate errors
xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, xyz_err, xyz_err_meas, R_err, R_err_meas = df.conversion_and_error(t, xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, img_dir)

# totals
xyz_filt_err_mean = np.mean(xyz_err)
xyz_meas_err_mean = np.mean(xyz_err_meas)
xyz_filt_err_norm = np.linalg.norm(xyz_err)
xyz_meas_err_norm = np.linalg.norm(xyz_err_meas)

R_meas_err_mean = np.mean(R_err_meas)
R_filt_err_mean = np.mean(R_err)
R_meas_err_norm = np.linalg.norm(R_err_meas)
R_filt_err_norm = np.linalg.norm(R_err)

# print
print('average xyz bias: ', np.mean(xyz - xyz_meas, axis=1))
print('xyz measur error norm: ',  xyz_meas_err_norm)
print('xyz filter error norm: ',  xyz_filt_err_norm)
print('R measur error norm:   ', R_meas_err_norm)
print('R filter error norm:   ', R_filt_err_norm)
print('R measur error mean:   ', R_meas_err_mean)
print('R filter error mean:   ', R_filt_err_mean)

# plot
fp.plot_errors()
fp.plot_3sigma()
fp.plot_velocities()
