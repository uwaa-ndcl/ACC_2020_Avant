import os
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.sim.dope_to_blender as db
import net_filter.dynamics.rigid_body as rb
import net_filter.dynamics.angular_velocity as av
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
q = np.full((4,n_ims), np.nan)
for i in range(n_ims):
    q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

regen_ims = 0 # regenerate images?
eval_ims = 0 # evaluate images?

# regenerate images?
if regen_ims:
    dg.generate_images(n_ims, dt, xyz, q, v, om, img_dir)
    dg.generate_snapshots(n_ims, inds, xyz, q)

# evaluate images?
if eval_ims:
    xyz, q, xyz_est, q_est = db.get_predictions(img_dir)

# load dope pose estimates
npz_file = os.path.join(img_dir, 'dope_xyzq.npz')
data = np.load(npz_file)
xyz_meas = data['xyz']
q_meas = data['q']

# convert quaternions to rotation matrices
R = np.full((3,3,n_ims),np.nan)
R_meas = np.full((3,3,n_ims),np.nan)
for i in range(n_ims):
    R[:,:,i] = t3d.quaternions.quat2mat(q[:,i])
    R_meas[:,:,i] = t3d.quaternions.quat2mat(q_meas[:,i])

# filter initial estimates
xyz0_hat = xyz_meas[:,0] # use  the true value to make it a fair comparison
R0_hat = R_meas[:,:,0] # use  the true value to make it a fair compariso
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

R_meas_err_norm = np.linalg.norm(R_err_meas)
R_filt_err_norm = np.linalg.norm(R_err)
R_meas_err_mean = np.mean(R_err_meas)
R_filt_err_mean = np.mean(R_err)

# print
print('average xyz bias: ', np.mean(xyz - xyz_meas, axis=1))
print('xyz measur error total: ',  xyz_meas_err_mean)
print('xyz filter error total: ',  xyz_filt_err_mean)

print('R measur error total: ', R_meas_err_norm)
print('R filter error total: ', R_filt_err_norm)
print('R measur error total: ', R_meas_err_mean)
print('R filter error total: ', R_filt_err_mean)

# plot
fp.plot_errors()
fp.plot_3sigma()
fp.plot_velocities()
