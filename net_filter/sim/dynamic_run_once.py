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
xyz0 = np.array([-.1, .7, -.14+.04]) # paper
q0 = t3d.euler.euler2quat(0.7*np.pi, .5*np.pi, .0*np.pi, 'sxyz')
q0 = t3d.euler.euler2quat(.45*np.pi, .8*np.pi, -.2*np.pi, 'sxyz') # not bad
#q0 = t3d.euler.euler2quat(-.7*np.pi, .3*np.pi, -.1*np.pi, 'sxyz') # paper
#q0 = t3d.euler.euler2quat(-.8*np.pi, .7*np.pi, -.1*np.pi, 'sxyz') # paper
q0 = t3d.euler.euler2quat(1.22, -1.31, .49, 'sxyz') # 10
q0 = t3d.euler.euler2quat(-.70, -1.31, 2.69, 'sxyz') # 19
q0 = t3d.euler.euler2quat(2.14, .05, -.91, 'sxyz') # 42 new
q0 = t3d.euler.euler2quat(-.70, -1.25, 2.69, 'sxyz') # 19 new, slightly modified
'''
# debug
# 7, 19, 22, 26
# 19?, 26?, 42, 45 
import pickle
to_render_pkl = os.path.join(img_dir, 'to_render.pkl')
with open(to_render_pkl, 'rb') as input:
    render_props = pickle.load(input)
q0 = render_props.quat[:,0]
print(t3d.euler.quat2euler(q0, 'sxyz'))
'''
v0 = np.array([0.9, 1.1, 2.3]) # paper
om0 = np.array([5, 8, 4]) # paper
#om0 = np.array([9, 11, 8])

# rigid body dynamics
q_dot0 = av.om_to_qdot(om0, q0)
x0 = np.concatenate((xyz0, q0, v0, q_dot0), axis=0)
X = rb.integrate(t, x0)
xyz = X[:3,:]
q = X[3:7,:]
v = X[7:10,:]
qdot = X[10:,:]

# convert q dot to omega
om = np.full((3,n_ims), np.nan)
for i in range(n_ims):
    om[:,i] = av.qdot_to_om(q[:,i], qdot[:,i])

# generate images (comment this if you don't want to do it again)
#dg.generate_images(n_ims, dt, xyz, q, v, om, img_dir)
#dg.generate_snapshots(n_ims, inds, xyz, q)

# evaluate images (comment this if you don't want to do it again)
#xyz, q, xyz_est, q_est = db.get_predictions(img_dir)

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
#xyz0_hat = xyz_meas[:,0] + np.array([-.2, .3, .1])
xyz0_hat = xyz_meas[:,0] # use  the true value to make it a fair comparison
#R0_noise = t3d.euler.euler2mat(.07, .03, .0, 'sxyz')
#R0_hat = R0_noise @ R_meas[:,:,0]
R0_hat = R_meas[:,:,0] # use  the true value to make it a fair compariso
#v0 = v[:,0]
v0_hat = v0 + .5*np.array([-1.1, 1.2, 1.1])
v0_hat = v0 + .2*np.array([-1.1, 1.2, 1.1])
#om0 = np.array([5, 8, 4])
#om0 = om[:,0]
om0_hat = om0 + .5*np.array([-1.0, 1.1, 1.0])
om0_hat = om0 + .6*np.array([-1.0, 1.1, 1.0])
#om0_hat = om0 + 1.2*np.array([-1.0, 1.1, 1.0])
#P0_hat = 1 * np.eye(n)
p0_hat = np.array([.01, .01, .01,     # p
                   .2, .2, .2,  # s
                   10, 80, 10,     # v
                   1, 1, 1]) # omega
                   #30, 30, 30]) # omega
p0_hat = np.ones(12)
P0_hat = np.diag(p0_hat)

# run filter
xyz_hat, R_hat, v_hat, om_hat, P_ALL = df.apply_filter(xyz0_hat, R0_hat, v0_hat, om0_hat, P0_hat, t, dt, xyz_meas, R_meas)

# convert outputs and calculate errors
xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, xyz_err, xyz_err_meas, R_err, R_err_meas = df.conversion_and_error(t, xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, img_dir)

# totals
xyz_err_mean = np.mean(xyz_err)
xyz_err_mean_meas = np.mean(xyz_err_meas)
R_err_mean_meas = np.mean(R_err_meas)
R_err_mean = np.mean(R_err)

# print
print('average xyz bias: ', np.mean(xyz - xyz_meas, axis=1))
#print('xyz filter error: ',  xyz_err)
#print('xyz measur error: ',  xyz_err_meas)
print('xyz measur error total: ',  np.linalg.norm(xyz_err_meas))
print('xyz filter error total: ',  np.linalg.norm(xyz_err))

print('R measur error total: ', np.linalg.norm(R_err_meas))
print('R filter error total: ', np.linalg.norm(R_err))
print('R measur error total: ', R_err_mean_meas)
print('R filter error total: ', R_err_mean)

# plot
#fp.plot_error_dynamic()
fp.plot_errors()
fp.plot_3sigma()
fp.plot_velocities()
