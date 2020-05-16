import os
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.sim.dope_to_blender as db
import net_filter.tools.so3 as so3
import net_filter.tools.unit_conversion as conv
import net_filter.dynamics.rigid_body as rb
import net_filter.dynamics.angular_velocity as av
import net_filter.sim.dynamic_gen as dg
import net_filter.sim.dynamic_filter as df
import net_filter.sim.dynamic_filter_plots as fp

# times
n_ims = 30*5
t = np.linspace(0, .5*5, n_ims)
inds = np.arange(n_ims)
dt = t[1] - t[0]

# good bad
xyz0 = np.array([-.1, .7, -.14]) # paper
q0 = t3d.euler.euler2quat(1.22, -1.31, .49, 'sxyz') # 10
v0 = np.array([0.9, 1.1, 2.3]) # paper
om0 = np.array([5, 8, 4]) # paper

# rigid body dynamics
q_dot0 = av.om_to_qdot(om0, q0)
x0 = np.concatenate((xyz0, q0, v0, q_dot0), axis=0)
X = rb.integrate(t, x0)
xyz = X[:3,:]
q = X[3:7,:]
xyzdot = X[7:10,:]
qdot = X[10:,:]

# convert q dot to omega
om = np.full((3,n_ims), np.nan)
for i in range(n_ims):
    om[:,i] = av.qdot_to_om(q[:,i], qdot[:,i])

# simulate measurements
xyz_meas = xyz + .01*(np.random.rand(3,n_ims) - .5)
R_meas = np.full((3,3,n_ims), np.nan)
for i in range(n_ims):
    R = t3d.quaternions.quat2mat(q[:,i])
    noise = .01*(np.random.rand(3) - .5)
    R_noise = so3.exp(so3.cross(noise))
    R_meas[:,:,i] = R_noise @ R

# convert q dot to omega
om = np.full((3,n_ims), np.nan)
for i in range(n_ims):
    om[:,i] = av.qdot_to_om(q[:,i], qdot[:,i])

# filter initial estimates
xyz0_hat = xyz_meas[:,0] # use the true value to make it a fair comparison
R0_hat = R_meas[:,:,0] # use the true value to make it a fair comparison
xyzdot0 = xyzdot[:,0]
xyzdot0_hat = xyzdot0 + .5*np.array([-1.1, 1.2, 1.1])
om0 = om[:,0]
om0_hat = om0 + .5*np.array([-1.0, 1.1, 1.0])
p0_hat = np.array([.01, .01, .01,     # p
                   .2, .2, .2,  # s
                   10, 80, 10,     # p dot
                   10, 10, 10]) # omega
P0_hat = np.diag(p0_hat)

# run filter
xyz_hat, R_hat, v_hat, om_hat, P_ALL = df.apply_filter(xyz0_hat, R0_hat, xyzdot0_hat, om0_hat, P0_hat, t, dt, xyz_meas, R_meas)

om *= conv.rad_to_deg
om_hat *= conv.rad_to_deg
# convert outputs and calculate errors
#xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, xyz_err, xyz_err_meas, R_err, R_err_meas = df.conversion_and_error(t, xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, '/tmp/')

import matplotlib.pyplot as pp
pp.figure()
err = om - om_hat
pp.plot(t, err.T)
pp.show()
