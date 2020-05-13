import os
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.sim.dope_to_blender as db
import net_filter.dynamics.rigid_body as rb
import net_filter.dynamics.angular_velocity as av
import net_filter.sim.dynamic_gen as dg
import net_filter.sim.dynamic_filter as df
import net_filter.sim.dynamic_filter_plots as fp

# set up
n_trials = 100
np.random.seed(82)

xyz_err_mean = np.full(n_trials, np.nan)
xyz_err_mean_meas = np.full(n_trials, np.nan)
R_err_mean_meas = np.full(n_trials, np.nan)
R_err_mean = np.full(n_trials, np.nan)

# times
n_ims = 30
t = np.linspace(0, .5, n_ims)
inds = np.arange(n_ims)
dt = t[1] - t[0]

# initial conditions
xyz0 = np.array([-.1, .7, -.14+.04]) # paper
v0 = np.array([0.9, 1.1, 2.3]) # paper
om0 = np.array([5, 8, 4]) # paper
q0_all = np.full((4,n_trials),np.nan)
for i in range(n_trials):
    R0 = so3.random_rotation_matrix()
    q0_all[:,i] = t3d.quaternions.mat2quat(R0)

for i in range(n_trials):
    # directory
    img_dir_i = os.path.join(dirs.trials_dir, str(i) + '/')
    if not os.path.exists(img_dir_i):
        os.makedirs(img_dir_i)

    #om0 = np.array([9, 11, 8])
    #R0 = so3.random_rotation_matrix()
    #q0 = t3d.quaternions.mat2quat(R0)
    q0 = q0_all[:,i]

    # get orientaion
    #q0 = q0_all[:,i]

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
    for j in range(n_ims):
        om[:,j] = av.qdot_to_om(q[:,j], qdot[:,j])

    # generate images (comment this if you don't want to do it again)
    #dg.generate_images(n_ims, dt, xyz, q, v, om, img_dir_i)

    # evaluate images (comment this if you don't want to do it again)
    #xyz, q, xyz_est, q_est = db.get_predictions(img_dir_i)

    # load dope pose estimates
    npz_file = os.path.join(img_dir_i, 'dope_xyzq.npz')
    data = np.load(npz_file)
    xyz_meas = data['xyz']
    q_meas = data['q']

    # convert quaternions to rotation matrices
    R = np.full((3,3,n_ims),np.nan)
    R_meas = np.full((3,3,n_ims),np.nan)
    for j in range(n_ims):
        R[:,:,j] = t3d.quaternions.quat2mat(q[:,j])
        R_meas[:,:,j] = t3d.quaternions.quat2mat(q_meas[:,j])

    # filter initial estimates
    #xyz0_hat = xyz_meas[:,0] + np.array([-.2, .3, .1])
    xyz0_hat = xyz_meas[:,0] # use the true value to make it a fair comparison
    #R0_noise = t3d.euler.euler2mat(.07, .03, .0, 'sxyz')
    #R0_hat = R0_noise @ R_meas[:,:,0]
    R0_hat = R_meas[:,:,0] # use the true value to make it a fair comparison
    #v0 = v[:,0]
    v0_nrm = np.linalg.norm(v0)
    v0_hat = v0 + .5*np.array([-1.1, 1.2, 1.1])
    v0_hat = v0 + .07*np.array([-1.1, 1.2, 1.1])
    v0_hat = v0 + 0*.17*np.array([-1.1, 1.2, 1.1])
    v0_hat = v0 + .2*np.random.uniform(-1,1,3)
    #v0_hat = v0 + .00*np.array([-1.1, 1.2, 1.1])
    #om0 = np.array([5, 8, 4])
    #om0 = om[:,0]
    om0_nrm = np.linalg.norm(om0)
    om0_hat = om0 + .5*np.array([-1.0, 1.1, 1.0])
    om0_hat = om0 + 1.2*np.array([-1.0, 1.1, 1.0])
    om0_hat = om0 + 0*.5*np.array([-1.0, 1.1, 1.0])
    om0_hat = om0 + .6*np.random.uniform(-1,1,3)
    #P0_hat = 1 * np.eye(n)
    p0_hat = np.array([.01, .01, .01,     # p
                       .2, .2, .2,  # s
                       10, 80, 10,     # v
                       #30, 30, 30]) # omega
                       1, 1, 1]) # omega
    p0_hat = np.ones(12)
    P0_hat = np.diag(p0_hat)

    # run filter
    xyz_hat, R_hat, v_hat, om_hat, P_ALL = df.apply_filter(xyz0_hat, R0_hat, v0_hat, om0_hat, P0_hat, t, dt, xyz_meas, R_meas)

    # convert outputs and calculate errors
    xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, xyz_err, xyz_err_meas, R_err, R_err_meas = df.conversion_and_error(t, xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, img_dir_i)

    # print
    print('xyz measur error total: ',  np.mean(xyz_err_meas))
    print('xyz filter error total: ',  np.mean(xyz_err))
    print('R measur error total: ', np.mean(R_err_meas))
    print('R filter error total: ', np.mean(R_err))
    '''
    import matplotlib.pyplot as pp
    pp.figure()
    pp.plot(t, R_err)
    pp.show()
    '''
    # totals
    xyz_err_mean[i] = np.mean(xyz_err)
    xyz_err_mean_meas[i] = np.mean(xyz_err_meas)
    R_err_mean[i] = np.mean(R_err)
    R_err_mean_meas[i] = np.mean(R_err_meas)

# save all results
trials_npz = os.path.join(dirs.trials_dir, 'trials.npz')
np.savez(trials_npz, xyz_err_mean=xyz_err_mean, xyz_err_mean_meas=xyz_err_mean_meas, R_err_mean=R_err_mean, R_err_mean_meas=R_err_mean_meas)
