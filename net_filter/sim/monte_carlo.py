import os
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.dope.dope_to_blender as db
import net_filter.dynamics.rigid_body as rb
import net_filter.sim.dynamic_gen as dg
import net_filter.sim.dynamic_filter as df
import net_filter.sim.dynamic_filter_plots as fp

# set up
n_trials = 100
np.random.seed(82)

p_err_mean = np.full(n_trials, np.nan)
p_err_mean_meas = np.full(n_trials, np.nan)
R_err_mean_meas = np.full(n_trials, np.nan)
R_err_mean = np.full(n_trials, np.nan)

# times
n_ims = 30
t = np.linspace(0, .5, n_ims)
inds = np.arange(n_ims)
dt = t[1] - t[0]

# initial conditions
p0 = np.array([-.1, .7, -.1])
pdot0 = np.array([0.9, 1.1, 2.3])
om0 = np.array([5, 8, 4])
R0_all = np.full((3,3,n_trials),np.nan)
for i in range(n_trials):
    R0_all[:,:,i] = so3.random_rotation_matrix()

for i in range(n_trials):
    # directory
    img_dir_i = os.path.join(dirs.monte_carlo_dir, str(i) + '/')
    if not os.path.exists(img_dir_i):
        os.makedirs(img_dir_i)

    R0 = R0_all[:,:,i]

    # rigid body dynamics
    p, R, pdot, om = rb.integrate(t, p0, R0, pdot0, om0)

    # regenerate and re-evaluate images?
    regen = 1
    if regen:
        dg.generate_images(n_ims, dt, p, R, pdot, om, img_dir_i)
        p, R, p_est, R_est = db.get_predictions(img_dir_i)

    # load dope pose estimates
    npz_file = os.path.join(img_dir_i, 'dope_pR.npz')
    data = np.load(npz_file)
    p_meas = data['p']
    R_meas = data['R']

    # filter initial estimates
    p0_hat = p_meas[:,0] # use the true value to make it a fair comparison
    R0_hat = R_meas[:,:,0] # use the true value to make it a fair comparison
    pdot0_nrm = np.linalg.norm(pdot0)
    pdot0_hat = pdot0 + .2*np.random.uniform(-1,1,3)
    om0_nrm = np.linalg.norm(om0)
    om0_hat = om0 + .6*np.random.uniform(-1,1,3)
    cov_xx_0_hat = np.ones(12)
    COV_xx_0_hat = np.diag(cov_xx_0_hat)

    # run filter
    p_hat, R_hat, pdot_hat, om_hat, COV_XX_ALL = df.apply_filter(p0_hat, R0_hat, pdot0_hat, om0_hat, COV_xx_0_hat, t, dt, p_meas, R_meas)

    # convert outputs and calculate errors
    p, R, v, om, p_hat, R_hat, pdot_hat, om_hat, p_meas, R_meas, p_err, p_err_meas, R_err, R_err_meas, COV_XX_ALL = df.conversion_and_error(t, p, R, pdot, om, p_hat, R_hat, pdot_hat, om_hat, p_meas, R_meas, COV_XX_ALL, img_dir_i)

    # print
    print('p measur error total: ',  np.mean(p_err_meas))
    print('p filter error total: ',  np.mean(p_err))
    print('R measur error total: ', np.mean(R_err_meas))
    print('R filter error total: ', np.mean(R_err))

    # totals
    p_err_mean[i] = np.mean(p_err)
    p_err_mean_meas[i] = np.mean(p_err_meas)
    R_err_mean[i] = np.mean(R_err)
    R_err_mean_meas[i] = np.mean(R_err_meas)

# save all results
trials_npz = os.path.join(dirs.monte_carlo_dir, 'trials.npz')
np.savez(trials_npz, p_err_mean=p_err_mean, p_err_mean_meas=p_err_mean_meas, R_err_mean=R_err_mean, R_err_mean_meas=R_err_mean_meas)
