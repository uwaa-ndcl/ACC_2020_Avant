import os
import pickle
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as pp

import net_filter.tools.unit_conversion as conv
import net_filter.tools.so3 as so3
import net_filter.dynamics.unscented_filter as uf


def apply_filter(xyz0_hat, R0_hat, xyzdot0_hat, om0_hat, P0_hat,
                 t, dt, xyz_meas, R_meas):

    # get number of time points
    n_t = len(t)

    # process noise covariance
    q_cov_xyz = .05*np.array([1, 1, 1])
    q_cov_R = .02*np.array([1, 1, 1])
    q_cov_v = .002*np.array([1, 1, 1])
    q_cov_om = .005*np.array([1, 1, 1])
    q_cov = .0001*np.block([q_cov_xyz, q_cov_R, q_cov_v, q_cov_om])
    Q_cov = np.diag(q_cov)

    # measurement noise covariance
    r_cov_xyz = (1/(100**2))*np.array([14, 979, 9]) # light energy 6
    r_cov_R = (1/((180/np.pi)**2))*np.array([198, 586, 230]) # light energy 6
    r_cov = np.block([r_cov_xyz, r_cov_R])
    R_cov = np.diag(r_cov)

    # run filter
    U = np.full((1, n_t), 0.0)

    # run the unscented filter
    xyz_hat, R_hat, xyzdot_hat, om_hat, P_XX_ALL = uf.filter(
            uf.f, uf.h, Q_cov, R_cov, xyz0_hat, R0_hat,
            xyzdot0_hat, om0_hat, P0_hat, U, xyz_meas, R_meas, dt)

    return xyz_hat, R_hat, xyzdot_hat, om_hat, P_XX_ALL


def conversion_and_error(t, xyz, R, v, om,
                         xyz_hat, R_hat, v_hat, om_hat,
                         P_ALL,
                         xyz_meas, R_meas, save_dir):

    # determine n_ims
    n_ims = xyz.shape[1]

    # translation: convert to cm
    xyz *= conv.m_to_cm
    xyz_meas *= conv.m_to_cm
    xyz_hat *= conv.m_to_cm
    v *= conv.m_to_cm
    v_hat *= conv.m_to_cm
    
    # translation: errors
    xyz_err = np.linalg.norm(xyz_hat - xyz, axis=0)
    xyz_err_meas = np.linalg.norm(xyz_meas - xyz, axis=0)

    # rotation: errors
    R_err = np.full(n_ims, np.nan)
    R_err_meas = np.full(n_ims, np.nan)
    s_err = np.full((3,n_ims), np.nan)
    for j in range(n_ims):
        R_err[j] = so3.geodesic_distance(R[:,:,j], R_hat[:,:,j])
        s_err[:,j] = so3.skew_elements(so3.log(R[:,:,j].T @ R_hat[:,:,j]))
        R_err_meas[j] = so3.geodesic_distance(R[:,:,j], R_meas[:,:,j])

    # rotation: convert to degrees
    R_err *= conv.rad_to_deg
    R_err_meas *= conv.rad_to_deg
    s_err *= conv.rad_to_deg
    om *= conv.rad_to_deg
    om_hat *= conv.rad_to_deg

    # translation and rotation: convert covariance
    xyz_unit_vec = np.tile(conv.m_to_cm, 6)
    R_unit_vec = np.tile(conv.rad_to_deg, 6)
    unit_vec = np.block([xyz_unit_vec, R_unit_vec])
    UNIT_MAT = np.diag(unit_vec)
    P_ALL_OLD = np.copy(P_ALL)
    for i in range(n_ims):
        P_ALL[:,:,i] = UNIT_MAT @ P_ALL[:,:,i] @ UNIT_MAT 

    # save to file
    filter_results_npz = os.path.join(save_dir, 'filter_results.npz')
    np.savez(filter_results_npz, t=t,
             xyz=xyz, xyz_meas=xyz_meas, xyz_hat=xyz_hat,
             R=R, R_meas=R_meas, R_hat=R_hat,
             R_err=R_err, s_err=s_err, R_err_meas=R_err_meas,
             v=v, v_hat=v_hat, om=om, om_hat = om_hat,
             P_ALL=P_ALL)

    return xyz, R, v, om, xyz_hat, R_hat, v_hat, om_hat, P_ALL, xyz_meas, R_meas, xyz_err, xyz_err_meas, R_err, R_err_meas
