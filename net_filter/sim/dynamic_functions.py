import os
import pickle
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as pp

import net_filter.tools.unit_conversion as conv
import net_filter.tools.so3 as so3
import net_filter.dynamics.unscented_filter as uf

def get_noise_covariances():

    # process noise (w) covariance
    cov_p = .05*np.array([1, 1, 1])
    cov_R = .02*np.array([1, 1, 1])
    cov_pdot = .002*np.array([1, 1, 1])
    cov_om = .005*np.array([1, 1, 1])
    cov_ww = .0001*np.block([cov_p, cov_R, cov_pdot, cov_om])
    COV_ww = np.diag(cov_ww)

    # measurement noise (v) covariance
    cov_p = (1/(100**2))*np.array([14, 979, 9]) # light energy 6
    cov_R = (1/((180/np.pi)**2))*np.array([198, 586, 230]) # light energy 6
    cov_vv = np.block([cov_p, cov_R])
    COV_vv = np.diag(cov_vv)

    return COV_ww, COV_vv


def conversion_and_error(t, p, R, pdot, om, p_filt, R_filt, pdot_filt, om_filt,
                         p_meas, R_meas, COV_XX_ALL, save_dir):

    # determine n_ims
    n_ims = p.shape[1]

    # translation: convert to cm
    p *= conv.m_to_cm
    p_meas *= conv.m_to_cm
    p_filt *= conv.m_to_cm
    pdot *= conv.m_to_cm
    pdot_filt *= conv.m_to_cm
    
    # translation: errors
    p_err_filt = np.linalg.norm(p_filt - p, axis=0)
    p_err_meas = np.linalg.norm(p_meas - p, axis=0)

    # rotation: errors
    R_err_filt = np.full(n_ims, np.nan)
    R_err_meas = np.full(n_ims, np.nan)
    s_err_filt = np.full((3,n_ims), np.nan)
    for j in range(n_ims):
        R_err_filt[j] = so3.geodesic_distance(R[:,:,j], R_filt[:,:,j])
        s_err_filt[:,j] = so3.skew_elements(so3.log(R[:,:,j].T @ R_filt[:,:,j]))
        R_err_meas[j] = so3.geodesic_distance(R[:,:,j], R_meas[:,:,j])

    # rotation: convert to degrees
    R_err_filt *= conv.rad_to_deg
    R_err_meas *= conv.rad_to_deg
    s_err_filt *= conv.rad_to_deg
    om *= conv.rad_to_deg
    om_filt *= conv.rad_to_deg

    # translation and rotation: convert covariance
    p_unit_vec = np.tile(conv.m_to_cm, 6)
    R_unit_vec = np.tile(conv.rad_to_deg, 6)
    unit_vec = np.block([p_unit_vec, R_unit_vec])
    UNIT_MAT = np.diag(unit_vec)
    COV_XX_ALL_OLD = np.copy(COV_XX_ALL)
    for i in range(n_ims):
        COV_XX_ALL[:,:,i] = UNIT_MAT @ COV_XX_ALL[:,:,i] @ UNIT_MAT 

    # save to file
    filter_results_npz = os.path.join(save_dir, 'filter_results.npz')
    np.savez(filter_results_npz, t=t,
             p=p, R=R, pdot=pdot, om=om,
             p_meas=p_meas, R_meas=R_meas,
             p_err_meas=p_err_meas, R_err_meas=R_err_meas,
             p_filt=p_filt, R_filt=R_filt,
             pdot_filt=pdot_filt, om_filt=om_filt,
             p_err_filt=p_err_filt, R_err_filt=R_err_filt,
             s_err_filt=s_err_filt, COV_XX_ALL=COV_XX_ALL)
