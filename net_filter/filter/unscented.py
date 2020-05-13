import numpy as np
import scipy as sp
from scipy import linalg
import transforms3d as t3d

import net_filter.tools.so3 as so3
import net_filter.dynamics.angular_velocity as av
import net_filter.dynamics.rigid_body as rb

def f_constant_ang(xyz_prev, R_prev, v_prev, om_prev, w, u, dt):
    '''
    dynamics and integration
    '''

    # translations
    g = 9.8
    vz_new = v_prev[2] - g*dt
    z_new = xyz_prev[2] + v_prev[2]*dt - .5*g*dt**2
    v_new = np.block([v_prev[:2], vz_new]) + w[6:9]
    xyz_new = np.block([xyz_prev[:2], z_new]) + w[:3]

    # rotations
    om_new = om_prev + w[9:]
    tangent = dt * om_prev 
    R_mat = so3.exp(so3.cross(tangent))
    R_noise = so3.exp(so3.cross(w[3:6]))
    #R_new = R_noise @ R_prev @ R_mat # orig
    R_new = R_prev @ R_mat @ R_noise

    return xyz_new, R_new, v_new, om_new


def f(xyz_prev, R_prev, v_prev, om_prev, w, u, dt):
    '''
    rigid body dynamics
    '''
    
    t = np.array([0, dt])
    q_prev = t3d.quaternions.mat2quat(R_prev)
    q_dot_prev = av.om_to_qdot(om_prev, q_prev)
    x0 = np.concatenate((xyz_prev, q_prev, v_prev, q_dot_prev))
    x = rb.integrate(t, x0)

    xf = x[:,-1]
    xyz_new = xf[:3]
    q_new = xf[3:7]
    R_new = t3d.quaternions.quat2mat(q_new)
    v_new = xf[7:10]
    q_dot_new = xf[10:]
    om_new = av.qdot_to_om(q_new, q_dot_new)

    return xyz_new, R_new, v_new, om_new


def h(xyz_prev, R_prev, u, v):
    '''
    measurement function
    '''

    xyz_new = xyz_prev + v[:3]
    R_noise = so3.exp(so3.cross(v[3:]))
    #R_new = R_noise @ R_prev 
    R_new = R_prev @ R_noise

    return xyz_new, R_new


def filter(f, h, Q_cov, R_cov, xyz0_hat, R0_hat, v0_hat, om0_hat, P0_hat,
           U, XYZ_MEAS, R_MEAS, dt, alpha=1e-2, beta=2, kappa=2, tan_rot=False):
    '''
    Unscented Filter from Section 3.7 of Optimal Estimation of Dynamic Systems
    (2nd ed.) by Crassidis & Junkins
    x is
    '''

    # system constants
    n = 12
    n_w = Q_cov.shape[0]
    n_v = R_cov.shape[0]
    n_t = XYZ_MEAS.shape[1] # number of time points
    n_y = 6

    # constants
    L = n + n_w + n_v # size of augmented state
    lam = (alpha**2)*(L + kappa) - L # eq. 3.256
    gam = np.sqrt(L + lam) # eq. 3.255

    # weights (eq. 3.259)
    W_mean = np.full(2*L + 1, np.nan)
    W_cov = np.full(2*L + 1, np.nan)
    W_mean[0] = lam/(L + lam)
    W_cov[0] = W_mean[0] + (1 - alpha**2 + beta)
    W_mean[1:] = 1/(2*(L + lam))
    W_cov[1:] = 1/(2*(L + lam))
    
    # create arrays for saving and load initial values for the loop
    XYZ_HAT = np.full((3, n_t), np.nan)
    R_HAT = np.full((3, 3, n_t), np.nan)
    V_HAT = np.full((3, n_t), np.nan)
    OM_HAT = np.full((3, n_t), np.nan)
    P_ALL = np.full((n,n,n_t), np.nan)

    XYZ_HAT[:,0] = xyz0_hat
    R_HAT[:,:,0] = R0_hat
    V_HAT[:,0] = v0_hat
    OM_HAT[:,0] = om0_hat

    y_chi = np.full((n_y, 2*L + 1), np.nan)
    tangent0_hat = so3.skew_elements(so3.log(R0_hat))
    x_hat = np.block([xyz0_hat, tangent0_hat, v0_hat, om0_hat])
    x_hat = x_hat[:,np.newaxis]
    P = P0_hat

    if tan_rot:
        R_hat = R0_hat
        x_hat[3:6,:] = 0

    for i in range(1, n_t):
        # load control and measurement at time i
        u = U[:,i]
        R_MEAS_i = R_MEAS[:,:,i]
        if tan_rot:
            R_MEAS_i = R_hat.T @ R_MEAS_i
            #R_MEAS_i = R_hat @ R_MEAS_i.T # new
        y = np.block([XYZ_MEAS[:,i], so3.skew_elements(so3.log(R_MEAS_i))])
        y = y[:,np.newaxis]
        test = so3.skew_elements(so3.log(R_MEAS_i))

        # cross-correlations (usually zero, see text after eq. 3.252)
        Pxw = np.full((n,n_w), 0.0)
        Pxv = np.full((n,n_v), 0.0)
        Pwv = np.full((n_w,n_v), 0.0)

        # sigma points
        Pa = np.block([[P,     Pxw,   Pxv],
                       [Pxw.T, Q_cov, Pwv],
                       [Pxv.T, Pwv.T, R_cov]])
        cols = gam * np.linalg.cholesky(Pa) # returns L such that Pa = L @ L.T
        sig = np.block([cols, -cols])
        x_hat_a = np.block([[x_hat], [np.zeros((n_w+n_v,1))]])
        chi = np.block([x_hat_a, x_hat_a + sig])
        x_chi = chi[:n,:]
        w_chi = chi[n:n+n_w,:]
        v_chi = chi[n+n_w:,:]

        # propagate
        for j in range(2*L + 1):

            # integrate
            R_j = so3.exp(so3.cross(x_chi[3:6,j]))
            om_j = x_chi[9:,j]

            if tan_rot:
                R_j = R_hat @ R_j
                #om_j = R_hat @ om_j
            xyz_new, R_new, v_new, om_new = f(
                    x_chi[:3,j], R_j, x_chi[6:9,j], om_j, w_chi[:,j], u, dt)

            tangent_new = so3.skew_elements(so3.log(R_new))
            if tan_rot:
                tangent_new = so3.skew_elements(so3.log(R_hat.T @ R_new))
                #tangent_new = so3.skew_elements(so3.log(R_hat.T @ R_new)) # new
                #om_new = R_hat.T @ om_new
                
            x_chi[:,j] = np.block([xyz_new, tangent_new, v_new, om_new]) 

            # measurement
            xyz_meas, R_meas = h(xyz_new, R_new, u, v_chi[:,j])
            tangent_meas = so3.skew_elements(so3.log(R_meas))
            if tan_rot:
                tangent_meas = so3.skew_elements(so3.log(R_hat.T @ R_meas))
                #tangent_meas = so3.skew_elements(so3.log(R_hat.T @ R_meas)) # new
            y_chi[:,j] = np.block([xyz_meas, tangent_meas]) 

        # predictions
        x_hat = np.sum(W_mean * x_chi, axis=1)
        x_hat = x_hat[:,np.newaxis]
        P = W_cov * (x_chi - x_hat) @ (x_chi - x_hat).T
        y_hat = np.sum(W_mean * y_chi, axis=1) # eq. 2.262
        y_hat = y_hat[:,np.newaxis]

        # covariances
        Pyy = W_cov * (y_chi - y_hat) @ (y_chi - y_hat).T
        Peyey = Pyy
        Pexey = W_cov * (x_chi - x_hat) @ (y_chi - y_hat).T

        # update
        e = y - y_hat # eq. 3.250
        K = Pexey @ np.linalg.inv(Peyey) # eq. 3.251
        x_hat = x_hat + K @ e # eq. 3.249a
        P = P - K @ Peyey @ K.T # eq. 3.249b

        # rotate frame back to identity
        XYZ_HAT[:,i] = x_hat[:3].ravel()
        R_delta = so3.exp(so3.cross(x_hat[3:6].ravel()))
        if tan_rot:
            #big_mat = sp.linalg.block_diag(np.eye(3), R_hat, np.eye(3), np.eye(3))
            big_mat = sp.linalg.block_diag(np.eye(3), np.eye(3), np.eye(3), np.eye(3))
            #big_mat = sp.linalg.block_diag(np.eye(3), R_delta, np.eye(3), np.eye(3))
            P = big_mat @ P @ big_mat.T
            R_hat = R_hat @ R_delta
            x_hat[3:6] = 0
            #om_hat = R_delta @ x_hat[9:].ravel()
            om_hat = x_hat[9:].ravel()
        else:
            R_hat = R_delta
            om_hat = x_hat[9:].ravel()

        # save
        P_ALL[:,:,i] = P
        R_HAT[:,:,i] = R_hat
        V_HAT[:,i] = x_hat[6:9].ravel()
        OM_HAT[:,i] = om_hat

    return XYZ_HAT, R_HAT, V_HAT, OM_HAT, P_ALL
