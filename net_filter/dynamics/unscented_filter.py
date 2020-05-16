import numpy as np
import scipy as sp
from scipy import linalg
import transforms3d as t3d

import net_filter.tools.so3 as so3
import net_filter.dynamics.angular_velocity as av
import net_filter.dynamics.rigid_body as rb

def f(xyz_prev, R_prev, xyzdot_prev, om_prev, w, u, dt):
    '''
    rigid body dynamics
    '''
    
    t = np.array([0, dt])
    xyz, R, xyzdot, om = rb.integrate(t, xyz_prev, R_prev, xyzdot_prev, om_prev)

    return xyz[:,-1], R[:,:,-1], xyzdot[:,-1], om[:,-1]


def h(xyz_prev, R_prev, u, v):
    '''
    measurement function
    '''

    xyz_new = xyz_prev + v[:3]
    R_noise = so3.exp(so3.cross(v[3:]))
    R_new = R_prev @ R_noise

    return xyz_new, R_new


def filter(f, h, Q_cov, R_cov, xyz0_hat, R0_hat, xyzdot0_hat, om0_hat, P_xx_0_hat,
           U, XYZ_MEAS, R_MEAS, dt):
    '''
    Unscented Filter from Section 3.7 of Optimal Estimation of Dynamic Systems
    (2nd ed.) by Crassidis & Junkins
    '''

    # system constants
    alpha = 1e-2
    beta = 2
    kappa = 2
    n = 12
    n_w = Q_cov.shape[0]
    n_v = R_cov.shape[0]
    n_t = XYZ_MEAS.shape[1] # number of time points
    n_y = 6

    # constants
    L = n + n_w + n_v # size of augmented state
    lam = (alpha**2)*(L + kappa) - L # eq. 3.256
    gam = np.sqrt(L + lam) # eq. 3.255

    # weights
    W_mean = np.full(2*L + 1, np.nan)
    W_cov = np.full(2*L + 1, np.nan)
    W_mean[0] = lam/(L + lam)
    W_cov[0] = W_mean[0] + (1 - alpha**2 + beta)
    W_mean[1:] = 1/(2*(L + lam))
    W_cov[1:] = 1/(2*(L + lam))
    
    # create arrays for saving values in the for loop
    XYZ_HAT = np.full((3, n_t), np.nan)
    R_HAT = np.full((3, 3, n_t), np.nan)
    XYZDOT_HAT = np.full((3, n_t), np.nan)
    OM_HAT = np.full((3, n_t), np.nan)
    P_XX_ALL = np.full((n,n,n_t), np.nan)
    Y_HAT = np.full((n_y, 2*L+1), np.nan) # y_hat^(i) for i=0,...,2L

    # fill arraysw with initial estimates
    XYZ_HAT[:,0] = xyz0_hat
    R_HAT[:,:,0] = R0_hat
    XYZDOT_HAT[:,0] = xyzdot0_hat
    OM_HAT[:,0] = om0_hat
    s0_hat = np.array([0,0,0])

    # initialize variables to be used in for loop
    x_hat = np.block([xyz0_hat, s0_hat, xyzdot0_hat, om0_hat])
    x_hat = x_hat[:,np.newaxis]
    P_xx = P_xx_0_hat
    R_hat = R0_hat

    # iterate over all time points
    for i in range(1, n_t):
        # load control and measurement at time i
        u = U[:,i]
        s_meas = so3.skew_elements(so3.log(R_hat.T @ R_MEAS[:,:,i])) # eq. 5
        y = np.block([XYZ_MEAS[:,i], s_meas])
        y = y[:,np.newaxis]

        # cross-correlations (usually zero, see Crassidis after eq. 3.252)
        Pxw = np.full((n,n_w), 0.0)
        Pxv = np.full((n,n_v), 0.0)
        Pwv = np.full((n_w,n_v), 0.0)

        # sigma points
        P_chi = np.block([[P_xx,  Pxw,   Pxv],
                          [Pxw.T, Q_cov, Pwv],
                          [Pxv.T, Pwv.T, R_cov]])
        M = gam * np.linalg.cholesky(P_chi) # returns L such that P_chi = L @ L.T
        sig = np.block([M, -M])
        chi_hat_pls = np.block([[x_hat], [np.zeros((n_w+n_v,1))]])
        CHI_HAT = np.block([chi_hat_pls, chi_hat_pls + sig]) # all chi_hat^(i) for i=0,...,2L
        X_HAT = CHI_HAT[:n,:] # all x_hat^(i) for i=0,...,2L
        W_HAT = CHI_HAT[n:n+n_w,:] # all w_hat^(i) for i=0,...,2L
        V_HAT = CHI_HAT[n+n_w:,:] # all v_hat^(i) for i=0,...,2L

        # iteration (over all sigma points)
        for k in range(2*L + 1):
            # dynamics
            om_k = X_HAT[9:,k]
            R_k = R_hat @ so3.exp(so3.cross(X_HAT[3:6,k]))
            xyz_new, R_new, v_new, om_new = f(
                    X_HAT[:3,k], R_k, X_HAT[6:9,k], om_k, W_HAT[:,k], u, dt)
            s_new = so3.skew_elements(so3.log(R_hat.T @ R_new))
            X_HAT[:,k] = np.block([xyz_new, s_new, v_new, om_new]) 

            # measurement
            xyz_meas, R_meas = h(xyz_new, R_new, u, V_HAT[:,k])
            s_meas = so3.skew_elements(so3.log(R_hat.T @ R_meas))
            Y_HAT[:,k] = np.block([xyz_meas, s_meas]) 

        # predictions
        x_hat = np.sum(W_mean * X_HAT, axis=1)
        x_hat = x_hat[:,np.newaxis]
        P_xx = W_cov * (X_HAT - x_hat) @ (X_HAT - x_hat).T # P_xx minus
        y_hat = np.sum(W_mean * Y_HAT, axis=1) # eq. 2.262
        y_hat = y_hat[:,np.newaxis]

        # covariances
        P_yy = W_cov * (Y_HAT - y_hat) @ (Y_HAT - y_hat).T
        P_xy = W_cov * (X_HAT - x_hat) @ (Y_HAT - y_hat).T

        # update
        e = y - y_hat # eq. 3.250
        K = P_xy @ np.linalg.inv(P_yy) # eq. 3.251
        x_hat = x_hat + K @ e # eq. 3.249a
        P_xx = P_xx - K @ P_yy @ K.T # eq. 3.249b

        # rotation
        R_hat = R_hat @ so3.exp(so3.cross(x_hat[3:6].ravel()))
        x_hat[3:6] = 0

        # save values
        XYZ_HAT[:,i] = x_hat[:3].ravel()
        R_HAT[:,:,i] = R_hat
        XYZDOT_HAT[:,i] = x_hat[6:9].ravel()
        OM_HAT[:,i] = x_hat[9:].ravel()
        P_XX_ALL[:,:,i] = P_xx

    return XYZ_HAT, R_HAT, XYZDOT_HAT, OM_HAT, P_XX_ALL
