import numpy as np
import scipy as sp
from scipy import integrate, interpolate
import transforms3d as t3d
import net_filter.tools.so3 as so3

# gravity
g = 9.8

# cylinder (i.e. the soup can)
# note: default rotation (when R=I) is when can's symmetric axis is pointed in
# the y-direction (away from the camera)
r = .04 # radius (meters) (POST-SUBMISSION NOTE: THIS MAY BE A TAD TOO LARGE)
h = .1 # height (meters)

# mass (kg)
# POST-SUBMISSION NOTE: THIS VALUE IS TOO LARGE, IT SHOULD BE ABOUT .35 kg,
# BUT THIS DOESN'T SEEM TO MAKE A BIG DIFFERENCE IN THE SIMULATION
m = 1
J = np.diag([(1/12)*m*(3*r**2 + h**2), .5*m*r**2, (1/12)*m*(3*r**2 + h**2)])

def newton_euler(t, pdot_om):
    '''
    Newton-Euler equations for a rigid body
    assumes velocity (pdot_A) is in inertial frame (A) coordinates,
    and omega (om_B) is in body-frame (B) coordinates
    ref: eq. 4.16 (p. 167) in A Mathematical Introduction to Robotic
    Manipulation
    '''

    pdot_A = pdot_om[:3]
    om_B = pdot_om[3:]
    om_col_B = om_B[:,np.newaxis] # omega as column vector
    
    # translational dynamics: all expressed in inertial frame A
    F_A = np.array([[0, 0, -m*g]]).T # inertial frame force of gravity
    pdot_dot_A = np.linalg.inv(m*np.eye(3)) @ F_A

    # rotational dynamics: all expressed in body frame B
    tau_B = np.array([[0, 0, 0]]).T     # inertial frame torque
    omdot_B = np.linalg.inv(J) @ (tau_B - so3.cross(om_B) @ J @ om_col_B)
    pdot_omdot = np.concatenate((pdot_dot_A, omdot_B)).ravel()

    return pdot_omdot


def integrate_kinematics(pdot, om, p0, R0, dt):
    '''
    forward Euler integration of angular velocities pdot and omega,
    to p and R

    inputs:
        pdot: trans velocity expressed in A frame, size (3, # time points)
        om: angular velocity expressed in B frame, size (3, # time points)
        p0: initial p, size (3)
        R0: initial rotation matrix, size (3,3)
        dt: timestep at which values are spaced
    '''

    n_pts = pdot.shape[1]

    # define arrays
    p = np.full((3,n_pts), np.nan)
    R = np.full((3,3,n_pts), np.nan)

    # 1st order forward Euler integration
    p[:,0] = p0
    R[:,:,0] = R0
    for i in range(1, n_pts):
        p[:,i] = p[:,i-1] + dt*pdot[:,i-1]
        R[:,:,i] = R[:,:,i-1] @ so3.exp(so3.cross(dt*om[:,i-1]))

    return p, R


def integrate(t, p0, R0, pdot0, om0):
    '''
    integrate Newton-Euler equations, then integrate kinematics (t=times)
    '''

    # setup
    t0 = t[0]
    tf = t[-1]
    n_t = t.size

    # integration times
    dt_int = .001
    t_int = np.arange(t0, tf+dt_int, dt_int) # make sure to include last point
    tf_int = t_int[-1]
    n_int = t_int.size

    # integrate newton-euler (state is pdot & om)
    pdot_om0 = np.concatenate((pdot0, om0))
    pdot_om0_col = pdot_om0[:,np.newaxis]
    sol = sp.integrate.solve_ivp(newton_euler, (t0, tf_int), pdot_om0,
                                 method='RK45', t_eval=t_int)
    pdot_om_int = sol.y
    pdot_int = pdot_om_int[:3,:]
    om_int = pdot_om_int[3:,:]
    
    # integrate kinematics
    p_int, R_int = integrate_kinematics(pdot_int, om_int, p0, R0, dt_int)

    # get tangent space representation of R_int so we can interpolate
    s_int = np.full((3,n_int), np.nan)
    for i in range(n_int):
        s_int[:,i] = so3.skew_elements(so3.log(R_int[:,:,i]))

    # interpolate t values using state x=(p,s,v,om)
    x_int = np.concatenate((p_int, s_int, pdot_int, om_int), axis=0)
    interp_fun = sp.interpolate.interp1d(t_int, x_int)
    x = interp_fun(t)

    # get x values
    p = x[:3,:]
    s = x[3:6,:]
    pdot = x[6:9,:]
    om = x[9:,:]

    # convert tangent space s back into rotations
    R = np.full((3,3,n_t), np.nan)
    for i in range(n_t):
        R[:,:,i] = so3.exp(so3.cross(s[:,i]))

    return p, R, pdot, om
