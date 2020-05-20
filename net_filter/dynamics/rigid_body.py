import numpy as np
import scipy as sp
from scipy import integrate, interpolate
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


def integrate(t_out, p0, R0, pdot0, om0):
    '''
    integrate Newton-Euler equations, then integrate kinematics
    t_out = times at which to return states
    '''

    # setup
    t0 = t_out[0]
    tf_out = t_out[-1]
    n_t_out = t_out.size

    # integration times
    dt = .001
    t = np.arange(t0, tf_out+dt, dt) # tf+dt makes sure tf is included
    tf = t[-1]
    n_t = t.size

    # integrate newton-euler (state is pdot & om)
    pdot_om0 = np.concatenate((pdot0, om0))
    sol = sp.integrate.solve_ivp(newton_euler, (t0, tf), pdot_om0,
                                 method='RK45', t_eval=t)
    pdot_om = sol.y
    pdot = pdot_om[:3,:]
    om = pdot_om[3:,:]

    # 1st-order forward Euler integration of kinematics to get p and R
    p = np.full((3,n_t), np.nan)
    R = np.full((3,3,n_t), np.nan)
    p[:,0] = p0
    R[:,:,0] = R0
    for i in range(1, n_t):
        p[:,i] = p[:,i-1] + dt*pdot[:,i-1]
        R[:,:,i] = R[:,:,i-1] @ so3.exp(so3.cross(dt*om[:,i-1]))

    # get tangent space representation of R so we can interpolate
    s = np.full((3,n_t), np.nan)
    for i in range(n_t):
        s[:,i] = so3.skew_elements(so3.log(R[:,:,i]))

    # interpolate t values using state x=(p,s,v,om)
    x = np.concatenate((p, s, pdot, om), axis=0)
    interp_fun = sp.interpolate.interp1d(t, x)
    x_out = interp_fun(t_out)

    # get x values
    p_out = x_out[:3,:]
    s_out = x_out[3:6,:]
    pdot_out = x_out[6:9,:]
    om_out = x_out[9:,:]

    # convert tangent space s back into rotations
    R_out = np.full((3,3,n_t_out), np.nan)
    for i in range(n_t_out):
        R_out[:,:,i] = so3.exp(so3.cross(s_out[:,i]))

    return p_out, R_out, pdot_out, om_out
