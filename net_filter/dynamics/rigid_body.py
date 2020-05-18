import numpy as np
import scipy as sp
from scipy import integrate, interpolate
import transforms3d as t3d

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

def cross(v):
    '''
    cross product matrix, for vectors v and w, v x w = cross(v) @ w
    '''
    mat = np.array([[    0, -v[2],  v[1]],
                    [ v[2],     0, -v[0]],
                    [-v[1],  v[0],    0]])

    return mat


def om_to_qdot(om, q):
    '''
    convert angular velocity om (expressed in body coordinates)
    to time derivative of quaternion q_dot based on q
    see eq. 3.104 (p. 110) Analytic Mechanics of Space Systems, 2nd ed.
        note: omega is expressed in B frame (see eq. 3.106)
    '''
    om_mat = np.array([[    0, -om[0], -om[1], -om[2]],
                       [om[0],      0,  om[2], -om[1]],
                       [om[1], -om[2],      0,  om[0]],
                       [om[2],  om[1], -om[0],      0]])
    qdot = .5 * om_mat @ q

    return qdot.ravel()


def qdot_to_om(q, qdot):
    '''
    convert time derivative of quaternion to angular velocity omega (in body
    frame)
    see 3.105 (p. 110) in Schaub & Junkins, and note that the inverse of the
    4x4 matrix is equal to applying that matrix for to the inverse of beta
    (this can be verfied numerically)
    '''
    om = 2 * t3d.quaternions.qmult(t3d.quaternions.qinverse(q), qdot)
    om = om[1:]

    return om


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
    omdot_B = np.linalg.inv(J) @ (tau_B - cross(om_B) @ J @ om_col_B)
    pdot_omdot = np.concatenate((pdot_dot_A, omdot_B)).ravel()

    return pdot_omdot


def integrate_kinematics(pdot, om, p0, q0, dt):
    '''
    forward Euler integration of angular velocities pdot and omega,
    to x and quaternions

    inputs:
        pdot, om: translational and angular velocities
              pdot expressed in A frame, om expressed in B frame
              size (3, # of time points)
        p_q0: initial p and quaternion, size (7)
        dt: timestep at which values are spaced
    '''

    n_pts = pdot.shape[1]

    # define arrays
    p = np.full((3, n_pts), np.nan)
    q = np.full((4, n_pts), np.nan)
    qdot = np.copy(q) # time derivative of quaternion

    # 1st order forward Euler integration
    p[:,0] = p0
    q[:,0] = q0
    qdot[:,0] = om_to_qdot(om[:,0], q[:,[0]]).ravel()
    for i in range(1, n_pts):
        p[:,i] = p[:,i-1] + dt*pdot[:,i-1]
        q[:,i] = q[:,i-1] + dt*qdot[:,i-1]
        q[:,i] = q[:,i]/np.linalg.norm(q[:,i]) # normalize quaternion

        # time derivative of quaternion (remember omega is in B coordinates!)
        qdot[:,i] = om_to_qdot(om[:,i], q[:,[i]]).ravel()
    p_q = np.concatenate((p, q)) 

    return p, q, qdot


def integrate(t, p0, R0, pdot0, om0):
    '''
    integrate Newton-Euler equations, then integrate kinematics (t=times)
    '''

    # setup
    t0 = t[0]
    tf = t[-1]
    n_t = t.size

    # convert rotation matrix to quaternions
    q0 = t3d.quaternions.mat2quat(R0)
    qdot0 = om_to_qdot(om0, q0)
    pdot_om0 = np.concatenate((pdot0, om0))

    # integration times
    dt_int = .001
    t_int = np.arange(t0, tf+dt_int, dt_int) # make sure to include last point
    tf_int = t_int[-1]

    # integrate newton-euler (state is pdot & om)
    pdot_om0_col = pdot_om0[:,np.newaxis]
    sol = sp.integrate.solve_ivp(newton_euler, (t0, tf_int), pdot_om0,
                                 method='RK45', t_eval=t_int)
    pdot_om_int = sol.y
    pdot_int = pdot_om_int[:3,:]
    om_int = pdot_om_int[3:,:]
    
    # integrate kinematics
    p_int, q_int, qdot_int = integrate_kinematics(
            pdot_int, om_int, p0, q0, dt_int) 

    # interpolate t values using state x=(p,q,v,qdot)
    x_int = np.concatenate((p_int, q_int, pdot_int, qdot_int), axis=0)
    interp_fun = sp.interpolate.interp1d(t_int, x_int)
    x = interp_fun(t)

    # convert quaterions to rotation matrix
    p = x[:3,:]
    q = x[3:7,:]
    pdot = x[7:10,:]
    qdot = x[10:,:]
    R = np.full((3,3,n_t), np.nan)
    om = np.full((3,n_t), np.nan)

    for i in range(n_t):
        R[:,:,i] = t3d.quaternions.quat2mat(q[:,i])
        om[:,i] = qdot_to_om(q[:,i], qdot[:,i]) 

    return p, R, pdot, om
