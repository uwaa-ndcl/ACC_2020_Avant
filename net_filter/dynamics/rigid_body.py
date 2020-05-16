import numpy as np
import scipy as sp
from scipy import integrate, interpolate
import transforms3d as t3d

import net_filter.dynamics.angular_velocity as av

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


def newton_euler(t, v_om):
    '''
    Newton-Euler equations for a rigid body
    assumes velocity (v_A) is in inertial frame (A) coordinates,
    and omega (om_B) is in body-frame (B) coordinates
    ref: eq. 4.16 (p. 167) in A Mathematical Introduction to Robotic
    Manipulation
    '''

    v_A = v_om[:3]
    om_B = v_om[3:]
    om_col_B = om_B[:,np.newaxis] # omega as column vector
    
    # translational dynamics: all expressed in inertial frame A
    F_A = np.array([[0, 0, -m*g]]).T # inertial frame force of gravity
    v_dot_A = np.linalg.inv(m*np.eye(3)) @ F_A

    # rotational dynamics: all expressed in body frame B
    tau_B = np.array([[0, 0, 0]]).T     # inertial frame torque
    om_dot_B = np.linalg.inv(J) @ (tau_B - cross(om_B) @ J @ om_col_B)
    v_om_dot = np.concatenate((v_dot_A, om_dot_B)).ravel()

    return v_om_dot


def integrate_kinematics(v, om, xyz0, q0, dt):
    '''
    forward Euler integration of angular velocities v and omega,
    to x and quaternions

    inputs:
        v_om: translational and angular velocities
              v expressed in A frame, omega expressed in B frame
              size (6, # of time points)
        xyz_q0: initial xyz and quaternion, size (7)
        dt: timestep at which values are spaced
    '''

    n_pts = v.shape[1]

    # define arrays
    xyz = np.full((3, n_pts), np.nan)
    q = np.full((4, n_pts), np.nan)
    q_dot = np.copy(q) # time derivative of quaternion

    # 1st order forward Euler integration
    xyz[:,0] = xyz0
    q[:,0] = q0
    q_dot[:,0] = av.om_to_qdot(om[:,0], q[:,[0]]).ravel()
    for i in range(1, n_pts):
        xyz[:,i] = xyz[:,i-1] + dt*v[:,i-1]
        q[:,i] = q[:,i-1] + dt*q_dot[:,i-1]
        q[:,i] = q[:,i]/np.linalg.norm(q[:,i]) # normalize quaternion

        # time derivative of quaternion (remember omega is in B coordinates!)
        q_dot[:,i] = av.om_to_qdot(om[:,i], q[:,[i]]).ravel()
    xyz_q = np.concatenate((xyz, q)) 

    return xyz, q, q_dot


def integrate(t, xyz0, R0, v0, om0):
    '''
    integrate Newton-Euler equations, then integrate kinematics (t=times)
    '''

    # setup
    t0 = t[0]
    tf = t[-1]
    n_t = t.size

    # convert rotation matrix to quaternions
    q0 = t3d.quaternions.mat2quat(R0)
    qdot0 = av.om_to_qdot(om0, q0)
    v_om0 = np.concatenate((v0, om0))

    # integration times
    dt_int = .001
    t_int = np.arange(t0, tf+dt_int, dt_int) # make sure to include last point
    tf_int = t_int[-1]

    # integrate newton-euler (state is v & om)
    v_om0_col = v_om0[:,np.newaxis]
    sol = sp.integrate.solve_ivp(newton_euler, (t0, tf_int), v_om0,
                                 method='RK45', t_eval=t_int)
    v_om_int = sol.y
    v_int = v_om_int[:3,:]
    om_int = v_om_int[3:,:]
    
    # integrate kinematics
    xyz_int, q_int, qdot_int = integrate_kinematics(
            v_int, om_int, xyz0, q0, dt_int) 

    # interpolate t values using state x=(xyz,q,v,qdot)
    x_int = np.concatenate((xyz_int, q_int, v_int, qdot_int), axis=0)
    interp_fun = sp.interpolate.interp1d(t_int, x_int)
    x = interp_fun(t)

    # convert quaterions to rotation matrix
    xyz = x[:3,:]
    q = x[3:7,:]
    v = x[7:10,:]
    qdot = x[10:,:]
    R = np.full((3,3,n_t), np.nan)
    om = np.full((3,n_t), np.nan)

    for i in range(n_t):
        R[:,:,i] = t3d.quaternions.quat2mat(q[:,i])
        om[:,i] = av.qdot_to_om(q[:,i], qdot[:,i]) 

    return xyz, R, v, om
