import numpy as np
import transforms3d as t3d

def om_to_eulerdot(omega, euler, axes='rxyz'):
    '''
    convert body-fixed angular velocity into time derivatives of Euler angles
    axes correspond to those defined in transforms3d
    reference: Appendix B in Schaub & Junkins
    note: 'rxyz' axes in transforms3d correspond to 1-2-3 Euler angles in
    Schaub & Junkins
    '''

    if axes == 'rxyz':
        s2 = np.sin(euler[1])
        c2 = np.cos(euler[1])
        s3 = np.sin(euler[2])
        c3 = np.cos(euler[2])
        mat = (1/c2) * np.array([[c3,       -s3,  0],
                                 [c2*s3,  c2*c3,  0],
                                 [-s2*c3, s2*s3, c2]])
        theta_dot = mat @ omega
       
        return theta_dot.ravel()


def eulerdot_to_om(euler, euler_dot, axes='rxyz'):
    '''
    convert euler angles to body-fixed angular velocity
    reference: Appendix B in Schaub & Junkins
    '''

    if axes == 'rxyz':
        s2 = np.sin(euler[1])
        c2 = np.cos(euler[1])
        s3 = np.sin(euler[2])
        c3 = np.cos(euler[2])

        mat = np.array([[   c2,   0, 1],
                        [s2*s3,  c3, 0],
                        [s2*c3, -s3, 0]])
        om = mat @ euler_dot[:,np.newaxis] 

        return om.ravel()


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


def qdot_to_om(q, q_dot):
    '''
    convert time derivative of quaternion to angular velocity omega (in body
    frame)
    see 3.105 (p. 110) in Schaub & Junkins, and note that the inverse of the
    4x4 matrix is equal to applying that matrix for to the inverse of beta
    (this can be verfied numerically)
    '''
    om = 2 * t3d.quaternions.qmult(t3d.quaternions.qinverse(q), q_dot)
    om = om[1:]

    return om
