import numpy as np


def cross(v):
    '''
    cross product matrix, for vectors v and w, v x w = cross(v) @ w
    '''

    mat = np.array([[    0, -v[2],  v[1]],
                    [ v[2],     0, -v[0]],
                    [-v[1],  v[0],    0]])

    return mat


def skew_elements(A):
    '''
    get the three elements of a skew-symmetric matrix
    '''

    x = np.array([A[2,1], A[0,2], A[1,0]])

    return x


def exp(X):
    '''
    use the Rodrigues formula to calculate the matrix exponential of a
    skew-symmetric matrix
    '''

    x = skew_elements(X) # vector of "angular velocities"
    theta = np.sqrt(x @ x)
    # TO DO: do this better for small angles
    if np.absolute(theta) < 1e-6:
        exp = np.eye(3) + 1*X + .5*(X@X)
    else:
        exp = np.eye(3) + (np.sin(theta)/theta)*X \
                        + ((1 - np.cos(theta))/theta**2)*(X@X)

    return exp


def log(R):
    '''
    matrix logarithm of a rotation matrix
    '''
    
    theta = np.arccos((np.trace(R) - 1)/2)
    if theta == 0.0:
        log = 0.5*(R - R.T)
    else:
        log = (theta/(2*np.sin(theta)))*(R - R.T)

    return log


def geodesic_distance(R1, R2):
    '''
    geodesic distance between two rotation matrices
    '''

    dist = (1/np.sqrt(2)) * np.linalg.norm(log(R1.T @ R2), ord='fro')

    return dist


def random_rotation_matrix():
    '''
    generate a random rotation matrix
    from Graphics Gems III
    '''

    (x1, x2, x3) = np.random.uniform(0, 1, size=3)
    theta = 2*np.pi*x1
    phi = 2*np.pi*x2
    z = x3
    V = np.array([np.cos(phi)*np.sqrt(z), np.sin(phi)*np.sqrt(z), np.sqrt(1-z)])
    R_z = np.array([[ np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [             0,             0, 1]])
    M = np.matmul(2*np.outer(V,V) - np.eye(3), R_z)

    return M
