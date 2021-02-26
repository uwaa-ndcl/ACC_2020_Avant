import numpy as np
import net_filter.tools.so3 as so3

# create a bunch of random roation matrices
n_test = 500
R = np.full((3,3,n_test), np.nan)
for i in range(n_test):
    R[:,:,i] = so3.random_rotation_matrix()

# check the geodesic distance between the rotiation matrices
for i in range(n_test):
    R_i = R[:,:,i]
    for j in range(n_test):
        R_j = R[:,:,j]
        dist = so3.geodesic_distance(R_i, R_j)

        if np.isnan(dist):
            print('NAN ERROR!!!')

        if np.all(R_i==R_j) and dist!=0:
            print('DISTANCE SHOULD BE ZERO!!!')
