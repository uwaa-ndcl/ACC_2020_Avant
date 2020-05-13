import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as pp

import net_filter.tools.so3 as so3
import net_filter.dynamics.rigid_body as rb
import net_filter.dynamics.angular_velocity as av

# set up
g = 9.8
n_t = 11
t = np.linspace(0, 1, n_t)
dt = t[1] - t[0]

# make arrays
xyz = np.full((3,n_t), np.nan)
v = np.full((3,n_t), np.nan)
R = np.full((3,3,n_t), np.nan)
quat = np.full((4,n_t), np.nan)
om = np.full((3,n_t), np.nan)

# initial values
# x: right, y: forward, z: up 
xyz0 = np.array([-.3, 1.5, -.3])
R0 = np.eye(3) 
quat0 = t3d.quaternions.mat2quat(R0)
v0 = np.array([.3, 0, 3.5])
omega0 = np.array([3, 2, -5])
qdot0 = av.om_to_qdot(omega0, quat0)

# so3
accel = np.array([0, 0, -g])
for i in range(n_t):
    # position
    v[:,i] = accel*t[i]
    xyz[:,i] = xyz0 + v0*t[i] + .5*accel*t[i]**2

    # rotations
    om[:,i] = om[:,i-1]
    R[:,:,i] = so3.exp(so3.cross(t[i] * omega0))
    quat[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

# newton-euler
x0 = np.block([xyz0, quat0, v0, qdot0])
x = rb.integrate(t, x0)

# convert to euler
euler_so3 = np.full((3,n_t), np.nan)
euler_ne = np.full((3,n_t), np.nan)
for i in range(n_t):
    euler_so3[:,i] = t3d.euler.mat2euler(R[:,:,i], 'rxyz')
    euler_ne[:,i] = t3d.euler.quat2euler(x[3:7,i], 'rxyz')

# plot
#print(t)
#print(euler_so3)
#print(euler_ne)
pp.figure()
pp.plot(t, euler_so3[0,:], 'k')
pp.plot(t, euler_so3[1,:], 'r')
pp.plot(t, euler_so3[2,:], 'b')
pp.plot(t, euler_ne[0,:], 'ko')
pp.plot(t, euler_ne[1,:], 'ro')
pp.plot(t, euler_ne[2,:], 'bo')
pp.show()
