import os
import pickle
import numpy as np
import scipy as sp
import scipy.interpolate
import transforms3d as t3d

import net_filter.blender.render as br
import net_filter.dynamics.rigid_body as rb
import net_filter.dynamics.angular_velocity as av
from net_filter.blender.render_properties import RenderProperties

import transforms3d as t3d
'''
###############################################################################
# test negating an axis
a, b, c = np.random.uniform(0, np.pi, 3)
R1 = t3d.euler.euler2mat(a, b, c, 'sxyz')
R2 = t3d.euler.euler2mat(a, b, -c, 'sxyz')
print(R1)
print('---')
print(R2)
'''
'''
###############################################################################
# test a single

# camera properties
f = 50 # focal length
pix_width = 640
pix_height = 480
sensor_width = 36 # in (mm)
sensor_height = 36*(pix_height/pix_width)
aov_w = 2*np.arctan((sensor_width/2)/f) # angle of view
aov_h = 2*np.arctan((sensor_height/2)/f) 
f_pix = pix_width*(f/sensor_width) # focal length in pixels

# Blender | x: right, y: forward, z: up
# DOPE    | x: right, y: down,    z: forward
# rxyz: body x  > body y  > body z
# sxyz: world x > world y > world z
seq = 'szyx'
xyz = np.array([-1, .0, .0])
q = t3d.euler.euler2quat(.1*np.pi, .3*np.pi, .7*np.pi, seq)
R = t3d.quaternions.quat2mat(q)
euler_R = t3d.euler.mat2euler(R, seq) 
euler_dope = np.array([euler_R[0], euler_R[1], -euler_R[2]])
R_dope = t3d.euler.euler2mat(*euler_dope, seq)

# render
save_dir = '/home/trevor/Downloads/test_dir/'
to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
render_props = RenderProperties()
render_props.n_renders = 1
render_props.model_name = 'soup_can'
render_props.xyz = xyz[:,np.newaxis]
render_props.quat = q[:,np.newaxis]
render_props.pix_width = pix_width
render_props.pix_height = pix_height
render_props.alpha = False
render_props.compute_gramian = False
render_props.cam_quat = t3d.euler.euler2quat(0, np.pi/2, 0, seq)
with open(to_render_pkl, 'wb') as output:
    pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
br.blender_render(save_dir)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#R_dope = R_dope.T
#R_dope[0,1] *= -1; R_dope[0,2] *= -1; R_dope[1,0] *= -1; R_dope[2,0] *= -1;
print('R:\n', R)
print('R_dope: \n', R_dope)
'''
"""
###############################################################################
# test a bunch in the default camera frame

import neural_network_filtering.tools.so3 as so3

# camera properties
f = 50 # focal length
pix_width = 640
pix_height = 480
sensor_width = 36 # in (mm)
sensor_height = 36*(pix_height/pix_width)
aov_w = 2*np.arctan((sensor_width/2)/f) # angle of view
aov_h = 2*np.arctan((sensor_height/2)/f) 
f_pix = pix_width*(f/sensor_width) # focal length in pixels

# Blender | x: right, y: forward, z: up
# DOPE    | x: right, y: down,    z: forward
# rxyz: body x  > body y  > body z
# sxyz: world x > world y > world z
n = 10
seq = 'szyx'

R = np.full((3,3,n), np.nan)
xyz = np.full((3,n), np.nan)
q = np.full((4,n), np.nan)
for i in range(n):
    R[:,:,i] = so3.random_rotation_matrix()
    q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
    '''
    # default camera
    xyz[:,i] = np.array([-1, np.random.uniform(-.2, .2),
                         np.random.uniform(-.2, .2)]) # default camera
    R_cam = t3d.euler.euler2mat(0, np.pi/2, 0, 'sxyz')
    q_cam = t3d.quaternions.mat2quat(R_cam)
    '''
    # my camera
    xyz[:,i] = np.array([np.random.uniform(-.2, .2), 1.4,
                         np.random.uniform(-.2, .2)]) # default camera
    R_cam = t3d.euler.euler2mat(np.pi/2, 0, 0, 'sxyz')
    q_cam = t3d.quaternions.mat2quat(R_cam)


# render
save_dir = '/home/trevor/Downloads/test_dir/'
to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
render_props = RenderProperties()
render_props.n_renders = n
render_props.model_name = 'soup_can'
render_props.xyz = xyz
render_props.quat = q
render_props.pix_width = pix_width
render_props.pix_height = pix_height
render_props.alpha = False
render_props.compute_gramian = False
render_props.cam_quat = q_cam
with open(to_render_pkl, 'wb') as output:
    pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
br.blender_render(save_dir)
"""
