import os
import pickle
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as pp

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.blender.render as br
from net_filter.blender.render_properties import RenderProperties
import net_filter.dope.eval as ev
import net_filter.sim.dope_to_blender as db

#mode = 'dim'
mode = 'bright'

if mode == 'dim':
    img_dir = dirs.cov_dim_dir
elif mode == 'bright':
    img_dir = dirs.cov_bright_dir

# camera properties
f = 50 # focal length
pix_width = 640
pix_height = 480
sensor_width = 36 # in (mm)
sensor_height = 36*(pix_height/pix_width)
aov_w = 2*np.arctan((sensor_width/2)/f) # angle of view
aov_h = 2*np.arctan((sensor_height/2)/f) 
f_pix = pix_width*(f/sensor_width) # focal length in pixels

# default camera
#R_cam = t3d.euler.euler2mat(0, np.pi/2, 0, 'sxyz')
R_cam = t3d.euler.euler2mat(np.pi/2, 0, 0, 'sxyz')
q_cam = t3d.quaternions.mat2quat(R_cam)


def generate_images(xyz, q, world_RGB, light_energy):
    '''
    generate images
    '''
    n_ims = xyz.shape[1]

    # render
    to_render_pkl = os.path.join(img_dir, 'to_render.pkl')
    render_props = RenderProperties()
    render_props.n_renders = n_ims
    render_props.model_name = 'soup_can'
    render_props.xyz = xyz
    render_props.quat = q
    render_props.pix_width = pix_width
    render_props.pix_height = pix_height
    render_props.alpha = False
    render_props.compute_gramian = False
    render_props.cam_quat = q_cam
    render_props.world_RGB = world_RGB 
    render_props.lighting_energy = light_energy
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    br.blender_render(img_dir)


###############################################################################
# monte carlo
np.random.seed(35)
n_ims = 1000

# min and max distance from the camera
y_min = .5
y_max = 1.2

# random x and z values within field of view
x = np.full(n_ims, np.nan)
z = np.full(n_ims, np.nan)
y = np.random.uniform(y_min, y_max, size=n_ims)
for i in range(n_ims):
    x_max = np.tan(aov_w/2)*y[i]
    x_max *= .8 # shrink it so the full object will be visible in the image
    z_max = np.tan(aov_h/2)*y[i]
    z_max *= .8 # shrink it so the full object will be visible in the image
    x[i] = x_max*np.random.uniform(-1, 1)
    z[i] = z_max*np.random.uniform(-1, 1)
    
    # debug
    #x[i] = x_max
    #z[i] = z_max
    #x[i] = 0
    #z[i] = 0

xyz = np.stack((x, y, z), axis=0)

# rotations
q = np.full((4, n_ims), np.nan) # to be filled
for i in range(0, n_ims):
    R_i = so3.random_rotation_matrix() # random orientations
    #R_i = tm.R_z(np.random.uniform(0, 2*np.pi)) # random yaw only
    q[:,i]  = t3d.quaternions.mat2quat(R_i)

# different environmental cases
if mode == 'dim':
    light_energy = 6.0
elif mode == 'bright':
    light_energy = 50.0
#elif case == 3:
    #light_energy = 0.2

# generate
world_RGB = np.full((3,n_ims), 0.0)
generate_images(xyz, q, world_RGB, light_energy)

# predict
xyz, q, xyz_est, q_est = db.get_predictions(img_dir, print_errors=False)
print('bias: ', xyz[1,:] - xyz_est[1,:])
print('average bias: ', np.mean(xyz[1,:] - xyz_est[1,:]))
