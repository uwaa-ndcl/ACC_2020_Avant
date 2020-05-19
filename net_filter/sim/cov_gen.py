import os
import pickle
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.blender.render as br
import net_filter.dope.eval as ev
import net_filter.dope.dope_to_blender as db

#mode = 'dim'
mode = 'bright'

if mode == 'dim':
    img_dir = dirs.cov_dim_dir
    light_energy = 6.0
elif mode == 'bright':
    img_dir = dirs.cov_bright_dir
    light_energy = 50.0

# camera properties (to determine angle of view to create random poses)
f = 50 # focal length
pix_width = 640
pix_height = 480
sensor_width = 36 # in (mm)
sensor_height = 36*(pix_height/pix_width)
aov_w = 2*np.arctan((sensor_width/2)/f) # angle of view
aov_h = 2*np.arctan((sensor_height/2)/f) 
f_pix = pix_width*(f/sensor_width) # focal length in pixels

# randomization
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

p = np.stack((x, y, z), axis=0)

# rotations
q = np.full((4, n_ims), np.nan) # to be filled
for i in range(n_ims):
    R_i = so3.random_rotation_matrix() # random orientations
    q[:,i]  = t3d.quaternions.mat2quat(R_i)

# render
world_RGB = np.full((3,n_ims), 0.0)
to_render_pkl = os.path.join(img_dir, 'to_render.pkl')
render_props = br.RenderProperties()
render_props.n_renders = n_ims
render_props.model_name = 'soup_can'
render_props.pos = p
render_props.quat = q
render_props.world_RGB = world_RGB 
render_props.lighting_energy = light_energy
with open(to_render_pkl, 'wb') as output:
    pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
br.blender_render(img_dir)

# predict
p, R, p_est, R_est = db.get_predictions(img_dir, print_errors=False)
