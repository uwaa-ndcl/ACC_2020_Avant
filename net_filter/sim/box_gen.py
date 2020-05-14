import os
import cv2
import glob
import pickle
import numpy as np
import transforms3d as t3d
from PIL import Image, ImageDraw

import net_filter.directories as dirs
import net_filter.tools.image as ti
import net_filter.tools.unit_conversion as conv
import net_filter.dope.eval as eval
import net_filter.dope.draw as dd

# model info
name = 'soup_can'
to_render_pkl = os.path.join(dirs.simulation_dir, 'to_render.pkl')
bbox_pkl = os.path.join(dirs.blender_models_dir, name + '_bounding_box.pkl')

# load render properties
with open(to_render_pkl, 'rb') as input:
    render_props = pickle.load(input)
pix_width = render_props.pix_width
pix_height = render_props.pix_height
# focal length in units of pixels
f_p = (pix_width/2)/np.tan(render_props.angle_w/2)

# load bounding box info
with open(bbox_pkl, 'rb') as file:
    bbox_data = pickle.load(file)

# define bounding box
x_min = bbox_data['x_min']
x_max = bbox_data['x_max']
y_min = bbox_data['y_min']
y_max = bbox_data['y_max']
z_min = bbox_data['z_min']
z_max = bbox_data['z_max']

# order given in Deep_Object_Pose/src/dope/inference/cuboid.py
box_def = np.array([[x_max, x_min, x_min, x_max, x_max, x_min, x_min, x_max],
                    [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max],
                    [z_max, z_max, z_min, z_min, z_max, z_max, z_min, z_min]])

def gen_boxes(xyz, q, clr, new_dir):
    '''
    overlay bounding boxes on rendered images
    '''

    # get rendered image files
    png_file = os.path.join(dirs.simulation_dir, '*.png')
    im_files = sorted(glob.glob(png_file))
    tails = [os.path.split(file)[1] for file in im_files]
    new_im_files = [os.path.join(new_dir, tail) for tail in tails]

    # loop over images
    n_renders = xyz.shape[1]
    for i in range(n_renders):
        # translate and rotate bounding box
        xyz_i = xyz[:,i]
        q_i = q[:,i]
        R_i = t3d.quaternions.quat2mat(q_i)
        xyz_mat = np.repeat(xyz_i[:,np.newaxis], 8, axis=1)
        box_i = np.copy(box_def)
        box_i = R_i @ box_i
        box_i += xyz_mat

        # convert bounding box x,y,z coordinates to pixels coordinates
        cx = pix_width//2
        cy = pix_height//2
        box = list()
        for j in range(8):
            x, y, z = box_i[0,j], box_i[1,j], box_i[2,j]
            px = cx + (x/y)*f_p
            pz = cy + (-z/y)*f_p
            tup_j = (px, pz)
            box.append(tup_j)

        # make new image file
        im_file_i = im_files[i]
        tail_i = os.path.split(im_file_i)[1]
        im_i = ti.load_im_np(im_file_i)
        #im = Image.fromarray(im0)
        im = Image.fromarray(np.uint8(im_i*255))
        draw_ob = ImageDraw.Draw(im)

        # draw the cube
        dd.draw_cube(draw_ob, box, clr)

        # open cv image
        open_cv_image = np.array(im)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_im_files[i], open_cv_image)

'''
name = 'original'
xyz = render_props.xyz
q = render_props.quat
clr = (0, 255, 0)
new_dir = os.path.join(dirs.simulation_dir, 'original/')
gen_boxes(xyz, q, clr, new_dir)
'''

# load data
npz_file = os.path.join(dirs.simulation_dir, 'filter_results.npz')
data = np.load(npz_file)
xyz = data['xyz']*conv.cm_to_m
xyz_meas = data['xyz_meas']*conv.cm_to_m
xyz_hat = data['xyz_hat']*conv.cm_to_m
R = data['R']
R_meas = data['R_meas']
R_hat = data['R_hat']
n_renders = xyz_meas.shape[1]

# convert R to quat
q = np.full((4, n_renders), np.nan)
q_meas = np.full((4, n_renders), np.nan)
q_hat = np.full((4, n_renders), np.nan)
for i in range(n_renders):
    q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
    q_meas[:,i] = t3d.quaternions.mat2quat(R_meas[:,:,i])
    q_hat[:,i] = t3d.quaternions.mat2quat(R_hat[:,:,i])

# make boxed images for neural net estimates
clr = (221, 0, 255)
new_dir = os.path.join(dirs.animation_dir, 'net/')
#gen_boxes(xyz, q, clr, new_dir) # sanity check: true values
gen_boxes(xyz_meas, q_meas, clr, new_dir)

# make boxed images for filter estimates
clr = (0, 221, 255)
new_dir = os.path.join(dirs.animation_dir, 'filter/')
gen_boxes(xyz_hat, q_hat, clr, new_dir)
