import os
import cv2
import glob
import pickle
import numpy as np
from PIL import Image, ImageDraw

import net_filter.directories as dirs
import net_filter.tools.image as ti
import net_filter.tools.unit_conversion as conv
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
f_p = (pix_width/2)/np.tan(render_props.angle_w/2) # focal length, units of pix

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

def gen_boxes(p, R, clr, save_dir):
    '''
    overlay bounding boxes on rendered images
    '''

    # get rendered image files
    png_file = os.path.join(dirs.simulation_dir, '*.png')
    im_files = sorted(glob.glob(png_file))
    tails = [os.path.split(file)[1] for file in im_files]
    new_im_files = [os.path.join(save_dir, tail) for tail in tails]

    # loop over images
    n_renders = p.shape[1]
    for i in range(n_renders):
        # translate and rotate bounding box
        p_i = p[:,i]
        R_i = R[:,:,i]
        p_mat = np.repeat(p_i[:,np.newaxis], 8, axis=1)
        box_i = np.copy(box_def)
        box_i = R_i @ box_i
        box_i += p_mat

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
        im = Image.fromarray(np.uint8(im_i*255))
        draw_ob = ImageDraw.Draw(im)

        # draw the cube
        dd.draw_cube(draw_ob, box, clr)

        # open cv image
        open_cv_image = np.array(im)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_im_files[i], open_cv_image)

# load filter and measurement data
npz_file = os.path.join(dirs.simulation_dir, 'filter_results.npz')
data = np.load(npz_file)
p = data['p']*conv.cm_to_m
p_meas = data['p_meas']*conv.cm_to_m
p_filt = data['p_filt']*conv.cm_to_m
R = data['R']
R_meas = data['R_meas']
R_filt = data['R_filt']

# make boxed images for neural net estimates
clr_net = (221, 0, 255)
net_dir = os.path.join(dirs.animation_dir, 'net/')
#gen_boxes(p, R, clr, net_dir) # sanity check: true values
gen_boxes(p_meas, R_meas, clr_net, net_dir)

# make boxed images for filter estimates
clr_filt = (0, 221, 255)
filt_dir = os.path.join(dirs.animation_dir, 'filter/')
gen_boxes(p_filt, R_filt, clr_filt, filt_dir)
