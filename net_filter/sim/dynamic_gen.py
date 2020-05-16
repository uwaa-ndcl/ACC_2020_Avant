import os
import pickle
import numpy as np
import scipy as sp
import scipy.interpolate
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.blender.render as br
import net_filter.tools.image as ti
from net_filter.blender.render_properties import RenderProperties

# model name
name = 'soup_can'

# camera properties
# NOTE: The my_config.yaml file has 6 parameters which correspond as follows to
# the variables below. If these variables are to be changed, the yaml file must
# be manually changed as well.
#
# yaml file | this file
# width       pix width
# height      pix_height
# fx          f_pix
# fy          f_pix
# cx          pix_width/2
# cy          pix_height/2
#
f = 50 # focal length
pix_width = 640
pix_height = 480
sensor_width = 36 # in (mm)
sensor_height = 36*(pix_height/pix_width)
aov_w = 2*np.arctan((sensor_width/2)/f) # angle of view
aov_h = 2*np.arctan((sensor_height/2)/f) 
f_pix = pix_width*(f/sensor_width) # focal length in pixels

# lighting
RGB_color = .0*np.array([1.0, 1.0, 1.0])
lighting_energy = 6.0

def generate_images(n_t, dt, xyz, R, v, om, save_dir):

    # convert rotation matrices to quaternions
    q = np.full((4,n_t), np.nan)
    for i in range(n_t):
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    render_props = RenderProperties()
    render_props.n_renders = n_t
    render_props.model_name = name
    render_props.xyz = xyz
    render_props.quat = q
    render_props.v = v # not used in rendering
    render_props.om = om # not used in rendering
    render_props.pix_width = pix_width
    render_props.pix_height = pix_height
    render_props.alpha = False
    render_props.compute_gramian = False
    render_props.world_RGB = np.repeat(RGB_color[:,np.newaxis], n_t, axis=1)
    render_props.lighting_energy = lighting_energy
    render_props.dt = dt
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    br.blender_render(save_dir)


def generate_snapshots(n_t, inds, xyz, R):
    # convert rotation matrices to quaternions
    q = np.full((4,n_t), np.nan)
    for i in range(n_t):
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

    # setup
    save_dir = dirs.snapshots_dir
    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    n_snapshot = 6 # number of snapshots for figure
    step_snapshot = int(np.floor(n_t/n_snapshot))
    inds_snapshot = inds[::step_snapshot]
    if n_t % n_snapshot != 0:
        inds_snapshot = inds_snapshot[:-1]
    png_name_snapshot = 'snapshot_%06d'

    # loop over snapshots
    for i in range(n_snapshot):
        ind_i = inds_snapshot[i]
        render_props = RenderProperties()
        render_props.model_name = name
        render_props.image_names = [png_name_snapshot % ind_i]
        render_props.xyz = xyz[:,[ind_i]]
        render_props.quat = q[:,[ind_i]]
        render_props.pix_width = pix_width
        render_props.pix_height = pix_height
        render_props.world_RGB = np.repeat(RGB_color[:,np.newaxis], n_t, axis=1)
        render_props.lighting_energy = lighting_energy

        # only make last snapshot have a background
        if i < n_snapshot-1:
            render_props.alpha = True
        else:
            render_props.alpha = False

        with open(to_render_pkl, 'wb') as output:
            pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
        br.blender_render(save_dir)

    # overlay snapshots
    im_file_0 = os.path.join(save_dir,
                             png_name_snapshot % inds_snapshot[-1] + '.png')
    im_snapshot = ti.load_im_np(im_file_0)
    for i in reversed(inds_snapshot[:-1]):
        im_file_i = os.path.join(save_dir, png_name_snapshot % i + '.png')
        im_overlay = ti.load_im_np(im_file_i)
        im_snapshot = ti.overlay(im_overlay, im_snapshot)
    ti.write_im_np(os.path.join(save_dir, 'snapshots.png'), im_snapshot)
