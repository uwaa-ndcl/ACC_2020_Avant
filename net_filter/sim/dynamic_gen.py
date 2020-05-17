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

# model name
name = 'soup_can'

# lighting
RGB_color = .0*np.array([1.0, 1.0, 1.0])
lighting_energy = 6.0

def generate_images(n_t, dt, p, R, v, om, save_dir):

    # convert rotation matrices to quaternions
    q = np.full((4,n_t), np.nan)
    for i in range(n_t):
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    render_props = br.RenderProperties()
    render_props.n_renders = n_t
    render_props.model_name = name
    render_props.pos = p
    render_props.quat = q
    #render_props.v = v # not used in rendering
    #render_props.om = om # not used in rendering
    render_props.world_RGB = np.repeat(RGB_color[:,np.newaxis], n_t, axis=1)
    render_props.lighting_energy = lighting_energy
    render_props.dt = dt
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    br.blender_render(save_dir)


def generate_snapshots(n_t, inds, p, R):
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
        render_props = br.RenderProperties()
        render_props.model_name = name
        render_props.image_names = [png_name_snapshot % ind_i]
        render_props.pos = p[:,[ind_i]]
        render_props.quat = q[:,[ind_i]]
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
