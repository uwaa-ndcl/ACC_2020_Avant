import os
import pickle
import pkgutil
import subprocess
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.tools.image as ti

class RenderProperties:

    def __init__(self):
        # name of .blend file
        self.model_name = None

        # list of names of images, if none then images will be named
        # 000000.png, 000001.png, ...
        self.image_names = None

        # directory to save output images, etc.
        self.save_dir = None # this will be filled later

        # object
        self.ob = None # this will be set inside of Blender
        self.n_renders = 1
        self.pos = np.array([[0],[0],[0]]) # size (3, n_renders)
        self.quat = np.array([[1],[0],[0],[0]]) # size (4, n_renders)

        # world lighting, size (3, n_renders)
        self.world_RGB = None  

        # lighting energy (sometimes called power in Blender) of all lights
        self.lighting_energy = None

        # transparent background?
        self.alpha = False 

        # camera
        self.cam_ob = None # this will be set inside of Blender
        self.cam_pos = [0, 0, 0]
        self.cam_quat = t3d.euler.euler2quat(np.pi/2, 0, 0, axes='sxyz')
        self.pix_width = 640
        self.pix_height = 480
        self.sensor_fit = 'AUTO'
        self.angle_w = 2*np.arctan(18/50) # Blender default
        self.angle_h = 2*np.arctan(18/50)


def blender_render(render_dir):
    '''
    call a blender command which will generate renders in render_dir
    '''

    # get path to render script
    mod_name = 'net_filter.blender.process_renders'
    pkg = pkgutil.get_loader(mod_name)
    render_script = pkg.get_filename()

    # run blender command
    blender_cmd = 'blender --background --python-use-system-env --python ' \
                  + render_script + ' -- ' + render_dir
    subprocess.run([blender_cmd], shell=True)


def soup_gen(dt, p, R, save_dir,
                    lighting_energy=6.0, world_RGB=np.array([.0, .0, .0])):

    # convert rotation matrices to quaternions
    n_ims = p.shape[1]
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    render_props = RenderProperties()
    render_props.n_renders = n_ims
    render_props.model_name = 'soup_can'
    render_props.pos = p
    render_props.quat = q
    render_props.world_RGB = np.repeat(world_RGB[:,np.newaxis], n_ims, axis=1)
    render_props.lighting_energy = lighting_energy
    render_props.dt = dt
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    blender_render(save_dir)


def soup_snapshots(p, R, inds,
                   lighting_energy=6.0, world_RGB=np.array([.0, .0, .0])):

    # convert rotation matrices to quaternions
    n_ims = p.shape[1]
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

    # setup
    save_dir = dirs.snapshots_dir
    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    n_snapshot = 6 # number of snapshots for figure
    step_snapshot = int(np.floor(n_ims/n_snapshot))
    inds_snapshot = inds[::step_snapshot]
    if n_ims % n_snapshot != 0:
        inds_snapshot = inds_snapshot[:-1]
    png_name_snapshot = 'snapshot_%06d'

    # loop over snapshots
    for i in range(n_snapshot):
        ind_i = inds_snapshot[i]
        render_props = RenderProperties()
        render_props.model_name = 'soup_can'
        render_props.image_names = [png_name_snapshot % ind_i]
        render_props.pos = p[:,[ind_i]]
        render_props.quat = q[:,[ind_i]]
        render_props.world_RGB = np.repeat(RGB_color[:,np.newaxis], n_ims, axis=1)
        render_props.lighting_energy = lighting_energy

        # only make last snapshot have a background
        if i < n_snapshot-1:
            render_props.alpha = True
        else:
            render_props.alpha = False

        with open(to_render_pkl, 'wb') as output:
            pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
        blender_render(save_dir)

    # overlay snapshots
    im_file_0 = os.path.join(save_dir,
                             png_name_snapshot % inds_snapshot[-1] + '.png')
    im_snapshot = ti.load_im_np(im_file_0)
    for i in reversed(inds_snapshot[:-1]):
        im_file_i = os.path.join(save_dir, png_name_snapshot % i + '.png')
        im_overlay = ti.load_im_np(im_file_i)
        im_snapshot = ti.overlay(im_overlay, im_snapshot)
    ti.write_im_np(os.path.join(save_dir, 'snapshots.png'), im_snapshot)
