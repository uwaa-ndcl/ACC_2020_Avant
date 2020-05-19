import os
import pickle
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.blender.render as br
import net_filter.dope.dope_to_blender as db
import net_filter.tools.so3 as so3
import net_filter.sim.one_dim_gen_plots as po

# upright can
R_upright = t3d.euler.euler2mat(-np.pi/2, 0, -np.pi/2, 'sxyz')

# parameters
n_ims = 30
rot_eps = 1e-6 # epsilon to add for generating angles -pi to pi
lighting_energy = 100.0


def gen_and_eval(mode):

    # to be filled
    p = np.full((3,n_ims), np.nan)
    R = np.full((3,3,n_ims), np.nan)

    if mode=='trans_x':
        x = np.linspace(-.2, .2, n_ims)
        for i in range(n_ims):
            R[:,:,i] = R_upright
            p[:,i] = np.array([x[i], .8, 0])
        img_dir = dirs.trans_x_dir

    elif mode=='trans_y':
        y = np.linspace(.4, 1.8, n_ims)
        for i in range(n_ims):
            R[:,:,i] = R_upright
            p[:,i] = np.array([0, y[i], 0])
        img_dir = dirs.trans_y_dir

    elif mode=='trans_z':
        z = np.linspace(-.15, .15, n_ims)
        for i in range(n_ims):
            R[:,:,i] = R_upright
            p[:,i] = np.array([0, .8, z[i]])
        img_dir = dirs.trans_z_dir

    elif mode=='rot_x':
        # rotate eps is needed to avoided wrapping of the exp/log maps
        ang = np.linspace(-np.pi+rot_eps, np.pi-rot_eps, n_ims)
        for i in range(n_ims):
            s_i = np.array([ang[i], 0, 0])
            R[:,:,i] = so3.exp(so3.cross(s_i)) @ R_upright
            p[:,i] = np.array([0, .8, 0])
        img_dir = dirs.rot_x_dir

    elif mode=='rot_y':
        ang = np.linspace(-np.pi+rot_eps, np.pi-rot_eps, n_ims)
        for i in range(n_ims):
            s_i = np.array([0, ang[i], 0])
            R[:,:,i] = so3.exp(so3.cross(s_i)) @ R_upright
            p[:,i] = np.array([0, .8, 0])
        img_dir = dirs.rot_y_dir

    elif mode=='rot_z':
        ang = np.linspace(-np.pi+rot_eps, np.pi-rot_eps, n_ims)
        for i in range(n_ims):
            s_i = np.array([0, 0, ang[i]])
            R[:,:,i] = so3.exp(so3.cross(s_i)) @ R_upright
            p[:,i] = np.array([0, .8, 0])
        img_dir = dirs.rot_z_dir

    # convert rotation matrices to quaternions
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])

    # render
    to_render_pkl = os.path.join(img_dir, 'to_render.pkl')
    render_props = br.RenderProperties()
    render_props.n_renders = n_ims
    render_props.model_name = 'soup_can'
    render_props.pos = p
    render_props.quat = q
    render_props.lighting_energy = lighting_energy
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    br.blender_render(img_dir)

    # predict
    p, R, p_est, R_est = db.get_predictions(img_dir, print_errors=False)

# regenerate and re-evaluate images?
regen = 0
if regen:
    # translations
    gen_and_eval('trans_x')
    gen_and_eval('trans_y')
    gen_and_eval('trans_z')

    # rotations
    gen_and_eval('rot_x')
    gen_and_eval('rot_y')
    gen_and_eval('rot_z')

# plots
po.plot_translation_and_rotation()
