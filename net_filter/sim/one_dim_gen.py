import os
import pickle
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.blender.render as br
from net_filter.blender.render_properties import RenderProperties
import net_filter.dope.eval as ev
import net_filter.sim.dope_to_blender as db
import net_filter.tools.image as ti
import net_filter.tools.so3 as so3
import net_filter.sim.one_dim_gen_plots as po
import matplotlib.pyplot as pp

# directories
img_dir_trans_x = dirs.trans_x_dir
img_dir_trans_y = dirs.trans_y_dir
img_dir_trans_z = dirs.trans_z_dir
img_dir_rot_x = dirs.rot_x_dir
img_dir_rot_y = dirs.rot_y_dir
img_dir_rot_z = dirs.rot_z_dir

# conversion    
rotate_eps = 1e-6 # epsilon to add for generating angles -pi to pi

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

# upright can
R_upright = t3d.euler.euler2mat(-np.pi/2, 0, -np.pi/2, 'sxyz')


def generate_images(xyz, q, img_dir):
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
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    br.blender_render(img_dir)


def translate_x():
    n_ims = 30
    x = np.linspace(-.2, .2, n_ims)

    xyz = np.full((3,n_ims), np.nan)
    R = np.full((3,3,n_ims), np.nan)
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        R[:,:,i] = R_upright
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
        xyz[:,i] = np.array([x[i], .8, 0])

    # generate
    generate_images(xyz, q, img_dir_trans_x)

    # predict
    xyz, q, xyz_est, q_est = db.get_predictions(img_dir_trans_x, print_errors=False)
    print('bias :', xyz[0,:] - xyz_est[0,:])
    print('average bias: ', np.mean(xyz[0,:] - xyz_est[0,:]))

    # plot
    pp.figure()
    pp.plot(xyz[0,:], xyz_est[0,:] - xyz[0,:], 'k-')
    pp.xlabel('x')
    pp.ylabel('x est - x')
    pp.grid()
    pp.show()


def translate_y():
    n_ims = 30
    y = np.linspace(.4, 1.8, n_ims)

    xyz = np.full((3,n_ims), np.nan)
    R = np.full((3,3,n_ims), np.nan)
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        R[:,:,i] = R_upright
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
        xyz[:,i] = np.array([0, y[i], 0])

    # generate
    generate_images(xyz, q, img_dir_trans_y)

    # predict
    xyz, q, xyz_est, q_est = db.get_predictions(img_dir_trans_y, print_errors=False)
    print('bias: ', xyz[1,:] - xyz_est[1,:])
    print('average bias: ', np.mean(xyz[1,:] - xyz_est[1,:]))

    # plot
    pp.figure()
    pp.plot(xyz[1,:], xyz_est[1,:] - xyz[1,:], 'k-')
    pp.xlabel('y')
    pp.ylabel('y est - y')
    pp.grid()
    pp.show()


def translate_z():
    n_ims = 30
    z = np.linspace(-.15, .15, n_ims)

    xyz = np.full((3,n_ims), np.nan)
    R = np.full((3,3,n_ims), np.nan)
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        R[:,:,i] = R_upright
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
        xyz[:,i] = np.array([0, .8, z[i]])

    # generate
    generate_images(xyz, q, img_dir_trans_z)

    # predict
    xyz, q, xyz_est, q_est = db.get_predictions(img_dir_trans_z, print_errors=False)
    print('bias: ', xyz[2,:] - xyz_est[2,:])
    print('average bias: ', np.mean(xyz[2,:] - xyz_est[2,:]))

    # plot
    pp.figure()
    pp.plot(xyz[2,:], xyz_est[2,:] - xyz[2,:], 'k-')
    pp.xlabel('z')
    pp.ylabel('z est - z')
    pp.grid()
    pp.show()


def rotate_x():
    n_ims = 30
    # rotate eps is needed to avoided wrapping of the exp/log maps
    ang = np.linspace(-np.pi+rotate_eps, np.pi-rotate_eps, n_ims)

    xyz = np.full((3,n_ims), np.nan)
    R = np.full((3,3,n_ims), np.nan)
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        s_i = np.array([ang[i], 0, 0])
        R[:,:,i] = so3.exp(so3.cross(s_i)) @ R_upright
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
        xyz[:,i] = np.array([0, .8, 0])

    # generate
    generate_images(xyz, q, img_dir_rot_x)

    # predict
    xyz, q, xyz_est, q_est = db.get_predictions(img_dir_rot_x, print_errors=False)
    s_offset = np.full((3,n_ims), np.nan)
    for i in range(n_ims):
        R_est_i = t3d.quaternions.quat2mat(q_est[:,i])
        R_offset_i = R[:,:,i].T @ R_est_i
        s_offset[:,i] = so3.skew_elements(so3.log(R_offset_i))

    print('bias: ', s_offset[0,:])
    print('average bias: ', np.mean(s_offset[0,:]))

    # plot
    pp.figure()
    pp.plot(ang, s_offset[0,:], 'k-')
    pp.xlabel('x rot ang')
    pp.ylabel('offset rot')
    pp.grid()
    pp.show()


def rotate_y():
    n_ims = 30
    ang = np.linspace(-np.pi+rotate_eps, np.pi-rotate_eps, n_ims)

    xyz = np.full((3,n_ims), np.nan)
    R = np.full((3,3,n_ims), np.nan)
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        s_i = np.array([0, ang[i], 0])
        R[:,:,i] = so3.exp(so3.cross(s_i)) @ R_upright
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
        xyz[:,i] = np.array([0, .8, 0])

    # generate
    generate_images(xyz, q, img_dir_rot_y)

    # predict
    xyz, q, xyz_est, q_est = db.get_predictions(img_dir_rot_y, print_errors=False)
    s_offset = np.full((3,n_ims), np.nan)
    for i in range(n_ims):
        R_est_i = t3d.quaternions.quat2mat(q_est[:,i])
        R_offset_i = R[:,:,i].T @ R_est_i
        s_offset[:,i] = so3.skew_elements(so3.log(R_offset_i))

    print('bias: ', s_offset[1,:])
    print('average bias: ', np.mean(s_offset[1,:]))

    # plot
    pp.figure()
    pp.plot(ang, s_offset[1,:], 'k-')
    pp.xlabel('y rot ang')
    pp.ylabel('offset rot')
    pp.grid()
    pp.show()


def rotate_z():
    n_ims = 30
    ang = np.linspace(-np.pi+rotate_eps, np.pi-rotate_eps, n_ims)

    xyz = np.full((3,n_ims), np.nan)
    R = np.full((3,3,n_ims), np.nan)
    q = np.full((4,n_ims), np.nan)
    for i in range(n_ims):
        s_i = np.array([0, 0, ang[i]])
        R[:,:,i] = so3.exp(so3.cross(s_i)) @ R_upright
        q[:,i] = t3d.quaternions.mat2quat(R[:,:,i])
        xyz[:,i] = np.array([0, .8, 0])

    # generate
    generate_images(xyz, q, img_dir_rot_z)

    # predict
    xyz, q, xyz_est, q_est = db.get_predictions(img_dir_rot_z, print_errors=False)
    s_offset = np.full((3,n_ims), np.nan)
    for i in range(n_ims):
        R_est_i = t3d.quaternions.quat2mat(q_est[:,i])
        R_offset_i = R[:,:,i].T @ R_est_i
        s_offset[:,i] = so3.skew_elements(so3.log(R_offset_i))

    print('bias: ', s_offset[2,:])
    print('average bias: ', np.mean(s_offset[2,:]))

    # plot
    pp.figure()
    pp.plot(ang, s_offset[2,:], 'k-')
    pp.xlabel('z rot ang')
    pp.ylabel('offset rot')
    pp.grid()
    pp.show()

# translations (uncomment to run)
#translate_x()
#translate_y()
#translate_z()

# rotations (uncomment to run)
#rotate_x()
#rotate_y()
#rotate_z()

# plots (uncomment to plot)
#po.plot_translation()
#po.plot_rotation()
po.plot_translation_and_rotation()
