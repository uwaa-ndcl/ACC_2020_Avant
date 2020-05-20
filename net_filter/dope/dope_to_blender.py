import os
import glob
import pickle
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.dope.eval as ev
import net_filter.tools.so3 as so3

def dope_to_blender(p_dope, q_dope, R_cam):
    '''
    convert dope position and orientation to blender
    
    Dope's reference coordinate frame for translation is
        x: right, y: down, z: away from the camera

    Dope's reference coordinate frame for rotation is default (R = identity
    matrix) when looking at the soup can upside down with the following
    coordinates:
        x: away from the camera, y: up, z: left

    this is a left-handed coordinate system, so we first convert to the
    right-handed system by negating x, which gives us the default Blender
    coordinate system of
        x: towards camera, y: up, z: left
    the Blender camera which produces these images has a rotation of (0,pi/2,0)
    in Blender's "XYZ Euler" coordinates

    we assume that the camera with the images are taken in Blender has the
    rotation matrix R_cam in Blender's coordinate system
    '''

    # translational conversion
    dist_ratio = 1/100 # Dope to Blender distance units, centimeters to meters

    # Blender cameras
    R_cam0 = t3d.euler.euler2mat(0,np.pi/2,0)
    R_cam0_to_cam = R_cam @ R_cam0.T

    # make arrays
    n = p_dope.shape[1]
    p_blend = np.full((3,n),np.nan)
    R_blend = np.full((3,3,n),np.nan)

    # convert position
    p_blend = np.copy(p_dope)
    p_blend[0,:] = -p_dope[2,:]
    p_blend[1,:] = -p_dope[1,:]
    p_blend[2,:] = -p_dope[0,:]
    p_blend *= dist_ratio
    p_blend = R_cam0_to_cam @ p_blend

    # loop over all images
    for i in range(n):
        # the network could not predict a pose, so set the predictions to some
        # default values
        if np.any(np.isnan(q_dope[:,i])) or np.any(np.isnan(p_dope[:,i])):
            print('image', i, 'could not be predicted')
            R_blend_i = np.eye(3)
            p_blend[:,i] = np.array([0,0,0])

        # the network predicted a pose, so convert it to our coordinate frame
        else:
            # convert orientation
            # is there a better way to do this?
            R_dope = t3d.quaternions.quat2mat(q_dope[:,i])
            ax, ang = t3d.axangles.mat2axangle(R_dope)
            ax[0] *= -1 # negate x-axis
            R_blend_i = t3d.axangles.axangle2mat(ax, ang)
            R_blend_i = R_cam0_to_cam @ R_blend_i
        R_blend[:,:,i] = R_blend_i

    return p_blend, R_blend


def get_predictions(img_dir, print_errors=True):
    '''
    get all predictions and truth values from a directory
    '''

    # true data
    data_pkl = os.path.join(img_dir, 'to_render.pkl')
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    p = data.pos
    R = data.rot_mat
    R_cam = data.cam_rot_mat

    # images in directory
    png_files = os.path.join(img_dir, '*.png')
    img_files = glob.glob(png_files)
    img_files = sorted(list(img_files))
    n_ims = len(img_files)

    # run Dope evaluation
    p_quat_pred = ev.eval(img_files, dirs.yaml_file, dirs.ckpt_file, draw=True, save_boxed_image=True)
    p_dope = p_quat_pred[:3,:]
    q_dope = p_quat_pred[3:,:]

    # convert Dope predictions to Blender
    p_blend, R_blend = dope_to_blender(p_dope, q_dope, R_cam)
    save_npz = os.path.join(img_dir, 'dope_pR.npz')
    np.savez(save_npz, p=p_blend, R=R_blend)

    # print errors
    if print_errors:
        for i in range(n_ims):
            print(i, 'pos error: ', np.linalg.norm(p[:,i] - p_blend[:,i]))
            print(i, 'ang error: ', so3.geodesic_distance(R_blend[:,:,i], R[:,:,i]))

    return p, R, p_blend, R_blend
