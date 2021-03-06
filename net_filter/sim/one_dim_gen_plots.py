import os
import pickle
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as pp

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.tools.unit_conversion as conv

# upright can
R_upright = t3d.euler.euler2mat(-np.pi/2, 0, -np.pi/2, 'sxyz')

def plot_translation_and_rotation():

    # load x data
    data_dope = np.load(os.path.join(dirs.trans_x_dir, 'dope_pR.npz'))
    p_dope_x = data_dope['p']

    with open(os.path.join(dirs.trans_x_dir, 'to_render.pkl'), 'rb') as file:
        data_true = pickle.load(file)
    p_true_x = data_true.pos

    # load y data
    data_dope = np.load(os.path.join(dirs.trans_y_dir, 'dope_pR.npz'))
    p_dope_y = data_dope['p']
    with open(os.path.join(dirs.trans_y_dir, 'to_render.pkl'), 'rb') as file:
        data_true = pickle.load(file)
    p_true_y = data_true.pos

    # load z data
    data_dope = np.load(os.path.join(dirs.trans_z_dir, 'dope_pR.npz'))
    p_dope_z = data_dope['p']
    with open(os.path.join(dirs.trans_z_dir, 'to_render.pkl'), 'rb') as file:
        data_true = pickle.load(file)
    p_true_z = data_true.pos

    # load x rotation data
    data_dope = np.load(os.path.join(dirs.rot_x_dir, 'dope_pR.npz'))
    R_dope_x = data_dope['R']
    with open(os.path.join(dirs.rot_x_dir, 'to_render.pkl'), 'rb') as file:
        data_true = pickle.load(file)
    R_true_x = data_true.rot_mat

    # load y rotation data
    data_dope = np.load(os.path.join(dirs.rot_y_dir, 'dope_pR.npz'))
    R_dope_y = data_dope['R']
    with open(os.path.join(dirs.rot_y_dir, 'to_render.pkl'), 'rb') as file:
        data_true = pickle.load(file)
    R_true_y = data_true.rot_mat

    # load z rotation data
    data_dope = np.load(os.path.join(dirs.rot_z_dir, 'dope_pR.npz'))
    R_dope_z = data_dope['R']
    with open(os.path.join(dirs.rot_z_dir, 'to_render.pkl'), 'rb') as file:
        data_true = pickle.load(file)
    R_true_z = data_true.rot_mat

    # convert position to cm
    p_dope_x *= conv.m_to_cm
    p_dope_y *= conv.m_to_cm
    p_dope_z *= conv.m_to_cm
    p_true_x *= conv.m_to_cm
    p_true_y *= conv.m_to_cm
    p_true_z *= conv.m_to_cm

    # convert rotation to tangent space representation
    n_ims = p_dope_x.shape[1]
    s_x = np.full((3,n_ims), np.nan)
    s_y = np.full((3,n_ims), np.nan)
    s_z = np.full((3,n_ims), np.nan)
    s_offset_x = np.full((3,n_ims), np.nan)
    s_offset_y = np.full((3,n_ims), np.nan)
    s_offset_z = np.full((3,n_ims), np.nan)
    for i in range(n_ims):
        # x
        R_i = R_true_x[:,:,i]
        R_est_i = R_dope_x[:,:,i]
        R_offset_i = R_i.T @ R_est_i
        s_x[:,i] = so3.skew_elements(so3.log(R_i @ R_upright.T))
        s_offset_x[:,i] = so3.skew_elements(so3.log(R_offset_i))

        # y
        R_i = R_true_y[:,:,i]
        R_est_i = R_dope_y[:,:,i]
        R_offset_i = R_i.T @ R_est_i
        s_y[:,i] = so3.skew_elements(so3.log(R_i @ R_upright.T))
        s_offset_y[:,i] = so3.skew_elements(so3.log(R_offset_i))

        # z
        R_i = R_true_z[:,:,i]
        R_est_i = R_dope_z[:,:,i]
        R_offset_i = R_i.T @ R_est_i
        s_z[:,i] = so3.skew_elements(so3.log(R_i @ R_upright.T))
        s_offset_z[:,i] = so3.skew_elements(so3.log(R_offset_i))

    # convert rotation to degrees
    s_x *= conv.rad_to_deg 
    s_y *= conv.rad_to_deg 
    s_z *= conv.rad_to_deg 
    s_offset_x *= conv.rad_to_deg 
    s_offset_y *= conv.rad_to_deg 
    s_offset_z *= conv.rad_to_deg 

    # plot settings
    pp.rc('text', usetex=True)
    pp.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{textcomp}')
    x_font_size = 18
    tick_font_size = 14

    # make figure
    fig = pp.figure(figsize=(7,10))
    pp.subplots_adjust(left=.15, bottom=.10, right=.95, top=.95,
                       wspace=None, hspace=0.9)

    # x
    ax1 = pp.subplot(6,1,1)
    pp.ylim(-1.0, 1.0)
    pp.plot(p_true_x[0,:], p_dope_x[0,:] - p_true_x[0,:])
    pp.xlabel('$x$ [cm]', fontsize=x_font_size)
    pp.ylabel('$\\hat{x} - x$ [cm]', fontsize=x_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    ax1.axhline(0, color='gray', linewidth=0.5)

    # y
    ax2 = pp.subplot(6,1,2)
    pp.ylim(-35.0, 35.0)
    pp.plot(p_true_y[1,:], p_dope_y[1,:] - p_true_y[1,:])
    pp.xlabel('$y$ [cm]', fontsize=x_font_size)
    pp.ylabel('$\\hat{y} - y$ [cm]', fontsize=x_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    ax2.axhline(0, color='gray', linewidth=0.5)

    # z
    ax3 = pp.subplot(6,1,3)
    pp.ylim(-1.0, 1.0)
    pp.plot(p_true_z[2,:], p_dope_z[2,:] - p_true_z[2,:])
    pp.xlabel('$z$ [cm]', fontsize=x_font_size)
    pp.ylabel('$\\hat{z} - z$ [cm]', fontsize=x_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    ax3.axhline(0, color='gray', linewidth=0.5)

    # s_1
    ax4 = pp.subplot(6,1,4)
    pp.plot(s_x[0,:], s_offset_x[0,:], 'r')
    pp.ylim(-3.4, 3.4)
    pp.xlabel('$s_1$ [\\textdegree]', fontsize=x_font_size)
    pp.ylabel('$\\hat{s}_1 - s_1$ [\\textdegree]', fontsize=x_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    ax4.axhline(0, color='gray', linewidth=0.5)

    # s_2
    ax5 = pp.subplot(6,1,5)
    pp.plot(s_y[1,:], s_offset_y[1,:], 'r')
    pp.ylim(-5.7, 5.7)
    pp.xlabel('$s_2$ [\\textdegree]', fontsize=x_font_size)
    pp.ylabel('$\\hat{s}_2 - s_2$ [\\textdegree]', fontsize=x_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    ax5.axhline(0, color='gray', linewidth=0.5)

    # s_3
    ax6 = pp.subplot(6,1,6)
    pp.plot(s_z[2,:], s_offset_z[2,:], 'r')
    pp.ylim(-2.3, 2.3)
    pp.xlabel('$s_3$ [\\textdegree]', fontsize=x_font_size)
    pp.ylabel('$\\hat{s}_3 - s_3$ [\\textdegree]', fontsize=x_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    ax6.axhline(0, color='gray', linewidth=0.5)

    # save plot
    pp.savefig(os.path.join(dirs.paper_figs_dir, 'trans_rot.png'), dpi=300)
    pp.show()
