import os
import numpy as np
import matplotlib.pyplot as pp

import net_filter.directories as dirs

filter_results_npz = os.path.join(dirs.simulation_dir, 'filter_results.npz')

# parameters for all plots
pp.rc('text', usetex=True)
pp.rc('text.latex',
      preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{textcomp}')
xlabel_font_size = 16
legend_font_size = 13
title_font_size = 18
tick_font_size = 14

def plot_errors():
    '''
    plot position and rotation errors
    '''

    # load results from file
    dat = np.load(filter_results_npz)
    t = dat['t']
    p = dat['p']
    p_meas = dat['p_meas']
    p_filt = dat['p_filt']
    R_err_meas = dat['R_err_meas']
    R_err_filt = dat['R_err_filt']

    # errors over time
    p_err_meas = np.linalg.norm(p_meas - p, axis=0)
    p_err_filt = np.linalg.norm(p_filt - p, axis=0)
    p_err_meas_mean = np.mean(p_err_meas)
    p_err_filt_mean = np.mean(p_err_filt)
    R_err_meas_mean = np.mean(R_err_meas)
    R_err_filt_mean = np.mean(R_err_filt)

    # setup
    pp.figure()
    pp.subplots_adjust(left=None, bottom=None, right=None, top=None,
                       wspace=None, hspace=.65)

    # position
    sp1 = pp.subplot(2,1,1)
    sp1.set_title('position error', fontsize=title_font_size)
    pp.plot(t, p_err_meas, 'k',
            label='neural network (mean = %.1f)' % p_err_meas_mean)
    pp.plot(t, p_err_filt, 'r:', 
            label='filter \hspace{48pt} (mean = %.1f)' % p_err_filt_mean)
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('$\| \hat{\mathbf{p}} - \mathbf{p} \|$ [cm]',
              fontsize=xlabel_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(fontsize=legend_font_size, loc='upper right')

    # rotation
    sp2 = pp.subplot(2,1,2)
    sp2.set_title('rotation error', fontsize=title_font_size)
    pp.plot(t, R_err_meas, 'k',
            label='neural network (mean = %.1f)' % R_err_meas_mean)
    pp.plot(t, R_err_filt, 'r:', 
            label='filter \\hspace{48pt} (mean = %.1f)' % R_err_filt_mean)
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('$\\text{dist}(\\widehat{\\mathbf{R}}, \\mathbf{R})$ [\\textdegree]',
              fontsize=xlabel_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    #pp.ylim([0, 42])
    pp.legend(fontsize=legend_font_size, loc='upper right')
    pp.savefig(os.path.join(dirs.paper_figs_dir, 'simulation_errors.png'),
               dpi=300)

    #pp.show()


def plot_3sigma():
    '''
    plot 3 sigma bounds
    '''

    # load results from file
    dat = np.load(filter_results_npz)
    t = dat['t']
    p = dat['p']
    p_filt = dat['p_filt']
    s_err_filt = dat['s_err_filt']
    COV_XX_ALL = dat['COV_XX_ALL']

    # setup
    pp.figure()
    pp.subplots_adjust(left=.15, bottom=None, right=.80, top=.90,
                       wspace=None, hspace=.65)
    legend_font_size = 12

    # position
    sp1 = pp.subplot(2,1,1)
    sp1.set_title('$3 \sigma$ bounds, position', fontsize=title_font_size)

    # x
    pp.plot(t, p_filt[0,:] - p[0,:], 'r', label='x error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[0,0,:]), 'r--', label='$3 \sigma$, x')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[0,0,:]), 'r--')

    # y
    pp.plot(t, p_filt[1,:] - p[1,:], 'g', label='y error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[1,1,:]), 'g--', label='$3 \sigma$, y')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[1,1,:]), 'g--')

    # z
    pp.plot(t, p_filt[2,:] - p[2,:], 'b', label='z error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[2,2,:]), 'b--', label='$3 \sigma$, z')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[2,2,:]), 'b--')

    # labels
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(bbox_to_anchor=(1,1.1), loc=2, fontsize=legend_font_size)
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('error [cm]', fontsize=xlabel_font_size)

    # rotation
    sp2 = pp.subplot(2,1,2)
    sp2.set_title('$3 \sigma$ bounds, rotation', fontsize=title_font_size)

    # rot x
    pp.plot(t, s_err_filt[0,:], 'r', label='$s_1$ error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[3,3,:]), 'r--', label='$3 \sigma$, $s_1$')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[3,3,:]), 'r--')

    # rot y
    pp.plot(t, s_err_filt[1,:], 'g', label='$s_2$ error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[4,4,:]), 'g--', label='$3 \sigma$, $s_2$')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[4,4,:]), 'g--')

    # rot z
    pp.plot(t, s_err_filt[2,:], 'b', label='$s_3$ error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[5,5,:]), 'b--', label='$3 \sigma$, $s_3$')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[5,5,:]), 'b--')

    # labels
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(bbox_to_anchor=(1,1.1), fontsize=legend_font_size)
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('error [\\textdegree]', fontsize=xlabel_font_size)
    file_name = os.path.join(dirs.paper_figs_dir, 'simulation_sigma_bounds.png')
    pp.savefig(file_name, dpi=300)
    #pp.show()


def plot_velocities():

    # load results from file
    dat = np.load(filter_results_npz)
    t = dat['t']
    pdot = dat['pdot']
    pdot_filt = dat['pdot_filt']
    om = dat['om']
    om_filt = dat['om_filt']

    # errors over time
    pdot_err = np.abs(pdot_filt - pdot)
    om_err = np.abs(om_filt - om)

    # setup
    pp.figure()
    pp.subplots_adjust(left=None, bottom=None, right=None, top=None,
                       wspace=None, hspace=.65)

    # velocity
    sp1 = pp.subplot(2,1,1)
    sp1.set_title('translational velocity errors', fontsize=title_font_size)
    sp1.plot(t, pdot_err[0,:], 'y', label='$| \hat{\dot{p}}_1 - \dot{p}_1 |$')
    sp1.plot(t, pdot_err[1,:], 'm', label='$| \hat{\dot{p}}_2 - \dot{p}_2 |$')
    sp1.plot(t, pdot_err[2,:], 'c', label='$| \hat{\dot{p}}_3 - \dot{p}_3 |$')

    # labels
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('[cm/s]', fontsize=xlabel_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(fontsize=legend_font_size)

    # angular velocity
    sp2 = pp.subplot(2,1,2)
    sp2.set_title('rotational velocity errors', fontsize=title_font_size)
    sp2.plot(t, om_err[0,:], 'y', label='$| \hat{\omega}_1 - \omega_1 |$')
    sp2.plot(t, om_err[1,:], 'm', label='$| \hat{\omega}_2 - \omega_2 |$')
    sp2.plot(t, om_err[2,:], 'c', label='$| \hat{\omega}_3 - \omega_3 |$')

    # labels
    pp.ylim([-5, 80])
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('[\\textdegree/s]', fontsize=xlabel_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(fontsize=legend_font_size, loc='upper right')

    # plot network's estimates of velocities based on a finite difference
    plot_fd = 0
    if plot_fd:
        dt = t[1]-t[0]
        pdot = dat['pdot']
        p_meas = dat['p_meas']
        R = dat['R']
        R_meas = dat['R_meas']
        t_fd = t[1:]
        n_fd = t_fd.size
        n_t = t.size

        p_meas_min1 = np.roll(p_meas, 1, axis=1)
        pdot_meas = (p_meas - p_meas_min1)/dt
        pdot_meas_err = np.abs(pdot - pdot_meas)
        pdot_meas_err = pdot_meas_err[:,1:]
        
        sp1.plot(t_fd, pdot_meas_err[0,:], 'y:') 
        sp1.plot(t_fd, pdot_meas_err[1,:], 'm:') 
        sp1.plot(t_fd, pdot_meas_err[2,:], 'c:') 
        print(np.min(pdot_meas_err))

        import net_filter.tools.so3 as so3 
        import net_filter.tools.unit_conversion as conv 
        dist = np.full(n_fd, np.nan)
        s = np.full((3,n_t), np.nan)
        s_meas = np.full((3,n_t), np.nan)
        for i in range(n_t):
            s_meas[:,i] = so3.skew_elements(so3.log(R_meas[:,:,i]))
        s_meas_min1 = np.roll(s_meas, 1, axis=1)
        om_meas = (s_meas - s_meas_min1)/dt 
        om_meas_err = np.abs(om - om_meas)
        om_meas_err *= conv.rad_to_deg 
        om_meas_err = om_meas_err[:,1:]

        import pdb; pdb.set_trace()
        sp2.plot(t_fd, om_meas_err[0,:], 'y:') 
        sp2.plot(t_fd, om_meas_err[1,:], 'm:') 
        sp2.plot(t_fd, om_meas_err[2,:], 'c:') 

    # save
    file_name = os.path.join(dirs.paper_figs_dir, 'simulation_velocities.png')
    pp.savefig(file_name, dpi=300)
    #pp.show()
