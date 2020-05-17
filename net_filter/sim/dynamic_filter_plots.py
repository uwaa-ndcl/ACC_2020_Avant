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
    p_hat = dat['p_hat']
    R_err = dat['R_err']
    R_err_meas = dat['R_err_meas']

    # errors over time
    p_err = np.linalg.norm(p_hat - p, axis=0)
    p_err_meas = np.linalg.norm(p_meas - p, axis=0)
    p_err_mean = np.mean(p_err)
    p_err_mean_meas = np.mean(p_err_meas)
    R_err_mean_meas = np.mean(R_err_meas)
    R_err_mean = np.mean(R_err)

    # setup
    pp.figure()
    pp.subplots_adjust(left=None, bottom=None, right=None, top=None,
                       wspace=None, hspace=.65)

    # position
    sp1 = pp.subplot(2,1,1)
    sp1.set_title('position error', fontsize=title_font_size)
    pp.plot(t, p_err_meas, 'k',
            label='neural network (mean = %.1f)' % p_err_mean_meas)
    pp.plot(t, p_err, 'r:', 
            label='filter \hspace{48pt} (mean = %.1f)' % p_err_mean)
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
            label='neural network (mean = %.1f)' % R_err_mean_meas)
    pp.plot(t, R_err, 'r:', 
            label='filter \\hspace{48pt} (mean = %.1f)' % R_err_mean)
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('$\\text{dist}(\\widehat{\\mathbf{R}}, \\mathbf{R})$ [\\textdegree]',
              fontsize=xlabel_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    #pp.ylim([0, 42])
    pp.legend(fontsize=legend_font_size, loc='upper right')
    pp.savefig(os.path.join(dirs.paper_figs_dir, 'simulation_errors.png'), dpi=300)

    pp.show()


def plot_3sigma():
    '''
    plot 3 sigma bounds
    '''

    # load results from file
    dat = np.load(filter_results_npz)
    t = dat['t']
    p = dat['p']
    p_hat = dat['p_hat']
    s_err = dat['s_err']
    COV_XX_ALL = dat['COV_XX_ALL']

    # setup
    pp.figure()
    pp.subplots_adjust(left=.15, bottom=None, right=.80, top=.90, wspace=None, hspace=.65)
    legend_font_size = 12

    # position
    sp1 = pp.subplot(2,1,1)
    sp1.set_title('$3 \sigma$ bounds, position', fontsize=title_font_size)

    # x
    pp.plot(t, p_hat[0,:] - p[0,:], 'r', label='x error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[0,0,:]), 'r--', label='$3 \sigma$, x')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[0,0,:]), 'r--')

    # y
    pp.plot(t, p_hat[1,:] - p[1,:], 'g', label='y error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[1,1,:]), 'g--', label='$3 \sigma$, y')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[1,1,:]), 'g--')

    # z
    pp.plot(t, p_hat[2,:] - p[2,:], 'b', label='z error')
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
    pp.plot(t, s_err[0,:], 'r', label='$s_1$ error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[3,3,:]), 'r--', label='$3 \sigma$, $s_1$')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[3,3,:]), 'r--')

    # rot y
    pp.plot(t, s_err[1,:], 'g', label='$s_2$ error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[4,4,:]), 'g--', label='$3 \sigma$, $s_2$')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[4,4,:]), 'g--')

    # rot z
    pp.plot(t, s_err[2,:], 'b', label='$s_3$ error')
    pp.plot(t, 3*np.sqrt(COV_XX_ALL[5,5,:]), 'b--', label='$3 \sigma$, $s_3$')
    pp.plot(t, -3*np.sqrt(COV_XX_ALL[5,5,:]), 'b--')

    # labels
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(bbox_to_anchor=(1,1.1), fontsize=legend_font_size)
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('error [\\textdegree]', fontsize=xlabel_font_size)
    pp.savefig(os.path.join(dirs.paper_figs_dir, 'simulation_sigma_bounds.png'), dpi=300)
    pp.show()


def plot_velocities():

    # load results from file
    dat = np.load(filter_results_npz)
    t = dat['t']
    pdot = dat['pdot']
    pdot_hat = dat['pdot_hat']
    om = dat['om']
    om_hat = dat['om_hat']

    # errors over time
    pdot_err = np.abs(pdot_hat - pdot)
    om_err = np.abs(om_hat - om)

    # setup
    pp.figure()
    pp.subplots_adjust(left=None, bottom=None, right=None, top=None,
                       wspace=None, hspace=.65)

    # velocity
    sp1 = pp.subplot(2,1,1)
    sp1.set_title('translational velocity errors', fontsize=title_font_size)
    pp.plot(t, pdot_err[0,:], 'y', label='$| \hat{\dot{p}}_1 - \dot{p}_1 |$')
    pp.plot(t, pdot_err[1,:], 'm', label='$| \hat{\dot{p}}_2 - \dot{p}_2 |$')
    pp.plot(t, pdot_err[2,:], 'c', label='$| \hat{\dot{p}}_3 - \dot{p}_3 |$')

    # labels
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('[cm/s]', fontsize=xlabel_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(fontsize=legend_font_size)

    # angular velocity
    sp2 = pp.subplot(2,1,2)
    sp2.set_title('rotational velocity errors', fontsize=title_font_size)
    pp.plot(t, om_err[0,:], 'y', label='$| \hat{\omega}_1 - \omega_1 |$')
    pp.plot(t, om_err[1,:], 'm', label='$| \hat{\omega}_2 - \omega_2 |$')
    pp.plot(t, om_err[2,:], 'c', label='$| \hat{\omega}_3 - \omega_3 |$')

    # labels
    pp.ylim([-5, 80])
    pp.xlabel('time [s]', fontsize=xlabel_font_size)
    pp.ylabel('[\\textdegree/s]', fontsize=xlabel_font_size)
    pp.xticks(fontsize=tick_font_size)
    pp.yticks(fontsize=tick_font_size)
    pp.legend(fontsize=legend_font_size, loc='upper right')

    # save
    pp.savefig(os.path.join(dirs.paper_figs_dir, 'simulation_velocities.png'), dpi=300)
    pp.show()
