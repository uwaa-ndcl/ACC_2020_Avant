import os
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as pp

import net_filter.directories as dirs
import net_filter.tools.image as ti
import net_filter.sim.dynamic_filter_plots as fp

clr_net = (221, 0, 255)
clr_net = tuple(i/255 for i in clr_net) # put into range [0,1]
clr_filt = (0, 221, 255)
clr_filt = tuple(i/255 for i in clr_filt) # put into range [0,1]

def plot(i):
    '''
    plot position and rotation errors
    '''

    # parameters for all plots
    pp.rc('text', usetex=True)
    pp.rc('text.latex',
          preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{textcomp}')
    xlabel_font_size = 16
    legend_font_size = 13
    title_font_size = 18
    tick_font_size = 14

    # load results from file
    npz_file = os.path.join(dirs.simulation_dir, 'filter_results.npz')
    dat = np.load(npz_file)
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
    pp.figure(figsize=[6.4,3.2]) # default figsize=[6.4,4.8]
    pp.subplots_adjust(left=None, bottom=None, right=None, top=None,
                       wspace=None, hspace=.15)

    # position
    sp1 = pp.subplot(2,1,1)
    #sp1.set_title('position error', fontsize=title_font_size)
    pp.plot(t, p_err_meas, color=clr_net,
            label='neural network (mean = %.1f)' % p_err_mean_meas)
    pp.plot(t, p_err, color=clr_filt, linestyle='-', 
            label='filter \hspace{48pt} (mean = %.1f)' % p_err_mean)
    pp.axvline(x=t[i], color='k', zorder=0)
    #pp.xlabel('time [s]', fontsize=xlabel_font_size)
    #pp.ylabel('$\| \hat{\mathbf{p}} - \mathbf{p} \|$ [cm]', fontsize=xlabel_font_size)
    pp.ylabel('pos error', fontsize=xlabel_font_size)
    #pp.xticks(fontsize=tick_font_size)
    #pp.yticks(fontsize=tick_font_size)
    pp.xticks([])
    pp.yticks([])
    #pp.legend(fontsize=legend_font_size, loc='upper right')
    #pp.axis('off')
    #pp.spines['top'].set_visible(False)
    #pp.box(False)

    # rotation
    sp2 = pp.subplot(2,1,2)
    #sp2.set_title('rotation error', fontsize=title_font_size)
    pp.plot(t, R_err_meas, color=clr_net,
            label='neural network (mean = %.1f)' % R_err_mean_meas)
    pp.plot(t, R_err, color=clr_filt, linestyle='-', 
            label='filter \\hspace{48pt} (mean = %.1f)' % R_err_mean)
    pp.axvline(x=t[i], color='k', zorder=0)
    #pp.xlabel('time [s]', fontsize=xlabel_font_size)
    #pp.ylabel('$\\text{dist}(\\widehat{\\mathbf{R}}, \\mathbf{R})$ [\\textdegree]', fontsize=xlabel_font_size)
    pp.ylabel('rot error', fontsize=xlabel_font_size)
    #pp.xticks(fontsize=tick_font_size)
    #pp.yticks(fontsize=tick_font_size)
    pp.xticks([])
    pp.yticks([])
    #pp.legend(fontsize=legend_font_size, loc='upper right')

    pp.savefig(plot_ims[i], dpi=300)

# directories
ani_dir = dirs.animation_dir
net_dir = os.path.join(ani_dir, 'net/')
filter_dir = os.path.join(ani_dir, 'filter/')
net_labeled_dir = os.path.join(net_dir, 'labeled/')
filter_labeled_dir = os.path.join(filter_dir, 'labeled/')
plot_dir = os.path.join(dirs.animation_dir, 'plots/')
combined_dir = os.path.join(ani_dir, 'combined/')

# image files
#sim_ims = sorted(glob.glob(os.path.join(sim_dir, '*.png')))
net_ims = sorted(glob.glob(os.path.join(net_dir, '*.png')))
filter_ims = sorted(glob.glob(os.path.join(filter_dir, '*.png')))
n_ims = len(net_ims)

# image files to be created
tails = [os.path.split(file)[1] for file in net_ims]
net_ims_labeled = [os.path.join(net_labeled_dir, tail) for tail in tails]
filter_ims_labeled = [os.path.join(filter_labeled_dir, tail) for tail in tails]
combined_ims = [os.path.join(combined_dir, tail) for tail in tails]
combined_im_placeholder = os.path.join(combined_dir, '%06d.png')
plot_ims = [os.path.join(plot_dir, tail) for tail in tails]
gif_file = os.path.join(ani_dir, 'animation.gif')

# debug: only do the first two images
net_ims = net_ims[:2]
filter_ims = filter_ims[:2]
n_ims = len(net_ims)

# geometry for convert command
h, w = ti.load_im_np(net_ims[0]).shape[:2]
w_space = 100
h_space = 100
h_upper_space = 80 # space above top of images (for title)
size_geom = str(w + w_space) + 'x' + str(h + h_space)

# geometry for -extent option in convert commands
# e.g. 740x580-16-80 
geom_1 = size_geom + '-' + str(w_space//6) + '-' + str(h_upper_space)
geom_2 = size_geom + '+' + str(w_space//6) + '-' + str(h_upper_space)


###############################################################################
# label neural net images
for i, f in enumerate(net_ims):
    # add border
    cmd = 'convert ' + f + ' -verbose ' + \
            "-gravity north -background '#ffffff' -extent " + geom_1 + ' ' + \
            net_ims_labeled[i]
    subprocess.run([cmd], shell=True)

    # add text
    cmd = 'convert ' + net_ims_labeled[i] + ' -verbose ' + \
            '-gravity north  -font NimbusSans-Regular -pointsize 60 ' + \
            "-fill '#000000' -annotate +0+20 'neural net' " + \
            net_ims_labeled[i]
    subprocess.run([cmd], shell=True)

###############################################################################
# label filter images
for i, f in enumerate(filter_ims): 
    # add border
    cmd = 'convert ' + f + ' -verbose ' + \
            "-gravity north -background '#ffffff' -extent " + geom_2 + ' ' + \
            filter_ims_labeled[i]
    subprocess.run([cmd], shell=True)

    # add text
    cmd = 'convert ' + filter_ims_labeled[i] + ' -verbose ' + \
            '-gravity north  -font NimbusSans-Regular -pointsize 60 ' + \
            "-fill '#000000' -annotate +0+20 'neural net + filter' " + \
            filter_ims_labeled[i]
    subprocess.run([cmd], shell=True)


###############################################################################
# plots
for i in range(n_ims):
    plot(i)

    # crop whitespace
    cmd = 'convert ' + plot_ims[i] + ' -trim ' + plot_ims[i]
    subprocess.run([cmd], shell=True)
    
    # resize
    cmd = 'convert ' + plot_ims[i] + ' -resize 1350x ' + plot_ims[i]
    subprocess.run([cmd], shell=True)

###############################################################################
# combine images
for i in range(n_ims):
    # combine frames
    cmd = 'convert ' + net_ims_labeled[i] + ' ' + filter_ims_labeled[i] + \
            ' +append ' + combined_ims[i]
    subprocess.run([cmd], shell=True)

    # add plot
    cmd = 'convert -gravity center ' + combined_ims[i] + ' ' + plot_ims[i] + \
            ' -append ' + combined_ims[i]
    subprocess.run([cmd], shell=True)

    # trim whitespace
    cmd = 'convert ' + combined_ims[i] + ' -trim ' + combined_ims[i]
    subprocess.run([cmd], shell=True)

###############################################################################
# make gif
cmd = 'ffmpeg -y -framerate 7 -i ' + combined_im_placeholder + ' ' + gif_file
print(cmd)
subprocess.run([cmd], shell=True)
