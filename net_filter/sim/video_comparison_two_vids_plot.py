# run shell commands to make a GIF in showing frames of neural net estimates,
# frames of filter estimates, and plots of the position & rotation errors
import os
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as pp

import net_filter.directories as dirs
import net_filter.tools.image as ti

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
    xlabel_font_size = 14
    legend_font_size = 13
    title_font_size = 18
    tick_font_size = 14

    # load results from file
    npz_file = os.path.join(dirs.simulation_dir, 'filter_results.npz')
    dat = np.load(npz_file)
    t = dat['t']
    p = dat['p']
    p_meas = dat['p_meas']
    p_hat = dat['p_filt']
    R_err = dat['R_err_filt']
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
                       wspace=None, hspace=.10)

    # position
    sp1 = pp.subplot(2,1,1)
    #sp1.set_title('position error', fontsize=title_font_size)
    pp.plot(t, p_err_meas, color=clr_net)
    pp.plot(t, p_err, color=clr_filt, linestyle='-')
    pp.axvline(x=t[i], color='k', zorder=0)
    pp.ylabel('pos error [cm]', fontsize=xlabel_font_size)
    pp.xticks([])
    #pp.yticks([])
    pp.grid(alpha=.1)
    #pp.box(False) # no outline box
    '''
    # alpha of outline box
    for spine_key, spine_val in sp1.axes.spines.items():
        spine_val.set_alpha(.1)
    '''
    # rotation
    sp2 = pp.subplot(2,1,2)
    #sp2.set_title('rotation error', fontsize=title_font_size)
    pp.plot(t, R_err_meas, color=clr_net)
    pp.plot(t, R_err, color=clr_filt, linestyle='-')
    pp.axvline(x=t[i], color='k', zorder=0)
    pp.ylabel('rot error [\\textdegree]', fontsize=xlabel_font_size)
    pp.xticks([])
    #pp.yticks([])
    pp.grid(alpha=.1)
    #pp.box(False) # no outline box

    pp.savefig(plot_ims[i], dpi=300)
    pp.close()

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

# debug: only do the first two images
debug = 0
if debug:
    net_ims = net_ims[:2]
    filter_ims = filter_ims[:2]
    n_ims = len(net_ims)

# geometry for convert command
h, w = ti.load_im_np(net_ims[0]).shape[:2]
w_space = 20
h_space = 100
h_upper_space = 80 # space above top of images (for title)
size_geom = str(w + w_space) + 'x' + str(h + h_space)

# geometry for -extent option in convert commands
# e.g. 740x580-0-80 
geom_net = size_geom + '-0-' + str(h_upper_space)
geom_filt = size_geom + '-0-' + str(h_upper_space)

###############################################################################
# label neural net images
for i, f in enumerate(net_ims):
    # add border
    cmd = 'convert ' + f + ' -verbose ' + \
            "-gravity north -background '#ffffff' " + \
            '-extent ' + geom_net + ' ' +  net_ims_labeled[i]
    subprocess.run([cmd], shell=True)

    # add text
    cmd = 'convert ' + net_ims_labeled[i] + ' -verbose ' + \
           '-gravity north  -font NimbusSans-Regular -pointsize 60 ' + \
           "-fill '#dd00ff' -annotate +0+20 'neural net' " + \
            net_ims_labeled[i]
    subprocess.run([cmd], shell=True)

###############################################################################
# label filter images
for i, f in enumerate(filter_ims): 
    # add border
    cmd = 'convert ' + f + ' -verbose ' + \
           "-gravity north -background '#ffffff' " + \
           '-extent ' + geom_filt + ' ' + filter_ims_labeled[i]
    subprocess.run([cmd], shell=True)

    # add text
    cmd = 'convert ' + filter_ims_labeled[i] + ' -verbose ' + \
            '-gravity north  -font NimbusSans-Regular -pointsize 60 ' + \
            "-fill '#00ddff' -annotate +0+20 'neural net + filter' " + \
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
    cmd = 'convert ' + plot_ims[i] + ' -resize 1300x ' + plot_ims[i]
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
gif_file = os.path.join(ani_dir, 'animation.gif')
palette_file = os.path.join(ani_dir, 'palette.png')

# make a palette of colors in the png files
# e.g.: ffmpeg -y -i im_%06d.png -vf palettegen palette.png
cmd = 'ffmpeg -y -i ' + combined_im_placeholder + ' -vf palettegen ' + palette_file
print(cmd)
subprocess.run([cmd], shell=True)

# make the gif
# e.g.: ffmpeg -y -i im_%03d.png -i palette.png -filter_complex "fps=60,setpts=0.175*PTS,paletteuse" animation.gif
cmd = 'ffmpeg -y -i ' + combined_im_placeholder + ' -i ' + palette_file + ' -filter_complex "fps=60,setpts=3.0*PTS,paletteuse" ' + gif_file
print(cmd)
subprocess.run([cmd], shell=True)
