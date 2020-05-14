import os
import glob
import subprocess

import net_filter.directories as dirs
import net_filter.tools.image as ti

# directories
sim_dir = dirs.simulation_dir
net_dir = os.path.join(sim_dir, 'net/')
filter_dir = os.path.join(sim_dir, 'filter/')
net_labeled_dir = os.path.join(net_dir, 'labeled/')
filter_labeled_dir = os.path.join(filter_dir, 'labeled/')
combined_dir = os.path.join(sim_dir, 'combined/')

# image files
sim_ims = sorted(glob.glob(os.path.join(sim_dir, '*.png')))
net_ims = sorted(glob.glob(os.path.join(net_dir, '*.png')))
filter_ims = sorted(glob.glob(os.path.join(filter_dir, '*.png')))
n_ims = len(sim_ims)

# image files to be created
tails = [os.path.split(file)[1] for file in sim_ims]
net_ims_labeled = [os.path.join(net_labeled_dir, tail) for tail in tails]
filter_ims_labeled = [os.path.join(filter_labeled_dir, tail) for tail in tails]
combined_ims = [os.path.join(combined_dir, tail) for tail in tails]

# debug: only do the first two images
sim_ims = sim_ims[:2]
net_ims = net_ims[:2]
filter_ims = filter_ims[:2]
n_ims = len(sim_ims)

# geometry for convert command
h, w = ti.load_im_np(sim_ims[0]).shape[:2]
w_space = 100
h_space = 100
h_upper_space = 80 # space above top of images (for title)
size_geom = str(w + w_space) + 'x' + str(h + h_space)

# geometry for -extent option in convert commands
# e.g. 740x580-16-80 
geom_1 = size_geom + '-' + str(w_space//6) + '-' + str(h_upper_space)
geom_2 = size_geom + '+' + str(w_space//6) + '-' + str(h_upper_space)

# label neural net images
for i, f in enumerate(net_ims):
    # add border
    cmd = 'convert ' + f + ' -verbose ' + \
            "-gravity north -background '#202020' -extent " + geom_1 + ' ' + \
            net_ims_labeled[i]
    subprocess.run([cmd], shell=True)

    # add text
    cmd = 'convert ' + net_ims_labeled[i] + ' -verbose ' + \
            '-gravity north  -font NimbusSans-Regular -pointsize 60 ' + \
            "-fill '#dd00ff' -annotate +0+20 'neural net' " + \
            net_ims_labeled[i]
    subprocess.run([cmd], shell=True)

# label filter images
for i, f in enumerate(filter_ims): 
    # add border
    cmd = 'convert ' + f + ' -verbose ' + \
            "-gravity north -background '#202020' -extent " + geom_2 + ' ' + \
            filter_ims_labeled[i]
    subprocess.run([cmd], shell=True)

    # add text
    cmd = 'convert ' + filter_ims_labeled[i] + ' -verbose ' + \
            '-gravity north  -font NimbusSans-Regular -pointsize 60 ' + \
            "-fill '#00ddff' -annotate +0+20 'neural net + filter' " + \
            filter_ims_labeled[i]
    subprocess.run([cmd], shell=True)

# combine images
for i in range(n_ims):
    cmd = 'convert ' + net_ims_labeled[i] + ' ' + filter_ims_labeled[i] + \
            ' +append ' + combined_ims[i]
    subprocess.run([cmd], shell=True)

