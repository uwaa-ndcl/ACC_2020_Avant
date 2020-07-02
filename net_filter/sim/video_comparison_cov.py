# run shell commands to stack corresponding covariance (bright & dim) images
import os
import glob
import subprocess

import net_filter.directories as dirs

# directories
ani_dir = dirs.animation_dir
dim_bright_dir = os.path.join(ani_dir, 'cov_dim_bright/')
dim_dir = dirs.cov_dim_dir
bright_dir = dirs.cov_bright_dir

# image files
dim_ims = sorted(glob.glob(os.path.join(dim_dir, '*.png')))
bright_ims = sorted(glob.glob(os.path.join(bright_dir, '*.png')))
n_ims = len(dim_ims)

# image files to be created
tails = [os.path.split(file)[1] for file in dim_ims]
dim_bright_ims = [os.path.join(dim_bright_dir, tail) for tail in tails]

n_im_go = 20
for i in range(n_im_go):
    # simulation + network
    cmd = 'montage ' + bright_ims[i] + ' ' + dim_ims[i] + \
            ' -tile x2 -geometry +0+27 ' + dim_bright_ims[i]
    subprocess.run([cmd], shell=True)

    # crop whitespace
    cmd = 'convert ' + dim_bright_ims[i] + ' -trim ' + dim_bright_ims[i]
    subprocess.run([cmd], shell=True)
