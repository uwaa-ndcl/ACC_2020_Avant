import os
import glob
import subprocess

import net_filter.directories as dirs

# directories
sim_dir = dirs.simulation_dir
ani_dir = dirs.animation_dir
net_dir = os.path.join(ani_dir, 'net/')
filter_dir = os.path.join(ani_dir, 'filter/')
sim_net_dir = os.path.join(ani_dir, 'sim_net/')
sim_net_filter_dir = os.path.join(ani_dir, 'sim_net_filter/')

# image files
sim_ims = sorted(glob.glob(os.path.join(sim_dir, '*.png')))
net_ims = sorted(glob.glob(os.path.join(net_dir, '*.png')))
filter_ims = sorted(glob.glob(os.path.join(filter_dir, '*.png')))
n_ims = len(net_ims)

# image files to be created
tails = [os.path.split(file)[1] for file in net_ims]
sim_net_ims = [os.path.join(sim_net_dir, tail) for tail in tails]
sim_net_filter_ims = [os.path.join(sim_net_filter_dir, tail) for tail in tails]

for i in range(n_ims):
    # simulation + network
    cmd = 'montage ' + sim_ims[i] + ' ' + net_ims[i] + \
            ' -geometry +10+0 ' + sim_net_ims[i]
    subprocess.run([cmd], shell=True)

    # crop whitespace
    cmd = 'convert ' + sim_net_ims[i] + ' -trim ' + sim_net_ims[i]
    subprocess.run([cmd], shell=True)

    # simulation + network + filter
    cmd = 'montage ' + sim_net_ims[i] + ' ' + filter_ims[i] + \
            ' -geometry +10+0 ' + sim_net_filter_ims[i] 
    subprocess.run([cmd], shell=True)

    # crop whitespace
    cmd = 'convert ' + sim_net_filter_ims[i] + ' -trim ' + sim_net_filter_ims[i]
    subprocess.run([cmd], shell=True)
