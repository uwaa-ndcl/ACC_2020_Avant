'''apply Dope to png images in a directory'''
import os
import glob
import numpy as np

import net_filter.directories as dirs
import net_filter.dope.eval as ev

# images
img_dir = dirs.simulation_dir
img_files = glob.iglob(os.path.join(img_dir, '*.png'))
img_files = sorted(list(img_files))

# run evaluation
xyz_quat_pred = ev.eval(img_files, dirs.yaml_file, dirs.ckpt_file)
