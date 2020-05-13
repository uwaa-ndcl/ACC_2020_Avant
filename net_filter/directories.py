'''
these are local directories for the entire package
they should be changed by each user
'''

import os.path

# blender
blender_models_dir = '/home/trevor/neural_network_filtering/blender_models/'

# renders
main_dir =       '/home/trevor/large_files/net_filter/'
simulation_dir = os.path.join(main_dir, 'simulation/')
#simulation_dir = '/home/trevor/large_files/net_filter/simulation_trials/19/'
simulation_dynamic_err_dir = os.path.join(simulation_dir, 'dynamic_err')
trials_dir =     os.path.join(main_dir, 'simulation_trials/')
#trials_dir =     os.path.join(main_dir, 'simulation_trials_bright/')
trans_x_dir =    os.path.join(main_dir, 'trans_x/')
trans_y_dir =    os.path.join(main_dir, 'trans_y/')
trans_z_dir =    os.path.join(main_dir, 'trans_z/')
rot_x_dir =      os.path.join(main_dir, 'rot_x/')
rot_y_dir =      os.path.join(main_dir, 'rot_y/')
rot_z_dir =      os.path.join(main_dir, 'rot_z/')
cov_bright_dir = os.path.join(main_dir, 'cov_bright/')
cov_dim_dir =    os.path.join(main_dir, 'cov_dim/')

snapshots_dir =  os.path.join(main_dir, 'snapshots/')
paper_figs_dir = os.path.join(main_dir, 'paper_figs/')
camera_calibration_images_dir = os.path.join(main_dir, 'camera_calibration_images/')

# dope
yaml_file = '/home/trevor/neural_network_filtering/net_filter/dope/my_config.yaml'
ckpt_file = '/home/trevor/neural_network_filtering/net_filter/dope/soup_60.pth'
