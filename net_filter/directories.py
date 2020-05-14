import os
import net_filter

# get directory of this python package
net_filter_file = net_filter.__file__
net_filter_dir = os.path.dirname(net_filter_file)
main_dir = os.path.dirname(net_filter_dir)

# blender
blender_models_dir = os.path.join(main_dir, 'blender_models')

# results
results_dir =     os.path.join(main_dir, 'results/')
simulation_dir =  os.path.join(results_dir, 'simulation/')
trials_dir =      os.path.join(results_dir, 'simulation_trials/')
trans_x_dir =     os.path.join(results_dir, 'trans_x/')
trans_y_dir =     os.path.join(results_dir, 'trans_y/')
trans_z_dir =     os.path.join(results_dir, 'trans_z/')
rot_x_dir =       os.path.join(results_dir, 'rot_x/')
rot_y_dir =       os.path.join(results_dir, 'rot_y/')
rot_z_dir =       os.path.join(results_dir, 'rot_z/')
cov_bright_dir =  os.path.join(results_dir, 'cov_bright/')
cov_dim_dir =     os.path.join(results_dir, 'cov_dim/')
snapshots_dir =   os.path.join(results_dir, 'snapshots/')
dynamic_err_dir = os.path.join(simulation_dir, 'dynamic_err')

# paper figures
paper_figs_dir =  os.path.join(results_dir, 'paper_figs/')

# dope
yaml_file = os.path.join(net_filter_dir, 'dope/my_config.yaml')
ckpt_file = os.path.join(net_filter_dir, 'dope/soup_60.pth')
