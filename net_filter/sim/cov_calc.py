import os
import pickle
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
import net_filter.tools.print as tp
import net_filter.tools.unit_conversion as conv

#mode = 'dim'
mode = 'bright'

if mode == 'dim':
    img_dir = dirs.cov_dim_dir
elif mode == 'bright':
    img_dir = dirs.cov_bright_dir

# load true pose
data_pkl = os.path.join(img_dir, 'to_render.pkl')
with open(data_pkl, 'rb') as f:
    data = pickle.load(f)
xyz = data.xyz
q = data.quat
n_ims = xyz.shape[1]

# load dope pose estimates
npz_file = os.path.join(img_dir, 'dope_xyzq.npz')
data = np.load(npz_file)
xyz_meas = data['xyz']
q_meas = data['q']

# convert to cm
xyz *= conv.m_to_cm
xyz_meas *= conv.m_to_cm

# convert quaternions to rotation matrices
R = np.full((3,3,n_ims),np.nan)
R_meas = np.full((3,3,n_ims),np.nan)
for i in range(n_ims):
    R[:,:,i] = t3d.quaternions.quat2mat(q[:,i])
    R_meas[:,:,i] = t3d.quaternions.quat2mat(q_meas[:,i])

# translations
xyz_err = xyz_meas - xyz
xyz_mean = np.mean(xyz_err, axis=1)
xyz_normal = xyz_err - np.tile(xyz_mean[:,np.newaxis],n_ims) # subtract mean

# rotations
R = np.full((3,3,n_ims), np.nan)
R_meas = np.full((3,3,n_ims), np.nan)
s = np.full((3,n_ims), np.nan)

# covariances
cov_s = np.full((3,3), 0.0)
cov_xyz = np.full((3,3), 0.0)
cov_state = np.full((6,6), 0.0)

for i in range(n_ims):
    # orientation
    R[:,:,i] = t3d.quaternions.quat2mat(q[:,i])
    R_meas[:,:,i] = t3d.quaternions.quat2mat(q_meas[:,i])
    S_i = so3.log(R[:,:,i].T @ R_meas[:,:,i])
    s[:,i] = so3.skew_elements(S_i)

# convert to deg
s *= conv.rad_to_deg

# rotation mean
s_mean = np.mean(s, axis=1)
s_normal = s - np.tile(s_mean[:,np.newaxis], n_ims)

# compute covariances
for i in range(n_ims):
    cov_xyz += xyz_normal[:,[i]] @ xyz_normal[:,[i]].T
    cov_s += s[:,[i]] @ s[:,[i]].T # mean s is 0, so we don't need to subtract it
    #cov_s += s_normal[:,[i]] @ s_normal[:,[i]].T # mean s is 0, so we don't need to subtract it

    # full state
    state = np.block([xyz_normal[:,i], s_normal[:,i]])
    state = state[:,np.newaxis]
    cov_state += state @ state.T

# average sum of covariances
cov_xyz *= 1/(n_ims - 1)
cov_s *= 1/(n_ims - 1)
cov_state *= 1/(n_ims - 1)

# covariance mean
state_mean = np.block([xyz_mean, s_mean])

# print results
#print('mean xyz')
#print(xyz_mean)
#print('covariance xyz')
#print(cov_xyz)
#print('covariance s')
#print(cov_s)
print('mean state')
print(state_mean)
print('covariance state')
print(cov_state)
print('covariance state diagonal elements')
print(np.diag(cov_state))

#tp.print_matrix_as_latex(cov_state, n_digs=3)
#tp.print_matrix_as_latex(state_mean[:,np.newaxis], n_digs=2)
#print(state_mean)
