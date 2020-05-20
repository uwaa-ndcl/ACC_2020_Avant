import os
import pickle
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs
import net_filter.tools.so3 as so3
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
p = data.pos
R = data.rot_mat
n_ims = p.shape[1]

# load dope pose estimates
npz_file = os.path.join(img_dir, 'dope_pR.npz')
data = np.load(npz_file)
p_meas = data['p']
R_meas = data['R']

# convert to cm
p *= conv.m_to_cm
p_meas *= conv.m_to_cm

# translations
p_err = p_meas - p
p_mean = np.mean(p_err, axis=1)
p_normal = p_err - np.tile(p_mean[:,np.newaxis],n_ims) # subtract mean

# rotations
s = np.full((3,n_ims), np.nan)
for i in range(n_ims):
    # orientation
    S_i = so3.log(R[:,:,i].T @ R_meas[:,:,i])
    s[:,i] = so3.skew_elements(S_i)
s *= conv.rad_to_deg # convert to deg

# rotation mean
s_mean = np.mean(s, axis=1)
s_normal = s - np.tile(s_mean[:,np.newaxis], n_ims)

# compute covariances
cov_s = np.full((3,3), 0.0)
cov_p = np.full((3,3), 0.0)
cov_state = np.full((6,6), 0.0)
for i in range(n_ims):
    cov_p += p_normal[:,[i]] @ p_normal[:,[i]].T
    cov_s += s[:,[i]] @ s[:,[i]].T # mean s is 0, so we don't need to subtract it
    #cov_s += s_normal[:,[i]] @ s_normal[:,[i]].T # mean s is 0, so we don't need to subtract it

    # full state
    state = np.block([p_normal[:,i], s_normal[:,i]])
    state = state[:,np.newaxis]
    cov_state += state @ state.T

# average sum of covariances
cov_p *= 1/(n_ims - 1)
cov_s *= 1/(n_ims - 1)
cov_state *= 1/(n_ims - 1)

# covariance mean
state_mean = np.block([p_mean, s_mean])

# print results
print('mean state')
print(state_mean)
print('mean state (round to 2 decimals)')
print(np.around(state_mean, 2))
#print('covariance state')
#print(cov_state)
print('covariance state diagonal elements')
print(np.diag(cov_state))
print('covariance state diagonal elements (round to int)')
print(np.rint(np.diag(cov_state)))
