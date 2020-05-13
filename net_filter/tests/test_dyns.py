import os
import pickle
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as pp

import net_filter.blender.render as br
import net_filter.dynamics.rigid_body as rb
import net_filter.dynamics.angular_velocity as av
from net_filter.blender.render_properties import RenderProperties

n_t = 100
def generate():
    # rigid body dynamics
    t = np.linspace(0, .7, n_t)
    # xyz: right, forward, up
    # singleshot pose: z is down
    xyz = np.array([-.3, 1.5, -.3])
    q = t3d.euler.euler2quat(.1, .4, -.5, 'sxyz')
    v = np.array([.8,0,3.0])
    om = np.array([0,0,0])
    #v = np.array([0,0,1])
    '''
    # spin
    xyz = np.array([0, 1, 0])
    q = t3d.euler.euler2quat(.0, .0, .0, 'sxyz')
    v = np.array([0,0,0])
    om = np.array([0,0,7])
    '''
    q_dot = av.om_to_qdot(om, q)
    x = np.concatenate((xyz, q, v, q_dot), axis=0)
    print(x.shape)
    X = rb.integrate(t, x)

    # render
    save_dir = '/home/trevor/Downloads/test_dir/'
    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    render_props = RenderProperties()
    render_props.n_renders = n_t
    render_props.model_name = 'watering_can'
    render_props.xyz = X[:3,:]
    render_props.quat = X[3:7,:]
    render_props.alpha = False
    render_props.compute_gramian = False
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    br.blender_render(save_dir)


def get_preds():
    file = '/home/trevor/Downloads/preds.npz'
    #preds_trans = np.array([2,3,4])
    #gts_trans = np.array([5,9,0])
    #np.savez(file, preds_trans=preds_trans, gts_trans=gts_trans)
    data = np.load(file)
    preds_trans = data['preds_trans']
    gts_trans = data['gts_trans']
    preds_rot = data['preds_rot']
    gts_rot = data['gts_rot']
    preds_trans = np.squeeze(preds_trans)
    preds_trans = preds_trans[:n_t,:].T
    preds_rot = preds_rot[:n_t,:,:]

    euler_xyz = np.full((3,n_t), np.nan)
    for i in range(n_t):
        euler_x, euler_y, euler_z = t3d.euler.mat2euler(preds_rot[i,:,:], 'rxyz')
        euler_xyz[:,i] = np.array([euler_x, euler_y, euler_z])

    print('2',preds_trans.shape)
    return preds_trans, euler_xyz

def evaluate():
    preds_trans, euler_xyz = get_preds()
    pp.figure()
    print(preds_trans[:,0].shape)
    #pp.plot(preds_trans[:,:])
    pp.plot(euler_xyz[2,:])
    pp.legend(['x', 'y', 'z'])
    pp.savefig('/tmp/plot.png')

#generate()
evaluate()
