import numpy as np
import asciiplotlib as ap
import matplotlib.pyplot as pp

import net_filter.dynamics.unscented_filter as uf 
import net_filter.filter.particle as pf 

def f(x, w, u, dt=1):
    '''
    dynamics and integration from example 3.7 (p. 196) in Optimal Estimation of
    Dynamic Systems (2nd ed.) by Crassidis & Junkins
    '''

    alp = 5e-5
    t = np.arange(0, 1, dt) 
    n_step = len(t)
    dx = lambda x: np.array([-x[1], -np.exp(-alp*x[0])*(x[1]**2)*x[2], 0])
    for i in range(n_step): 
        k1 = dt*dx(x)
        k2 = dt*dx(x + .5*k1)
        k3 = dt*dx(x + .5*k2)
        k4 = dt*dx(x + k3)
        x = x + (k1/6 + k2/3 + k3/3 + k4/6)

    return x


def h(x, u, v):
    '''
    measurement function from example 3.7 (p. 196) in Optimal Estimation of
    Dynamic Systems (2nd ed.) by Crassidis & Junkins
    '''

    M = 1e5
    Z = 1e5
    h = np.sqrt(M**2 + (x[0] - Z)**2) + v

    return h


'''
example 3.7 (p. 196) in Optimal Estimation of Dynamic Systems (2nd ed.) by
Crassidis & Junkins
'''

# setup
n = 3
Q = np.array([[0]])
R = np.array([[1e4]])
x_true = np.array([3e5, 2e4, 1e-3])

# estimates
x0_hat = np.array([3e5, 2e4, 3e-5])
P0_hat = np.diag([1e6, 4e6, 1e-4])

# actual system
dt = 1
tf = 60
t = np.arange(0, tf, dt)
m = t.shape[0]
X = np.full((n, m), np.nan)
Y = np.full((1, m), np.nan)
U = np.full((1, m), 0.0)
X[:,0] = x_true
Y[:,0] = h(x_true, 0, np.sqrt(R)*np.random.randn(1))
for i in range(1,m):
    X[:,i] = f(X[:,i-1], 0, 0, dt=1/64)
    Y[:,i] = h(X[:,i], 0, np.sqrt(R)*np.random.randn(1))

# filter parameters
alpha = 1
beta = 2
kappa = 0

# perform multiple trials and average them
n_trials = 1
X_ABS = np.full((m,n_trials), np.nan)
X_ABS_PF = np.full((m,n_trials), np.nan)
for i in range(n_trials): 
    X_HAT = uf.filter(f, h, Q, R, x0_hat, P0_hat, U, Y,
                             alpha, beta, kappa)
    X_ABS[:,i] = np.abs(X[1,:] - X_HAT[1,:])

    # particle filter
    H = np.array([[1, 0, 0]])
    Ups = np.array([[0, 0, 0]]).T
    n_particle = 100
    fxu = lambda x, u: f(x, 0, u)
    X_HAT_PF, X_PARTICLE = pf.filter(fxu, U, Y, H, R, Ups, Q,
                                    x0_hat, P0_hat, n_particle)
    X_ABS_PF[:,i] = np.abs(X[1,:] - X_HAT_PF[1,:])

X_ERR = np.mean(X_ABS, axis=1)
X_ERR_PF = np.mean(X_ABS_PF, axis=1)

# plot
'''
fig = ap.figure()
fig.plot(t, X_ERR)
fig.show()
'''
pp.figure()
pp.plot(t, X_ERR, label='uf')
pp.plot(t, X_ERR_PF, label='pf')
pp.legend()
pp.savefig('/tmp/test.png')
