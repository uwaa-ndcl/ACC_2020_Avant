import numpy as np
import matplotlib.pyplot as pp

import net_filter.dynamics.unscented_filter as uf 
import net_filter.filter.particle as pf 

def f(x_prev, w, u):
    #x = np.array([np.exp(2*x_prev[0]), x_prev[1]**2])
    #x = np.array([2*x_prev[0], 2*x_prev[1] + 2])
    x = np.array([x_prev**2/(1 + x_prev**3) + w])
    return x

def h(x, u, v):
    return x + v

# initial conditions
n = 1
n_y = 1
x_0 = np.array([1])

# noises
Q = 1 * np.diag([.1])
R = 1 * np.diag([1])
n_v = R.shape[0]
#Ups = np.zeros((n, 1))
#Ups = np.ones((n, 1))
Ups = np.eye(n)
#H = np.block([np.eye(n_y), np.zeros((n_y, n-n_y))])
H = np.array([[1]])

# estimates
#x0_hat = x_0 + np.array([.9, .1])
#x0_hat = x_0 + np.array([.1, .1])
#x0_hat = x_0
x0_hat = np.array([.8])
P0_hat = 1 * np.eye(n)

# actual system
tf = 51
dt = 1
t = np.arange(0, tf, dt)
n_t = t.shape[0]
X = np.full((n, n_t), np.nan)
Y = np.full((n_y, n_t), np.nan)
U = np.full((1, n_t), 0.0)
V = np.random.multivariate_normal(np.zeros(n_v), R, n_t).T
#V = .1 * np.random.uniform(-1, 1, (n_y, n_t))
X[:,0] = x_0
Y[:,0] = h(x_0, 0, V[:,0])
for i in range(1, n_t):
    X[:,i] = f(X[:,i-1], 0, 0)
    Y[:,i] = h(X[:,i], 0, V[:,i])

# filter parameters
alpha = 1
beta = 2
kappa = 2

# perform multiple trials and average them
n_trials = 1
X_ERR_UF = np.full((n_t, n_trials), np.nan)
X_ERR_PF = np.copy(X_ERR_UF)
for i in range(n_trials): 
    # unscented
    X_HAT = uf.filter(f, h, Q, R, x0_hat, P0_hat, U, Y,
                             alpha, beta, kappa)
    X_ERR_UF[:,i] = np.linalg.norm(X - X_HAT, axis=0)

    # y error
    Y_MEAS_ERR = np.linalg.norm(Y - X[:6,:], axis=0)
    Y_FILT_ERR = np.linalg.norm(X_HAT[:6,:] - X[:6,:], axis=0)

    # particle
    n_particle = 500
    fxu = lambda x, u: f(x, 0, u)
    X_HAT_PF, X_PARTICLE = pf.filter(fxu, U, Y, H, R, Ups, Q,
                                    x0_hat, P0_hat, n_particle)
    X_ERR_PF[:,i] = np.linalg.norm(X - X_HAT_PF, axis=0)

X_ERR_UF = np.mean(X_ERR_UF, axis=1)
X_ERR_PF = np.mean(X_ERR_PF, axis=1)

# plot
pp.figure()
pp.plot(t, X_ERR_UF, label='total UF')
pp.plot(t, X_ERR_PF, label='total PF')
#pp.plot(t, Y_MEAS_ERR, label='y meas')
#pp.plot(t, Y_FILT_ERR, label='y filter')
pp.legend()
pp.savefig('/tmp/test.png')
