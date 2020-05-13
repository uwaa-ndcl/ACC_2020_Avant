import numpy as np
import matplotlib.pyplot as pp

n = 10000

# indeterminant forms in exponential map
theta_a = np.linspace(-1e-12, 1e-12, n)
theta_b = np.linspace(-1e-6, 1e-6, n)
a = np.sin(theta_a)/theta_a
b = (1 - np.cos(theta_b))/(theta_b**2)

# indeterminant form in the logarithmic map
theta_c = np.linspace(-1e-6, 1e-6, n+1)
c = theta_c/(2*np.sin(theta_c))

# print any nans
for i in range(n):
    if np.isnan(a[i]):
        print('THETA A IS NAN AT THETA=', theta_a[i], '!!!')
    if np.isnan(b[i]):
        print('THETA B IS NAN AT THETA=', theta_b[i], '!!!')
    if np.isnan(c[i]):
        print('THETA C IS NAN AT THETA=', theta_c[i], '!!!')

# very small values for logarithmic indeterminant form
eps1 = 1e-323
eps2 = 1e-324 # this turns into 0.0!
out1 = eps1/(2*np.sin(eps1))
out2 = eps2/(2*np.sin(eps2))
print(eps1, out1)
print(eps2, out2)

# print
print('a: ', a)
print('c: ', c)

pp.figure()
pp.subplot(3,1,1)
pp.plot(theta_a, a)
pp.xlabel('theta')
pp.ylabel('(sine theta)/theta')
pp.subplot(3,1,2)
pp.plot(theta_b, b)
pp.xlabel('theta')
pp.ylabel('(1 - cosine theta)/(theta**2)')
pp.subplot(3,1,3)
pp.plot(theta_c, c)
pp.xlabel('theta')
pp.ylabel('(theta)/(2*sine theta)')
pp.show()
