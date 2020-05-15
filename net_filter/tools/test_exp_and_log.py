'''
test the indeterminant forms of the exponential and log maps

the results of this script show:

exp map: the indeterminant form (sin theta)/theta is only a problem when theta=0

         the indeterminant form (1 - cos theta)/(theta^2) becomes a problem
         about when abs(theta)<1e-3, we can rewrite the term 1-cos(x)/x^2 as
         .5*((sin(x/2)/(x/2))^2), which is more numerically stable and only
         is a problem when theta=0

         see: https://math.stackexchange.com/questions/1334164/finding-the-limit-of-1-cos-x-x2

log map: the indeterminant form is only a problem when theta=0
'''
import numpy as np
import matplotlib.pyplot as pp

pp.rc('text.latex',
      preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{textcomp}')

n = 10001 # make it odd so linspace includes 0

# indeterminant forms in exponential map
theta_a = np.linspace(-1e-12, 1e-12, n)
theta_b = np.linspace(-1e-6, 1e-6, n)
theta_b_XL = np.linspace(-1e-3, 1e-3, n)
a = np.sin(theta_a)/theta_a
b = (1 - np.cos(theta_b))/(theta_b**2)
b_new =.5*((np.sin(theta_b/2)/(theta_b/2))**2) # alternative formula
b_XL = (1 - np.cos(theta_b_XL))/(theta_b_XL**2)
b_new_XL =.5*((np.sin(theta_b_XL/2)/(theta_b_XL/2))**2)

# indeterminant form in the logarithmic map
theta_c = np.linspace(-1e-6, 1e-6, n)
c = theta_c/(2*np.sin(theta_c))

# print any nans
a_nans = np.isnan(a)
b_nans = np.isnan(b)
b_new_nans = np.isnan(b_new)
c_nans = np.isnan(c)
print('THETA A IS NAN AT THETA =', *theta_a[a_nans], '!!!') 
print('THETA B IS NAN AT THETA =', *theta_b[b_nans], '!!!') 
print('THETA B NEW IS NAN AT THETA =', *theta_b[b_new_nans], '!!!') 
print('THETA C IS NAN AT THETA =', *theta_c[c_nans], '!!!') 

# very small values for logarithmic indeterminant form
eps1 = 1e-323 # this still works! 
eps2 = 1e-324 # this turns into 0.0!
out1 = eps1/(2*np.sin(eps1))
out2 = eps2/(2*np.sin(eps2))
print(eps1, out1)
print(eps2, out2)

# print
print('a: ', a)
print('c: ', c)

pp.figure()
pp.subplot(5,1,1)
pp.plot(theta_a, a)
pp.axvline(x=theta_a[a_nans], color='k', linestyle='--')
pp.xlabel('$\\theta$')
pp.ylabel('$\\frac{\\sin \\theta}{\\theta}$ (exp)')

pp.subplot(5,1,2)
pp.plot(theta_b, b, label='old formula')
pp.plot(theta_b, b_new, color='r', linestyle=':', label='new formula')
pp.axvline(x=theta_b[b_nans], color='k', linestyle='-')
pp.axvline(x=theta_b[b_new_nans], color='r', linestyle='--')
pp.xlabel('$\\theta$')
pp.ylabel('$\\frac{1 - \\cos \\theta}{\\theta^2}$ (exp)')
pp.legend()

pp.subplot(5,1,3)
pp.plot(theta_b, b_new, color='r', linestyle=':')
pp.axvline(x=theta_b[b_new_nans], color='r', linestyle='--')
pp.xlabel('$\\theta$')
pp.ylabel('$\\frac{1 - \\cos \\theta}{\\theta^2}$ (exp, new formula)')

pp.subplot(5,1,4)
clean_inds = np.abs(theta_b_XL)>1e-4
pp.plot(theta_b_XL[clean_inds], b_XL[clean_inds], label='old formula')
pp.plot(theta_b_XL[clean_inds], b_new_XL[clean_inds], color='r', linestyle=':', label='new formula')
pp.legend()
pp.xlabel('$\\theta$')
pp.ylabel('$\\frac{1 - \\cos \\theta}{\\theta^2}$ (exp, no small values)')

pp.subplot(5,1,5)
pp.plot(theta_c, c)
pp.axvline(x=theta_c[c_nans], color='k', linestyle='--')
pp.xlabel('$\\theta$')
pp.ylabel('$\\frac{\\theta}{2*\\sin \\theta}$ (log)')
pp.show()
