#This script is an example of how to numerically compute the integral
#I = \int_0^3 (a+b*x^2) dx
import numpy as np
import scipy.integrate as si


def func(y,x,a,b):
    return a+b*x**2

############## INPUT ##############
a = 0.0;  b = 1.0
###################################

yinit    = [0.0];     x_limits = [0.0, 3.0] 
I = si.odeint(func, yinit, x_limits, args=(a,b), mxstep=1000000,
              rtol=1e-8, atol=1e-10,  h0=1e-10)[1][0]
print 'I =',I
