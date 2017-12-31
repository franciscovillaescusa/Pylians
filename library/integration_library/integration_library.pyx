import numpy as np 
cimport numpy as np
import time,sys,os
cimport cython
cimport integration as CI
from libc.math cimport sqrt,pow,sin,cos,floor,fabs

############### example 1 ################
cdef void example(double x, double y[], double dydx[],
                   double a[], double b[], long elements):
    dydx[0] = x*x + 2.0*x + 3.0
##########################################

############### example 2 ################
cdef void example2(double x, double y[], double dydx[],
                   double a[], double b[], long elements):

    cdef int index
    cdef double a_interp

    if x<=a[0]:             a_interp = a[0]
    elif x>=a[elements-1]:  a_interp = a[elements-1]
    else:
        index = <int>((x-a[0])/(a[elements-1]-a[0])*(elements-1))
        a_interp = (b[index+1] - b[index])/(a[index+1]-a[index])*(x-a[index]) + b[index]
    dydx[0] = a_interp
##########################################

    
############### Trapezoidal rule ##############
cpdef double trapezoidal(double[::1] x_array, double[::1] y_array):
    return CI.rectangular(x_array[0], x_array[-1], &x_array[0], &y_array[0],
                          x_array.shape[0])
###############################################

############### Simpson rule ##################
cpdef double simpson(double[::1] x_array, double[::1] y_array, long steps):
    return CI.simpson(x_array[0], x_array[-1], &x_array[0], &y_array[0],
                      x_array.shape[0], steps)
###############################################

# RK4 example1 for debugging
cpdef RK4_example(double[::1] yinit, double x1, double x2, long nstep):

    cdef double *result      
    result =  CI.rkdumb(&yinit[0], yinit.shape[0], x1, x2, nstep,
	                NULL, NULL, 0, example)
    # cast pointer to cython memory view. Then memory view to numpy array
    return np.asarray(<double[:yinit.shape[0]]> result)

# RK4 example2 for debugging
cpdef RK4_example2(double[::1] yinit, double x1, double x2, long nstep,
                   double[::1] a, double[::1] b):

    cdef double *result      
    result =  CI.rkdumb(&yinit[0], yinit.shape[0], x1, x2, nstep,
	                &a[0], &b[0], a.shape[0], example2)
    # cast pointer to cython memory view. Then memory view to numpy array
    return np.asarray(<double[:yinit.shape[0]]> result)

    
