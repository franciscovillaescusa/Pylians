import numpy as np 
cimport numpy as np
import time,sys,os
cimport cython
cimport cyintegration as CYI

cpdef double trapezoidal(double[::1] x_array, double[::1] y_array):
    return CYI.rectangular(x_array[0], x_array[-1], &x_array[0], &y_array[0],
                           x_array.shape[0])

cpdef double simpson(double[::1] x_array, double[::1] y_array, long steps):
    return CYI.simpson(x_array[0], x_array[-1], &x_array[0], &y_array[0],
                           x_array.shape[0], steps)
