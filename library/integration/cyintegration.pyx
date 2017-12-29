import numpy as np 
cimport numpy as np
import time,sys,os
cimport cython
cimport cyintegration as CYI

cpdef void trapezoidal(double[::1] x_array, double[::1] y_array, long steps):
    return CYI.rectangular(x_array[0], x_array[1], &x_array[0], &y_array[0],
                           x_array.shape[0], steps)
