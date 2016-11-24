import numpy as np
import time,sys,os
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin


# This function computes the density field of a cubic distribution of particles
def CIC(np.ndarray[np.float32_t,ndim=3] pos, float BoxSize, int dims):

    start = time.time()
    cdef int i,j,k,axis
    cdef float inv_mean_dist
    cdef np.ndarray[np.float32_t,ndim=3] number
    cdef np.ndarray[np.float32_t,ndim=1] dist
    cdef np.ndarray[np.int32_t,ndim=1] index

    # inverse of the cell distance 
    inv_mean_dist = dims/BoxSize

    # define arrays
    number = np.zeros((dims,dims,dims),dtype=np.float32)
    dist   = np.zeros(3,dtype=np.float32)
    index  = np.zeros(3,dtype=np.int32)

    for i in xrange(dims):
        for j in xrange(dims):
            for k in xrange(dims):
                number[i,j,k] = 0.0

                for axis in xrange(3):
                    dist[axis] = pos[i,j,k]*inv_mean_dist
                    index[axis] = <int>dist[axis]
    

    
    
    #print 'Time = %.2f ms'%((time.time()-start)*1000.0)
    t = 1000.0*(time.time()-start)
    return t
