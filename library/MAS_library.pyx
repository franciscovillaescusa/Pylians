import numpy as np
import time,sys,os
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,floor


# This function computes the density field of a cubic distribution of particles
def CIC(np.ndarray[np.float32_t,ndim=2] pos,
        np.ndarray[np.float32_t,ndim=3] number, float BoxSize):
        
    cdef int i,axis,dims
    cdef long particles
    cdef float inv_cell_size
    cdef np.ndarray[np.float32_t,ndim=1] dist,u,d
    cdef np.ndarray[np.int32_t,ndim=1] index_d,index_u

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  inv_cell_size = dims/BoxSize;  dims = len(number)

    # define arrays
    dist    = np.zeros(3,dtype=np.float32)
    index_d = np.zeros(3,dtype=np.int32)
    index_u = np.zeros(3,dtype=np.int32)
    u       = np.zeros(3,dtype=np.float32) #for up
    d       = np.zeros(3,dtype=np.float32) #for down
    
    # do a loop over all particles
    for i in xrange(particles):

        # $ (denotes grid point), X (denotes particle position)
        # $........$.X......$
        # -----------> dist
        # --------->   index_d
        # ------------------> index_u
        #          --> u
        #            -------> d
        for axis in xrange(3):
            dist[axis]    = pos[i,axis]*inv_cell_size
            u[axis]       = dist[axis]-floor(dist[axis])
            d[axis]       = 1.0-u[axis]
            index_d[axis] = (<int>dist[axis])%dims
            index_u[axis] = index_d[axis]+1
            index_u[axis] = index_u[axis]%dims #seems this is faster

        number[index_d[0],index_d[1],index_d[2]] += d[0]*d[1]*d[2]
        number[index_d[0],index_d[1],index_u[2]] += d[0]*d[1]*u[2]
        number[index_d[0],index_u[1],index_d[2]] += d[0]*u[1]*d[2]
        number[index_d[0],index_u[1],index_u[2]] += d[0]*u[1]*u[2]
        number[index_u[0],index_d[1],index_d[2]] += u[0]*d[1]*d[2]
        number[index_u[0],index_d[1],index_u[2]] += u[0]*d[1]*u[2]
        number[index_u[0],index_u[1],index_d[2]] += u[0]*u[1]*d[2]
        number[index_u[0],index_u[1],index_u[2]] += u[0]*u[1]*u[2]
        
    return number
