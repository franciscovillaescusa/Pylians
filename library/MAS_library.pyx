import numpy as np
import time,sys,os
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin


# This function computes the density field of a cubic distribution of particles
def CIC(np.ndarray[np.float32_t,ndim=2] pos, float BoxSize, int dims):

    start = time.time()
    cdef int i,j,k,axis,num,particles
    cdef float inv_mean_dist
    cdef np.ndarray[np.float32_t,ndim=3] number
    cdef np.ndarray[np.float32_t,ndim=1] dist,u,d
    cdef np.ndarray[np.int32_t,ndim=1] index_d,index_u

    # find number of particles
    particles = len(pos)
    
    # inverse of the cell distance 
    inv_mean_dist = dims/BoxSize

    # define arrays
    number  = np.zeros((dims,dims,dims),dtype=np.float32)
    dist    = np.zeros(3,dtype=np.float32)
    index_d = np.zeros(3,dtype=np.int32)
    index_u = np.zeros(3,dtype=np.int32)
    u       = np.zeros(3,dtype=np.float32) #for up
    d       = np.zeros(3,dtype=np.float32) #for down
    

    for num in xrange(particles):

        for axis in xrange(3):
            dist[axis]    = pos[num,axis]*inv_mean_dist
            index_d[axis] = <int>dist[axis]
            u[axis]       = dist[axis]-index_d[axis]
            d[axis]       = 1.0-u[axis]
            index_d[axis] = index_d[axis]%dims
            index_u[axis] = (index_d[axis]+1) 
            index_u[axis] = index_u[axis]%dims #seems this is faster


        number[index_d[0],index_d[0],index_d[2]] += d[0]*d[1]*d[2]
        number[index_d[0],index_d[0],index_u[2]] += d[0]*d[1]*u[2]
        number[index_d[0],index_u[0],index_d[2]] += d[0]*u[1]*d[2]
        number[index_d[0],index_u[0],index_u[2]] += d[0]*u[1]*u[2]
        number[index_u[0],index_d[0],index_d[2]] += u[0]*d[1]*d[2]
        number[index_u[0],index_d[0],index_u[2]] += u[0]*d[1]*u[2]
        number[index_u[0],index_u[0],index_d[2]] += u[0]*u[1]*d[2]
        number[index_u[0],index_u[0],index_u[2]] += u[0]*u[1]*u[2]
        
    t = 1000.0*(time.time()-start)
    return number,t
