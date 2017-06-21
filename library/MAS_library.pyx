import numpy as np 
import time,sys,os
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,floor,fabs

################################################################################
################################# ROUTINES #####################################
# NGP(pos,number,BoxSize)
# CIC(pos,number,BoxSize)
# TSC(pos,number,BoxSize)
# PCS(pos,number,BoxSize)
# NGPW(pos,number,BoxSize,W)
# CICW(pos,number,BoxSize,W)
# TSCW(pos,number,BoxSize,W)
# PCSW(pos,number,BoxSize,W)
# CIC_interp(pos,density,BoxSize,dens)
################################################################################
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] CIC(np.ndarray[np.float32_t,ndim=2] pos,
                                          np.ndarray[np.float32_t,ndim=3] number,
                                          float BoxSize):
    cdef int axis,dims
    cdef long i,particles
    cdef float inv_cell_size,dist
    cdef float u[3]
    cdef float d[3]
    cdef int index_u[3]
    cdef int index_d[3]

    # define arrays. This is slower than defining this as cython arrays
    #cdef np.ndarray[np.float32_t,ndim=1] u,d
    #cdef np.ndarray[np.int32_t,  ndim=1] index_d,index_u
    #u       = np.zeros(3,dtype=np.float32) #for up
    #d       = np.zeros(3,dtype=np.float32) #for down
    #index_u = np.zeros(3,dtype=np.int32)
    #index_d = np.zeros(3,dtype=np.int32)
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    
    # do a loop over all particles
    for i in xrange(particles):

        # $: grid point, X: particle position
        # $.........$..X......$
        # ------------>         dist    (1.3)
        # --------->            index_d (1)
        # --------------------> index_u (2)
        #           --->        u       (0.3)
        #              -------> d       (0.7)
        for axis in xrange(3):
            dist          = pos[i,axis]*inv_cell_size
            u[axis]       = dist - <int>dist
            d[axis]       = 1.0 - u[axis]
            index_d[axis] = (<int>dist)%dims
            index_u[axis] = index_d[axis] + 1
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
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# using weights
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
# W --------> weights of the particles
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] CICW(np.ndarray[np.float32_t,ndim=2] pos,
                                           np.ndarray[np.float32_t,ndim=3] number,
                                           float BoxSize,
                                           np.ndarray[np.float32_t,ndim=1] W):
    cdef int axis,dims
    cdef long i,particles
    cdef float inv_cell_size,dist
    cdef np.ndarray[np.float32_t,ndim=1] u,d
    cdef np.ndarray[np.int32_t,  ndim=1] index_d,index_u
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    
    # define arrays
    u       = np.zeros(3,dtype=np.float32) #for up
    d       = np.zeros(3,dtype=np.float32) #for down
    index_u = np.zeros(3,dtype=np.int32)
    index_d = np.zeros(3,dtype=np.int32)
    
    # do a loop over all particles
    for i in xrange(particles):

        # $: grid point, X: particle position
        # $.........$..X......$
        # ------------>         dist    (1.3)
        # --------->            index_d (1)
        # --------------------> index_u (2)
        #           --->        u       (0.3)
        #              -------> d       (0.7)
        for axis in xrange(3):
            dist          = pos[i,axis]*inv_cell_size
            u[axis]       = dist - <int>dist
            d[axis]       = 1.0 - u[axis]
            index_d[axis] = (<int>dist)%dims
            index_u[axis] = index_d[axis] + 1
            index_u[axis] = index_u[axis]%dims #seems this is faster

        number[index_d[0],index_d[1],index_d[2]] += d[0]*d[1]*d[2]*W[i]
        number[index_d[0],index_d[1],index_u[2]] += d[0]*d[1]*u[2]*W[i]
        number[index_d[0],index_u[1],index_d[2]] += d[0]*u[1]*d[2]*W[i]
        number[index_d[0],index_u[1],index_u[2]] += d[0]*u[1]*u[2]*W[i]
        number[index_u[0],index_d[1],index_d[2]] += u[0]*d[1]*d[2]*W[i]
        number[index_u[0],index_d[1],index_u[2]] += u[0]*d[1]*u[2]*W[i]
        number[index_u[0],index_u[1],index_d[2]] += u[0]*u[1]*d[2]*W[i]
        number[index_u[0],index_u[1],index_u[2]] += u[0]*u[1]*u[2]*W[i]
        
    return number
################################################################################

################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] NGP(np.ndarray[np.float32_t,ndim=2] pos,
                                          np.ndarray[np.float32_t,ndim=3] number,
                                          float BoxSize):
    cdef int axis,dims
    cdef long i,particles
    cdef float inv_cell_size
    cdef np.ndarray[np.int32_t,  ndim=1] index
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    index = np.zeros(3,dtype=np.int32)

    # do a loop over all particles
    for i in xrange(particles):
        for axis in xrange(3):
            index[axis] = <int>(pos[i,axis]*inv_cell_size + 0.5)
            index[axis] = index[axis]%dims
        number[index[0],index[1],index[2]] += 1.0

    return number

################################################################################

################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] NGPW(np.ndarray[np.float32_t,ndim=2] pos,
                                           np.ndarray[np.float32_t,ndim=3] number,
                                           float BoxSize,
                                           np.ndarray[np.float32_t,ndim=1] W):
    cdef int axis,dims
    cdef long i,particles
    cdef float inv_cell_size
    cdef np.ndarray[np.int32_t,  ndim=1] index
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    index = np.zeros(3,dtype=np.int32)

    # do a loop over all particles
    for i in xrange(particles):
        for axis in xrange(3):
            index[axis] = <int>(pos[i,axis]*inv_cell_size + 0.5)
            index[axis] = index[axis]%dims
        number[index[0],index[1],index[2]] += W[i]

    return number
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] TSC(np.ndarray[np.float32_t,ndim=2] pos,
                                          np.ndarray[np.float32_t,ndim=3] number,
                                          float BoxSize):
    cdef int axis,dims,minimum,j,l,m,n
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef np.ndarray[np.float32_t,ndim=2] C
    cdef np.ndarray[np.int32_t,  ndim=2] index
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    
    # define arrays
    C     = np.zeros((3,4),dtype=np.float32) #contribution of particle to grid point
    index = np.zeros((3,4),dtype=np.int32)   #index of the grid point

    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the three axes of the particle
        for axis in xrange(3):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-1.5)
            for j in xrange(4): #only 4 cells/dimension can contribute
                index[axis,j] = (minimum+j+dims)%dims
                diff = fabs(minimum+j - dist)
                if diff<0.5:    C[axis,j] = 0.75-diff*diff
                elif diff<1.5:  C[axis,j] = 0.5*(1.5-diff)*(1.5-diff)
                else:           C[axis,j] = 0.0

        for l in xrange(4):  
            for m in xrange(4):  
                for n in xrange(4): 
                    number[index[0,l],index[1,m],index[2,n]] += C[0,l]*C[1,m]*C[2,n]
            
    return number
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] TSCW(np.ndarray[np.float32_t,ndim=2] pos,
                                           np.ndarray[np.float32_t,ndim=3] number,
                                           float BoxSize,
                                           np.ndarray[np.float32_t,ndim=1] W):
    cdef int axis,dims,minimum,j,l,m,n
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef np.ndarray[np.float32_t,ndim=2] C
    cdef np.ndarray[np.int32_t,  ndim=2] index
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    
    # define arrays
    C     = np.zeros((3,4),dtype=np.float32) #contribution of particle to grid point
    index = np.zeros((3,4),dtype=np.int32)   #index of the grid point

    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the three axes of the particle
        for axis in xrange(3):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-1.5)
            for j in xrange(4): #only 4 cells/dimension can contribute
                index[axis,j] = (minimum+j+dims)%dims
                diff = fabs(minimum+j - dist)
                if diff<0.5:    C[axis,j] = 0.75-diff*diff
                elif diff<1.5:  C[axis,j] = 0.5*(1.5-diff)*(1.5-diff)
                else:           C[axis,j] = 0.0

        for l in xrange(4):  
            for m in xrange(4):  
                for n in xrange(4): 
                    number[index[0,l],index[1,m],index[2,n]] += C[0,l]*C[1,m]*C[2,n]*W[i]
            
    return number
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] PCS(np.ndarray[np.float32_t,ndim=2] pos,
                                          np.ndarray[np.float32_t,ndim=3] number,
                                          float BoxSize):
    cdef int axis,dims,minimum,j,l,m,n
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef np.ndarray[np.float32_t,ndim=2] C
    cdef np.ndarray[np.int32_t,  ndim=2] index
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    
    # define arrays
    C     = np.zeros((3,5),dtype=np.float32) #contribution of particle to grid point
    index = np.zeros((3,5),dtype=np.int32)   #index of the grid point

    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the three axes of the particle
        for axis in xrange(3):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-2.0)
            for j in xrange(5): #only 5 cells/dimension can contribute
                index[axis,j] = (minimum+j+dims)%dims
                diff = fabs(minimum+j - dist)
                if diff<1.0:    C[axis,j] = (4.0 - 6.0*diff*diff + 3.0*diff*diff*diff)/6.0
                elif diff<2.0:  C[axis,j] = (2.0 - diff)*(2.0 - diff)*(2.0 - diff)/6.0
                else:           C[axis,j] = 0.0

        for l in xrange(5):  
            for m in xrange(5):  
                for n in xrange(5): 
                    number[index[0,l],index[1,m],index[2,n]] += C[0,l]*C[1,m]*C[2,n]
            
    return number
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] PCSW(np.ndarray[np.float32_t,ndim=2] pos,
                                           np.ndarray[np.float32_t,ndim=3] number,
                                           float BoxSize,
                                           np.ndarray[np.float32_t,ndim=1] W):
    cdef int axis,dims,minimum,j,l,m,n
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef np.ndarray[np.float32_t,ndim=2] C
    cdef np.ndarray[np.int32_t,  ndim=2] index
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(number);  inv_cell_size = dims/BoxSize
    
    # define arrays
    C     = np.zeros((3,5),dtype=np.float32) #contribution of particle to grid point
    index = np.zeros((3,5),dtype=np.int32)   #index of the grid point

    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the three axes of the particle
        for axis in xrange(3):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-2.0)
            for j in xrange(5): #only 5 cells/dimension can contribute
                index[axis,j] = (minimum+j+dims)%dims
                diff = fabs(minimum+j - dist)
                if diff<1.0:    C[axis,j] = (4.0 - 6.0*diff*diff + 3.0*diff*diff*diff)/6.0
                elif diff<2.0:  C[axis,j] = (2.0 - diff)*(2.0 - diff)*(2.0 - diff)/6.0
                else:           C[axis,j] = 0.0

        for l in xrange(5):  
            for m in xrange(5):  
                for n in xrange(5): 
                    number[index[0,l],index[1,m],index[2,n]] += C[0,l]*C[1,m]*C[2,n]*W[i]
            
    return number
################################################################################

################################################################################
# This function takes a 3D grid called density. The routine finds the CIC 
# interpolated value of the grid onto the positions input as pos
# density --> 3D array with containing the density field
# BoxSize --> Size of the box
# pos ------> positions where the density field will be interpolated
# den ------> array with the interpolated density field at pos
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t,ndim=2] CIC_interp(np.ndarray[np.float32_t,ndim=3] density,
                                                 float BoxSize,
                                                 np.ndarray[np.float32_t,ndim=2] pos,
                                                 np.ndarray[np.float32_t,ndim=1] den):
    cdef int axis,dims
    cdef long i,particles
    cdef float inv_cell_size,dist
    cdef float u[3]
    cdef float d[3]
    cdef int index_u[3]
    cdef int index_d[3]
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  dims = len(density);  inv_cell_size = dims/BoxSize

    # do a loop over all particles
    for i in xrange(particles):

        # $: grid point, X: particle position
        # $.........$..X......$
        # ------------>         dist    (1.3)
        # --------->            index_d (1)
        # --------------------> index_u (2)
        #           --->        u       (0.3)
        #              -------> d       (0.7)
        for axis in xrange(3):
            dist          = pos[i,axis]*inv_cell_size
            u[axis]       = dist - <int>dist
            d[axis]       = 1.0 - u[axis]
            index_d[axis] = (<int>dist)%dims
            index_u[axis] = index_d[axis] + 1
            index_u[axis] = index_u[axis]%dims #seems this is faster

        den[i] = density[index_d[0],index_d[1],index_d[2]]*d[0]*d[1]*d[2]+\
                 density[index_d[0],index_d[1],index_u[2]]*d[0]*d[1]*u[2]+\
                 density[index_d[0],index_u[1],index_d[2]]*d[0]*u[1]*d[2]+\
                 density[index_d[0],index_u[1],index_u[2]]*d[0]*u[1]*u[2]+\
                 density[index_u[0],index_d[1],index_d[2]]*u[0]*d[1]*d[2]+\
                 density[index_u[0],index_d[1],index_u[2]]*u[0]*d[1]*u[2]+\
                 density[index_u[0],index_u[1],index_d[2]]*u[0]*u[1]*d[2]+\
                 density[index_u[0],index_u[1],index_u[2]]*u[0]*u[1]*u[2]
################################################################################
