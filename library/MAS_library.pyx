import numpy as np 
import time,sys,os
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,floor,fabs


################################################################################
################################# ROUTINES #####################################
# MA(pos,number,BoxSize,MAS='CIC',W=None) ---> main routine
# NGP(pos,number,BoxSize)
# CIC(pos,number,BoxSize)
# TSC(pos,number,BoxSize)
# PCS(pos,number,BoxSize)
# NGPW(pos,number,BoxSize,W)
# CICW(pos,number,BoxSize,W)
# TSCW(pos,number,BoxSize,W)
# PCSW(pos,number,BoxSize,W)
# CIC_interp(pos,density,BoxSize,dens)
# TO-DO: 2D computations are suboptimal for CIC,TSC and PCS as particles along
# the axis 2 are repeated 2,3 and 4 times, respectively
################################################################################
################################################################################

# This is the main function to use when performing the mass assignment
# pos --------> array containing the positions of the particles: 2D or 3D
# numbers ----> array containing the density field: 2D or 3D
# BoxSize ----> size of the simulation box
# MAS --------> mass assignment scheme: NGP, CIC, TSC or PCS
# W ----------> array containing the weights to be used: 1D array (optional)
def MA(pos, number, BoxSize, MAS='CIC', W=None):

    #number of coordinates to work in 2D or 3D
    coord,coord_aux = len(pos[0]), len(number.shape) 

    # check that the number of dimensions match
    if coord!=coord_aux:
        print 'pos have %d dimensions and the density %d!!!'%(coord,coord_aux)
        sys.exit()

    if W is None:  print '\nUsing %s mass assignment scheme'%MAS;
    else:          print '\nUsing %s mass assignment scheme with weights'%MAS;
    start = time.clock()
    if coord==3: 
        if   MAS=='NGP' and W is None:  NGP(pos,number,BoxSize)
        elif MAS=='CIC' and W is None:  CIC(pos,number,BoxSize)
        elif MAS=='TSC' and W is None:  TSC(pos,number,BoxSize)
        elif MAS=='PCS' and W is None:  PCS(pos,number,BoxSize)
        elif MAS=='NGP' and W is not None:  NGPW(pos,number,BoxSize,W)
        elif MAS=='CIC' and W is not None:  CICW(pos,number,BoxSize,W)
        elif MAS=='TSC' and W is not None:  TSCW(pos,number,BoxSize,W)
        elif MAS=='PCS' and W is not None:  PCSW(pos,number,BoxSize,W)
        else:
            print 'option not valid!!!';  sys.exit()

    if coord==2:
        number2 = np.expand_dims(number,axis=2)
        if   MAS=='NGP' and W is None:  
            NGP(pos,number2,BoxSize)
        elif MAS=='CIC' and W is None:  
            CIC(pos,number2,BoxSize);  number2 /= 2.0
        elif MAS=='TSC' and W is None:  
            TSC(pos,number2,BoxSize);  number2 /= 3.0
        elif MAS=='PCS' and W is None:  
            PCS(pos,number2,BoxSize);  number2 /= 4.0
        elif MAS=='NGP' and W is not None:  
            NGPW(pos,number2,BoxSize,W)
        elif MAS=='CIC' and W is not None:  
            CICW(pos,number2,BoxSize,W);  number2 /= 2.0
        elif MAS=='TSC' and W is not None:  
            TSCW(pos,number2,BoxSize,W);  number2 /= 3.0
        elif MAS=='PCS' and W is not None:  
            PCSW(pos,number2,BoxSize,W);  number2 /= 4.0
        else:
            print 'option not valid!!!';  sys.exit()
        number = number2[:,:,0]
    print 'Time taken = %.3f seconds\n'%(time.clock()-start)
    

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def CIC(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):
        
    cdef int axis,dims,coord
    cdef long i,particles
    cdef float inv_cell_size,dist
    cdef float u[3]
    cdef float d[3]
    cdef int index_u[3]
    cdef int index_d[3]

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize
    
    # when computing things in 2D, use the index_ud[2]=0 plane
    for i in xrange(3):
        index_d[i] = 0;  index_u[i] = 0;  d[i] = 1.0;  u[i] = 1.0

    # do a loop over all particles
    for i in xrange(particles):

        # $: grid point, X: particle position
        # $.........$..X......$
        # ------------>         dist    (1.3)
        # --------->            index_d (1)
        # --------------------> index_u (2)
        #           --->        u       (0.3)
        #              -------> d       (0.7)
        for axis in xrange(coord):
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
def CICW(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize,
         np.float32_t[:] W):

    cdef int axis,dims,coord
    cdef long i,particles
    cdef float inv_cell_size,dist
    cdef float u[3]
    cdef float d[3]
    cdef int index_d[3]
    cdef int index_u[3]
    
    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize
    
    # when computing things in 2D, use the index_ud[2]=0 plane
    for i in xrange(3):
        index_d[i] = 0;  index_u[i] = 0;  d[i] = 1.0;  u[i] = 1.0

    # do a loop over all particles
    for i in xrange(particles):

        for axis in xrange(coord):
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
################################################################################

################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def NGP(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):

    cdef int axis,dims,coord
    cdef long i,particles
    cdef float inv_cell_size
    cdef int index[3]

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize

    # when computing things in 2D, use the index[2]=0 plane
    for i in xrange(3):  index[i] = 0

    # do a loop over all particles
    for i in xrange(particles):
        for axis in xrange(coord):
            index[axis] = <int>(pos[i,axis]*inv_cell_size + 0.5)
            index[axis] = index[axis]%dims
        number[index[0],index[1],index[2]] += 1.0
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
def NGPW(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize,
        np.float32_t[:] W):

    cdef int axis,dims,coord
    cdef long i,particles
    cdef float inv_cell_size
    cdef int index[3]

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize

    # when computing things in 2D, use the index[2]=0 plane
    for i in xrange(3):  index[i] = 0

    # do a loop over all particles
    for i in xrange(particles):
        for axis in xrange(coord):
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
def TSC(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):

    cdef int axis, dims, minimum
    cdef int j, l, m, n, coord
    cdef long i, particles
    cdef float inv_cell_size, dist, diff
    cdef float C[3][3]
    cdef int index[3][3]

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize
    
    # define arrays: for 2D set we have C[2,:] = 1.0 and index[2,:] = 0
    for i in xrange(3):
        for j in xrange(3):
            C[i][j] = 1.0;  index[i][j] = 0
            
    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the axes of the particle
        for axis in xrange(coord):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-1.5)
            for j in xrange(3): #only 3 cells/dimension can contribute
                index[axis][j] = (minimum+j+1+dims)%dims
                diff = fabs(minimum + j+1 - dist)
                if diff<0.5:    C[axis][j] = 0.75-diff*diff
                elif diff<1.5:  C[axis][j] = 0.5*(1.5-diff)*(1.5-diff)
                else:           C[axis][j] = 0.0

        for l in xrange(3):  
            for m in xrange(3):  
                for n in xrange(3): 
                    number[index[0][l],index[1][m],index[2][n]] += C[0][l]*C[1][m]*C[2][n]
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
def TSCW(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize,
         np.float32_t[:] W):

    cdef int axis,dims,minimum,j,l,m,n,coord
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef float C[3][3]
    cdef int index[3][3]

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize
    
    # define arrays: for 2D set we have C[2,:] = 1.0 and index[2,:] = 0
    for i in xrange(3):
        for j in xrange(3):
            C[i][j] = 1.0;  index[i][j] = 0
    
    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the three axes of the particle
        for axis in xrange(coord):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-1.5)
            for j in xrange(3): #only 3 cells/dimension can contribute
                index[axis][j] = (minimum+ j+1+ dims)%dims
                diff = fabs(minimum+ j+1 - dist)
                if diff<0.5:    C[axis][j] = 0.75-diff*diff
                elif diff<1.5:  C[axis][j] = 0.5*(1.5-diff)*(1.5-diff)
                else:           C[axis][j] = 0.0

        for l in xrange(3):  
            for m in xrange(3):  
                for n in xrange(3): 
                    number[index[0][l],index[1][m],index[2][n]] += C[0][l]*C[1][m]*C[2][n]*W[i]
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def PCS(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):

    cdef int axis,dims,minimum,j,l,m,n,coord
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef float C[3][4]
    cdef int index[3][4]

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize
        
    # define arrays: for 2D set we have C[2,:] = 1.0 and index[2,:] = 0
    for i in xrange(3):
        for j in xrange(4):
            C[i][j] = 1.0;  index[i][j] = 0

    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the three axes of the particle
        for axis in xrange(coord):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-2.0)
            for j in xrange(4): #only 4 cells/dimension can contribute
                index[axis][j] = (minimum + j+1 + dims)%dims
                diff = fabs(minimum + j+1 - dist)
                if diff<1.0:    C[axis][j] = (4.0 - 6.0*diff*diff + 3.0*diff*diff*diff)/6.0
                elif diff<2.0:  C[axis][j] = (2.0 - diff)*(2.0 - diff)*(2.0 - diff)/6.0
                else:           C[axis][j] = 0.0

        for l in xrange(4):  
            for m in xrange(4):  
                for n in xrange(4): 
                    number[index[0][l],index[1][m],index[2][n]] += C[0][l]*C[1][m]*C[2][n]
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
def PCSW(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize,
         np.float32_t[:] W):

    cdef int axis,dims,minimum,j,l,m,n,coord
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef float C[3][4]
    cdef int index[3][4]

    # find number of particles, the inverse of the cell size and dims
    particles = len(pos);  coord = len(pos[0]);  dims = len(number);  
    inv_cell_size = dims/BoxSize

    # define arrays: for 2D set we have C[2,:] = 1.0 and index[2,:] = 0
    for i in xrange(3):
        for j in xrange(4):
            C[i][j] = 1.0;  index[i][j] = 0
    
    # do a loop over all particles
    for i in xrange(particles):

        # do a loop over the three axes of the particle
        for axis in xrange(coord):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-2.0)
            for j in xrange(4): #only 4 cells/dimension can contribute
                index[axis][j] = (minimum + j+1 + dims)%dims
                diff = fabs(minimum + j+1 - dist)
                if diff<1.0:    C[axis][j] = (4.0 - 6.0*diff*diff + 3.0*diff*diff*diff)/6.0
                elif diff<2.0:  C[axis][j] = (2.0 - diff)*(2.0 - diff)*(2.0 - diff)/6.0
                else:           C[axis][j] = 0.0

        for l in xrange(4):
            for m in xrange(4):
                for n in xrange(4): 
                    number[index[0][l],index[1][m],index[2][n]] += C[0][l]*C[1][m]*C[2][n]*W[i]
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
def CIC_interp(np.ndarray[np.float32_t,ndim=3] density, float BoxSize,
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
