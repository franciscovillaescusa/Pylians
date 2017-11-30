import numpy as np
import sys,os,time
import Pk_library as PKL
import units_library as UL
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt,pow,sin,cos,log,log10,fabs,round
from libc.stdlib cimport malloc, free


############################### ROUTINES ####################################
# V = void_finder(delta, BoxSize, threshold, Rmax, Rmin, bins, Omega_m, 
#               threads, void_field=False)
# V.void_pos  ----> void positions
# V.void_mass ----> void masses
# V.void_radius --> void radii  
# V.Rbins --------> void mass function (R bins)
# V.void_mf ------> void mass function (voids/(Volume*log(R)))
# V.in_void ------> grid with 0 and 1. 1 is a cell within a void (optional)

# gaussian_smoothing(delta, BoxSize, R, threads)

# V = void_safety_check(delta, void_pos, void_radius, BoxSize, Omega_m)
# V.mean_overdensity ----> mean overdensity of the cells in a void
# V.mean_radius ---------> effective radius of the void from the cells in it
# V.mean_mass -----------> sum of the cell masses in a void

# V = random_spheres(BoxSize, Rmin, Rmax, Nvoids, dims)
# V.void_pos ------> array with the positions of the input voids
# V.void_radius ---> array with the radii of the input void
# V.delta ---------> density field with zeros everywhere except in voids
#############################################################################


# The function takes a density field and smooth it with a 3D top-hat filter
# of radius R:  W = 3/(4*pi*R^3) if r<R;  W = 0  otherwise
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def gaussian_smoothing(delta, float BoxSize, float R, int threads=1):
                       
    
    cdef int dims = delta.shape[0]
    cdef int middle = dims/2
    cdef float prefact,kR,fact
    cdef int kxx, kyy, kzz, kx, ky, kz, kx2, ky2, kz2
    cdef np.complex64_t[:,:,::1] delta_k

    ## compute FFT of the field (change this for double precision) ##
    delta_k = PKL.FFT3Dr_f(delta,threads) 

    # do a loop over the independent modes.
    prefact = R*2.0*np.pi/BoxSize
    for kxx in prange(dims, nogil=True):
        kx  = (kxx-dims if (kxx>middle) else kxx)
        kx2 = kx*kx
        
        for kyy in xrange(dims):
            ky  = (kyy-dims if (kyy>middle) else kyy)
            ky2 = ky*ky
            
            for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz  = (kzz-dims if (kzz>middle) else kzz)
                kz2 = kz*kz

                if kxx==0 and kyy==0 and kzz==0:
                    continue

                # compute the value of |k|
                kR = prefact*sqrt(kx2 + ky2 + kz2)
                if fabs(kR)<1e-5:  fact = 1.0
                else:              fact = 3.0*(sin(kR) - cos(kR)*kR)/(kR*kR*kR)
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*fact
                                       
    # Fourier transform back
    return PKL.IFFT3Dr_f(delta_k,threads)

# This routine finds voids in the density field delta. The input values needed
# are the box size, the maximum and minimum radii, the number of bins, the
# density threshold, Omega_m (to compute void masses) and the threads number
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
class void_finder:
    def __init__(self, np.ndarray[np.float32_t, ndim=3] delta, float BoxSize, 
        float threshold, float Rmax, float Rmin, int bins, 
        float Omega_m, int threads, void_field=False):

        cdef float R, dist2, R_grid, R_grid2, rho_crit, mean_rho, pi
        cdef long voids_found, total_voids_found
        cdef long max_num_voids,local_voids,ID,dims3
        cdef int i,j,k,p,q,Ncells,l,m,n,i1,j1,k1
        cdef int dims, dims2, cells_in_other_void
        cdef int[:,:,::1] in_void
        cdef np.ndarray[np.float32_t, ndim=1] delta_v
        cdef np.ndarray[np.int64_t, ndim=1] indexes, IDs
        cdef float[:] Radii,mf
        cdef int[:] Nvoids
        cdef float[:,:,::1] delta_sm
        cdef list void_pos, void_mass, void_radius

        dims = delta.shape[0]
        dims2 = dims**2;  dims3 = dims**3;  pi = np.pi

        # check that Rmin is larger than the grid resolution
        if Rmin<BoxSize*1.0/dims:
            raise Exception("Rmin=%.3f Mpc/h below grid resolution=%.3f Mpc/h"\
                            %(Rmin, BoxSize*1.0/dims))

        # find the value of the mean matter density of the Universe
        rho_crit = (UL.units()).rho_crit #h^2 Msun/Mpc^3
        mean_rho = rho_crit*Omega_m

        # define list containing void positions, radii and masses
        void_pos = [];  void_mass = [];  void_radius = []

        # find the maximum possible number of voids
        max_num_voids = int(BoxSize**3/(4.0*pi/3.0*Rmin**3))
        print 'maximum number of voids = %d\n'%max_num_voids
        
        # define the in_void and delta_v array
        in_void = np.zeros((dims,dims,dims), dtype=np.int32)
        delta_v = np.zeros(dims3,            dtype=np.float32)
        IDs     = np.zeros(dims3,            dtype=np.int64)

        Radii = np.logspace(np.log10(Rmax),np.log10(Rmin), bins, 
                dtype=np.float32)
        Nvoids = np.zeros(bins,   dtype=np.int32)
        mf     = np.zeros(bins-1, dtype=np.float32)

        total_voids_found = 0
        for q in xrange(bins):

            # Get the smoothing length
            R = Radii[q]

            # smooth the density field with a top-hat radius of R
            start = time.time()
            print 'Smoothing field with top-hat filter of radius %.2f'%R
            delta_sm = gaussian_smoothing(delta, BoxSize, R, threads)
            print '%.3f < delta < %.3f  :  <delta> = %.3f'\
                %(np.min(delta_sm), np.max(delta_sm), np.mean(delta))
            print '<delta_sm^2> =%.3e  :  <delta^2>=%.3e'\
                %(np.std(delta_sm),np.std(delta))
            print 'Density smoothing took %.3f seconds'%(time.time()-start)
            if np.min(delta_sm)>threshold:
                print 'No cells with delta < %.2f\n'%threshold
                continue

            # find cells with delta<threshold and not belonging to existing voids
            start = time.time()
            local_voids = 0
            for i in xrange(dims):
                for j in xrange(dims):
                    for k in xrange(dims):

                        if delta_sm[i,j,k]<threshold and in_void[i,j,k]==0:
                            IDs[local_voids]     = dims2*i + dims*j + k
                            delta_v[local_voids] = delta_sm[i,j,k]
                            local_voids += 1
            print 'Searching underdense cells took %.3f seconds'%(time.time()-start)
            print 'Found %08d cells below threshold'%(local_voids)

            # sort the cell underdensities
            start = time.time()
            indexes = np.argsort(delta_v[:local_voids])
            delta_v[:local_voids] = delta_v[:local_voids][indexes]
            IDs[:local_voids] = IDs[:local_voids][indexes]
            print 'Sorting took %.3f seconds'%(time.time()-start)

            # do a loop over all underdense cells and identify voids
            start   = time.time()
            R_grid  = (R/BoxSize)*1.0*dims;  Ncells = <int>R_grid + 1
            R_grid2 = R_grid*R_grid
            voids_found = 0
            for p in xrange(local_voids):

                # find the grid coordinates of the underdense cell
                ID = IDs[p]
                i,j,k = ID/dims2, (ID%dims2)/dims, (ID%dims2)%dims

                # if cell belong to a void continue
                if in_void[i,j,k] == 1:  continue
                    
                # check that all cells in the void are not in other void
                cells_in_other_void = 0
                for l in prange(-Ncells,Ncells+1, nogil=True):
                    i1 = (i+l+dims)%dims
                    for m in xrange(-Ncells,Ncells+1):
                        j1 = (j+m+dims)%dims
                        for n in xrange(-Ncells,Ncells+1):
                            k1 = (k+n+dims)%dims
                                
                            dist2 = l*l + m*m + n*n
                            if dist2<R_grid2 and in_void[i1,j1,k1]==1:
                                cells_in_other_void += 1
                                                 
                # a void is found
                if cells_in_other_void==0:

                    voids_found += 1;  total_voids_found += 1
                    void_pos.append([i*BoxSize/dims, j*BoxSize/dims, k*BoxSize/dims])
                    void_radius.append(R)
                    void_mass.append(4.0/3.0*pi*R**3*(1.0+threshold)*mean_rho)

                    in_void[i,j,k] = 1                        
                    for l in prange(-Ncells, Ncells+1, nogil=True):
                        i1 = (i+l+dims)%dims
                        for m in xrange(-Ncells, Ncells+1):
                            j1 = (j+m+dims)%dims
                            for n in xrange(-Ncells, Ncells+1):
                                k1 = (k+n+dims)%dims
                                
                                dist2 = l*l + m*m + n*n
                                if dist2<R_grid2:  in_void[i1,j1,k1] = 1
            
            print 'Found %06d voids with radius R=%.3f Mpc/h'%(voids_found, R)
            print 'void finding took %.3f seconds\n'%(time.time()-start)  
            Nvoids[q] = voids_found   

        print 'Void volume filling fraction = %.3f'\
            %(np.sum(in_void, dtype=np.int64)*1.0/dims3)
        print 'Found a total of %d voids\n'%total_voids_found

        # compute the void mass function (voids/Volume/log(R))
        for i in xrange(bins-1):
            mf[i] = Nvoids[i]/(BoxSize**3*log(Radii[i+1]-Radii[i]))

        if void_field:
            self.in_void = np.asarray(in_void)
        self.void_pos    = np.asarray(void_pos,       dtype=np.float32)
        self.void_mass   = np.asarray(void_mass,      dtype=np.float32)
        self.void_radius = np.asarray(void_radius,    dtype=np.float32)
        self.Rbins       = np.asarray(Radii[:bins-1], dtype=np.float32)
        self.void_mf     = mf

# This routine takes the density field, the void positions and radii
# and finds the cells inside each void. From those cells it computes
# the mean overdensity, the effective radius and the mass of the void
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
class void_safety_check:
    def __init__(self, float[:,:,:] delta, float[:,:] void_pos, 
        float[:] void_radius, float BoxSize, float Omega_m):

        cdef long number_of_voids, p
        cdef int dims, i, j, k, Nshells, ii, jj, kk, i1, j1, k1
        cdef float dist, R_void, prefact, ratio, pi, V_cell, mean_rho
        cdef double[:] mean_overdensity, mean_mass, mean_radius
        cdef long[:] cells

        # find the number of voids and the number of cells in the field
        number_of_voids = void_pos.shape[0]
        dims            = delta.shape[0]

        # define the array containing the mean overdensities
        mean_overdensity = np.zeros(number_of_voids, dtype=np.float64)
        mean_mass        = np.zeros(number_of_voids, dtype=np.float64)
        mean_radius      = np.zeros(number_of_voids, dtype=np.float64)
        cells            = np.zeros(number_of_voids, dtype=np.int64)

        # compute the mean matter density
        mean_rho = Omega_m*(UL.units()).rho_crit #h^2 Msun/Mpc^3

        pi      = np.pi
        prefact = dims*1.0/BoxSize
        V_cell  = (BoxSize*1.0/dims)**3


        # do a loop over all the voids
        for p in xrange(number_of_voids):

            # find void coordinate in grid units
            i = <int>round(void_pos[p,0]*prefact)
            j = <int>round(void_pos[p,1]*prefact)
            k = <int>round(void_pos[p,2]*prefact)

            # compute the void radius in grid cell units
            R_void  = void_radius[p]*prefact
            Nshells = <int>R_void+1

            for ii in xrange(-Nshells, Nshells+1):
                i1 = (ii+i+dims)%dims
                for jj in xrange(-Nshells, Nshells+1):
                    j1 = (jj+j+dims)%dims
                    for kk in xrange(-Nshells, Nshells+1):
                        k1 = (kk+k+dims)%dims

                        dist = sqrt(ii*ii + jj*jj + kk*kk)
                        if dist<R_void:
                            mean_overdensity[p] += delta[i1,j1,k1]
                            cells[p] += 1

            # compute the mean overdensity of the cells in the void
            mean_overdensity[p] = mean_overdensity[p]*1.0/cells[p]

            # compute volume occupied by cells in void and effective radius
            mean_radius[p] = (3.0*cells[p]*V_cell/(4.0*pi))**(1.0/3.0)

            # compute the mean mass of the void
            mean_mass[p] = (1.0+mean_overdensity[p])*cells[p]*V_cell*mean_rho


        self.mean_overdensity = np.asarray(mean_overdensity, dtype=np.float32)
        self.mean_radius      = np.asarray(mean_radius,      dtype=np.float32)
        self.mean_mass        = np.asarray(mean_mass,        dtype=np.float32)


# This routine creates a density field filled with 0. It then places random
# spheres (not overlapping) with a profile delta(r) = -1*(1-(r/R)^3)
# It returns that density field. A void finder can be run a the identified
# voids can be compared with the input ones.
# For this density profile, the average delta within R is -0.5
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
class random_spheres:
    def __init__(self, float BoxSize, float Rmin, float Rmax, int Nvoids,
        int dims):

        cdef int new_void
        cdef int i, j, l, m, n, Nshells, ii, i1, jj, j1, kk, k1
        cdef float R, diff_x, diff_y, diff_z
        cdef float[:,:,::1] delta
        cdef float[::1] pos, radii
        cdef float [:,::1] positions

        # define the density field
        delta = np.zeros((dims,dims,dims), dtype=np.float32)

        # define the arrays with the positions and radii of the voids
        positions = np.zeros((Nvoids,3), dtype=np.float32)
        radii     = np.zeros(Nvoids,     dtype=np.float32)

        # do a loop over all the voids
        for i in xrange(Nvoids):

            while(1):

                new_void = 1

                # generate the position and radius of new void
                pos = np.random.random(3).astype(np.float32)*BoxSize
                R   = np.random.random(1).astype(np.float32)*(Rmax-Rmin) + Rmin

                for j in xrange(i):

                    diff_x = pos[0]-positions[j,0]
                    if diff_x<-BoxSize/2.0:  diff_x += BoxSize
                    if diff_x>BoxSize/2.0:   diff_x -= BoxSize

                    diff_y = pos[1]-positions[j,1]
                    if diff_y<-BoxSize/2.0:  diff_y += BoxSize
                    if diff_y>BoxSize/2.0:   diff_y -= BoxSize

                    diff_z = pos[2]-positions[j,2]
                    if diff_z<-BoxSize/2.0:  diff_z += BoxSize
                    if diff_z>BoxSize/2.0:   diff_z -= BoxSize

                    dist = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)

                    if dist<R+radii[j]:
                        new_void = 0
                        break

                if new_void==1:  
                    positions[i] = pos
                    radii[i]     = R
                    break

            # find the position and radius of the void in grid units
            l       = <int>(positions[i,0]*dims*1.0/BoxSize)
            m       = <int>(positions[i,1]*dims*1.0/BoxSize)
            n       = <int>(positions[i,2]*dims*1.0/BoxSize)
            Nshells = <int>(radii[i]*dims*1.0/BoxSize)

            for ii in xrange(-Nshells, Nshells+1):
                i1 = (ii+l+dims)%dims
                for jj in xrange(-Nshells, Nshells+1):
                    j1 = (jj+m+dims)%dims
                    for kk in xrange(-Nshells, Nshells+1):
                        k1 = (kk+n+dims)%dims

                        dist = sqrt(ii**2 + jj**2 + kk**2)*BoxSize*1.0/dims

                        if dist<R:
                            delta[i1,j1,k1]=-1.0*(1.0-(dist/R)**3)

        self.void_pos    = np.asarray(positions)
        self.void_radius = np.asarray(radii)
        self.delta       = np.asarray(delta)
