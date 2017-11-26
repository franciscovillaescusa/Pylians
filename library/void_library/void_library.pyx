import numpy as np
import sys,os,time
import Pk_library as PKL
import units_library as UL
import pyfftw
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt,pow,sin,cos,log10,fabs
from libc.stdlib cimport malloc, free

# The function takes a density field and smooth it with a 3D top-hat filter
# of radius R:  W = 3/(4*pi*R^3) if r<R;  W = 0  otherwise
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def gaussian_smoothing(delta, float BoxSize, float R, int threads=1):
                       
    
    cdef int dims = delta.shape[0]
    cdef int middle = dims/2
    cdef float prefact,kR,fact
    cdef int kxx, kyy, kzz, kx, ky, kz
    cdef np.complex64_t[:,:,::1] delta_k

    ## compute FFT of the field (change this for double precision) ##
    delta_k = PKL.FFT3Dr_f(delta,threads) 

    # do a loop over the independent modes.
    prefact = 2.0*np.pi/BoxSize
    for kxx in prange(dims, nogil=True):
        kx = (kxx-dims if (kxx>middle) else kxx)
        
        for kyy in xrange(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            
            for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)

                if kxx==0 and kyy==0 and kzz==0:
                    continue

                # compute the value of |k|
                kR = prefact*sqrt(kx*kx + ky*ky + kz*kz)*R
                if fabs(kR)<1e-5:  fact = 1.0
                else:             fact = 3.0*(sin(kR) - cos(kR)*kR)/(kR*kR*kR)
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*fact
                                       
    # Fourier transform back
    return PKL.IFFT3Dr_f(delta_k,threads)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
class void_finder:
    def __init__(self, np.ndarray[np.float32_t, ndim=3] delta, float BoxSize, 
        float threshold, float Rmax, float Rmin, float Omega_m, int threads):

        cdef float R = Rmax
        cdef float dist, R_gridx, rho_crit, mean_rho, pi
        cdef long Nvoids = 0
        cdef long max_num_voids,local_voids,ID,dims3
        cdef int i,j,k,p,dims2,Ncells,l,m,n,i1,j1,k1,near_voids
        cdef int dims = delta.shape[0]
        cdef int[:,:,::1] in_void
        cdef np.ndarray[np.float32_t, ndim=2] delta_v
        cdef float[:] Radii
        cdef float[:,:,::1] delta_sm
        cdef list void_pos, void_mass, void_radius

        dims2 = dims**2;  dims3 = dims**3
        pi = np.pi

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
        delta_v = np.zeros((dims3,2),        dtype=np.float32)

        Radii = np.logspace(np.log10(Rmax),np.log10(Rmin), 20, dtype=np.float32)


        for R in Radii:

            # smooth the density field with a top-hat radius of R
            start = time.time()
            print 'Smoothing field with top-hat filter of radius %.2f'%R
            delta_sm = gaussian_smoothing(delta, BoxSize, R, threads)
            print '%.3f < delta < %.3f'%(np.min(delta_sm), np.max(delta_sm))
            print '<delta> = %.3f'%(np.mean(delta, dtype=np.float64))
            if np.min(delta_sm)>threshold:
                print 'No cells with delta < %.2f\n'%threshold
                continue
            print 'Density smoothing took %.3f seconds'%(time.time()-start)

            # find cells with delta<threshold and not belonging to existing voids
            start = time.time()
            local_voids = 0
            for i in xrange(dims):
                for j in xrange(dims):
                    for k in xrange(dims):

                        if delta_sm[i,j,k]<threshold and in_void[i,j,k]==0:
                            delta_v[local_voids,0] = delta_sm[i,j,k]
                            delta_v[local_voids,1] = dims2*i+dims*j+k
                            local_voids += 1

            print 'Searching underdense cells took %.3f seconds'%(time.time()-start)
            print 'Found %08d cells below threshold'%(local_voids)

            # sort the cell underdensities
            start = time.time()
            delta_v[:local_voids].sort(axis=0)
            print 'Sorting took %.3f seconds'%(time.time()-start)
            

            # do a loop over all underdense cells and identify voids
            start = time.time()
            Ncells = <int>((R/BoxSize)*1.0*dims) + 1
            R_grid = (R/BoxSize)*1.0*dims
            for p in xrange(local_voids):
                near_voids = 0
                ID = <int>delta_v[p,1]
                i,j,k = ID/dims2, (ID%dims2)/dims, (ID%dims2)%dims
                
                if in_void[i,j,k] == 1:
                    continue
                    
                for l in prange(-Ncells,Ncells+1, nogil=True):
                    i1 = (i+l+dims)%dims
                    for m in xrange(-Ncells,Ncells+1):
                        j1 = (j+m+dims)%dims
                        for n in xrange(-Ncells,Ncells+1):
                            k1 = (k+n+dims)%dims
                                
                            dist = sqrt(l*l + m*m + n*n)
                            if dist<R_grid and in_void[i1,j1,k1]==1:
                                near_voids += 1
                                             
                                     
                if near_voids==0:
                    Nvoids += 1
                    void_pos.append([i*BoxSize/dims, j*BoxSize/dims, k*BoxSize/dims])
                    void_radius.append(R)
                    void_mass.append(4.0/3.0*pi*R**3*(1.0+threshold)*mean_rho)
                    in_void[i,j,k] = 1
                        
                    for l in prange(-Ncells,Ncells+1, nogil=True):
                        i1 = (i+l+dims)%dims
                        for m in xrange(-Ncells,Ncells+1):
                            j1 = (j+m+dims)%dims
                            for n in xrange(-Ncells,Ncells+1):
                                k1 = (k+n+dims)%dims
                                
                                dist = sqrt(l*l + m*m + n*n)
                                if dist<R_grid:
                                    in_void[i1,j1,k1] = 1
            

            print 'void finding took %.3f seconds\n'%(time.time()-start)     

        print 'Void volume filling fraction = %.3f'\
            %(np.sum(in_void, dtype=np.int64)*1.0/dims3)
        print 'Found %d voids\n'%Nvoids

        self.in_void     = np.asarray(in_void)
        self.void_pos    = np.array(void_pos)
        self.void_mass   = np.array(void_mass)
        self.void_radius = np.array(void_radius)
