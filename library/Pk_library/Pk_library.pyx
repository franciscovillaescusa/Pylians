import numpy as np 
import time,sys,os
import pyfftw
import scipy.integrate as si
cimport numpy as np
cimport cython
#from cython.parallel import prange
from libc.math cimport sqrt,pow,sin,log10,abs
from libc.stdlib cimport malloc, free

################################ ROUTINES ####################################
# Pk(delta, BoxSize, axis=2, MAS='CIC', threads=1)
#   1D: k1D[k]   Pk1D[k]    Nmodes1D[k] 
#   2D: kpar[k]  kper[k]    Pk2D[k]   Nmodes2D[k]
#   3D: k3D[k]   Pk[k,ell]  Nmodes3D[k] ===> Pk0 = Pk[:,0];  Pk2 = Pk[:,1]

# XPk([delta1,delta2], BoxSize, axis=2, MAS=['CIC','CIC'], threads=1)
#   1D: k1D[k]  Pk1D[k,field]   PkX1D[k,field]   Nmodes1D[k]
#   2D: kpar[k] kper[k]         Pk2D[k,field]    PkX2D[k,field]  Nmodes2D[k]
#   3D: k3D[k]  Pk[k,ell,field] XPk[k,ell,field] Nmodes3D[k]

# Pk_theta(Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk_theta,Nmodes]

# XPk_dv(delta,Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk1,Pk2,PkX,Nmodes]

# XPk_2D(delta1,delta2,BoxSize,axis=2,MAS1='CIC',MAS2='CIC',threads=1)
#   [kpar,kper,Pk1,Pk2,PkX,Nmodes]
 
# Pk_1D_from_3D(delta,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk_1D,Pk_3D]

# Pk_1D_from_2D(delta,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k_1D,Pk_1D,kpar_2D,kper_2D,Pk2D,Nmodes_2D]

# correct_MAS(delta,BoxSize,MAS='CIC',threads=1)
#   delta_corrected
##############################################################################

# This function determines the fundamental (kF) and Nyquist (kN) frequencies
# It also finds the maximum frequency sampled in the box, the maximum 
# frequency along the parallel and perpendicular directions in units of kF
def frequencies(BoxSize,dims):
    kF = 2.0*np.pi/BoxSize;  middle = dims/2;  kN = middle*kF
    kmax_par = middle
    kmax_per = int(np.sqrt(middle**2 + middle**2))
    kmax     = int(np.sqrt(middle**2 + middle**2 + middle**2))
    return kF,kN,kmax_par,kmax_per,kmax
    
# This function finds the MAS correction index and return the array used
def MAS_function(MAS):
    MAS_index = 0;  #MAS_corr = np.ones(3,dtype=np.float64)
    if MAS=='NGP':  MAS_index = 1
    if MAS=='CIC':  MAS_index = 2
    if MAS=='TSC':  MAS_index = 3
    if MAS=='PCS':  MAS_index = 4
    return MAS_index#,MAS_corr

# This function implement the MAS correction to modes amplitude
#@cython.cdivision(False)
#@cython.boundscheck(False)
cpdef inline double MAS_correction(double x, int MAS_index):
    return (1.0 if (x==0.0) else pow(x/sin(x),MAS_index))

# This function checks that all independent modes have been counted
def check_number_modes(Nmodes,dims):
    # (0,0,0) own antivector, while (n,n,n) has (-n,-n,-n) for dims odd
    if dims%2==1:  own_modes = 1 
    # (0,0,0),(0,0,n),(0,n,0),(n,0,0),(n,n,0),(n,0,n),(0,n,n),(n,n,n)
    else:          own_modes = 8 
    repeated_modes = (dims**3 - own_modes)/2  
    indep_modes    = repeated_modes + own_modes

    if int(np.sum(Nmodes))!=indep_modes:
        print 'WARNING: Not all modes counted'
        print 'Counted  %d independent modes'%(int(np.sum(Nmodes)))
        print 'Expected %d independent modes'%indep_modes
        sys.exit() 

# This function performs the 3D FFT of a field in single precision
def FFT3Dr_f(np.ndarray[np.float32_t,ndim=3] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')
    a_out = pyfftw.empty_aligned((dims,dims,dims/2+1),dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in double precision
def FFT3Dr_d(np.ndarray[np.float64_t,ndim=3] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float64')
    a_out = pyfftw.empty_aligned((dims,dims,dims/2+1),dtype='complex128')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in single precision
def IFFT3Dr_f(np.complex64_t[:,:,::1] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims/2+1),dtype='complex64')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in double precision
def IFFT3Dr_d(np.complex128_t[:,:,::1] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims/2+1),dtype='complex128')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out


################################################################################
################################################################################
# This routine computes the 1D, 2D & 3D P(k) of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
# P_1D(k_par) = \int d^2k_per/(2pi)^2 P_3D(k_par,k_per)
# we approximate the 2D integral by the average of the modes sampled by the 
# field and we carry it out using k_per modes such as |k|<kN. The perpendicular
# modes sample the circle in an almost uniform way, thus, we approximate the 
# integral by a sum of the amplitude of each mode times the area it covers, 
# which is pi*kmax_per^2/Nmodes, where kmax_per=sqrt(kN^2-kpar^2)
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
class Pk:
    def __init__(self,delta,BoxSize,int axis=2,MAS='CIC',threads=1):

        start = time.time()
        cdef int kxx, kyy, kzz, kx, ky, kz,dims, middle, k_index, MAS_index
        cdef int kmax_par, kmax_per, kmax, k_par, k_per, index_2D, i
        cdef double k, delta2, prefact, mu, mu2, real, imag, kmaxper
        cdef double MAS_corr[3]
        ####### change this for double precision ######
        cdef float MAS_factor
        cdef np.complex64_t[:,:,::1] delta_k
        ###############################################
        cdef np.float64_t[::1] k1D, kpar, kper, k3D, Pk1D, Pk2D
        cdef np.float64_t[::1] Nmodes1D, Nmodes2D, Nmodes3D
        cdef np.float64_t[:,::1] Pk3D 

        # find dimensions of delta: we assume is a (dims,dims,dims) array
        # determine the different frequencies and the MAS_index
        print '\nComputing power spectrum of the field...'
        dims = len(delta);  middle = dims/2
        kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
        MAS_index = MAS_function(MAS)

        ## compute FFT of the field (change this for double precision) ##
        delta_k = FFT3Dr_f(delta,threads)
        #################################

        # define arrays containing k1D, Pk1D and Nmodes1D. We need kmax_par+1
        # bins since modes go from 0 to kmax_par
        k1D      = np.zeros(kmax_par+1, dtype=np.float64)
        Pk1D     = np.zeros(kmax_par+1, dtype=np.float64)
        Nmodes1D = np.zeros(kmax_par+1, dtype=np.float64)

        # define arrays containing Pk2D and Nmodes2D
        Pk2D     = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
        Nmodes2D = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)

        # define arrays containing k3D, Pk3D and Nmodes3D. We need kmax+1
        # bins since the mode (middle,middle, middle) has an index = kmax
        k3D      = np.zeros(kmax+1,     dtype=np.float64)
        Pk3D     = np.zeros((kmax+1,3), dtype=np.float64)
        Nmodes3D = np.zeros(kmax+1,     dtype=np.float64)


        # do a loop over the independent modes.
        # compute k,k_par,k_per, mu for each mode. k's are in kF units
        start2 = time.time();  prefact = np.pi/dims
        for kxx in xrange(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
        
            for kyy in xrange(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

                for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                    # kz=0 and kz=middle planes are special
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue


                    # compute |k| of the mode and its integer part
                    k       = sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = <int>k

                    # compute the value of k_par and k_perp
                    if axis==0:   
                        k_par, k_per = kx, <int>sqrt(ky*ky + kz*kz)
                    elif axis==1: 
                        k_par, k_per = ky, <int>sqrt(kx*kx + kz*kz)
                    else:         
                        k_par, k_per = kz, <int>sqrt(kx*kx + ky*ky)

                    # find the value of mu
                    if k==0:  mu = 0.0
                    else:     mu = k_par/k
                    mu2 = mu*mu

                    # take the absolute value of k_par
                    if k_par<0:  k_par = -k_par

                    # correct modes amplitude for MAS
                    MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                    # compute |delta_k|^2 of the mode
                    real = delta_k[kxx,kyy,kzz].real
                    imag = delta_k[kxx,kyy,kzz].imag
                    delta2 = real*real + imag*imag

                    # Pk1D: only consider modes with |k|<kF
                    if k<=middle:
                        k1D[k_par]      += k_par
                        Pk1D[k_par]     += delta2
                        Nmodes1D[k_par] += 1.0

                    # Pk2D: P(k_per,k_par)
                    # index_2D goes from 0 to (kmax_par+1)*(kmax_per+1)-1
                    index_2D = (kmax_par+1)*k_per + k_par
                    Pk2D[index_2D]     += delta2
                    Nmodes2D[index_2D] += 1.0

                    # Pk3D.
                    k3D[k_index]      += k
                    Pk3D[k_index,0]   += delta2
                    Pk3D[k_index,1]   += (delta2*(3.0*mu2-1.0)/2.0)
                    Pk3D[k_index,2]   += (delta2*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
                    Nmodes3D[k_index] += 1.0
        print 'Time to complete loop = %.2f'%(time.time()-start2)

        # Pk1D. Discard DC mode bin and give units
        # the perpendicular modes sample an area equal to pi*kmax_per^2
        # we assume that each mode has an area equal to pi*kmax_per^2/Nmodes
        k1D  = k1D[1:];  Nmodes1D = Nmodes1D[1:];  Pk1D = Pk1D[1:]
        for i in xrange(len(k1D)):
            Pk1D[i] = Pk1D[i]*(BoxSize/dims**2)**3 #give units
            k1D[i]  = (k1D[i]/Nmodes1D[i])*kF      #give units
            kmaxper = sqrt(kN**2 - k1D[i]**2)
            Pk1D[i] = Pk1D[i]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
        self.k1D = np.asarray(k1D);  self.Pk1D = np.asarray(Pk1D)
        self.Nmodes1D = np.asarray(Nmodes1D  )

        # Pk2D. Keep DC mode bin, give units to Pk2D and find kpar & kper
        kpar = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
        kper = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
        for k_par in xrange(kmax_par+1):
            for k_per in xrange(kmax_per+1):
                index_2D = (kmax_par+1)*k_per + k_par
                kpar[index_2D] = 0.5*(k_par + k_par+1)*kF
                kper[index_2D] = 0.5*(k_per + k_per+1)*kF
        for i in xrange(len(kpar)):
            Pk2D[i] = Pk2D[i]*(BoxSize/dims**2)**3/Nmodes2D[i]
        self.kpar = np.asarray(kpar);  self.kper = np.asarray(kper)
        self.Pk2D = np.asarray(Pk2D);  self.Nmodes2D = np.asarray(Nmodes2D)

        # Pk3D. Check modes, discard DC mode bin and give units
        # we need to multiply the multipoles by (2*ell + 1)
        check_number_modes(Nmodes3D,dims)
        k3D  = k3D[1:];  Nmodes3D = Nmodes3D[1:];  Pk3D = Pk3D[1:,:]
        for i in xrange(len(k3D)):
            k3D[i]    = (k3D[i]/Nmodes3D[i])*kF
            Pk3D[i,0] = (Pk3D[i,0]/Nmodes3D[i])*(BoxSize/dims**2)**3
            Pk3D[i,1] = (Pk3D[i,1]*5.0/Nmodes3D[i])*(BoxSize/dims**2)**3
            Pk3D[i,2] = (Pk3D[i,2]*9.0/Nmodes3D[i])*(BoxSize/dims**2)**3
        self.k3D = np.asarray(k3D);  self.Nmodes3D = np.asarray(Nmodes3D)
        self.Pk = np.asarray(Pk3D)

        print 'Time taken = %.2f seconds'%(time.time()-start)
################################################################################
################################################################################

################################################################################
################################################################################
# This routine computes the auto- and cross-power spectra of various 
# density fields
# delta -------> list with the density fields: [delta1,delta2,delta3,...]
# each density field should be a: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight for the multipoles
# MAS ---------> list with the mass assignment scheme used to compute density 
# fields: ['CIC','CIC','PCS',...]
# threads -----> number of threads (OMP) used to make the FFTW
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
class XPk:
    def __init__(self,delta,BoxSize,int axis=2,MAS=None,threads=1):

        start = time.time()
        cdef int dims, middle, fields, Xfields, num_unique_MAS, i, j
        cdef int index, index_z, index_2D, index_X
        cdef int kxx, kyy, kzz, kx, ky, kz, k_index#, begin, end
        cdef int kmax_par, kmax_per, kmax, k_par, k_per
        cdef double k, prefact, mu, mu2, val1, val2
        cdef double delta2, delta2_X, fact
        ####### change this for double precision ######
        cdef float MAS_factor
        cdef np.ndarray[np.complex64_t,ndim=3] delta_k
        #cdef np.complex64_t[:,:,::1] delta_k
        ###############################################
        cdef np.int32_t[::1] MAS_index, unique_MAS_id
        cdef np.float64_t[::1] real_part, imag_part, k1D, k3D
        cdef np.float64_t[::1] kpar, kper, Nmodes1D, Nmodes2D, Nmodes3D
        cdef np.float64_t[:,::1] MAS_corr, Pk1D, Pk2D, PkX1D, PkX2D
        cdef np.float64_t[:,:,::1] Pk3D, PkX3D

        print '\nComputing power spectra of the fields...'

        # find the number and dimensions of the density fields
        # we assume the density fields are (dims,dims,dims) arrays
        dims = len(delta[0]);  middle = dims/2;  fields = len(delta)
        Xfields = fields*(fields-1)/2  #number of independent cross-P(k)

        # check that the dimensions of all fields are the same
        for i in xrange(1,fields):
            if len(delta[i])!=dims:
                print 'Fields have different grid sizes!!!'; sys.exit()

        # find the different relevant frequencies
        kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)

        # find the independent MAS and the arrays relating both.
        # if MAS = ['CIC','PCS','CIC','CIC'] ==> unique_MAS = ['CIC','PCS']
        # num_unique_MAS = 2 : unique_MAS_id = [0,1,0,0]
        unique_MAS     = np.array(list(set(MAS))) #array with independent MAS
        num_unique_MAS = len(unique_MAS)          #number of independent MAS
        unique_MAS_id  = np.empty(fields,dtype=np.int32) 
        for i in xrange(fields):
            unique_MAS_id[i] = np.where(MAS[i]==unique_MAS)[0][0]

        # define and fill the MAS_corr and MAS_index arrays
        MAS_corr  = np.ones((num_unique_MAS,3), dtype=np.float64)
        MAS_index = np.zeros(num_unique_MAS,    dtype=np.int32)
        for i in xrange(num_unique_MAS):
            MAS_index[i] = MAS_function(unique_MAS[i])

        # define the real_part and imag_part arrays
        real_part = np.zeros(fields,dtype=np.float64)
        imag_part = np.zeros(fields,dtype=np.float64)

        ## compute FFT of the field (change this for double precision) ##
        # to try to have the elements of the different fields as close as 
        # possible we stack along the z-direction (major-row)
        delta_k = np.empty((dims,dims,(middle+1)*fields),dtype=np.complex64)
        for i in xrange(fields):
            begin = i*(middle+1);  end = (i+1)*(middle+1)
            delta_k[:,:,begin:end] = FFT3Dr_f(delta[i],threads)
        print 'Time FFTS = %.2f'%(time.time()-start)
        #################################

        # define arrays having k1D, Pk1D, PkX1D & Nmodes1D. We need kmax_par+1
        # bins since modes go from 0 to kmax_par. Is better if we define the
        # arrays as (kmax_par+1,fields) rather than (fields,kmax_par+1) since
        # in memory arrays numpy arrays are row-major
        k1D      = np.zeros(kmax_par+1,           dtype=np.float64)
        Pk1D     = np.zeros((kmax_par+1,fields),  dtype=np.float64)
        PkX1D    = np.zeros((kmax_par+1,Xfields), dtype=np.float64)
        Nmodes1D = np.zeros(kmax_par+1,           dtype=np.float64)

        # define arrays containing Pk2D and Nmodes2D. We define the arrays
        # in this way to have them as close as possible in row-major
        Pk2D     = np.zeros(((kmax_par+1)*(kmax_per+1),fields), 
                            dtype=np.float64)
        PkX2D    = np.zeros(((kmax_par+1)*(kmax_per+1),Xfields),
                            dtype=np.float64)
        Nmodes2D = np.zeros((kmax_par+1)*(kmax_per+1), 
                            dtype=np.float64)

        # define arrays containing k3D, Pk3D,PkX3D & Nmodes3D. We need kmax+1
        # bins since the mode (middle,middle, middle) has an index = kmax.
        # We define the arrays in this way to benefit of row-major
        k3D      = np.zeros(kmax+1,             dtype=np.float64)
        Pk3D     = np.zeros((kmax+1,3,fields),  dtype=np.float64)
        PkX3D    = np.zeros((kmax+1,3,Xfields), dtype=np.float64)
        Nmodes3D = np.zeros(kmax+1,             dtype=np.float64)

        # do a loop over the independent modes.
        # compute k,k_par,k_per, mu for each mode. k's are in kF units
        start2 = time.time();  prefact = np.pi/dims
        for kxx in xrange(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            for i in xrange(num_unique_MAS):
                MAS_corr[i,0] = MAS_correction(prefact*kx,MAS_index[i])

            for kyy in xrange(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                for i in xrange(num_unique_MAS):
                    MAS_corr[i,1] = MAS_correction(prefact*ky,MAS_index[i])

                for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    for i in xrange(num_unique_MAS):
                        MAS_corr[i,2] = MAS_correction(prefact*kz,MAS_index[i])

                    # kz=0 and kz=middle planes are special
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue

                    ###### k, k_index, k_par,k_per, mu ######
                    # compute |k| of the mode and its integer part
                    k       = sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = <int>k

                    # compute the value of k_par and k_perp
                    if axis==0:   
                        k_par, k_per = kx, <int>sqrt(ky*ky + kz*kz)
                    elif axis==1: 
                        k_par, k_per = ky, <int>sqrt(kx*kx + kz*kz)
                    else:         
                        k_par, k_per = kz, <int>sqrt(kx*kx + ky*ky)

                    # find the value of mu
                    if k==0:  mu = 0.0
                    else:     mu = k_par/k
                    mu2 = mu*mu  
                    val1 = (3.0*mu2-1.0)/2.0
                    val2 = (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0

                    # take the absolute value of k_par
                    if k_par<0:  k_par = -k_par
                    #########################################

                    ####### fill the general arrays #########
                    # Pk1D(k)
                    if k<=middle:
                        k1D[k_par]      += k_par
                        Nmodes1D[k_par] += 1.0
                    
                    # Pk2D: index_2D goes from 0 to (kmax_par+1)*(kmax_per+1)-1
                    index_2D = (kmax_par+1)*k_per + k_par
                    Nmodes2D[index_2D] += 1.0

                    # Pk3D
                    k3D[k_index]      += k
                    Nmodes3D[k_index] += 1.0
                    #########################################

                    #### correct modes amplitude for MAS ####
                    for i in xrange(fields):
                        index = unique_MAS_id[i]
                        MAS_factor = MAS_corr[index,0]*\
                                     MAS_corr[index,1]*\
                                     MAS_corr[index,2]
                        index_z = i*(middle+1) + kzz
                        delta_k[kxx,kyy,index_z] = delta_k[kxx,kyy,index_z]*\
                                                    MAS_factor
                        real_part[i] = delta_k[kxx,kyy,index_z].real
                        imag_part[i] = delta_k[kxx,kyy,index_z].imag

                        ########## compute auto-P(k) ########
                        delta2 = real_part[i]*real_part[i] +\
                                 imag_part[i]*imag_part[i]

                        # Pk1D: only consider modes with |k|<kF
                        if k<=middle:
                            Pk1D[k_par,i] += delta2

                        # Pk2D: P(k_per,k_par)
                        Pk2D[index_2D,i] += delta2

                        # Pk3D
                        Pk3D[k_index,0,i] += (delta2)
                        Pk3D[k_index,1,i] += (delta2*val1)
                        Pk3D[k_index,2,i] += (delta2*val2)
                    #########################################

                    ####### compute XPk for each pair #######
                    index_X  = 0
                    for i in xrange(fields):
                        for j in xrange(i+1,fields):
                            delta2_X = real_part[i]*real_part[j] +\
                                       imag_part[i]*imag_part[j]            

                            # Pk1D: only consider modes with |k|<kF
                            if k<=middle:
                                PkX1D[k_par,index_X] += delta2_X

                            # Pk2D: P(k_per,k_par)
                            PkX2D[index_2D,index_X] += delta2_X
                            
                            # Pk3D
                            PkX3D[k_index,0,index_X] += delta2_X
                            PkX3D[k_index,1,index_X] += (delta2_X*val1)
                            PkX3D[k_index,2,index_X] += (delta2_X*val2)

                            index_X += 1
                    #########################################

        print 'Time loop = %.2f'%(time.time()-start2)
        fact = (BoxSize/dims**2)**3

        # Pk1D. Discard DC mode bin and give units
        # the perpendicular modes sample an area equal to pi*kmax_per^2
        # we assume that each mode has an area equal to pi*kmax_per^2/Nmodes
        k1D  = k1D[1:];  Nmodes1D = Nmodes1D[1:]
        Pk1D = Pk1D[1:,:];  PkX1D = PkX1D[1:,:]
        for i in xrange(len(k1D)):
            k1D[i]  = (k1D[i]/Nmodes1D[i])*kF  #give units
            kmaxper = sqrt(kN**2 - k1D[i]**2)

            for j in xrange(fields):
                Pk1D[i,j] = Pk1D[i,j]*fact #give units
                Pk1D[i,j] = Pk1D[i,j]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
            for j in xrange(Xfields):
                PkX1D[i,j] = PkX1D[i,j]*fact #give units
                PkX1D[i,j] = PkX1D[i,j]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
        self.k1D = np.asarray(k1D);    self.Nmodes1D = np.asarray(Nmodes1D)  
        self.Pk1D = np.asarray(Pk1D);  self.PkX1D = np.asarray(PkX1D)

        # Pk2D. Keep DC mode bin, give units to Pk2D and find kpar & kper
        kpar = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
        kper = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
        for k_par in xrange(kmax_par+1):
            for k_per in xrange(kmax_per+1):
                index_2D = (kmax_par+1)*k_per + k_par
                kpar[index_2D] = 0.5*(k_par + k_par+1)*kF
                kper[index_2D] = 0.5*(k_per + k_per+1)*kF
        for i in xrange(len(kpar)):
            for j in xrange(fields):
                Pk2D[i,j] = Pk2D[i,j]*fact/Nmodes2D[i]
            for j in xrange(Xfields):
                PkX2D[i,j] = PkX2D[i,j]*fact/Nmodes2D[i]
        self.kpar = np.asarray(kpar);  self.kper = np.asarray(kper)
        self.Nmodes2D = np.asarray(Nmodes2D)
        self.Pk2D = np.asarray(Pk2D);  self.PkX2D = np.asarray(PkX2D)

        # Pk3D. Check modes, discard DC mode bin and give units
        # we need to multiply the multipoles by (2*ell + 1)
        check_number_modes(Nmodes3D,dims)
        k3D  = k3D[1:];  Nmodes3D = Nmodes3D[1:];  
        Pk3D = Pk3D[1:,:,:];  PkX3D = PkX3D[1:,:,:]
        for i in xrange(len(k3D)):
            k3D[i] = (k3D[i]/Nmodes3D[i])*kF

            for j in xrange(fields):
                Pk3D[i,0,j] = (Pk3D[i,0,j]/Nmodes3D[i])*fact
                Pk3D[i,1,j] = (Pk3D[i,1,j]*5.0/Nmodes3D[i])*fact
                Pk3D[i,2,j] = (Pk3D[i,2,j]*9.0/Nmodes3D[i])*fact

            for j in xrange(Xfields):
                PkX3D[i,0,j] = (PkX3D[i,0,j]/Nmodes3D[i])*fact
                PkX3D[i,1,j] = (PkX3D[i,1,j]*5.0/Nmodes3D[i])*fact
                PkX3D[i,2,j] = (PkX3D[i,2,j]*9.0/Nmodes3D[i])*fact

        self.k3D = np.asarray(k3D);  self.Nmodes3D = np.asarray(Nmodes3D)
        self.Pk = np.asarray(Pk3D);  self.XPk = np.asarray(PkX3D)

        print 'Time taken = %.2f seconds'%(time.time()-start)
################################################################################
################################################################################

################################################################################
################################################################################
# This routine computes the real auto- and imaginary cross-power spectra 
# of various density fields
# delta -------> list with the density fields: [delta1,delta2,delta3,...]
# each density field should be a: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight for the multipoles
# MAS ---------> list with the mass assignment scheme used to compute density 
# fields: ['CIC','CIC','PCS',...]
# threads -----> number of threads (OMP) used to make the FFTW
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
class XPk_imag:
    def __init__(self,delta,BoxSize,int axis=2,MAS=None,threads=1):

        start = time.time()
        cdef int dims, middle, fields, Xfields, num_unique_MAS, i, j
        cdef int index, index_z, index_2D, index_X
        cdef int kxx, kyy, kzz, kx, ky, kz, k_index#, begin, end
        cdef int kmax_par, kmax_per, kmax, k_par, k_per
        cdef double k, prefact, mu, mu2, val1, val2
        cdef double delta2, delta2_X, fact
        ####### change this for double precision ######
        cdef float MAS_factor
        cdef np.ndarray[np.complex64_t,ndim=3] delta_k
        #cdef np.complex64_t[:,:,::1] delta_k
        ###############################################
        cdef np.int32_t[::1] MAS_index, unique_MAS_id
        cdef np.float64_t[::1] real_part, imag_part, k1D, k3D
        cdef np.float64_t[::1] kpar, kper, Nmodes1D, Nmodes2D, Nmodes3D
        cdef np.float64_t[:,::1] MAS_corr, Pk1D, Pk2D, PkX1D, PkX2D
        cdef np.float64_t[:,:,::1] Pk3D, PkX3D

        print '\nComputing power spectra of the fields...'

        # find the number and dimensions of the density fields
        # we assume the density fields are (dims,dims,dims) arrays
        dims = len(delta[0]);  middle = dims/2;  fields = len(delta)
        Xfields = fields*(fields-1)/2  #number of independent cross-P(k)

        # check that the dimensions of all fields are the same
        for i in xrange(1,fields):
            if len(delta[i])!=dims:
                print 'Fields have different grid sizes!!!'; sys.exit()

        # find the different relevant frequencies
        kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)

        # find the independent MAS and the arrays relating both.
        # if MAS = ['CIC','PCS','CIC','CIC'] ==> unique_MAS = ['CIC','PCS']
        # num_unique_MAS = 2 : unique_MAS_id = [0,1,0,0]
        unique_MAS     = np.array(list(set(MAS))) #array with independent MAS
        num_unique_MAS = len(unique_MAS)          #number of independent MAS
        unique_MAS_id  = np.empty(fields,dtype=np.int32) 
        for i in xrange(fields):
            unique_MAS_id[i] = np.where(MAS[i]==unique_MAS)[0][0]

        # define and fill the MAS_corr and MAS_index arrays
        MAS_corr  = np.ones((num_unique_MAS,3), dtype=np.float64)
        MAS_index = np.zeros(num_unique_MAS,    dtype=np.int32)
        for i in xrange(num_unique_MAS):
            MAS_index[i] = MAS_function(unique_MAS[i])

        # define the real_part and imag_part arrays
        real_part = np.zeros(fields,dtype=np.float64)
        imag_part = np.zeros(fields,dtype=np.float64)

        ## compute FFT of the field (change this for double precision) ##
        # to try to have the elements of the different fields as close as 
        # possible we stack along the z-direction (major-row)
        delta_k = np.empty((dims,dims,(middle+1)*fields),dtype=np.complex64)
        for i in xrange(fields):
            begin = i*(middle+1);  end = (i+1)*(middle+1)
            delta_k[:,:,begin:end] = FFT3Dr_f(delta[i],threads)
        print 'Time FFTS = %.2f'%(time.time()-start)
        #################################

        # define arrays having k1D, Pk1D, PkX1D & Nmodes1D. We need kmax_par+1
        # bins since modes go from 0 to kmax_par. Is better if we define the
        # arrays as (kmax_par+1,fields) rather than (fields,kmax_par+1) since
        # in memory arrays numpy arrays are row-major
        k1D      = np.zeros(kmax_par+1,           dtype=np.float64)
        Pk1D     = np.zeros((kmax_par+1,fields),  dtype=np.float64)
        PkX1D    = np.zeros((kmax_par+1,Xfields), dtype=np.float64)
        Nmodes1D = np.zeros(kmax_par+1,           dtype=np.float64)

        # define arrays containing Pk2D and Nmodes2D. We define the arrays
        # in this way to have them as close as possible in row-major
        Pk2D     = np.zeros(((kmax_par+1)*(kmax_per+1),fields), 
                            dtype=np.float64)
        PkX2D    = np.zeros(((kmax_par+1)*(kmax_per+1),Xfields),
                            dtype=np.float64)
        Nmodes2D = np.zeros((kmax_par+1)*(kmax_per+1), 
                            dtype=np.float64)

        # define arrays containing k3D, Pk3D,PkX3D & Nmodes3D. We need kmax+1
        # bins since the mode (middle,middle, middle) has an index = kmax.
        # We define the arrays in this way to benefit of row-major
        k3D      = np.zeros(kmax+1,             dtype=np.float64)
        Pk3D     = np.zeros((kmax+1,3,fields),  dtype=np.float64)
        PkX3D    = np.zeros((kmax+1,3,Xfields), dtype=np.float64)
        Nmodes3D = np.zeros(kmax+1,             dtype=np.float64)

        # do a loop over the independent modes.
        # compute k,k_par,k_per, mu for each mode. k's are in kF units
        start2 = time.time();  prefact = np.pi/dims
        for kxx in xrange(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            for i in xrange(num_unique_MAS):
                MAS_corr[i,0] = MAS_correction(prefact*kx,MAS_index[i])

            for kyy in xrange(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                for i in xrange(num_unique_MAS):
                    MAS_corr[i,1] = MAS_correction(prefact*ky,MAS_index[i])

                for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    for i in xrange(num_unique_MAS):
                        MAS_corr[i,2] = MAS_correction(prefact*kz,MAS_index[i])

                    # kz=0 and kz=middle planes are special
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue

                    ###### k, k_index, k_par,k_per, mu ######
                    # compute |k| of the mode and its integer part
                    k       = sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = <int>k

                    # compute the value of k_par and k_perp
                    if axis==0:   
                        k_par, k_per = kx, <int>sqrt(ky*ky + kz*kz)
                    elif axis==1: 
                        k_par, k_per = ky, <int>sqrt(kx*kx + kz*kz)
                    else:         
                        k_par, k_per = kz, <int>sqrt(kx*kx + ky*ky)

                    # find the value of mu
                    if k==0:  mu = 0.0
                    else:     mu = k_par/k
                    mu2 = mu*mu  
                    val1 = (3.0*mu2-1.0)/2.0
                    val2 = (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0

                    # take the absolute value of k_par
                    if k_par<0:  k_par = -k_par
                    #########################################

                    ####### fill the general arrays #########
                    # Pk1D(k)
                    if k<=middle:
                        k1D[k_par]      += k_par
                        Nmodes1D[k_par] += 1.0
                    
                    # Pk2D: index_2D goes from 0 to (kmax_par+1)*(kmax_per+1)-1
                    index_2D = (kmax_par+1)*k_per + k_par
                    Nmodes2D[index_2D] += 1.0

                    # Pk3D
                    k3D[k_index]      += k
                    Nmodes3D[k_index] += 1.0
                    #########################################

                    #### correct modes amplitude for MAS ####
                    for i in xrange(fields):
                        index = unique_MAS_id[i]
                        MAS_factor = MAS_corr[index,0]*\
                                     MAS_corr[index,1]*\
                                     MAS_corr[index,2]
                        index_z = i*(middle+1) + kzz
                        delta_k[kxx,kyy,index_z] = delta_k[kxx,kyy,index_z]*\
                                                    MAS_factor
                        real_part[i] = delta_k[kxx,kyy,index_z].real
                        imag_part[i] = delta_k[kxx,kyy,index_z].imag

                        ########## compute auto-P(k) ########
                        delta2 = real_part[i]*real_part[i] +\
                                 imag_part[i]*imag_part[i]

                        # Pk1D: only consider modes with |k|<kF
                        if k<=middle:
                            Pk1D[k_par,i] += delta2

                        # Pk2D: P(k_per,k_par)
                        Pk2D[index_2D,i] += delta2

                        # Pk3D
                        Pk3D[k_index,0,i] += (delta2)
                        Pk3D[k_index,1,i] += (delta2*val1)
                        Pk3D[k_index,2,i] += (delta2*val2)
                    #########################################

                    ####### compute XPk for each pair #######
                    index_X  = 0
                    for i in xrange(fields):
                        for j in xrange(i+1,fields):
                            #delta2_X = real_part[i]*real_part[j] +\
                            #           imag_part[i]*imag_part[j]            
                            delta2_X = imag_part[i]*real_part[j] -\
			               real_part[i]*imag_part[j]

                            # Pk1D: only consider modes with |k|<kF
                            if k<=middle:
                                PkX1D[k_par,index_X] += delta2_X

                            # Pk2D: P(k_per,k_par)
                            PkX2D[index_2D,index_X] += delta2_X
                            
                            # Pk3D
                            PkX3D[k_index,0,index_X] += delta2_X
                            PkX3D[k_index,1,index_X] += (delta2_X*val1)
                            PkX3D[k_index,2,index_X] += (delta2_X*val2)

                            index_X += 1
                    #########################################

        print 'Time loop = %.2f'%(time.time()-start2)
        fact = (BoxSize/dims**2)**3

        # Pk1D. Discard DC mode bin and give units
        # the perpendicular modes sample an area equal to pi*kmax_per^2
        # we assume that each mode has an area equal to pi*kmax_per^2/Nmodes
        k1D  = k1D[1:];  Nmodes1D = Nmodes1D[1:]
        Pk1D = Pk1D[1:,:];  PkX1D = PkX1D[1:,:]
        for i in xrange(len(k1D)):
            k1D[i]  = (k1D[i]/Nmodes1D[i])*kF  #give units
            kmaxper = sqrt(kN**2 - k1D[i]**2)

            for j in xrange(fields):
                Pk1D[i,j] = Pk1D[i,j]*fact #give units
                Pk1D[i,j] = Pk1D[i,j]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
            for j in xrange(Xfields):
                PkX1D[i,j] = PkX1D[i,j]*fact #give units
                PkX1D[i,j] = PkX1D[i,j]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
        self.k1D = np.asarray(k1D);    self.Nmodes1D = np.asarray(Nmodes1D)  
        self.Pk1D = np.asarray(Pk1D);  self.PkX1D = np.asarray(PkX1D)

        # Pk2D. Keep DC mode bin, give units to Pk2D and find kpar & kper
        kpar = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
        kper = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
        for k_par in xrange(kmax_par+1):
            for k_per in xrange(kmax_per+1):
                index_2D = (kmax_par+1)*k_per + k_par
                kpar[index_2D] = 0.5*(k_par + k_par+1)*kF
                kper[index_2D] = 0.5*(k_per + k_per+1)*kF
        for i in xrange(len(kpar)):
            for j in xrange(fields):
                Pk2D[i,j] = Pk2D[i,j]*fact/Nmodes2D[i]
            for j in xrange(Xfields):
                PkX2D[i,j] = PkX2D[i,j]*fact/Nmodes2D[i]
        self.kpar = np.asarray(kpar);  self.kper = np.asarray(kper)
        self.Nmodes2D = np.asarray(Nmodes2D)
        self.Pk2D = np.asarray(Pk2D);  self.PkX2D = np.asarray(PkX2D)

        # Pk3D. Check modes, discard DC mode bin and give units
        # we need to multiply the multipoles by (2*ell + 1)
        check_number_modes(Nmodes3D,dims)
        k3D  = k3D[1:];  Nmodes3D = Nmodes3D[1:];  
        Pk3D = Pk3D[1:,:,:];  PkX3D = PkX3D[1:,:,:]
        for i in xrange(len(k3D)):
            k3D[i] = (k3D[i]/Nmodes3D[i])*kF

            for j in xrange(fields):
                Pk3D[i,0,j] = (Pk3D[i,0,j]/Nmodes3D[i])*fact
                Pk3D[i,1,j] = (Pk3D[i,1,j]*5.0/Nmodes3D[i])*fact
                Pk3D[i,2,j] = (Pk3D[i,2,j]*9.0/Nmodes3D[i])*fact

            for j in xrange(Xfields):
                PkX3D[i,0,j] = (PkX3D[i,0,j]/Nmodes3D[i])*fact
                PkX3D[i,1,j] = (PkX3D[i,1,j]*5.0/Nmodes3D[i])*fact
                PkX3D[i,2,j] = (PkX3D[i,2,j]*9.0/Nmodes3D[i])*fact

        self.k3D = np.asarray(k3D);  self.Nmodes3D = np.asarray(Nmodes3D)
        self.Pk = np.asarray(Pk3D);  self.XPk = np.asarray(PkX3D)

        print 'Time taken = %.2f seconds'%(time.time()-start)
################################################################################
################################################################################




################################################################################
################################################################################
# This function computes the power spectrum of theta = div V
# Vx ----------> 3D velocity field of the x-component: (dims,dims,dims) np array
# Vy ----------> 3D velocity field of the y-component: (dims,dims,dims) np array
# Vz ----------> 3D velocity field of the z-component: (dims,dims,dims) np array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight for the multipoles
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
#@cython.boundscheck(False)
#@cython.cdivision(False)
def Pk_theta(Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,k_index,kmax,MAS_index
    cdef double kmod,prefact,real,imag,theta2
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] Vx_k,Vy_k,Vz_k
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] k,Nmodes,MAS_corr
    cdef np.ndarray[np.float64_t,ndim=1] Pk 

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing power spectrum of theta...'
    dims = len(Vx);  middle = dims/2
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
    MAS_index = MAS_function(MAS)
    MAS_corr = np.ones(3, dtype=np.float64)
                
    ## compute FFT of the field (change this for double precision) ##
    #delta_k = FFT3Dr_f(delta,threads)
    Vx_k = FFT3Dr_f(Vx,threads)
    Vy_k = FFT3Dr_f(Vy,threads)
    Vz_k = FFT3Dr_f(Vz,threads)
    #################################

    # define arrays containing k, Pk0,Pk2,Pk4 and Nmodes. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    k      = np.zeros(kmax+1,dtype=np.float64)
    Pk     = np.zeros(kmax+1,dtype=np.float64)
    Nmodes = np.zeros(kmax+1,dtype=np.float64)

    # do a loop over all modes, computing their k,Pk. k's are in k_F units
    start2 = time.time();  prefact = np.pi/dims
    for kxx in xrange(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
        
        for kyy in xrange(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

            for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)
                MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==middle and dims%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==middle and dims%2==0):
                        if ky<0.0: continue

                # compute |k| of the mode and its integer part
                kmod    = sqrt(kx*kx + ky*ky + kz*kz)
                k_index = <int>kmod

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                Vx_k[kxx,kyy,kzz] = Vx_k[kxx,kyy,kzz]*MAS_factor
                Vy_k[kxx,kyy,kzz] = Vy_k[kxx,kyy,kzz]*MAS_factor
                Vz_k[kxx,kyy,kzz] = Vz_k[kxx,kyy,kzz]*MAS_factor
                #delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                # compute theta for each mode: theta(k) = ik*V(k)
                real = -(kx*Vx_k[kxx,kyy,kzz].imag + \
                         ky*Vy_k[kxx,kyy,kzz].imag + \
                         kz*Vz_k[kxx,kyy,kzz].imag)
            
                imag = kx*Vx_k[kxx,kyy,kzz].real + \
                       ky*Vy_k[kxx,kyy,kzz].real + \
                       kz*Vz_k[kxx,kyy,kzz].real
                
                theta2 = real*real + imag*imag

                # add mode to the k,Pk and Nmodes arrays
                k[k_index]      += kmod
                Pk[k_index]     += theta2
                Nmodes[k_index] += 1.0
    print 'Time compute modulus = %.2f'%(time.time()-start2)

    # check modes, discard fundamental frequency bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    check_number_modes(Nmodes,dims)
    k  = k[1:];    Nmodes = Nmodes[1:];   k = (k/Nmodes)*kF; 
    Pk = Pk[1:]*(BoxSize/dims**2)**3*kF**2;  Pk *= (1.0/Nmodes);
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [k,Pk,Nmodes]
################################################################################
################################################################################

################################################################################
################################################################################
# This function computes the auto- and cross-power spectra of the density
# field and the field div[(1+delta)*V]
# delta -------> 3D density field: (dims,dims,dims) np array
# Vx ----------> 3D velocity field of the x-component: (dims,dims,dims) np array
# Vy ----------> 3D velocity field of the y-component: (dims,dims,dims) np array
# Vz ----------> 3D velocity field of the z-component: (dims,dims,dims) np array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight for the multipoles
# MAS ---------> mass assignment scheme used to compute the fields
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
#@cython.boundscheck(False)
#@cython.cdivision(False)
def XPk_dv(delta,Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,k_index,kmax
    cdef int MAS_index
    cdef double kmod,prefact,real1,real2,imag1,imag2
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] delta_k,Vx_k,Vy_k,Vz_k
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] k,Nmodes,MAS_corr
    cdef np.ndarray[np.float64_t,ndim=1] Pk1,Pk2,PkX

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing power spectra of the fields...'
    dims = len(delta);  middle = dims/2
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index, MAS_corr = MAS_function(MAS)
                            
    # compute the fields (1+delta)*V
    Vx *= (1.0 + delta);  Vy *= (1.0 + delta);  Vz *= (1.0 + delta)

    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    Vx_k    = FFT3Dr_f(Vx,threads)
    Vy_k    = FFT3Dr_f(Vy,threads)
    Vz_k    = FFT3Dr_f(Vz,threads)
    #################################

    # define arrays containing k, Pk0,Pk2,Pk4 and Nmodes. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    k      = np.zeros(kmax+1, dtype=np.float64)
    Pk1    = np.zeros(kmax+1, dtype=np.float64)
    Pk2    = np.zeros(kmax+1, dtype=np.float64)
    PkX    = np.zeros(kmax+1, dtype=np.float64)
    Nmodes = np.zeros(kmax+1, dtype=np.float64)

    # do a loop over all modes, computing their k,Pk. k's are in k_F units
    start2 = time.time();  prefact = np.pi/dims
    for kxx in xrange(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
        
        for kyy in xrange(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

            for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)
                MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==middle and dims%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==middle and dims%2==0):
                        if ky<0.0: continue

                # compute |k| of the mode and its integer part
                kmod    = sqrt(kx*kx + ky*ky + kz*kz)
                k_index = <int>kmod

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor
                Vx_k[kxx,kyy,kzz]    = Vx_k[kxx,kyy,kzz]*MAS_factor
                Vy_k[kxx,kyy,kzz]    = Vy_k[kxx,kyy,kzz]*MAS_factor
                Vz_k[kxx,kyy,kzz]    = Vz_k[kxx,kyy,kzz]*MAS_factor

                # compute |delta_k|^2 of the mode
                real1 = delta_k[kxx,kyy,kzz].real
                imag1 = delta_k[kxx,kyy,kzz].imag

                real2 = +(kx*Vx_k[kxx,kyy,kzz].imag + \
                          ky*Vy_k[kxx,kyy,kzz].imag + \
                          kz*Vz_k[kxx,kyy,kzz].imag)
            
                imag2 = -(kx*Vx_k[kxx,kyy,kzz].real + \
                          ky*Vy_k[kxx,kyy,kzz].real + \
                          kz*Vz_k[kxx,kyy,kzz].real)

                # add mode to the k,Pk and Nmodes arrays
                k[k_index]      += kmod
                Pk1[k_index]    += (real1*real1 + imag1*imag1)
                Pk2[k_index]    += (real2*real2 + imag2*imag2)
                PkX[k_index]    += (real1*real2 + imag1*imag2)
                Nmodes[k_index] += 1.0
    print 'Time compute modulus = %.2f'%(time.time()-start2)

    # check modes, discard fundamental frequency bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    check_number_modes(Nmodes,dims)
    k   = k[1:];    Nmodes = Nmodes[1:];       k = (k/Nmodes)*kF; 
    Pk1 = Pk1[1:]*(BoxSize/dims**2)**3;        Pk1 *= (1.0/Nmodes)
    Pk2 = Pk2[1:]*(BoxSize/dims**2)**3*kF**2;  Pk2 *= (1.0/Nmodes)
    PkX = PkX[1:]*(BoxSize/dims**2)**3*kF;     PkX *= (1.0/Nmodes)
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [k,Pk1,Pk2,PkX,Nmodes]
################################################################################
################################################################################

################################################################################
################################################################################
# This is the integrant of the 1D P(k) integral
cdef double func_1D(double y, double x, log10_k, Pk, double k_par):
    cdef double log10_kmod,Pk_3D
    log10_kmod = log10(sqrt(x*x + k_par*k_par))
    Pk_3D = np.interp(log10_kmod,log10_k,Pk)
    return x*Pk_3D

"""
# This routines computes first the 3D P(k) and then computes the 1D P(k) using
# P_1D(k_par) = 1/(2*pi) \int dk_per k_per P_3D(k_par,k_perp), where kmax_per
# is set to kN. This routine will in general be slower than Pk_1D and it is 
# only valid in real-space, so we recommend use Pk_1D to compute the 1D P(k) in
# general and use this one only for validation purposes. The routine
# returns the 3D and 1D power spectra of the field. The reason why the outcome
# of this procedure is slightly different to the one from Pk_1D or Pk_1D_from_2D
# is because P(k_par,k_per), for a given k_par is slighty different to 
# P(k), where k=sqrt(k_par,k_per). In the latter we are averaging over all modes
# with k, but the power spectrum near the Nyquist frequency exhibits a large
# variance and therefore a large dependence on k_par. In this case, the correct
# thing to do is to use Pk_1D or Pk_1D_from_2D.
def Pk_1D_from_3D(delta,BoxSize,axis=2,MAS='CIC',threads=1):

    # compute 3D P(k) 
    [k,Pk0,Pk2,Pk4,Nmodes] = Pk(delta,BoxSize,axis,MAS,threads)
                  
    # define the 1D P(k) array
    Pk_1D = np.zeros(len(k),dtype=np.float64)
    
    # only take perpendicular modes with |k_perp|<kN to make the integral
    dims = len(delta);  kN = (2.0*np.pi/BoxSize)*(dims/2)

    print 'Computing 1D P(k) from 3D P(k)...';  start = time.time()
    log10_k = np.log10(k)
    for i,k_par in enumerate(k):

        if k_par>=kN:  continue
        kmax_per = np.sqrt(kN**2 - k_par**2)
            
        yinit = [0.0];  kper_limits = [0,kmax_per]
        Pk_1D[i] = si.odeint(func_1D, yinit, kper_limits, 
                             args=(log10_k,Pk0,k_par),
                             mxstep=1000000,rtol=1e-8, atol=1e-10, 
                             h0=1e-10)[1][0]/(2.0*np.pi)
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [k,Pk_1D,Pk0]

# This routines computes first the 2D P(k) and then computes the 1D P(k) using
# P_1D(k_par) = 1/(2*pi) \int dk_perp k_perp P_3D(k_par,k_perp), where kmax_per
# is set to sqrt(kN^2-kpar^2), i.e. taking only modes with |k|<kN. 
# This routine will in general be slower than Pk_1D but its results should 
# be similar to those from Pk_1D, so can be used for validation purposes.
def Pk_1D_from_2D(delta,BoxSize,axis=2,MAS='CIC',threads=1):

    # find number of modes along parallel and perpendicular directions
    dims = len(delta);  middle = dims/2
    kF = 2.0*np.pi/BoxSize;  kN = kF*middle
    Nmodes_par = middle+1
    Nmodes_per = int(np.sqrt(middle**2 + middle**2))+1

    # define 1D P(k)
    Pk_1D = np.zeros(Nmodes_par,dtype=np.float64)

    # compute 2D P(k) 
    kpar_2D,kper_2D,Pk2D,Nmodes_2D = Pk_2D(delta,BoxSize,axis=axis,
                                           MAS=MAS,threads=threads)
                                           
    # put 2D power spectrum as matrix instead of 1D array: Pk[kper,kpar]
    Pk = np.reshape(Pk2D,(Nmodes_per,Nmodes_par))

    # find the value of the parallel and perpendicular modes sampled
    k_par = (np.arange(0,Nmodes_par)+0.5)*kF  #h/Mpc
    k_per = (np.arange(0,Nmodes_per)+0.5)*kF  #h/Mpc

    # for each k_par mode compute \int dk_per k_per P_3D(k_par,k_per)/(2pi)
    print 'Computing 1D P(k) from 2D P(k)...';  start = time.time()
    for i in xrange(Nmodes_par):

        # obtain the value of k_par, |k| and Pk(k_par,k_per)
        kpar = k_par[i];  Pk_val = Pk[:,i]
        k    = np.sqrt(kpar**2 + k_per**2);  log10_k = np.log10(k)

        # find the value of kmax_per
        if kpar>kN:  continue
        kmax_per = np.sqrt(kN**2 - kpar**2)
        
        yinit = [0.0];  kper_limits = [0,kmax_per]
        Pk_1D[i] = si.odeint(func_1D, yinit, kper_limits, 
                             args=(log10_k,Pk_val,kpar),
                             mxstep=1000000,rtol=1e-8, atol=1e-10, 
                             h0=1e-10)[1][0]/(2.0*np.pi)
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [k_par,Pk_1D,kpar_2D,kper_2D,Pk2D,Nmodes_2D]
"""
################################################################################
################################################################################

################################################################################
################################################################################
# This routine takes two density fields, delta1 and delta2 and computes the 
# 2D P(k) of the auto-fields and the 2D P(k) of the cross-power spectrum
def XPk_2D(delta1,delta2,BoxSize,axis=2,MAS1='CIC',MAS2='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,kmax,index
    cdef int MAS_index1,MAS_index2,imax_par,imax_per,ipar,iper
    cdef double prefact,real1,real2,imag1,imag2
    cdef double delta2_1,delta2_2,delta2_X
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] delta_k1,delta_k2
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] Nmodes,MAS_corr1,MAS_corr2
    cdef np.ndarray[np.float64_t,ndim=1] kpar,kper,Pk1,Pk2,PkX

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing power spectra of the fields...'
    dims = len(delta1);  middle = dims/2
    if dims!=len(delta2):
        print 'Different grids in the two fields!!!';  sys.exit()
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index1, MAS_corr1 = MAS_function(MAS1)
    MAS_index2, MAS_corr2 = MAS_function(MAS2)

    # find maximum wavenumbers, in kF units, along the par and perp directions
    imax_par = middle 
    imax_per = int(np.sqrt(middle**2 + middle**2))
                            
    ## compute FFT of the field (change this for double precision) ##
    delta_k1 = FFT3Dr_f(delta1,threads)
    delta_k2 = FFT3Dr_f(delta2,threads)
    #################################

    # define arrays containing kpar,kper, Pk1,Pk2,PkX and Nmodes
    kpar   = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    kper   = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    Pk1    = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    Pk2    = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    PkX    = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    Nmodes = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)

    # do a loop over all modes, computing their k,Pk. k's are in k_F units
    start2 = time.time();  prefact = np.pi/dims
    for kxx in xrange(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        MAS_corr1[0] = MAS_correction(prefact*kx,MAS_index1)
        MAS_corr2[0] = MAS_correction(prefact*kx,MAS_index2)
        
        for kyy in xrange(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            MAS_corr1[1] = MAS_correction(prefact*ky,MAS_index1)
            MAS_corr2[1] = MAS_correction(prefact*ky,MAS_index2)

            for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)
                MAS_corr1[2] = MAS_correction(prefact*kz,MAS_index1)  
                MAS_corr2[2] = MAS_correction(prefact*kz,MAS_index2)  

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==middle and dims%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==middle and dims%2==0):
                        if ky<0.0: continue

                # compute the value of k_par and k_perp
                if axis==0:   
                    k_par, k_per = abs(kx), <int>sqrt(ky*ky + kz*kz)
                elif axis==1: 
                    k_par, k_per = abs(ky), <int>sqrt(kx*kx + kz*kz)
                else:         
                    k_par, k_per = abs(kz), <int>sqrt(kx*kx + ky*ky)

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr1[0]*MAS_corr1[1]*MAS_corr1[2]
                delta_k1[kxx,kyy,kzz] = delta_k1[kxx,kyy,kzz]*MAS_factor
                MAS_factor = MAS_corr2[0]*MAS_corr2[1]*MAS_corr2[2]
                delta_k2[kxx,kyy,kzz] = delta_k2[kxx,kyy,kzz]*MAS_factor

                # compute |delta_k|^2 of the mode
                real1 = delta_k1[kxx,kyy,kzz].real
                imag1 = delta_k1[kxx,kyy,kzz].imag
                real2 = delta_k2[kxx,kyy,kzz].real
                imag2 = delta_k2[kxx,kyy,kzz].imag
                delta2_1 = real1*real1 + imag1*imag1
                delta2_2 = real2*real2 + imag2*imag2
                delta2_X = real1*real2 + imag1*imag2

                # we have one big 1D array to store the Pk(kper,kpar)
                # add mode to the Pk and Nmodes arrays
                index = (imax_par+1)*k_per + k_par
                Pk1[index]    += delta2_1
                Pk2[index]    += delta2_2
                PkX[index]    += delta2_X
                Nmodes[index] += 1.0
    print 'Time compute modulus = %.2f'%(time.time()-start2)

    # obtain the value of the kpar and kper for each bin
    for ipar in xrange(0,imax_par+1):
        for iper in xrange(0,imax_per+1):
            index = (imax_par+1)*iper + ipar
            kpar[index] = 0.5*(ipar + ipar+1)*kF
            kper[index] = 0.5*(iper + iper+1)*kF

    # keep fundamental frequency and give units
    Pk1 = Pk1*(BoxSize/dims**2)**3/Nmodes
    Pk2 = Pk2*(BoxSize/dims**2)**3/Nmodes
    PkX = PkX*(BoxSize/dims**2)**3/Nmodes
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [kpar,kper,Pk1,Pk2,PkX,Nmodes]
################################################################################
################################################################################

################################################################################
################################################################################
# This routine takes a field in real-space, Fourier transform it to get it in
# Fourier-space and then correct the modes amplitude to account for MAS. It then
# Fourier transform back and return the field in real-space 
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def correct_MAS(delta,BoxSize,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx, kyy, kzz, kx, ky, kz, dims, middle, MAS_index
    cdef int kmax_par, kmax_per, kmax, i
    cdef double k, prefact
    cdef double MAS_corr[3]
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.complex64_t[:,:,::1] delta_k
    ###############################################

    # find dimensions of delta: we assume is a (dims,dims,dims) array
    # determine the different frequencies and the MAS_index
    print '\nComputing power spectrum of the field...'
    dims = len(delta);  middle = dims/2
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
    MAS_index = MAS_function(MAS)
    
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################


    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time();  prefact = np.pi/dims
    for kxx in xrange(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
        
        for kyy in xrange(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

            for kzz in xrange(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)
                MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==middle and dims%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==middle and dims%2==0):
                        if ky<0.0: continue

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

    print 'Time to complete loop = %.2f'%(time.time()-start2)

    ## compute IFFT of the field (change this for double precision) ##
    delta = IFFT3Dr_f(delta_k,threads)
    #################################

    return delta

    print 'Time taken = %.2f seconds'%(time.time()-start)
################################################################################
################################################################################
