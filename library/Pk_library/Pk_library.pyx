import numpy as np
import time,sys,os
import pyfftw
import scipy.integrate as si
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,log10

################################ ROUTINES ####################################
# Pk(delta,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk0,Pk2,Pk4,Nmodes]

# XPk(delta1,delta2,BoxSize,axis=2,MAS1='CIC',MAS2='CIC',threads=1)
#   [k,
#    PkX_0,PkX_2,PkX_4,
#    Pk1_0,Pk1_2,Pk1_4,
#    Pk2_0,Pk2_2,Pk2_4,
#    Nmodes]

# Pk_theta(Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk_theta,Nmodes]

# XPk_dv(delta,Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk1,Pk2,PkX,Nmodes]

# Pk_1D(delta,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk,Nmodes]

# Pk_2D(delta,BoxSize,axis=2,MAS='CIC',threads=1)
#   [kpar,kper,Pk,Nmodes]
 
# Pk_1D_from_3D(delta,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k,Pk_1D,Pk_3D]

# Pk_1D_from_2D(delta,BoxSize,axis=2,MAS='CIC',threads=1)
#   [k_1D,Pk_1D,kpar_2D,kper_2D,Pk2D,Nmodes_2D]
##############################################################################

# This function determines the fundamental (kF) and Nyquist (kN) frequencies
# It also detemine the maximum frequency sampled in the box in units of kF
def frequencies(BoxSize,dims):
    kF = 2.0*np.pi/BoxSize;  kN = (dims/2.0)*kF
    kmax = np.sqrt(3.0*(dims/2.0)**2) # used to define the size of P(k) array
    return kF,kN,int(kmax)
    
# This function finds the MAS correction index and return the array used
def MAS_function(MAS):
    MAS_index = 0;  MAS_corr = np.ones(3,dtype=np.float64)
    if MAS=='NGP':  MAS_index = 1
    if MAS=='CIC':  MAS_index = 2
    if MAS=='TSC':  MAS_index = 3
    if MAS=='PCS':  MAS_index = 4
    return MAS_index,MAS_corr

# This function implement the MAS correction to modes amplitude
#@cython.cdivision(False)
#@cython.boundscheck(False)
cdef double MAS_correction(double x, int MAS_index):
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


################################################################################
################################################################################
# This function computes the power spectrum of the density field delta
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight for the multipoles
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
#@cython.boundscheck(False)
#@cython.cdivision(False)
def Pk(delta,BoxSize,axis=2,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,k_index,kmax,MAS_index
    cdef double kmod,delta2,prefact,mu,mu2,real,imag
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] delta_k
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] k,Nmodes,MAS_corr
    cdef np.ndarray[np.float64_t,ndim=2] Pk #monopole, quadrupole and hexadecapole

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing power spectrum of the field...'
    dims = len(delta);  middle = dims/2
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index, MAS_corr = MAS_function(MAS)
                                        
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################

    # define arrays containing k, Pk0,Pk2,Pk4 and Nmodes. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    k      = np.zeros(kmax+1,     dtype=np.float64)
    Pk     = np.zeros((kmax+1,3), dtype=np.float64)
    Nmodes = np.zeros(kmax+1,     dtype=np.float64)

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

                # find the value of mu
                if kmod==0:    mu = 0.0
                elif axis==0:  mu = kx/kmod
                elif axis==1:  mu = ky/kmod
                else:          mu = kz/kmod
                mu2 = mu*mu

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                # compute |delta_k|^2 of the mode
                real = delta_k[kxx,kyy,kzz].real
                imag = delta_k[kxx,kyy,kzz].imag
                delta2 = real*real + imag*imag

                # add mode to the k,Pk and Nmodes arrays
                k[k_index]      += kmod
                Pk[k_index,0]   += delta2
                Pk[k_index,1]   += (delta2*(3.0*mu2-1.0)/2.0)
                Pk[k_index,2]   += (delta2*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
                Nmodes[k_index] += 1.0
    print 'Time compute modulus = %.2f'%(time.time()-start2)

    # check modes, discard fundamental frequency bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    check_number_modes(Nmodes,dims)
    k  = k[1:];    Nmodes = Nmodes[1:];   k = (k/Nmodes)*kF; 
    Pk = Pk[1:,:]*(BoxSize/dims**2)**3
    Pk[:,0] *= (1.0/Nmodes);  Pk[:,1] *= (5.0/Nmodes);  Pk[:,2] *= (9.0/Nmodes) 
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [k,Pk[:,0],Pk[:,1],Pk[:,2],Nmodes]
################################################################################
################################################################################


################################################################################
################################################################################
# This function computes the auto- and cross-power spectra of two density fields
# delta1 ------> 3D density field1: (dims,dims,dims) numpy array
# delta2 ------> 3D density field2: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight for the multipoles
# MAS1 --------> mass assignment scheme used to compute density field 1
# MAS2 --------> mass assignment scheme used to compute density field 2
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
#@cython.boundscheck(False)
#@cython.cdivision(False)
def XPk(delta1,delta2,BoxSize,axis=2,MAS1='CIC',MAS2='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,k_index,kmax,
    cdef int MAS_index1,MAS_index2
    cdef double kmod,prefact,mu,mu2,real1,real2,imag1,imag2
    cdef double delta2_1,delta2_2,delta2_X
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] delta_k1,delta_k2
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] k,Nmodes,MAS_corr1,MAS_corr2
    cdef np.ndarray[np.float64_t,ndim=2] Pk1,Pk2,PkX

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing power spectra of the fields...'
    dims = len(delta1);  middle = dims/2
    if dims!=len(delta2):
        print 'Different grids in the two fields!!!';  sys.exit()
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index1, MAS_corr1 = MAS_function(MAS1)
    MAS_index2, MAS_corr2 = MAS_function(MAS2)
                            
    ## compute FFT of the field (change this for double precision) ##
    delta_k1 = FFT3Dr_f(delta1,threads)
    delta_k2 = FFT3Dr_f(delta2,threads)
    #################################

    # define arrays containing k, Pk0,Pk2,Pk4 and Nmodes. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    k      = np.zeros(kmax+1,     dtype=np.float64)
    Pk1    = np.zeros((kmax+1,3), dtype=np.float64)
    Pk2    = np.zeros((kmax+1,3), dtype=np.float64)
    PkX    = np.zeros((kmax+1,3), dtype=np.float64)
    Nmodes = np.zeros(kmax+1,     dtype=np.float64)

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

                # compute |k| of the mode and its integer part
                kmod    = sqrt(kx*kx + ky*ky + kz*kz)
                k_index = <int>kmod

                # find the value of mu
                if kmod==0:    mu = 0.0
                elif axis==0:  mu = kx/kmod
                elif axis==1:  mu = ky/kmod
                else:          mu = kz/kmod
                mu2 = mu*mu

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

                # add mode to the k,Pk and Nmodes arrays
                k[k_index]      += kmod

                Pk1[k_index,0]  += delta2_1
                Pk2[k_index,0]  += delta2_2
                PkX[k_index,0]  += delta2_X

                Pk1[k_index,1]  += (delta2_1*(3*mu2-1.0)/2.0)
                Pk2[k_index,1]  += (delta2_2*(3*mu2-1.0)/2.0)
                PkX[k_index,1]  += (delta2_X*(3*mu2-1.0)/2.0)

                Pk1[k_index,2]  += (delta2_1*(35*mu2*mu2 - 30*mu2 + 3.0)/8.0)
                Pk2[k_index,2]  += (delta2_2*(35*mu2*mu2 - 30*mu2 + 3.0)/8.0)
                PkX[k_index,2]  += (delta2_X*(35*mu2*mu2 - 30*mu2 + 3.0)/8.0)

                Nmodes[k_index] += 1.0
    print 'Time compute modulus = %.2f'%(time.time()-start2)

    # check modes, discard fundamental frequency bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    check_number_modes(Nmodes,dims)
    k  = k[1:];    Nmodes = Nmodes[1:];   k = (k/Nmodes)*kF; 

    Pk1 = Pk1[1:,:]*(BoxSize/dims**2)**3;  Pk1[:,0] *= (1.0/Nmodes)
    Pk1[:,1] *= (5.0/Nmodes);  Pk1[:,2] *= (9.0/Nmodes) 

    Pk2 = Pk2[1:,:]*(BoxSize/dims**2)**3;  Pk2[:,0] *= (1.0/Nmodes)
    Pk2[:,1] *= (5.0/Nmodes);  Pk2[:,2] *= (9.0/Nmodes) 

    PkX = PkX[1:,:]*(BoxSize/dims**2)**3;  PkX[:,0] *= (1.0/Nmodes)
    PkX[:,1] *= (5.0/Nmodes);  PkX[:,2] *= (9.0/Nmodes) 
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [k,
            PkX[:,0],PkX[:,1],PkX[:,2],
            Pk1[:,0],Pk1[:,1],Pk1[:,2],
            Pk2[:,0],Pk2[:,1],Pk2[:,2],
            Nmodes]
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
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index, MAS_corr = MAS_function(MAS)
                                        
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
# This routine computes the 1D power spectrum from a 3D density field
# The routine takes a 3D field, delta, and Fourier transform it to get delta(k)
# it then computes the 1D P(k) as:
# P_1D(k_par) = \int d^2k_per/(2pi)^2 P_3D(k_par,k_per)
# we approximate the 2D integral by the average of the modes sampled by the 
# field and we carry it out using k_per modes such as |k|<kN. The perpendicular
# modes sample the circle in an almost uniform way, thus, we approximate the 
# integral by a sum of the amplitude of each mode times the area it covers, 
# which is pi*kmax_per^2/Nmodes, where kmax_per=sqrt(kN^2-kpar^2)
def Pk_1D(delta,BoxSize,axis=2,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,kmax,MAS_index,k_par
    cdef double kmod,delta2,prefact,real,imag,k_per#,kmax_per
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] delta_k
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] k,Nmodes,MAS_corr,Pk

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing power spectrum of the field...'
    dims = len(delta);  middle = dims/2
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index, MAS_corr = MAS_function(MAS)  

    # do not consider modes with |k_perp| > kN. kmax_perp is in units of kF
    #kmax_per = middle
                                        
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################

    # define arrays containing k, Pk_1D and Nmodes. We need middle+1
    # bins since modes go from 0 to middle
    k      = np.zeros(middle+1, dtype=np.float64)
    Pk     = np.zeros(middle+1, dtype=np.float64)
    Nmodes = np.zeros(middle+1, dtype=np.float64)

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

                # compute the value of k_par and k_perp
                if axis==0:   
                    k_par, k_per = abs(kx), sqrt(ky*ky + kz*kz)
                elif axis==1: 
                    k_par, k_per = abs(ky), sqrt(kx*kx + kz*kz)
                else:         
                    k_par, k_per = abs(kz), sqrt(kx*kx + ky*ky)
                kmod = sqrt(k_par*k_par + k_per*k_per)

                # only consider modes with |k|<kN
                if kmod>middle:  continue

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                # compute |delta_k|^2 of the mode
                real = delta_k[kxx,kyy,kzz].real
                imag = delta_k[kxx,kyy,kzz].imag
                delta2 = real*real + imag*imag

                # add mode to the k,Pk and Nmodes arrays
                k[k_par]      += k_par
                Pk[k_par]     += delta2
                Nmodes[k_par] += 1.0
    print 'Time compute modulus = %.2f'%(time.time()-start2)

    # discard fundamental frequency bin and give units
    k  = k[1:];    Nmodes = Nmodes[1:];   k = (k/Nmodes)*kF; 
    Pk = Pk[1:]*(BoxSize/dims**2)**3
    
    # the perpendicular modes sample an area equal to pi*kmax_perp^2
    # we are assuming that each mode has an area equal to pi*kmax_perp^2/Nmodes
    kmax_per = np.sqrt(kN**2-k**2)
    #kmax_perp = kmax_perp*kF
    Pk = Pk*(np.pi*kmax_per**2/Nmodes)/(2.0*np.pi)**2
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [k,Pk,Nmodes]
################################################################################
################################################################################

################################################################################
################################################################################
# This routine computes the 2D power spectrum from a 3D density field
# The routine returns 1D arrays with kpar, kper, Pk and Nmodes
def Pk_2D(delta,BoxSize,axis=2,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,kmax,MAS_index,k_par,k_per
    cdef int ipar,iper,imax_par,imax_per,index
    cdef double kmod,delta2,prefact,real,imag
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] delta_k
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] MAS_corr,kpar,kper,Pk,Nmodes

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing 2D power spectrum of the field...'
    dims = len(delta);  middle = dims/2
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index, MAS_corr = MAS_function(MAS)  

    # find maximum wavenumbers, in kF units, along the par and perp directions
    imax_par  = middle 
    imax_per  = int(np.sqrt(middle**2 + middle**2))
                                        
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################

    # define arrays containing k, Pk_2D and Nmodes
    kpar   = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    kper   = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    Pk     = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)
    Nmodes = np.zeros((imax_par+1)*(imax_per+1), dtype=np.float64)

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

                # compute the value of k_par and k_perp
                if axis==0:   
                    k_par, k_per = abs(kx), <int>sqrt(ky*ky + kz*kz)
                elif axis==1: 
                    k_par, k_per = abs(ky), <int>sqrt(kx*kx + kz*kz)
                else:         
                    k_par, k_per = abs(kz), <int>sqrt(kx*kx + ky*ky)
                    
                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                # compute |delta_k|^2 of the mode
                real = delta_k[kxx,kyy,kzz].real
                imag = delta_k[kxx,kyy,kzz].imag
                delta2 = real*real + imag*imag
                                                
                # we have one big 1D array to store the Pk(kper,kpar)
                # add mode to the Pk and Nmodes arrays
                index = (imax_par+1)*k_per + k_par
                Pk[index]     += delta2
                Nmodes[index] += 1.0
    print 'Time compute modulus = %.2f'%(time.time()-start2)

    # obtain the value of the kpar and kper for each bin
    for ipar in xrange(0,imax_par+1):
        for iper in xrange(0,imax_per+1):
            index = (imax_par+1)*iper + ipar
            kpar[index] = 0.5*(ipar + ipar+1)*kF
            kper[index] = 0.5*(iper + iper+1)*kF

    # keep fundamental frequency and give units
    Pk = Pk*(BoxSize/dims**2)**3/Nmodes
    print 'Time taken = %.2f seconds'%(time.time()-start)

    return [kpar,kper,Pk,Nmodes]
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
################################################################################
################################################################################
