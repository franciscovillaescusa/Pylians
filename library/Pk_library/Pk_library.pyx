import numpy as np
import time,sys,os
import pyfftw
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin

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
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
#@cython.boundscheck(False)
#@cython.cdivision(False)
def XPk_dv(delta,Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,k_index,kmax,MAS_index
    cdef double kmod,prefact,real,imag,theta2
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.ndarray[np.complex64_t,ndim=3] delta_k,Vx_k,Vy_k,Vz_k
    ###############################################
    cdef np.ndarray[np.float64_t,ndim=1] k,Nmodes,MAS_corr
    cdef np.ndarray[np.float64_t,ndim=1] Pk,Pk_dv

    # find dimensions of delta: we assuming is a (dims,dims,dims) array
    # determine the different frequencies, the MAS_index and the MAS_corr
    print 'Computing power spectrum of theta...'
    dims = len(delta);  middle = dims/2
    kF,kN,kmax = frequencies(BoxSize,dims)
    MAS_index, MAS_corr = MAS_function(MAS)
                    
    # we need to FT (1+delta)*V
    Vx *= (1.0 + delta)
    Vy *= (1.0 + delta)
    Vz *= (1.0 + delta)
                    
    ## compute FFT of the field (change this for double precision) ##
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
                Vx_k[kxx,kyy,kzz]    = Vx_k[kxx,kyy,kzz]*MAS_factor
                Vy_k[kxx,kyy,kzz]    = Vy_k[kxx,kyy,kzz]*MAS_factor
                Vz_k[kxx,kyy,kzz]    = Vz_k[kxx,kyy,kzz]*MAS_factor

                # compute theta for each mode: theta(k) = -ik*V(k)
                real = +(kx*Vx_k[kxx,kyy,kzz].imag + \
                         ky*Vy_k[kxx,kyy,kzz].imag + \
                         kz*Vz_k[kxx,kyy,kzz].imag)
            
                imag = -(kx*Vx_k[kxx,kyy,kzz].real + \
                         ky*Vy_k[kxx,kyy,kzz].real + \
                         kz*Vz_k[kxx,kyy,kzz].real)
                
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
