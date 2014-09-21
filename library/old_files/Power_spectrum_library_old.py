#This file contains the routines used to compute the P(k) of a given point set
#within an N-body simulation. The routines are:

#power_spectrum(pos,dims,BoxSize,shoot_noise_correction=True)
#power_spectrum_2comp(pos1,pos2,Omega1,Omega2,dims,BoxSize)
#cross_power_spectrum(pos1,pos2,dims,BoxSize)
#cross_power_spectrum_DM(pos1,pos2,posh,Omega1,Omega2,dims,BoxSize)
#power_spectrum_full_analysis(pos1,pos2,Omega1,Omega2,dims,BoxSize)


#from mpi4py import MPI
import CIC_library as CIC #version 2.0
import numpy as np
import scipy.fftpack
import scipy.weave as wv
import sys
import time

#This routine computes the P(k) in the following situations:
#1) halo catalogue from an N-body
#2) the CDM particles of an N-body simulations
#3) the NU particles of an N-body simulations
#If there are more than one particle type, use the power_spectrum_2comp routine
#pos: an array containing the positions of the particles/galaxies/halos
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
#method: method to compute the value of the overdensity in the grid points
#choose between 'CIC' or 'TSC'. By default it is 'CIC'
def power_spectrum(pos,dims,BoxSize,method='CIC',shoot_noise_correction=True):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #compute the delta (rho/mean_rho-1) in the grid points by cic interp
    delta=np.zeros(dims3,dtype=np.float32)
    if method=='CIC':
        CIC.CIC_serial(pos,dims,BoxSize,delta) #computes the density
    elif method=='TSC':
        CIC.TSC_serial(pos,dims,BoxSize,delta) #computes the density
    else:
        print 'choose method among "CIC" or "TSC"' 
    print 'numbers should be equal:',np.sum(delta,dtype=np.float64),len(pos)
    delta=delta*(dims3*1.0/len(pos))-1.0  #computes the delta
    print np.min(delta),'< delta <',np.max(delta)
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    if method=='CIC':
        [array,k]=CIC_correction(dims)
    elif method=='TSC':
        [array,k]=TSC_correction(dims)
    delta_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic
    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute delta(k)^2, delete delta(k) and delta(k)*
    print 'computing delta(k)^2'
    delta_k_conj=np.conj(delta_k)
    delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj

    #compute the P(k)=<delta_k^2>
    Pk=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta_k2)
    Pk=Pk/count

    #final processing
    bins_k=np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    n=len(pos)*1.0/BoxSize**3 #mean density
    if shoot_noise_correction:
        Pk=Pk-1.0/n #correct for the shot noise

    #compute the error on P(k)
    delta_Pk=np.sqrt(2.0/count)*(1.0+1.0/(Pk*n))*Pk

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]; delta_Pk=delta_Pk[1:]
    Pk=np.array([k,Pk,delta_Pk])

    #k1=(bins_k[:-1]*2.0*np.pi/BoxSize)[1:]
    #k2=(bins_k[1:]*2.0*np.pi/BoxSize)[1:]
    #Pk=np.array([k1,k2,Pk,delta_Pk])

    return Pk
##############################################################################

#This routine computes the P(k) when there are 2 types of particles:
#e.g. CDM+neutrinos.
#It assumes that the fraction of the first component is: Omega1/(Omega1+Omega2),
#whereas the fraction of the second component is: Omega2/(Omega1+Omega2)
#For CDM and neutrinos set Omega1=OmegaCDM and Omega2=OmegaNU
#pos1: an array containing the positions of the particle type 1 
#pos2: an array containing the positions of the particle type 2
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
def power_spectrum_2comp(pos1,pos2,Omega1,Omega2,dims,BoxSize):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 1
    delta1=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos1,dims,BoxSize,delta1) #computes the density
    print 'numbers should be equal:',np.sum(delta1,dtype=np.float64),len(pos1)
    delta1=delta1*(dims3*1.0/len(pos1))-1.0  #computes the delta
    print np.min(delta1),'< delta1 <',np.max(delta1)

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 2
    delta2=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos2,dims,BoxSize,delta2) #computes the density
    print 'numbers should be equal:',np.sum(delta2,dtype=np.float64),len(pos2)
    delta2=delta2*(dims3*1.0/len(pos2))-1.0  #computes the delta
    print np.min(delta2),'< delta2 <',np.max(delta2)
    
    #compute the total delta (rho/mean_rho-1)
    delta=np.empty(dims3,dtype=np.float32)
    delta=Omega1/(Omega1+Omega2)*delta1+Omega2/(Omega1+Omega2)*delta2
    del delta1,delta2
    print np.min(delta),'< delta <',np.max(delta)
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    delta_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic
    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute delta(k)^2, delete delta(k) and delta(k)*
    print 'computing delta(k)^2'
    delta_k_conj=np.conj(delta_k)
    delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj

    #compute the P(k)=<delta_k^2>
    Pk=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta_k2)
    Pk=Pk/count

    #final processing
    bins_k=np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #n=len(pos)*1.0/BoxSize**3 #mean density
    #if shoot_noise_correction:
    #    Pk=Pk-1.0/n #correct for the shot noise

    #compute the error on P(k)
    #delta_Pk=np.sqrt(2.0/count)*(1.0+1.0/(Pk*n))*Pk

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]  #; delta_Pk=delta_Pk[1:]
    Pk=np.array([k,Pk])
    return Pk
##############################################################################

#This routine computes the cross-P(k) of 2 types of particles:
#e.g. CDM-neutrinos, CDM-halos, NU-halos...etc
#if there are more than one fluid, and it is desired the DM-halos cross-P(k)
#use the function cross_power_spectrum_DM
#pos1: an array containing the positions of the particle type 1 
#pos2: an array containing the positions of the particle type 2
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
def cross_power_spectrum(pos1,pos2,dims,BoxSize):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 1
    delta1=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos1,dims,BoxSize,delta1) #computes the density
    print 'numbers should be equal:',np.sum(delta1,dtype=np.float64),len(pos1)
    delta1=delta1*(dims3*1.0/len(pos1))-1.0  #computes the delta
    print np.min(delta1),'< delta1 <',np.max(delta1)

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 2
    delta2=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos2,dims,BoxSize,delta2) #computes the density
    print 'numbers should be equal:',np.sum(delta2,dtype=np.float64),len(pos2)
    delta2=delta2*(dims3*1.0/len(pos2))-1.0  #computes the delta
    print np.min(delta2),'< delta2 <',np.max(delta2)
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta1=np.reshape(delta1,(dims,dims,dims))
    delta2=np.reshape(delta2,(dims,dims,dims))
    print 'Computing the FFT of the field1...'; start_fft=time.clock()
    delta1_k=scipy.fftpack.ifftn(delta1,overwrite_x=True); del delta1
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    print 'Computing the FFT of the field2...'; start_fft=time.clock()
    delta2_k=scipy.fftpack.ifftn(delta2,overwrite_x=True); del delta2
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta1_k=np.ravel(delta1_k); delta2_k=np.ravel(delta2_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    delta1_k*=array; delta2_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic
    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute delta_12(k)^2, delete delta_1(k)* and delta_2(k)*
    print 'computing delta_12(k)^2'
    delta1_k=np.conj(delta1_k)
    delta12_k2=np.real(delta1_k*delta2_k); del delta1_k,delta2_k

    #compute the P(k)=<delta_k^2>
    Pk=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta12_k2)
    Pk=Pk/count

    #final processing
    bins_k=np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #n=len(pos)*1.0/BoxSize**3 #mean density
    #if shoot_noise_correction:
    #    Pk=Pk-1.0/n #correct for the shot noise

    #compute the error on P(k)
    #delta_Pk=np.sqrt(2.0/count)*(1.0+1.0/(Pk*n))*Pk

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]  #; delta_Pk=delta_Pk[1:]
    Pk=np.array([k,Pk])
    return Pk
##############################################################################

#This routine computes the cross-P(k) of DM-halos when DM=CDM+NU
#It assumes that the fraction of the first component is: Omega1/(Omega1+Omega2),
#whereas the fraction of the second component is: Omega2/(Omega1+Omega2)
#For CDM and neutrinos set Omega1=OmegaCDM and Omega2=OmegaNU
#pos1: an array containing the positions of the particle type 1 
#pos2: an array containing the positions of the particle type 2
#posh: an array containing the positions of the halos
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
def cross_power_spectrum_DM(pos1,pos2,posh,Omega1,Omega2,dims,BoxSize):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 1
    delta1=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos1,dims,BoxSize,delta1) #computes the density
    print 'numbers should be equal:',np.sum(delta1,dtype=np.float64),len(pos1)
    delta1=delta1*(dims3*1.0/len(pos1))-1.0  #computes the delta
    print np.min(delta1),'< delta1 <',np.max(delta1)

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 2
    delta2=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos2,dims,BoxSize,delta2) #computes the density
    print 'numbers should be equal:',np.sum(delta2,dtype=np.float64),len(pos2)
    delta2=delta2*(dims3*1.0/len(pos2))-1.0  #computes the delta
    print np.min(delta2),'< delta2 <',np.max(delta2)
    
    #compute the total delta (rho/mean_rho-1)
    delta=np.empty(dims3,dtype=np.float32)
    delta=Omega1/(Omega1+Omega2)*delta1+Omega2/(Omega1+Omega2)*delta2
    del delta1,delta2
    print np.min(delta),'< delta <',np.max(delta)

    #compute the delta (rho/mean_rho-1) in the mesh points for the halos
    deltah=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(posh,dims,BoxSize,deltah) #computes the density
    print 'numbers should be equal:',np.sum(deltah,dtype=np.float64),len(posh)
    deltah=deltah*(dims3*1.0/len(posh))-1.0  #computes the delta
    print np.min(deltah),'< deltah <',np.max(deltah)
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    deltah=np.reshape(deltah,(dims,dims,dims))
    print 'Computing the FFT of the DM field...'; start_fft=time.clock()
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    print 'Computing the FFT of the halos field...'; start_fft=time.clock()
    deltah_k=scipy.fftpack.ifftn(deltah,overwrite_x=True); del deltah
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k); deltah_k=np.ravel(deltah_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    delta_k*=array; deltah_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic
    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute delta_12(k)^2, delete delta_1(k)* and delta_2(k)*
    print 'computing delta_12(k)^2'
    delta_k=np.conj(delta_k)
    delta12_k2=np.real(delta_k*deltah_k); del delta_k,deltah_k

    #compute the P(k)=<delta_k^2>
    Pk=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta12_k2)
    Pk=Pk/count; del delta12_k2

    #final processing
    bins_k=np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #n=len(pos)*1.0/BoxSize**3 #mean density
    #if shoot_noise_correction:
    #    Pk=Pk-1.0/n #correct for the shot noise

    #compute the error on P(k)
    #delta_Pk=np.sqrt(2.0/count)*(1.0+1.0/(Pk*n))*Pk

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]  #; delta_Pk=delta_Pk[1:]
    Pk=np.array([k,Pk])
    return Pk
##############################################################################

#This routine is thought to be used when having more than 1 particle type.
#It computes the P(k) for each component, the P(k) for the total density
#field and the cross-P(k).
#For the total P(k) it assumes that the fraction of the first component is:
#Omega1/(Omega1+Omega2), whereas the fraction of the second component is:
#Omega2/(Omega1+Omega2). For CDM and neutrinos set Omega1=OmegaCDM and 
#Omega2=OmegaNU
#pos1: an array containing the positions of the particle type 1 
#pos2: an array containing the positions of the particle type 2
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
#SNC1: Correct component-1 for shot-noise (True or False). False by default
#SNC2: Correct component-1 for shot-noise (True or False). False by default
class power_spectrum_full_analysis:
    def __init__(self,pos1,pos2,Omega1,Omega2,dims,BoxSize,SNC1=False,SNC2=False):

        dims3=dims**3; start_time=time.clock()
        bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

        #compute k-bins
        bins_k=np.linspace(0.0,bins_r,bins_r+1)

        #compute the delta in the mesh points for the component 1
        delta1=np.zeros(dims3,dtype=np.float32)
        CIC.CIC_serial(pos1,dims,BoxSize,delta1) #computes the density
        print np.sum(delta1,dtype=np.float64),'should be equal to',len(pos1)
        delta1=delta1*(dims3*1.0/len(pos1))-1.0  #computes the delta
        print np.min(delta1),'< delta1 <',np.max(delta1)

        #compute the delta in the mesh points for the component 2
        delta2=np.zeros(dims3,dtype=np.float32)
        CIC.CIC_serial(pos2,dims,BoxSize,delta2) #computes the density
        print np.sum(delta2,dtype=np.float64),'should be equal to',len(pos2)
        delta2=delta2*(dims3*1.0/len(pos2))-1.0  #computes the delta
        print np.min(delta2),'< delta2 <',np.max(delta2)
    
        #compute the total delta (rho/mean_rho-1). Formula easily obtained.
        delta=np.empty(dims3,dtype=np.float32)
        delta=Omega1/(Omega1+Omega2)*delta1+Omega2/(Omega1+Omega2)*delta2
        print np.min(delta),'< delta <',np.max(delta)

########################## FIRST COMPONENT #################################
        #FFT of the delta1 field (scipy.fftpack seems superior to numpy.fft)
        delta1=np.reshape(delta1,(dims,dims,dims))
        print 'Computing the FFT of the field...'; start_fft=time.clock()
        delta1_k=scipy.fftpack.ifftn(delta1,overwrite_x=True); del delta1
        print 'done: time taken for computing the FFT=',time.clock()-start_fft
        delta1_k=np.ravel(delta1_k)

        #apply the cic correction to the modes
        #at the same time computes the modulus of k at each mesh point
        print 'Applying CIC correction to the modes...';start_cic=time.clock()
        #because we are using complex numbers: 1) compute the correction over a
        #np.ones(dims3) array  2) multiply the results
        [array,k]=CIC_correction(dims)
        delta1_k*=array
        print 'done: time taken for the correction=   ',time.clock()-start_cic
        #count modes
        count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

        #compute bins in k-space and give them physical units (h/Mpc), (h/kpc)
        k=k.astype(np.float64) #to avoid problems with np.histogram
        k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count
        self.k=(2.0*np.pi*k/BoxSize)[1:] #ignore first bin

        #compute bins in k-space and give them physical units (h/Mpc), (h/kpc)
        k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count
        self.k=k[1:] #ignore first bin

        #compute delta_1(k)^2, delete delta_1(k) and keep delta_1(k)*
        print 'computing delta_1(k)^2'
        delta1_k_conj=np.conj(delta1_k)
        delta1_k2=np.real(delta1_k*delta1_k_conj); del delta1_k

        #compute the P_1(k)=<delta_1(k)^2>
        Pk1=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta1_k2)
        Pk1=Pk1/count; del delta1_k2
        #give physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
        Pk1=Pk1*BoxSize**3 

        #correct for shot-noise. Compute error
        n=len(pos1)*1.0/BoxSize**3 #mean density
        if SNC1:
            self.Pk1=(Pk1-1.0/n)[1:]
        else:
            self.Pk1=Pk1[1:]
        self.dPk1=(np.sqrt(2.0/count)*(1.0+1.0/(Pk1*n))*Pk1)[1:]
########################## SECOND COMPONENT ################################
        #FFT of the delta2 field (scipy.fftpack seems superior to numpy.fft)
        delta2=np.reshape(delta2,(dims,dims,dims))
        print 'Computing the FFT of the field...'; start_fft=time.clock()
        delta2_k=scipy.fftpack.ifftn(delta2,overwrite_x=True); del delta2
        print 'done: time taken for computing the FFT=',time.clock()-start_fft
        delta2_k=np.ravel(delta2_k)

        #apply the cic correction to the modes
        print 'Applying CIC correction to the modes...';start_cic=time.clock()
        delta2_k*=array
        print 'done: time taken for the correction=   ',time.clock()-start_cic

        #compute delta_2(k)^2, delete delta_2(k)* and keep delta_2(k)
        print 'computing delta_2(k)^2'
        delta2_k_conj=np.conj(delta2_k)
        delta2_k2=np.real(delta2_k*delta2_k_conj); del delta2_k_conj

        #compute the P_2(k)=<delta_2(k)^2>
        Pk2=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta2_k2)
        Pk2=Pk2/count; del delta2_k2
        #give physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
        Pk2=Pk2*BoxSize**3 

        #correct for shot-noise and compute error
        n=len(pos1)*1.0/BoxSize**3 #mean density
        if SNC2:
            self.Pk2=(Pk2-1.0/n)[1:]
        else:
            self.Pk2=Pk2[1:]
        self.dPk2=(np.sqrt(2.0/count)*(1.0+1.0/(Pk2*n))*Pk2)[1:]
########################### CROSS-COMPONENT ################################
        #compute delta_12(k)^2, delete delta_1(k)* and delta_2(k)
        print 'computing delta_12(k)^2'
        delta12_k2=np.real(delta1_k_conj*delta2_k); del delta1_k_conj,delta2_k

        #compute the P(k)=<delta_12(k)^2>
        Pk12=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta12_k2)
        Pk12=Pk12/count; del delta12_k2
        #give physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
        Pk12*=BoxSize**3
        self.Pk12=Pk12[1:]
########################### TOTAL COMPONENT ################################
        #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
        delta=np.reshape(delta,(dims,dims,dims))
        print 'Computing the FFT of the field...'; start_fft=time.clock()
        delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
        print 'done: time taken for computing the FFT=',time.clock()-start_fft
        delta_k=np.ravel(delta_k)

        #apply the cic correction to the modes
        print 'Applying CIC correction to the modes...';start_cic=time.clock()
        delta_k*=array; del array
        print 'done: time taken for the correction=   ',time.clock()-start_cic

        #compute delta(k)^2, delete delta(k) and delta(k)*
        print 'computing delta(k)^2'
        delta_k_conj=np.conj(delta_k)
        delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj

        #compute the P(k)=<delta_k^2>
        Pk=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta_k2)
        Pk=Pk/count; del delta_k2
        #give physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
        Pk*=BoxSize**3
        self.Pk=Pk[1:]
########################### CONSISTENCY CHECK ##############################
        #compute the total P(k) through the individual P(k)s and the cross
        #and check whether it is the same as using the P(k) computed using
        #the total density field in the cic interpolation
        total_Pk=Omega1**2*Pk1+Omega2**2*Pk2+2.0*Omega1*Omega2*Pk12
        total_Pk/=(Omega1+Omega2)**2
        self.check=((total_Pk-Pk)/Pk)[1:]

        print 'time used to perform calculation=',time.clock()-start_time,' s'
##############################################################################

#This routine computes the P(k) if the values of delta(r) are given.
#It is useful when the values of delta(r) have to be computed for a particular
#quantity such as delta(r)=nH0(r)/<nH0>-1

#This routine by default does not perform any shot-noise correction!!!
#It also does not compute any error for the power spectrum
def power_spectrum_given_delta(delta,dims,BoxSize,do_CIC_correction=True):

    start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    if do_CIC_correction:
        delta_k*=array; del array
    else:
        del array; print 'CIC correction not performed'
    print 'done: time taken for the correction=   ',time.clock()-start_cic
    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute delta(k)^2, delete delta(k) and delta(k)*
    print 'computing delta(k)^2'
    delta_k_conj=np.conj(delta_k)
    delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj
    #delta_k2=np.real(delta_k); del delta_k,delta_k_conj

    #compute the P(k)=<delta_k^2>
    Pk=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta_k2)
    Pk=Pk/count

    #final processing
    bins_k=np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]
    Pk=np.array([k,Pk])

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    return Pk
##############################################################################

#This routine computes the cross-P(k) if the values of delta_1(r) and 
#delta_2(r) are given.
#It is useful when the values of delta(r) have to be computed for a particular
#quantity such as delta(r)=nH0(r)/<nH0>-1

#This routine by default does not perform any shot-noise correction!!!
#It also does not compute any error for the power spectrum
def cross_power_spectrum_given_delta(delta1,delta2,dims,BoxSize):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta1=np.reshape(delta1,(dims,dims,dims))
    delta2=np.reshape(delta2,(dims,dims,dims))
    print 'Computing the FFT of the field1...'; start_fft=time.clock()
    delta1_k=scipy.fftpack.ifftn(delta1,overwrite_x=True); del delta1
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    print 'Computing the FFT of the field2...'; start_fft=time.clock()
    delta2_k=scipy.fftpack.ifftn(delta2,overwrite_x=True); del delta2
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta1_k=np.ravel(delta1_k); delta2_k=np.ravel(delta2_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    delta1_k*=array; delta2_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic
    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute delta_12(k)^2, delete delta_1(k)* and delta_2(k)*
    print 'computing delta_12(k)^2'
    delta1_k=np.conj(delta1_k)
    delta12_k2=np.real(delta1_k*delta2_k); del delta1_k,delta2_k

    #compute the P(k)=<delta_k^2>
    Pk=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta12_k2)
    Pk=Pk/count

    #final processing
    bins_k=np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #n=len(pos)*1.0/BoxSize**3 #mean density
    #if shoot_noise_correction:
    #    Pk=Pk-1.0/n #correct for the shot noise

    #compute the error on P(k)
    #delta_Pk=np.sqrt(2.0/count)*(1.0+1.0/(Pk*n))*Pk

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]  #; delta_Pk=delta_Pk[1:]
    Pk=np.array([k,Pk])
    return Pk
##############################################################################

#This routine computes the P(k) in (k_perp,k_par) or (|k|,\mu) bins
#pos: an array containing the positions of the particles/galaxies/halos
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
#do_k_mu: if True computes the Power spectrum in (|k|,\mu) bins
#         if False the bins are in (k_perp,k_par)
#shoot_noise_correction: correct the power spectrum for shot-noise
#This routine returns the intervals in |k| and \mu used 
#(not the mean of interval!!) and the value of P(k) on each interval
def power_spectrum_2D(pos,dims,BoxSize,bins_mu,do_k_mu=False,
                      shoot_noise_correction=False):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #compute the delta (rho/mean_rho-1) in the grid points by cic interp
    delta=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos,dims,BoxSize,delta) #computes the density
    print 'numbers should be equal:',np.sum(delta,dtype=np.float64),len(pos)
    delta=delta*(dims3*1.0/len(pos))-1.0  #computes the delta
    print np.min(delta),'< delta <',np.max(delta)
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    delta_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic

    #compute delta(k)^2, delete delta(k) and delta(k)*
    print 'computing delta(k)^2'
    delta_k_conj=np.conj(delta_k)
    delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj

    #compute the P(k)=<delta_k^2>
    if do_k_mu:
        Pk,number_of_modes=modes_k_mu(dims,delta_k2,bins_mu)
    else:
        Pk=modes(dims,delta_k2)[0]

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    #we should add 1 since we want to equal the number of intervals
    if do_k_mu:
        bins_k=np.arange(int(np.sqrt(3*int(0.5*(dims+1))**2))+1+1)
        bins_mu=np.linspace(-1.0,1.0,Pk.shape[1]+1)
        
        k=2.0*np.pi*bins_k/BoxSize
        mu=bins_mu
        #k=2.0*np.pi*(0.5*(bins_k[:-1]+bins_k[1:]))/BoxSize
        #mu=0.5*(bins_mu[:-1]+bins_mu[1:])

    else:
        bins_perp=np.arange(int(np.sqrt(2*(dims/2+1)**2))+1+1)
        bins_par=np.arange(dims/2+1+1)

        k_perp=2.0*np.pi*(0.5*(bins_perp[:-1]+bins_perp[1:]))/BoxSize
        k_par=2.0*np.pi*(0.5*(bins_par[:-1]+bins_par[1:]))/BoxSize

    n=len(pos)*1.0/BoxSize**3 #mean density
    if shoot_noise_correction:
        Pk=Pk-1.0/n #correct for the shot noise

    #compute the error on P(k)
    #delta_Pk=np.sqrt(2.0/count)*(1.0+1.0/(Pk*n))*Pk

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #Note that for the bins in (|k|,mu) we return the intervals, not the 
    #mean of the interval
    if do_k_mu:
        Pk=np.array([k,mu,Pk,number_of_modes])
    else:
        Pk=np.array([k_perp,k_par,Pk])

    return Pk
##############################################################################

#This routine computes the cross-P(k) of 2 types of particles in (k_perp,k_par):
#e.g. CDM-neutrinos, CDM-halos, NU-halos...etc
#if there are more than one fluid, and it is desired the DM-halos cross-P(k)
#use the function cross_power_spectrum_DM
#pos1: an array containing the positions of the particle type 1 
#pos2: an array containing the positions of the particle type 2
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
def cross_power_spectrum_2D(pos1,pos2,dims,BoxSize):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 1
    delta1=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos1,dims,BoxSize,delta1) #computes the density
    print 'numbers should be equal:',np.sum(delta1,dtype=np.float64),len(pos1)
    delta1=delta1*(dims3*1.0/len(pos1))-1.0  #computes the delta
    print np.min(delta1),'< delta1 <',np.max(delta1)

    #compute the delta (rho/mean_rho-1) in the mesh points for the component 2
    delta2=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos2,dims,BoxSize,delta2) #computes the density
    print 'numbers should be equal:',np.sum(delta2,dtype=np.float64),len(pos2)
    delta2=delta2*(dims3*1.0/len(pos2))-1.0  #computes the delta
    print np.min(delta2),'< delta2 <',np.max(delta2)
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta1=np.reshape(delta1,(dims,dims,dims))
    delta2=np.reshape(delta2,(dims,dims,dims))
    print 'Computing the FFT of the field1...'; start_fft=time.clock()
    delta1_k=scipy.fftpack.ifftn(delta1,overwrite_x=True); del delta1
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    print 'Computing the FFT of the field2...'; start_fft=time.clock()
    delta2_k=scipy.fftpack.ifftn(delta2,overwrite_x=True); del delta2
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta1_k=np.ravel(delta1_k); delta2_k=np.ravel(delta2_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    delta1_k*=array; delta2_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic

    #compute delta_12(k)^2, delete delta_1(k)* and delta_2(k)*
    print 'computing delta_12(k)^2'
    delta1_k=np.conj(delta1_k)
    delta12_k2=np.real(delta1_k*delta2_k); del delta1_k,delta2_k

    #compute the P(k)=<delta12_k^2>
    Pk=modes(dims,delta12_k2)[0]

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    #we should add 1 since we want to equal the number of intervals
    bins_perp=np.arange(int(np.sqrt(2*(dims/2+1)**2))+1+1)
    bins_par=np.arange(dims/2+1+1)

    k_perp=2.0*np.pi*(0.5*(bins_perp[:-1]+bins_perp[1:]))/BoxSize
    k_par=2.0*np.pi*(0.5*(bins_par[:-1]+bins_par[1:]))/BoxSize

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #ignore the first bin
    #k_perp=k_perp[1:]; k_par=k_par[1:]
    #Pk=Pk[1:]; Pk=#delta_Pk=delta_Pk[1:]
    Pk=np.array([k_perp,k_par,Pk])

    return Pk
##############################################################################

#This routine computes P_l(k) where l=0,2 or l=4 
#pos: an array containing the positions of the particles/galaxies/halos
#dims: the number of points per dimension in the grid
#BoxSize: Size of the simulation. Units must be equal to those of pos
#ell: multipole wanted. Choose among 0,2 or 4
#axis: axis over which compute the multipole, 0(x), 1(y) or 2(z)
#shoot_noise_correction: correct the power spectrum for shot-noise
def multipole(pos,dims,BoxSize,ell,axis,shoot_noise_correction=False):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #compute delta (rho/<rho>-1) in the grid points by cic interp
    delta=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos,dims,BoxSize,delta) #computes the density
    print 'numbers should be equal:',np.sum(delta,dtype=np.float64),len(pos)
    delta=delta*(dims3*1.0/len(pos))-1.0  #computes the delta
    print np.min(delta),'< delta <',np.max(delta)
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #apply the cic correction to the modes
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #because we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    [array,k]=CIC_correction(dims)
    delta_k*=array; del array
    print 'done: time taken for the correction=   ',time.clock()-start_cic

    #compute delta(k)^2, delete delta(k) and delta(k)*
    print 'computing delta(k)^2'
    delta_k_conj=np.conj(delta_k)
    delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj

    #compute the P_l(k)=(2*l+1)<delta_k^2(k,mu)*L_l(mu)>
    Pk=modes_multipole(dims,delta_k2,ell,axis)[0]

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    #we should add 1 since we want to equal the number of intervals
    bins_k=np.arange(int(np.sqrt(3*int(0.5*(dims+1))**2))+1+1)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    n=len(pos)*1.0/BoxSize**3 #mean density
    if shoot_noise_correction:
        Pk=Pk-1.0/n #correct for the shot noise

    #compute the error on P(k)
    #delta_Pk=np.sqrt(2.0/count)*(1.0+1.0/(Pk*n))*Pk

    print 'time used to perform calculation=',time.clock()-start_time,' s'

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]; #delta_Pk=delta_Pk[1:]
    Pk=np.array([k,Pk])

    return Pk
##############################################################################


#this function computes:
#1) the CIC correction to the modes 
#2) the module of k for a given point in the fourier grid
def CIC_correction(dims):
    array=np.empty(dims**3,dtype=np.float32)
    mod_k=np.empty(dims**3,dtype=np.float32)

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int dims3=dims2*dims;
       int middle=dims/2;
       int i,j,k;
       float value_i,value_j,value_k;

       for (long l=0;l<dims3;l++){
           i=l/dims2;
           j=(l%dims2)/dims;
           k=(l%dims2)%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;
           k = (k>middle) ? k-dims : k;

           value_i = (i==0) ? 1.0 : pow((i*M_PI/dims)/sin(i*M_PI/dims),2);
           value_j = (j==0) ? 1.0 : pow((j*M_PI/dims)/sin(j*M_PI/dims),2);
           value_k = (k==0) ? 1.0 : pow((k*M_PI/dims)/sin(k*M_PI/dims),2);

           array(l)=value_i*value_j*value_k;
           mod_k(l)=sqrt(i*i+j*j+k*k);
       } 
    """
    wv.inline(code,['dims','array','mod_k'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])
    return [array,mod_k]

#this function computes:
#1) the TSC correction to the modes 
#2) the module of k for a given point in the fourier grid
def TSC_correction(dims):
    array=np.empty(dims**3,dtype=np.float32)
    mod_k=np.empty(dims**3,dtype=np.float32)

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int dims3=dims2*dims;
       int middle=dims/2;
       int i,j,k;
       float value_i,value_j,value_k;

       for (long l=0;l<dims3;l++){
           i=l/dims2;
           j=(l%dims2)/dims;
           k=(l%dims2)%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;
           k = (k>middle) ? k-dims : k;

           value_i = (i==0) ? 1.0 : pow((i*M_PI/dims)/sin(i*M_PI/dims),3);
           value_j = (j==0) ? 1.0 : pow((j*M_PI/dims)/sin(j*M_PI/dims),3);
           value_k = (k==0) ? 1.0 : pow((k*M_PI/dims)/sin(k*M_PI/dims),3);

           array(l)=value_i*value_j*value_k;
           mod_k(l)=sqrt(i*i+j*j+k*k);
       } 
    """
    wv.inline(code,['dims','array','mod_k'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])
    return [array,mod_k]


#This function computes an histogram of the values of array
#between minimum and maximum, having bins bins.
#Can also be used to create an histogram of the weighted values of array
def lin_histogram(bins,minimum,maximum,array,weights=None):
    #the elements which are equal to the maximum may not lie in the bins
    #we create an extra bin to place those elements there
    #at the end we put those elements in the last bin of the histogram

    histo=np.zeros(bins+1,np.float32) #allow an extra bin for elements at edge
    length=array.shape[0]

    support = "#include <math.h>"

    code1 = """
    int index;
    float delta=(maximum-minimum)*1.0/bins;  /* the size of a bin */

    for (int k=0;k<length;k++){
        index=(int)(array(k)-minimum)/delta;
        if (index>=0 && index<=bins)
            histo(index)+=1.0;
    }
    histo(bins-1)+=histo(bins); 
    """

    code2 = """
    int index;
    float delta=(maximum-minimum)*1.0/bins;   /* the size of a bin */

    for (int k=0;k<length;k++){
        //if (k%1000000==0)
           //printf("%d\\n",k);
        index=(int)(array(k)-minimum)/delta;
        if (index>=0 && index<=bins)
            histo(index)+=weights(k);
    }
    histo(bins-1)+=histo(bins); 
    """
    
    if weights==None:
        wv.inline(code1,['length','minimum','maximum','bins','histo','array'],
                  type_converters = wv.converters.blitz,
                  support_code = support,libraries = ['m'],
                  extra_compile_args =['-O3'])
    else:
        if length!=weights.shape[0]:
            print 'the lengths of the array and its weights must be the same'
            sys.exit()

        wv.inline(code2,['length','minimum','maximum','bins','histo','array','weights'],type_converters = wv.converters.blitz,              
                  support_code = support,libraries = ['m'],
                  extra_compile_args =['-O3'])

    return histo[:-1]


#this function computes:
#1) The sum of |delta(k)^2| for each bin in (k_perp,k_par)
#2) The number of modes on each bin (k_perp,k_par)
#The function returns the power spectrum = sum_modes |delta(k)^2|/ #_modes
def modes(dims,delta_k2):

    if len(delta_k2)!=dims**3:
        print 'sizes are different!!'
        sys.exit()

    bins_perp=int(np.sqrt(2*(dims/2+1)**2))+1
    bins_par=dims/2+1

    number_of_modes=np.zeros((bins_perp,bins_par),dtype=np.float32)
    Pk=np.zeros((bins_perp,bins_par),dtype=np.float32)

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int dims3=dims2*dims;
       int middle=dims/2;
       int i,j,k,k_perp,k_par;

       for (long l=0;l<dims3;l++){
           i=l/dims2;
           j=(l%dims2)/dims;
           k=(l%dims2)%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;
           k = (k>middle) ? k-dims : k;

           k_perp = (int)sqrt(i*i+j*j);
           k_par  = abs(k);

           number_of_modes(k_perp,k_par)+=1.0;
           Pk(k_perp,k_par)+=delta_k2(l);
       } 

       for (i=0;i<bins_perp;i++){
           for (j=0;j<bins_par;j++){
              if (number_of_modes(i,j)>0){
                  Pk(i,j)=Pk(i,j)/number_of_modes(i,j);
              }
           }
       }
    """
    wv.inline(code,['dims','delta_k2','number_of_modes','Pk',
                    'bins_perp','bins_par'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return Pk,number_of_modes

#this function computes:
#1) The sum of |delta(k)^2| for each bin in (|k|,\mu)
#2) The number of modes on each bin (|k|,\mu)
#The function returns the power spectrum = sum_modes |delta(k)^2|/ #_modes
def modes_k_mu(dims,delta_k2,bins_mu,axis):

    if len(delta_k2)!=dims**3:
        print 'sizes are different!!'
        sys.exit()

    bins_k=int(np.sqrt(3*int(0.5*(dims+1))**2))+1
    #bins_mu=dims/2+1

    #bins_perp=int(np.sqrt(2*(dims/2+1)**2))+1
    #bins_par=dims/2+1

    number_of_modes=np.zeros((bins_k,bins_mu),dtype=np.float32)
    Pk=np.zeros((bins_k,bins_mu),dtype=np.float32)

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int dims3=dims2*dims;
       int middle=dims/2;
       int i,j,k,k_bin,mu_bin;
       float mu,mod_k;

       for (long l=0;l<dims3;l++){
           i=l/dims2;
           j=(l%dims2)/dims;
           k=(l%dims2)%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;
           k = (k>middle) ? k-dims : k;

           mod_k = sqrt(i*i+j*j+k*k);
           k_bin=(int)mod_k;

           if (mod_k>0.0){
              switch (axis){
              case 0:
                  mu = i/mod_k;
                  break;
              case 1:
                  mu = j/mod_k;
                  break;
              case 2:
                  mu = k/mod_k;
                  break;
              default:
                  printf("Error with the chosen axis!!\\n");
                  break;
              }
           } 
           else
              mu = 0.0;

           mu_bin=(int)((mu+1.0)/(2.0/bins_mu));
           if (mu_bin==bins_mu)
              mu_bin=bins_mu-1;

           number_of_modes(k_bin,mu_bin)+=1.0;
           Pk(k_bin,mu_bin)+=delta_k2(l);
       } 

       for (i=0;i<bins_k;i++){
           for (j=0;j<bins_mu;j++){
              if (number_of_modes(i,j)>0){
                  Pk(i,j)=Pk(i,j)/number_of_modes(i,j);
              }
           }
       }
    """
    wv.inline(code,['dims','delta_k2','number_of_modes','Pk',
                    'bins_k','bins_mu','axis'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return Pk,number_of_modes

#this function computes:
#1) The sum of |delta(k)^2*P_l(\mu)| for each bin in k
#2) The number of modes on each k-bin 
#The function returns the power spectrum = sum_modes |delta(k)^2*P_l(\mu)|/ #_modes
def modes_multipole(dims,delta_k2,ell,axis):

    if len(delta_k2)!=dims**3:
        print 'sizes are different!!'
        sys.exit()

    bins_k=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    number_of_modes=np.zeros(bins_k,dtype=np.float32)
    Pk=np.zeros(bins_k,dtype=np.float32)

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int dims3=dims2*dims;
       int middle=dims/2;
       int i,j,k,k_bin,mu_bin;
       float mu,mod_k,P_ell;

       for (long l=0;l<dims3;l++){
           i=l/dims2;
           j=(l%dims2)/dims;
           k=(l%dims2)%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;
           k = (k>middle) ? k-dims : k;

           mod_k = sqrt(i*i+j*j+k*k);
           k_bin=(int)mod_k;
           if (mod_k>0.0){
              switch (axis){
              case 0:
                  mu = i/mod_k;
                  break;
              case 1:
                  mu = j/mod_k;
                  break;
              case 2:
                  mu = k/mod_k;
                  break;
              default:
                  printf("Error with the chosen axis!!\\n");
                  break;
              }
           }
           else
              mu = 0.0;

           switch (ell){
           case 0:
               P_ell=1.0;
               break;
           case 2:
               P_ell=0.5*(3.0*mu*mu-1.0);
               break;
           case 4:
               P_ell=(35.0*mu*mu*mu*mu-30.0*mu*mu+3.0)/8.0;
               break;
           default:
               printf("Error with the chosen ell!!\\n");
               break;
           }
     
           number_of_modes(k_bin)+=1.0;
           Pk(k_bin)+=(2.0*ell+1.0)*delta_k2(l)*P_ell;
       } 

       for (i=0;i<bins_k;i++){
           if (number_of_modes(i)>0)
               Pk(i)=Pk(i)/number_of_modes(i);
       }
    """
    wv.inline(code,['dims','delta_k2','number_of_modes','Pk',
                    'bins_k','ell','axis'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return Pk,number_of_modes

#this function computes:
#1) The value of |k| 
#2) The angle theta with respect to the z-direction
def k_theta(dims):

    mod_k=np.empty(dims**3,dtype=np.float32)
    theta=np.empty(dims**3,dtype=np.float32)

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int dims3=dims2*dims;
       int middle=dims/2;
       int i,j,k;

       for (long l=0;l<dims3;l++){
           i=l/dims2;
           j=(l%dims2)/dims;
           k=(l%dims2)%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;
           k = (k>middle) ? k-dims : k;

           mod_k(l)=sqrt(i*i+j*j+k*k);
           theta(l)=atan2(sqrt(i*i+j*j),abs(k));
       } 
    """
    wv.inline(code,['dims','mod_k','theta'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return mod_k,theta



########################### EXAMPLE OF USAGE ###########################
#### one component (CDM, NU or halos): power_spectrum ####
"""
BoxSize=1000.0 #Mpc/h
dims=256 #number of points in the grid in each direction
n=256**3 #number of particles in the catalogue

pos=(np.random.random((n,3))*BoxSize).astype(np.float32) #positions in Mpc/h

Pk=power_spectrum(pos,dims,BoxSize,method='CIC')
Pk=power_spectrum(pos,dims,BoxSize,method='TSC')

print Pk

#f=open('borrar.dat','w')
#for i in range(len(Pk[0])):
#    f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+str(Pk[2][i])+'\n')
#f.close()
"""

#### 2 components (CDM+NU, CDM+Baryons, ...): power_spectrum_2comp ####
"""
BoxSize=1000.0 #Mpc/h
dims=512 #number of points in the grid in each direction
n1=512**3 #number of particles type-1 in the catalogue
n2=512**3 #number of particles type-2 in the catalogue
Omega1=0.25 #the Omega of the particles type-1
Omega2=0.01 #the Omega of the particles type-2

pos1=(np.random.random((n1,3))*BoxSize).astype(np.float32) #positions in Mpc/h
pos2=(np.random.random((n2,3))*BoxSize).astype(np.float32) #positions in Mpc/h

Pk=power_spectrum_2comp(pos1,pos2,Omega1,Omega2,dims,BoxSize)

print Pk

#f=open('borrar.dat','w')
#for i in range(len(Pk[0])):
#    f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
#f.close()
"""

#### cross-P(k) (CDM-NU, CDM-halos, NU-halos...): cross_power_spectrum ####
"""
BoxSize=1000.0 #Mpc/h
dims=512 #number of points in the grid in each direction
n1=512**3 #number of particles type-1 in the catalogue
n2=512**3 #number of particles type-2 in the catalogue

pos1=(np.random.random((n1,3))*BoxSize).astype(np.float32) #positions in Mpc/h
pos2=(np.random.random((n2,3))*BoxSize).astype(np.float32) #positions in Mpc/h

Pk=cross_power_spectrum(pos1,pos2,dims,BoxSize)

print Pk

f=open('borrar.dat','w')
for i in range(len(Pk[0])):
    f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
f.close()
"""

#### cross-P(k) (DM-halos): cross_power_spectrum_DM ####
"""
BoxSize=1000.0 #Mpc/h
dims=512 #number of points in the grid in each direction
n1=512**3 #number of particles type-1 in the catalogue
n2=512**3 #number of particles type-2 in the catalogue
n3=512**3 #number of halos in the catalogue

Omega1=0.25 #the Omega of the particles type-1
Omega2=0.01 #the Omega of the particles type-2

pos1=(np.random.random((n1,3))*BoxSize).astype(np.float32) #positions in Mpc/h
pos2=(np.random.random((n2,3))*BoxSize).astype(np.float32) #positions in Mpc/h
pos3=(np.random.random((n3,3))*BoxSize).astype(np.float32) #positions in Mpc/h

Pk=cross_power_spectrum_DM(pos1,pos2,pos3,Omega1,Omega2,dims,BoxSize)

print Pk

f=open('borrar.dat','w')
for i in range(len(Pk[0])):
    f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
f.close()
"""

#### full analysis (CDM-NU): power_spectrum_full_analysis ####
#it computes the P(k) for the CDM and the NU alone, correcting for the shot 
#noise and giving the error. It also computes the CDM-NU cross-P(k), without 
#correction. Finally it computes P(k) for the total CDM+NU field.
"""
BoxSize=1000.0 #Mpc/h
dims=512 #number of points in the grid in each direction
n1=512**3 #number of particles type-1 in the catalogue
n2=512**3 #number of particles type-2 in the catalogue
Omega1=0.25 #the Omega of the particles type-1
Omega2=0.01 #the Omega of the particles type-2

pos1=(np.random.random((n1,3))*BoxSize).astype(np.float32) #positions in Mpc/h
pos2=(np.random.random((n2,3))*BoxSize).astype(np.float32) #positions in Mpc/h

A=power_spectrum_full_analysis(pos1,pos2,Omega1,Omega2,dims,BoxSize,SNC1=True,SNC2=True)

k=A.k
Pk1=A.Pk1; dPk1=A.dPk1
Pk2=A.Pk2; dPk2=A.dPk2
Pk12=A.Pk12
Pk=A.Pk

f=open('borrar.dat','w')
g=open('borrar1.dat','w')
h=open('borrar2.dat','w')
l=open('borrar3.dat','w')
for i in range(len(k)):
    f.write(str(k[i])+' '+str(Pk1[i])+' '+str(dPk1[i])+'\n')
    g.write(str(k[i])+' '+str(Pk2[i])+' '+str(dPk2[i])+'\n')
    h.write(str(k[i])+' '+str(Pk12[i])+'\n')
    l.write(str(k[i])+' '+str(Pk[i])+'\n')
f.close()
g.close()
h.close()
l.close()

print np.min(A.check),'< diff <',np.max(A.check)
"""


#### one component (CDM, NU or halos): power_spectrum in (|k|,\mu) bins ####
"""
BoxSize=1000.0 #Mpc/h
dims=256 #number of points in the grid in each direction
n=256**3 #number of particles in the catalogue

pos=(np.random.random((n,3))*BoxSize).astype(np.float32) #positions in Mpc/h

[k,mu,Pk,number_of_modes]=power_spectrum_2D(pos,dims,BoxSize,do_k_mu=True,
                                            shoot_noise_correction=False)

print Pk[-10]
print len(Pk)
print mu
print k

#f=open('borrar.dat','w')
#for i in range(len(Pk[0])):
#    f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+str(Pk[2][i])+'\n')
#f.close()
"""


#### multipole (CDM) ####
"""
BoxSize=1000.0 #Mpc/h
dims=256 #number of points in the grid in each direction
n=256**3 #number of particles in the catalogue

ell=0
axis=2

pos=(np.random.random((n,3))*BoxSize).astype(np.float32) #positions in Mpc/h

[k,Pk]=multipole(pos,dims,BoxSize,ell,shoot_noise_correction=False)

print k
print Pk
"""
