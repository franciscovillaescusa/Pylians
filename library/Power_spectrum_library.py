################################################################################
################################################################################

#This library contains the routines needed to compute power spectra.

############## AVAILABLE ROUTINES ##############
#power_spectrum_given_delta
      #CIC_correction
      #TSC_correction
      #lin_histogram
#cross_power_spectrum_given_delta
#power_spectrum_2D
      #modes_k_mu
      #modes
#cross_power_spectrum_2D
#multipole
      #modes_multipole
#EH_Pk
################################################

######## COMPILATION ##########
#If the library needs to be compiled type: 
#python Power_spectrum_library.py compile
###############################

#IMPORTANT!! If the c/c++ functions need to be modified, the code has to be
#compiled by calling those functions within this file, otherwise it gives errors

################################################################################
################################################################################


import CIC_library as CIC
import numpy as np
import scipy.fftpack
import scipy.weave as wv
import mass_function_library as MFL
import sys
import time


###############################################################################
#This routine computes the P(k) if the values of delta(r) are given.
#It is useful when the values of delta(r) have to be computed for a particular
#quantity such as delta_HI(r)=HI(r)/<HI>-1
#This routine by default does not perform any shot-noise correction!!!
#It also does not compute any error for the power spectrum
#delta ----------------> array containing the values of delta(r)
#dims -----------------> number of cell per dimension used to compute the P(k)
#BoxSize --------------> size of the simulation box
#aliasing_method ------> method used to compute the deltas(r): CIC, TSC, other
#Notice that if delta(r) is computed using using a different kernel (not CIC),
#as the SPH kernel for baryons, set do_CIC_correction=False
def power_spectrum_given_delta(delta,dims,BoxSize,aliasing_method='CIC'):

    start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT =',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #correct modes amplitude to account for aliasing when computing delta(r)
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #since we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    if aliasing_method in ['CIC','TSC']:
        if aliasing_method=='CIC':
            [array,k]=CIC_correction(dims)
        else:
            [array,k]=TSC_correction(dims)
        delta_k*=array; del array
    else:
        [array,k]=CIC_correction(dims); del array
        print 'aliasing correction not performed'
    print 'done: time taken for the correction =  ',time.clock()-start_cic

    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute |delta(k)|^2, delete delta(k) and delta(k)*
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
    Pk*=BoxSize**3 

    #ignore the first bin (fundamental frequency)
    k=k[1:]; Pk=Pk[1:]

    #keep only with modes below 1.1*k_Nyquist
    k_N=np.pi*dims/BoxSize; indexes=np.where(k<1.1*k_N)
    k=k[indexes]; Pk=Pk[indexes]; del indexes
    print 'time used to perform calculation=',time.clock()-start_time,' s'

    return [k,Pk]


###################################################################
#this function computes:
#1) the CIC correction to the modes amplitude 
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

###################################################################
#this function computes:
#1) the TSC correction to the modes amplitude 
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

###################################################################
#This function computes an histogram of the values of the array
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

        wv.inline(code2,['length','minimum','maximum','bins','histo','array',
                         'weights'],type_converters = wv.converters.blitz,
                  support_code = support,libraries = ['m'],
                  extra_compile_args =['-O3'])

    return histo[:-1]


###############################################################################
#This routine computes the cross-P(k) if the values of delta1(r) and delta2(r)
#are given. It is useful when the values of delta(r) have to be computed for
#a particular quantity such as delta_HI(r)=HI(r)/<HI>-1
#The routines also computes the power spectra of the fields 1 and 2, returning:
#[k,Pk12,Pk1,Pk2]
#This routine by default does not perform any shot-noise correction!!!
#It also does not compute any error for the power spectrum
#delta1 ---------------> array containing the values of delta1(r)
#delta2 ---------------> array containing the values of delta2(r)
#dims -----------------> number of cell per dimension used to compute the P(k)
#BoxSize --------------> size of the simulation box
#aliasing_method1 -----> method used to compute the delta1(r): CIC, TSC, other
#aliasing_method2 -----> method used to compute the delta2(r): CIC, TSC, other
#Notice that if delta(r) is computed using using a different kernel (not CIC),
#as the SPH kernel for baryons, set do_CIC_correction=False
def cross_power_spectrum_given_delta(delta1,delta2,dims,BoxSize,
                                     aliasing_method1='CIC',
                                     aliasing_method2='CIC'):

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

    #correct modes amplitude to account for aliasing when computing delta1(r)
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #since we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    if aliasing_method1 in ['CIC','TSC']:
        if aliasing_method1=='CIC':
            [array,k]=CIC_correction(dims)
        else:
            [array,k]=TSC_correction(dims)
        delta1_k*=array; 
        if aliasing_method1==aliasing_method2:
            delta2_k*=array
        del array
    else:
        [array,k]=CIC_correction(dims); del array
        print 'aliasing correction not performed on modes1'

    if aliasing_method1 != aliasing_method2:
        if aliasing_method1 in ['CIC','TSC']:
            if aliasing_method2=='CIC':
                [array,k]=CIC_correction(dims)
            else:
                [array,k]=TSC_correction(dims)
            delta2_k*=array
        else:
            print 'aliasing correction not performed on modes2'
    print 'done: time taken for the correction =  ',time.clock()-start_cic

    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute |delta_1(k)|^2
    print 'computing delta_1(k)^2'
    delta1_k_conj=np.conj(delta1_k)
    delta1_k2=np.real(delta1_k*delta1_k_conj); del delta1_k
    Pk1=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta1_k2)
    Pk1=Pk1/count; del delta1_k2

    #compute |delta_2(k)|^2
    print 'computing delta_2(k)^2'
    delta2_k_conj=np.conj(delta2_k)
    delta2_k2=np.real(delta2_k*delta2_k_conj); del delta2_k_conj
    Pk2=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta2_k2)
    Pk2=Pk2/count; del delta2_k2

    #compute delta_12(k)^2
    print 'computing delta_12(k)^2'
    delta12_k2=np.real(delta1_k_conj*delta2_k); del delta1_k_conj,delta2_k
    Pk12=lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta12_k2)
    Pk12=Pk12/count

    #final processing
    bins_k=np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk1=Pk1*BoxSize**3; Pk2=Pk2*BoxSize**3; Pk12=Pk12*BoxSize**3 

    #ignore the first bin
    k=k[1:]; Pk1=Pk1[1:]; Pk2=Pk2[1:]; Pk12=Pk12[1:]

    #keep only with modes below 1.1*k_Nyquist
    k_N=np.pi*dims/BoxSize; indexes=np.where(k<1.1*k_N)
    k=k[indexes]; Pk1=Pk1[indexes]; Pk2=Pk2[indexes]; Pk12=Pk12[indexes]
    del indexes
    print 'time used to perform calculation=',time.clock()-start_time,' s'

    return [k,Pk12,Pk1,Pk2]
###############################################################################


#This routine computes the P(k) in (k_perp,k_par) or (|k|,\mu) bins
#delta ----------------> array containing the values of delta(r)
#dims -----------------> number of cell per dimension used to compute the P(k)
#BoxSize --------------> size of the simulation box
#axis -----------------> axis along which compute P(|k|,mu) or P(k_perp,k_par)
#bins_mu --------------> number of bins for mu in P(|k|,mu)
#do_k_mu --------------> if True computes the Power spectrum in (|k|,\mu) bins
#                        if False the bins are in (k_perp,k_par)
#aliasing_method ------> method used to compute the deltas(r): CIC, TSC, other
#This routine returns the intervals in |k| and \mu used 
#(not the mean of interval!!) and the value of P(k) on each interval
def power_spectrum_2D(delta,dims,BoxSize,axis,bins_mu,do_k_mu=False,
                      aliasing_method='CIC'):
                      
    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #correct modes amplitude to account for aliasing when computing delta(r)
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #since we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    if aliasing_method in ['CIC','TSC']:
        if aliasing_method=='CIC':
            [array,k]=CIC_correction(dims)
        else:
            [array,k]=TSC_correction(dims)
        delta_k*=array; del array
    else:
        [array,k]=CIC_correction(dims); del array
        print 'aliasing correction not performed'
    print 'done: time taken for the correction =  ',time.clock()-start_cic

    #compute delta(k)^2, delete delta(k) and delta(k)*
    print 'computing delta(k)^2'
    delta_k_conj=np.conj(delta_k)
    delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj

    #compute the P(k)=<delta_k^2>
    if do_k_mu:
        Pk,number_of_modes=modes_k_mu(dims,delta_k2,bins_mu,axis)
    else:
        Pk,number_of_modes=modes(dims,delta_k2,axis)

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    #we should add 1 since we want to equal the number of intervals
    if do_k_mu:
        bins_k=np.arange(int(np.sqrt(3*int(0.5*(dims+1))**2))+1+1)
        bins_mu=np.linspace(-1.0,1.0,Pk.shape[1]+1)
        
        k=2.0*np.pi*bins_k/BoxSize; mu=bins_mu
        #k=2.0*np.pi*(0.5*(bins_k[:-1]+bins_k[1:]))/BoxSize
        #mu=0.5*(bins_mu[:-1]+bins_mu[1:])

    else:
        bins_perp=np.arange(int(np.sqrt(2*(dims/2+1)**2))+1+1)
        bins_par=np.arange(dims/2+1+1)

        k_perp=2.0*np.pi*(0.5*(bins_perp[:-1]+bins_perp[1:]))/BoxSize
        k_par=2.0*np.pi*(0.5*(bins_par[:-1]+bins_par[1:]))/BoxSize

    #Note that for the bins in (|k|,mu) we return the intervals, not the 
    #mean of the interval
    if do_k_mu:
        Pk=np.array([k,mu,Pk,number_of_modes])
    else:
        Pk=np.array([k_perp,k_par,Pk])
    print 'time used to perform calculation=',time.clock()-start_time,' s'

    return Pk



###################################################################
#this function computes:
#1) The sum of |delta(k)^2| for each bin in (|k|,\mu)
#2) The number of modes on each bin (|k|,\mu)
#The function returns the power spectrum = sum_modes |delta(k)^2|/ #_modes
def modes_k_mu(dims,delta_k2,bins_mu,axis):

    if len(delta_k2)!=dims**3:
        print 'sizes are different!!';  sys.exit()

    bins_k=int(np.sqrt(3*int(0.5*(dims+1))**2))+1
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

###################################################################
#this function computes:
#1) The sum of |delta(k)^2| for each bin in (k_perp,k_par)
#2) The number of modes on each bin (k_perp,k_par)
#The function returns the power spectrum = sum_modes |delta(k)^2|/ #_modes
def modes(dims,delta_k2,axis):

    if len(delta_k2)!=dims**3:
        print 'sizes are different!!'; sys.exit()

    bins_perp=int(np.sqrt(2*(dims/2+1)**2))+1; bins_par=dims/2+1
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

           switch (axis){
           case 0:
               k_perp = (int)sqrt(j*j+k*k);
               k_par  = abs(i);
               break;
           case 1:
               k_perp = (int)sqrt(i*i+k*k);
               k_par  = abs(j);
               break;
           case 2:
               k_perp = (int)sqrt(i*i+j*j);
               k_par  = abs(k);
               break;
           default:
               printf("Error with the chosen axis!!\\n");
               break;
           }

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
                    'bins_perp','bins_par','axis'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return Pk,number_of_modes
##############################################################################

#This routine computes the cross-P(k) in (k_perp,k_par) or (|k|,\mu) bins
#delta1 ----------------> array containing the values of delta1(r)
#delta2 ----------------> array containing the values of delta2(r)
#dims -----------------> number of cell per dimension used to compute the P(k)
#BoxSize --------------> size of the simulation box
#axis -----------------> axis along which compute P(|k|,mu) or P(k_perp,k_par)
#bins_mu --------------> number of bins for mu in P(|k|,mu)
#do_k_mu --------------> if True computes the Power spectrum in (|k|,\mu) bins
#                        if False the bins are in (k_perp,k_par)
#aliasing_method1------> method used to compute the delta1(r): CIC, TSC, other
#aliasing_method2------> method used to compute the delta2(r): CIC, TSC, other
#This routine returns the intervals in |k| and \mu used 
#(not the mean of interval!!) and the value of P(k) on each interval
def cross_power_spectrum_2D(delta1,delta2,dims,BoxSize,axis,bins_mu,
                            do_k_mu=False,
                            aliasing_method1='CIC',aliasing_method2='CIC'):

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

    #correct modes amplitude to account for aliasing when computing delta1(r)
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #since we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    if aliasing_method1 in ['CIC','TSC']:
        if aliasing_method1=='CIC':
            [array,k]=CIC_correction(dims)
        else:
            [array,k]=TSC_correction(dims)
        delta1_k*=array; 
        if aliasing_method1==aliasing_method2:
            delta2_k*=array
        del array
    else:
        [array,k]=CIC_correction(dims); del array
        print 'aliasing correction not performed on modes1'

    if aliasing_method1 != aliasing_method2:
        if aliasing_method1 in ['CIC','TSC']:
            if aliasing_method2=='CIC':
                [array,k]=CIC_correction(dims)
            else:
                [array,k]=TSC_correction(dims)
            delta2_k*=array
        else:
            print 'aliasing correction not performed on modes2'
    print 'done: time taken for the correction =  ',time.clock()-start_cic

    #compute delta_12(k)^2, delete delta_1(k)* and delta_2(k)*
    print 'computing delta_12(k)^2'
    delta1_k=np.conj(delta1_k)
    delta12_k2=np.real(delta1_k*delta2_k); del delta1_k,delta2_k

    #compute the P(k)=<delta12_k^2>
    if do_k_mu:
        Pk,number_of_modes=modes_k_mu(dims,delta12_k2,bins_mu,axis)
    else:
        Pk,number_of_modes=modes(dims,delta12_k2,axis)

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    #we should add 1 since we want to equal the number of intervals
    if do_k_mu:
        bins_k=np.arange(int(np.sqrt(3*int(0.5*(dims+1))**2))+1+1)
        bins_mu=np.linspace(-1.0,1.0,Pk.shape[1]+1)
        
        k=2.0*np.pi*bins_k/BoxSize; mu=bins_mu
        #k=2.0*np.pi*(0.5*(bins_k[:-1]+bins_k[1:]))/BoxSize
        #mu=0.5*(bins_mu[:-1]+bins_mu[1:])

    else:
        bins_perp=np.arange(int(np.sqrt(2*(dims/2+1)**2))+1+1)
        bins_par=np.arange(dims/2+1+1)

        k_perp=2.0*np.pi*(0.5*(bins_perp[:-1]+bins_perp[1:]))/BoxSize
        k_par=2.0*np.pi*(0.5*(bins_par[:-1]+bins_par[1:]))/BoxSize

    #Note that for the bins in (|k|,mu) we return the intervals, not the 
    #mean of the interval
    if do_k_mu:
        Pk=np.array([k,mu,Pk,number_of_modes])
    else:
        Pk=np.array([k_perp,k_par,Pk])
    print 'time used to perform calculation=',time.clock()-start_time,' s'

    return Pk
##############################################################################

#This routine computes P_l(k) where l=0,2 or l=4 
#delta ----------------> array containing the values of delta(r)
#dims -----------------> number of cell per dimension used to compute the P(k)
#BoxSize --------------> size of the simulation box
#ell ------------------> multipole: l=0 (monopole); l=2 (quadrupole)l l=4 ...
#axis -----------------> axis along which compute P(|k|,mu) or P(k_perp,k_par)
#aliasing_method ------> method used to compute the deltas(r): CIC, TSC, other
def multipole(delta,dims,BoxSize,ell,axis,aliasing_method='CIC'):

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1
 
    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta=np.reshape(delta,(dims,dims,dims))
    print 'Computing the FFT of the field...'; start_fft=time.clock()
    #delta_k=np.fft.ifftn(delta)
    delta_k=scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    delta_k=np.ravel(delta_k)

    #correct modes amplitude to account for aliasing when computing delta(r)
    #at the same time computes the modulus of k at each mesh point
    print 'Applying the CIC correction to the modes...';start_cic=time.clock()
    #since we are using complex numbers: 1) compute the correction over a
    #np.ones(dims3) array  2) multiply the results
    if aliasing_method in ['CIC','TSC']:
        if aliasing_method=='CIC':
            [array,k]=CIC_correction(dims)
        else:
            [array,k]=TSC_correction(dims)
        delta_k*=array; del array
    else:
        [array,k]=CIC_correction(dims); del array
        print 'aliasing correction not performed'
    print 'done: time taken for the correction =  ',time.clock()-start_cic

    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute delta(k)^2, delete delta(k) and delta(k)*
    print 'computing delta(k)^2'
    delta_k_conj=np.conj(delta_k)
    delta_k2=np.real(delta_k*delta_k_conj); del delta_k,delta_k_conj

    #compute the P_l(k)=(2*l+1)<delta_k^2(k,mu)*L_l(mu)>
    Pk,number_of_modes=modes_multipole(dims,delta_k2,ell,axis)

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk=Pk*BoxSize**3 

    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    #we should add 1 since we want to equal the number of intervals
    bins_k=np.arange(int(np.sqrt(3*int(0.5*(dims+1))**2))+1+1)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #ignore the first bin
    k=k[1:]; Pk=Pk[1:]

    #keep only with modes below 1.1*k_Nyquist
    k_N=np.pi*dims/BoxSize; indexes=np.where(k<1.1*k_N)
    k=k[indexes]; Pk=Pk[indexes]; del indexes
    print 'time used to perform calculation=',time.clock()-start_time,' s'

    return [k,Pk]

###################################################################
#this function computes:
#1) The sum of |delta(k)^2*P_l(\mu)| for each bin in k
#2) The number of modes on each k-bin 
#The function returns the power spectrum = sum_modes |delta(k)^2*P_l(\mu)|/ #_modes
def modes_multipole(dims,delta_k2,ell,axis):

    if len(delta_k2)!=dims**3:
        print 'sizes are different!!'; sys.exit()

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
##############################################################################

#This routine computes the Eisenstein & Hu matter power spectrum, with no wiggles.
#It returns [k,Pk_EH]
#Omega_m -----------> value of Omega_matter, at z=0
#Omega_b -----------> value of Omega_baryon, at z=0
#h -----------------> value of the hubble constant, at z=0, in 100 km/s/(Mpc) units
#ns ----------------> value of the spectral index
#sigma_8 -----------> value of sigma8 at z=0
def EH_Pk(Omega_m,Omega_b,h,ns,sigma8):

    #define the k-binning
    k = np.logspace(-3,3,10000)

    ommh2 = Omega_m*h**2;  ombh2 = Omega_b*h**2
    
    theta = 2.728 / 2.7
    s = 44.5*np.log(9.83/ommh2)/np.sqrt(1.0+10.0*np.exp(0.75*np.log(ombh2)))*h
    a = 1.0-0.328*np.log(431.0*ommh2)*ombh2/ommh2 +\
        0.380*np.log(22.3*ommh2)*(ombh2/ommh2)*(ombh2/ommh2)
    gamma = a+(1.0-a)/(1.0+np.exp(4*np.log(0.43*k*s)))
    gamma *= (Omega_m * h)
    q = k*theta*theta/gamma
    L0 = np.log(2.0*np.exp(1.0)+1.8*q)
    C0 = 14.2+731./(1.0+62.5*q)
    tmp = L0/(L0+C0*q*q)

    Pk_EH = k**ns*tmp**2

    #Normalize the amplitude of the P(k) to have the correct sigma8 value
    Norm = sigma8/MFL.sigma(k,Pk_EH,8.0)
    Pk_EH *= Norm**2

    print 'sigma8 =',MFL.sigma(k,Pk_EH,8.0)

    return [k,Pk_EH]







############################### EXAMPLE OF USAGE ###############################
if len(sys.argv)==2:
    if sys.argv[0]=='Power_spectrum_library.py' and sys.argv[1]=='compile':

        ################################################################
        ### power spectrum given delta (CIC and TSC) ###
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h

        np.random.seed(seed=1)
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta using CIC
        delta=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos,dims,BoxSize,delta) #compute densities
        delta=delta/(n*1.0/dims**3)-1.0

        [k,Pk]=power_spectrum_given_delta(delta,dims,BoxSize,
                                          aliasing_method='CIC')
        print Pk

        #compute delta using TSC
        delta=np.zeros(dims**3,dtype=np.float32) 
        CIC.TSC_serial(pos,dims,BoxSize,delta) #compute densities
        delta=delta/(n*1.0/dims**3)-1.0

        [k,Pk]=power_spectrum_given_delta(delta,dims,BoxSize,
                                          aliasing_method='TSC')
        print Pk

        ################################################################
        ### cross power spectrum given delta (CIC and TSC) ###
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h

        np.random.seed(seed=1)
        pos1=(np.random.random((n,3))*BoxSize).astype(np.float32)
        pos2=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta1 using CIC
        delta1=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos1,dims,BoxSize,delta1) #compute densities
        delta1=delta1/(n*1.0/dims**3)-1.0

        #compute delta2 using CIC
        delta2=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos2,dims,BoxSize,delta2) #compute densities
        delta2=delta2/(n*1.0/dims**3)-1.0

        [k,Pk12,Pk1,Pk2]=cross_power_spectrum_given_delta(delta1,delta2,dims,
                         BoxSize,aliasing_method1='CIC',aliasing_method2='CIC')
        print k; print Pk12; print Pk1; print Pk2
        
        ################################################################
        ### power spectrum 2D ###    P(|k|,mu) 
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h
        axis=0

        np.random.seed(seed=1)
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta using CIC
        delta=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos,dims,BoxSize,delta) #compute densities
        delta=delta/(n*1.0/dims**3)-1.0

        [k,mu,Pk,number_of_modes]=power_spectrum_2D(delta,dims,BoxSize,axis,
                                                    bins_mu=10,do_k_mu=True,
                                                    aliasing_method='CIC')
            
        print k; print mu; print Pk

        ################################################################
        ### power spectrum 2D ###    P(k_perp,k_par) 
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h
        axis=0

        np.random.seed(seed=1)
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta using CIC
        delta=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos,dims,BoxSize,delta) #compute densities
        delta=delta/(n*1.0/dims**3)-1.0

        [k_perp,k_par,Pk]=power_spectrum_2D(delta,dims,BoxSize,axis,
                                            bins_mu=10,do_k_mu=False,
                                            aliasing_method='CIC')
            
        print k_perp; print k_par; print Pk

        ################################################################
        ### cross power spectrum 2D ###    P(|k|,mu) 
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h

        np.random.seed(seed=1)
        pos1=(np.random.random((n,3))*BoxSize).astype(np.float32)
        pos2=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta1 using CIC
        delta1=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos1,dims,BoxSize,delta1) #compute densities
        delta1=delta1/(n*1.0/dims**3)-1.0

        #compute delta2 using CIC
        delta2=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos2,dims,BoxSize,delta2) #compute densities
        delta2=delta2/(n*1.0/dims**3)-1.0

        [k,mu,Pk,number_of_modes]=cross_power_spectrum_2D(delta1,delta2,dims,
                                                  BoxSize,
                                                  axis,bins_mu=10,do_k_mu=True,
                                                  aliasing_method1='CIC',
                                                  aliasing_method2='CIC')
        print k; print mu; print Pk

        ################################################################
        ### cross power spectrum 2D ###    P(k_perp,k_par) 
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h

        np.random.seed(seed=1)
        pos1=(np.random.random((n,3))*BoxSize).astype(np.float32)
        pos2=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta1 using CIC
        delta1=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos1,dims,BoxSize,delta1) #compute densities
        delta1=delta1/(n*1.0/dims**3)-1.0

        #compute delta2 using CIC
        delta2=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos2,dims,BoxSize,delta2) #compute densities
        delta2=delta2/(n*1.0/dims**3)-1.0

        [k_perp,k_par,Pk]=cross_power_spectrum_2D(delta1,delta2,dims,BoxSize,
                                                  axis,bins_mu=10,do_k_mu=False,
                                                  aliasing_method1='CIC',
                                                  aliasing_method2='CIC')
        print k_perp; print k_par; print Pk

        ################################################################
        ### mutipole ### l=0, l=2, l=4
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h

        np.random.seed(seed=1)
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta using CIC
        delta=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos,dims,BoxSize,delta) #compute densities
        delta=delta/(n*1.0/dims**3)-1.0

        #monopole (l=0)
        ell=0
        [k,Pk]=multipole(delta,dims,BoxSize,ell,axis,aliasing_method='CIC')
        print Pk

        #quadrupole (l=2)
        ell=2
        [k,Pk]=multipole(delta,dims,BoxSize,ell,axis,aliasing_method='CIC')
        print Pk

        #hexadrupole (l=4)
        ell=4
        [k,Pk]=multipole(delta,dims,BoxSize,ell,axis,aliasing_method='CIC')
        print Pk

        ################################################################
        ### EH_Pk ### 
        Omega_m = 0.3175
        Omega_b = 0.0490
        h       = 0.6711
        sigma8  = 0.8338
        ns      = 0.9624

        [k,Pk] = EH_Pk(Omega_m,Omega_b,h,ns,sigma8)
        print k;  print Pk
        
        
