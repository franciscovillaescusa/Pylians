################################################################################
################################################################################

#This library contains the routines needed to compute power spectra.

############## AVAILABLE ROUTINES ##############
#power_spectrum_given_delta
      #NGP_correction
      #CIC_correction
      #TSC_correction
      #lin_histogram
#cross_power_spectrum_given_delta
#power_spectrum_snapshot
#power_spectrum_2D
      #modes_k_mu
      #modes
#cross_power_spectrum_2D
#multipole
#multipole_cross
      #modes_multipole
#angular_power_spectrum
      #Cl_modulus
#EH_Pk
#CAMB_Pk
#CF_Taruya
#Gaussian_smoothing
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
import readsnap
import numpy as np
import scipy.fftpack
import scipy.weave as wv
import mass_function_library as MFL
import redshift_space_library as RSL
import sys
import time


rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3
###############################################################################
#This routine computes the P(k) if the values of delta(r) are given.
#It is useful when the values of delta(r) have to be computed for a particular
#quantity such as delta_HI(r)=HI(r)/<HI>-1
#This routine by default does not perform any shot-noise correction!!!
#It also does not compute any error for the power spectrum
#delta ----------------> array containing the values of delta(r)
#dims -----------------> number of cell per dimension used to compute the P(k)
#BoxSize --------------> size of the simulation box
#aliasing_method ------> method used to compute the deltas(r): NGP, CIC, TSC, other
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
    if aliasing_method in ['NGP','CIC','TSC']:
        if   aliasing_method=='NGP':   [array,k]=NGP_correction(dims)
        elif aliasing_method=='CIC':   [array,k]=CIC_correction(dims)
        else:                          [array,k]=TSC_correction(dims)
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
#1) the NGP correction to the modes amplitude 
#2) the module of k for a given point in the fourier grid
def NGP_correction(dims):
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

           value_i = (i==0) ? 1.0 : pow((i*M_PI/dims)/sin(i*M_PI/dims),1);
           value_j = (j==0) ? 1.0 : pow((j*M_PI/dims)/sin(j*M_PI/dims),1);
           value_k = (k==0) ? 1.0 : pow((k*M_PI/dims)/sin(k*M_PI/dims),1);

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
# This routine computes the auto- and cross-power spectra of a Gadget snapshot
# in real or redshift-space. Can compute the total matter power spectrum or the
# auto- cross-power spectra of different particle types.
# If one only wants the total matter P(k), set particle_type=[-1]. If the P(k)
# of the different components is wanted set for instance particle_type=[0,1,4]
# snapshot_fname -----------> name of the Gadget snapshot
# dims ---------------------> Total number of cells is dims^3 to compute Pk
# particle_type ------------> compute Pk of those particles, e.g. [1,2]
# do_RSD -------------------> Pk in redshift-space (True) or real-space (False)
# axis ---------------------> axis along which move particles in redshift-space
# hydro --------------------> whether snapshot is hydro (True) or not (False)
def Power_spectrum_snapshot(snapshot_fname,dims,particle_type,do_RSD,axis,
                            hydro):

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    print '\nREADING SNAPSHOTS PROPERTIES'
    head     = readsnap.snapshot_header(snapshot_fname)
    BoxSize  = head.boxsize/1e3  #Mpc/h
    Nall     = head.nall
    Masses   = head.massarr*1e10 #Msun/h
    Omega_m  = head.omega_m
    Omega_l  = head.omega_l
    redshift = head.redshift
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #km/s/(Mpc/h)
    h        = head.hubble
    z        = '%.3f'%redshift
    dims3    = dims**3

    # compute the values of Omega_cdm, Omega_nu, Omega_gas and Omega_s
    Omega_c = Masses[1]*Nall[1]/BoxSize**3/rho_crit
    Omega_n = Masses[2]*Nall[2]/BoxSize**3/rho_crit
    Omega_g, Omega_s = 0.0, 0.0
    if Nall[0]>0:
        if Masses[0]>0:  
            Omega_g = Masses[0]*Nall[0]/BoxSize**3/rho_crit
            Omega_s = Masses[4]*Nall[4]/BoxSize**3/rho_crit
        else:    
            # mass in Msun/h
            mass = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 
            Omega_g = np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit
            mass = readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10
            Omega_s = np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit
            del mass

    # some verbose
    print 'Omega_gas    = ',Omega_g
    print 'Omega_cdm    = ',Omega_c
    print 'Omega_nu     = ',Omega_n
    print 'Omega_star   = ',Omega_s
    print 'Omega_m      = ',Omega_g + Omega_c + Omega_n + Omega_s
    print 'Omega_m snap = ',Omega_m

    ######################################################################
    # for total matter just use all particles
    if particle_type==[-1]:

        print 'Computing total matter power spectrum...'
        
        # read the positions of all the particles
        pos = readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
        print '%.3f < X [Mpc/h] < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0]))
        print '%.3f < Y [Mpc/h] < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1]))
        print '%.3f < Z [Mpc/h] < %.3f\n'%(np.min(pos[:,2]),np.max(pos[:,2]))

        if do_RSD:
            print 'moving particles to redshift-space'
            # read the velocities of all the particles
            vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=-1) #km/s
            RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis)
            del vel

        # read the masses of all the particles
        if not(hydro):
            Ntotal = np.sum(Nall,dtype=np.int64)#compute the number of particles
            M = np.zeros(Ntotal,dtype=np.float32) #define the mass array
            offset = 0
            for ptype in [0,1,2,3,4,5]:
                M[offset:offset+Nall[ptype]] = Masses[ptype]
                offset += Nall[ptype]
        else:
            M = readsnap.read_block(snapshot_fname,"MASS",parttype=-1)*1e10
            print '%.3e < M [Msun/h] < %.3e'%(np.min(M),np.max(M))
            print 'Omega_M = %.4f\n'\
                %(np.sum(M,dtype=np.float64)/rho_crit/BoxSize**3)

        # compute the mean mass per grid cell
        mean_M = np.sum(M,dtype=np.float64)/dims**3

        # compute the mass within each grid cell
        delta = np.zeros(dims**3,dtype=np.float32)
        CIC.CIC_serial(pos,dims,BoxSize,delta,M); del pos
        print '%.6e should be equal to \n%.6e\n'\
            %(np.sum(M,dtype=np.float64),np.sum(delta,dtype=np.float64)); del M

        # compute the density constrast within each grid cell
        delta /= mean_M;  delta-=1.0
        print '%.3e < delta < %.3e\n'%(np.min(delta),np.max(delta))

        # compute the P(k)
        Pk = power_spectrum_given_delta(delta,dims,BoxSize)

        # write P(k) to output file
        f_out = 'Pk_m_z='+z+'.dat'
        np.savetxt(f_out,np.transpose([Pk[0],Pk[1]]));  sys.exit()
    #####################################################################

    # set the label of the output files
    root_fout = {'0' :'GAS',  '01':'CDMG',  '02':'GNU',    '04':'Gstars',
                 '1' :'CDM',                '12':'CDMNU',  '14':'CDMStars',
                 '2' :'NU',                                '24':'NUStars',
                 '4' :'Stars'                                             }

    # define the arrays containing the positions and deltas and power spectra
    delta = [[],[],[],[]]   #array  containing the gas, CDM, NU and stars deltas
    Pk    = [[[],[],[],[]], #matrix containing the auto- and cross-power spectra
             [[],[],[],[]],
             [[],[],[],[]],
             [[],[],[],[]]]

    # dictionary among particle type and the index in the delta and Pk arrays
    # delta of stars (ptype=4) is delta[3] not delta[4]
    index_dict = {0:0, 1:1, 2:2, 4:3} 

    #####################################################################
    # do a loop over all particle types and compute the deltas
    for ptype in particle_type:
    
        # read particle positions in #Mpc/h
        pos = readsnap.read_block(snapshot_fname,"POS ",parttype=ptype)/1e3 

        # move particle positions to redshift-space
        if do_RSD:
            vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=ptype)#km/s
            RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis)
            del vel

        # find the index of the particle type in the delta array
        index = index_dict[ptype]

        # compute the deltas
        delta[index] = np.zeros(dims3,dtype=np.float32)
        CIC.CIC_serial(pos,dims,BoxSize,delta[index])
        print '%.6e should be equal to \n%.6e\n'\
            %(len(pos),np.sum(delta[index],dtype=np.float64))

        # compute the density constrast within each grid cell
        delta[index] = delta[index]*dims3*1.0/len(pos)-1.0;  del pos
        print '%.3e < delta < %.3e\n'\
            %(np.min(delta[index]),np.max(delta[index]))
    #####################################################################

    #####################################################################
    # compute the auto-power spectrum when there is only one component
    if len(particle_type) == 1:

        ptype = particle_type[0];  index = index_dict[ptype]
        fout = 'Pk_'+root_fout[str(ptype)]+'_z='+z+'.dat'
        print '\nComputing the power spectrum of the particle type: ',ptype
        data = power_spectrum_given_delta(delta[index],dims,BoxSize)
        k = data[0];  Pk[index][index] = data[1];  del data
        np.savetxt(fout,np.transpose([k,Pk[index][index]]));  return None
    #####################################################################

    #####################################################################
    # if there are two or more particles compute auto- and cross-power spectra
    for i,ptype1 in enumerate(particle_type):
        for ptype2 in particle_type[i+1:]:

            # find the indexes of the particle types
            index1 = index_dict[ptype1];  index2 = index_dict[ptype2]

            # choose the name of the output files
            if do_RSD:  root_fname = 'Pk_RS_'+str(axis)+'_'
            else:       root_fname = 'Pk_'
            suffix = '_z='+z+'.dat'
            fout1  = root_fname+root_fout[str(ptype1)]+suffix
            fout2  = root_fname+root_fout[str(ptype2)]+suffix
            fout12 = root_fname+root_fout[str(ptype1)+str(ptype2)]+suffix

            # some verbose
            print '\nComputing the auto- and cross-power spectra of types: '\
                ,ptype1,'-',ptype2
            print 'saving results in:';  print fout1,'\n',fout2,'\n',fout12

            # This routine computes the auto- and cross-power spectra
            data = cross_power_spectrum_given_delta(delta[index1],
                                                    delta[index2],dims,
                                                    BoxSize)
                                                        
            k                  = data[0]
            Pk[index1][index2] = data[1];   Pk[index2][index1] = data[1]; 
            Pk[index1][index1] = data[2]
            Pk[index2][index2] = data[3]

            # save power spectra results in the output files
            np.savetxt(fout1,  np.transpose([k,Pk[index1][index1]]))
            np.savetxt(fout2,  np.transpose([k,Pk[index2][index2]]))
            np.savetxt(fout12, np.transpose([k,Pk[index1][index2]]))
    #####################################################################

    #####################################################################
    # compute the power spectrum of the sum of all components
    print '\ncomputing P(k) of all components'

    # dictionary giving the value of Omega for each component
    Omega_dict = {0:Omega_g, 1:Omega_c, 2:Omega_n, 4:Omega_s}

    Pk_m = np.zeros(len(k),dtype=np.float64);  
    name = '_'+root_fout[str(particle_type[0])]
    Omega_tot = Omega_dict[particle_type[0]]
    for ptype in particle_type[1:]:  
        name += '+'+root_fout[str(ptype)]
        Omega_tot += Omega_dict[ptype]
    
    if do_RSD:  f_out_m = 'Pk'+name+'_RS_'+str(axis)+'_z='+z+'.dat'
    else:       f_out_m = 'Pk'+name+'_z='+z+'.dat'

    for ptype1 in particle_type:
        for ptype2 in particle_type:
        
            # find the indexes of the particle types
            index1 = index_dict[ptype1];  index2 = index_dict[ptype2]

            Pk_m += Omega_dict[ptype1]*Omega_dict[ptype2] * Pk[index1][index2]

    Pk_m /= Omega_tot**2
    np.savetxt(f_out_m,np.transpose([k,Pk_m])) #write results to output file
    #####################################################################


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
#This routine computes P_l(k) where l=0,2 or l=4 for a cross-power spectrum
#delta1 ----------------> array containing the values of delta1(r)
#delta2 ----------------> array containing the values of delta2(r)
#dims -----------------> number of cell per dimension used to compute the P(k)
#BoxSize --------------> size of the simulation box
#ell ------------------> multipole: l=0 (monopole); l=2 (quadrupole)l l=4 ...
#axis -----------------> axis along which compute P(|k|,mu) or P(k_perp,k_par)
#aliasing_method ------> method used to compute the deltas(r): CIC, TSC, other
def multipole_cross(delta1,delta2,dims,BoxSize,ell,axis,
                    aliasing_method1='CIC',aliasing_method2='CIC'):
                    

    dims3=dims**3; start_time=time.clock()
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1

    #FFT of the delta field (scipy.fftpack seems superior to numpy.fft)
    delta1 = np.reshape(delta1,(dims,dims,dims))
    delta2 = np.reshape(delta2,(dims,dims,dims))
    print 'Computing the FFT of the field1...'; start_fft=time.clock()
    delta1_k = scipy.fftpack.ifftn(delta1,overwrite_x=True); del delta1
    print 'done: time taken for computing the FFT=',time.clock()-start_fft
    print 'Computing the FFT of the field2...'; start_fft=time.clock()
    delta2_k = scipy.fftpack.ifftn(delta2,overwrite_x=True); del delta2
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

    #compute delta_12(k)^2
    print 'computing delta_12(k)^2'
    delta1_k_conj=np.conj(delta1_k);  del delta1_k
    delta_k2=np.real(delta1_k_conj*delta2_k); del delta1_k_conj,delta2_k
 
    #count modes
    count=lin_histogram(bins_r,0.0,bins_r*1.0,k)

    #compute the P_l(k)=(2*l+1)<delta_k^2(k,mu)*L_l(mu)>
    Pk,number_of_modes=modes_multipole(dims,delta_k2,ell,axis)

    #given the physical units to P(k) (Mpc/h)^3, (kpc/h)^3 ...
    Pk = Pk*BoxSize**3 

    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    #we should add 1 since we want to equal the number of intervals
    bins_k = np.arange(int(np.sqrt(3*int(0.5*(dims+1))**2))+1+1)
    k = k.astype(np.float64) #to avoid problems with np.histogram
    k = 2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #ignore the first bin
    k = k[1:]; Pk = Pk[1:]

    #keep only with modes below 1.1*k_Nyquist
    k_N = np.pi*dims/BoxSize; indexes = np.where(k<1.1*k_N)
    k = k[indexes]; Pk = Pk[indexes]; del indexes
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

# delta --------------> 2D field. Should be input as a 1D numpy array
# dims ---------------> The field contains dims**2 pixels
# BoxSize ------------> BoxSize of the field in degrees
def angular_power_spectrum(delta,dims,BoxSize):

    # maximum value of k probed
    middle = dims/2
    bins_r = int(np.sqrt(2*middle**2))+1  

    # reshape field
    delta = np.reshape(delta,(dims,dims))

    # compute delta^2(k) of the field
    delta_k = scipy.fftpack.ifftn(delta,overwrite_x=True);  del delta
    delta_k = np.ravel(delta_k);  delta2_k = np.absolute(delta_k)**2

    # for each cell compute value of |k|
    l_value  = Cl_modulus(dims)

    # define bins in multipoles
    bins_l = np.linspace(0.0,bins_r*1.0,bins_r+1)

    # compute power spectrum
    Pl    = np.histogram(l_value,bins=bins_l,weights=delta2_k)[0]
    modes = np.histogram(l_value,bins=bins_l)[0]
    Pl    = Pl*1.0/modes;  del delta2_k

    # compute average value of l in each bin
    bin_l = np.histogram(l_value,bins=bins_l,weights=l_value)[0]
    bin_l = bin_l*1.0/modes

    # given proper units to multipoles and angular power spectrum
    factor = 2.0*np.pi/(BoxSize*np.pi/180.0)
    bin_l  = bin_l*factor
    Pl     = Pl/factor**2

    # avoid fundamental frequency
    bin_l = bin_l[1:];  Pl = Pl[1:]

    return [bin_l,Pl]


# This function computes the value of |k| for each grid cell of a 2D grid
def Cl_modulus(dims):
    mod_k = np.empty(dims**2,dtype=np.float64)

    support = "#include <math.h>"
    code = """
       int middle=dims/2;
       int i,j;

       for (long l=0;l<dims*dims;l++){
           i=l/dims;
           j=l%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;

           mod_k(l)=sqrt(i*i+j*j);
       } 
    """
    wv.inline(code,['dims','mod_k'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])
    return mod_k
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
    k = np.logspace(-4,3,10000)

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
##############################################################################

#This routine reads the CAMB P(k) and transfer function files and returns 
#all the different power spectra
#f_Pk ----------------> file containing the CAMB matter power spectrum
#f_transfer ----------> file containing the CAMB transfer functions
#Omega_cdm -----------> Value of Omega_cdm (only if the CDM+B P(k) is wanted)
#Omega_b 00-----------> Value of Omega_n   (only if the CDM+B P(k) is wanted)
class CAMB_Pk:
    def __init__(self,f_Pk,f_transfer,Omega_cdm=None,Omega_b=None):
        
        # read CAMB matter power spectrum file
        k_m,Pk_m = np.loadtxt(f_Pk,unpack=True)

        # read CAMB transfer function file
        k,Tcdm,Tb,dumb,dumb,Tnu,Tm = np.loadtxt(f_transfer,unpack=True)
        self.k = k

        #Interpolate to find P(k)_matter in the same ks as the transfer functions
        Pk_m = 10**(np.interp(np.log10(k),np.log10(k_m),np.log10(Pk_m)))
        self.Pk_m = Pk_m

        #compute the different power spectra and save them
        self.Pk_c = Pk_m*(Tcdm/Tm)**2   
        self.Pk_b = Pk_m*(Tb/Tm)**2     
        self.Pk_n = Pk_m*(Tnu/Tm)**2    

        self.Pk_x_c_b = Pk_m*Tcdm*Tb/Tm**2   
        self.Pk_x_c_n = Pk_m*Tcdm*Tnu/Tm**2  
        self.Pk_x_b_n = Pk_m*Tb*Tnu/Tm**2    

        #compute the CDM+B transfer function
        if Omega_cdm!=None and Omega_b!=None:
            Tcdmb = (Omega_CDM*Tcdm+Omega_B*Tb)/(Omega_CDM+Omega_B)
        
            self.Pk_cb = Pk_m*(Tcdmb/Tm)**2  
            self.Pk_x_cb_n = Pk_m*Tcdmb*Tnu/Tm**2

################################################################################
#This routine computes the 2-point correlation function using the Taruya et al.
#estimator (Eq. 4.2 of 0906.0507)
#delta --------------> array containing the value of delta(r)
#dims ---------------> number of cells per dimension
#BoxSize ------------> Size of the simulation box
#bins_CF ------------> Number of bins between 0 and BoxSize/2 to compute CF
#MAS ----------------> Mass assignment scheme ('NGP','CIC','TSC' or 'None')
#If bins_CF='None' then bins_CF = dims/2+1
#The MAS is needed to correct the modes amplitudes
def CF_Taruya(delta,dims,BoxSize,bins_CF='None',MAS='CIC'):

    dims3 = dims**3;  start = time.clock()

    if bins_CF=='None':  bins_CF = dims/2+1

    #compute delta(k) by FFT delta(r)
    delta = np.reshape(delta,(dims,dims,dims))
    print '\nComputing the FFT of the field...'
    delta_k = scipy.fftpack.ifftn(delta,overwrite_x=True); del delta
    delta_k = np.ravel(delta_k)

    #correct modes amplitude to account for MAS
    print 'Correcting modes amplitude...'
    if MAS in ['NGP','CIC','TSC']:
        if   MAS=='NGP':   [array,k] = NGP_correction(dims)
        elif MAS=='CIC':   [array,k] = CIC_correction(dims)
        else:              [array,k] = TSC_correction(dims)
        delta_k *= array;  del array
    else:
        [array,k] = CIC_correction(dims);  del array
        print 'aliasing correction not performed'

    #compute the value of r in each point of the grid 
    d_grid  = k*BoxSize/dims; del k
    print np.min(d_grid),'< d <',np.max(d_grid)

    #define the array with the bins in r
    distances = np.linspace(0.0,BoxSize/2.0,bins_CF)

    #compute |delta(k)|^2
    print 'Computing |delta(k)|^2...'
    delta_k = (np.absolute(delta_k))**2

    #compute xi(r) by FFT |delta(k)|^2
    delta_k = np.reshape(delta_k,(dims,dims,dims))
    print 'Computing the IFFT of the field...' 
    xi_delta = scipy.fftpack.fftn(delta_k,overwrite_x=True); del delta_k
    xi_delta = np.real(np.ravel(xi_delta))

    xi    = np.histogram(d_grid,bins=distances,weights=xi_delta)[0]
    modes = np.histogram(d_grid,bins=distances)[0]; del d_grid
    xi    /= modes

    distances_bin = 0.5*(distances[:-1]+distances[1:])

    return distances_bin,xi


################################################################################
# This function computes the smoothing factor for a Gaussian kernel
# If we have a field F(x) then its Fourier-transform is F(k), and we smooth
# it by doing FFT^{-1}F(k)*exp(-k^2*R^2/2.0)
# This routine returns the array exp(-k^2*R^2/2.0)
# field -------------> 1D array with dims^3 elements with the value of the field
# dims --------------> value of dims
# R -----------------> The smoothing scale: sigma=R
# BoxSize -----------> The size of the simulation box
def Gaussian_smoothing(dims,R,BoxSize):

    smooth_array = np.empty(dims**3,dtype=np.float32)
    R = np.array([R]);  BoxSize = np.array([BoxSize])

    support = "#include <math.h>"
    code = """
       long dims2=dims*dims;
       long dims3=dims*dims*dims;
       int middle=dims/2;
       int i,j,k;
       float kR;

       for (long l=0;l<dims3;l++){
           i=l/dims2;
           j=(l%dims2)/dims;
           k=(l%dims2)%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;
           k = (k>middle) ? k-dims : k;

           kR=sqrt(i*i+j*j+k*k)*2.0*M_PI/BoxSize(0)*R(0);
           smooth_array(l)=exp(-kR*kR/2.0);
       } 
    """
    wv.inline(code,['dims','smooth_array','R','BoxSize'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return smooth_array
################################################################################

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

        
        ### multipole cross ### l=0, l=2, l=4
        n = 100**3;  dims = 128;  BoxSize = 500.0 #Mpc/h
        np.random.seed(seed=1)

        pos1=(np.random.random((n,3))*BoxSize).astype(np.float32)
        pos2=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta1 using CIC
        delta1 = np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos1,dims,BoxSize,delta) #compute densities
        delta1=delta1/(n*1.0/dims**3)-1.0

        #compute delta2 using CIC
        delta2 = np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos2,dims,BoxSize,delta) #compute densities
        delta2=delta2/(n*1.0/dims**3)-1.0

        #monopole (l=0)
        ell=0
        [k,Pk]=multipole_cross(delta1,delta2,dims,BoxSize,ell,axis,
                               aliasing_method1='CIC',aliasing_method2='CIC')
        print Pk

        #quadrupole (l=2)
        ell=2
        [k,Pk]=multipole_cross(delta1,delta2,dims,BoxSize,ell,axis,
                               aliasing_method1='CIC',aliasing_method2='CIC')
        print Pk

        #hexadecapole (l=4)
        ell=4
        [k,Pk]=multipole_cross(delta1,delta2,dims,BoxSize,ell,axis,
                               aliasing_method1='CIC',aliasing_method2='CIC')
        print Pk

        ################################################################
        ### Angular power spectrum ###
        dims  = 128;  BoxSize = 5.0 #degrees
        delta = np.random.random(dims**2)
        
        l,Cl = angular_power_spectrum(delta,dims,BoxSize)
        print l,Cl

        ################################################################
        ### EH_Pk ### 
        Omega_m = 0.3175
        Omega_b = 0.0490
        h       = 0.6711
        sigma8  = 0.8338
        ns      = 0.9624

        [k,Pk] = EH_Pk(Omega_m,Omega_b,h,ns,sigma8)
        print k;  print Pk
        
        
        ################################################################
        ### CF_Taruya ### 
        n=100**3; dims=128; BoxSize=500.0 #Mpc/h

        np.random.seed(seed=1)
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)

        #compute delta using CIC
        delta=np.zeros(dims**3,dtype=np.float32) 
        CIC.CIC_serial(pos,dims,BoxSize,delta) #compute densities
        delta=delta/(n*1.0/dims**3)-1.0

        #compute CF using Taruya et al estimator
        r,xi = CF_Taruya(delta,dims,BoxSize,MAS='CIC')
        print r,xi

        ################################################################
        ### Gaussian_smoothing ### 
        dims = 128;  BoxSize = 1000.0 #Mpc/h
        R = 20.0  #Mpc/h

        # create a density field
        field = np.random.random(dims**3)
        print 'field pre-smoothing =';  print field
        print '< field > =',np.mean(field,dtype=np.float64)

        # go to Fourier space
        field   = np.reshape(field,(dims,dims,dims))
        field_k = scipy.fftpack.ifftn(field,overwrite_x=True);  del field
        field_k = np.ravel(field_k)

        # compute the smoothing array
        smoothing_array = Gaussian_smoothing(dims,R,BoxSize)
        field_k *= smoothing_array
        field_k = np.reshape(field_k,(dims,dims,dims))
        
        # go back to real-space
        field = scipy.fftpack.fftn(field_k,overwrite_x=True);  del field_k
        field = np.real(np.ravel(field))
        print 'field post-smoothing =';  print field
        print '< field > =',np.mean(field,dtype=np.float64)
        
        
