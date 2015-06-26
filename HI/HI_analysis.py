#This script is capable of computing different things regarding the HI:
# 1) The function M_HI
# 2) The HI CDDF
# 3) The HI/21cm power spectrum in real-space
# 4) The bias between HI and matter b_HI(k) = P_HI-m(k)/P_m(k)
# 5) The HI/21cm power spectrum in redshift-space (averaging over the 3 axes)
# 6) The 21cm quadrupole and hexadecapole

import numpy as np
import readsnap
import readfof
import CIC_library as CIC
import cosmology_library as CL
import scipy.weave as wv
import scipy.integrate as si
import Power_spectrum_library as PSL
import redshift_space_library as RSL
import HI_library as HIL
import sys


################################# UNITS #####################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

Mpc=3.0856e24 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams

pi=np.pi
#############################################################################

################################ INPUT ######################################
if len(sys.argv)>1:
    sa=sys.argv

    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=int(sa[3])
    method=sa[4]

    fac=float(sa[5]); HI_frac=float(sa[6]); Omega_HI_ref=float(sa[7])
    method_Bagla=int(sa[8]); long_ids_flag=bool(int(sa[9]))
    SFR_flag=bool(int(sa[10])); f_MF=sa[11]; TREECOOL_file=sa[12]
    
    dims=int(sa[13])
    
    HI_in_halos=bool(int(sa[14]));  f_HI_in_halos=sa[15]
    HI_CDDF=bool(int(sa[16])); divisions=int(sa[17]); cells=int(sa[18])
    threads=int(sa[19]); f_HI_bins=int(sa[20]); f_HI_CDDF=sa[21]
    Pk_HI_real_space=bool(int(sa[22])); f_Pk_HI_real_space=sa[23]
    f_Pk_21cm_real_space=sa[24]; HI_matter_bias=bool(int(sa[25]))
    f_Pk_matter=sa[26]; f_Pk_cross=sa[27]; f_HI_bias=sa[28]
    Pk_HI_redshift_space=bool(int(sa[29])); f_Pk_HI_redshift_space=sa[30]
    f_Pk_21cm_redshift_space=sa[31]; do_quad=bool(int(sa[32]))
    do_hexa=bool(int(sa[33])); f_quad=sa[34]; f_hexa=sa[35]
    
    print '################# INFO ##############\n',sa

else:
    snapshot_fname = '../0.6eV/snapdir_007/snap_007'
    groups_fname   = '../0.6eV'
    groups_number  = 7

    #'Dave','method_1','Bagla','Barnes','Paco','Nagamine'
    method = 'Rahmati'

    fac     = 1.194564 #factor to obtain <F> = <F>_obs from the Lya 
    HI_frac = 0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref = 1e-3 #for method_1, Bagla and Paco and for computing x_HI
    method_Bagla = 3 #only for Bagla
    long_ids_flag = True;  SFR_flag = True #flags for reading the FoF file
    f_MF = '../0.6eV/Mass_function/Crocce_MF_z=3.dat' #mass function file
    TREECOOL_file = '../fiducial/TREECOOL_bestfit_g1.3'

    dims = 512

    ###### HI in halos #######
    HI_in_halos            = False
    f_HI_in_halos          = 'M_HI_halos_Rahmati_z=3.dat'
    ##########################

    ######## HI CDDF #########
    HI_CDDF                = False
    divisions              = 1 #divisions to the BoxSize to compute the N_HI
    cells                  = 1000 #LOS grid will have cells x cells elements 
    threads                = 32 #number of openmp threads
    f_HI_bins              = 300 #bins for computing f_HI
    f_HI_CDDF              = 'f_HI_Rahmati_z=3.dat'
    ##########################

    ###### HI/21cm P(k)#######
    Pk_HI_real_space       = False
    f_Pk_HI_real_space     = 'Pk_HI_Rahmati_real-space_z=3.dat'
    f_Pk_21cm_real_space   = 'Pk_21cm_Rahmati_real-space_z=3.dat'
    HI_matter_bias         = True
    f_Pk_matter            = 'Pk_matter_z=3.dat'
    f_Pk_cross             = 'Pk_HI-matter_Rahmati_z=3.dat'
    f_HI_bias              = 'HI-matter_bias_Rahmati_z=3.dat'

    Pk_HI_redshift_space     = False
    f_Pk_HI_redshift_space   = 'Pk_HI_Rahmati_z=3.dat'
    f_Pk_21cm_redshift_space = 'Pk_21cm_Rahmati_z=3.dat'
    do_quad                  = False   #whether compute 21cm P(k) quadrupole
    do_hexa                  = False   #whether compute 21cm P(k) hexadecapole
    f_quad                   = 'Pk_21cm_Rahmati_quadrupole_z=3.dat'
    f_hexa                   = 'Pk_21cm_Rahmati_hexadecapole_z=3.dat'
    ##########################

#############################################################################
bins_mu = 10

#for computing the HI bias we need the HI P(k) in real space
if HI_matter_bias:  Pk_HI_real_space=True

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head     = readsnap.snapshot_header(snapshot_fname)
BoxSize  = head.boxsize/1e3 #Mpc/h
Nall     = head.nall
Masses   = head.massarr*1e10 #Msun/h
Omega_m  = head.omega_m
Omega_l  = head.omega_l
redshift = head.redshift
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h        = head.hubble

#find the total number of particles in the simulation
Ntotal = np.sum(Nall,dtype=np.uint64)
print 'Total number of particles in the simulation =',Ntotal

#sort the pos and vel array
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 #normalized
print 'sorting the POS array...'
pos_unsort=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort; 
del pos_unsort
if Pk_HI_redshift_space:
    print 'sorting the VEL array...'
    vel_unsort=readsnap.read_block(snapshot_fname,"VEL ",parttype=-1) #km/s
    vel=np.empty((Ntotal,3),dtype=np.float32); vel[ID_unsort]=vel_unsort; 
    del vel_unsort
del ID_unsort

#find the IDs and HI masses of the particles to which HI has been assigned
if method == 'Dave':
    [IDs,M_HI]=HIL.Dave_HI_assignment(snapshot_fname,HI_frac,fac)
elif method=='method_1': 
    [IDs,M_HI]=HIL.method_1_HI_assignment(snapshot_fname,HI_frac,Omega_HI_ref)
elif method == 'Barnes':
    [IDs,M_HI]=HIL.Barnes_Haehnelt(snapshot_fname,groups_fname,
                                   groups_number,long_ids_flag,SFR_flag)
elif method == 'Paco':
    [IDs,M_HI]=HIL.Paco_HI_assignment(snapshot_fname,groups_fname,
                                      groups_number,long_ids_flag,SFR_flag)
elif method == 'Nagamine':
    [IDs,M_HI]=HIL.Nagamine_HI_assignment(snapshot_fname,
                                          correct_H2=False)
elif method == 'Bagla':
    [IDs,M_HI]=HIL.Bagla_HI_assignment(snapshot_fname,groups_fname,
                                       groups_number,Omega_HI_ref,method_Bagla,
                                       f_MF,long_ids_flag,SFR_flag)
elif method == 'Rahmati':
    [IDs,M_HI]=HIL.Rahmati_HI_assignment(snapshot_fname,fac,TREECOOL_file,
                                         Gamma_UVB=None,correct_H2=True,
                                         IDs=None)
else:
    print 'Incorrect method selected!!!'; sys.exit()


############################### HI in halos ################################
if HI_in_halos:
    # we create the array Star_mass that only contain the masses of the stars
    Star_mass = np.zeros(Ntotal,dtype=np.float32)
    Star_IDs  = readsnap.read_block(snapshot_fname,"ID  ",parttype=4)-1 #normalized
    Star_mass[Star_IDs]=\
        readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10 #Msun/h
    del Star_IDs

    # read FoF halos information
    halos   = readfof.FoF_catalog(groups_fname,groups_number,
                                  long_ids=long_ids_flag,swap=False,SFR=SFR_flag)
    pos_FoF = halos.GroupPos/1e3   #Mpc/h
    M_FoF   = halos.GroupMass*1e10 #Msun/h
    ID_FoF  = halos.GroupIDs-1     #normalize IDs
    Len     = halos.GroupLen       #number of particles in the halo
    Offset  = halos.GroupOffset    #offset of the halo in the ID array
    del halos

    # some verbose
    print 'Number of FoF halos:',len(pos_FoF),len(M_FoF)
    print '%f < X [Mpc/h] < %f'%(np.min(pos_FoF[:,0]),np.max(pos_FoF[:,0]))
    print '%f < Y [Mpc/h] < %f'%(np.min(pos_FoF[:,1]),np.max(pos_FoF[:,1]))
    print '%f < Z [Mpc/h] < %f'%(np.min(pos_FoF[:,2]),np.max(pos_FoF[:,2]))
    print '%e < M [Msun/h] < %e\n'%(np.min(M_FoF),np.max(M_FoF))

    # make a loop over the different FoF halos and populate with HI
    No_gas_halos=0; f=open(f_HI_in_halos,'w')
    for index in range(len(M_FoF)):

        indexes = ID_FoF[Offset[index]:Offset[index]+Len[index]]
    
        Num_gas = len(indexes)
        if Num_gas>0:
            HI_mass =      np.sum(M_HI[indexes],dtype=np.float64)
            Stellar_mass = np.sum(Star_mass[indexes],dtype=np.float64)
            f.write(str(M_FoF[index])+' '+str(HI_mass)+' '+\
                        str(Stellar_mass)+'\n')
        else:
            No_gas_halos+=1
    f.close(); print '\nNumber of halos with no gas particles=',No_gas_halos
    del Star_mass,pos_FoF,M_FoF,ID_FoF,Len,Offset,indexes,
    del HI_mass,Stellar_mass
###########################################################################


#keep only the particles having HI masses
M_HI = M_HI[IDs]; pos = pos[IDs] 
if Pk_HI_redshift_space:  vel = vel[IDs]

############################### HI CDDF ################################
# for computing the HI CDDF we need the SPH radii of the particles
if HI_CDDF:

    X=pos[:,0]; Y=pos[:,1]; Z=pos[:,2]

    # sort the R array (note that only gas particles have an associated R)
    ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1#normalized
    R_unsort=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3 #Mpc/h
    R = np.zeros(Ntotal,dtype=np.float32);  R[ID_unsort] = R_unsort
    del R_unsort,ID_unsort;  R = R[IDs];  del IDs

    # find the border size: to avoid problems with boundary conditions, we 
    # restrict our LOS region to X,Y=[Border_size,BoxSize-Border_size]
    Border_size = np.max(R) #Mpc/h
    print 'Border size = %.4f Mpc/h'%Border_size

    # compute the value of dX ----> dX/dz = H0*(1+z)^2/H(z)
    dX = CL.absorption_distance(Omega_m,Omega_l,redshift,BoxSize/divisions)
    print 'dX=%f\n'%dX

    # keep only with the LOS that are not affected by boundary conditions
    xy_min = int(Border_size*cells/BoxSize)
    xy_max = int((BoxSize-Border_size)*cells/BoxSize)+1; del Border_size
    indexes_los = np.empty((xy_max-xy_min+1)**2,dtype=np.int32)

    offset = 0;  length = xy_max-xy_min+1;  numbers = np.arange(xy_min,xy_max+1)
    for i in xrange(xy_min,xy_max+1):
        indexes_los[offset:offset+length] = (cells*i+numbers); offset+=length
    del numbers

    # compute f_HI = #_of_absorbers / dN_HI / dX
    log10_N_HI_min = 16.0;  log10_N_HI_max = 23.0
    bins_histo = np.logspace(log10_N_HI_min,log10_N_HI_max,f_HI_bins+1)
    delta_bins_histo = bins_histo[1:]-bins_histo[:-1] #dN_HI
    N_HI_histo = 10**(0.5*(np.log10(bins_histo[:-1])+np.log10(bins_histo[1:])))
    f_HI = np.zeros(f_HI_bins,dtype=np.float64)

    total_los = len(indexes_los)*divisions
    prefactor = (Msun/h)/mH/(Mpc/h/(1.0+redshift))**2
    for i in range(divisions):

        z_min = BoxSize*i*1.0/divisions; z_max = BoxSize*(i+1)*1.0/divisions
        indexes = np.where((Z>=z_min) & (Z<z_max))[0]

        # make a subplot
        #BoxSize = 1.0;  X_min = 13.5;  Y_min = 9.0 #everything in Mpc/h
        #indexes = np.where((X>X_min)  & (X<X_min+BoxSize) & \
        #                   (Y>Y_min)  & (Y<Y_min+BoxSize) & \
        #                   (Z>=z_min) & (Z<z_max))[0]
        #X -= X_min; Y -= Y_min
        
        N_HI = HIL.NHI_los_sph(R[indexes],X[indexes],Y[indexes],M_HI[indexes],
                               BoxSize,cells,threads)*prefactor

        #write column density file
        #f=open(f_cd,'w')
        #for l in xrange(len(N_HI)):
        #    #y=l/cells*BoxSize*1.0/cells; x=l%cells*BoxSize*1.0/cells
        #   y=Y_min+l/cells*BoxSize*1.0/cells; x=X_min+l%cells*BoxSize*1.0/cells
        #    f.write(str(x)+' '+str(y)+' '+str(N_HI[l])+'\n')
        #f.close(); sys.exit()

        # take only the LOS not affected by the boundary conditions
        N_HI = N_HI[indexes_los]

        N_DLAs = len(np.where(N_HI>10**(20.3))[0]) #number of DLAs found 
        # compute dN/dX = number of DLAs per absorption distance
        print '\ndN/dX = %f'%(N_DLAs*1.0/len(N_HI)/dX) 
        print '%e < N_HI < %e\n'%(np.min(N_HI),np.max(N_HI))
        f_HI+=np.histogram(N_HI,bins=bins_histo)[0]\
            /delta_bins_histo/(total_los*dX)
    del R,X,Y,Z,indexes,indexes_los,N_HI

    # sanity check
    print 'dN/dX = %f'%(HIL.incidence_rate(N_HI_histo,f_HI))

    # save results to file
    np.savetxt(f_HI_CDDF,np.transpose([N_HI_histo,f_HI]))
    del bins_histo,f_HI

else:
    del IDs

###########################################################################



#mean HI mass per grid point
mean_M_HI = np.sum(M_HI,dtype=np.float64)/dims**3
print '< M_HI > = %e'%(mean_M_HI)
print 'Omega_HI = %e'%(mean_M_HI*dims**3/BoxSize**3/rho_crit)

#compute \delta T_b(z)---> prefactor to compute \delta T_b(x)
#note that when computing M_H we have to use the total Omega_B, not only the
#Hydrogen from the gas particles. Notice that the brigthness temperature excess
#will be computed as: delta_Tb = <delta_Tb> * M_HI/<M_HI>
#Therefore, the value of <M_HI> used to compute X_HI has to be the same of this
#used when computing M_HI/<M_HI>. We just take the <M_HI> of the simulation
Omega_cdm = Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_nu  = Nall[2]*Masses[2]/BoxSize**3/rho_crit
Omega_b   = Omega_m-Omega_cdm-Omega_nu
X_HI = np.sum(M_HI,dtype=np.float64)/(0.76*Omega_b*rho_crit*BoxSize**3) #HI/H
mean_delta_Tb = 23.44*(Omega_b*h**2/0.02)\
    *np.sqrt(0.15*(1.0+redshift)/(10.0*Omega_m*h**2))*X_HI #mK
print '\nOmega_CDM =',Omega_cdm; print 'Omega_B   =',Omega_b; 
print 'Omega_nu  =',  Omega_nu;  print 'X_HI (simulation) =',X_HI; 
print 'mean_delta_Tb =',mean_delta_Tb,'mK'


###################### HI power spectrum real space ######################
if Pk_HI_real_space:
    delta_HI = np.zeros(dims**3,dtype=np.float32)
    delta_HI = CIC.CIC_serial(pos,dims,BoxSize,delta_HI,M_HI)
    print '%.4e should be equal to \n%.4e'%(np.sum(delta_HI,dtype=np.float64),
                                            np.sum(M_HI,dtype=np.float64))
    delta_HI = delta_HI/mean_M_HI - 1.0

################ HI-Matter bias ##################
if HI_matter_bias:
    # read particle positions in #Mpc/h
    pos_m = readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 

    print '%.3f < X [Mpc/h] < %.3f'%(np.min(pos_m[:,0]),np.max(pos_m[:,0]))
    print '%.3f < Y [Mpc/h] < %.3f'%(np.min(pos_m[:,1]),np.max(pos_m[:,1]))
    print '%.3f < Z [Mpc/h] < %.3f'%(np.min(pos_m[:,2]),np.max(pos_m[:,2]))
        
    # create an array with the masses of the particles
    mass = np.zeros(Ntotal,dtype=np.float32);  offset = 0

    # read masses of the gas particles
    mass_gas = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10
    mass[offset:offset+len(mass_gas)]=mass_gas; offset+=len(mass_gas)
    del mass_gas

    # use masses of the CDM particles
    mass_CDM = np.ones(Nall[1],dtype=np.float32)*Masses[1]
    mass[offset:offset+len(mass_CDM)]=mass_CDM; offset+=len(mass_CDM)
    del mass_CDM

    # use masses of the neutrino particles
    if Nall[2]!=0:
        mass_NU = np.ones(Nall[2],dtype=np.float32)*Masses[2]
        mass[offset:offset+len(mass_NU)]=mass_NU; offset+=len(mass_NU)
        del mass_NU

    # read masses of the star particles
    mass_star = readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10
    mass[offset:offset+len(mass_star)]=mass_star; offset+=len(mass_star)
    del mass_star

    if np.any(mass==0.0):
        print 'Something went wrong!!!'; sys.exit()

    # compute the value of Omega_m
    Omega_m   = np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit
    Omega_CDM = Nall[1]*Masses[1]/BoxSize**3/rho_crit
    Omega_NU  = Nall[2]*Masses[2]/BoxSize**3/rho_crit
    print 'Omega_m   = %.4f'%Omega_m
    print 'Omega_CDM = %.4f'%Omega_CDM
    print 'Omega_NU  = %.4f'%Omega_NU
    print 'Omega_b   = %.4f'%(Omega_m-Omega_CDM-Omega_NU)

    # compute the mean mass per cell
    mean_mass = np.sum(mass,dtype=np.float64)/dims**3

    # compute the mass within each cell
    delta_m = np.zeros(dims**3,dtype=np.float32)
    CIC.CIC_serial(pos_m,dims,BoxSize,delta_m,weights=mass)
    print '%.6e should be equal to \n%.6e\n'\
        %(np.sum(delta_m,dtype=np.float64),np.sum(mass,dtype=np.float64))
    delta_m = delta_m/mean_mass - 1.0;  del mass,pos_m
    print np.min(delta_m),'< delta_m <',np.max(delta_m)

    # compute P_HI(k), P_m(k), P_HI-m(k)
    [k,Pk_cross,Pk_HI,Pk_m] = \
        PSL.cross_power_spectrum_given_delta(delta_HI,delta_m,dims,BoxSize,
                                             aliasing_method1='CIC',
                                             aliasing_method2='CIC')
    del delta_HI,delta_m
                                                               
    # save results to file
    np.savetxt(f_Pk_HI_real_space,   np.transpose([k,Pk_HI]))
    np.savetxt(f_Pk_21cm_real_space, np.transpose([k,Pk_HI*mean_delta_Tb**2]))
    np.savetxt(f_Pk_matter,          np.transpose([k,Pk_m]))
    np.savetxt(f_Pk_cross,           np.transpose([k,Pk_cross]))
    np.savetxt(f_HI_bias,            np.transpose([k,Pk_cross/Pk_m]))

if Pk_HI_real_space and not(HI_matter_bias): 
    k,Pk = PSL.power_spectrum_given_delta(delta_HI,dims,BoxSize)
    np.savetxt(f_Pk_HI_real_space,   np.transpose([k,Pk]))
    np.savetxt(f_Pk_21cm_real_space, np.transpose([k,Pk*mean_delta_Tb**2]))
    del delta_HI
###########################################################################


#################### HI power spectrum redshift space #####################
if Pk_HI_redshift_space:

    # define arrays containing the quadrupole and hexadecople
    if do_quad:   Pk_quad=[]
    if do_hexa:   Pk_hexa=[]
    Pk_axis=[]

    for axis in range(0,3):

        print '\nComputing the 21 cm P(k) along axis:',axis

        # compute the value of M_HI in each grid point
        delta_HI = np.zeros(dims**3,dtype=np.float32)

        # create a copy of the pos array
        pos_RSD = np.copy(pos)

        # do RSD along the axis
        RSL.pos_redshift_space(pos_RSD,vel,BoxSize,Hubble,redshift,axis)

        CIC.CIC_serial(pos_RSD,dims,BoxSize,delta_HI,M_HI); del pos_RSD
        print '%.4e should be equal to\n%.4e'%\
            (np.sum(delta_HI,dtype=np.float64),np.sum(M_HI,dtype=np.float64))
        print 'Omega_HI = %.4e'\
            %(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

        delta_HI = delta_HI/mean_M_HI - 1.0

        # compute 21 cm P(k)
        print '\nComputing monopole'
        Pk = PSL.power_spectrum_given_delta(delta_HI,dims,BoxSize)
        Pk_axis.append(Pk[1])

        # write P(k) to output file
        # np.savetxt(f_out[axis],np.transpose([Pk[0],Pk[1]]))
    
        #compute P(k,\mu)
        #k2D,mu2D,Pk2D,modes2D=PSL.power_spectrum_2D(delta_Tb,dims,BoxSize,
        #                                            axis,bins_mu,do_k_mu=True,
        #                                            aliasing_method='CIC')
        #                                            
        #f=open('Pk_2D_axis='+str(axis)+'.dat','w')
        #for i in xrange(len(k2D)-1):
        #    f.write(str(k2D[i])+' '+str(k2D[i+1])+' ')
        #    for j in xrange(len(mu2D)-1):
        #        f.write(str(Pk2D[i,j])+' ')
        #    f.write('\n')
        #f.close()

        #compute the quadrupole
        if do_quad:
            print '\nComputing quadrupole'
            Pk_q = PSL.multipole(delta_HI,dims,BoxSize,2,axis,
                                 aliasing_method='CIC')
            Pk_quad.append(Pk_q[1])
        
        #compute the hexadecapole
        if do_hexa:
            print '\nComputing hexadecapole'
            Pk_h = PSL.multipole(delta_HI,dims,BoxSize,4,axis,
                                 aliasing_method='CIC')
            Pk_hexa.append(Pk_h[1])

    del delta_HI


    # save monopole results to file
    Pk_axis=np.array(Pk_axis);  k=Pk[0];  
    f=open(f_Pk_HI_redshift_space,'w')
    g=open(f_Pk_21cm_redshift_space,'w')
    for i in range(len(k)):
        f.write(str(k[i])+' '+str(np.mean(Pk_axis[:,i]))+'\n')
        g.write(str(k[i])+' '+str(np.mean(Pk_axis[:,i])*mean_delta_Tb**2)+'\n')
    f.close(); g.close()

    # save quadrupole results to file
    if do_quad: 
        Pk_quad = np.array(Pk_quad); k=Pk_q[0]; f=open(f_quad,'w')
        for i in range(len(k)):
            f.write(str(k[i])+' '+str(np.mean(Pk_quad[:,i])*mean_delta_Tb**2)+'\n')
        f.close()

    # save hexadecapole results to file
    if do_hexa:  
        Pk_hexa = np.array(Pk_hexa); k=Pk_h[0]; f=open(f_hexa,'w')
        for i in range(len(k)):
            f.write(str(k[i])+' '+str(np.mean(Pk_hexa[:,i])*mean_delta_Tb**2)+'\n')
        f.close()
###########################################################################


