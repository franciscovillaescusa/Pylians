#This code computes different power spectrums for different objects:
#CDM --------------> cdm particles
#GAS --------------> gas particles
#STARS ------------> star particles
#HI_gas -----------> HI mass in gas particles
#HI_stars ---------> HI mass in star particles
#HI_baryons -------> HI mass in baryon (gas+stars) particles

#residing in different enviroments:
#ALL --------> all particles in the simulation
#HALOS ------> particles residing within halos in a given mass interval
#FILAMENTS --> particles residing in filaments (outside any dark matter halo)

#If the desired enviroment is halos, dark matter halos within a mass interval
#can be selected by setting mass_interval=True and setting the minimim/maximum
#masses. If all halos are wanted just set mass_interval=False

#Notice that the splitting among the different enviroments is performed by the
#routine particle_indexes in HI_library.py

import numpy as np
import HI_library as HIL
import readsnap
import Power_spectrum_library as PSL
import CIC_library as CIC
import sys

################################# UNITS #######################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

Mpc=3.0856e24 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams

pi=np.pi
###############################################################################

################################ INPUT ########################################
if len(sys.argv)>1:
    sa=sys.argv
    
    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=int(sa[3])
    method=sa[4]

    fac=float(sa[5]); HI_frac=float(sa[6]); Omega_HI_ref=float(sa[7])
    method_Bagla=int(sa[8]); long_ids_flag=bool(int(sa[9]))
    SFR_flag=bool(int(sa[10])); f_MF=sa[11]
    
    dims=int(sa[12]); divisions=int(sa[13]); threads=int(sa[14])
    obj=sa[15]; enviroment=sa[16]

    mass_interval=bool(int(sa[17]))
    min_mass=float(sa[18]); max_mass=float(sa[19])

    M_HI_stars=float(sa[20]); Pk_method=sa[21]; f_out=sa[22]

    print '################# INFO ##############\n',sa

else:
    #snapshot and halo catalogue
    snapshot_fname='../Efective_model_60Mpc_winds/snapdir_008/snap_008'
    groups_fname='../Efective_model_60Mpc_winds/FoF_0.2'
    groups_number=8

    #'Dave','method_1','Bagla','Barnes','Paco','Nagamine'
    method='Dave'

    #1.362889 (60 Mpc/h z=3) 1.436037 (30 Mpc/h z=3) 1.440990 (15 Mpc/h z=3)
    #1.369705 (60 Mpc/h z=3 winds)
    fac=1.369705 #factor to obtain <F> = <F>_obs from the Lya : only for Dave
    HI_frac=0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref=1e-3 #for method_1, Bagla and Paco
    method_Bagla=3 #only for Bagla
    long_ids_flag=False; SFR_flag=True #flags for reading the FoF file
    f_MF='../mass_function/ST_MF_z=3.dat' #file containing the mass function
    
    dims=1024
    divisions=7 #the SPH gas sphere will be divided into divisions^3 points
    threads=20 #number of openmp threads to use in the SPH_gas routine

    obj='HI_gas'  #'CDM':'GAS':'STARS':'HI_gas':'HI_stars':'HI_baryons' 
    enviroment='ALL'  #'ALL'  'HALOS'  'FILAMENTS' 

    #if enviroment='HALOS' select here the mass range
    mass_interval=False #if all halos are wanted set mass_interval=False
    min_mass=1e10 #Msun/h 
    max_mass=1e11 #Msun/h

    M_HI_stars=1.25e5 #Msun/h --> HI mass to assign to the star particles

    Pk_method='CIC' #choose between 'SPH' or 'CIC'

    f_out='Pk_HI_Dave_60Mpc_winds_z=3.dat'
###############################################################################
dims3=dims**3


#read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h=head.hubble

#split the particle IDs among CDM, gas and stars and also among enviroment
indexes=HIL.particle_indexes(snapshot_fname,groups_fname,groups_number,
                             long_ids_flag,SFR_flag,mass_interval,min_mass,
                             max_mass)

#find the total number of particles in the simulation
Ntotal=np.sum(Nall,dtype=np.uint64)
print 'Total number of particles in the simulation =',Ntotal

#sort the pos array
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 #normalized
pos_unsort=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort

#sort the SPH radii array (only required if SPH_gas used)
if Pk_method=='SPH':
    ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1#normalized
    radii_unsort=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3#Mpc/h
    radii=np.empty(Ntotal,dtype=np.float32); radii[ID_unsort]=radii_unsort
    del radii_unsort, ID_unsort

#select the IDs of the object wanted
if obj in ['CDM','GAS','STARS']:

    if obj=='CDM':
        if enviroment=='ALL':
            IDs=np.hstack([indexes.cdm_halos,indexes.cdm_halos2,
                           indexes.cdm_filaments])
        elif enviroment=='HALOS':
            IDs=indexes.cdm_halos
        elif enviroment=='FILAMENTS':
            IDs=indexes.cdm_filaments
        else:
            print 'error'; sys.exit()

    if obj=='GAS':
        if enviroment=='ALL':
            IDs=np.hstack([indexes.gas_halos,indexes.gas_halos2,
                           indexes.gas_filaments])
        elif enviroment=='HALOS':
            IDs=indexes.gas_halos
        elif enviroment=='FILAMENTS':
            IDs=indexes.gas_filaments
        else:
            print 'error'; sys.exit()

    if obj=='STARS':
        if enviroment=='ALL':
            IDs=np.hstack([indexes.star_halos,indexes.star_halos2,
                           indexes.star_filaments])
        elif enviroment=='HALOS':
            IDs=indexes.star_halos
        elif enviroment=='FILAMENTS':
            IDs=indexes.star_filaments
        else:
            print 'error'; sys.exit()
    del indexes

    #keep only with the positions of the particles of interest
    pos=pos[IDs]; del IDs

    #compute the mean number of particles per grid point
    mean_density=len(pos)*1.0/dims3
    print 'mean # of particles per grid point = %e'%mean_density

    #compute the value of delta = rho / <rho> - 1
    delta=np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos,dims,BoxSize,delta) #computes the density
    print '%e should be equal to %e' %(len(IDs),np.sum(delta,dtype=np.float64))
                                       
    delta=delta/mean_density-1.0  #computes delta
    print 'numbers should be equal:',np.sum(delta,dtype=np.float64),0.0
    print np.min(delta),'< delta <',np.max(delta)

    #compute the PS
    Pk=PSL.power_spectrum_given_delta(delta,dims,BoxSize)

    #write total HI P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

#if we are working with HI find the HI mass array
else:

    #find the IDs and HI masses of the particles to which HI has been assigned
    #note that these IDs_g correspond to all the particles to which HI has been
    #assigned
    if method=='Dave':
        [IDs_g,M_HI]=HIL.Dave_HI_assignment(snapshot_fname,HI_frac,fac)
    elif method=='method_1':
        [IDs_g,M_HI]=HIL.method_1_HI_assignment(snapshot_fname,HI_frac,
                                                Omega_HI_ref)
    elif method=='Barnes':
        [IDs_g,M_HI]=HIL.Barnes_Haehnelt(snapshot_fname,groups_fname,
                                         groups_number,long_ids_flag,SFR_flag)
    elif method=='Paco':
        [IDs_g,M_HI]=HIL.Paco_HI_assignment(snapshot_fname,groups_fname,
                                          groups_number,long_ids_flag,SFR_flag)
    elif method=='Nagamine':
        [IDs_g,M_HI]=HIL.Nagamine_HI_assignment(snapshot_fname,
                                                correct_H2=False)
    elif method=='Bagla':
        [IDs_g,M_HI]=HIL.Bagla_HI_assignment(snapshot_fname,groups_fname,
                                             groups_number,Omega_HI_ref,
                                             method_Bagla,f_MF,long_ids_flag,
                                             SFR_flag)
                                             

    #compute the value of Omega_HI
    Omega_HI=np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit
    print '\nOmega_HI (simulation) = %e'%Omega_HI

    #compute the total expected value of M_HI, not the one from the simulation
    #M_HI_Total=np.sum(M_HI,dtype=np.float64)
    M_HI_Total=Omega_HI_ref*BoxSize**3*rho_crit 


    if obj=='HI_gas':
        if enviroment=='ALL':
            IDs=np.hstack([indexes.gas_halos,indexes.gas_halos2,
                           indexes.gas_filaments])
        elif enviroment=='HALOS':
            IDs=indexes.gas_halos
        elif enviroment=='FILAMENTS':
            IDs=indexes.gas_filaments
        else:
            print 'error'; sys.exit()        
            
        #keep only with the particles of interest
        del IDs_g; M_HI=M_HI[IDs]; pos=pos[IDs]
        if Pk_method=='SPH':
            radii=radii[IDs]
            if np.any(radii==0.0): #check that the SPH radii are fine
                print 'found gas particles with R<0!!!'; sys.exit()
        del IDs

    elif obj in ['HI_stars','HI_baryons']:
        if enviroment=='ALL':
            IDs=np.hstack([indexes.star_halos,indexes.star_halos2,
                           indexes.star_filaments])
        elif enviroment=='HALOS':
            IDs=indexes.star_halos
        elif enviroment=='FILAMENTS':
            IDs=indexes.star_filaments
        else:
            print 'error'; sys.exit()   

        #keep only with the particles of interest
        if obj=='HI_stars':
            M_HI=np.ones(len(IDs),dtype=np.float32)*M_HI_stars
            pos=pos[IDs]; del IDs,IDs_g
        else:
            M_HI[IDs]=M_HI_stars; IDs=np.hstack([IDs_g,IDs])
            M_HI=M_HI[IDs]; pos=pos[IDs]; del IDs,IDs_g


    #only keep particles with M_HI>0
    indexes=np.where(M_HI>0.0)[0]; M_HI=M_HI[indexes]; pos=pos[indexes]
    if Pk_method=='SPH':
        radii=radii[indexes]
    del indexes

    #compute the value of Omega_HI
    print 'Omega_HI (enviroment) = %e\n'\
        %(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)
    
    #compute the mean neutral hydrogen mass per grid point from the expected HI
    mean_M_HI=M_HI_Total/dims3
    #mean_M_HI=np.sum(M_HI,dtype=np.float64)/dims3
    print 'mean HI mass per grid point=',mean_M_HI,'\n'

    #compute the value of delta_HI = rho_HI / <rho_HI> - 1
    delta_HI=np.zeros(dims3,dtype=np.float32)
    if Pk_method=='CIC': #computes the density
        print 'Computing the M_HI in the grid cells using CIC'
        CIC.CIC_serial(pos,dims,BoxSize,delta_HI,M_HI) 
    else:
        print 'Computing the M_HI in the grid cells using gas SPH radii'
        CIC.SPH_gas(pos,radii,divisions,dims,BoxSize,threads,delta_HI,M_HI) 
    print '%e should be equal to:\n%e' %(np.sum(M_HI,dtype=np.float64),
                                       np.sum(delta_HI,dtype=np.float64))
    delta_HI=delta_HI/mean_M_HI-1.0  #computes delta
    print 'numbers may be equal:',np.sum(delta_HI,dtype=np.float64),0.0
    print np.min(delta_HI),'< delta_HI <',np.max(delta_HI)

    #compute the HI PS
    if Pk_method=='CIC':
        Pk=PSL.power_spectrum_given_delta(delta_HI,dims,BoxSize,
                                          do_CIC_correction=True)
    else:
        Pk=PSL.power_spectrum_given_delta(delta_HI,dims,BoxSize,
                                          do_CIC_correction=False)

    #write total HI P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()


