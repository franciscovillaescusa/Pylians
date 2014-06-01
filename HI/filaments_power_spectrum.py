#This code computes the power spectrum of the HI outside halos

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
    
    dims=int(sa[12])

    mass_interval=bool(int(sa[13]))
    min_mass=float(sa[14]); max_mass=float(sa[15])

    f_out=sa[16]

    print '################# INFO ##############\n',sa

else:
    #snapshot and halo catalogue
    snapshot_fname='../Efective_model_60Mpc/snapdir_006/snap_006'
    groups_fname='../Efective_model_60Mpc/FoF_0.2'
    groups_number=6

    #'Dave','method_1','Bagla','Barnes','Paco','Nagamine'
    method='Dave'

    #1.362889 (60 Mpc/h z=3) 1.436037 (30 Mpc/h z=3) 1.440990 (15 Mpc/h z=3)
    #1.369705 (60 Mpc/h z=3 winds)
    fac=0.429351 #factor to obtain <F> = <F>_obs from the Lya : only for Dave
    HI_frac=0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref=1e-3 #for method_1, Bagla and Paco
    method_Bagla=3 #only for Bagla
    long_ids_flag=False; SFR_flag=True #flags for reading the FoF file
    f_MF='../mass_function/ST_MF_z=3.dat' #file containing the mass function
    
    dims=1024

    #if enviroment='HALOS' select here the mass range
    mass_interval=False #if all halos are wanted set mass_interval=False
    min_mass=1e10 #Msun/h 
    max_mass=1e11 #Msun/h

    f_out='Pk_HI_Dave_filaments_60Mpc_z=6.dat'
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

#find the IDs of the gas particles residing in filaments
IDs=indexes.gas_filaments

#find the IDs and HI masses of the particles to which HI has been assigned
#note that these IDs_g correspond to all the particles to which HI has been
#assigned
if method=='Bagla' or method=='Paco':

    #sort the HI/H array: only use gas particles
    ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1#normalized
    nH0_unsort=readsnap.read_block(snapshot_fname,"NH  ",parttype=0)*fac #HI/H
    nH0=np.zeros(Ntotal,dtype=np.float32); nH0[ID_unsort]=nH0_unsort
    del nH0_unsort

    #sort the mass array: only use gas particles
    mass_unsort=readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Ms/h
    mass=np.zeros(Ntotal,dtype=np.float32); mass[ID_unsort]=mass_unsort
    del mass_unsort, ID_unsort

    #define the M_HI array
    M_HI=0.76*mass*nH0; del nH0,mass
    print 'Omega_HI (0.76*HI/H*m) = %e'\
        %(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)
    
elif method=='Dave':
    [IDs_g,M_HI]=HIL.Dave_HI_assignment(snapshot_fname,HI_frac,fac)

    #compute the value of Omega_HI
    Omega_HI=np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit
    print '\nOmega_HI (simulation) = %e'%Omega_HI; del IDs_g

elif method=='Nagamine':
    [IDs_g,M_HI]=HIL.Nagamine_HI_assignment(snapshot_fname,correct_H2=False)
    
    #compute the value of Omega_HI
    Omega_HI=np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit
    print '\nOmega_HI (simulation) = %e'%Omega_HI; del IDs_g


#keep only with the gas particles in filaments
pos=pos[IDs]; M_HI=M_HI[IDs]; del IDs
print 'Omega_HI_filaments    = %e'\
    %(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

#compute the total expected value of M_HI, not the one from the simulation
#M_HI_Total=np.sum(M_HI,dtype=np.float64)
M_HI_Total=Omega_HI_ref*BoxSize**3*rho_crit 

#compute the value of Omega_HI
print 'Omega_HI (enviroment) = %e\n'\
    %(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)
    
#compute the mean neutral hydrogen mass per grid point from the expected HI
mean_M_HI=M_HI_Total/dims3
#mean_M_HI=np.sum(M_HI,dtype=np.float64)/dims3
print 'mean HI mass per grid point=',mean_M_HI,'\n'

#compute the value of delta_HI = rho_HI / <rho_HI> - 1
delta_HI=np.zeros(dims3,dtype=np.float32)
CIC.CIC_serial(pos,dims,BoxSize,delta_HI,M_HI) 
print '%e should be equal to:\n%e' %(np.sum(M_HI,dtype=np.float64),
                                     np.sum(delta_HI,dtype=np.float64))
delta_HI=delta_HI/mean_M_HI-1.0  #computes delta
print 'numbers may be equal:',np.sum(delta_HI,dtype=np.float64),0.0
print np.min(delta_HI),'< delta_HI <',np.max(delta_HI)

#compute the HI PS
Pk=PSL.power_spectrum_given_delta(delta_HI,dims,BoxSize,do_CIC_correction=True)

#write total HI P(k) file
f=open(f_out,'w')
for i in range(len(Pk[0])):
    f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
f.close()


