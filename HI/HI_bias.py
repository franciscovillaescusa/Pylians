#This scripts compute the HI, matter and the HI-matter power spectra
import numpy as np
import readsnap
import CIC_library as CIC
import Power_spectrum_library as PSL
import HI_library as HIL
import sys,os


rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
################################ INPUT ######################################
if len(sys.argv)>1:
    sa=sys.argv
    
    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=int(sa[3])
    method=sa[4]

    fac=float(sa[5]); HI_frac=float(sa[6]); Omega_HI_ref=float(sa[7])
    method_Bagla=int(sa[8]); long_ids_flag=bool(int(sa[9]))
    SFR_flag=bool(int(sa[10])); f_MF=sa[11]; TREECOOL_file=sa[12]
    
    dims=int(sa[13]); 

    f_Pk_HI=sa[14];  f_Pk_m=sa[15];  f_Pk_cross=sa[16] 

    print '################# INFO ##############\n',sa

else:
    snapshot_fname = '../fiducial/snapdir_016/snap_016'
    groups_fname   = '../fiducial'
    groups_number  = 16

    #'Dave','method_1','Bagla','Barnes','Paco','Nagamine'
    method = 'Rahmati'

    fac     = 1.435028 #factor to obtain <F> = <F>_obs from the Lya
    HI_frac = 0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref = 1e-3 #for method_1, Bagla and Paco and for computing x_HI
    method_Bagla = 3 #only for Bagla
    long_ids_flag = True;   SFR_flag = True #flags for reading the FoF file
    f_MF = '../fiducial/Mass_function/Crocce_MF_z=3.dat' #mass function file
    TREECOOL_file = '../fiducial/TREECOOL_bestfit_g1.3'

    dims = 512

    f_Pk_HI    = 'Pk_HI_Rahmati_z=3.dat'
    f_Pk_m     = 'Pk_matter_z=3.dat'
    f_Pk_cross = 'Pk_HI-matter_Rahmati_z=3.dat'
#############################################################################



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

#find the total number of particles in the simulation
Ntotal=np.sum(Nall,dtype=np.uint64)
print 'Total number of particles in the simulation =',Ntotal



################################## HI ###########################################
#sort the pos and vel array
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 #normalized
print 'sorting the POS array...'
pos_unsort=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort; del pos_unsort
del ID_unsort

#find the IDs and HI masses of the particles to which HI has been assigned
if method=='Dave':
    [IDs,M_HI]=HIL.Dave_HI_assignment(snapshot_fname,HI_frac,fac)
elif method=='method_1': 
    [IDs,M_HI]=HIL.method_1_HI_assignment(snapshot_fname,HI_frac,Omega_HI_ref)
elif method=='Barnes':
    [IDs,M_HI]=HIL.Barnes_Haehnelt(snapshot_fname,groups_fname,
                                   groups_number,long_ids_flag,SFR_flag)
elif method=='Paco':
    [IDs,M_HI]=HIL.Paco_HI_assignment(snapshot_fname,groups_fname,
                                      groups_number,long_ids_flag,SFR_flag)
elif method=='Nagamine':
    [IDs,M_HI]=HIL.Nagamine_HI_assignment(snapshot_fname,
                                          correct_H2=False)
elif method=='Bagla':
    [IDs,M_HI]=HIL.Bagla_HI_assignment(snapshot_fname,groups_fname,
                                       groups_number,Omega_HI_ref,method_Bagla,
                                       f_MF,long_ids_flag,SFR_flag)
elif method=='Rahmati':
    [IDs,M_HI]=HIL.Rahmati_HI_assignment(snapshot_fname,fac,TREECOOL_file,
                                         Gamma_UVB=None,correct_H2=True,IDs=None)
else:
    print 'Incorrect method selected!!!'; sys.exit()

#keep only the particles having HI masses
M_HI=M_HI[IDs]; pos=pos[IDs]; del IDs

#mean HI mass per grid point
mean_M_HI=np.sum(M_HI,dtype=np.float64)/dims**3
print '< M_HI > = %e'%(mean_M_HI)
print 'Omega_HI = %e'%(mean_M_HI*dims**3/BoxSize**3/rho_crit)

#define the delta_HI array
delta_HI = np.zeros(dims**3,dtype=np.float32)

#compute the HI mass within each grid cell
CIC.CIC_serial(pos,dims,BoxSize,delta_HI,M_HI); del pos
print '%.6e should be equal to %.6e'%(np.sum(delta_HI,dtype=np.float64),
                                      np.sum(M_HI,dtype=np.float64)); del M_HI
print 'Omega_HI = %e'\
    %(np.sum(delta_HI,dtype=np.float64)/BoxSize**3/rho_crit)

delta_HI = delta_HI/mean_M_HI - 1.0
print '%.3f < delta_HI < %.3f'%(np.min(delta_HI),np.max(delta_HI))
#################################################################################


################################## MATTER #######################################
#read particle positions in #Mpc/h
pos = readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 

print '%.3f < X [Mpc/h] < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0]))
print '%.3f < Y [Mpc/h] < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1]))
print '%.3f < Z [Mpc/h] < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2]))
        
#create an array with the masses of the particles
mass = np.zeros(Ntotal,dtype=np.float32);  offset = 0

#read masses of the gas particles
mass_gas = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10  #Msun/h
mass[offset:offset+len(mass_gas)]=mass_gas; offset+=len(mass_gas)
del mass_gas

#use masses of the CDM particles
mass_CDM = np.ones(Nall[1],dtype=np.float32)*Masses[1]
mass[offset:offset+len(mass_CDM)]=mass_CDM; offset+=len(mass_CDM)
del mass_CDM

#use masses of the neutrino particles
if Nall[2]!=0:
    mass_NU = np.ones(Nall[2],dtype=np.float32)*Masses[2]
    mass[offset:offset+len(mass_NU)]=mass_NU; offset+=len(mass_NU)
    del mass_NU

#read masses of the star particles
mass_star = readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10  #Msun/h
mass[offset:offset+len(mass_star)]=mass_star; offset+=len(mass_star)
del mass_star

if np.any(mass==0.0):
    print 'Something went wrong!!!'; #sys.exit()

#compute the value of Omega_m
Omega_m   = np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit
Omega_CDM = Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_NU  = Nall[2]*Masses[2]/BoxSize**3/rho_crit
print 'Omega_m   = %.4f'%Omega_m
print 'Omega_CDM = %.4f'%Omega_CDM
print 'Omega_NU  = %.4f'%Omega_NU
print 'Omega_b   = %.4f'%(Omega_m-Omega_CDM-Omega_NU)

#compute the mean mass per cell
mean_mass = np.sum(mass,dtype=np.float64)/dims**3

#compute the mass within each cell
delta_m = np.zeros(dims**3,dtype=np.float32)
CIC.CIC_serial(pos,dims,BoxSize,delta_m,weights=mass)
print '%.6e should be equal to \n%.6e\n'\
    %(np.sum(delta_m,dtype=np.float64),np.sum(mass,dtype=np.float64))
delta_m = delta_m/mean_mass
print np.min(delta_m),'< delta_m <',np.max(delta_m)
#################################################################################



#compute P_HI(k), P_m(k), P_HI-m(k)
[k,Pk_cross,Pk_HI,Pk_m] = PSL.cross_power_spectrum_given_delta(delta_HI,delta_m,
                               dims,BoxSize,aliasing_method1='CIC',
                                            aliasing_method2='CIC')
                                                               
#save results to file
np.savetxt(f_Pk_HI,    np.transpose([k,Pk_HI]))
np.savetxt(f_Pk_m,     np.transpose([k,Pk_m]))
np.savetxt(f_Pk_cross, np.transpose([k,Pk_cross]))
