#This script computes, for every single FoF halo in the simulation the HI and 
#stellar mass within it. The output file contains three columns:
#M_halo M_HI_halo M_stars_halo
import numpy as np
import HI_library as HIL
import readsnap
import readfof
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

    f_out=sa[13]

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

    f_out    = 'M_HI_halos_Rahmati_z=3.dat'
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
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort
del pos_unsort,ID_unsort

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

#compute the value of Omega_HI
print 'Omega_HI = %.4e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)



#we create the array Star_mass that only contain the masses of the stars
Star_mass = np.zeros(Ntotal,dtype=np.float32)
Star_IDs  = readsnap.read_block(snapshot_fname,"ID  ",parttype=4)-1 #normalized
Star_mass[Star_IDs]=\
    readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10 #Msun/h
del Star_IDs

# read FoF halos information
halos = readfof.FoF_catalog(groups_fname,groups_number,
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
No_gas_halos=0; f=open(f_out,'w')
for index in range(len(M_FoF)):

    indexes=ID_FoF[Offset[index]:Offset[index]+Len[index]]
    
    Num_gas=len(indexes)
    if Num_gas>0:
        HI_mass=np.sum(M_HI[indexes],dtype=np.float64)
        Stellar_mass=np.sum(Star_mass[indexes],dtype=np.float64)
        f.write(str(M_FoF[index])+' '+str(HI_mass)+' '+\
                    str(Stellar_mass)+'\n')
    else:
        No_gas_halos+=1
f.close(); print '\nNumber of halos with no gas particles=',No_gas_halos
   
