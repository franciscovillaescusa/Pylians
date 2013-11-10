#version 1.1 
#read random catalogue from a file
from mpi4py import MPI
import numpy as np
import readsnap
import readsubf
import correlation_function_library as CF
import snap_chooser as SC
import random
import sys

comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

########### INPUT #############
if len(sys.argv)>1:
    sa=sys.argv
    Mnu=float(sa[1]); z=float(sa[2]); Box=int(sa[3]); som=sa[4]

    BoxSize=float(sa[5]); points_r=int(sa[6])

    DD_action=sa[7]; DR_action=sa[8]; RR_action=sa[9]
    DD_name=sa[10]; DR_name=sa[11]; RR_name=sa[12]

    D1D2_action=sa[13]; D1R_action=sa[14]; D2R_action=sa[15]
    D1D2_name=sa[16]; D1R_name=sa[17]; D2R_name=sa[18]

    random_file=sa[19]

    bins=int(sa[20]); Rmin=float(sa[21]); Rmax=float(sa[22])
    Np=int(sa[23])

    f_CDM=sa[24]; f_NU=sa[25]; f_CDM_NU=sa[26]; f_DM=sa[27]

    Omega_CDM=float(sa[28]); Omega_NU=float(sa[29])

else:
    Mnu=0.3; z=0.5; Box=1000; som='som2'

    BoxSize=500.0 #Mpc/h
    points_r=10000000

    DD_action='compute'; DR_action='compute'; RR_action='read'
    DD_name='DD.dat'; DR_name='DR.dat'; RR_name='RR_0.1_50_1e7.dat'

    D1D2_action='compute'; D1R_action='compute'; D2R_action='compute'
    D1D2_name='D1D2.dat'; D1R_name='D1R.dat'; D2R_name='D2R.dat'
    
    random_file='/disk/disksom2/villa/Correlation_function/Random_catalogue/'
    random_file+='random_catalogue_1e7.dat'

    bins=20
    Rmin=0.1 #Mpc/h
    Rmax=50.0 #Mpc/h

    Np=3000000 #number of randomly picked particles from the simulation

    f_CDM='TPCF_CDM_0.3_z=0.5.dat'
    f_NU='TPCF_NU_0.3_z=0.5.dat'
    f_CDM_NU='TPCF_CDM-NU_0.3_z=0.5.dat'

    Omega_CDM=0.2642266
    Omega_NU=0.006573383
###############################
Omega_DM=Omega_CDM+Omega_NU


#### MASTER ####
if myrank==0:
    #obtain subfind group file name
    F=SC.snap_chooser(Mnu,z,Box,som)
    snapshot_fname=F.snap

    #create the random catalogue or read it: positions in Mpc/h
    dt=np.dtype((np.float32,3)); pos_r=np.fromfile(random_file,dtype=dt)*BoxSize
    #pos_r=np.random.random((points_r,3))*BoxSize  


    ########################## CDM ##########################
    PAR_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    IDs=np.arange(len(PAR_pos))
    IDs=random.sample(IDs,Np)
    pos_g1=PAR_pos[IDs]

    #compute the 2pt correlation function
    r,xi_CDM,error_xi=CF.TPCF(pos_g1,pos_r,BoxSize,DD_action,
                            RR_action,DR_action,DD_name,RR_name,
                            DR_name,bins,Rmin,Rmax)

    f=open(f_CDM,'w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_CDM[i])+' '+str(error_xi[i])+'\n')
    f.close()


    ########################## NU ##########################
    PAR_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h
    IDs=np.arange(len(PAR_pos))
    IDs=random.sample(IDs,Np)
    pos_g2=PAR_pos[IDs]

    #compute the 2pt correlation function
    r,xi_NU,error_xi=CF.TPCF(pos_g2,pos_r,BoxSize,DD_action,
                            RR_action,DR_action,DD_name,RR_name,
                            DR_name,bins,Rmin,Rmax)

    f=open(f_NU,'w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_NU[i])+' '+str(error_xi[i])+'\n')
    f.close()


    ######################### CDM-NU #########################
    r,xi_CDM_NU=CF.TPCCF(pos_g1,pos_g2,pos_r,BoxSize,
                 D1D2_action,D1R_action,D2R_action,RR_action,
                 D1D2_name,D1R_name,D2R_name,RR_name,
                 bins,Rmin,Rmax)

    f=open(f_CDM_NU,'w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_CDM_NU[i])+'\n')
    f.close()

    ######################## total DM ########################
    xi_DM=(Omega_CDM/Omega_DM)**2*xi_CDM+(Omega_NU/Omega_DM)**2*xi_NU+2.0*Omega_CDM*Omega_NU/Omega_DM**2*xi_CDM_NU

    f=open(f_DM,'w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_DM[i])+'\n')
    f.close()


#### SLAVES ####
else:
    pos_g=None; pos_r=None
    CF.TPCF(pos_g,pos_r,BoxSize,DD_action,RR_action,DR_action,
            DD_name,RR_name,DR_name,bins,Rmin,Rmax)

    pos_g=None; pos_r=None
    CF.TPCF(pos_g,pos_r,BoxSize,DD_action,RR_action,DR_action,
            DD_name,RR_name,DR_name,bins,Rmin,Rmax)

    pos_g1=None; pos_g2=None; pos_r=None
    CF.TPCCF(pos_g1,pos_g2,pos_r,BoxSize,D1D2_action,D1R_action,D2R_action,
          RR_action,D1D2_name,D1R_name,D2R_name,RR_name,bins,Rmin,Rmax)
