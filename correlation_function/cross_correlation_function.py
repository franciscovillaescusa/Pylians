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
Mnu=0.6
z=0.0
som='som2'

mass_criteria='t200'
min_mass=2e12 #Msun/h
max_mass=2e15 #Msun/h
mode='DM' #'halos','subhalos' or 'DM'

BoxSize=500.0 #Mpc/h
points_r=4000000

DD_action='compute'; RR_action='compute'; DR_action='compute'
DD_name='DD.dat'; RR_name='RR.dat'; DR_name='DR.dat'

D1D2_action='compute'; D1R_action='compute'; D2R_action='compute'
D1D2_name='D1D2.dat'; D1R_name='D1R.dat'; D2R_name='D2R.dat'

bins=20
Rmin=0.1
Rmax=50.0

Np=1000000 #number of randomly picked particles from the simulation

f_out='borrar.dat'
###############################


#### MASTER ####
if myrank==0:
    #obtain subfind group file name
    F=SC.snap_chooser(Mnu,z,som)
    snapshot_fname=F.snap


    ##### CDM #####
    PAR_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    IDs=np.arange(len(PAR_pos))
    IDs=random.sample(IDs,Np)
    pos_g1=PAR_pos[IDs]

    pos_r=np.random.random((points_r,3))*BoxSize

    #compute the 2pt correlation function
    r,xi_r,error_xi=CF.TPCF(pos_g1,pos_r,BoxSize,DD_action,
                            RR_action,DR_action,DD_name,RR_name,
                            DR_name,bins,Rmin,Rmax)

    f=open('borrar_CDM.dat','w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_r[i])+' '+str(error_xi[i])+'\n')
    f.close()


    ##### NU #####
    PAR_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h
    IDs=np.arange(len(PAR_pos))
    IDs=random.sample(IDs,Np)
    pos_g2=PAR_pos[IDs]

    #compute the 2pt correlation function
    r,xi_r,error_xi=CF.TPCF(pos_g2,pos_r,BoxSize,DD_action,
                            RR_action,DR_action,DD_name,RR_name,
                            DR_name,bins,Rmin,Rmax)

    f=open('borrar_NU.dat','w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_r[i])+' '+str(error_xi[i])+'\n')
    f.close()


    ##### CDM-NU #####
    r,xi_r=CF.TPCCF(pos_g1,pos_g2,pos_r,BoxSize,
                 D1D2_action,D1R_action,D2R_action,RR_action,
                 D1D2_name,D1R_name,D2R_name,RR_name,
                 bins,Rmin,Rmax)

    f=open('borrar_CDM-NU.dat','w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_r[i])+'\n')
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
