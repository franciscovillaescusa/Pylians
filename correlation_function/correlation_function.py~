#version 1.1 
#read random catalogue from a file

#This code computes the 2pt autocorrelation function of any the following:
#1) CDM halos
#2) CDM subhalos
#3) DM particles from an N-body simulation

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
    
    Mnu=float(sa[1]); z=float(sa[2]); Box=int(sa[3]); location=sa[4]

    mass_criteria=sa[5]; min_mass=float(sa[6]); max_mass=float(sa[7])
    mode=sa[8]

    N_par=int(sa[9]); random_points=int(sa[10])

    DD_action=sa[11]; RR_action=sa[12]; DR_action=sa[13]
    DD_name=sa[14];   RR_name=sa[15];   DR_name=sa[16]

    random_file=sa[17]

    BoxSize=float(sa[18])
    bins=int(sa[19]); Rmin=float(sa[20]); Rmax=float(sa[21])

    f_out=sa[22]
else:
    Mnu=0.0; z=0.0; Box=1000; location='cosmos'

    mass_criteria='m200' #'t200' 'm200' or 'c200'
    min_mass=2e12*8.0 #Msun/h
    max_mass=3e15 #Msun/h
    mode='halos' #'halos','subhalos' or 'DM'

    N_par=300000 #number of random particles from the simulation (only for DM/NU)
    random_points=10000000

    random_file='/home/cosmos/users/mv249/RUNSG2/Paco/Random_catalogue/'
    random_file+='random_catalogue_1e7.dat'

    DD_action='compute'; RR_action='read'; DR_action='compute'
    DD_name='DD.dat'; RR_name='RR_0.206_103_40_1e7.dat'; DR_name='DR.dat'

    BoxSize=1000.0 #Mpc/h
    bins=40
    Rmin=0.103*2.0
    Rmax=51.5*2.0

    f_out='borrar.dat'
#'TPCF_DM_0.0_z=2.dat'
###############################


#### MASTER ####
if myrank==0:
    #obtain snapshot/subfind-group file name
    F=SC.snap_chooser(Mnu,z,Box,location)
    snapshot_fname=F.snap
    groups_fname=F.group
    groups_number=F.group_number

    #read CDM halos/subhalos information
    halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                   group_veldisp=True,masstab=True,
                                   long_ids=True,swap=False)
    halos_pos=halos.group_pos/1e3                 #positions in Mpc/h
    if mass_criteria=='t200':
        halos_mass=halos.group_m_tophat200*1e10   #masses in Msun/h
        halos_radius=halos.group_r_tophat200/1e3  #radius in Mpc/h
    elif mass_criteria=='m200':
        halos_mass=halos.group_m_mean200*1e10     #masses in Msun/h
        halos_radius=halos.group_r_mean200/1e3    #radius in Mpc/h
    elif mass_criteria=='c200':    
        halos_mass=halos.group_m_crit200*1e10     #masses in Msun/h
        halos_radius=halos.group_r_crit200/1e3    #radius in Mpc/h
    else:
        print 'bad mass_criteria'
        sys.exit()
    halos_indexes=np.where((halos_mass>min_mass) & (halos_mass<max_mass))[0]

    subhalos_mass=halos.sub_mass*1e10             #masses in Msun/h
    subhalos_pos=halos.sub_pos/1e3                #positions in Mpc/h
    subhalos_indexes=np.where((subhalos_mass>min_mass) & (subhalos_mass<max_mass))[0]
    del halos

    #chose the object of which compute the correlation function
    if mode=='DM':
        PAR_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
        IDs=np.arange(len(PAR_pos))
        IDs=random.sample(IDs,N_par)
        pos_g=PAR_pos[IDs]; del PAR_pos,IDs
    elif mode=='subhalos':
        pos_g=subhalos_pos[subhalos_indexes]
    elif mode=='halos':
        pos_g=halos_pos[halos_indexes]
    else:
        print 'bad mode chosen'
        sys.exit()

    #create the random catalogue or read it
    dt=np.dtype((np.float32,3)); pos_r=np.fromfile(random_file,dtype=dt)*BoxSize
    #pos_r=np.random.random((random_points,3))*BoxSize  #positions in Mpc/h

    print 'This is done'

    #compute the 2pt correlation function
    r,xi_r,error_xi=CF.TPCF(pos_g,pos_r,BoxSize,DD_action,
                            RR_action,DR_action,DD_name,RR_name,
                            DR_name,bins,Rmin,Rmax)

    f=open(f_out,'w')
    for i in range(len(r)):
        f.write(str(r[i])+' '+str(xi_r[i])+' '+str(error_xi[i])+'\n')
    f.close()


#### SLAVES ####
else:
    pos_g=None; pos_r=None
    CF.TPCF(pos_g,pos_r,BoxSize,DD_action,RR_action,DR_action,
            DD_name,RR_name,DR_name,bins,Rmin,Rmax)
