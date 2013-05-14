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
Mnu=0.0
z=0.0
som='som1'

mass_criteria='t200'
min_mass=2e12 #Msun/h
max_mass=2e15 #Msun/h
subhalos=True #True or False

BoxSize=500.0 #Mpc/h
points_r=2000000

DD_action='compute'; RR_action='compute'; DR_action='compute'
DD_name='DD.dat'; RR_name='RR.dat'; DR_name='DR.dat'

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
    groups_fname=F.group
    groups_number=F.group_number

    PAR_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h

    #read CDM halos information
    halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                   group_veldisp=True,masstab=True,
                                   long_ids=True,swap=False)
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
    halos_pos=halos.group_pos/1e3 #positions in Mpc/h
    halos_indexes=np.where((halos_mass>min_mass) & (halos_mass<max_mass))[0]

    subhalos_mass=halos.sub_mass*1e10 #masses in Msun/h
    subhalos_pos=halos.sub_pos/1e3 #positions in Mpc/h
    subhalos_indexes=np.where((subhalos_mass>min_mass) & (subhalos_mass<max_mass))[0]
    del halos

    if subhalos:
        pos_g=subhalos_pos[subhalos_indexes]
    else:
        pos_g=halos_pos[halos_indexes]
    IDs=np.arange(len(PAR_pos))
    IDs=random.sample(IDs,Np)
    pos_g=PAR_pos[IDs]


    pos_r=np.random.random((points_r,3))*BoxSize

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
