#version 2.1
#It will automatically create/read the random catalogue and compute/read the 
#RR pairs

#This code computes the 2pt autocorrelation function of any the following:
#1) FoF halos
#2) SO halos (from Subfind)
#3) Subhalos (from Subfind)
#4) CDM particles from an N-body simulation

from mpi4py import MPI
import numpy as np
import readsnap, readsubf, readfof
import correlation_function_library as CFL
import redshift_space_library as RSL
import random, sys,os

comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

###################################### INPUT ####################################
if len(sys.argv)>1:

    parameter_file = sys.argv[1]
    print '\nLoading the parameter file ',parameter_file
    if os.path.exists(parameter_file):
        parms = imp.load_source("name",parameter_file)
        globals().update(vars(parms))
    else:   print 'file does not exists!!!'

else:

    #### GALAXY CATALOGUE ####
    snapshot_fname = '../snapdir_008/snap_008'   #only for obj=DM
    groups_fname   = '../FoF_b=0.2/'             #only for FoF/subfind
    groups_number  = 4                           #only for FoF/subfind
    SO_halo_mass   = 'm200' #'t200','m200','c200' only for subfind
    min_mass       = 1e6                         #Msun/h
    max_mass       = 1e16                        #Msun/h
    N_par          = 300000 #number of particles for subsampling (only for DM/NU)
    data_file      = '../pinocchio.3.0000.example.catalog.out' #only for data

    #### OBJECT USED AS GALAXY ####
    obj = 'data'     #'FoF_halos','halos','subhalos', 'DM' or 'data'

    #### RANDOM CATALOG ####
    random_points = 500000  #number of points in the random catalogue

    #### PARTIAL RESULTS NAMES ####
    DD_name = 'DD.dat';   DR_name = 'DR.dat'

    #### CF PARAMETERS ####
    BoxSize  = 60.0  #Mpc/h
    bins     = 30    #number of bins in the CF
    Rmin     = 0.1   #Mpc/h
    Rmax     = 20.0  #Mpc/h

    #### REDSHIFT-SPACE DISTORTIONS ####
    do_RSD   = True
    axis     = 0
    redshift = 3.0    #only needed to correct velocities of FoF halos

    #### OUTPUT ####
    f_out = 'CF_Pinocchio_RS_X_z=3.dat2'
################################################################################

#obtain the positions of the random particles reading/creating a random catalogue
pos_r,RR_name = CFL.create_random_catalogue(random_points,Rmin,Rmax,bins,BoxSize)

#we set here the actions
DD_action = 'compute'
RR_action = 'read'      #if needed, the RR pairs are computed above
DR_action = 'compute'

#Only the master will read the positions of the galaxies
pos_g = None   

#### MASTER ####
if myrank==0:

    #read FoF-halos/subfind-halos/subhalos information    
    print '\nReading galaxy catalogue'
    if obj == 'FoF_halos':                     #read FoF file
        halos = readfof.FoF_catalog(groups_fname,groups_number,
                                    long_ids=False,swap=False)

        halos_pos = halos.GroupPos/1e3;           #Mpc/h
        halos_mass = halos.GroupMass*1e10         #Msun/h
        halos_vel = halos.GroupVel*(1.0+redshift) #km/s

        halos_indexes = np.where((halos_mass>min_mass) & \
                                 (halos_mass<max_mass))[0]
        pos_g = halos_pos[halos_indexes]
        if do_RSD:
            vel_g = halos_vel[halos_indexes]
            RSL.pos_redshift_space(pos_g,vel_g,BoxSize,Hubble,redshift,axis)

    elif obj == 'halos' or obj == 'subhalos':  #read subfind file
        halos = readsubf.subfind_catalog(groups_fname,groups_number,
                                         group_veldisp=True,masstab=True,
                                         long_ids=True,swap=False)
        #read SO halos positions/masses
        halos_pos = halos.group_pos/1e3                 #positions in Mpc/h
        if mass_criteria=='t200':
            halos_mass = halos.group_m_tophat200*1e10   #masses in Msun/h
            halos_radius = halos.group_r_tophat200/1e3  #radius in Mpc/h
        elif mass_criteria=='m200':
            halos_mass = halos.group_m_mean200*1e10     #masses in Msun/h
            halos_radius = halos.group_r_mean200/1e3    #radius in Mpc/h
        elif mass_criteria=='c200':    
            halos_mass = halos.group_m_crit200*1e10     #masses in Msun/h
            halos_radius = halos.group_r_crit200/1e3    #radius in Mpc/h
        else:
            print 'bad mass_criteria';  sys.exit()

        #read subhalos positions/masses
        subhalos_mass = halos.sub_mass*1e10             #masses in Msun/h
        subhalos_pos = halos.sub_pos/1e3                #positions in Mpc/h
        
        if obj == 'halos':
            halos_indexes = np.where((halos_mass>min_mass) & \
                                     (halos_mass<max_mass))[0]
            pos_g = halos_pos[halos_indexes]
        else:
            halos_indexes = np.where((subhalos_mass>min_mass) & \
                                     (subhalos_mass<max_mass))[0]
            pos_g = halos_pos[halos_indexes]
    
    elif obj == 'DM':                           #read snapshot file
        PAR_pos = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3#Mpc/h
        IDs = np.arange(len(PAR_pos));  IDs=random.sample(IDs,N_par)
        pos_g = PAR_pos[IDs];           del PAR_pos,IDs

    elif obj == 'data':
        #read Pinocchio halos information
        data = np.loadtxt(data_file,comments='#')
        Pin_pos = data[:,5:8];  Pin_mass = data[:,1];  Pin_vel = data[:,8:11]
        Pin_pos  = Pin_pos.astype(np.float32);  
        Pin_mass = Pin_mass.astype(np.float32)
        Pin_vel  = Pin_vel.astype(np.float32)
        halos_indexes = np.where((Pin_mass>min_mass) & \
                                 (Pin_mass<max_mass))[0]
        pos_g = Pin_pos[halos_indexes]
        if do_RSD:
            vel_g = Pin_vel[halos_indexes]
            RSL.pos_redshift_space(pos_g,vel_g,BoxSize,Hubble,redshift,axis)

    else:
        print 'bad object chosen';    sys.exit()


#compute the 2pt correlation function
if myrank==0:
    print '\nComputing the 2pt-correlation function...'

r,xi_r,error_xi = CFL.TPCF(pos_g,     pos_r,     BoxSize,
                           DD_action, RR_action, DR_action,
                           DD_name,   RR_name,   DR_name,
                           bins,      Rmin,      Rmax)
                            
#save results to file
if myrank==0:
    np.savetxt(f_out,np.transpose([r,xi_r,error_xi]))


