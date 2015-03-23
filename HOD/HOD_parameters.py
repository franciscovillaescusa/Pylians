#This code computes the Xi^2 for a set of different HOD parameters

#Be careful with the IDs. In Gadget the IDs start from 1 whereas when we sort
#them the first one will be 0, for instance:
#import numpy as np
#a=np.array([1,2,8,5,4,9,6,3,7])
#b=a.argsort(axis=0)
#b
#array([0, 1, 7, 4, 3, 6, 8, 2, 5])
#i.e. b[1] will return 1, whereas it should be 0

#The code will create/read the random catalogue and also compute the number of 
#pairs in it if needed

#The code can also be used to find the minium chi2 in the results file.
#For this type:
#python HOD_parameters.py results_file.txt
#where results_file.txt is the fine containing the chi2 values, M1...etc

#IMPORTANT!!!! check that halo velocities are correct (not sqrt(a) factors...)

from mpi4py import MPI
import numpy as np
import scipy.integrate as si
import readsnap
import readsubf
import HOD_library as HOD
import correlation_function_library as CFL
import sys,os,imp
import random

#function used to compute wp(rp): d(wp) / dr = 2r*xi(r) / sqrt(r^2-rp^2)
def deriv(y,x,r,xi,rp):
    value=2.0*x*np.interp(x,r,xi)/np.sqrt(x**2-rp**2)
    return np.array([value])


###### MPI DEFINITIONS ######
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

########################### INPUT ###############################
if len(sys.argv)>1:

    #look for the minimum
    if len(sys.argv)==2:
        results_file=sys.argv[1]
        M1,alpha,Mmin,seed,chi2,gal_density=np.loadtxt(results_file,unpack=True)
        index = np.where(chi2==np.min(chi2))
        print 'M1           =',M1[index][0]
        print 'alpha        =',alpha[index][0]
        print 'Mmin         =',Mmin[index][0]
        print 'seed         =',seed[index][0]
        print 'gal. density =',gal_density[index][0]
        print 'chi2         =',chi2[index][0]
        sys.exit()

    parameter_file = sys.argv[1]
    print '\nLoading the parameter file ',parameter_file
    if os.path.exists(parameter_file):
        parms = imp.load_source("name",parameter_file)
        globals().update(vars(parms))
    else:   print 'file does not exists!!!'

else:
    #### SNAPSHOT TO SELECT GALAXIES WITHIN CDM HALOS ####
    snapshot_fname = '../snapdir_003/snap_003'
    groups_fname   = '../'
    groups_number  = 3

    #### HALO CATALOGUE PARAMETERS ####
    mass_criteria = 'm200' #'t200' 'm200' or 'c200'
    min_mass      = 3e10   #Msun/h
    max_mass      = 2e15   #Msun/h

    #### HOD PARAMETERS ####
    fiducial_density = 0.00111 #mean number density for galaxies with Mr<-21
    M1_min = 1.15e+14;   M1_max = 1.15e14   #M1 range
    alpha_min = 1.27;    alpha_max =1.27    #alpha range
    seed = 955    #if None random seeds will be selected
    iterations = 200   #number of random points in the M1-alpha plane

    #### RANDOM CATALOG ####
    random_points = 500000  #number of points in the random catalogue

    #### PARAMETERS ####
    Rmin = 0.1   #Mpc/h
    Rmax = 75.0  #Mpc/h
    bins = 60

    #### PARTIAL RESULTS NAMES ####
    DD_name = 'DD.dat';  DR_name = 'DR.dat' 
    
    #### wp FILE ####
    wp_file            = 'w_p_21.dat'
    wp_covariance_file = 'wp_covar_21.0.dat'

    #### OUTPUT ####
    results_file = 'xi_0.0eV_z=0.dat'
######################################################

#read snapshot head and obtain BoxSize, Omega_m and Omega_L                   
head     = readsnap.snapshot_header(snapshot_fname)
BoxSize  = head.boxsize/1e3  #Mpc/h                                               
Nall     = head.nall
Masses   = head.massarr*1e10 #Msun/h                                              
Omega_m  = head.omega_m
Omega_l  = head.omega_l
redshift = head.redshift
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc          
h        = head.hubble

#obtain the positions of the random particles reading/creating a random catalogue
pos_r,RR_name = CFL.create_random_catalogue(random_points,Rmin,Rmax,bins,BoxSize)

#we set here the actions                                                      
DD_action = 'compute'
RR_action = 'read'      #if needed, the RR pairs are computed above           
DR_action = 'compute'

#Only the master will read the positions of the galaxies                      
pos_g = None

if myrank==0:

    #read positions, velocities and IDs of DM particles: sort the IDs array
    DM_pos = readsnap.read_block(snapshot_fname,"POS ",parttype=-1)  #kpc/h
    DM_vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=-1)  #km/s
    #IDs should go from 0 to N-1, instead from 1 to N
    DM_ids = readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1
    if np.min(DM_ids)!=0 or np.max(DM_ids)!=(len(DM_pos)-1):
        print 'Error!!!!'; print 'IDs should go from 0 to N-1'
    print len(DM_ids),np.min(DM_ids),np.max(DM_ids)
    sorted_ids = DM_ids.argsort(axis=0); del DM_ids
    #the particle whose ID is N is located in the position sorted_ids[N]
    #i.e. DM_ids[sorted_ids[N]]=N
    #the position of the particle whose ID is N would be:
    #DM_pos[sorted_ids[N]]

    #read the IDs of the particles belonging to the CDM halos
    #again the IDs should go from 0 to N-1
    halos_ID = readsubf.subf_ids(groups_fname,groups_number,0,0,
                                 long_ids=True,read_all=True)
    IDs = halos_ID.SubIDs-1;  del halos_ID
    print 'subhalos IDs =[',np.min(IDs),'-',np.max(IDs),']'

    #read CDM halos information
    halos = readsubf.subfind_catalog(groups_fname,groups_number,
                                     group_veldisp=True,masstab=True,
                                     long_ids=True,swap=False)
    if mass_criteria=='t200':
        halos_mass   = halos.group_m_tophat200*1e10 #masses in Msun/h
        halos_radius = halos.group_r_tophat200      #radius in kpc/h
    elif mass_criteria=='m200':
        halos_mass   = halos.group_m_mean200*1e10   #masses in Msun/h
        halos_radius = halos.group_r_mean200        #radius in kpc/h
    elif mass_criteria=='c200':    
        halos_mass   = halos.group_m_crit200*1e10   #masses in Msun/h
        halos_radius = halos.group_r_crit200        #radius in kpc/h
    else:
        print 'bad mass_criteria';  sys.exit()
    halos_pos          = halos.group_pos        #kpc/h
    halos_len          = halos.group_len
    halos_offset       = halos.group_offset
    halos_main_subhalo = halos.group_firstsub
    halos_vel          = halos.sub_vel[halos_main_subhalo]/np.sqrt(1.0+redshift)
    del halos
    
    print '\ntotal halos found =',len(halos_pos)
    print 'halos number density =',len(halos_pos)/BoxSize**3

    #keep only the halos in the given mass range 
    halos_indexes = np.where((halos_mass>min_mass) & (halos_mass<max_mass))[0]
    halo_mass = halos_mass[halos_indexes];  halo_pos    = halos_pos[halos_indexes]
    halo_vel  = halos_vel[halos_indexes];   halo_radius = halos_radius[halos_indexes]
    halo_len  = halos_len[halos_indexes];   halo_offset = halos_offset[halos_indexes]
    del halos_indexes; 
    if np.any(halo_len==[]):  print 'something wrong!!!'
    
    #read the wp file
    wp = np.loadtxt(wp_file)

    #read covariance matrix file
    f=open(wp_covariance_file,'r');  Cov=[]
    for line in f.readlines():
        a=line.split()
        for value in a:
            Cov.append(float(value))
    f.close(); Cov=np.array(Cov)
    if len(Cov)!=len(wp)**2:
        print 'problem with point numbers in the covariance file'; sys.exit()
    Cov=np.reshape(Cov,(len(wp),len(wp)));  Cov=np.matrix(Cov)


min_chi2 = 1e5  #we will save galaxy catalogues whose chi2 is lower than min_chi2
for g in range(iterations):

    ##### MASTER #####
    if myrank==0:

        #set here the range of M1, alpha to vary
        #print 'M1=';      M1=float(raw_input())
        #print 'alpha=';   alpha=float(raw_input())
        
        M1    = M1_min    + (M1_max-M1_min)      *np.random.random()
        alpha = alpha_min + (alpha_max-alpha_min)*np.random.random()
        if seed == None:   seed = np.random.randint(0,3000,1)[0]
            
        #create the galaxy catalogue through the HOD parameters
        hod = HOD.hod_fast(DM_pos,DM_vel,sorted_ids,IDs,halo_mass,halo_pos,
                           halo_vel,halo_radius,halo_len,halo_offset,BoxSize,
                           min_mass,max_mass,fiducial_density,M1,
                           alpha,seed,model='standard',verbose=True)

        pos_g = hod.pos_galaxies/1e3 #Mpc/h
        vel_g = hod.vel_galaxies     #km/s

        #only keep the catalogues with the correct galaxy number density
        if abs((hod.galaxy_density-fiducial_density)/fiducial_density)<0.02:

            #compute the 2pt correlation function
            r,xi_r,error_xi = CFL.TPCF(pos_g,pos_r,BoxSize,
                                       DD_action,RR_action,DR_action,
                                       DD_name,RR_name,DR_name,
                                       bins,Rmin,Rmax)
                                       
            #compute the projected 2pt correlation function
            r_max = np.max(r);  yinit = np.array([0.0]);  wp_HOD = []
            h = 1e-13 #discontinuity at r=rp. We integrate from r=rp+h to r_max
            for rp in wp[:,0]:
                x = np.array([rp+h,r_max])
                y = si.odeint(deriv,yinit,x,args=(r,xi_r,rp),mxstep=100000)
                wp_HOD.append(y[1][0])
            wp_HOD=np.array(wp_HOD)

            #compute the value of the chi2
            chi2_bins = (wp_HOD-wp[:,1])**2/wp[:,2]**2
            for min_bin in [2]:
                for max_bin in [12]:
                    elements = np.arange(min_bin,max_bin)
                
                    #X^2 without covariance matrix
                    chi2_nocov = np.sum(chi2_bins[elements])

                    #X^2 with covariance matrix 
                    wp_aux = wp[elements,1]; wp_HOD_aux = wp_HOD[elements]
                    Cov_aux = Cov[elements,:][:,elements]
                    diff = np.matrix(wp_HOD_aux-wp_aux)
                    chi2 = diff*Cov_aux.I*diff.T;  chi2=chi2[0,0]

                    if chi2<min_chi2:
                        min_chi2 = chi2
                        #save the positions, CF and projected-CF of the mock catalogue
                        np.savetxt('galaxy_catalogue.dat',np.hstack([pos_g,vel_g]))
                        np.savetxt('correlation_function.dat',
                                   np.transpose([r,xi_r,error_xi]))
                        np.savetxt('projected_correlation_function.dat',
                                   np.transpose([wp[:,0],wp_HOD]))

                    print 'M1 =',M1;  print 'alpha =',alpha
                    print 'X2('+str(min_bin)+'-'+str(max_bin)+')=',chi2_nocov,chi2
                    g=open(results_file,'a')
                    g.write(str(M1)+ ' '+str(alpha)+' '+str(hod.Mmin)+\
                                ' '+str(seed)+' '+str(chi2)+' '+\
                                str(hod.galaxy_density)+'\n')
                    g.close()


    ##### SLAVES #####
    else:
        r,xi_r,error_xi = CFL.TPCF(pos_g,pos_r,BoxSize,
                                   DD_action,RR_action,DR_action,
                                   DD_name,RR_name,DR_name,
                                   bins,Rmin,Rmax)



