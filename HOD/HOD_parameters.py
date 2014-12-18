#LATEST MODIFICATION: 10/11/2013
#This code computes the Xi^2 for a set of different HOD parameters

#to generate always the same results for a particular value of M1 & alpha
#edit the HOD_library.py code and comment out the lines with the seeds

#the range over which M1 and alpha wants to be varied has to be specified 
#below: not in the INPUT

#Be careful with the IDs. In Gadget the IDs start from 1 whereas when we sort
#them the first one will be 0, for instance:
#import numpy as np
#a=np.array([1,2,8,5,4,9,6,3,7])
#b=a.argsort(axis=0)
#b
#array([0, 1, 7, 4, 3, 6, 8, 2, 5])
#i.e. b[1] will return 1, whereas it should be 0

from mpi4py import MPI
import numpy as np
import scipy.integrate as si
import snap_chooser as SC
import readsnap
import readsubf
import HOD_library as HOD
import correlation_function_library as CF
import sys
import os
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
    sa=sys.argv

    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=sa[3]

    mass_criteria=sa[4]; min_mass=float(sa[5]); max_mass=float(sa[6])

    fiducial_density=float(sa[7])
    M1_min=float(sa[8]); M1_max=float(sa[9]); M1_bins=int(sa[10]); 
    alpha_min=float(sa[11]); alpha_max=float(sa[12]); alpha_bins=int(sa[13])
    
    random_file=sa[14]

    BoxSize=float(sa[15])
    Rmin=float(sa[16]); Rmax=float(sa[17]); bins=int(sa[18])

    DD_name=sa[19];   RR_name=sa[20];   DR_name=sa[21]
    DD_action=sa[22]; RR_action=sa[23]; DR_action=sa[24]

    wp_file=sa[25]; results_file=sa[26]

else:
    #### SNAPSHOTS TO SELECT GALAXIES WITHIN CDM HALOS ####
    snapshot_fname='../../snapdir_003/snap_003'
    groups_fname='../../'
    groups_number=3

    #### HALO CATALOGUE PARAMETERS ####
    mass_criteria='m200' #'t200' 'm200' or 'c200'
    min_mass=3e10 #Msun/h
    max_mass=2e15 #Msun/h

    ### HOD PARAMETERS ###
    fiducial_density=0.00111 #mean number density for galaxies with Mr<-21
    #M1_min=6.0e13;     M1_max=1.0e14;   M1_bins=20
    #alpha_min=1.05;  alpha_max=1.60;  alpha_bins=20
    
    M1_min=6.9e+13; M1_max= 6.9e+13; M1_bins=100
    alpha_min=1.20;  alpha_max=1.20;  alpha_bins=100

    #### RANDOM CATALOG ####
    random_file='/home/villa/disksom2/Correlation_function/Random_catalogue/random_catalogue_4e5.dat'

    #### PARAMETERS ####
    BoxSize=500.0 #Mpc/h
    Rmin=0.1 #Mpc/h
    Rmax=75.0 #Mpc/h
    bins=60

    #### PARTIAL RESULTS NAMES ####
    DD_name='DD.dat' #name for the file containing DD results
    RR_name='../RR_0.1_75_60_4e5.dat' #name for the file containing RR results
    DR_name='DR.dat' #name for the file containing DR results

    #### ACTIONS ####
    DD_action='compute' #'compute' or 'read' (from DD_name file)
    RR_action='read' #'compute' or 'read' (from RR_name file)
    DR_action='compute' #'compute' or 'read' (from DR_name file)
    
    #### wp FILE ####
    wp_file='../w_p_21.dat'
    wp_covariance_file='../wp_covar_21.0.dat'

    #### OUTPUT ####
    results_file='borrar.dat'
######################################################

if myrank==0:

    #read positions and IDs of DM particles: sort the IDs array
    DM_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)
    DM_vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=-1)
    #IDs should go from 0 to N-1, instead from 1 to N
    DM_ids=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1
    if np.min(DM_ids)!=0 or np.max(DM_ids)!=(len(DM_pos)-1):
        print 'Error!!!!'
        print 'IDs should go from 0 to N-1'
    print len(DM_ids),np.min(DM_ids),np.max(DM_ids)
    sorted_ids=DM_ids.argsort(axis=0)
    del DM_ids
    #the particle whose ID is N is located in the position sorted_ids[N]
    #i.e. DM_ids[sorted_ids[N]]=N
    #the position of the particle whose ID is N would be:
    #DM_pos[sorted_ids[N]]

    #read the IDs of the particles belonging to the CDM halos
    #again the IDs should go from 0 to N-1
    halos_ID=readsubf.subf_ids(groups_fname,groups_number,0,0,
                               long_ids=True,read_all=True)
    IDs=halos_ID.SubIDs-1
    del halos_ID

    print 'subhalos IDs=',np.min(IDs),np.max(IDs)

    #read CDM halos information
    halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                   group_veldisp=True,masstab=True,
                                   long_ids=True,swap=False)
    if mass_criteria=='t200':
        halos_mass=halos.group_m_tophat200*1e10   #masses in Msun/h
        halos_radius=halos.group_r_tophat200      #radius in kpc/h
    elif mass_criteria=='m200':
        halos_mass=halos.group_m_mean200*1e10     #masses in Msun/h
        halos_radius=halos.group_r_mean200        #radius in kpc/h
    elif mass_criteria=='c200':    
        halos_mass=halos.group_m_crit200*1e10     #masses in Msun/h
        halos_radius=halos.group_r_crit200        #radius in kpc/h
    else:
        print 'bad mass_criteria'
        sys.exit()
    halos_pos=halos.group_pos
    halos_len=halos.group_len
    halos_offset=halos.group_offset
    halos_indexes=np.where((halos_mass>min_mass) & (halos_mass<max_mass))[0]
    del halos
    
    print ' '
    print 'total halos found=',len(halos_pos)
    print 'halos number density=',len(halos_pos)/BoxSize**3

    #keep only the halos in the given mass range 
    halo_mass=halos_mass[halos_indexes]
    halo_pos=halos_pos[halos_indexes]
    halo_radius=halos_radius[halos_indexes]
    halo_len=halos_len[halos_indexes]
    halo_offset=halos_offset[halos_indexes]
    del halos_indexes

    if np.any(halo_len==[]):
        print 'something bad'

    #read the random catalogue (new version)
    dt=np.dtype((np.float32,3))
    pos_r=np.fromfile(random_file,dtype=dt)*BoxSize #Mpc/h

    #read the wp file
    f=open(wp_file,'r');  wp=[]
    for line in f.readlines():
        a=line.split()
        wp.append([float(a[0]),float(a[1]),float(a[2])])
    f.close();  wp=np.array(wp)

    #read covariance matrix file
    f=open(wp_covariance_file,'r')
    Cov=[]
    for line in f.readlines():
        a=line.split()
        for value in a:
            Cov.append(float(value))
    f.close(); Cov=np.array(Cov)
    if len(Cov)!=len(wp)**2:
        print 'problem with point numbers in the covariance file'
        sys.exit()
    Cov=np.reshape(Cov,(len(wp),len(wp)))
    Cov=np.matrix(Cov)

for g in range(100):

    ##### MASTER #####
    if myrank==0:

        #set here the range of M1, alpha to vary
        #print 'M1=';      M1=float(raw_input())
        #print 'alpha=';   alpha=float(raw_input())
        
        #M1=1.0e14+0.4e14*np.random.random()
        #alpha=1.10+0.3*np.random.random()
        #seed=np.random.randint(0,3000,1)[0]

        M1=1.15e14
        alpha=1.27
        seed=955

        #create the galaxy catalogue through the HOD parameters
        pos_g=HOD.hod_fast(DM_pos,DM_vel,sorted_ids,IDs,halo_mass,halo_pos,
                           halo_vel,halo_radius,halo_len,halo_offset,BoxSize,
                           min_mass,max_mass,fiducial_density,M1,
                           alpha,seed,model='standard',verbose=True)/1e3

        #compute the 2pt correlation function
        r,xi_r,error_xi=CF.TPCF(pos_g,pos_r,BoxSize,DD_action,
                                RR_action,DR_action,DD_name,RR_name,
                                DR_name,bins,Rmin,Rmax)

        f=open('correlation_function.dat','w')
        for i in range(len(r)):
            f.write(str(r[i])+' '+str(xi_r[i])+' '+str(error_xi[i])+'\n')
        f.close()
                                    
        r_max=np.max(r)
        h=1e-13 #discontinuity at r=rp. We integrate from r=rp+h to r_max
        yinit=np.array([0.0])

        f=open('projected_correlation_function.dat','w')
        wp_HOD=[]
        for rp in wp[:,0]:
            x=np.array([rp+h,r_max])
            y=si.odeint(deriv,yinit,x,args=(r,xi_r,rp),mxstep=100000)
            wp_HOD.append(y[1][0])
            f.write(str(rp)+' '+str(y[1][0])+'\n')
        wp_HOD=np.array(wp_HOD)
        f.close()

        print 'M1=',M1
        print 'alpha=',alpha

        chi2_bins=(wp_HOD-wp[:,1])**2/wp[:,2]**2
            
        for min_bin in [2]:
            for max_bin in [12]:
                elements=np.arange(min_bin,max_bin)
                
                #X^2 without covariance matrix
                chi2_nocov=np.sum(chi2_bins[elements])

                #X^2 with covariance matrix 
                wp_aux=wp[elements,1]; wp_HOD_aux=wp_HOD[elements]
                Cov_aux=Cov[elements,:][:,elements]
                diff=np.matrix(wp_HOD_aux-wp_aux)
                chi2=diff*Cov_aux.I*diff.T

                print 'X2('+str(min_bin)+'-'+str(max_bin)+')=',chi2_nocov,chi2
                g=open(results_file,'a')
                g.write(str(M1)+ ' '+str(alpha)+' '+str(seed)+' '+str(chi2)+'\n')
                g.close()


    ##### SLAVES #####
    else:
        pos_g=None; pos_r=None
        CF.TPCF(pos_g,pos_r,BoxSize,DD_action,RR_action,DR_action,
                DD_name,RR_name,DR_name,bins,Rmin,Rmax)



