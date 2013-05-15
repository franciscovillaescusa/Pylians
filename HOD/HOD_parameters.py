#LATEST MODIFICATION: 14/05/2013
#This code computes the Xi^2 for a set of different HOD parameters

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

    Mnu=float(sa[1]); z=float(sa[2]); som=sa[3] 

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
    Mnu=0.0
    z=0.0
    som='som2'

    #### HALO CATALOGUE PARAMETERS ####
    mass_criteria='m200' #'t200' 'm200' or 'c200'
    min_mass=2e12 #Msun/h
    max_mass=2e15 #Msun/h

    ### HOD PARAMETERS ###
    fiducial_density=0.00111 #mean number density for galaxies with Mr<-21
    M1_min=9e13;     M1_max=1.2e14;   M1_bins=15
    alpha_min=1.25;  alpha_max=1.60;  alpha_bins=15

    #### RANDOM CATALOG ####
    random_file='/disk/disksom2/villa/Correlation_function/Random_catalogue/random_catalogue_2e5.dat'

    #### PARAMETERS ####
    BoxSize=500.0 #Mpc/h
    Rmin=0.1 #Mpc/h
    Rmax=60.0 #Mpc/h
    bins=30

    #### PARTIAL RESULTS NAMES ####
    DD_name='DD.dat' #name for the file containing DD results
    RR_name='RR_0.1_60_30.dat' #name for the file containing RR results
    DR_name='DR.dat' #name for the file containing DR results

    #### ACTIONS ####
    DD_action='compute' #'compute' or 'read' (from DD_name file)
    RR_action='read' #'compute' or 'read' (from RR_name file)
    DR_action='compute' #'compute' or 'read' (from DR_name file)
    
    #### wp FILE ####
    wp_file='w_p_21.dat'

    #### OUTPUT ####
    results_file='results_0.6_mean200_new1.dat'
######################################################

M1_array=np.linspace(M1_min, M1_max, M1_bins)
alpha_array=np.linspace(alpha_min, alpha_max, alpha_bins)

if myrank==0:

    #obtain the names of snapshots and groups files
    F=SC.snap_chooser(Mnu,z,som)
    snapshot_fname=F.snap
    groups_fname=F.group
    groups_number=F.group_number
    del F

    #read positions and IDs of DM particles: sort the IDs array
    DM_pos=readsnap.read_block(snapshot_fname,"POS ",parttype=1)
    DM_ids=readsnap.read_block(snapshot_fname,"ID  ",parttype=1)
    sorted_ids=DM_ids.argsort(axis=0)
    del DM_ids
    #the particle whose ID is N is located in the position sorted_ids[N]
    #i.e. DM_ids[sorted_ids[N]]=N
    #the position of the particle whose ID is N would be:
    #DM_pos[sorted_ids[N]]

    #read the IDs of the particles belonging to the CDM halos
    halos_ID=readsubf.subf_ids(groups_fname,groups_number,0,0,
                               long_ids=True,read_all=True)
    IDs=halos_ID.SubIDs
    del halos_ID

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

    #read the random catalogue
    f=open(random_file,'r');  pos_r=[]
    for line in f.readlines():
        a=line.split()
        pos_r.append([float(a[0]),float(a[1]),float(a[2])])
    f.close()
    pos_r=np.array(pos_r)*BoxSize #rand catalog:dimensionless box size 1 

    #read the wp file
    f=open(wp_file,'r');  wp=[]
    for line in f.readlines():
        a=line.split()
        wp.append([float(a[0]),float(a[1]),float(a[2])])
    f.close();  wp=np.array(wp)

    


g=open(results_file,'a')
for M1 in M1_array:
    for alpha in alpha_array:

        ##### MASTER #####
        if myrank==0:
            #create the galaxy catalogue through the HOD parameters
            pos_g=HOD.hod_fast(DM_pos,sorted_ids,IDs,halo_mass,halo_pos,
                               halo_radius,halo_len,halo_offset,BoxSize,
                               min_mass,max_mass,fiducial_density,M1,
                               alpha,verbose=True)/1e3

            #compute the 2pt correlation function
            r,xi_r,error_xi=CF.TPCF(pos_g,pos_r,BoxSize,DD_action,
                                    RR_action,DR_action,DD_name,RR_name,
                                    DR_name,bins,Rmin,Rmax)
                                    
            r_max=np.max(r)
            h=1e-13 #discontinuity at r=rp. We integrate from r=rp+h to r_max
            yinit=np.array([0.0])

            wp_HOD=[]
            for rp in wp[:,0]:
                x=np.array([rp+h,r_max])
                y=si.odeint(deriv,yinit,x,args=(r,xi_r,rp),mxstep=1000)
                wp_HOD.append(y[1][0])
            wp_HOD=np.array(wp_HOD)

            print 'M1=',M1
            print 'alpha=',alpha

            chi2=np.sum((wp_HOD-wp[:,1])**2/wp[:,2]**2)
            print 'X2=',chi2
            g.write(str(M1)+ ' '+str(alpha)+' '+str(chi2)+'\n')

        ##### SLAVES #####
        else:
            pos_g=None; pos_r=None
            CF.TPCF(pos_g,pos_r,BoxSize,DD_action,RR_action,DR_action,
                    DD_name,RR_name,DR_name,bins,Rmin,Rmax)

if myrank==0:
    g.close()



