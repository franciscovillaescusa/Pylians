#LATEST MODIFICATION: 03/05/2013
#This code computes the correlation function of CDM halos from N-body simulations
#It uses the fast routines DDR_histogram3
from mpi4py import MPI
import numpy as np
import library as lb
import readsnap
import readsubf
import HOD_library as HOD
import correlation_function_library as CF
import sys,os

###### MPI DEFINITIONS ######
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

##################### INPUT ##########################
#### SNAPSHOTS TO SELECT GALAXIES WITHIN CDM HALOS ####
#snapshot_fname='/data1/villa/b500p512nu0z99tree/snapdir_017/snap_017'
#groups_fname='/data1/villa/b500p512nu0z99tree'
#groups_number=17

snapshot_fname='/disk/disksom2/villa/b500p512nu0z99tree/snapdir_017/snap_017'
groups_fname='/disk/disksom2/villa/b500p512nu0z99tree'
groups_number=17

#### HALO CATALOGUE PARAMETERS ####
mass_criteria='m200' #'t200' 'm200' or 'c200'
min_mass=2e12 #Msun/h
max_mass=2e15 #Msun/h

### HOD PARAMETERS ###
fiducial_density=0.00111 #mean number density for galaxies with Mr<-21
M1=8e13
alpha=1.5

#### RANDOM CATALOG ####
random_file='/disk/disksom2/villa/Correlation_function/Random_catalogue/random_catalogue_2e5.dat'

#### PARAMETERS ####
Rmin=0.1       #Mpc/h
Rmax=60.0      #Mpc/h
BoxSize=500.0  #Mpc/h
bins=30 #number of bins in the correlation function
dims=10 #divisions of the box for the particle sorting

#### PARTIAL RESULTS NAMES ####
DD_name='DD.dat' #name for the file containing DD results
RR_name='RR_0.1_60_30.dat' #name for the file containing RR results
DR_name='DR.dat' #name for the file containing DR results

#### ACTIONS ####
DD_action='compute' #'compute' or 'read' (from DD_name file)
RR_action='read' #'compute' or 'read' (from RR_name file)
DR_action='compute' #'compute' or 'read' (from DR_name file)
    
#### OUTPUT ####
out_fname='borrar20.dat'
results_file='results_0.0_mean200.dat'

#measured correlation function from Zehavi et al 2011
wp=np.array([[586.2,19.5],[402.9,11.7],[258.7,6.7],[163.2,4.7],
             [105.5,3.0],[68.9,2.2],[50.2,2.1],[35.5,1.8],
             [24.5,1.6],[15.3,1.3],[8.54,0.94],[4.11,0.71],
             [2.73,0.54]])
######################################################

M1_array=np.linspace(1.21e14,1.3e14,9)
alpha_array=np.linspace(1.25,1.60,25)

#M1_array=[1.11e14]
#alpha_array=[1.45]




if myrank==0:

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
    halo_radius=halos_radius[halos_indexes]
    halo_pos=halos_pos[halos_indexes]
    halo_len=halos_len[halos_indexes]
    halo_offset=halos_offset[halos_indexes]
    del halos_indexes

    #read the random catalogue
    f=open(random_file,'r')
    pos_r=[]
    for line in f.readlines():
        a=line.split()
        pos_r.append([float(a[0]),float(a[1]),float(a[2])])
    f.close()
    pos_r=np.array(pos_r)*BoxSize #rand catalog:dimensionless box size 1 




g=open(results_file,'a')
for M1 in M1_array:
    for alpha in alpha_array:

        ##### MASTER #####
        if myrank==0:
            #create the galaxy catalogue through HOD parameters
            pos_g=HOD.hod_fast(DM_pos,sorted_ids,IDs,halo_mass,halo_pos,
                               halo_radius,halo_len,halo_offset,BoxSize,
                               min_mass,max_mass,fiducial_density,M1,
                               alpha,verbose=True)/1e3 #kpc/h --> Mpc/h

            #compute the 2pt correlation function
            r,xi_r,error_xi=CF.TPCF(pos_g,pos_r,BoxSize,dims,DD_action,RR_action,
                                    DR_action,DD_name,RR_name,DR_name,bins,
                                    Rmin,Rmax)

            #save the results to a file
            f=open(out_fname,'w')
            for i in range(len(r)):
                f.write(str(r[i])+' '+str(xi_r[i])+' '+str(error_xi[i])+'\n')
            f.close()

            os.system("./run_fortran0")

            print 'M1=',M1
            print 'alpha=',alpha

            #read the wp file and compute chi^2
            f=open('borrar30.dat','r')
            wp_HOD=[]
            for line in f.readlines():
                a=line.split()
                wp_HOD.append(float(a[1]))
            f.close()

            chi2=np.sum((wp_HOD-wp[:,0])**2/wp[:,1]**2)
            print 'X2=',chi2
            g.write(str(M1)+ ' '+str(alpha)+' '+str(chi2)+'\n')

        ##### SLAVES #####
        else:
            pos_g=None; pos_r=None
            CF.TPCF(pos_g,pos_r,BoxSize,dims,DD_action,RR_action,DR_action,
                    DD_name,RR_name,DR_name,bins,Rmin,Rmax)

if myrank==0:
    g.close()



