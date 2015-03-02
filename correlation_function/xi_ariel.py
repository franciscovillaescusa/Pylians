from mpi4py import MPI
import numpy as np
import CIC_library as CIC
import correlation_function_library as CFL
import redshift_space_library as RSL
import readsnap
import sys,os
import scipy.weave as wv
import time


###### MPI DEFINITIONS ######                                                 
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
################################### INPUT #####################################
if len(sys.argv)>1:
    sa=sys.argv

    snapshot_fname = sa[1];  bins = int(sa[2]);  
    Rmin = float(sa[3]);     Rmax = float(sa[4]);        bins_type = sa[5]
    dims = int(sa[6]);       do_RSD = bool(int(sa[7]));  axis = int(sa[8])
    f_out = sa[9];           particle_type = []
    
    for i in xrange(6):
        if int(sa[10+i])==1:  particle_type.append(i)
            
    print sa; print 'particle_type =',particle_type
    
else:
    snapshot_fname = '1/snapdir_003/snap_003'

    bins = 50
    Rmin = 10.0   #Mpc/h
    Rmax = 150.0  #Mpc/h
    bins_type = 'log'  #whether use 'linear' bins or 'log' bins in xi(r)

    dims   = 160
    do_RSD = False  #whether compute the CF in real or redshift space
    axis   = 0      #axis along which go to redshift-space (only if do_RSD=True)

    f_out = 'xi_matter_160_z=0.dat'

    particle_type = [1,2] #types of particle to use to compute the CF

do_serial   = True
omp_threads = 1
###############################################################################
dims2 = dims**2;  dims3 = dims**3
delta = None;  pos_grid = None;  BoxSize = None #to gather later from master

#dictionary with the names of the different particle types
pname = {0:'gas', 1:'CDM', 2:'NU', 4:'star'}


#only master read the positions and masses and computes delta
if myrank==0:

    #read snapshot header
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Nall=head.nall
    Masses=head.massarr*1e10 #Msun/h
    Omega_m=head.omega_m
    Omega_l=head.omega_l
    redshift=head.redshift
    Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #(km/s)/(Mpc/h) 
    
    #define the delta array and the mean_mass variable
    delta     = np.zeros(dims3,dtype=np.float32)
    mean_mass = 0.0   #Msun/h

    #make a loop over all particle types and sum their masses in the grid
    for ptype in particle_type:

        #read particle positions 
        pos  = readsnap.read_block(snapshot_fname,"POS ",
                                   parttype=ptype)/1e3  #Mpc/h

        #displace particle positions to redshift-space
        if do_RSD:
            vel  = readsnap.read_block(snapshot_fname,"VEL ",
                                       parttype=ptype)  #km/s
            RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis)
            del vel

        #read particle masses
        mass = readsnap.read_block(snapshot_fname,"MASS",
                                   parttype=ptype)*1e10 #Msun/h

        print 'Number of '+pname[ptype]+' particles =',len(pos)
        print '%.4f < X < %.4f'%(np.min(pos[:,0]), np.max(pos[:,0]))
        print '%.4f < Y < %.4f'%(np.min(pos[:,1]), np.max(pos[:,1]))
        print '%.4f < Z < %.4f'%(np.min(pos[:,2]), np.max(pos[:,2]))
        print '%.4e < Mass < %.4e'%(np.min(mass), np.max(mass))

        #compute the value of Omega
        print 'Omega_'+pname[ptype]+' = %.4f'\
            %(np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit)

        #sum all masses
        mean_mass += np.sum(mass,dtype=np.float64)

        CIC.NGP_serial(pos,dims,BoxSize,delta,weights=mass) #mass in each cell
        #CIC.CIC_serial(pos,dims,BoxSize,delta,weights=mass) #mass in each cell
        print '%d should be equal to\n%d'%(np.sum(delta,dtype=np.float64),
                                           mean_mass); del pos,mass; print ' '

    #compute the mean mass in each grid cell
    mean_mass = mean_mass*1.0/dims3

    #compute the value of delta from the total mass in each cell
    delta = delta/mean_mass - 1.0
    print '%f < [(n - <n>) / <n>] < %f'%(np.min(delta),np.max(delta))
    print '%f should be close to 0'%(np.sum(delta,dtype=np.float64))

    #find the coordinates of each cell in the grid
    pos_grid = np.empty((dims3,3),dtype=np.float32)
    pos_grid[:,0] = (np.arange(dims3)/dims2)
    pos_grid[:,1] = (np.arange(dims3)%dims2)/dims
    pos_grid[:,2] = (np.arange(dims3)%dims2)%dims
    pos_grid *= BoxSize/dims
    print 'Total number of pairs in the grid = %.3e'%(dims3*(dims3-1)/2)


#send arrays computed by master to slaves
pos_grid = comm.bcast(pos_grid,root=0)
delta    = comm.bcast(delta,root=0)
BoxSize  = comm.bcast(BoxSize,root=0) 
BoxSize  = float(BoxSize) #to avoid problems with scipy weave

#compute distances between the cell 0 and all others cells 
diff    = pos_grid-pos_grid[0]
indexes = np.where(diff>BoxSize/2.0);  diff[indexes]-=BoxSize; del indexes
indexes = np.where(diff<-BoxSize/2.0); diff[indexes]+=BoxSize; del indexes
distances_ref = np.sqrt(diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2); del diff

#keep only with the cells at distances between Rmin and Rmax
indexes       = np.where((distances_ref>Rmin) & (distances_ref<Rmax))[0]
distances_ref = distances_ref[indexes]

#define CF binning and find bin in which the distances to neighbor cell lie
if bins_type=='linear':
    #R_i = Rmin + (Rmax-Rmin)/bins*i   for i=0,1,2,...,bins
    r_bins = np.linspace(Rmin,Rmax,bins+1)
    indexes_distances = ((distances_ref-Rmin)*bins/(Rmax-Rmin)).astype(np.int32)
elif bins_type=='log':
    #R_i = 10**[np.log10(Rmin) + (np.log10(Rmax)-np.log10(Rmin))/bins*i]   
    #for i=0,1,2,...,bins
    r_bins = np.logspace(np.log10(Rmin),np.log10(Rmax),bins+1)
    indexes_distances = (np.log10(distances_ref/Rmin)*bins\
                             /np.log10(Rmax/Rmin)).astype(np.int32)
else:
    print 'Wrong bins_type value';  sys.exit()
r_xi = 0.5*(r_bins[:-1] + r_bins[1:])




if myrank==0:
    print '%.3f < distances grid [Mpc/h] < %.3f'\
        %(np.min(distances_ref),np.max(distances_ref))
    print np.min(indexes_distances),'< bin <',np.max(indexes_distances)

#compute the i,j and k indexes such as index = dims2*i + dims*j + k
indexes_coord = np.empty((len(indexes),3),dtype=np.int32)
indexes_coord[:,0] = np.mod(indexes/dims2,dims)
indexes_coord[:,1] = np.mod(indexes/dims,dims)
indexes_coord[:,2] = np.mod(indexes,dims); del indexes

#define the number of pairs and xi arrays
pairs = np.zeros(bins,dtype=np.int64)
xi    = np.zeros(bins,dtype=np.float64)

divisions = 1000  #each cpu will divide its part into divisions tasks
length = int(dims3/(divisions*nprocs))+1 #number of particles 

#compute the correlation function
if myrank==0:    print '\nComputing the correlation function'
start_time=time.clock();  
for i in xrange(divisions):
    N1 = (myrank+i*nprocs)*length
    N2 = min((myrank+1+i*nprocs)*length,dims3)

    #CFL.all_distances_grid(pos_grid,delta,BoxSize,bins,Rmin,Rmax,pairs,xi,
    #                       N1,N2,serial=do_serial,threads=omp_threads)

    #This function computes the product of delta(r1)*delta(r2) using 
    #an already computed list of neighbors and distances
    CFL.distances_grid(N1,N2,dims,delta,pairs,xi,indexes_coord,
                       indexes_distances)             
print 'cpu ',myrank,' time =',time.clock()-start_time

#send results to master
pairs = comm.gather(pairs,root=0)
xi    = comm.gather(xi,root=0)

#master sum the results and write file
if myrank==0:

    #sum the partial results from all processors
    pairs = np.sum(pairs,axis=0,dtype=np.int64)
    xi    = np.sum(xi,axis=0,dtype=np.float64)

    #remove bins with no pairs in it
    indexes = np.where(pairs!=0)[0];
    pairs=pairs[indexes]; xi=xi[indexes]; r_xi=r_xi[indexes]; del indexes

    #compute correlation function
    xi *= (1.0/pairs)
    print 'Sum contributions from %.4e pairs'%(np.sum(pairs,dtype=np.int64))
    print xi

    #save results to file
    np.savetxt(f_out,np.transpose([r_xi,xi]))



