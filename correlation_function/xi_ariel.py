from mpi4py import MPI
import numpy as np
import CIC_library as CIC
import correlation_function_library as CFL
import readsnap
import sys,os
import scipy.weave as wv
import time


###### MPI DEFINITIONS ######                                                 
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

################################### INPUT #####################################
snapshot_fname = 'ics'

bins = 50
Rmin = 10.0
Rmax = 150.0

dims = 100

do_serial = True
omp_threads = 1

f_out = 'xi_75_z=99_4.dat'
###############################################################################
dims2 = dims**2;  dims3 = dims**3
delta = None;  pos_grid = None;  BoxSize = None #to gather later from master


#only master read the positions and computes delta
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

    #read particle positions
    pos = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h  

    """
    BoxSize = 500.0 #Mpc/h
    x = np.fromfile('posxALPTCICz0.000G512S1.dat',dtype=np.float32,count=-1)
    y = np.fromfile('posyALPTCICz0.000G512S1.dat',dtype=np.float32,count=-1)
    z = np.fromfile('poszALPTCICz0.000G512S1.dat',dtype=np.float32,count=-1)

    pos = np.empty((len(x),3),dtype=np.float32)
    pos[:,0] = x; pos[:,1] = y; pos[:,2] = z; del x,y,z
    """

    print len(pos)
    print np.min(pos[:,0]),'< X <',np.max(pos[:,0])
    print np.min(pos[:,1]),'< Y <',np.max(pos[:,1])
    print np.min(pos[:,2]),'< Z <',np.max(pos[:,2])

    #compute the mean number of particles per cell
    mean_particle_number = len(pos)*1.0/dims**3

    delta = np.zeros(dims**3,dtype=np.float32)
    CIC.NGP_serial(pos,dims,BoxSize,delta) #number of particles in each cell
    #CIC.CIC_serial(pos,dims,BoxSize,delta) #number of particles in each cell
    print '%d should be equal to\n%d'%(np.sum(delta,dtype=np.float64),len(pos))
    delta = delta/mean_particle_number - 1.0
    print '%f < [(n - <n>) / <n>] < %f'%(np.min(delta),np.max(delta))
    print '%f should be close to 0'%(np.sum(delta,dtype=np.float64))

    pos_grid = np.empty((dims**3,3),dtype=np.float32)
    for i in xrange(dims):
        for j in xrange(dims):
            for k in xrange(dims):
                index = dims2*i + dims*j + k
                pos_grid[index] = [i,j,k]
    pos_grid*=BoxSize/dims; #print pos_grid
    print 'Total number of pairs = %.3e'%(dims3*(dims3-1)/2)


#send arrays computed by master to slaves
pos_grid = comm.bcast(pos_grid,root=0)
delta    = comm.bcast(delta,root=0)
BoxSize  = comm.bcast(BoxSize,root=0) 
BoxSize  = float(BoxSize) #to avoid problems with scipy weave

#define r-binning
#r_bins = np.linspace(0.0,500,bins+1)
r_bins = np.logspace(np.log10(Rmin),np.log10(Rmax),bins+1)
r_xi   = 0.5*(r_bins[:-1] + r_bins[1:])

#compute distances between the particle 0 and all others between Rmin and Rmax
diff = pos_grid-pos_grid[0]
indexes = np.where(diff>BoxSize/2.0);  diff[indexes]-=BoxSize; del indexes
indexes = np.where(diff<-BoxSize/2.0); diff[indexes]+=BoxSize; del indexes
distances_ref = np.sqrt(diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2); del diff
indexes = np.where((distances_ref>Rmin) & (distances_ref<Rmax))[0]
distances_ref=distances_ref[indexes]
indexes_distances = (np.log10(distances_ref/Rmin)\
                         /(np.log10(Rmax/Rmin)/bins)).astype(np.int32)
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

start_time=time.clock();  
for i in xrange(divisions):
    N1 = (myrank+i*nprocs)*length
    N2 = min((myrank+1+i*nprocs)*length,dims3)

    #CFL.all_distances_grid(pos_grid,delta,BoxSize,bins,Rmin,Rmax,pairs,xi,
    #                       N1,N2,serial=do_serial,threads=omp_threads)

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
    indexes=np.where(pairs!=0)[0];
    pairs=pairs[indexes]; xi=xi[indexes]; r_xi=r_xi[indexes]; del indexes

    #compute correlation function
    xi*=1.0/pairs
    print 'Sum contributions from %.4e pairs'%(np.sum(pairs,dtype=np.int64))
    print xi

    #save results to file
    np.savetxt(f_out,np.transpose([r_xi,xi]))



