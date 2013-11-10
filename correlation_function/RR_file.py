#This code computes the RR file, needed for computing the 2pt-correlation
#function. If one needs to compute the 2pt correlation function the code 
#correlation_function.py code also does it. 
#This code is useful when the DD file is heavily used: e.g.
#when computing the HOD parameters

from mpi4py import MPI
import numpy as np
import correlation_function_library as CF
import random
import sys

comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

################################### INPUT ####################################
random_file = '/home/villa/disksom2/Correlation_function/Random_catalogue/'
random_file += 'random_catalogue_1e7.dat'

#Be carefull!!! before create a new file
create_random_file = False #True or False !!!!!!

points=10000000

RR_name='RR_0.1_100_60_1e7_1000Mpc.dat'

BoxSize=1000.0 #Mpc/h
bins=60
Rmin=0.1    #Mpc/h
Rmax=100.0  #Mpc/h
##############################################################################


#### MASTER ####
if myrank==0:

    if create_random_file:
        #create the random catalogue
        print 'creating random catalogue...'
        pos_r=np.random.random((points,3)).astype(np.float32)
        f=open(random_file,'wb')
        for i in range(len(pos_r)):
            f.write(pos_r[i])
        f.close(); print 'done'
        pos_r*=BoxSize #muliply for the correct units
    else:
        print 'reading random catalogue...'
        #read the random catalogue
        dt=np.dtype((np.float32,3))
        pos_r=np.fromfile(random_file,dtype=dt)*BoxSize #Mpc/h
        print 'done'

    #compute the RR file
    print 'computing number of random pairs...'
    CF.DD_file(pos_r,BoxSize,RR_name,bins,Rmin,Rmax)
    print 'done'

#### SLAVES ####
else:
    pos_r=None
    CF.DD_file(pos_r,BoxSize,RR_name,bins,Rmin,Rmax)
               
