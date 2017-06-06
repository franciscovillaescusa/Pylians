from mpi4py import MPI
import numpy as np
import sys,os

###### MPI DEFINITIONS ###### 
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

files = 10000

# find the numbers that each cpu will work with
numbers = np.where(np.arange(files)%nprocs==myrank)[0]

for i in numbers:
    print 'Cpu %3d working with number %%4d'%(myrank,i)
