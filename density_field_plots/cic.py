#DESCRIPTION:
#This code is used to extract a slice of the density field from 
#an N-body snapshot.

#VARIABLES:
#snapshot_fname --> name of the N-body snapshot
#dims ------------> number of points per dimension to compute the density field
#f_out -----------> name of the output file
#cells -----------> number of cells to use in the slice
#for instance, if the box has a size of 100 Mpc/h and dims=100 and we want 
#a slice of 13 Mpc/h set cells=13

#USAGE:
#copy this code in the folder containing the snapshot over which the 
#density field slice wants to be extracted. Set the values of the variables
#and type pyhon cic.py

import numpy as np
import readsnap
import CIC_library as CIC
import sys

rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
############################# INPUT ############################
snapshot_fname='snapdir_013/snap_013'

dims=512
f_out='overdensity_GR_0.0.dat'

cells=5
################################################################
dims2=dims**2; dims3=dims**3

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc

#read CDM and NU positions
pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)
pos2=readsnap.read_block(snapshot_fname,"POS ",parttype=2)

#computes OmegaCDM and OmegaNU
OmegaCDM = Nall[1]*Masses[1]/(BoxSize**3*rho_crit*1e-9)
OmegaNU  = Nall[2]*Masses[2]/(BoxSize**3*rho_crit*1e-9)
print 'OmegaCDM=',OmegaCDM; print 'OmegaNU= ',OmegaNU
print 'OmegaDM= ',OmegaCDM+OmegaNU

#compute the delta in the mesh points for the component 1
delta1=np.zeros(dims3,dtype=np.float32)
CIC.CIC_serial(pos1,dims,BoxSize,delta1) #computes the density
print np.sum(delta1,dtype=np.float64),'should be equal to',len(pos1)
delta1=delta1*(dims3*1.0/len(pos1))-1.0  #computes the delta
print np.min(delta1),'< delta1 <',np.max(delta1)

#compute the delta in the mesh points for the component 2
delta2=np.zeros(dims3,dtype=np.float32)
if len(pos2!=0):
    CIC.CIC_serial(pos2,dims,BoxSize,delta2) #computes the density
    print np.sum(delta2,dtype=np.float64),'should be equal to',len(pos2)
    delta2=delta2*(dims3*1.0/len(pos2))-1.0  #computes the delta
    print np.min(delta2),'< delta2 <',np.max(delta2)
    
#compute the total delta (rho/mean_rho-1). Formula easily obtained.
delta=np.empty(dims3,dtype=np.float32)
delta=OmegaCDM/(OmegaCDM+OmegaNU)*delta1+OmegaNU/(OmegaCDM+OmegaNU)*delta2
print np.min(delta),'< delta <',np.max(delta)

overdensity=np.zeros(dims2,dtype=np.float32)
numbers=np.arange(cells)*dims2

for i in range(dims2):
    #move from delta to overdensity
    overdensity[i]=np.sum(delta[numbers+i]+1.0)/cells

f=open(f_out,'w')
for i in range(dims2):
    index_x=i/dims2
    index_y=(i%dims2)/dims
    index_z=(i%dims2)%dims

    x=index_x*BoxSize/dims
    y=index_y*BoxSize/dims
    z=index_z*BoxSize/dims

    f.write(str(y)+' '+str(z)+' '+str(overdensity[i])+'\n')
f.close()
