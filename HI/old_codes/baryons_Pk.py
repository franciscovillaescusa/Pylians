#This code computes the P(k) of the gas particles taking into account their
#smoothing lengths

import numpy as np
import readsnap
import sys
import Power_spectrum_library as PSL
import CIC_library as CIC


pi=np.pi
################################### INPUT #####################################
#snapshot and halo catalogue
snapshot_fname='../Efective_model_15Mpc/snapdir_008/snap_008'
groups_fname='../Efective_model_15Mpc/FoF_0.2'
groups_number=8

divisions=3 #the SPH spheres will be split among divisions^3 points

dims=512
threads=8

f_out='borrar.dat'
###############################################################################


#read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h=head.hubble

#read the positions and SPH smoothing lengths of the gas particles
pos=readsnap.read_block(snapshot_fname,"POS ",parttype=0)/1e3 #Mpc/h
radii=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3 #Mpc/h

"""
print len(np.where(radii<(BoxSize/1024.0))[0])*1.0/len(radii)
bins_histo=np.logspace(np.log10(np.min(radii)),np.log10(np.max(radii)),101)
middle_bin=0.5*(bins_histo[1:]+bins_histo[:-1])
H=np.histogram(radii,bins=bins_histo)[0]*1.0/len(radii)
print np.sum(H,dtype=np.float64)
f=open('borrar.dat','w')
for i in range(100):
    f.write(str(middle_bin[i])+' '+str(H[i])+'\n')
f.close()
"""

#compute the density in the grid cells
delta=np.zeros(dims**3,dtype=np.float32)
#CIC.CIC_serial(pos,dims,BoxSize,delta) #computes the density
#CIC.NGP_serial(pos,dims,BoxSize,delta) #computes the density
CIC.SPH_gas(pos,radii,divisions,dims,BoxSize,threads,delta,weights=None)

print delta
print np.sum(delta,dtype=np.float64)
print len(pos)

#compute the mean number of particles per grid cell
mean_density=len(pos)*1.0/dims**3

#compute the value of delta = density / <density> - 1
delta=delta/mean_density-1.0  #computes delta
print 'numbers should be equal:',np.sum(delta,dtype=np.float64),0.0
print np.min(delta),'< delta <',np.max(delta)

#compute the PS
Pk=PSL.power_spectrum_given_delta(delta,dims,BoxSize,do_CIC_correction=False)

#write total HI P(k) file
f=open(f_out,'w')
for i in range(len(Pk[0])):
    f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
f.close()
