# This script takes an uniform density field and places random spheres
# of random sizes with a profile of delta(r)=-1*(1-(r/R)^3)
# Then it identifies the voids using the void finder. Finally, it plots
# the average density field across the entire box of the input and 
# recovered void field

import numpy as np 
import sys,os,time
import void_library as VL
from pylab import *
from matplotlib.colors import LogNorm

############################### INPUT #####################################
BoxSize = 1000.0 #Mpc/h
Nvoids  = 5      #number of random voids 
dims    = 512    #grid resolution to find voids

threshold = -0.5 #for delta(r)=-1*(1-(r/R)^3)

Rmax = 200.0   #maximum radius of the input voids
Rmin = 20.0    #minimum radius of the input voids
bins = 50      #number of radii between Rmin and Rmax to find voids

Omega_m = 0.3175 #only needed for the void finder

threads = 28 #openmp threads

f_out = 'Spheres_test.png'
###########################################################################


fig=figure(figsize=(15,7))
ax1=fig.add_subplot(121) 
ax2=fig.add_subplot(122) 

# create density field with random spheres
V = VL.random_spheres(BoxSize, Rmin, Rmax, Nvoids, dims)
delta = V.delta

# find voids
V2 = VL.void_finder(delta, BoxSize, threshold, Rmax, Rmin, bins, Omega_m, 
              threads, void_field=True)
delta2 = V2.in_void

ax1.imshow(np.mean(delta[:,:,:],axis=0),
	cmap=get_cmap('nipy_spectral'),origin='lower',
	vmin=-1, vmax=0.0,
	extent=[0, BoxSize, 0, BoxSize])

ax2.imshow(np.mean(delta2[:,:,:],axis=0),
	cmap=get_cmap('nipy_spectral_r'),origin='lower',
	vmin=0, vmax=1.0,
	extent=[0, BoxSize, 0, BoxSize])

savefig(f_out, bbox_inches='tight')
show()

pos = V.void_pos
R   = V.void_radius
print '#### Input void positions ####'
print '  X      Y      Z      R'
for i in xrange(Nvoids):
	print "%6.2f %6.2f %6.2f %6.2f"%(pos[i,0], pos[i,1], pos[i,2], R[i])

print ' '

pos = V2.void_pos
R   = V2.void_radius
print '### Output void positions ###'
print '  X      Y      Z      R'
for i in xrange(pos.shape[0]):
	print "%6.2f %6.2f %6.2f %6.2f"%(pos[i,0], pos[i,1], pos[i,2], R[i])









