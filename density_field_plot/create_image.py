import numpy as np
import readsnap
import CIC_library as CIC
import sys

from pylab import *
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm

################################### INPUT ######################################
snapshot_fname = 'snapdir_003/snap_003'

x_min = None;    x_max = None
y_min = None;    y_max = None
z_min = 0.0;     z_max = 10.0

dims = 512

plane = 'XY'   #'XY','YZ' or 'XZ'

save_density_field = True   #whether save the density field into a file
density_field_fname = 'density_field.txt'

min_overdensity = 0.1    #minimum overdensity to plot
max_overdensity = 100.0  #maximum overdensity to plot

fout = 'Image.png'
################################################################################

plane_dict = {'XY':[0,1], 'XZ':[0,2], 'YZ':[1,2]}

#read snapshot head and obtain BoxSize, Omega_m and Omega_L                   
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h                                               
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h                                              
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc 

#set the limits of slice
if x_min==None:  x_min = 0.0     
if y_min==None:  y_min = 0.0   
if z_min==None:  z_min = 0.0
if x_max==None:  x_max = BoxSize
if y_max==None:  y_max = BoxSize
if z_max==None:  z_max = BoxSize

#check that the plane is square
if plane=='XY':
    length1 = x_max-x_min;  length2 = y_max-y_min;  depth = z_max-z_min 
    offset1 = x_min;        offset2 = y_min
elif plane=='XZ':
    length1 = x_max-x_min;  length2 = z_max-z_min;  depth = y_max-y_min 
    offset1 = x_min;        offset2 = z_min
else:
    length1 = y_max-y_min;  length2 = z_max-z_min;  depth = x_max-x_min 
    offset1 = y_min;        offset2 = z_min
if length1!=length2:
    print 'Plane has to be a square!!!'; sys.exit()
BoxSize_slice = length1

#read positions and masses of the CDM particles
pos  = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3  #Mpc/h 
mass = readsnap.read_block(snapshot_fname,"MASS",parttype=1)*1e10 #Msun/h 

#compute the mean mass in each cell of the slice 
mass_density = np.sum(mass,dtype=np.float64)*1.0/BoxSize**3 #mass/(Mpc/h)^3
V_cell = BoxSize_slice**2*depth*1.0/dims**2   #slice cell volume in (Mpc/h)^3
mean_mass = mass_density*V_cell 

#keep only with the particles in the slice
indexes = np.where((pos[:,0]>x_min) & (pos[:,0]<x_max) &
                   (pos[:,1]>y_min) & (pos[:,1]<y_max) &
                   (pos[:,2]>z_min) & (pos[:,2]<z_max) )
pos = pos[indexes];   mass = mass[indexes]
print 'Coordinates of the particles in the slice:'
print '%.4f < X < %.4f'%(np.min(pos[:,0]),np.max(pos[:,0]))
print '%.4f < Y < %.4f'%(np.min(pos[:,1]),np.max(pos[:,1]))
print '%.4f < Z < %.4f'%(np.min(pos[:,2]),np.max(pos[:,2]))

#project particle positions into a 2D plane
pos = pos[:,plane_dict[plane]]
print '\nCoordinates of the particles in the plane:'
print '%.4f < axis 1 < %.4f'%(np.min(pos[:,0]),np.max(pos[:,0]))
print '%.4f < axis 2 < %.4f'%(np.min(pos[:,1]),np.max(pos[:,1]))

#define the density array
overdensity = np.zeros(dims**2,dtype=np.float32)

#compute the mass in each cell of the grid
CIC.CIC_serial_2D(pos,dims,BoxSize_slice,overdensity,weights=mass)
print '%.4e should be equal to\n%.4e'\
    %(np.sum(overdensity,dtype=np.float64),np.sum(mass,dtype=np.float64))
del pos,mass

#compute overdensities
overdensity = overdensity/mean_mass
print np.min(overdensity),'< rho/<rho> <',np.max(overdensity)

#reshape the array to make it a matrix
overdensity = np.reshape(overdensity,(dims,dims))

#save density field to file
if save_density_field:  np.save(density_field_fname,overdensity)
    




############### IMAGE ###############
fig = figure()    #create the figure
ax1 = fig.add_subplot(111) 

ax1.set_xlim([offset1,offset1+length1])  #set the range for the x-axis
ax1.set_ylim([offset2,offset2+length2])  #set the range for the y-axis

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #y-axis label

if min_overdensity==None:  min_overdensity = np.min(overdensity)
if max_overdensity==None:  max_overdensity = np.max(overdensity)

overdensity[np.where(overdensity<min_overdensity)] = min_overdensity

cax = ax1.imshow(overdensity,cmap=get_cmap('jet'),origin='lower',
                 extent=[offset1, offset1+length1, offset2, offset2+length2],
                 #vmin=min_density,vmax=max_density)
                 norm = LogNorm(vmin=min_overdensity,vmax=max_overdensity))

cbar = fig.colorbar(cax)
cbar.set_label(r"$\rho/\bar{\rho}$",fontsize=20)

savefig(fout, bbox_inches='tight')
close(fig)
