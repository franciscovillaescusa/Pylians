import numpy as np
import sys,os,h5py
import groupcat
import MAS_library as MASL
import scipy.ndimage
import redshift_space_library as RSL

from pylab import *
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm

rho_crit = 2.77537e11 #h^2 Msun/Mpc^3
################################ INPUT ####################################
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG_DM'

snapnum = 50

alpha, M0, Mmin = 0.53, 1.5e10, 6e11 #z=1

x_min, x_max = 0.0, 75.0 #Mpc/h
y_min, y_max = 0.0, 75.0 #Mpc/h
BoxSize = 75.0 #Mpc/h
axis    = 2

dims = 5000

Omega_m = 0.3089
Omega_l = 0.6911
h       = 0.6774
z       = 1.0
#######################################################################

H0 = 100.0 #km/s/(Mpc/h)
Hz = 100.0*np.sqrt(Omega_m*(1.0+z)**3 + Omega_l)

V_cell = BoxSize**3*(5.0/BoxSize)/dims**2 #(Mpc/h)^3



print '\nReading halo catalogue...'
snapshot_root = '%s/output/'%run
halos = groupcat.loadHalos(snapshot_root, snapnum, 
                           fields=['GroupPos','GroupMass','GroupVel'])
halo_pos  = halos['GroupPos']/1e3     #Mpc/h
halo_vel  = halos['GroupVel']*(1.0+z) #km/s
halo_mass = halos['GroupMass']*1e10   #Msun/h
del halos

# move halo positions to redshift-space                  
RSL.pos_redshift_space(halo_pos, halo_vel, BoxSize, Hz, z, axis)


print np.min(halo_pos[:,0]),np.max(halo_pos[:,0])
print np.min(halo_pos[:,1]),np.max(halo_pos[:,1])
print np.min(halo_pos[:,2]),np.max(halo_pos[:,2])


indexes = np.where((halo_pos[:,2]>=0.0) & (halo_pos[:,2]<5.0))[0]

halo_pos  = halo_pos[indexes]
halo_mass = halo_mass[indexes]
M_HI = M0*(halo_mass/Mmin)**alpha*np.exp(-(Mmin/halo_mass)**0.35)

print np.min(halo_pos[:,0]),np.max(halo_pos[:,0])
print np.min(halo_pos[:,1]),np.max(halo_pos[:,1])
print np.min(halo_pos[:,2]),np.max(halo_pos[:,2])

HI_mass = np.zeros((dims,dims), dtype=np.float32)

halo_pos = halo_pos[:,0:2]

MASL.MA(halo_pos,HI_mass,BoxSize,'NGP',W=M_HI)

f = h5py.File('HI_mass_map_5Mpc_RS_Nbody_z=1.hdf5','w')
f.create_dataset('HI_mass', data=HI_mass)
f.close()



Tb = 189.0*(H0*(1.0+z)**2/Hz)*h*(HI_mass/V_cell)/rho_crit #mK

print np.min(Tb),np.max(Tb)
print np.sum(Tb, dtype=np.float64)

Tb = scipy.ndimage.filters.gaussian_filter(Tb, 10)

print np.sum(Tb, dtype=np.float64)
print np.min(Tb),np.max(Tb)


vmin, vmax, cmap = 1e-6, 10, 'gnuplot'




fig = figure(figsize=(18.8,15)) #create the figure
ax1 = fig.add_subplot(111) 

ax1.set_xlim([x_min, x_max])
ax1.set_ylim([y_min, y_max])

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #y-axis label

cax = ax1.imshow(Tb, cmap=cmap,
                 origin='lower', extent=[x_min, x_max, y_min, y_max],
                 #vmin=vmin, vmax=vmax,
                 norm = LogNorm(vmin=vmin,vmax=vmax),
                 #interpolation='sinc')
                 interpolation='nearest')

cbar = fig.colorbar(cax)
cbar.set_label(r"$T_b\,[mK]$",fontsize=20)


savefig('HI_map3.png', bbox_inches='tight', dpi=300)
close(fig)
