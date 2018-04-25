import numpy as np
import sys,os,h5py
import MAS_library as MASL
import time
import scipy.ndimage

from pylab import *
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm


rho_crit = 2.77537e11 #h^2 Msun/Mpc^3
############################ INPUT ####################################
f1 = '../HI_image/HI_region_None_0.00-75.00-0.00-75.00-0.00-5.00_z=1.0.hdf5'
f2 = '../HI_image/HI_region_None_0.00-75.00-0.00-75.00-0.00-5.00_RS=1_2_z=1.0.hdf5'

fout1 = 'HI_mass_map_5Mpc_z=1.hdf5'
fout2 = 'HI_mass_map_5Mpc_RS_0_z=1.hdf5'


dims = 5000
BoxSize = 75.0 #Mpc/h

x_min, x_max = 0.0, 75.0 #Mpc/h
y_min, y_max = 0.0, 75.0 #Mpc/h

Omega_m = 0.3089
Omega_l = 0.6911
h       = 0.6774
z       = 1.0
#######################################################################

H0 = 100.0 #km/s/(Mpc/h)
Hz = 100.0*np.sqrt(Omega_m*(1.0+z)**3 + Omega_l)

V_cell = BoxSize**3*(5.0/BoxSize)/dims**2 #(Mpc/h)^3

for fin, fout in zip([f1,f2],[fout1,fout2]):

    # read positions, radii and HI masses of particles
    f     = h5py.File(fin, 'r')
    pos   = f['pos'][:]
    M_HI  = f['M_HI'][:]
    radii = f['radii'][:]
    f.close()

    print np.min(pos[:,0]),np.max(pos[:,0])
    print np.min(pos[:,1]),np.max(pos[:,1])
    print np.min(pos[:,2]),np.max(pos[:,2])

    delta = np.zeros((dims,dims), dtype=np.float64)

    pos = pos[:,0:2]

    start = time.time()
    MASL.voronoi_NGP_2D(delta, pos, M_HI, radii, 0.0, 0.0, BoxSize, 10, 2, True)
    print 'Time taken = %.1f seconds'%(time.time()-start)

    print np.sum(M_HI, dtype=np.float64)
    print np.sum(delta, dtype=np.float64)

    f = h5py.File(fout,'w')
    f.create_dataset('HI_mass', data=delta)
    f.close()



"""
f = h5py.File('HI_mass_map_5Mpc_RS_0_z=1.hdf5', 'r')
HI_mass = f['HI_mass'][:]
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


savefig('HI_map2.png', bbox_inches='tight', dpi=300)
close(fig)
"""


