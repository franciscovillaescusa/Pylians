import numpy as np
import plotting_library as PL
import sys,os

from pylab import *
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm

################################# INPUT #######################################
snapshot_fname = '../snapdir_003/snap_003'

x_min = 0.0;  x_max = 1000.0
y_min = 0.0;  y_max = 1000.0
z_min = 0.0;  z_max = 10.0

dims   = 512
ptypes = [1]   # 0-Gas, 1-CDM, 2-NU, 4-Stars; can deal with several species
plane  = 'XY'  #'XY','YZ' or 'XZ'
MAS    = 'PCS' #'NGP', 'CIC', 'TSC', 'PCS' 

min_overdensity = 0.1    #minimum overdensity to plot
max_overdensity = 100.0  #maximum overdensity to plot
scale           = 'linear' #'linear' or 'log'
cmap            = 'nipy_spectral'

fout = 'Image.png'

save_density_field = True  #whether save the density field into a file
###############################################################################

# find the name of the density field
f_df = PL.density_field_name(snapshot_fname, x_min, x_max, y_min, y_max, 
                             z_min, z_max, dims, ptypes, plane, MAS)

# find the geometric values of the density field square
len_x, off_x, len_y, off_y, depth, BoxSize_slice = PL.geometry(snapshot_fname, 
    plane, x_min, x_max, y_min, y_max, z_min, z_max)

# compute/read density field
if os.path.exists(f_df):
    print '\nDensity field already computed. Reading it from file...'
    overdensity = np.load(f_df)
else:
    print '\nComputing density field...'
    overdensity = PL.density_field_2D(snapshot_fname, x_min, x_max, y_min, y_max, 
                                      z_min, z_max, dims, ptypes, plane, MAS, 
                                      save_density_field)

############### IMAGE ###############
print '\nCreating the figure...'
fig = figure()    #create the figure
ax1 = fig.add_subplot(111) 

ax1.set_xlim([off_x, off_y+len_x])  #set the range for the x-axis
ax1.set_ylim([off_y, off_y+len_y])  #set the range for the y-axis

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #y-axis label

if min_overdensity==None:  min_overdensity = np.min(overdensity)
if max_overdensity==None:  max_overdensity = np.max(overdensity)

overdensity[np.where(overdensity<min_overdensity)] = min_overdensity

if scale=='linear':
    cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                     extent=[off_x, off_x+len_x, off_y, off_y+len_y],
                     interpolation='sinc',
                     vmin=min_overdensity,vmax=max_overdensity)
else:
    cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                     extent=[off_x, off_x+len_x, off_y, off_y+len_y],
                     interpolation='sinc',
                     norm = LogNorm(vmin=min_overdensity,vmax=max_overdensity))

cbar = fig.colorbar(cax)
cbar.set_label(r"$\rho/\bar{\rho}$",fontsize=20)

savefig(fout, bbox_inches='tight')
close(fig)
#####################################

