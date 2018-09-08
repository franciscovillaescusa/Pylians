from mpi4py import MPI
import numpy as np
import readsnapHDF5 as rs
import HI_library as HIL
import sys,os,glob,h5py,time
import MAS_library as MASL
import HI.HI_image_library as HIIL
import groupcat
import sorting_library as SL
import units_library as UL

from pylab import *
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm



####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
################################ INPUT ########################################
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

snapnums = [99] #17(z=5) 21(z=4) 25(z=3) 33(z=2) 50(z=1) 99(z=0)

# if halo_num = None then coordinates are set from x_min, x_max...etc
halo_num = 36 #261299 277266 428572 ## 219150 287753 294114 436118
BoxSize_image = 3.0

# this only applies if halo_num = None
x_min, x_max = 0, 75 #11.75, 21.75
y_min, y_max = 0, 75 #62.35, 72.35
z_min, z_max = 0.0, 5.0

padding = 0.1 #add to x- and y- axis to get the proper density at the borders

redshift_space = False
axis = 0

dims = 5000

component = 'gas'  #'HI', 'gas', 'CDM'
obj       = 'halo'  #'region', 'halo'
###############################################################################

if component=='HI':
	vmin, vmax, cmap = 1e15, 1e23, 'gnuplot'
	ptype = 0
elif component in ['gas','CDM']:
	vmin, vmax, cmap = 1e19, 1e23, 'jet' #1e19-1e23
	if component=='gas':  ptype = 0
	if component=='CDM':  ptype = 1
else:  raise Exception("wrong component")


for snapnum in snapnums:

	comm.Barrier()

	density = np.zeros((dims,dims), dtype=np.float32)  

	# read snapshot and find number of subfiles
	snapshot_root = '%s/output/'%run
	snapshot = snapshot_root + 'snapdir_%03d/snap_%03d'%(snapnum,snapnum)
	header   = rs.snapshot_header(snapshot)
	redshift = header.redshift
	h        = header.hubble

	if halo_num is not(None):
		xmin = x_max = y_min = y_max = z_min = z_max = None

	# only master reads halo catalogue
	if myrank==0:
		# read number of particles in halos and subhalos and number of subhalos
		print '\nReading halo catalogue...'
		halos = groupcat.loadHalos(snapshot_root, snapnum, 
		                           fields=['GroupPos','GroupMass','Group_R_TopHat200'])
		halo_pos  = halos['GroupPos']/1e3          #Mpc/h
		halo_R    = halos['Group_R_TopHat200']/1e3 #Mpc/h
		halo_mass = halos['GroupMass']*1e10        #Msun/h
		del halos

		# find the coordinates of the region
		if halo_num is not (None):
			x_max, y_max, z_max = halo_pos[halo_num] + BoxSize_image/2.0
			x_min, y_min, z_min = halo_pos[halo_num] - BoxSize_image/2.0
			print 'M_halo = %.3e Msun/h'%(halo_mass[halo_num])
			print 'R_halo = %.3e Mpc/h\n'%(halo_R[halo_num])

		print '%6.3f < X < %6.3f'%(x_min, x_max)
		print '%6.3f < Y < %6.3f'%(y_min, y_max)
		print '%6.3f < Z < %6.3f'%(z_min, z_max)

	x_min = comm.bcast(x_min, root=0);  x_max = comm.bcast(x_max, root=0)
	y_min = comm.bcast(y_min, root=0);  y_max = comm.bcast(y_max, root=0)
	z_min = comm.bcast(z_min, root=0);  z_max = comm.bcast(z_max, root=0)

	# find the names of the output files
	if ptype==0:
	        #fout = 'HI_%s_%s_%.2f-%.2f-%.2f-%.2f-%.2f-%.2f_RS=%d_%d_z=%.1f.hdf5'\
		fout = 'HI_%s_%s_%.2f-%.2f-%.2f-%.2f-%.2f-%.2f_z=%.1f.hdf5'\
		%(obj, str(halo_num), x_min, x_max, y_min, y_max, z_min, z_max, 
		  #int(redshift_space), axis, 
		  redshift)
	if ptype==1:
		fout = 'CDM_%s_%s_%.2f-%.2f-%.2f-%.2f-%.2f-%.2f_z=%.1f.hdf5'\
		    %(obj, str(halo_num), x_min, x_max, y_min, y_max, z_min, z_max, 
		  #int(redshift_space), axis, 
		  redshift)
	fout_image = 'image_%s_%s_%s_%.2f-%.2f-%.2f-%.2f-%.2f-%.2f_z=%.1f.pdf'\
		%(obj,component,str(halo_num),x_min, x_max, y_min, y_max, z_min, z_max, redshift)
	fout_df = 'df_%s_%s_%s_%.2f-%.2f-%.2f-%.2f-%.2f-%.2f_%d_z=%.1f.hdf5'\
		%(obj, component, str(halo_num), x_min, x_max, y_min, y_max, 
			z_min, z_max, dims, redshift)

	if obj=='halo':
		if myrank>0:  sys.exit()
		if not(os.path.exists(fout)):
			HIIL.Illustris_halo(snapshot_root, snapnum, halo_num, TREECOOL_file, fout, ptype)

	if obj=='region':
		if not(os.path.exists(fout)):
			HIIL.Illustris_region(snapshot_root, snapnum, TREECOOL_file, x_min, x_max, 
	    		         	      y_min, y_max, z_min, z_max, padding, fout, redshift_space, axis)
		if myrank>0:  continue 




	##### compute/read density field #####
	if os.path.exists(fout_df):
		f = h5py.File(fout_df, 'r')
		density = f['df'][:];  f.close()

	else:

		print '\nFinding density field...'
		# read particle positions, HI masses, gas masses and radii
		f = h5py.File(fout,'r')
		if ptype==0:
			pos,  radii = f['pos'][:],  f['radii'][:]
			M_HI, mass  = f['M_HI'][:], f['mass'][:]
		if ptype==1:
			pos,  radii = f['pos'][:],  f['radii'][:]
			mass        = f['mass_c'][:]
		f.close()

		density = np.zeros((dims,dims), dtype=np.float64)
		pos2D = np.ascontiguousarray(pos[:,0:2])
		if component=='HI':
			MASL.voronoi_RT_2D(density, pos2D, M_HI, radii, 
					   x_min, y_min, 0, 1, x_max-x_min, 
					   False)
		if component in ['gas','CDM']:
			MASL.voronoi_RT_2D(density, pos2D, mass, radii, 
					   x_min, y_min, 0, 1, x_max-x_min, 
					   False)

		# convert (Msun/h)/(Mpc/h)^2 to cm^{-2}
		factor = h*(1.0+redshift)**2*\
			(UL.units().Msun_g)/(UL.units().mH_g)/(UL.units().Mpc_cm)**2
		####### CAUTION!!!! ########
		#factor/=(1.0+redshift)**2
		###########################

		density *= factor



		# save density field to file
		f = h5py.File(fout_df, 'w')
		f.create_dataset('df', data=density)
		f.close()



	# set minimum/maximum values of the density field
	print '%.5e < column density (cm^{-2}) < %.5e'\
		%(np.min(density), np.max(density))
	density[np.where(density<vmin)]=vmin
	density = np.transpose(density)




	############### IMAGE ###############
	print '\nMaking image...'

	pos_halo = halo_pos[halo_num]

	indexes = np.where((halo_pos[:,0]>x_min) & (halo_pos[:,0]<x_max) &\
	                   (halo_pos[:,1]>y_min) & (halo_pos[:,1]<y_max) &\
	                   (halo_pos[:,2]>z_min) & (halo_pos[:,2]<z_max) &\
	                   (halo_mass>3e9))[0]

	halo_R    = halo_R[indexes]
	halo_pos  = halo_pos[indexes,0:2]
	halo_mass = halo_mass[indexes]

	indexes   = np.where(halo_R>0.2)[0]
	halo_R    = halo_R[indexes]
	halo_pos  = halo_pos[indexes]
	halo_mass = halo_mass[indexes]

	fig = figure(figsize=(18.8,15)) #create the figure
	ax1 = fig.add_subplot(111) 

	ax1.set_xlim([x_min, x_max])
	ax1.set_ylim([y_min, y_max])

	ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #x-axis label
	ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #y-axis label

	cax = ax1.imshow(density, cmap=cmap,#get_cmap(my_cmap), 
			 origin='lower', extent=[x_min, x_max, y_min, y_max],
	                 #vmin=0,vmax=np.max(density))
	                 norm = LogNorm(vmin=vmin,vmax=vmax),interpolation='sinc')

	cbar = fig.colorbar(cax)
	cbar.set_label(r"${\rm cm^{-2}}$",fontsize=20)

	for i in xrange(len(halo_pos)):
		circle = Circle((halo_pos[i,0],halo_pos[i,1]), halo_R[i],
	    	            color='white', fill=False)
		#ax1.add_artist(circle)

	#ax1.plot([pos_halo[0]],[pos_halo[1]], marker='o',
		#markersize=4,color='white')

	savefig(fout_image, bbox_inches='tight', dpi=300)
	close(fig)




