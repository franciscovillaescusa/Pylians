from mpi4py import MPI
import numpy as np
import snapshot as sn
import readsnapHDF5 as rs
import HI_library as HIL
import sys,os,glob,h5py,time
import MAS_library as MASL
import HI.HI_image_library as HIIL
import groupcat
import sorting_library as SL

####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
################################ INPUT ########################################
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

snapnum = 17 #17(z=5) 21(z=4) 25(z=3) 33(z=2) 50(z=1) 99(z=0)

Mmin = 1.0e8   #Msun/h
Mmax = 1.0e15  #Msun/h

cell_size = 1.1 #Mpc/h must be larger than virial radii of the used halos
bins      = 50 #number of bins in the density profile
R1        = 1e-5 #Mpc/h first bin in the profiles goes from 0 to R1
###############################################################################

# find offset_root and snapshot_root                             
snapshot_root = '%s/output/'%run

# read header
snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
f        = h5py.File(snapshot+'.0.hdf5', 'r')
redshift = f['Header'].attrs[u'Redshift']
BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
Omega_m  = f['Header'].attrs[u'Omega0']
Omega_L  = f['Header'].attrs[u'OmegaLambda']
h        = f['Header'].attrs[u'HubbleParam']
Masses   = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
f.close()

if myrank==0:
    print '\nBoxSize         = %.1f Mpc/h'%BoxSize
    print 'Number of files = %d'%filenum
    print 'Omega_m         = %.3f'%Omega_m
    print 'Omega_l         = %.3f'%Omega_L
    print 'redshift        = %.3f'%redshift

# find the output name
fout1 = 'HI_profiles_%.1e-%.1e_%d_z=%.2f.hdf5'%(Mmin,Mmax,bins,redshift)
fout2 = 'HI_profile_new_%.1e-%.1e-z=%.1f.txt'%(Mmin,Mmax,redshift)

# read number of particles in halos and subhalos and number of subhalos
if myrank==0:  print '\nReading halo catalogue...'
halos = groupcat.loadHalos(snapshot_root, snapnum, 
                           fields=['GroupPos','GroupMass',
                                   'Group_R_TopHat200','Group_M_TopHat200'])
halo_pos  = halos['GroupPos']/1e3           #Mpc/h
halo_R    = halos['Group_R_TopHat200']/1e3  #Mpc/h
halo_mass = halos['Group_M_TopHat200']*1e10 #Msun/h
#halo_mass = halos['GroupMass']*1e10        #Msun/h
del halos

# consider only halos in the mass range and with R>0
indexes   = np.where((halo_mass>Mmin) & (halo_mass<Mmax) & (halo_R>0.0))[0]
halo_pos  = halo_pos[indexes]
halo_R    = halo_R[indexes]
halo_mass = halo_mass[indexes]

if myrank==0:
    print 'Found %d halos with masses %.2e < M < %.2e'\
        %(len(indexes), np.min(halo_mass), np.max(halo_mass))
    print 'Radii in the range %.5f < R < %.5f'%(np.min(halo_R), np.max(halo_R))
    print 'Using a cell size of %.3f Mpc/h'%cell_size

if np.max(halo_R>cell_size):
    raise Exception("cell size should be larger than biggest halo radius!!!")

comm.Barrier() # just to make the above output clear

# sort halo positions and find their ids
data = SL.sort_3D_pos(halo_pos, BoxSize, cell_size, return_indexes=True, 
		return_offset=False)
halo_pos  = data.pos_sorted
halo_R    = halo_R[data.indexes]
halo_mass = halo_mass[data.indexes]
halos     = halo_pos.shape[0]

# find the id = dims2*i + dims*j + k of the cell where halo is
halo_id = SL.indexes_3D_cube(halo_pos, BoxSize, cell_size)

# define the array containing the HI mass in each spherical shell
HI_mass_shell = np.zeros((halos, bins), dtype=np.float64)
part_in_halo  = np.zeros(halos,         dtype=np.int64)

# find the numbers each cpu will work on
array   = np.arange(filenum)
numbers = np.where(array%nprocs==myrank)[0]

# do a loop over each subsnapshot
for i in numbers:

    # find subfile name and read the number of particles in it
    snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(snapnum, snapnum, i)
    header = rs.snapshot_header(snapshot)
    npart  = header.npart 

    pos  = rs.read_block(snapshot, 'POS ', parttype=0, verbose=False)/1e3
    pos  = pos.astype(np.float32)
    MHI  = rs.read_block(snapshot, 'NH  ', parttype=0, verbose=False)#HI/H
    mass = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10
    SFR  = rs.read_block(snapshot, 'SFR ', parttype=0, verbose=False)
    indexes = np.where(SFR>0.0)[0];  del SFR

    # find the metallicity of star-forming particles
    metals = rs.read_block(snapshot, 'GZ  ', parttype=0, verbose=False)
    metals = metals[indexes]/0.0127

    # find densities of star-forming particles: units of h^2 Msun/Mpc^3
    rho = rs.read_block(snapshot, 'RHO ', parttype=0, verbose=False)*1e19
    Volume = mass/rho                            #(Mpc/h)^3
    radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 

    # find density and radius of star-forming particles
    radii_SFR = radii[indexes]    
    rho       = rho[indexes]

    # find HI/H fraction for star-forming particles
    MHI[indexes] = HIL.Rahmati_HI_Illustris(rho, radii_SFR, metals, redshift, 
                                            h, TREECOOL_file, Gamma=None,
                                            fac=1, correct_H2=True) #HI/H
    MHI *= (0.76*mass)


    # sort the positions of the particles
    data = SL.sort_3D_pos(pos, BoxSize, cell_size, return_indexes=True, 
        return_offset=True)
    pos    = data.pos_sorted
    MHI    = MHI[data.indexes]
    offset = data.offset    

    HIL.HI_profile(halo_pos, halo_R, halo_id, pos, MHI, offset,
        HI_mass_shell, part_in_halo, BoxSize, R1)

    print '\nDone with subfile %03d : %d'%(i,myrank)
    print 'Total HI so far = %.8e'%(np.sum(HI_mass_shell))


# sum the results of each indivual core
HI_mass_shell_total  = np.zeros((halos, bins), dtype=np.float64)
comm.Reduce([HI_mass_shell, MPI.DOUBLE], [HI_mass_shell_total, MPI.DOUBLE], 
    op=MPI.SUM, root=0)

part_in_halo_total  = np.zeros(halos, dtype=np.int64)
comm.Reduce([part_in_halo, MPI.LONG], [part_in_halo_total, MPI.LONG], 
    op=MPI.SUM, root=0)

if myrank==0:
    print HI_mass_shell_total
    print '%.3e'%(np.sum(HI_mass_shell_total))

    rho_HI = np.zeros((halos,bins), dtype=np.float64)
    r      = np.zeros((halos,bins), dtype=np.float64)

    # do a loop over all halos
    for i in xrange(halos):

    	# find r-bins and shell volume
	    r_bins = np.empty(bins+1, dtype=np.float64)
	    r_bins[0] = 1e-15
	    r_bins[1:] = np.logspace(np.log10(R1), np.log10(halo_R[i]), bins)
	    r[i] = 10**(0.5*(np.log10(r_bins[1:]) + np.log10(r_bins[:-1])))
	    V = 4.0*np.pi/3.0*(r_bins[1:]**3 - r_bins[:-1]**3)

	    # compute HI profile
	    rho_HI[i] = HI_mass_shell_total[i]*1.0/V

    f = h5py.File(fout1, 'w')
    f.create_dataset('r',          data=r)
    f.create_dataset('rho_HI',     data=rho_HI)
    f.create_dataset('M_HI_shell', data=HI_mass_shell_total)
    f.create_dataset('Mass',       data=halo_mass)
    f.create_dataset('Particles',  data=part_in_halo_total)
    f.close()

















