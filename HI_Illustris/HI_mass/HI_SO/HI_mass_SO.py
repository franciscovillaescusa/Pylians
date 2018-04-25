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
import argparse

####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

parser = argparse.ArgumentParser(description="This code computes the HI mass inside SO halos")

# non-optional arguments
parser.add_argument("snapnum", type=int, help="snapnum")
parser.add_argument("R_field", help="Group_R_Mean200, Group_R_TopHat200 or Group_R_Crit200")
args = parser.parse_args()



TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
################################ INPUT ########################################
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

snapnum = args.snapnum
#snapnum = 50  #17(z=5) 21(z=4) 25(z=3) 33(z=2) 50(z=1) 99(z=0)

cell_size = 3.0 #Mpc/h should be larger than virial radius of the halos used

R_field = args.R_field
#R_field = 'Group_R_TopHat200'
#R_field = 'Group_R_Crit200'
#R_field = 'Group_R_Mean200'
###############################################################################

# find offset_root and snapshot_root                             
snapshot_root = '%s/output/'%run

# read header
snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
f = h5py.File(snapshot+'.0.hdf5', 'r')
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

# find output name
if R_field=='Group_R_Crit200':
    M_field = 'Group_M_Crit200'
    fout    = 'M_HI_SO_Crit200_z=%.1f.hdf5'%redshift
elif R_field=='Group_R_TopHat200':
    M_field = 'Group_M_TopHat200'
    fout    = 'M_HI_SO_TopHat200_z=%.1f.hdf5'%redshift
elif R_field=='Group_R_Mean200':
    M_field = 'Group_M_Mean200'
    fout    = 'M_HI_SO_Mean200_z=%.1f.hdf5'%redshift
else:  raise Exception("%s not allowed!"%R_field)

# read number of particles in halos and subhalos and number of subhalos
if myrank==0:  print '\nReading halo catalogue...'
halos = groupcat.loadHalos(snapshot_root, snapnum, 
                           fields=['GroupPos','GroupMass',
                                   R_field, M_field])
halo_pos   = halos['GroupPos']/1e3          #Mpc/h
halo_R     = halos[R_field]/1e3             #Mpc/h
halo_mass  = halos['GroupMass']*1e10        #Msun/h
halo_mass2 = halos[M_field]*1e10            #Msun/h 
del halos

####### CAUTION ######
#halo_R = halo_R*1.7
######################

# consider only halos with radii larger than 0
indexes    = np.where(halo_R>0.0)[0]
halo_pos   = halo_pos[indexes]
halo_R     = halo_R[indexes]
halo_mass  = halo_mass[indexes]
halo_mass2 = halo_mass2[indexes] 

if myrank==0:
    print 'Found %d halos with masses %.2e < M < %.2e'\
        %(len(indexes),np.min(halo_mass), np.max(halo_mass))
    print 'Radii in the range %.5f < R < %.5f'%(np.min(halo_R), np.max(halo_R))
    print 'Using a cell size of %.3f Mpc/h'%cell_size

if np.max(halo_R>cell_size):
    raise Exception("cell size should be larger than biggest halo radius!!!")

comm.Barrier() #just to print the above nicely

# sort halo positions and find their ids
data = SL.sort_3D_pos(halo_pos, BoxSize, cell_size, return_indexes=True, 
		return_offset=False)
halo_pos   = data.pos_sorted
halo_R     = halo_R[data.indexes]
halo_mass  = halo_mass[data.indexes]
halo_mass2 = halo_mass2[data.indexes]
halos      = halo_pos.shape[0];  del data

# find the id = dims2*i + dims*j + k of the cell where halo is
halo_id = SL.indexes_3D_cube(halo_pos, BoxSize, cell_size)

# define the array containing the HI mass in each halo
M_HI_halo = np.zeros(halos, dtype=np.float64)

# find the numbers each cpu will work on
array   = np.arange(0, filenum)
numbers = np.where(array%nprocs==myrank)[0]

# do a loop over each subsnapshot
for i in numbers:

    # find subfile name and read the number of particles in it
    snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(snapnum, snapnum, i)
    header = rs.snapshot_header(snapshot)
    npart  = header.npart 

    print '\nWorking with subfile %03d : %d'%(i,myrank)
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
    offset = data.offset;  del data

    HIL.HI_mass_SO(halo_pos, halo_R, pos, MHI, offset, M_HI_halo, BoxSize)

    print '%.8e'%(np.sum(M_HI_halo))


# sum the results of each indivual core
M_HI_halo_total = np.zeros(halos, dtype=np.float64)
comm.Reduce([M_HI_halo, MPI.DOUBLE], [M_HI_halo_total, MPI.DOUBLE], 
    op=MPI.SUM, root=0)

if myrank==0:
    f = h5py.File(fout, 'w')
    f.create_dataset("M_HI_SO",  data=M_HI_halo_total)
    f.create_dataset("pos",      data=halo_pos)
    f.create_dataset("R",        data=halo_R)
    f.create_dataset("mass_FoF", data=halo_mass)
    f.create_dataset("mass_SO",  data=halo_mass2)
    f.close()
















