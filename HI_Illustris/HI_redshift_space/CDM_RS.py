import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL
import HI_library as HIL
import redshift_space_library as RSL
import Pk_library as PKL

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
################################ INPUT #######################################
#snapnums = np.array([99, 50, 33, 25, 21, 17])
#snapnums = np.array([17,21,25])
#snapnums = np.array([33,25])
snapnums = np.array([99])

TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

dims = 1024
MAS  = 'CIC'
##############################################################################

# do a loop over the different redshifts
for snapnum in snapnums:

    for axis in [0,1,2]:

        # read header
        snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
        f            = h5py.File(snapshot+'.0.hdf5', 'r')
        scale_factor = f['Header'].attrs[u'Time']   
        redshift     = f['Header'].attrs[u'Redshift']
        BoxSize      = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
        filenum      = f['Header'].attrs[u'NumFilesPerSnapshot']
        Omega_m      = f['Header'].attrs[u'Omega0']
        Omega_L      = f['Header'].attrs[u'OmegaLambda']
        h            = f['Header'].attrs[u'HubbleParam']
        Masses       = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
        f.close()
        Hubble = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_L) #km/s/(Mpc/h)
        print 'Working with snapshot at z=%.0f and axis %d'%(redshift,axis)

        fout = 'CDM_field_RS_%d_z=%.1f.hdf5'%(axis,redshift)

        # define the array hosting delta_HI and delta_m
        delta_c = np.zeros((dims,dims,dims), dtype=np.float32)

        # do a loop over all subfiles in a given snapshot
        MHI_total, M_total, start = 0.0, 0.0, time.time()
        for i in xrange(118):

            snapshot = '%s/output/snapdir_%03d/snap_%03d.%d.hdf5'\
                %(run,snapnum,snapnum,i)
            f = h5py.File(snapshot, 'r')

            # read pos, radii, densities, HI/H and masses of gas particles 
            pos  = (f['PartType1/Coordinates'][:]/1e3).astype(np.float32)
            vel  = f['PartType1/Velocities'][:]*np.sqrt(scale_factor)

            # move gas particles to redshift-space
            RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)

            ######## delta_c ########
            MASL.MA(pos, delta_c, BoxSize, MAS)

            print 'i = %d'%i


        f = h5py.File(fout,'w')
        f.create_dataset('delta_c', data=delta_c)
        f.close()


        delta_c  /= np.mean(delta_c,  dtype=np.float64);  delta_c -= 1.0

        Pk = PKL.Pk(delta_c, BoxSize, axis, 'CIC', threads=8)

        np.savetxt('Pk_CDM_RS_%d_z=%.1f.txt'%(axis,redshift), 
                   np.transpose([Pk.k3D, Pk.Pk[:,0]]))

        np.savetxt('Pk2D_CDM_RS_%d_z=%.1f.txt'%(axis,redshift),
                   np.transpose([Pk.kpar, Pk.kper, Pk.Pk2D]))
