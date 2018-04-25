# This script reads the HI masses and velocities of all gas particles inside 
# halos and save a file with that.
import numpy as np
import sys,os,h5py
import groupcat
import HI_library as HIL

################################ INPUT ######################################
snapshot_root = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/'
TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
#############################################################################


for snapnum in [99, 50, 33, 25, 21, 17]:

    # read number of particles in halos and subhalos and number of subhalos
    print '\nReading halo catalogue...'
    halos = groupcat.loadHalos(snapshot_root, snapnum, 
                               fields=['GroupPos','GroupMass','GroupLenType'])
    halo_pos  = halos['GroupPos']/1e3   #Mpc/h
    halo_mass = halos['GroupMass']*1e10 #Msun/h
    halo_len  = halos['GroupLenType'][:,0]
    del halos

    Nparticles = np.sum(halo_len)
    print 'Number of halos = %d'%len(halo_pos)
    print 'Number of particles in all halos = %d'%Nparticles

    # read snapshot header
    snapshot = '%s/snapdir_%03d/snap_%03d.0.hdf5'\
        %(snapshot_root, snapnum, snapnum)
    f        = h5py.File(snapshot, 'r')
    redshift = f['Header'].attrs[u'Redshift']
    filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
    f.close()

    fout = 'HI_mass_vel_z=%d.hdf5'%(round(redshift))
    if os.path.exists(fout):  continue

    # define arrays containing the HI masses and velocities of gas particles
    M_HI_tot = np.zeros(Nparticles,      dtype=np.float32)
    vel_tot  = np.zeros((Nparticles,3),  dtype=np.float32)

    # do a loop over all subfiles
    offset = 0
    for i in xrange(filenum):

        print 'z=%d ---> %03d'%(round(redshift),i)

        # get snapshot subfile name
        snapshot = '%s/snapdir_%03d/snap_%03d.%d.hdf5'\
            %(snapshot_root, snapnum, snapnum, i)

        # read positions and HI masses from subfile
        pos, M_HI = HIL.HI_mass_from_Illustris_snap(snapshot, TREECOOL_file)

        # read gas velocities 
        f   = h5py.File(snapshot, 'r')
        vel = f['PartType0/Velocities'][:]/np.sqrt(1.0+redshift) #km/s
        f.close()

        end = offset+len(M_HI)

        if end<Nparticles:
            M_HI_tot[offset:end] = M_HI
            vel_tot[offset:end]  = vel
        else:
            M_HI_tot[offset:] = M_HI[:Nparticles-offset]
            vel_tot[offset:]  = vel[:Nparticles-offset]
            break

        offset = end

    # save results to file
    f = h5py.File(fout,'w')
    f.create_dataset('M_HI', data=M_HI_tot)
    f.create_dataset('vel',  data=vel_tot)
    f.close()
