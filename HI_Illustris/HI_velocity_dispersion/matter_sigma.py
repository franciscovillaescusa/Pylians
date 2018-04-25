import numpy as np
import snapshot as sn
import readsnapHDF5 as rs
import units_library as UL
import HI_library as HIL
import sys,os,glob,h5py
import groupcat

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
################################ INPUT ########################################
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'
fout = 'sigma_matter_75_1820.txt'

snaps = np.array([17, 21, 25, 33, 50, 99])
###############################################################################


snapshot_root = '%s/output/'%run

# do a loop over the different realizations
for num in snaps:

    # find snapshot name
    snapshot = snapshot_root + 'snapdir_%03d/snap_%03d'%(num,num)

    # read header
    header   = rs.snapshot_header(snapshot)
    nall     = header.nall
    redshift = header.redshift
    BoxSize  = header.boxsize/1e3 #Mpc/h
    filenum  = header.filenum
    Omega_m  = header.omega0
    Omega_L  = header.omegaL
    h        = header.hubble

    print '\n'
    print 'BoxSize         = %.3f Mpc/h'%BoxSize
    print 'Number of files = %d'%filenum
    print 'Omega_m         = %.3f'%Omega_m
    print 'Omega_l         = %.3f'%Omega_L
    print 'redshift        = %.3f'%redshift

    if os.path.exists(fout[:-4]+'_z=%.3f.hdf5'%redshift):  continue
    print 'ciao'

    f = h5py.File(fout[:-4]+'_z=%.3f.hdf5'%redshift, 'w')
    for ptype in [0,1,4,5]:

        # read number of particles in halos and subhalos and number of subhalos
        halos = groupcat.loadHalos(snapshot_root, num, 
                                   fields=['GroupLenType','GroupVel'])
        halo_len = halos['GroupLenType'][:,ptype]  
        halo_vel = halos['GroupVel']*(1.0+redshift) #km/s 
        del halos

        # define the M_HI and sigma_HI array
        M_ptype      = np.zeros(len(halo_len), dtype=np.float64)
        sigma2_ptype = np.zeros(len(halo_len), dtype=np.float64)


        # do a loop over all files
        Omega_ptype, done = 0.0, False
        #start, end, end_gal = 0, end_halos[0], end_all_galaxies[0]
        #pars = [Number, start_h, end_h, start_g, end_g, halo_num, gal_num,
        #        gal_in_local_halo]
        pars = np.array([0, 0, halo_len[0], 0], dtype=np.int64)
        for i in xrange(filenum):

            snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(num,num,i)
        
            vel = rs.read_block(snapshot, 'VEL ', parttype=ptype, verbose=False)/np.sqrt(1.0+redshift) #km/s                         

            mass  = rs.read_block(snapshot, 'MASS', parttype=ptype, verbose=False)*1e10
            mass = mass.astype(np.float32)

            # compute the HI mass in halos and galaxies
            if not(done):
                done = HIL.sigma_HI_halos(pars, done, halo_len, halo_vel,
                                          mass, vel, sigma2_ptype, M_ptype)
                             
            Omega_ptype += np.sum(mass,  dtype=np.float64)

            print '\ni = %03d: z=%.3f'%(i,redshift)
            print 'Omega(%d,z=%d) = %.6e'%(ptype,round(redshift),Omega_ptype/(BoxSize**3*rho_crit))
            
            if done: break                                

        # save results to file
        f.create_dataset('M_%d'%ptype,      data=M_ptype)
        f.create_dataset('sigma2_%d'%ptype, data=sigma2_ptype)
    f.close()

