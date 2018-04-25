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
fout = 'sigma_HI_75_1820.txt'

snaps = np.array([17, 21, 25, 33, 50, 99])
###############################################################################


snapshot_root = '%s/output/'%run

# do a loop over the different realizations
for num in snaps:

    # find snapshot name and offset file name
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

    # read number of particles in halos and subhalos and number of subhalos
    halos = groupcat.loadHalos(snapshot_root, num, 
                               fields=['GroupLenType','GroupVel'])
    halo_len = halos['GroupLenType'][:,0]  
    halo_vel = halos['GroupVel']*(1.0+redshift) #km/s 
    del halos

    # read HI bulk velocity of halos
    f = h5py.File('../HI_bulk_velocity/V_HI_75_1820_z=%d.hdf5'%round(redshift),
                  'r')
    V_HI = f['V_HI'][:] #km/s
    f.close()
    V_HI = V_HI.astype(np.float32)


    # define the M_HI and sigma_HI array
    M_HI      = np.zeros(len(halo_len), dtype=np.float64)
    sigma2_HI = np.zeros(len(halo_len), dtype=np.float64)


    # do a loop over all files
    Omega_HI, done = 0.0, False
    #start, end, end_gal = 0, end_halos[0], end_all_galaxies[0]
    #pars = [Number, start_h, end_h, start_g, end_g, halo_num, gal_num,
    #        gal_in_local_halo]
    pars = np.array([0, 0, halo_len[0], 0], dtype=np.int64)
    for i in xrange(filenum):

        snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(num,num,i)
        
        vel = rs.read_block(snapshot, 'VEL ', parttype=0, verbose=False)/np.sqrt(1.0+redshift) #km/s                         

        MHI   = rs.read_block(snapshot, 'NH  ', parttype=0, verbose=False)#HI/H
        mass  = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10
        SFR   = rs.read_block(snapshot, 'SFR ', parttype=0, verbose=False)
        indexes = np.where(SFR>0.0)[0];  del SFR

        # find the metallicity of star-forming particles
        metals = rs.read_block(snapshot, 'GZ  ', parttype=0, verbose=False)
        metals = metals[indexes]/0.0127

        # find densities of star-forming particles: units of h^2 Msun/Mpc^3
        rho = rs.read_block(snapshot, 'RHO ', parttype=0, verbose=False)*1e19
        rho = rho[indexes]

        # find volume and radius of star-forming particles
        Volume = mass[indexes]/rho                   #(Mpc/h)^3
        radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 
        

        # find HI/H fraction for star-forming particles
        MHI[indexes] = HIL.Rahmati_HI_Illustris(rho, radii, metals, redshift, 
                                                h, TREECOOL_file, Gamma=None,
                                                fac=1, correct_H2=True) #HI/H
        MHI *= (0.76*mass)

        # compute the HI mass in halos and galaxies
        if not(done):
            done = HIL.sigma_HI_halos(pars, done, halo_len, V_HI, #halo_vel,
                                      MHI, vel, sigma2_HI, M_HI)
                             
        Omega_HI += np.sum(MHI,  dtype=np.float64)

        print '\ni = %03d: z=%.3f'%(i,redshift)
        print 'Omega_HI = %.6e'%(Omega_HI/(BoxSize**3*rho_crit))
            
        if done: break                                

    Omega_HI = Omega_HI/(BoxSize**3*rho_crit)

    # save results to file
    f = h5py.File(fout[:-4]+'_z=%.3f.hdf5'%redshift, 'w')
    f.create_dataset('M_HI',      data=M_HI)
    f.create_dataset('sigma2_HI', data=sigma2_HI)
    f.close()

