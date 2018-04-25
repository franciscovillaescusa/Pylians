from mpi4py import MPI
import numpy as np
import snapshot as sn
import readsnapHDF5 as rs
import units_library as UL
import HI_library as HIL
import sys,os,glob,h5py
import groupcat

####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
################################ INPUT ########################################
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'
fout = 'M_HI_new_75_1820.txt'

snaps = np.array([17, 21, 25, 33, 50, 99])
###############################################################################

# find offset_root and snapshot_root
snapshot_root = '%s/output/'%run

# find the numbers each cpu will work on
array   = np.arange(0, len(snaps))
numbers = np.where(array%nprocs==myrank)[0]
numbers = snaps[numbers]


# do a loop over the different realizations
for num in numbers:

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

    if myrank==0:
        print '\n'
        print 'BoxSize         = %.3f Mpc/h'%BoxSize
        print 'Number of files = %d'%filenum
        print 'Omega_m         = %.3f'%Omega_m
        print 'Omega_l         = %.3f'%Omega_L
        print 'redshift        = %.3f'%redshift

    if os.path.exists(fout[:-4]+'_z=%.3f.hdf5'%redshift):
        continue

    # read number of particles in halos and subhalos and number of subhalos
    halos = groupcat.loadHalos(snapshot_root, num, 
                               fields=['GroupLenType','GroupNsubs'])
    halo_len   = halos['GroupLenType'][:,0]  
    gal_in_halo = halos['GroupNsubs']
    subhalos = groupcat.loadSubhalos(snapshot_root, num, 
                               fields=['SubhaloLenType','SubhaloMassType'])
    gal_len    = subhalos['SubhaloLenType'][:,0]
    mass_ratio = subhalos['SubhaloMassType'][:]*1e10 #Msun/h
    total_mass = np.sum(mass_ratio, axis=1) #M_matter
    mass_ratio = mass_ratio[:,1]/total_mass #M_CDM / M_matter
    del halos, subhalos

    # define the M_HI array
    M_HI     = np.zeros(len(halo_len), dtype=np.float64)
    M_HI_gal = np.zeros(len(halo_len), dtype=np.float64)
    M_HI_cen = np.zeros(len(halo_len), dtype=np.float64)
    M_HI_sat = np.zeros(len(halo_len), dtype=np.float64)

    # do a loop over all files
    Omega_HI, Omega_g, done = 0.0, 0.0, False
    #start, end, end_gal = 0, end_halos[0], end_all_galaxies[0]
    #pars = [Number, start_h, end_h, start_g, end_g, halo_num, gal_num,
    #        gal_in_local_halo]
    pars = np.array([0, 0, halo_len[0], 0, gal_len[0], 0, 0, 0],
                    dtype=np.int64)
    for i in xrange(filenum):

        snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(num,num,i)
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
            done = HIL.M_HI_halos_gal_cen_sat(pars, done, halo_len, gal_len,
                                              gal_in_halo, mass_ratio, MHI,
                                              M_HI, M_HI_gal, M_HI_cen, 
                                              M_HI_sat)
                             
        Omega_HI += np.sum(MHI,  dtype=np.float64)
        Omega_g  += np.sum(mass, dtype=np.float64)

        print '\ni = %03d: z=%.3f ----> %02d'%(i,redshift,myrank)
        print 'Omega_g        = %.6e'%(Omega_g/(BoxSize**3*rho_crit))
        print 'Omega_HI       = %.6e'%(Omega_HI/(BoxSize**3*rho_crit))
        print 'Omega_HI_halos = %.6e'%(np.sum(M_HI)/(BoxSize**3*rho_crit))
        print 'Omega_HI_gal   = %.6e'%(np.sum(M_HI_gal)/(BoxSize**3*rho_crit))
        print 'Omega_HI_cen   = %.6e'%(np.sum(M_HI_cen)/(BoxSize**3*rho_crit))
        print 'Omega_HI_sat   = %.6e'%(np.sum(M_HI_sat)/(BoxSize**3*rho_crit))
            
        if done: break                                

    Omega_HI       = Omega_HI/(BoxSize**3*rho_crit)
    Omega_g        = Omega_g/(BoxSize**3*rho_crit)
    Omega_HI_halos = np.sum(M_HI)/(BoxSize**3*rho_crit)
    Omega_HI_gal   = np.sum(M_HI_gal)/(BoxSize**3*rho_crit)

    # read masses and positions of the halos
    fields  = ['GroupCM','GroupMass','GroupVel','GroupMassType',
               'Group_R_TopHat200']
    halos   = groupcat.loadHalos(snapshot_root,num,fields=fields)
    mass_h  = halos['GroupMass']*1e10          #Msun/h
    mass_hc = halos['GroupMassType'][:,1]*1e10 #Msun/h
    pos_h   = halos['GroupCM']/1e3             #Mpc/h
    vel_h   = halos['GroupVel']*(1.0+redshift) #km/s
    R_h     = halos['Group_R_TopHat200']/1e3   #Mpc/h
    del halos

    # save results to file
    f = h5py.File(fout[:-4]+'_z=%.3f.hdf5'%redshift, 'w')
    f.create_dataset('Mass',              data=mass_h)
    f.create_dataset('M_HI',              data=M_HI)
    f.create_dataset('M_HI_gal',          data=M_HI_gal)
    f.create_dataset('M_HI_cen',          data=M_HI_cen)
    f.create_dataset('M_HI_sat',          data=M_HI_sat)
    f.create_dataset('POS',               data=pos_h)
    f.create_dataset('R',                 data=R_h)
    f.close()



