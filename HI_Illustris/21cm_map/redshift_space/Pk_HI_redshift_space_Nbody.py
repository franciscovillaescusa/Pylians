import numpy as np
import sys,os,h5py
import groupcat
import MAS_library as MASL
import Pk_library as PKL
import redshift_space_library as RSL

################################# INPUT #####################################
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG_DM'

BoxSize = 75.0 #Mpc/h

dims = 1200
MAS  = 'CIC'

Omega_m = 0.3089
Omega_l = 1.0-Omega_m
#############################################################################

for z,snapnum in zip([0,1,2,3,4,5],[99,50,33,25,21,17]):

    Hubble = 100.0*np.sqrt(Omega_m*(1.0+z)**3+Omega_l) #km/s/(Mpc/h)

    for axis in [0,1,2]:

        if z==0:
            alpha, M0, Mmin = 0.24, 4.3e10, 2e12 
        elif z==1:
            alpha, M0, Mmin = 0.53, 1.5e10, 6e11
        elif z==2:
            alpha, M0, Mmin = 0.60, 1.3e10, 3.6e11
        elif z==3:
            alpha, M0, Mmin = 0.76, 2.9e9, 6.7e10
        elif z==4:
            alpha, M0, Mmin = 0.79, 1.4e9, 2.1e10
        elif z==5:
            alpha, M0, Mmin = 0.74, 1.9e9, 2e10
        else: raise Exception('wrong redshift')

        print '\nReading halo catalogue...'
        snapshot_root = '%s/output/'%run
        halos = groupcat.loadHalos(snapshot_root, snapnum, 
                             fields=['GroupPos','GroupMass','GroupVel'])
        halo_pos  = halos['GroupPos']/1e3          #Mpc/h
        halo_mass = halos['GroupMass']*1e10        #Msun/h
        halo_vel  = halos['GroupVel']*(1.0+z)      #km/s
        del halos

        # move halo positions to redshift-space
        RSL.pos_redshift_space(halo_pos, halo_vel, BoxSize, Hubble, z, axis)

        print np.min(halo_pos[:,0]),np.max(halo_pos[:,0])
        print np.min(halo_pos[:,1]),np.max(halo_pos[:,1])
        print np.min(halo_pos[:,2]),np.max(halo_pos[:,2])
        
        M_HI = M0*(halo_mass/Mmin)**alpha*np.exp(-(Mmin/halo_mass)**(0.35))

        delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)

        MASL.MA(halo_pos, delta_HI, BoxSize, MAS, W=M_HI)
        delta_HI /= np.mean(delta_HI, dtype=np.float64);  delta_HI -= 1.0

        Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, 8)

        np.savetxt('Pk_HI_Nbody_redshift_space_%d_z=%.1f.txt'%(axis,z), 
                   np.transpose([Pk.k3D, Pk.Pk[:,0]]))
