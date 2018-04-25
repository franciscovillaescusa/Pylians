import numpy as np
import sys,os,h5py
import groupcat
import MAS_library as MASL
import Pk_library as PKL
import units_library as UL

rho_crit = UL.units().rho_crit
################################# INPUT #####################################
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG_DM'

BoxSize = 75.0 #Mpc/h

dims = 1200
MAS  = 'CIC'
axis = 0
#############################################################################

for z,snapnum in zip([0,1,2,3,4,5],[99,50,33,25,21,17]):

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
                               fields=['GroupPos','GroupMass','Group_R_TopHat200'])
    halo_pos  = halos['GroupPos']/1e3          #Mpc/h
    halo_R    = halos['Group_R_TopHat200']/1e3 #Mpc/h
    halo_mass = halos['GroupMass']*1e10        #Msun/h
    del halos

    print np.min(halo_pos[:,0]),np.max(halo_pos[:,0])
    print np.min(halo_pos[:,1]),np.max(halo_pos[:,1])
    print np.min(halo_pos[:,2]),np.max(halo_pos[:,2])

    M_HI = M0*(halo_mass/Mmin)**alpha*np.exp(-(Mmin/halo_mass)**(0.35))

    Omega_HI = np.sum(M_HI, dtype=np.float64)/(BoxSize**3*rho_crit)
    print 'Omega_HI(z=%d) = %.3e'%(z,Omega_HI)
    #continue

    delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)

    MASL.MA(halo_pos, delta_HI, BoxSize, MAS, W=M_HI)
    delta_HI /= np.mean(delta_HI, dtype=np.float64);  delta_HI -= 1.0

    Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, 8)

    np.savetxt('Pk_HI_Nbody_real_space_z=%.1f.txt'%z, 
               np.transpose([Pk.k3D, Pk.Pk[:,0]]))
