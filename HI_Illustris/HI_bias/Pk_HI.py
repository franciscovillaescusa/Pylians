import numpy as np
import sys,os,h5py
import Pk_library as PKL
import units_library as UL

rho_crit = UL.units().rho_crit
################################# INPUT ######################################
redshifts = [0, 1, 2, 3, 4, 5]

BoxSize = 75.0 #Mpc/h
##############################################################################

for z in redshifts:
    f = h5py.File('fields_z=%.1f.hdf5'%z, 'r')
    delta_HI = f['delta_HI'][:]
    delta_m  = f['delta_m'][:]
    f.close()

    Omega_HI = np.sum(delta_HI, dtype=np.float64)/(BoxSize**3*rho_crit)
    Omega_m  = np.sum(delta_m,  dtype=np.float64)/(BoxSize**3*rho_crit)
    print 'z=%.1f ------> Omega_HI = %.5f ---> Omega_m = %.4f'\
        %(z,Omega_HI,Omega_m)

    delta_HI /= np.mean(delta_HI, dtype=np.float64);  delta_HI -= 1.0
    delta_m  /= np.mean(delta_m,  dtype=np.float64);  delta_m -= 1.0

    Pk = PKL.XPk([delta_HI, delta_m], BoxSize, axis=0, MAS=['CIC','CIC'],
                 threads=8)

    np.savetxt('Pk_HI_z=%.1f.txt'%z,
               np.transpose([Pk.k3D, Pk.Pk[:,0,0]]))

    np.savetxt('Pk_m_z=%.1f.txt'%z,
               np.transpose([Pk.k3D, Pk.Pk[:,0,1]]))

    np.savetxt('Pk_HI-m_z=%.1f.txt'%z,
               np.transpose([Pk.k3D, Pk.XPk[:,0,0]]))

