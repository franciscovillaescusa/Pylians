# This routine computes the HI power spectrum when the HI inside halos is placed
# in the halo center. This is done to avoid the presence of the 1-halo term
import numpy as np
import sys,os,h5py
import MAS_library as MASL
import Pk_library as PKL

############################ INPUT #############################
dims    = 1200
BoxSize = 75.0  #Mpc/h
MAS     = 'CIC'
axis    = 0
threads = 10
################################################################

z_dict = {0:0, 1:0.997, 2:2.002, 3:3.008, 4:4.008, 5:4.996}

for z in [0, 1, 2, 3, 4, 5]:

    # read positions, radii, HI and total masses of halos
    f1   = '../HI_mass/HI_FoF_galaxies/M_HI_new_75_1820_z=%.3f.hdf5'%z_dict[z]
    f    = h5py.File(f1, 'r')
    M_HI = f['M_HI'][:]
    Mass = f['Mass'][:]
    R    = f['R'][:]
    pos  = f['POS'][:]
    f.close()

    Omega_HI = np.sum(M_HI, dtype=np.float64)/(2.775e11*75**3)
    print '\nOmega_HI(z=%.1f) = %.3e'%(z,Omega_HI)

    """
    # consider only halos with R>0
    indexes = np.where(R>0.0)[0]
    M_HI    = M_HI[indexes]
    Mass    = Mass[indexes]
    R       = R[indexes]
    pos     = pos[indexes]
    Omega_HI = np.sum(M_HI, dtype=np.float64)/(2.775e11*75**3)
    print 'Omega_HI = %.3e'%Omega_HI
    """

    M_HI = M_HI.astype(np.float32)

    # compute density field
    delta = np.zeros((dims,dims,dims), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, W=M_HI)
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)

    # save data to file
    fout = 'Pk_HI_center_halos_z=%.1f.txt'%z
    np.savetxt(fout, np.transpose([Pk.k3D, Pk.Pk[:,0]]))


