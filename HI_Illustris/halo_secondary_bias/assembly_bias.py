import numpy as np
import sys,os,h5py
import MAS_library as MASL
import Pk_library as PKL
import units_library as UL

rho_crit = UL.units().rho_crit
######################## INPUT ###########################
bins    = 1
dims    = 512
BoxSize = 75.0
MAS     = 'CIC'
axis    = 0
threads = 10
##########################################################

z_dict = {0:0, 1:0.997, 2:2.002, 3:3.008, 4:4.008, 5:4.996}

for z in [0,1,2,3,4,5]:
    for Mmin,Mmax in zip([1e8,1e9,1e10],[1e9,1e10,1e11]):

        # read positions, radii, HI and total masses of halos
        f1   = '../HI_mass/HI_FoF_galaxies/M_HI_new_75_1820_z=%.3f.hdf5'\
            %z_dict[z]
        f    = h5py.File(f1, 'r')
        M_HI = f['M_HI'][:];  M_HI = M_HI.astype(np.float32)
        Mass = f['Mass'][:]
        R    = f['R'][:]
        pos  = f['POS'][:]
        f.close()

        Omega_HI = np.sum(M_HI, dtype=np.float64)/(rho_crit*BoxSize**3)
        print 'Omega_HI(z=%.1f) = %.3e'%(z,Omega_HI)

        # consider only halos with R>0 and Masses between Mmin and Mmax
        indexes = np.where((R>0.0) & (Mass>Mmin) & (Mass<Mmax))[0]
        M_HI    = M_HI[indexes]
        Mass    = Mass[indexes]
        R       = R[indexes]
        pos     = pos[indexes]
        Omega_HI = np.sum(M_HI, dtype=np.float64)/(rho_crit*BoxSize**3)
        print 'Omega_HI(z=%.1f) = %.3e'%(z,Omega_HI)
        print '%d halos found'%len(pos)

        mass_bins = np.logspace(np.log10(Mmin), np.log10(np.max(Mass)), bins+1)

        pos1 = None
        pos2 = None
        for i in xrange(bins):
            indexes  = np.where((Mass>=mass_bins[i]) & (Mass<mass_bins[i+1]))[0]
            Mass_bin = Mass[indexes]
            M_HI_bin = M_HI[indexes]
            pos_bin  = pos[indexes]

            M_HI_mean = np.median(M_HI_bin)
            indexes1 = np.where(M_HI_bin>M_HI_mean)[0]
            indexes2 = np.where(M_HI_bin<=M_HI_mean)[0]

            #M_HI_mean = np.median(Mass_bin)
            #indexes1 = np.where(Mass_bin>M_HI_mean)[0]
            #indexes2 = np.where(Mass_bin<=M_HI_mean)[0]

            if pos1 is None:  pos1 = pos_bin[indexes1]
            else:             pos1 = np.vstack([pos1, pos_bin[indexes1]])
    
            if pos2 is None:  pos2 = pos_bin[indexes2]
            else:             pos2 = np.vstack([pos2, pos_bin[indexes2]])

        N0 = len(pos)
        N1 = len(pos1)
        N2 = len(pos2)

        # compute density field
        delta0 = np.zeros((dims,dims,dims), dtype=np.float32)
        delta1 = np.zeros((dims,dims,dims), dtype=np.float32)
        delta2 = np.zeros((dims,dims,dims), dtype=np.float32)

        MASL.MA(pos, delta0, BoxSize, MAS)
        delta0 /= np.mean(delta0, dtype=np.float64);  delta0 -= 1.0

        MASL.MA(pos1, delta1, BoxSize, MAS)
        delta1 /= np.mean(delta1, dtype=np.float64);  delta1 -= 1.0

        MASL.MA(pos2, delta2, BoxSize, MAS)
        delta2 /= np.mean(delta2, dtype=np.float64);  delta2 -= 1.0

        # compute power spectrum
        Pk = PKL.XPk([delta1,delta2,delta0], BoxSize, axis, 
                     [MAS,MAS,MAS], threads)

        # save data to file
        fout0 = 'Pk_halos_Mmin=%.1e_Mmax=%.1e_%d_z=%.1f.txt'%(Mmin,Mmax,bins,z)
        fout1 = 'Pk_halos_HI_rich_Mmin=%.1e_Mmax=%.1e_%d_z=%.1f.txt'%(Mmin,Mmax,bins,z)
        fout2 = 'Pk_halos_HI_poor_Mmin=%.1e_Mmax=%.1e_%d_z=%.1f.txt'%(Mmin,Mmax,bins,z)
        fout3 = 'Pk_halos_cross_Mmin=%.1e_Mmax=%.1e_%d_z=%.1f.txt'%(Mmin,Mmax,bins,z)
        fout4 = 'Pk_halos_check_Mmin=%.1e_Mmax=%.1e_%d_z=%.1f.txt'%(Mmin,Mmax,bins,z)
        np.savetxt(fout1, np.transpose([Pk.k3D, Pk.Pk[:,0,0]]))
        np.savetxt(fout2, np.transpose([Pk.k3D, Pk.Pk[:,0,1]]))
        np.savetxt(fout3, np.transpose([Pk.k3D, Pk.XPk[:,0,0]]))
        np.savetxt(fout0, np.transpose([Pk.k3D, Pk.Pk[:,0,2]]))

        Pk_tot = (N1*1.0/N0)**2*Pk.Pk[:,0,0] + (N2*1.0/N0)**2*Pk.Pk[:,0,1] +\
            2.0*N1*N2*1.0/N0**2*Pk.XPk[:,0,0]
        np.savetxt(fout4, np.transpose([Pk.k3D, Pk_tot]))


