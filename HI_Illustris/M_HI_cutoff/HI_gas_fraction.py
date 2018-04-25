import numpy as np
import sys,os,h5py
import groupcat
import readgadget


root = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/'
################################# INPUT ########################################
Mmin = 3.7e8 #3e9 #3.7e8
Mmax = 1e14
bins = 60

fout = 'ratio_HI_gas_75Mpc_1820'
#fout = 'ratio_g_75Mpc_1820_UVB'
################################################################################

z_dict = {99:0.000, 50: 0.997, 33:2.002, 25:3.008, 21:4.008, 17:4.996}

for snapnum in [17,21,25,33,50,99]:

    # read header
    snapshot = root + 'snapdir_%03d/snap_%03d'%(snapnum,snapnum)
    header = readgadget.header(snapshot)
    BoxSize = header.boxsize/1e3 #Mpc/h
    redshift = header.redshift

    print 'L = %.1f Mpc/h'%BoxSize
    print 'z = %.1f'%redshift

    halos = groupcat.loadHalos(root, snapnum,
                               fields=['GroupMassType','GroupMass',
                                       'Group_R_TopHat200'])
    halo_mass = halos['GroupMassType'][:]*1e10    #Msun/h
    Mass      = halos['GroupMass'][:]*1e10        #Msun/h
    R         = halos['Group_R_TopHat200'][:]/1e3 #Mpc/h

    f = h5py.File('../HI_mass/HI_FoF_galaxies/M_HI_new_75_1820_z=%.3f.hdf5'%z_dict[snapnum], 'r')
    M_HI = f['M_HI'][:]
    f.close()

    indexes   = np.where((R>0.0) & (halo_mass[:,0]>0.0))[0]
    halo_mass = halo_mass[indexes]
    Mass      = Mass[indexes]
    M_HI      = M_HI[indexes]


    ratio = M_HI/halo_mass[:,0]

    M_bins = np.logspace(np.log10(Mmin), np.log10(Mmax), bins+1)
    M_mean = 10**(0.5*(np.log10(M_bins[1:]) + np.log10(M_bins[:-1])))

    ratio_mean = np.zeros(bins)
    ratio_std  = np.zeros(bins)

    for i in xrange(bins):
        indexes = np.where((Mass>=M_bins[i]) & (Mass<M_bins[i+1]))[0]
        if len(indexes)==0:  continue
        ratio_mean[i] = np.mean(ratio[indexes])
        ratio_std[i]  = np.std(ratio[indexes])
    
    np.savetxt('%s_z=%.1f.txt'%(fout,redshift),
               np.transpose([M_mean, ratio_mean, ratio_std]))

