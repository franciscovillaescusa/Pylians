import numpy as np
import sys,os,h5py
import groupcat


############################### INPUT ######################################
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

bins = 35
############################################################################

snapshot_root = '%s/output/'%run

for snapnum,z in zip([17,21,25,33,50,99],[4.996,4.008,3.008,2.002,0.997,0.000]):

    # read number of particles in halos and subhalos and number of subhalos
    halos = groupcat.loadHalos(snapshot_root, snapnum, 
                               fields=['GroupLenType','GroupMass'])
    halo_len = halos['GroupLenType'][:,1]  
    halo_mass = halos['GroupMass']*1e10 #Msun/h
    del halos

    f_HI = 'sigma_HI_75_1820_z=%.3f.hdf5'%z
    f_m  = 'sigma_matter_75_1820_z=%.3f.hdf5'%z

    f = h5py.File(f_HI, 'r')
    M_HI = f['M_HI'][:]
    sigma2_HI = f['sigma2_HI'][:]
    f.close()

    # only consider halos with HI masses above 1e5
    indexes   = np.where(M_HI>1e5)[0]
    M_HI      = M_HI[indexes]
    sigma2_HI = sigma2_HI[indexes]
    halo_mass = halo_mass[indexes]
    sigma_HI  = np.sqrt(sigma2_HI/M_HI)

    f = h5py.File(f_m, 'r')
    M_0 = f['M_0'][:]
    M_1 = f['M_1'][:]
    M_4 = f['M_4'][:]
    M_5 = f['M_5'][:]
    sigma2_0 = f['sigma2_0'][:]
    sigma2_1 = f['sigma2_1'][:]
    sigma2_4 = f['sigma2_4'][:]
    sigma2_5 = f['sigma2_5'][:]
    f.close()

    #indexes = np.where(M_CDM==0.0)[0]
    sigma_m = np.sqrt((sigma2_0+sigma2_1+sigma2_4+sigma2_5)/(M_0+M_1+M_4+M_5))

    M_bins        = np.logspace(8,14,bins+1)
    M_mean        = 0.5*(M_bins[1:] + M_bins[:-1])
    sigma_HI_mean = np.zeros(bins) 
    sigma_HI_std  = np.zeros(bins) 
    sigma_m_mean  = np.zeros(bins) 
    sigma_m_std   = np.zeros(bins) 
    
    for i in xrange(bins):
        indexes = np.where((halo_mass>=M_bins[i]) & (halo_mass<M_bins[i+1]))[0]
        if len(indexes)==0:  continue

        sigma1 = sigma_HI[indexes]
        sigma2 = sigma_m[indexes]

        sigma_HI_mean[i] = np.mean(sigma1)
        sigma_HI_std[i]  = np.std(sigma1)
        sigma_m_mean[i]  = np.mean(sigma2)
        sigma_m_std[i]   = np.std(sigma2)


    np.savetxt('HI_sigma_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, sigma_HI_mean, sigma_HI_std]))
    np.savetxt('matter_sigma_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, sigma_m_mean, sigma_m_std]))

