import numpy as np
import sys,os,h5py


############################### INPUT #######################################
Mmin = 1e8
Mmax = 1e15
bins = 30
#############################################################################

M_bins    = np.logspace(np.log10(Mmin), np.log10(Mmax), bins+1)
M_mean    = 10**(0.5*(np.log10(M_bins[1:]) + np.log10(M_bins[:-1])))

for z in [4.996, 4.008, 3.008, 2.002, 0.997, 0.000]:

    f = h5py.File('M_HI_new_75_1820_z=%.3f.hdf5'%z, 'r')
    Mass      = f['Mass'][:]
    M_HI_halo = f['M_HI'][:]
    M_HI_gal  = f['M_HI_gal'][:]
    M_HI_cen  = f['M_HI_cen'][:]
    M_HI_sat  = f['M_HI_sat'][:]
    R         = f['R'][:]
    f.close()

    # select only halos with R>0
    indexes = np.where(R>0.0)[0]
    Mass      = Mass[indexes]
    M_HI_halo = M_HI_halo[indexes]
    M_HI_gal  = M_HI_gal[indexes]
    M_HI_cen  = M_HI_cen[indexes]
    M_HI_sat  = M_HI_sat[indexes]

    M_HI_halo_mean = np.zeros(bins, dtype=np.float64) 
    M_HI_halo_std  = np.zeros(bins, dtype=np.float64) 
    M_HI_gal_mean  = np.zeros(bins, dtype=np.float64) 
    M_HI_gal_std   = np.zeros(bins, dtype=np.float64) 
    M_HI_cen_mean  = np.zeros(bins, dtype=np.float64) 
    M_HI_cen_std   = np.zeros(bins, dtype=np.float64) 
    M_HI_sat_mean  = np.zeros(bins, dtype=np.float64) 
    M_HI_sat_std   = np.zeros(bins, dtype=np.float64) 

    for i in xrange(bins):
        indexes = np.where((Mass>=M_bins[i]) & (Mass<M_bins[i+1]))[0]
        if len(indexes)==0:  continue
        
        M_HI_halo_mean[i] = np.mean(M_HI_halo[indexes])
        M_HI_halo_std[i]  = np.std(M_HI_halo[indexes])

        M_HI_gal_mean[i] = np.mean(M_HI_gal[indexes])
        M_HI_gal_std[i]  = np.std(M_HI_gal[indexes])

        M_HI_cen_mean[i] = np.mean(M_HI_cen[indexes])
        M_HI_cen_std[i]  = np.std(M_HI_cen[indexes])

        M_HI_sat_mean[i] = np.mean(M_HI_sat[indexes])
        M_HI_sat_std[i]  = np.std(M_HI_sat[indexes])

    np.savetxt('M_HI_halo_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, M_HI_halo_mean, M_HI_halo_std]))
    np.savetxt('M_HI_gal_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, M_HI_gal_mean, M_HI_gal_std]))
    np.savetxt('M_HI_cen_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, M_HI_cen_mean, M_HI_cen_std]))
    np.savetxt('M_HI_sat_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, M_HI_sat_mean, M_HI_sat_std]))
