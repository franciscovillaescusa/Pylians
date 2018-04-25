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

    f = h5py.File('V_HI_75_1820_z=%.3f.hdf5'%z,'r')
    V_HI = f['V_HI'][:]
    V    = f['V'][:]
    Mass = f['M'][:]
    M_HI = f['M_HI'][:]
    f.close()

    # only consider halos with HI masses above 1e5
    indexes = np.where(M_HI>1e5)[0]
    V_HI = V_HI[indexes]
    V    = V[indexes]
    Mass = Mass[indexes]
    M_HI = M_HI[indexes]

    
    V_HI2 = V_HI[:,0]**2 + V_HI[:,1]**2 + V_HI[:,2]**2
    V2    = V[:,0]**2 + V[:,1]**2 + V[:,2]**2

    cos_a = V_HI[:,0]*V[:,0] + V_HI[:,1]*V[:,1] + V_HI[:,2]*V[:,2]
    cos_a = cos_a/(np.sqrt(V_HI2)*np.sqrt(V2))

    V_HI  = np.sqrt(V_HI2)
    V     = np.sqrt(V2)
    ratio = V_HI/V

    cos_a_mean = np.zeros(bins, dtype=np.float64) 
    cos_a_std  = np.zeros(bins, dtype=np.float64) 

    ratio_mean = np.zeros(bins, dtype=np.float64) 
    ratio_std  = np.zeros(bins, dtype=np.float64) 
    for i in xrange(bins):
        indexes = np.where((Mass>=M_bins[i]) & (Mass<M_bins[i+1]))[0]
        if len(indexes)==0:  continue

        cos_a_mean[i] = np.mean(cos_a[indexes])
        cos_a_std[i]  = np.std(cos_a[indexes])

        ratio_mean[i] = np.mean(ratio[indexes])
        ratio_std[i]  = np.std(ratio[indexes])

    np.savetxt('cos_a_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, cos_a_mean, cos_a_std]))
    np.savetxt('ratio_mean_std_z=%d.txt'%round(z), 
               np.transpose([M_mean, ratio_mean, ratio_std]))





