# we use this script to compare the total mass in FoF and FoF-SO halos
# and their differences in HI masses
import numpy as np
import sys,os,h5py
import groupcat

###################### INPUT ############################
f1 = '../HI_SO/M_HI_SO_TopHat200_z=5.0.hdf5'
f2 = '../HI_FoF_galaxies/M_HI_new_75_1820_z=4.996.hdf5'
#########################################################

# read FoF and FoF-SO masses and FoF-SO HI masses
f = h5py.File(f1, 'r')
M_HI_SO  = f['M_HI_SO'][:]
mass_FoF = f['mass_FoF'][:]
mass_SO  = f['mass_SO'][:]
R        = f['R'][:]
f.close()

indexes  = np.argsort(mass_FoF)[::-1]
M_HI_SO  = M_HI_SO[indexes]
mass_FoF = mass_FoF[indexes]
mass_SO  = mass_SO[indexes]

f = h5py.File(f2, 'r')
M_HI_FoF  = f['M_HI'][:]
mass_FoF2 = f['Mass'][:]
R         = f['R'][:]
f.close()

indexes   = np.where(R>0)[0]
M_HI_FoF  = M_HI_FoF[indexes]
mass_FoF2 = mass_FoF2[indexes]
indexes = np.argsort(mass_FoF2)[::-1]
mass_FoF2 = mass_FoF2[indexes]
M_HI_FoF = M_HI_FoF[indexes]

ratio = mass_FoF2/mass_FoF
print np.min(ratio), np.max(ratio)

num = -1
Omega_HI1 = np.sum(M_HI_FoF[:num], dtype=np.float64)/(2.775e11*75**3)
Omega_HI2 = np.sum(M_HI_SO[:num],  dtype=np.float64)/(2.775e11*75**3)
Omega_m1  = np.sum(mass_FoF[:num], dtype=np.float64)/(2.775e11*75**3)
Omega_m2  = np.sum(mass_SO[:num],  dtype=np.float64)/(2.775e11*75**3)
print 'Omega_HI1 = %.5e'%Omega_HI1
print 'Omega_HI2 = %.5e'%Omega_HI2
print 'Omega_m1  = %.5e'%Omega_m1
print 'Omega_m2  = %.5e'%Omega_m2
print 'ratio_HI = %.3f'%(Omega_HI1/Omega_HI2)
print 'ratio_m  = %.3f'%(Omega_m1/Omega_m2)

for i in xrange(10):
    print '%.3e ---> %.3e  %.3e'%(mass_FoF[i],M_HI_FoF[i],M_HI_SO[i])








