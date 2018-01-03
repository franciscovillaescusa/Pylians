import numpy as np
import bias_library as BL
import sys,os,time

################################### INPUT ########################################
# input Pk at wanted MF redshift. For neutrinos use CDM+B Pk
f_in    = 'Pk_m_z=0.dat'
bins_k  = 10000  #number of bins to use in the input Pk

# For neutrinos use Omega_{CDM+B} instead of Omega_m
Omega_m = 0.3175
Mmin    = 1e10 #Msun/h
Mmax    = 1e16 #Msun/h
bins_MF = 300  #number of bins in the HMF

author = 'SMT01'  #'SMT01', 'Tinker'
f_out  = 'bias_SMT_z=0.txt'
##################################################################################

# read input Pk
k, Pk = np.loadtxt(f_in, unpack=True)

# define the array with the masses where to evaluate the halo bias
Masses = np.logspace(np.log10(Mmin), np.log10(Mmax), bins_MF)

# compute halo bias
b = BL.bias(k, Pk, Omega_m, Masses, author, bins_k)

# save results to file    
np.savetxt(f_out, np.transpose([Masses, b]))



# This computes the effective bias for halos in the range
# [min(Masses), max(Masses)]
z = 0.0
b_eff =  BL.bias_eff(k, Pk, Omega_m, Masses, z, author)
