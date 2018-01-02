import numpy as np
import mass_function_library as MFL

################################# INPUT ######################################
# input Pk at wanted MF redshift. For neutrinos use CDM+B Pk
f_Pk   = 'Pk_m_z=0.dat'
bins_k = 10000 #number of bins to use in the input Pk

# For neutrinos use Omega_{CDM+B} instead of Omega_m
Omega_m = 0.3175
M_min   = 1e10 #Msun/h
M_max   = 1e16 #Msun/h
bins_MF = 300  #number of bins in the HMF

author = 'ST'
f_out  = 'ST_MF_z=0.dat'

# optional arguments
z      = 0.0  # only for 'Tinker', 'Tinker10' and Crocce
delta  = 200  # only for 'Tinker' and 'Tinker10'
##############################################################################

# read input Pk
k, Pk = np.loadtxt(f_Pk, unpack=True)

# compute the masses at which compute the halo mass function
M = np.logspace(np.log10(M_min), np.log10(M_max), bins_MF)

# compute the MF
MF = MFL.MF_theory(k, Pk, Omega_m, M, author, bins_k, z, delta)

# save results to file
np.savetxt(f_out, np.transpose([M,MF]))






