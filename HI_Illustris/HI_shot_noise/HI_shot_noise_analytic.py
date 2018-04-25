# This scripts computes the value of Omega_HI and HI shot-noise
# assuming a M_HI(M) = M_0*(M/Mmin)^alpha*exp(-Mmin/M)
# P_SN = 1/(rho_c^0*Omega_HI)^2*\int_0^oo n(M)*M_HI(M)^2dM
# Omega_HI = 1/rho_c^0*\int_0^oo n(M)*M_HI(M) dM
import numpy as np
import sys,os
import mass_function_library as MFL
import units_library as UL
import integration_library as IL

rho_crit = UL.units().rho_crit
################################# INPUT ######################################
# input Pk at wanted MF redshift. For neutrinos use CDM+B Pk
f_Pk   = '../../CAMB/illustris_matterpower_z=0.dat'
bins_k = 10000 #number of bins to use in the input Pk

# For neutrinos use Omega_{CDM+B} instead of Omega_m
Omega_m = 0.3089
M_min   = 1e8  #Msun/h
M_max   = 1e16 #Msun/h
bins_MF = 800  #number of bins in the HMF

author = 'ST'
f_out  = 'ST_MF_z=0.dat'

# optional arguments
z     = 0.0  # only for 'Tinker', 'Tinker10' and Crocce
delta = 200  # only for 'Tinker' and 'Tinker10'

# M_HI parameters (M0 only needed for Omega_HI, not P_SN)
alpha, M0, Mmin = 0.63, 4.7e8, 2.2e10  #z=0
#alpha, M0, Mmin = 0.88, 7.7e7, 7.9e9  #z=1
#alpha, M0, Mmin = 0.98, 2.7e7, 3.0e9  #z=2
#alpha, M0, Mmin = 1.04, 9.4e6, 9.5e8  #z=3
#alpha, M0, Mmin = 0.99, 1.2e7, 5.3e8  #z=4
#alpha, M0, Mmin = 0.99, 1.1e7, 3.4e8  #z=5


# integration parameterrs
eps  = 1e-15
h1   = 1e0
hmin = 0.0
##############################################################################



if os.path.exists(f_out):
    M, MF = np.loadtxt(f_out, unpack=True)
else:
    # read input Pk
    k, Pk = np.loadtxt(f_Pk, unpack=True)
    
    # compute the masses at which compute the halo mass function
    M = np.logspace(np.log10(M_min), np.log10(M_max), bins_MF)

    # compute the MF
    MF = MFL.MF_theory(k, Pk, Omega_m, M, author, bins_k, z, delta)

    # save results to file
    np.savetxt(f_out, np.transpose([M,MF]))

M  = np.ascontiguousarray(M)
MF = np.ascontiguousarray(MF)

M_HI = M0*(M/Mmin)**alpha*np.exp(-Mmin/M)

A = MF*M_HI**2
B = MF*M_HI

yinit = np.array([0.0], dtype=np.float64)
numerator = IL.odeint(yinit, 1e8, M_max, eps, h1, hmin, 
                      np.log10(M), np.log10(A), 
                      'sigma', verbose=True)[0]

yinit = np.array([0.0], dtype=np.float64)
denominator = IL.odeint(yinit, 1e8, M_max, eps, h1, hmin, 
                        np.log10(M), np.log10(B), 
                        'sigma', verbose=True)[0]

print 'numerator       = %.5e'%numerator
print 'denominator     = %.5e'%denominator
print 'Omega_HI(z=%.1f) = %.3e'%(z,denominator/rho_crit)
print 'P_HI(z=%.1f)     = %.5f'%(z,numerator/denominator**2)
