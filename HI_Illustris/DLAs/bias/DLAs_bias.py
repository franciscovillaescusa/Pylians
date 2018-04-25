import numpy as np
import sys,os,h5py
import mass_function_library as MFL
import bias_library as BL
import integration_library as IL

def sigma_DLAs_func(x, M0, N_HI=10**20.3):
	if M0<4 or M0>15:
		return 0

	A = -1.85-M0
	beta = 0.85*np.log10(N_HI*1.0)-16.35
	return 10**A*x**0.82*(1.0-np.exp(-(x/10**M0)**beta))

############################## INPUT ########################################
Omega_m = 0.3089
z       = 3

Mmin, Mmax = 1e8, 1e15
bins = 300
#############################################################################

# read input Pk
k,Pk = np.loadtxt('../../CAMB/illustris_matterpower_z=%.d.dat'%z, unpack=True)

# define halo masses
M = np.logspace(np.log10(Mmin), np.log10(Mmax), bins+1)


# read DLAs cross-section
f = h5py.File('../cross_section_new_z=%d.hdf5'%z,'r')
M_sigma = f['M'][:]
sigma   = f['sigma'][:]
N_HI    = f['N_HI'][:]
f.close()

# compute HMF and DLAs cross-section
bins2 = 40
M1         = np.logspace(np.log10(Mmin), np.log10(Mmax), bins2+1)
sigma_DLAs = np.zeros(bins2, dtype=np.float64)
MF         = np.zeros(bins2, dtype=np.float64)
for i in xrange(bins2):
    Ma,Mb = M1[i],M1[i+1]
    indexes = np.where((M_sigma>=Ma) & (M_sigma<Mb))[0]
    if len(indexes)==0:  continue
    sigma_DLAs[i] = np.mean(sigma[indexes,0])
    print 'M = %.3e ---> sigma = %.3e'%(0.5*(Ma+Mb),sigma_DLAs[i])
    MF[i] = len(indexes)/((Mb-Ma)*75.0**3)

M_mean = 0.5*(M1[1:] + M1[:-1])
np.savetxt('DLAs_sigma_mean_z=%.d.txt'%z, np.transpose([M_mean,sigma_DLAs]))
np.savetxt('HMF_mean_z=%.d.txt'%z,        np.transpose([M_mean,MF]))



# compute halo mass function
#MF = MFL.MF_theory(k, Pk, Omega_m, M, 'Crocce', 10000, z, delta=200.0)
#np.savetxt('MF_z=%d.txt'%z, np.transpose([M,MF]))

#M1,MF         = np.loadtxt('HMF_mean_z=%d.txt'%z, unpack=True)
#M1,sigma_DLAs = np.loadtxt('DLAs_sigma_mean_z=%d.txt'%z, unpack=True)

# interpolate HMF and DLAs cross-sections
MF         = np.interp(M,M_mean,MF,left=0,right=0)
sigma_DLAs = np.interp(M,M_mean,sigma_DLAs,left=0,right=0)

indexes = np.where((MF>0.0) & (sigma_DLAs>0.0))[0]
MF = MF[indexes]
sigma_DLAs = sigma_DLAs[indexes]
M = M[indexes]

Mmin = np.min(M)
Mmax = np.max(M)

# compute halo bias 'SMT01'
b = BL.bias(k, Pk, Omega_m, M, 'SMT01', 10000)
np.savetxt('bias_z=%d.txt'%z, np.transpose([M,b]))


#sigma_DLAs = np.zeros(bins+1, dtype=np.float64)
#if z==2:  M0 = 10.22955643 #value of 10^20
#if z==3:  M0 = 9.89018152  #value of 10^20
#for i in xrange(bins+1):
#    sigma_DLAs[i] = sigma_DLAs_func(M[i], M0, N_HI=10**20.0)


# integral value, its limits and precision parameters
eps   = 1e-14 
h1    = 1e-15 
hmin  = 0.0   

# integral method and integrand function
function = 'log'

yinit = np.zeros(1, dtype=np.float64) 
I1 = IL.odeint(yinit, Mmin, Mmax, eps, h1, hmin, np.log10(M), 
               np.log10(MF*sigma_DLAs*b), function, verbose=True)[0]

yinit = np.zeros(1, dtype=np.float64) 
I2 = IL.odeint(yinit, Mmin, Mmax, eps, h1, hmin, np.log10(M), 
               np.log10(MF*sigma_DLAs), function, verbose=True)[0]

print I1
print I2
print 'b_DLAs(z=%d) = %.3f'%(z,I1/I2)



