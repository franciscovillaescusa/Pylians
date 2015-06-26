import numpy as np
import scipy.integrate as si
import scipy.special as se
import sys,os




def int_sigma(y,x,k,Pk):
    return [np.interp(np.log(x),np.log(k),Pk)]

################################## INPUT #####################################
f_Pk = '../fiducial/CAMB_TABLES/ics_NL_matterpow_0.dat'

z       = 0.0 
Omega_m = 0.3175
Omega_l = 0.6825

bias_HI = 1.0

f_out1 = 'Pk_PT2_z=0.dat'
f_out2 = 'Pk_PT2_RS_z=0.dat'
##############################################################################

#compute the growth rate
Omega_mz = Omega_m*(1.0+z)**3/(Omega_m*(1.0+z)**3+Omega_l)
f = Omega_mz**(6.0/11.0)

#read the file with the linear matter power spectrum
k,Pk = np.loadtxt(f_Pk,unpack=True)

#compute the value of sigma_v
yinit = [0.0];  k_limits = [np.min(k),np.max(k)]
I = si.odeint(int_sigma, yinit, k_limits, args=(k,Pk),mxstep=1000000,
              rtol=1e-7,atol=1e-9,h0=1e-10)[1][0]
sigma_v = np.sqrt(I/(6.0*np.pi**2))
print 'sigma_v =',sigma_v

sigma_v = 1e-9


####################### REAL SPACE #######################
#compute P_PT(k) = e^(-k^2*sigma_v^2)*P_lin(k)
Pk_PT = np.exp(-k**2*sigma_v**2)*Pk

#save results to file
np.savetxt(f_out1,np.transpose([k,Pk_PT]))



##################### REDSHIFT SPACE #####################
#compute P_PT(k) in redshift-space
q = np.sqrt(2.0+f)*k*sigma_v
bias = bias_HI
prefactor = (3.0+4.0*q**2*(bias+bias**2*q**2))/(8.0*q**4)*\
            np.sqrt(np.pi)*se.erf(np.sqrt(f)*q)/(np.sqrt(f)*q) - \
            (3.0+2.0*(2.0*bias+f)*q**2)*np.exp(-f*q**2)/(4.0*q**4)

Pk_PT_RS = prefactor*np.exp(-k**2*sigma_v**2)*Pk

#save results to file
np.savetxt(f_out2,np.transpose([k,Pk_PT_RS]))


