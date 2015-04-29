#This script computes the HIMF and Omega_HI given the Salpeter parameters.
#We have taken the Haynes et al. 2011 2DSWML,15000 km/s parameters
#(see table 5). This is the HIMF used by Dave et al. 2013
#The HIMF (phi(M_HI)) gives the number density of galaxies with masses in 
# the interval [M_HI,M_HI+dM_HI] divided by log10(dM_HI), i.e. 
#phi(M_HI) = dN/dlog10(M_HI) = log(10)*phi_star*(M_HI/M_star)^(alpha+1)*
#exp(-M_HI/M_star)  see Eq. 3. of 1408.3392
#To compute Omega_HI = 1/rho_crit int_0^infty phi(M_HI)*M_HI*dlog10(M_HI)
#which is equivalent to 1/rho_crit int_-infty^infty phi(x)*10^x dx
#with x=log10(M_HI)
import numpy as np
import scipy.integrate as si


def Omega_HI_from_HIMF(y,x,M_HI,HIMF):
    return np.interp(10**x,M_HI,HIMF)*10**x


rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
########################## INPUT ##########################
alpha    = -1.34
phi_star = 4.7e-3    #h_70^3 Mpc^-3 dex^-1
M_star   = 10**9.96  #Msun/h_70^2: h_70 = h/70 km/s/Mpc

h = 0.7   #H0 in units of 100 km/s/Mpc; needed to compute Omega_HI

f_HIMF = 'ALFALFA.txt'
###########################################################

bins = 1000
M_HI = np.logspace(5,12,bins)  #HI mass in Msun

HIMF = np.log(10.0)*phi_star*(M_HI/M_star)**(1.0+alpha)*\
       np.exp(-M_HI/M_star)  #h_70 Mpc^-3 dex^-1

#Save results to file: M_HI(Msun), HIMF(h_70 Mpc^-3 dex^-1)
np.savetxt(f_HIMF,np.transpose([M_HI,HIMF]))


#compute the value of Omega_HI
yinit = [0.0];  MHI_limits = [np.log10(np.min(M_HI)),np.log10(np.max(M_HI))]
I = si.odeint(Omega_HI_from_HIMF,yinit,MHI_limits,args=(M_HI,HIMF),
              mxstep=100000,rtol=1e-10,atol=1e-20,h0=1e-10)[1][0]
print 'Omega_HI = %.3e h_70'%(I/(rho_crit*h**2))


