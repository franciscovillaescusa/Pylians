import numpy as np
import scipy.integrate as si

#################### FUNCTIONS USED TO COMPUTE INTEGRALS #####################

#we use this function to compute the comoving distance to a given redshift
def func(y,x,Omega_m,Omega_L):
    return [1.0/np.sqrt(Omega_m*(1.0+x)**3+Omega_L)]
##############################################################################




#This functions computes the comoving distance to redshift z, in Mpc/h
#As input it needs z, Omega_m and Omega_L. It assumes a flat cosmology
def comoving_distance(z,Omega_m,Omega_L):
    H0=100.0 #km/s/(Mpc/h)
    c=3e5    #km/s

    #compute the comoving distance to redshift z
    yinit=[0.0]
    z_limits=[0.0,z]
    I=si.odeint(func,yinit,z_limits,args=(Omega_m,Omega_L),
                rtol=1e-8,atol=1e-8,mxstep=100000,h0=1e-6)[1][0]
    r=c/H0*I

    return r
