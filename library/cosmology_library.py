import numpy as np
import scipy.integrate as si

#################### FUNCTIONS USED TO COMPUTE INTEGRALS #####################

#we use this function to compute the comoving distance to a given redshift
def func(y,x,Omega_m,Omega_L):
    return [1.0/np.sqrt(Omega_m*(1.0+x)**3+Omega_L)]
##############################################################################



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
##############################################################################

#This function computes the absoption distance:
#dX = H0*(1+z)^2/H(z)*dz
#Omega_m ----> value of the Omega_m cosmological parameter
#Omega_L ----> value of the Omega_L cosmological parameter
#z ----------> cosmological redshift
#BoxSize ----> size of the simulation box in Mpc/h
def absorption_distance(Omega_m,Omega_L,z,BoxSize):
    iterations=40; tol=1e-4; i=0; final=False
    dz_max=10.0; dz_min=0.0; dz=0.5*(dz_min+dz_max)
    r0=comoving_distance(z,Omega_m,Omega_L) #Mpc/h
    while not(final):
        dr=comoving_distance(z+dz,Omega_m,Omega_L)-r0
        if (np.absolute(dr-BoxSize)/BoxSize)<tol or i>iterations:
            final=True
        else:
            i+=1
            if dr>BoxSize:
                dz_max=dz
            else:
                dz_min=dz
            dz=0.5*(dz_min+dz_max)

    dX=(1.0+z)**2/np.sqrt(Omega_m*(1.0+z)**3+Omega_L)*dz
    return dX
##############################################################################









###############################################################################
################################### USAGE #####################################
###############################################################################

###### comoving distance ######
"""
z=3.0
Omega_m=0.3
Omega_L=0.7

r=comoving_distance(z,Omega_m,Omega_L)
print 'comoving distance to z = %2.2f ---> %f Mpc/h'%(z,r)
"""

###### absorption distance ######
"""
Omega_m=0.274247
Omega_L=0.725753
z=3.0
BoxSize=60.0 #Mpc/h

dX=absorption_distance(Omega_m,Omega_L,z,BoxSize)
print 'dX =',dX
"""
