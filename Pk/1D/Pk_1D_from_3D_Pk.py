# This script reads a 3D P(k) file and computes the 1D power spectrum that it
# will measured directly from the simulation. It requires the 3D P(k) file and
# the value of the BoxSize and the value of the parameter dims used to estimate
# the 3D P(k). We provide three different options to compute the integral that
# relates the 1D and 3D power spectra
# In order to compare with the 1D P(k) measured directly from simulations it is
# important to account for the MAS in the perpendicular direction. Here we 
# implement a 2D top-hat as an approximation to NGP and a fully NGP that 
# requires a 2D integral (very expensive).
import numpy as np
import scipy.integrate as si
import scipy.special as ss
import sys,os

# The routine computes 1D P(k) from 3D P(k) making the integral over |k|
# P_1D(k_par) = 1/(2*pi) \int_(k_par)^infty k*P_3D(k) dk
# Implement your own MAS function here. Here we assume that MAS is NGP and we 
# approximate it by a 2D top-hat function of radius pi*R^2 = (BoxSize/dims)^2
def Pk_1D_integrand_k(y,x,k,Pk,k_par,R):
    k_perp = np.sqrt(x**2-k_par**2)
    if k_perp == 0.0:  W = 1.0
    else:              W = (2.0*ss.jv(1,k_perp*R)/(k_perp*R))
    return np.interp(x,k,Pk,left=0,right=0)*x*W**2

# The routine computes 1D P(k) from 3D P(k) making the integral over k_perp
# P_1D(k_par) = 1/(2*pi) \int_0^infty k_perp*P_3D(k_par,k_perp) dk_perp
# Implement your own MAS function here. Here we assume that MAS is NGP and we 
# approximate it by a 2D top-hat function of radius pi*R^2 = (BoxSize/dims)^2
def Pk_1D_integrand_k_perp(y,x,k,Pk,k_par,R):
    k_mod = np.sqrt(x**2 + k_par**2)  #k = sqrt(k_perp^2 + k_par^2)
    W = (2.0*ss.jv(1,x*R)/(x*R))
    return np.interp(k_mod,k,Pk,left=0,right=0)*x*W**2

# The routine computes 1D P(k) from 3D P(k) making the integral over kx and ky
# P_1D(k_par) = 1/(2*pi)^2 \int_{-infty}^infty P_3D(k_par,kx,ky) dkx dky
# The MAS implemented here is the exact NGP
def Pk_1D_integrand_kx_ky(kx,ky,k,Pk,k_par,BoxSize,dims):
    k_mod = np.sqrt(kx**2 + ky**2 + k_par**2)
    value_x = kx*BoxSize/(2.0*dims);  value_y = ky*BoxSize/(2.0*dims)    
    if value_x ==0.0:  Wx = 1.0
    else:              Wx = np.sin(value_x)/value_x
    if value_y ==0.0:  Wy = 1.0
    else:              Wy = np.sin(value_y)/value_y
    return Wx**2*Wy**2*np.interp(k_mod,k,Pk,left=0.0,right=0.0)


################################## INPUT #####################################
f_3D_Pk = '../new_fR5_1keV_60_512/Pk_m_512_z=3.000.dat'
dims    = 512
BoxSize = 60.0  #Mpc/h
f_1D_Pk = 'Pk_1D_from_3D_512_fR5_1keV_z=3.dat'

# choose among: 'over_k', 'over_k_perp' and 'over_kx_ky'. 
# MAS is 2D top-hat for 'over_k' and 'over_k_perp' while is NGP for 'over_kx_ky'
# the option 'over_kx_ky' can be computationally very expensive. Results among
# 'over_k' and 'over_k_perp' should be the same
integral = 'over_k_perp' 
##############################################################################

# read the 3D P(k) file
k,Pk = np.loadtxt(f_3D_Pk,unpack=True)

# compute the Nyquist frequency
k_ny = np.pi*dims/BoxSize  #h/Mpc

# for 'over_k' and 'over_k_perp' we assume that MAS is NGP and approximate 
# W_NGP(k_x)*W_NGP(k_y) by a 2D top-hat filter with radius 
# pi*R^2 = (BoxSize/dims)^2
R = (BoxSize*1.0/dims)/np.sqrt(np.pi)

# define the k_1D and 1D P(k) arrays
k_1D  = k[np.where(k<k_ny)[0]]
Pk_1D = np.empty(len(k_1D),dtype=np.float32)

# P_1D(k_par) = 1/(2*pi) \int_(k_par)^infty k*P_3D(k)dk
if integral == 'over_k':
    for i,k_par in enumerate(k_1D):
        yinit = [0.0];  k_limits = [k_par,k_ny]
        Pk_1D[i] = si.odeint(Pk_1D_integrand_k,yinit,k_limits,
                             args=(k,Pk,k_par,R),
                             mxstep=100000,rtol=1e-10,atol=1e-12,
                             h0=1e-10)[1][0]/(2*np.pi)
        print k_par,Pk_1D[i]

# P_1D(k_par) = 1/(2*pi) \int_0^infty k_perp*P_3D(k_par,k_perp) dk_perp
elif integral == 'over_k_perp':
    for i,k_par in enumerate(k_1D):
        k_perp_max = np.sqrt(k_ny**2-k_par**2)
        yinit = [0.0];  k_limits = [1e-10,k_perp_max]
        Pk_1D[i] = si.odeint(Pk_1D_integrand_k_perp,yinit,k_limits,
                             args=(k,Pk,k_par,R),
                             mxstep=100000,rtol=1e-10,atol=1e-12,
                             h0=1e-10)[1][0]/(2*np.pi)
        print k_par,Pk_1D[i]

# P_1D(k_par) = 1/(2*pi)^2 \int_{-\infty}^infty P_3D(k_par,kx,ky) dkx dky
elif integral == 'over_kx_ky':

    bins = 1000 # only set when making a grid to compute the integral
    for i,k_par in enumerate(k_1D):

        # integration using dblquad
        I,dI = si.dblquad(Pk_1D_integrand_kx_ky, 0, k_ny,
                          lambda x:0, lambda x:k_ny,
                          args=(k,Pk,k_par,BoxSize,dims),
                          epsabs=1e-8, epsrel=1e-8)
        # factor 4 arises since integral goes from kx,ky=[-infty,infty]
        # We just do kx,ky=[0,infty] and multiply by 4
        Pk_1D[i] = 4.0*I/(2.0*np.pi)**2
        print k_par, Pk_1D[i], dI/(2.0*np.pi)**2

        """
        # Make integral by computing the value of the function in a regular grid
        I = 0
        for i in xrange(bins):
            kx = k[-1]*i*1.0/bins
            for j in xrange(bins):
                ky = k[-1]*j*1.0/bins
                I += Pk_1D_integrand_kx_ky(kx,ky,k,Pk,k_par,BoxSize,dims)
        Pk_1D[i] = 4.0*I*(k_ny*1.0/bins)**2/(2.0*np.pi)**2
        print k_par,Pk_1D[i]
        """

else:
    print 'bad choice for integral!!!';  sys.exit()

np.savetxt(f_1D_Pk,np.transpose([k_1D,Pk_1D]))    

if integral in ['over_k','over_k_perp']:
    print '1D P(k) computed from 3D P(k) approximating NGP MAS by 2D top-hat'
if integral=='over_kx_ky':
    print '1D P(k) computed from 3D P(k) assuming NGP MAS'
