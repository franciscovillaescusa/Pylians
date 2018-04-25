import numpy as np
import sys,os,h5py
import units_library as UL

# This routine returns the circular velocity of a halo of mass M at redshift z
def Vcirc(M,z,Omega_m,Omega_l):

    x = Omega_m*(1.0+z)**3/(Omega_m*(1.0+z)**3 + Omega_l) - 1.0
    delta_c = 18*np.pi**2 + 82.0*x - 39.0*x**2

    rho_crit_z = rho_crit*(Omega_m*(1.0+z)**3 + Omega_l)
    Hz = 100.0*np.sqrt(Omega_m*(1.0+z)**3 + Omega_l) #km/s/(Mpc/h)
    
    R = ((3.0*M)/(4.0*np.pi*delta_c*rho_crit_z))**(1.0/3.0) #Mpc/h
    
    V = np.sqrt(0.5*Hz**2*delta_c*R**2)

    return V

rho_crit = UL.units().rho_crit
############################### INPUT #######################################
Omega_m = 0.3089
Omega_l = 1.0-Omega_m

threshold = 0.98 #find Mmin such as lighter halos only host 1-threshold % of HI
#############################################################################

for z in [0,1,2,3,4,5]:

    f = h5py.File('M_HI_SO_TopHat200_z=%.1f.hdf5'%z, 'r')
    M_HI = f['M_HI_SO'][:]
    M    = f['mass_SO'][:]
    R    = f['R'][:]
    f.close()

    Omega_HI = np.sum(M_HI, dtype=np.float64)/(75.0**3*2.775e11)
    M_HI_tot = np.sum(M_HI, dtype=np.float64)
    print '\nOmega_HI(z=%d) = %.3e'%(z,Omega_HI)

    Mmin_a, Mmin_b = 1e8, 1e15
    for i in xrange(20):
        Mmin = 10**(0.5*(np.log10(Mmin_a) + np.log10(Mmin_b)))
        indexes = np.where(M>Mmin)[0]
        ratio_HI = np.sum(M_HI[indexes], dtype=np.float64)/M_HI_tot

        if ratio_HI>threshold:  Mmin_a = Mmin
        else:                   Mmin_b = Mmin

    print 'Mmin = %.2e ---> Vcirc = %.2f km/s ---> ratio = %.3f'\
        %(Mmin, Vcirc(Mmin,z,Omega_m,Omega_l), ratio_HI)
