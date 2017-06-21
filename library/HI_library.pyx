import numpy as np 
import time,sys,os 
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,log10,abs,exp
import readsnap
import time


################################# UNITS #####################################
cdef double rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

cdef double yr=3.15576e7  #seconds
cdef double km=1e5        #cm
cdef double Mpc=3.0856e24 #cm
cdef double kpc=3.0856e21 #cm
cdef double Msun=1.989e33 #g
cdef double Ymass=0.24   #helium mass fraction
cdef double mH=1.6726e-24  #proton mass in grams
cdef double gamma=5.0/3.0  #ideal gas
cdef double kB=1.3806e-26  #gr (km/s)^2 K^{-1}
cdef double nu0=1420.0     #21-cm frequency in MHz

pi=np.pi 
#############################################################################


# This routine implements reads Gadget gas output and correct HI/H fractions
# to account for self-shielding and H2
# snapshot_fname -------> name of the N-body snapshot
# fac ------------------> factor to reproduce the mean Lya flux
# TREECOOL_file --------> TREECOOL file used in the N-body
# Gamma_UVB ------------> value of the UVB photoionization rate
# correct_H2 -----------> correct the HI/H fraction to account for H2
# if Gamma_UVB is set to None the value of the photoionization rate will be read
# from the TREECOOL file, otherwise it is used the Gamma_UVB value
def Rahmati_HI_assignment(snapshot_fname, fac, TREECOOL_file, Gamma_UVB=None,
                          correct_H2=False, IDs=None):

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    print '\nREADING SNAPSHOT PROPERTIES'
    head     = readsnap.snapshot_header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h
    Nall     = head.nall
    redshift = head.redshift

    # Rahmati et. al. 2013 self-shielding parameters (table A1)
    z_t       = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00])
    n0_t      = np.array([-2.94,-2.29,-2.06,-2.13,-2.23,-2.35]); n0_t=10**n0_t
    alpha_1_t = np.array([-3.98,-2.94,-2.22,-1.99,-2.05,-2.63])
    alpha_2_t = np.array([-1.09,-0.90,-1.09,-0.88,-0.75,-0.57])
    beta_t    = np.array([1.29, 1.21, 1.75, 1.72, 1.93, 1.77])
    f_t       = np.array([0.01, 0.03, 0.03, 0.04, 0.02, 0.01])

    # compute the self-shielding parameters at the redshift of the N-body
    n0      = np.interp(redshift,z_t,n0_t)
    alpha_1 = np.interp(redshift,z_t,alpha_1_t)
    alpha_2 = np.interp(redshift,z_t,alpha_2_t)
    beta    = np.interp(redshift,z_t,beta_t)
    f       = np.interp(redshift,z_t,f_t)
    print 'n0 = %e\nalpha_1 = %2.3f\nalpha_2 = %2.3f\nbeta = %2.3f\nf = %2.3f'\
        %(n0,alpha_1,alpha_2,beta,f)

    # find the value of the photoionization rate
    if Gamma_UVB is None:
        data=np.loadtxt(TREECOOL_file); logz=data[:,0]; Gamma_UVB=data[:,1]
        Gamma_UVB=np.interp(np.log10(1.0+redshift),logz,Gamma_UVB); del data
        print 'Gamma_UVB(z=%2.2f) = %e s^{-1}'%(redshift,Gamma_UVB)
    
    # Correct to reproduce the Lya forest mean flux
    Gamma_UVB /= fac

    #compute the HI/H fraction 
    nH0 = HI_from_UVB(snapshot_fname, Gamma_UVB, True, correct_H2,
                      f, n0, alpha_1, alpha_2, beta, SF_temperature=1e4)
    
    #read the gas masses
    mass = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h

    #create the array M_HI and fill it
    M_HI = 0.76*mass*nH0;  del nH0,mass
    print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return M_HI



#This function compute the HI/H fraction given the photoionization rate and 
#the density of the gas particles
#snapshot_fname ------------> name of the N-body snapshot
#Gamma_UVB -----------------> value of the UVB photoionization rate
#self_shielding_correction -> apply (True) or not (False) the self-shielding
#correct_H2 ----------------> correct the HI masses to account for H2
#f -------------------------> parameter of the Rahmati et al model (see eq. A1)
#n0 ------------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_1 -------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_2 -------------------> parameter of the Rahmati et al model (see eq. A1)
#beta ----------------------> parameter of the Rahmati et al model (see eq. A1)
#SF_temperature ------------> associate temperature of star forming particles
#If SF_temperature is set to None it will compute the temperature of the SF
#particles from their density and internal energy
def HI_from_UVB(snapshot_fname, double Gamma_UVB,
                self_shielding_correction=False,correct_H2=False,
                double f=0.0, double n0=0.0, double alpha_1=0.0,
                double alpha_2=0.0, double beta=0.0,
                double SF_temperature=0.0):

    cdef long gas_part,i 
    cdef double mean_mol_weight, yhelium, A, B, C, R_surf, P, prefact2
    cdef double nH, Lambda, alpha_A, Lambda_T, Gamma_phot, prefact, T
    cdef np.float32_t[:] rho,U,ne,SFR,nH0

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head     = readsnap.snapshot_header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h
    Nall     = head.nall
    Masses   = head.massarr*1e10 #Msun/h
    Omega_m  = head.omega_m
    Omega_l  = head.omega_l
    redshift = head.redshift
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
    h        = head.hubble
    gas_part = Nall[0]

    # read density, internal energy, electron fraction and star formation rate
    # density units: h^2 Msun / Mpc^3
    yhelium = (1.0-0.76)/(4.0*0.76) 
    U   = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
    ne  = readsnap.read_block(snapshot_fname,"NE  ",parttype=0) #electron frac
    rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10/1e-9 
    SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR

    # define HI/H fraction
    nH0 = np.zeros(gas_part, dtype=np.float32)

    #### Now compute the HI/H fraction following Rahmati et. al. 2013 ####
    prefact  = 0.76*h**2*Msun/Mpc**3/mH*(1.0+redshift)**3 
    prefact2 = (gamma-1.0)*(1.0+redshift)**3*h**2*Msun/kpc**3/kB 

    print 'doing loop...';  start = time.clock()    
    for i in xrange(gas_part):

        # compute particle density in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
        nH = rho[i]*prefact

        # compute particle temperature
        if SFR[i]>0.0:
            T = SF_temperature
            P = prefact2*U[i]*rho[i]*1e-9  #K/cm^3
        else:
            mean_mol_weight = (1.0+4.0*yhelium)/(1.0+yhelium+ne[i])
            T = U[i]*(gamma-1.0)*mH*mean_mol_weight/kB
        
        #compute the case A recombination rate
        Lambda  = 315614.0/T
        alpha_A = 1.269e-13*pow(Lambda,1.503)\
                  /pow(1.0 + pow((Lambda/0.522),0.47),1.923) #cm^3/s

        #Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
        Lambda_T = 1.17e-10*sqrt(T)*\
                   exp(-157809.0/T)/(1.0+sqrt(T/1e5)) #cm^3/s

        #compute the photoionization rate
        Gamma_phot = Gamma_UVB
        if self_shielding_correction:
            Gamma_phot *= ((1.0 - f)*pow(1.0 + pow(nH/n0,beta), alpha_1) +\
                           f*pow(1.0 + nH/n0, alpha_2))

        #compute the coeficients A,B and C to calculate the HI/H fraction
        A = alpha_A + Lambda_T
        B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
        C = alpha_A

        #compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
        nH0[i] = (B-sqrt(B*B-4.0*A*C))/(2.0*A)

        # correct for the presence of H2
        if correct_H2 and SFR[i]>0.0:
            R_surf = (P/1.7e4)**0.8
            nH0[i] = nH0[i]/(1.0 + R_surf)

    print 'Time taken = %.3f s'%(time.clock()-start)
    return np.asarray(nH0)



