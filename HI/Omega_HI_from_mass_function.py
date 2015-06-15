#This script will compute the value of Omega_HI by integrating the mass
#function and using a model for the function M_HI(M). Initially it will look
#whether there is already a file containing the mass function. Otherwise the
#code will compute it and save it.

import numpy as np
import scipy.integrate as si
import bias_library as BL
import mass_function_library as MFL
import sys,os


def M_HI(M):
    M_HI_min = 2e9  #Msun/h
    if M<M_HI_min:  M_HI = 0.0
    else:           M_HI = (15*M**0.7+5e7)*(1.0-np.exp(-M/2e9)**0.3) 
                    #M_HI = 15.0*M**0.7 #2.6*M**0.8  
    return M_HI

#This function performs the integral \int_Mmin^Mmax dn/dm*M_HI(H) dM
def func(y,x,M,MF):
    MF_interp = 10**(np.interp(np.log10(x), np.log10(M), np.log10(MF)))
    return [MF_interp*M_HI(x)]
                      
                      
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
################################# INPUT ######################################
M_min = 1e7  #Msun/h
M_max = 1e16 #Msun/h
bins  = 100

z          = 3.0
f_Pk_DM    = '../fiducial/CAMB_TABLES/ics_matterpow_3.dat'
f_transfer = '../fiducial/CAMB_TABLES/ics_transfer_3.dat'

do_CDM    = False  #whether use the matter or only CDM power spectrum
Omega_CDM = 0.2685 #set the values for do_CDM = True or do_CDM = False   
Omega_B   = 0.0490 #set the values for do_CDM = True or do_CDM = False   

f_MF = 'ST_MF_z=3.dat'
##############################################################################


#compute/read the halo mass function
if os.path.exists(f_MF):
    M,dndM = np.loadtxt(f_MF,unpack=True)

else:
    OmegaM = Omega_CDM+Omega_B
    M = np.logspace(np.log10(M_min),np.log10(M_max),bins+1)

    if do_CDM:  [k,Pk] = BL.CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)    
    else:       [k,Pk] = BL.DM_Pk(f_Pk_DM)

    dndM = MFL.ST_mass_function(k,Pk,OmegaM,None,None,None,M)[1]
    #dndM = MFL.Tinker_mass_function(k,Pk,OmegaM,None,None,None,M)[1]
    #dndM = MFL.Crocce_mass_function(k,Pk,OmegaM,z,None,None,None,M)[1]
    #dndM = MFL.Jenkins_mass_function(k,Pk,OmegaM,None,None,None,M)[1]
    #dndM = MFL.Warren_mass_function(k,Pk,OmegaM,None,None,None,M)[1]

    np.savetxt(f_MF,np.transpose([M,dndM]))




#compute the value Omega_HI
yinit = [0.0];  M_limits = [np.min(M),np.max(M)]
I = si.odeint(func,yinit,M_limits,args=(M,dndM),mxstep=1000000,
              rtol=1e-10,atol=1e-7,h0=1e1)[1][0]
Omega_HI = I/rho_crit
print 'Omega_HI = %.4e'%Omega_HI
