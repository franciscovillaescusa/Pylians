import numpy as np
import mass_function_library as MFL
import bias_library as BL
import scipy.integrate as si
import sys,os

def M_HI_model(M,z,h):

    v_min, v_max = 30.0, 200.0 #km/s
    
    Mmin_HI = 1e10*h*(v_min/60.0)**3*((1.0+z)/4.0)**(-3.0/2.0)  #Msun/h
    Mmax_HI = 1e10*h*(v_max/60.0)**3*((1.0+z)/4.0)**(-3.0/2.0)  #Msun/h

    M_HI = np.empty(len(M),dtype=np.float64)
    M_HI = M/(1.0+M/Mmax_HI)

    indexes       = np.where(M<Mmin_HI)[0]
    M_HI[indexes] = 0.0;  del indexes

    """
    #Model M_HI(M) of the paper with neutrinos
    M_HI = np.empty(len(M),dtype=np.float64)
    M_HI = (M/0.012)**0.699

    indexes       = np.where(M<2e9)[0]
    M_HI[indexes] = 0.0;  del indexes
    """

    return M_HI

    #Mmin_HI, Mmax_HI = 1e10, 1e16  #Msun/h
    #M_HI = M*1.0
    #indexes = np.where(M<Mmin_HI)[0]; M_HI[indexes]=0.0
    #indexes = np.where(M>Mmax_HI)[0]; M_HI[indexes]=0.0
    #compute minimum and maximum circular velocity
    #v_min = 60.0*((Mmin_HI/h)/1e10*((1.0+z)/4.0)**1.5)**(1.0/3.0) #km/s
    #v_max = 60.0*((Mmax_HI/h)/1e10*((1.0+z)/4.0)**1.5)**(1.0/3.0) #km/s
    #print 'Minimum circular velocity =',v_min,'km/s'
    #print 'Maximum circular velocity =',v_max,'km/s'
    #return M_HI

#This routine computes: \int_0^infty n(M,z)*b(M,z)*M_HI(M,z)
def integral1(M,dndM,b,M_HI):
    yinit = [0.0];  M_limits = [M[0],M[-1]]
    I = si.odeint(int1, yinit, M_limits, args=(M,dndM,b,M_HI), mxstep=10000000,
                  rtol=1e-10, atol=1e-8, h0=1.0)[1][0]
    return I

def int1(y,x,M,dndM,b,M_HI):
    dndM_interp = np.interp(np.log10(x),np.log10(M),dndM)
    b_interp    = np.interp(np.log10(x),np.log10(M),b)
    M_HI_interp = np.interp(np.log10(x),np.log10(M),M_HI)
    return [dndM_interp*b_interp*M_HI_interp]

#This routine computes: \int_0^infty n(M,z)*M_HI(M,z)
def integral2(M,dndM,M_HI):
    yinit = [0.0];  M_limits = [M[0],M[-1]]
    I = si.odeint(int2, yinit, M_limits, args=(M,dndM,M_HI), mxstep=1000000,
                  rtol=1e-10, atol=1e-8, h0=1.0)[1][0]
    return I

def int2(y,x,M,dndM,M_HI):
    dndM_interp = np.interp(np.log10(x),np.log10(M),dndM)
    M_HI_interp = np.interp(np.log10(x),np.log10(M),M_HI)
    return [dndM_interp*M_HI_interp]

rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3 
################################# INPUT ######################################
if len(sys.argv)>1:
    sa=sys.argv

    z = float(sa[1]);  f_Pk_DM = sa[2];  f_transfer = sa[3]
    do_CDM = bool(int(sa[4]))
    Omega_CDM = float(sa[5]);  Omega_B = float(sa[6]);  h = float(sa[7])
    fix_Omega_HI = bool(int(sa[8]))
    f_MF = sa[9];  f_bias = sa[10];  
    f_21cm_real_space = sa[11];  f_21cm_redshift_space = sa[12]

    print sa

else:
    z          = 3.0
    f_Pk_DM    = '../0.6eV/CAMB_TABLES/ics_matterpow_3.dat'
    f_transfer = '../0.6eV/CAMB_TABLES/ics_transfer_3.dat'

    do_CDM     = True   #whether use the matter or the CDM+B P(k) to compute MF
    Omega_CDM  = 0.25435
    Omega_B    = 0.0490 
    h          = 0.6711

    fix_Omega_HI = False  #whether fix Omega_HI to 4e-4*(1+z)^0.6

    f_MF                  = 'ST_MF_0.6eV_z=3.dat'
    f_bias                = 'ST_bias_0.6eV_z=3.dat'
    f_21cm_real_space     = 'Pk_21cm_0.6eV_z=3.dat'
    f_21cm_redshift_space = 'Pk_21cm_0.6eV_z=3.dat'
##############################################################################

#compute the value of OmegaM
OmegaM = Omega_CDM + Omega_B

#parameters for the mass functions and bias
M_min = 1e7  #Msun/h
M_max = 1e16 #Msun/h
bins  = 200

#From Crighton et al. 2015
Omega_HI = 4e-4*(1.0+z)**0.60

#define the bins in mass
M = np.logspace(np.log10(M_min),np.log10(M_max),bins+1)


#read the matter power spectrum
if do_CDM:  [k,Pk] = BL.CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)
else:       [k,Pk] = BL.DM_Pk(f_Pk_DM)

#compute/read the halo mass function
if not(os.path.exists(f_MF)):
    dndM = MFL.ST_mass_function(k,Pk,OmegaM,None,None,None,M)[1]
    np.savetxt(f_MF,np.transpose([M,dndM]))
else:
    M_MF,dndM = np.loadtxt(f_MF,unpack=True)
    if np.any(M_MF!=M):
        print 'error!!\nbins in mass function are different'; sys.exit()

#compute/read the halo bias
if not(os.path.exists(f_bias)):
    b = np.empty(bins+1,dtype=np.float64)
    for i,mass in enumerate(M):
        b[i] = BL.bias(k,Pk,OmegaM,mass,'SMT01')
    np.savetxt(f_bias,np.transpose([M,b]))
else:
    M_b,b = np.loadtxt(f_bias,unpack=True)
    if np.any(M_b!=M):
        print 'error!!\nbins in halo bias are different'; sys.exit()

#compute the M_HI function
M_HI = M_HI_model(M,z,h)

#compute the bias of the HI
numerator   = integral1(M,dndM,b,M_HI)
denominator = integral2(M,dndM,M_HI)

if fix_Omega_HI:
    # Omega_HI = f3/rho_crit*\int_0^infty n(M)M_HI(M)dM
    f3 = Omega_HI*rho_crit/denominator
    print 'Omega_HI(z=%.1f) = %.3e'%(z,Omega_HI)
    print 'f3 = %.3e'%f3
    numerator*=f3; denominator*=f3
else:
    Omega_HI = denominator/rho_crit
    print 'Omega_HI(z=%.1f) = %.3e'%(z,Omega_HI)

b_HI = numerator/denominator

print 'numerator   =',numerator
print 'denominator =',denominator
print 'b_HI(z=%.1f) = %.3f'%(z,b_HI)

delta_Tb = 23.88*Omega_HI*h**2/(0.02*0.76)*\
           np.sqrt(0.15*(1.0+z)/(10.0*OmegaM*h**2)) #mK
print 'delta_Tb(z=%.1f) = %.4f mK'%(z,delta_Tb)

OmegaM_z = OmegaM*(1.0+z)**3/(OmegaM*(1.0+z)**3 + (1.0-OmegaM))
beta     = OmegaM_z**(6.0/11.0)/b_HI

#compute the 21cm power spectrum and save results to file
prefactor = delta_Tb**2 * b_HI**2
Pk_21cm_real_space     = prefactor * Pk
Pk_21cm_redshift_space = prefactor * (1.0 + 2.0*beta/3.0 + beta**2/5.0) * Pk
np.savetxt(f_21cm_real_space,     np.transpose([k,Pk_21cm_real_space]))
np.savetxt(f_21cm_redshift_space, np.transpose([k,Pk_21cm_redshift_space]))
