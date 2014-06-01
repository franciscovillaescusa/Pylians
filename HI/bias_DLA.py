#In this code we compute the cross section of dark matter halos as a function
#of mass following the model of Barnes & Haehnelt 2014

import numpy as np
import readsnap
import scipy.integrate as si
import sys
import mass_function_library as MFL
import bias_library as BL


#Returns the DLA cross-section for a given halo mass, redshift and model
#it returns the DLA cross-section in (kpc/h)^2 physical units
def cross_section_model(M,redshift,model):

    if model=='Bagla':
        z_t       = [2.4, 3.0, 4.0]
        sigma0_t  = [1641.0, 2429.0, 4886.0]    #comoving (kpc/h)^2
        M0_t      = [3.41e11, 2.67e11, 2.81e11] #Msun/h
        alpha_t   = [0.79, 0.85, 0.79]
        beta_t    = [1.81, 1.89, 2.04]
        Mmin_t    = [1.12e9, 8.75e8, 6.26e8]    #Msun/h

        sigma0  = np.interp(redshift,z_t,sigma0_t)
        M0      = np.interp(redshift,z_t,M0_t)
        alpha   = np.interp(redshift,z_t,alpha_t)
        beta    = np.interp(redshift,z_t,beta_t)
        Mmin    = np.interp(redshift,z_t,Mmin_t)
        
        print 'sigma0 = %3.0f\nM0=%1.3e\nalpha=%1.3f\nbeta=%1.3f\nMmin=%1.3e'\
            %(sigma0,M0,alpha,beta,Mmin)
        
        if M>Mmin:
            sigma=sigma0*(M/M0)**alpha\
                *(1.0+(M/M0)**beta)**(-alpha/beta)/(1.0+redshift)**2
        else:
            sigma=1e-20

    elif model=='Dave':
        z_t      = [2.4, 3.0, 4.0]
        sigma0_t = [18.1, 29.1, 44.4] #comoving (kpc/h)^2
        M0_t     = [7e8, 6e8, 5e8]        #Msun/h
        beta_t   = [0.721, 0.739, 0.750]

        sigma0  = np.interp(redshift,z_t,sigma0_t)
        M0      = np.interp(redshift,z_t,M0_t)
        beta    = np.interp(redshift,z_t,beta_t)


        sigma=sigma0*(M/M0)**beta*np.exp(-(M0/M)**3)/(1.0+redshift)**2

    else:
        print 'Incorrect model!!!'; sys.exit()

    return sigma

#This function computes the DLA incidence rate
def DLA_incidence_rate(M,dndM,cross_section):

    M_min=np.min(M); M_max=np.max(M)
    M_limits=[M_min,M_max]; yinit=[0.0]
    I=si.odeint(deriv_2,yinit,M_limits,rtol=1e-10,atol=1e-20,
                args=(M,dndM,cross_section),mxstep=10000000)[1][0]
    return I*3e5/100.0/1e6

#This function computes the DLA bias
#M should be in Msun/h, dndM in halos/(Mpc/h)^3/(Msun/h) and cross_section in
#(kpc/h)^2
def DLA_bias(M,bias,dndM,cross_section):

    M_min=np.min(M); M_max=np.max(M)

    M_limits=[M_min,M_max]; yinit=[0.0]
    I1=si.odeint(deriv_1,yinit,M_limits,rtol=1e-9,atol=1e-10,
                 args=(M,bias,dndM,cross_section),mxstep=10000000)[1][0]

    M_limits=[M_min,M_max]; yinit=[0.0]
    I2=si.odeint(deriv_2,yinit,M_limits,rtol=1e-9,atol=1e-10,
                 args=(M,dndM,cross_section),mxstep=10000000)[1][0]

    return I1/I2

def deriv_1(y,x,M,bias,dndM,cross_section):
    b    =10**(np.interp(np.log10(x),np.log10(M),np.log10(bias)))
    n_M  =10**(np.interp(np.log10(x),np.log10(M),np.log10(dndM)))
    sigma=10**(np.interp(np.log10(x),np.log10(M),np.log10(cross_section)))
    return[b*n_M*sigma]

def deriv_2(y,x,M,dndM,cross_section):
    n_M  =10**(np.interp(np.log10(x),np.log10(M),np.log10(dndM)))
    sigma=10**(np.interp(np.log10(x),np.log10(M),np.log10(cross_section)))
    return[n_M*sigma]



################################# UNITS #####################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

c=3e5 #km/s
kpc=3.0856e21 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams

pi=np.pi
#############################################################################

################################### INPUT #####################################
snapshot_fname='../Efective_model_60Mpc/snapdir_007/snap_007'

#files containing the power spectrum, mass function and halo bias
f_Pk='../mass_function/CAMB_TABLES/ics_matterpower_z=4.dat' #units should be 
f_MF='ST_MF_BH_z=4.dat' #units should be halos/(Mpc/h)^3/(Msun/h)
f_bias='bias_BH_z=4.dat'

model='Dave'

#mass range for the dark matter halos in Msun/h
Mmin=1e8; Mmax=1e15

#compute bias and mass function
compute_bias=False
compute_MF=False

bins=100

f_out_cross_section='cross_section_Dave_z=4.dat'
###############################################################################

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h=head.hubble

#read the CAMB power spectrum file
[k,Pk]=BL.DM_Pk(f_Pk)

#select the masses of the dark matter halos over which the quantities will be 
#computed (bias, mass function and cross_section)
M=np.logspace(np.log10(Mmin),np.log10(Mmax),bins) #in Msun/h

#compute the halo mass function or read it from a file
if compute_MF:
    dndM=MFL.ST_mass_function(k,Pk,Omega_m,None,None,None,M)[1]
    np.savetxt(f_MF,np.transpose([M,dndM]))
else:
    M_MF,MF=np.loadtxt(f_MF,unpack=True)
    if Mmin<np.min(M_MF): #check that the file explores the minimum mass
        print 'compute the mass function for smaller masses'; sys.exit()
    if Mmax>np.max(M_MF): #check that the file explores the maximum mass
        print 'compute the mass function for larger masses'; sys.exit()
    dndM=10**(np.interp(np.log10(M),np.log10(M_MF),np.log10(MF))); del M_MF,MF

#compute the halo bias or read it from a file
if compute_bias:
    bias=np.zeros(bins,dtype=np.float64)
    for i in xrange(bins):
        bias[i]=BL.bias(k,Pk,Omega_m,M[i],author='SMT01') #'ST01' 'Tinker'
        print 'M = %1.3e Msun/h ---> bias = %1.3f'%(M[i],bias[i])
    np.savetxt(f_bias,np.transpose([M,bias]))
else:
    M_b,b=np.loadtxt(f_bias,unpack=True)
    if Mmin<np.min(M_b): #check that the file explores the minimum mass
        print 'compute the bias for smaller masses'; sys.exit()
    if Mmax>np.max(M_b): #check that the file explores the maximum mass
        print 'compute the bias for larger masses'; sys.exit()
    bias=10**(np.interp(np.log10(M),np.log10(M_b),np.log10(b))); del M_b,b

#compute the DLA cross section as a function of the dark matter halo mass
cross_section=np.zeros(bins,dtype=np.float64)
f=open(f_out_cross_section,'w')
for i in xrange(bins):
    cross_section[i]=cross_section_model(M[i],redshift,model)
    print 'M = %1.3e Msun/h ----> sigma = %1.3e (kpc/h)^2'\
        %(M[i],cross_section[i])    
    f.write(str(M[i])+' '+str(cross_section[i])+'\n')
f.close()

#compute the DLA incidence rate
print 'dN/dX = %f'%DLA_incidence_rate(M,dndM,cross_section)

#compute the DLA bias
print 'b_D=',DLA_bias(M,bias,dndM,cross_section)





