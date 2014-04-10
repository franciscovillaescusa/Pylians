#In this code we compute the cross section of dark matter halos as a function
#of mass following the model of Barnes & Haehnelt 2014

import numpy as np
import readsnap
import scipy.integrate as si
import sys
import HI_library as HIL
import mass_function_library as MFL
import bias_library as BL
import readfof

#sanity check
def sigma_DLA(M):

    N_HI_min=10**(20.3); N_HI_max=10**(23.0)
    yinit=[0.0]; N_HI_limits=[N_HI_min,N_HI_max]

    #compute d(pi y^2)/d(N_HI) in a series of points
    points=30
    N_HI=np.logspace(np.log10(N_HI_min),np.log10(N_HI_max),points)
    dcsdNHI=np.empty(points,dtype=np.float64); epsilon=0.01
    f=open('borrar.dat','w')
    for i in xrange(points):
        cs=HIL.cross_section_BH(Omega_m,Omega_b,Omega_l,redshift,h,
                                M,c0,f_HI,alpha_e,V0c,N_HI_ref=N_HI[i],
                                verbose=False)[0]
        cs_e=HIL.cross_section_BH(Omega_m,Omega_b,Omega_l,redshift,h,
                                  M,c0,f_HI,alpha_e,V0c,
                                  N_HI_ref=N_HI[i]*(1.0+epsilon))[0]
        dcsdNHI[i]=np.absolute((cs_e-cs)/(N_HI[i]*epsilon))
        #dcsdNHI[i]=(cs_e-cs)/(N_HI[i]*epsilon)
        f.write(str(N_HI[i])+' '+str(cs)+' '+str(dcsdNHI[i])+'\n')
    f.close()

    I=si.odeint(deriv_sigma_DLA,yinit,N_HI_limits,rtol=1e-8,atol=1e-4,
                args=(N_HI,dcsdNHI),h0=1e15,mxstep=10000000)[1][0]


    
    return I

def deriv_sigma_DLA(y,x,N_HI,dcsdNHI):
    derivate=10**(np.interp(np.log10(x),np.log10(N_HI),np.log10(dcsdNHI)))
    return [derivate]
    

#This function returns the cross-section for a given mass and redshift for a 
#certain model
def cross_section_model(M,redshift):
    #Font et al. 2012 toy models
    #model 1
    #if M[i]>7.1e8:
    #    cross_section[i]=0.994*(M[i]/1e9)
    #else:
    #    cross_section[i]=1e-10
    
    #model 2
    #cross_section[i]=12.22*(M[i]/7.1e9)**2*(1.0+M[i]/7.1e9)**(-1)

    #model 3
    #cross_section[i]=1801.0*(M[i]/2.13e11)**2*(1.0+M[i]/2.13e11)**(-1.5)

    #model 4
    #cross_section[i]=20.16*(M[i]/3.56e9)**2*(1.0+M[i]/3.56e9)**(-1.5)

    #model 5
    #cross_section[i]=2400.0*(M[i]/3.0e11)**0.8\
    #    /(1.0+(M[i]/3e11)**2.2)**(-0.36)/(1.0+redshift)**2

    #model 6
    if M>1.12e9:
        sigma=1640.0*(M/3.46e11)**0.78\
            /(1.0+(M/3e46)**1.84)**(-0.42)/(1.0+redshift)**2
    else:
        sigma=0.0

    return sigma

#This function computes f_HI for the model of Barnes & Haehnelt 2014
def column_density_BH(N_HI_cd,M,dndM,cross_section,Omega_m,Omega_b,Omega_l,
                      redshift,h,c0,f_HI,alpha_e,V0c):

    M_min=np.min(M); M_max=np.max(M)
    M_limits=[M_min,M_max]; yinit=[0.0]

    points=30 #number of points to compute M,dndM and dsigma/dN_HI

    #compute d(pi y^2)/d(N_HI) in a series of points
    Masses=np.logspace(np.log10(M_min),np.log10(M_max),points)
    MF=10**(np.interp(np.log10(Masses),np.log10(M),np.log10(dndM)))
    dcsdNHI=np.empty(points,dtype=np.float64); epsilon=0.1
    for i in xrange(points):
        cs=HIL.cross_section_BH(Omega_m,Omega_b,Omega_l,redshift,h,
                                Masses[i],c0,f_HI,alpha_e,V0c,N_HI_ref=N_HI_cd,
                                verbose=False)[0]
        cs_e=HIL.cross_section_BH(Omega_m,Omega_b,Omega_l,redshift,h,
                                  Masses[i],c0,f_HI,alpha_e,V0c,
                                  N_HI_ref=N_HI_cd*(1.0+epsilon))[0]
        dcsdNHI[i]=np.absolute((cs_e-cs)/(N_HI_cd*epsilon))

        """
        cs=cross_section_model(Masses[i],redshift)
        cs_e=cross_section_model(Masses[i]*(1.0+epsilon),redshift)
        dcsdNHI[i]=np.absolute((cs_e-cs)/(N_HI_cd*epsilon))
        """

    dcsdNHI[np.where(dcsdNHI==0.0)[0]]=10**(-40)

    I=si.odeint(deriv_column_density,yinit,M_limits,rtol=1e-7,atol=1e-25,
                args=(Masses,MF,dcsdNHI),h0=1e4,mxstep=10000000)[1][0]

    return I*3e5/100.0/1e6

def deriv_column_density(y,x,M,dndM,dcsdNHI):
    n_M     =10**(np.interp(np.log10(x),np.log10(M),np.log10(dndM)))
    derivate=10**(np.interp(np.log10(x),np.log10(M),np.log10(dcsdNHI)))
    return [n_M*derivate]


#This function computes the DLA incidence rate
def DLA_incidence_rate(M,dndM,cross_section):

    M_min=np.min(M); M_max=np.max(M)
    M_limits=[M_min,M_max]; yinit=[0.0]
    I=si.odeint(deriv_2,yinit,M_limits,rtol=1e-9,atol=1e-10,
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
snapshot_fname='../Efective_model_60Mpc/snapdir_013/snap_013'
groups_fname='../Efective_model_60Mpc'
groups_number=13

#files containing the power spectrum, mass function and halo bias
#f_Pk='../mass_function/CAMB_TABLES/ics_matterpower_z=2.4.dat' #units should be 
#f_MF='ST_MF_Paco_z=2.4.dat' #units should be halos/(Mpc/h)^3/(Msun/h)
#f_bias='bias_Paco_z=2.4.dat'

f_Pk='CAMB_TABLES/ics_matterpow_2.5.dat' #units should be 
f_MF='ST_MF_BH_z=2.5.dat' #units should be halos/(Mpc/h)^3/(Msun/h)
f_bias='bias_BH_z=2.5.dat'

#Barnes & Haehnelt parameters
f_HI=0.19   #0.37
alpha_e=3
V0c=37.0 #km/s
c0=3.4

#cosmological parameter values
Omega_m=0.281
Omega_b=0.0462
Omega_l=0.719
redshift=2.5
h=0.71

#mass range for the dark matter halos
Mmin=1e8; Mmax=1e15

#compute bias and mass function
compute_bias=False
compute_MF=False

bins=50

f_out_cross_section='cross_section_BH_v0=70.dat'
f_out_f_HI='New_BH.dat6'
###############################################################################

"""
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

#compute the value of Omega_b
Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b=Omega_m-Omega_cdm
print 'Omega_cdm =',Omega_cdm; print 'Omega_b   =',Omega_b
print 'Omega_m   =',Omega_m,'\n'
"""

#read the CAMB power spectrum file
[k,Pk]=BL.DM_Pk(f_Pk)

#select the masses of the dark matter halos over which the quantities will be 
#computed (bias, mass function and cross_section)
M=np.logspace(np.log10(Mmin),np.log10(Mmax),bins)

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
        print '%e %f'%(M[i],bias[i])
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
M_HI=np.zeros(bins,dtype=np.float64); f=open(f_out_cross_section,'w')
for i in xrange(bins):
    #M[i]=1e12*h
    [cross_section[i],M_HI[i]]=HIL.cross_section_BH(Omega_m,Omega_b,Omega_l,
                                                    redshift,h,M[i],c0,f_HI,
                                                    alpha_e,V0c,verbose=False)

    #cross_section_check=sigma_DLA(M[i])
    #cross_section[i]=cross_section_model(M[i],redshift)

    print 'M = %e Msun/h ----> sigma = %e (kpc/h)^2'%(M[i],cross_section[i])    
    #print 'M = %e Msun/h ----> sigma = %e (kpc/h)^2'%(M[i],cross_section_check)
    f.write(str(M[i])+' '+str(cross_section[i])+'\n')
f.close()

#take only the points where sigma>0.0
#indexes=np.where(cross_section>0.0)[0]
#dndM=dndM[indexes]; cross_section=cross_section[indexes]
#bias=bias[indexes]; M=M[indexes]; del indexes

#compute the DLA incidence rate
print 'dN/dX = %f'%DLA_incidence_rate(M,dndM,cross_section)

#compute the DLA bias
print 'b_D=',DLA_bias(M,bias,dndM,cross_section)
#sys.exit()

"""
N_HI=10**20
f_HI_BH=column_density_BH(N_HI,M,dndM,cross_section,Omega_m,Omega_b,
                          Omega_l,redshift,h,c0,f_HI,alpha_e,V0c)
print '%e %e'%(N_HI,f_HI_BH)
sys.exit()
"""

#compute the column density distribution
bins=10; N_HI=np.logspace(20.0,22.3,bins); f=open(f_out_f_HI,'w')
for i in xrange(bins):
    f_HI_BH=column_density_BH(N_HI[i],M,dndM,cross_section,Omega_m,Omega_b,
                              Omega_l,redshift,h,c0,f_HI,alpha_e,V0c)
    print np.log10(N_HI[i]),np.log10(f_HI_BH)
    f.write(str(np.log10(N_HI[i]))+' '+str(np.log10(f_HI_BH))+'\n')
f.close()

