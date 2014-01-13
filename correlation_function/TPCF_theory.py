import numpy as np
import scipy.integrate as si
import bias_library as BL
import sys


############################ CONSTANTS ############################
pi=np.pi
###################################################################

#derivative function for integrating sigma(R)
def deriv(y,x,k,Pk,R):
    Pkp=np.interp(np.log10(x),np.log10(k),np.log10(Pk)); Pkp=10**Pkp
    kR=x*R
    return np.array([x**2*Pkp*np.sin(kR)/kR])

def correlation_function_from_Pk(k,Pk,R):
    yinit=np.array([0.0])
    k_limits=np.array([k[0],k[-1]])    

    I=si.odeint(deriv,yinit,k_limits,args=(k,Pk,R),
                rtol=1e-7,atol=1e-7,
                mxstep=10000000)[1][0]/(2.0*pi**2)

    return I


################################# INPUT ######################################
f_Pk_DM='../../CAMB_TABLES/ics_matterpow_0.dat'
f_transfer='../../CAMB_TABLES/ics_transfer_0.dat'

Rmin=0.1
Rmax=100.0
bins=100

Omega_CDM=0.207653233885351
Omega_B=0.05


##### z=0 #####

#DM
[k,Pk]=BL.DM_Pk(f_Pk_DM)

f_out='CF_DM_z=0.dat'

f=open(f_out,'w')
for i in range(bins+1):
    print i
    R=10**(np.log10(Rmin)+np.log10(Rmax/Rmin)*i*1.0/bins)
    f.write(str(R)+' '+str(correlation_function_from_Pk(k,Pk,R))+'\n')
f.close()
"""
#CDM
[k,Pk]=BL.CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)

f_out='CF_CDM_z=1.dat'

f=open(f_out,'w')
for i in range(bins+1):
    print i
    R=10**(np.log10(Rmin)+np.log10(Rmax/Rmin)*i*1.0/bins)
    f.write(str(R)+' '+str(correlation_function_from_Pk(k,Pk,R))+'\n')
f.close()

#NU
[k,Pk]=BL.NU_Pk(f_Pk_DM,f_transfer)

f_out='CF_NU_z=1.dat'

f=open(f_out,'w')
for i in range(bins+1):
    print i
    R=10**(np.log10(Rmin)+np.log10(Rmax/Rmin)*i*1.0/bins)
    f.write(str(R)+' '+str(correlation_function_from_Pk(k,Pk,R))+'\n')
f.close()

#CDM-NU
[k,Pk]=BL.CDM_NU_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)

f_out='CF_CDM-NU_z=1.dat'

f=open(f_out,'w')
for i in range(bins+1):
    print i
    R=10**(np.log10(Rmin)+np.log10(Rmax/Rmin)*i*1.0/bins)
    f.write(str(R)+' '+str(correlation_function_from_Pk(k,Pk,R))+'\n')
f.close()
"""
