import numpy as np
import sys

############## INPUT ################
f_CDM='TPCF_CDM_0.3_z=2.dat'
f_NU='TPCF_NU_0.3_z=2.dat'
f_CDM_NU='TPCF_CDM-NU_0.3_z=2.dat'
f_DM='TPCF_DM_0.3_z=2.dat'

Omega_CDM=0.2642266
Omega_NU=0.006573383
#####################################
Omega_DM=Omega_CDM+Omega_NU


f=open(f_CDM,'r')
r,xi_CDM=[],[]
for line in f.readlines():
    a=line.split()
    r.append(float(a[0]))
    xi_CDM.append(float(a[1]))
f.close()
r=np.array(r); xi_CDM=np.array(xi_CDM)

f=open(f_NU,'r')
xi_NU=[]
for line in f.readlines():
    a=line.split()
    xi_NU.append(float(a[1]))
f.close()
xi_NU=np.array(xi_NU)

f=open(f_CDM_NU,'r')
xi_CDM_NU=[]
for line in f.readlines():
    a=line.split()
    xi_CDM_NU.append(float(a[1]))
f.close()
xi_CDM_NU=np.array(xi_CDM_NU)


xi_DM=(Omega_CDM/Omega_DM)**2*xi_CDM+(Omega_NU/Omega_DM)**2*xi_NU+2.0*Omega_CDM*Omega_NU/Omega_DM**2*xi_CDM_NU
f=open(f_DM,'w')
for i in range(len(r)):
    f.write(str(r[i])+' '+str(xi_DM[i])+'\n')
f.close()
