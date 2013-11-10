#This file contains the routines needed to compute the bias for:
#1) halos and DM PS files are provided
#2) halos-DM and DM PS files are provided
#3) Theoretical bias from Tinker and SMT01

import numpy as np
import scipy.integrate as si
import sys
import Power_spectrum_library as PSL
import mass_function_library as MFL

############################ CONSTANTS ############################
pi=np.pi
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
deltac=1.686
###################################################################

##############################################################################

#This functions computes the b(M)
def bias(k,Pk,OmegaM,M,author):
    rhoM=rho_crit*OmegaM

    R=(3.0*M/(4.0*pi*rhoM))**(1.0/3.0)

    if author=='SMT01':
        a=0.707; b=0.5; c=0.6
        anu=a*(deltac/MFL.sigma(k,Pk,R))**2
        
        bias=np.sqrt(a)*anu+np.sqrt(a)*b*anu**(1.0-c)
        bias=bias-anu**c/(anu**c+b*(1.0-c)*(1.0-0.5*c))
        bias=1.0+bias/(np.sqrt(a)*deltac)

    if author=='Tinker':
        Delta=200.0
        y=np.log10(Delta)

        A=1.0+0.24*y*np.exp(-(4.0/y)**4);          a=0.44*y-0.88
        B=0.183;                                   b=1.5
        C=0.019+0.107*y+0.19*np.exp(-(4.0/y)**4);  c=2.4
        
        nu=deltac/MFL.sigma(k,Pk,R)

        bias=1.0-A*nu**a/(nu**a+deltac**a)+B*nu**b+C*nu**c

    return bias

##############################################################################

#This function returns the effective bias, see eq. 15 of Marulli 2011
def bias_eff(k,Pk,OmegaM,z,M1,M2,author,bins=100):

    M=np.logspace(np.log10(M1),np.log10(M2),bins+1) #bins in mass
    M_middle=10**(0.5*(np.log10(M[1:])+np.log10(M[:-1]))) #center of the bin
    deltaM=M[1:]-M[:-1] #size of the bin
    
    if author=='SMT01':
        dndM=MFL.ST_mass_function(k,Pk,OmegaM,None,None,None,M_middle)[1]

    if author=='Tinker':
        dndM=MFL.Tinker_mass_function(k,Pk,OmegaM,z,None,None,None,M_middle)[1]
    
    b=np.empty(bins,dtype=np.float64)
    for i in range(bins):
        b[i]=bias(k,Pk,OmegaM,M_middle[i],author)

    bias_eff=np.sum(dndM*deltaM*b)/np.sum(dndM*deltaM)

    return bias_eff

###############################################################################
#This routine computes the efective bias
#Author-------> Choose between 'Tinker' or 'SMT01'
#Mmin---------> Value of the minimum mass in the halo catalogue
#Mmax---------> Value of the maximum mass in the halo catalogue
#f_Pk_DM------> This is the CAMB linear DM Pk file at the desired redshift
#f_transfer---> This is the CAMB transfer function at the desired redshift
#Omega_CDM----> Omega_CDM of the simulation (without the baryons contributions!)
#Omega_B------> Omega_B of the simulation
#do_DM--------> if the total matter Pk is wanted, then set this to True

def halo_bias(Author,z,Mmin,Mmax,f_Pk_DM,f_transfer,
              Omega_CDM,Omega_B,do_DM=False):

    #if using the total DM P(k) read the CAMB file
    if do_DM:
        f=open(f_Pk_DM,'r'); k,Pk=[],[]
        for line in f.readlines():
            a=line.split()
            k.append(float(a[0])); Pk.append(float(a[1]))
        f.close(); k=np.array(k); Pk=np.array(Pk)

    #if not, compute the CDM P(k) through the DM P(k) and the transfer function
    else:
        [k,Pk]=CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)

    #compute the effective halo bias
    print Author,bias_eff(k,Pk,Omega_CDM+Omega_B,z,Mmin,Mmax,Author,100)

##############################################################################

# This routine just read the DM P(k) from the CAMB output and return it

def DM_Pk(f_Pk_DM):

    # read CAMB matter power spectrum file
    f=open(f_Pk_DM,'r'); k_DM,Pk_DM=[],[]
    for line in f.readlines():
        a=line.split()
        k_DM.append(float(a[0])); Pk_DM.append(float(a[1]))
    f.close(); k_DM=np.array(k_DM); Pk_DM=np.array(Pk_DM)

    return [k_DM,Pk_DM]

##############################################################################

# This routine computes the CDM P(k) from the total DM P(k) by using the 
# transfer function. It also requires the values of Omega_CDM and Omega_B

def CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B):

    # read CAMB matter power spectrum file
    f=open(f_Pk_DM,'r'); k_DM,Pk_DM=[],[]
    for line in f.readlines():
        a=line.split()
        k_DM.append(float(a[0])); Pk_DM.append(float(a[1]))
    f.close(); k_DM=np.array(k_DM); Pk_DM=np.array(Pk_DM)

    # read CAMB transfer function file
    f=open(f_transfer,'r'); k,Tcdm,Tb,Tnu,Tm=[],[],[],[],[]
    for line in f.readlines():
        a=line.split()
        k.append(float(a[0]))
        Tcdm.append(float(a[1]))
        Tb.append(float(a[2]))
        Tnu.append(float(a[5]))
        Tm.append(float(a[6]))
    f.close(); k=np.array(k); Tcdm=np.array(Tcdm); Tb=np.array(Tb)    
    Tnu=np.array(Tnu); Tm=np.array(Tm)

    # DM P(k)
    Pk_DM=np.interp(np.log10(k),np.log10(k_DM),np.log10(Pk_DM))
    Pk_DM=10**(Pk_DM)

    # CDM P(k)
    transfer_CDM=(Tcdm*Omega_CDM+Tb*Omega_B)/(Omega_CDM+Omega_B)
    Pk_CDM=Pk_DM*(transfer_CDM/Tm)**2

    return [k,Pk_CDM]

##############################################################################

# This routine computes the NU P(k) from the total DM P(k) by using the 
# transfer function. 

def NU_Pk(f_Pk_DM,f_transfer):

    # read CAMB matter power spectrum file
    f=open(f_Pk_DM,'r'); k_DM,Pk_DM=[],[]
    for line in f.readlines():
        a=line.split()
        k_DM.append(float(a[0])); Pk_DM.append(float(a[1]))
    f.close(); k_DM=np.array(k_DM); Pk_DM=np.array(Pk_DM)

    # read CAMB transfer function file
    f=open(f_transfer,'r'); k,Tcdm,Tb,Tnu,Tm=[],[],[],[],[]
    for line in f.readlines():
        a=line.split()
        k.append(float(a[0]))
        Tcdm.append(float(a[1]))
        Tb.append(float(a[2]))
        Tnu.append(float(a[5]))
        Tm.append(float(a[6]))
    f.close(); k=np.array(k); Tcdm=np.array(Tcdm); Tb=np.array(Tb)    
    Tnu=np.array(Tnu); Tm=np.array(Tm)

    # DM P(k)
    Pk_DM=np.interp(np.log10(k),np.log10(k_DM),np.log10(Pk_DM))
    Pk_DM=10**(Pk_DM)

    # NU P(k)
    Pk_NU=Pk_DM*(Tnu/Tm)**2

    return [k,Pk_NU]

##############################################################################

# This routine computes the CDM-NU cross-P(k) from the total DM P(k)
# by using the transfer function. 
# It requires the values of Omega_CDM and Omega_B

def CDM_NU_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B):

    # read CAMB matter power spectrum file
    f=open(f_Pk_DM,'r'); k_DM,Pk_DM=[],[]
    for line in f.readlines():
        a=line.split()
        k_DM.append(float(a[0])); Pk_DM.append(float(a[1]))
    f.close(); k_DM=np.array(k_DM); Pk_DM=np.array(Pk_DM)

    # read CAMB transfer function file
    f=open(f_transfer,'r'); k,Tcdm,Tb,Tnu,Tm=[],[],[],[],[]
    for line in f.readlines():
        a=line.split()
        k.append(float(a[0]))
        Tcdm.append(float(a[1]))
        Tb.append(float(a[2]))
        Tnu.append(float(a[5]))
        Tm.append(float(a[6]))
    f.close(); k=np.array(k); Tcdm=np.array(Tcdm); Tb=np.array(Tb)    
    Tnu=np.array(Tnu); Tm=np.array(Tm)

    # DM P(k)
    Pk_DM=np.interp(np.log10(k),np.log10(k_DM),np.log10(Pk_DM))
    Pk_DM=10**(Pk_DM)

    # CDM P(k)
    transfer_CDM=(Tcdm*Omega_CDM+Tb*Omega_B)/(Omega_CDM+Omega_B)
    Pk_CDM=Pk_DM*(transfer_CDM*Tnu/Tm**2)

    return [k,Pk_CDM]

##############################################################################







#This function computes the bias between two files, one containing the DM PS
#and the other containing the halos PS or the halos-DM PS
#file1----> file containing the halos or the halos-DM power spectrum
#file2----> file containing the DM power spectrum
#f_out----> file with the bias to be written
#cross_spectrum----> if file1 is the halos PS, set to False, otherwise set True
def bias_from_files(file1,file2,f_out,cross_spectrum=False):
    
    #read halos (or halos-DM) (cross-)power spectrum file
    k,Pk=[],[]; f=open(file1,'r')
    for line in f.readlines():
        a=line.split()
        k.append(float(a[0])); Pk.append(float(a[1]))
    f.close(); k=np.array(k); Pk=np.array(Pk)

    #read DM power spectrum file
    k_DM,Pk_DM=[],[]; f=open(file2,'r')
    for line in f.readlines():
        a=line.split()
        k_DM.append(float(a[0])); Pk_DM.append(float(a[1]))
    f.close(); k_DM=np.array(k_DM); Pk_DM=np.array(Pk_DM)
        
    #check that k-arrays are the same
    if np.any(k!=k_DM):
        print 'k arrays are different'
        sys.exit()

    if cross_spectrum:
        b=Pk/Pk_DM
    else:
        b=np.sqrt(Pk/Pk_DM)
        
    #write output file
    f=open(f_out,'w')
    for i in range(len(k_DM)):
        f.write(str(k_DM[i])+' '+str(b[i])+'\n')
    f.close()


#This function computes the bias between the DM-halos PS and the DM PS
#It also computes the error on the bias as a function of k
#file1----> file containing DM-halos power spectrum
#file2----> file containing the DM power spectrum
#f_out----> file with the bias to be written
#dims-----> number of points in the grid per dimension used to compute the PS
#n_halos--> number density of the halos used to compute the cross-PS
def bias_from_files_cross(file1,file2,f_out,dims,n_halos):
    
    #read the DM-halos cross-power spectrum file
    k_cross,Pk_cross=[],[]; f=open(file1,'r')
    for line in f.readlines():
        a=line.split()
        k_cross.append(float(a[0])); Pk_cross.append(float(a[1]))
    f.close(); k_cross=np.array(k_cross); Pk_cross=np.array(Pk_cross)

    #read the DM power spectrum file
    k_DM,Pk_DM=[],[]; f=open(file2,'r')
    for line in f.readlines():
        a=line.split()
        k_DM.append(float(a[0])); Pk_DM.append(float(a[1]))
    f.close(); k_DM=np.array(k_DM); Pk_DM=np.array(Pk_DM)
        
    #check that k-arrays are the same
    if np.any(k_cross!=k_DM):
        print 'k arrays are different'
        sys.exit()

    #compute the bias
    b=Pk_cross/Pk_DM

    #count the number of modes in the k-bins
    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1
    [array,k]=PSL.CIC_correction(dims)
    count=PSL.lin_histogram(bins_r,0.0,bins_r*1.0,k)[1:]    

    #compute the error on the bias
    db=1.0/np.sqrt(n_halos*count*Pk_DM)
        
    #write output file
    f=open(f_out,'w')
    for i in range(len(k_DM)):
        f.write(str(k_DM[i])+' '+str(b[i])+' '+str(db[i])+'\n')
    f.close()










    




################################# USAGE ######################################

##### bias from files #####
"""
#file1 ------> halos PS file
#file2 ------> DM PS file
#f_out ------> file to write with the bias

file1='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/Pk_halos_0.0_z=0.dat' 
file2='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/Pk_DM_0.0_z=0.dat'
f_out='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/bias_halos_0.0_z=0.dat'

bias_from_files(file1,file2,f_out,cross_spectrum=False)

#file1 -----> halos-DM PS file
#file2 ------> DM PS file
#f_out ------> file to write with the bias

file1='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/Pk_DM-halos_0.0_z=0.dat'
file2='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/Pk_DM_0.0_z=0.dat'
f_out='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/bias_DM-halos_0.0_z=0.dat'

bias_from_files(file1,file2,f_out,cross_spectrum=True)
"""

##### DM-halos cross bias from files ####
"""
file1='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/Pk_DM-halos_0.0_z=0.dat'
file2='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/Pk_DM_0.0_z=0.dat'
f_out='/home/villa/disksom2/1000Mpc_z=99/CDM/Pk/bias_DM-halos_0.0_z=0.dat'
dims=512
n_halos=190950/1000.0**3 #in (h/Mpc)^3

bias_from_files_cross(file1,file2,f_out,dims,n_halos)
"""

##### halo bias #####
"""
Omega_DM=0.2708
Omega_NU=0.006573383*0.0 #*0--0.0eV  *1--0.3eV   *2--0.6eV
Omega_B=0.05

z=0

Author='Tinker' #'SMT01' or 'Tinker'
reds=['0']   #['0','0.5','1','2']

Mmin=2.0e13
Mmax=3.0e15

Omega_CDM=Omega_DM-Omega_NU-Omega_B
f_Pk_DM='/home/villa/disksom2/1000Mpc_z=99/CDM/CAMB_TABLES/ics_matterpow_0.dat'
f_transfer='/home/villa/disksom2/1000Mpc_z=99/CDM/CAMB_TABLES/ics_transfer_0.dat'

halo_bias(Author,z,Mmin,Mmax,f_Pk_DM,f_transfer,Omega_CDM,Omega_B,do_DM=False)
"""

##### read the DM P(k) from CAMB output #####
"""
f_Pk_DM='/home/villa/disksom2/1000Mpc_z=99/CDM/CAMB_TABLES/ics_matterpow_0.dat'

[k,Pk]=DM_Pk(f_Pk_DM)

f=open('borrar.dat','w')
for i in range(len(k)):
    f.write(str(k[i])+' '+str(Pk[i])+'\n')
f.close()
"""

##### compute CDM P(k) from DM P(k) + transfer function #####
"""
f_Pk_DM='/home/villa/disksom2/1000Mpc_z=99/CDM/CAMB_TABLES/ics_matterpow_0.dat'
f_transfer='/home/villa/disksom2/1000Mpc_z=99/CDM/CAMB_TABLES/ics_transfer_0.dat'
Omega_CDM=0.2208
Omega_B=0.05

[k,Pk]=CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)

f=open('borrar.dat','w')
for i in range(len(k)):
    f.write(str(k[i])+' '+str(Pk[i])+'\n')
f.close()
"""

##### compute NU P(k) from DM P(k) + transfer function #####
"""
f_Pk_DM='/home/villa/disksom2/1000Mpc_z=99/CDM/CAMB_TABLES/ics_matterpow_0.dat'
f_transfer='/home/villa/disksom2/1000Mpc_z=99/CDM/CAMB_TABLES/ics_transfer_0.dat'

[k,Pk]=NU_Pk(f_Pk_DM,f_transfer)

f=open('borrar.dat','w')
for i in range(len(k)):
    f.write(str(k[i])+' '+str(Pk[i])+'\n')
f.close()
"""

