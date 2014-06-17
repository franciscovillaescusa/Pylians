#This code can be used to compute the CDM, baryon and neutrinos power spectra
#from an IC N-body file. It also computes the CDM-baryon, CDM-neutrinos and
#baryon-neutrinos cross-power spectra and the overall matter power spectrum.
#The code also computes the above spectra from the CAMB matter power spectrum
#and transfer files.

#By default the code will compute all the above power spectra. If the N-body
#does not content a given particle type (e.g. neutrinos) the generated file
#will contain nan values.



import numpy as np
import readsnap
import CIC_library as CIC
import Power_spectrum_library as PSL
import sys


rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
################################## INPUT ######################################
####### CAMB #######
f_Pk_DM='./CAMB_TABLES/ics_matterpow_99.dat'
f_transfer='./CAMB_TABLES/ics_transfer_99.dat'

f_camb_c  = 'CDM_Pk_CAMB.dat'
f_camb_b  = 'B_Pk_CAMB.dat'
f_camb_n  = 'NU_Pk_CAMB.dat'
f_camb_cb = 'CDMB_Pk_CAMB.dat'
f_camb_cn = 'CDMNU_Pk_CAMB.dat'
f_camb_bn = 'BNU_Pk_CAMB.dat'

####### IC file #######
snapshot_fname='ics'
dims=512

f_out_c  = 'Pk_CDM_z=99.dat'
f_out_b  = 'Pk_B_z=99.dat'
f_out_n  = 'Pk_NU_z=99.dat'
f_out_cb = 'Pk_CDMB_z=99.dat'
f_out_cn = 'Pk_CDMNU_z=99.dat'
f_out_bn = 'Pk_BNU_z=99.dat'
f_out_m  = 'Pk_matter_z=99.dat'
###############################################################################

"""
#This is to read the positions of the particles in the glass file
pos_1=readsnap.read_block('../N-GenIC/GLASS/dummy_glass_CDM_B_64_64.dat',
                          "POS ",parttype=1) #kpc/h

pos_2=readsnap.read_block('../N-GenIC/GLASS/dummy_glass_CDM_B_64_64.dat',
                          "POS ",parttype=2) #kpc/h

print pos_1
print pos_2
print len(pos_1),len(pos_2)
"""

####################### CAMB ###########################
# read CAMB matter power spectrum file
k_DM,Pk_DM=np.loadtxt(f_Pk_DM,unpack=True)

# read CAMB transfer function file
k,Tcdm,Tb,dumb,dumb,Tnu,Tm=np.loadtxt(f_transfer,unpack=True)

#DM P(k)
Pk_DM=10**(np.interp(np.log10(k),np.log10(k_DM),np.log10(Pk_DM)))

#compute the different P(k)
Pk_C  = Pk_DM*(Tcdm/Tm)**2;   np.savetxt(f_camb_c ,np.transpose([k,Pk_C]))
Pk_B  = Pk_DM*(Tb/Tm)**2;     np.savetxt(f_camb_b ,np.transpose([k,Pk_B]))
Pk_N  = Pk_DM*(Tnu/Tm)**2;    np.savetxt(f_camb_n ,np.transpose([k,Pk_N]))
Pk_CB = Pk_DM*Tcdm*Tb/Tm**2;  np.savetxt(f_camb_cb,np.transpose([k,Pk_CB]))
Pk_CN = Pk_DM*Tcdm*Tnu/Tm**2; np.savetxt(f_camb_cn,np.transpose([k,Pk_CN]))
Pk_BN = Pk_DM*Tb*Tnu/Tm**2;   np.savetxt(f_camb_bn,np.transpose([k,Pk_BN]))



####################### IC file ###########################
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

#compute total number of particles
Ntotal=np.sum(Nall,dtype=np.int64)
print 'Total number of particles in the simulation =',Ntotal

#compute the values of Omega_CDM and Omega_B
Omega_c = Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b = Nall[0]*Masses[0]/BoxSize**3/rho_crit
Omega_n = Nall[2]*Masses[2]/BoxSize**3/rho_crit
print '\nOmega_CDM = %.4f\nOmega_B   = %0.4f\nOmega_NU  = %.4f'\
    %(Omega_c,Omega_b,Omega_n)
print 'Omega_m   = %.4f\n'%Omega_m



######################################
#first: compute auto-power spectra
######################################

pos   = [[],[],[]] #define the array that hosts the CDM, B and NU positions
M     = [[],[],[]] #define the array that hosts the CDM, B and NU masses
delta = [[],[],[]] #define the array that hosts the CDM, B and NU deltas

for ptype in [0,1,2]:

    #positions in #Mpc/h
    pos[ptype]=readsnap.read_block(snapshot_fname,"POS ",parttype=ptype)/1e3 
    M[ptype]=np.empty(len(pos[ptype]),dtype=np.float32)
    M[ptype][:]=Masses[ptype]

    if ptype==0:
        print 'Omega_B   = %.4f'\
            %(np.sum(M[ptype],dtype=np.float64)/BoxSize**3/rho_crit)
        f_out=f_out_b
    elif ptype==1:
        print 'Omega_CDM = %.4f'\
            %(np.sum(M[ptype],dtype=np.float64)/BoxSize**3/rho_crit) 
        f_out=f_out_c
    elif ptype==2:
        print 'Omega_NU  = %.4f'\
            %(np.sum(M[ptype],dtype=np.float64)/BoxSize**3/rho_crit)
        f_out=f_out_n    

    #compute the mean mass per grid cell
    mean_M=np.sum(M[ptype],dtype=np.float64)/dims**3

    #compute the mass within each grid cell
    delta[ptype]=np.zeros(dims**3,dtype=np.float32)
    CIC.CIC_serial(pos[ptype],dims,BoxSize,delta[ptype],M[ptype])
    print '%.6e should be equal to \n%.6e\n'\
     %(np.sum(M[ptype],dtype=np.float64),np.sum(delta[ptype],dtype=np.float64))

    #compute the density constrast within each grid cell
    delta[ptype]/=mean_M; delta[ptype]-=1.0
    print '%.3e < delta < %.3e\n'%(np.min(delta[ptype]),np.max(delta[ptype]))

    #compute the P(k)
    Pk=PSL.power_spectrum_given_delta(delta[ptype],dims,BoxSize)

    #write P(k) to output file
    np.savetxt(f_out,np.transpose([Pk[0],Pk[1]])); print '\n\n'


#######################################
#second: compute cross-power spectra
#######################################
for ptype1 in [0,1,2]:
    for ptype2 in xrange(ptype1+1,3):

        #compute the cross-P(k)
        print '\ncomputing cross-P(k) of fields %d and %d'%(ptype1,ptype2)
        Pk=PSL.cross_power_spectrum_given_delta(delta[ptype1],delta[ptype2],
                                                dims,BoxSize)
        if ptype1==0:
            if ptype2==1:
                f_out=f_out_cb
            elif ptype2==2:
                f_out=f_out_bn
        elif ptype1==1:
            if ptype2==2:
                f_out=f_out_cn

        #write P(k) to output file
        np.savetxt(f_out,np.transpose([Pk[0],Pk[1]]))

    delta[ptype1]=[]



####################################################
#third: compute total matter auto-power spectrum   
####################################################

print '/ncomputing total matter P(k)'
pos=np.vstack([pos[0],pos[1],pos[2]])
M=np.hstack([M[0],M[1],M[2]])

#compute the mean mass per grid cell
mean_M=np.sum(M,dtype=np.float64)/dims**3

#compute the mass within each grid cell
delta=np.zeros(dims**3,dtype=np.float32)
CIC.CIC_serial(pos,dims,BoxSize,delta,M); del pos
print '%.6e should be equal to \n%.6e\n'\
    %(np.sum(M,dtype=np.float64),np.sum(delta,dtype=np.float64)); del M

#compute the density constrast within each grid cell
delta/=mean_M; delta-=1.0
print '%.3e < delta < %.3e\n'%(np.min(delta),np.max(delta))

#compute the P(k)
Pk=PSL.power_spectrum_given_delta(delta,dims,BoxSize)

#write P(k) to output file
np.savetxt(f_out_m,np.transpose([Pk[0],Pk[1]]))




