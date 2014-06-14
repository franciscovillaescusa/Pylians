#This code can be used to compute the power spectrum of the CDM and baryons
#in hydro simulations (simulations containing only CDM and baryons).
#It computes the CDM and baryon power spectrum, together with their 
#cross-power spectrum and the total matter P(k). Moreover, this code reads
#the CAMB matter power spectrum and the transfer function and computes the
#CDM, baryon and CDM-baryons (cross-)power spectra.


import numpy as np
import readsnap
import CIC_library as CIC
import Power_spectrum_library as PSL
import sys


rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
################################## INPUT ######################################
####### CAMB #######
f_Pk_DM='../CAMB_TABLES/ics_matterpower_z=99.dat'
f_transfer='../CAMB_TABLES/ics_transfer_out_z=99.dat'

f_camb_c='CDM_Pk_CAMB.dat'
f_camb_b='B_Pk_CAMB.dat'
f_camb_cb='CDMB_Pk_CAMB.dat'

####### IC file #######
snapshot_fname='ics'
dims=512

f_out_c='Pk_CDM_z=99.dat'
f_out_b='Pk_B_z=99.dat'
f_out_cb='Pk_CDMB_z=99.dat'
f_out_m='Pk_matter_z=99.dat'
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
Pk_CDM=Pk_DM*(Tcdm/Tm)**2
Pk_B=Pk_DM*(Tb/Tm)**2
Pk_CDMB=Pk_DM*Tb*Tcdm/Tm**2

np.savetxt(f_camb_c,np.transpose([k,Pk_CDM]))
np.savetxt(f_camb_b,np.transpose([k,Pk_B]))
np.savetxt(f_camb_cb,np.transpose([k,Pk_CDMB]))




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
Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b=Omega_m-Omega_cdm
print '\nOmega_CDM = %.3f\nOmega_B   = %0.3f\nOmega_M   = %.3f\n'\
    %(Omega_cdm,Omega_b,Omega_m)


############### CDM ##################
pos_c=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
M_c=np.empty(len(pos_c),dtype=np.float32); M_c[:]=Masses[1]

print 'Omega_CDM = ',np.sum(M_c,dtype=np.float64)/BoxSize**3/rho_crit

#compute the mean mass per grid cell
mean_M_c=np.sum(M_c,dtype=np.float64)/dims**3

#compute the mass within each grid cell
delta_c=np.zeros(dims**3,dtype=np.float32)
CIC.CIC_serial(pos_c,dims,BoxSize,delta_c,M_c)
print '%.6e should be equal to \n%.6e\n'\
    %(np.sum(M_c,dtype=np.float64),np.sum(delta_c,dtype=np.float64))

#compute the density constrast within each grid cell
delta_c/=mean_M_c; delta_c-=1.0
print '%.3e < delta < %.3e\n'%(np.min(delta_c),np.max(delta_c))

#compute the P(k)
Pk=PSL.power_spectrum_given_delta(delta_c,dims,BoxSize)

#write P(k) to output file
np.savetxt(f_out_c,np.transpose([Pk[0],Pk[1]]))

############### Baryons ##################
pos_b=readsnap.read_block(snapshot_fname,"POS ",parttype=0)/1e3 #Mpc/h
M_b=np.empty(len(pos_b),dtype=np.float32); M_b[:]=Masses[0]

#pos_b-=0.5*BoxSize/512.0
#indexes=np.where(pos_b<0.0); pos_b[indexes]+=BoxSize; del indexes
#indexes=np.where(pos_b>BoxSize); pos_b[indexes]-=BoxSize; del indexes

print 'Omega_B   = ',np.sum(M_b,dtype=np.float64)/BoxSize**3/rho_crit

#compute the mean mass per grid cell
mean_M_b=np.sum(M_b,dtype=np.float64)/dims**3

#compute the mass within each grid cell
delta_b=np.zeros(dims**3,dtype=np.float32)
CIC.CIC_serial(pos_b,dims,BoxSize,delta_b,M_b)
print '%.6e should be equal to \n%.6e\n'\
    %(np.sum(M_b,dtype=np.float64),np.sum(delta_b,dtype=np.float64))

#compute the density constrast within each grid cell
delta_b/=mean_M_b; delta_b-=1.0
print '%.3e < delta < %.3e\n'%(np.min(delta_b),np.max(delta_b))

#compute the P(k)
Pk=PSL.power_spectrum_given_delta(delta_b,dims,BoxSize)

#write P(k) to output file
np.savetxt(f_out_b,np.transpose([Pk[0],Pk[1]]))

############### CDM-Baryons ##################
#compute the P(k)
Pk=PSL.cross_power_spectrum_given_delta(delta_c,delta_b,dims,BoxSize)
del delta_c,delta_b

#write P(k) to output file
np.savetxt(f_out_cb,np.transpose([Pk[0],Pk[1]]))

############### matter ##################
pos=np.vstack([pos_c,pos_b]); del pos_c, pos_b
M=np.hstack([M_c,M_b]); del M_c,M_b

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




