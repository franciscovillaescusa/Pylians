#This code can be used to compute the CDM, baryon and neutrinos power spectra
#from an IC N-body file. It also computes the CDM-baryon, CDM-neutrinos and
#baryon-neutrinos cross-power spectra and the overall matter power spectrum.

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
snapshot_fname='ics'
dims=512

Omega_c = 0.2685
Omega_b = 0.049
Omega_n = 0.0
z       = 99

particle_type = [0,1,2]

f_out_c  = 'Pk_CDM_z='    +str(z)+ '.dat'
f_out_b  = 'Pk_B_z='      +str(z)+ '.dat'
f_out_n  = 'Pk_NU_z='     +str(z)+ '.dat'
f_out_cb = 'Pk_CDMB_z='   +str(z)+ '.dat'
f_out_cn = 'Pk_CDMNU_z='  +str(z)+ '.dat'
f_out_bn = 'Pk_BNU_z='    +str(z)+ '.dat'
f_out_m  = 'Pk_matter_z=' +str(z)+ '.dat'
###############################################################################

#some verbose
Omega_m = Omega_c + Omega_b + Omega_n
print '\nValue of the cosmological parameters: (check!!!!)'
print 'Omega_CDM = ',Omega_c
print 'Omega_B   = ',Omega_b
print 'Omega_NU  = ',Omega_n,'\n'

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

#define the arrays containing the positions and deltas and power spectra
delta = [[],[],[]]   #array  containing the CDM, B and NU deltas
Pk    = [[[],[],[]], #matrix containing the auto- and cross-power spectra
         [[],[],[]],
         [[],[],[]]] 


#do a loop over all particle types and compute the deltas
for ptype in particle_type:
    
    #read particle positions in #Mpc/h
    pos=readsnap.read_block(snapshot_fname,"POS ",parttype=ptype)/1e3 

    #compute the deltas
    delta[ptype]=np.zeros(dims**3,dtype=np.float32)
    CIC.CIC_serial(pos,dims,BoxSize,delta[ptype])
    print '%.6e should be equal to \n%.6e\n'\
        %(len(pos),np.sum(delta[ptype],dtype=np.float64)); del pos

    #compute the density constrast within each grid cell
    delta[ptype]=delta[ptype]*1.0/len(pos)-1.0;
    print '%.3e < delta < %.3e\n'%(np.min(delta[ptype]),np.max(delta[ptype]))
                                   

#compute the auto-power spectrum when there is only one component
if len(particle_type)==1:

    ptype=particle_type[0]
    if ptype==0:   f_out=f_out_b
    elif ptype==1: f_out=f_out_c
    elif ptype==2: f_out=f_out_n
    print '\nComputing the power spectrum of the particle type: ',ptype
    data=PSL.power_spectrum_given_delta(delta[ptype],dims,BoxSize)
    k=data[0]; Pk[ptype][ptype]=data[1]; del data
    np.savetxt(f_out,np.transpose([k,Pk[ptype][ptype]])); print '\n'


#if there are two or more particles compute auto- and cross-power spectra
for ptype1 in particle_type:
    for ptype2 in particle_type[ptype1+1:]:

        #choose the name of the output files
        if ptype1==0:
            f_out1 = f_out_b
            if ptype2==1:
                f_out2  = f_out_c
                f_out12 = f_out_cb
            elif ptype2==2:
                f_out2  = f_out_n
                f_out12 = f_out_bn

        elif ptype1==1:
            f_out1 = f_out_c
            if ptype2==2:
                f_out2  = f_out_n 
                f_out12 = f_out_cn

        #some verbose
        print '\nComputing the auto- and cross-power spectra of types: '\
            ,ptype1,'-',ptype2
        print 'saving results in:'; print f_out1; print f_out2; print f_out12

        #This routine computes the auto- and cross-power spectra
        data=PSL.cross_power_spectrum_given_delta(delta[ptype1],delta[ptype2],
                                                  dims,BoxSize)

        k                  = data[0]
        Pk[ptype1][ptype2] = data[1];   Pk[ptype2][ptype1] = data[1]; 
        Pk[ptype1][ptype1] = data[2]
        Pk[ptype2][ptype2] = data[3]

        #save power spectra results in the output files
        np.savetxt(f_out1,  np.transpose([k,Pk[ptype1][ptype1]]))
        np.savetxt(f_out2,  np.transpose([k,Pk[ptype2][ptype2]]))
        np.savetxt(f_out12, np.transpose([k,Pk[ptype1][ptype2]]))



#compute total matter auto-power spectrum   
print '\ncomputing total matter P(k)'
Pk_m = np.zeros(len(k),dtype=np.float64)

for ptype1 in particle_type:
    for ptype2 in particle_type:

        if ptype1==0:  factor1 = Omega_b
        if ptype1==1:  factor1 = Omega_c
        if ptype1==2:  factor1 = Omega_n

        if ptype2==0:  factor2 = Omega_b
        if ptype2==1:  factor2 = Omega_c
        if ptype2==2:  factor2 = Omega_n
        
        Pk_m += factor1*factor2 * Pk[ptype1][ptype2]

Pk_m /= Omega_m**2
np.savetxt(f_out_m,np.transpose([k,Pk_m])) #write results to output file




