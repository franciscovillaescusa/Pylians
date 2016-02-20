#This code can be used to compute the CDM, baryon and neutrinos power spectra
#from an N-body snapshot file. It also computes the CDM-baryon, CDM-neutrinos 
#and baryon-neutrinos cross-power spectra and the overall matter power spectrum

import numpy as np
import readsnap
import CIC_library as CIC
import Power_spectrum_library as PSL
import redshift_space_library as RSL
import sys


rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3
################################## INPUT ######################################
snapshot_fname = 'ics'
dims           = 512
particle_type  = [0,1,2]
do_RSD         = False
axis           = 0
###############################################################################
dims3 = dims**3

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head     = readsnap.snapshot_header(snapshot_fname)
BoxSize  = head.boxsize/1e3  #Mpc/h
Nall     = head.nall
Masses   = head.massarr*1e10 #Msun/h
Omega_m  = head.omega_m
Omega_l  = head.omega_l
redshift = head.redshift
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #km/s/(Mpc/h)
h        = head.hubble

z = '%.3f'%redshift

#set the label of the output files
root_fout = {'0':'GAS',  '01':'CDMG',  '02':'GNU',    '04':'Gstars',
             '1':'CDM',                '12':'CDMNU',  '14':'CDMStars',
             '2':'NU',                                '24':'NUStars',
             '4':'Stars'                                             } 

#compute the values of Omega_cdm, Omega_nu, Omega_gas and Omega_s
Omega_c = Masses[1]*Nall[1]/BoxSize**3/rho_crit
Omega_n = Masses[2]*Nall[2]/BoxSize**3/rho_crit
Omega_g, Omega_s = 0.0, 0.0
if Nall[0]>0:
    if Masses[0]>0:  
        Omega_g = Masses[0]*Nall[0]/BoxSize**3/rho_crit
        Omega_s = Masses[4]*Nall[4]/BoxSize**3/rho_crit
    else:    
        mass = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h
        Omega_g = np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit; del mass
        mass = readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10 #Msun/h
        Omega_s = np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit; del mass

#some verbose
print 'Omega_gas  = ',Omega_g
print 'Omega_cdm  = ',Omega_c
print 'Omega_nu   = ',Omega_n
print 'Omega_star = ',Omega_s
print 'Omega_m    = ',Omega_m

#define the arrays containing the positions and deltas and power spectra
delta = [[],[],[],[]]     #array  containing the gas, CDM, NU and stars deltas
Pk    = [[[],[],[],[]],   #matrix containing the auto- and cross-power spectra
         [[],[],[],[]],
         [[],[],[],[]],
         [[],[],[],[]]]

#dictionary relating the particle type to the index in the delta and Pk arrays
index_dict = {0:0, 1:1, 2:2, 4:3} #delta of stars (ptype=4) is delta[3] not delta[4]

########################################################################
# do a loop over all particle types and compute the deltas
for ptype in particle_type:
    
    # read particle positions in #Mpc/h
    pos = readsnap.read_block(snapshot_fname,"POS ",parttype=ptype)/1e3 

    # move particle positions to redshift-space
    if do_RSD:
        vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=ptype) #km/s
        RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis);  del vel

    #find the index of the particle type in the delta array
    index = index_dict[ptype]

    #compute the deltas
    delta[index] = np.zeros(dims3,dtype=np.float32)
    CIC.CIC_serial(pos,dims,BoxSize,delta[index])
    print '%.6e should be equal to \n%.6e\n'\
        %(len(pos),np.sum(delta[index],dtype=np.float64))

    #compute the density constrast within each grid cell
    delta[index] = delta[index]*dims3*1.0/len(pos)-1.0;  del pos
    print '%.3e < delta < %.3e\n'%(np.min(delta[index]),np.max(delta[index]))
########################################################################

########################################################################
#compute the auto-power spectrum when there is only one component
if len(particle_type) == 1:

    ptype = particle_type[0];  index = index_dict[ptype]
    fout = 'Pk_'+root_fout[str(ptype)]+'_z='+z+'.dat'
    print '\nComputing the power spectrum of the particle type: ',ptype
    data = PSL.power_spectrum_given_delta(delta[index],dims,BoxSize)
    k = data[0];  Pk[index][index] = data[1];  del data
    np.savetxt(fout,np.transpose([k,Pk[index][index]])); print '\n'; sys.exit()
########################################################################

########################################################################
#if there are two or more particles compute auto- and cross-power spectra
for i,ptype1 in enumerate(particle_type):
    for ptype2 in particle_type[i+1:]:

        #find the indexes of the particle types
        index1 = index_dict[ptype1];  index2 = index_dict[ptype2]

        #choose the name of the output files
        if do_RSD:  root_fname = 'Pk_RS_'+str(axis)+'_'
        else:       root_fname = 'Pk_'
        fout1  = root_fname+root_fout[str(ptype1)]+'_z='+z+'.dat'
        fout2  = root_fname+root_fout[str(ptype2)]+'_z='+z+'.dat'
        fout12 = root_fname+root_fout[str(ptype1)+str(ptype2)]+'_z='+z+'.dat'

        #some verbose
        print '\nComputing the auto- and cross-power spectra of types: '\
            ,ptype1,'-',ptype2
        print 'saving results in:';  print fout1,'\n',fout2,'\n',fout12

        #This routine computes the auto- and cross-power spectra
        data = PSL.cross_power_spectrum_given_delta(delta[index1],delta[index2],
                                                    dims,BoxSize)

        k                  = data[0]
        Pk[index1][index2] = data[1];   Pk[index2][index1] = data[1]; 
        Pk[index1][index1] = data[2]
        Pk[index2][index2] = data[3]

        #save power spectra results in the output files
        np.savetxt(fout1,  np.transpose([k,Pk[index1][index1]]))
        np.savetxt(fout2,  np.transpose([k,Pk[index2][index2]]))
        np.savetxt(fout12, np.transpose([k,Pk[index1][index2]]))
########################################################################

########################################################################
# compute total matter auto-power spectrum   
print '\ncomputing total matter P(k)'
Pk_m = np.zeros(len(k),dtype=np.float64)
if do_RSD:  f_out_m = 'Pk_RS_'+str(axis)+'_matter_z='+z+'.dat'
else:       f_out_m = 'Pk_matter_z='+z+'.dat'

# dictionary giving the value of Omega for each component
Omega_dict = {0:Omega_g, 1:Omega_c, 2:Omega_n, 4:Omega_s}

for ptype1 in particle_type:
    for ptype2 in particle_type:
        
        # find the indexes of the particle types
        index1 = index_dict[ptype1];  index2 = index_dict[ptype2]

        Pk_m += Omega_dict[ptype1]*Omega_dict[ptype2] * Pk[index1][index2]

Pk_m /= Omega_m**2
np.savetxt(f_out_m,np.transpose([k,Pk_m])) #write results to output file
########################################################################



