import numpy as np
import readsnap
import CIC_library as CIC
import Power_spectrum_library as PSL
import sys

#Pos is an array containing the positions of the particles along one axis
#Vel is an array containing the velocities of the particle along the above axis
def RSD(Pos,Vel,Hubble,redshift):
    #transform coordinates to redshift space
    delta_y=(Vel/Hubble)*(1.0+redshift)  #displacement in Mpc/h
    Pos+=delta_y #add distorsion to position of particle in real-space
    del delta_y

    #take care of the boundary conditions
    beyond=np.where(Pos>BoxSize)[0]; Pos[beyond]-=BoxSize
    beyond=np.where(Pos<0.0)[0];     Pos[beyond]+=BoxSize
    del beyond

rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
################################## INPUT ######################################
if len(sys.argv)>1:
    sa=sys.argv
    snapshot_fname=sa[1]; dims=int(sa[2]); f_out=sa[3]
    do_RSD=bool(int(sa[4])); axis=int(sa[5])

else:
    snapshot_fname='../Efective_model_60Mpc/snapdir_007/snap_007'
    dims=1024
    f_out='Pk_matter_RS_60Mpc_z=4.dat'
    do_RSD=True
    axis=0
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

#compute the values of Omega_CDM and Omega_B
Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b=Omega_m-Omega_cdm
print '\nOmega_CDM = %.3f\nOmega_B   = %0.3f\nOmega_M   = %.3f\n'\
    %(Omega_cdm,Omega_b,Omega_m)


#read the positions of all the particles
pos=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
print '%.3f < X [Mpc/h] < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0]))
print '%.3f < Y [Mpc/h] < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1]))
print '%.3f < Z [Mpc/h] < %.3f\n'%(np.min(pos[:,2]),np.max(pos[:,2]))

#read the velocities of all the particles
vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=-1) #kms

if do_RSD:
    print 'moving particles to redshift-space'
    RSD(pos[:,axis],vel[:,axis],Hubble,redshift)

#read the masses of all the particles
M=readsnap.read_block(snapshot_fname,"MASS",parttype=-1)*1e10 #Msun/h
print '%.3e < M [Msun/h] < %.3e'%(np.min(M),np.max(M))
print 'Omega_m = %.3f\n'%(np.sum(M,dtype=np.float64)/rho_crit/BoxSize**3)

#compute the mean mass per grid cell
mean_M=np.sum(M,dtype=np.float64)/dims**3

#compute the mass within each grid cell
delta=np.zeros(dims**3,dtype=np.float32)
CIC.CIC_serial(pos,dims,BoxSize,delta,M); del pos
print '%.6e should be equal to \n%.6e\n'\
    %(np.sum(M,dtype=np.float64),np.sum(delta,dtype=np.float64)); del M

#compute the density constrast within each grid cell
#delta=delta/mean_M - 1.0
delta/=mean_M; delta-=1.0
print '%.3e < delta < %.3e\n'%(np.min(delta),np.max(delta))

#compute the P(k)
Pk=PSL.power_spectrum_given_delta(delta,dims,BoxSize)

#write P(k) to output file
np.savetxt(f_out,np.transpose([Pk[0],Pk[1]]))



