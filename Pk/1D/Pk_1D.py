# This code computes the 1D power spectrum from a simulation snapshot
# From the 1D power spectrum it also estimates the 3D P(k)
# It corrects the modes amplitude to account for MAS in the radial direction
import numpy as np
import readsnap
import CIC_library as CIC
import redshift_space_library as RSL
import scipy.fftpack
import scipy.weave as wv
import sys,os,time

rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3
################################ INPUT ################################
snapshot_fname = '../snapdir_003/snap_003'
dims           = 512
axis           = 2       #axis along which compute the 1D P(k)
do_RSD         = False   #whether do RSD along the above axis
ptype          = -1      #0:GAS, 1:CDM, 4:stars, -1: matter
f_1D_Pk        = 'Pk2_1D_512_z=3.dat'
f_3D_Pk        = 'Pk2_3D_512_from_1D_z=3.dat'
#######################################################################

start_time = time.clock()

# we perform the FFT only along one axis. We will then have a 3D matrix
# delta_k[i,j,k]. If we perform the FFT along the z-axis then we want to
# stack the results along the other directions delta_k[:,:,k]. To make
# the stacking we need to define the other two directions here
axis1 = {0:1, 1:0, 2:0} #after stack along one axis delta_k has 2 dimensions
axis2 = {0:1, 1:1, 2:0}  

# read snapshot head and obtain BoxSize, Omega_m and Omega_L     
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

# read particle positions and masses         
pos  = readsnap.read_block(snapshot_fname,"POS ",parttype=ptype)/1e3  #Mpc/h
mass = readsnap.read_block(snapshot_fname,"MASS",parttype=ptype)*1e10 #Msun/h

# move particle positions to redshift-space    
if do_RSD:
    vel  = readsnap.read_block(snapshot_fname,"VEL ",parttype=ptype) #km/s  
    RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis);  del vel

# some verbose
print '%.3f < X [Mpc/h] < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0]))
print '%.3f < Y [Mpc/h] < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1]))
print '%.3f < Z [Mpc/h] < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2]))
print 'Omega_ptype = %.4f'%(np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit)
print 'Omega_m     =',Omega_m

# compute the mean mass per cell
mean_mass = np.sum(mass,dtype=np.float64)/dims**3

# compute the density contrast in each point of the grid
delta = np.zeros(dims**3,dtype=np.float32)
CIC.CIC_serial(pos,dims,BoxSize,delta,weights=mass)
print '%.5e should be equal to\n%.5e'\
    %(np.sum(delta,dtype=np.float64),np.sum(mass,dtype=np.float64))
delta = delta/mean_mass - 1.0
print '%.3e < delta < %.3e'%(np.min(delta),np.max(delta))

# reshape the density constrast field array
delta = np.reshape(delta,(dims,dims,dims))

# Fourier transform only along the given axis
print '\nFFT the overdensity field along axis',axis
delta_k = scipy.fftpack.ifftn(delta,overwrite_x=True,axes=(axis,));  del delta

# compute the value of |delta(k)|^2
delta_k2 = np.absolute(delta_k)**2;  del delta_k

# take all LOS and compute the average value for each mode
print 'stacking delta_k[i,j,k] along axis:',axis1[axis]
delta_k2 = np.mean(delta_k2,dtype=np.float64,axis=axis1[axis])
print 'stacking  delta_k[i,j]  along axis:',axis2[axis]
delta_k2 = np.mean(delta_k2,dtype=np.float64,axis=axis2[axis])

# compute the values of k for the 1D P(k)
k = np.arange(dims,dtype=np.float64);  middle = dims/2
indexes = np.where(k>middle)[0];  k[indexes] = k[indexes]-dims;  del indexes
k *= (2.0*np.pi/BoxSize)  #k in h/Mpc
k = np.absolute(k)  #just take the modulus of the wavenumber

# correct modes for MAS in the direction where P(k) is computed
argument = k*BoxSize/(2.0*dims)
W_k = np.ones(len(k),dtype=np.float32)  # define here to avoid division by 0
W_k[1:] = (argument[1:]/np.sin(argument[1:]))**2 # CIC correction!!! 
delta_k2 *= (W_k**2)

# define the k-bins
k_bins = np.linspace(0,dims/2,dims/2+1)*(2.0*np.pi/BoxSize)

# compute the number of modes and the averange number-weighted value of k 
modes = np.histogram(k,bins=k_bins)[0]
k_bin = np.histogram(k,bins=k_bins,weights=k)[0]/modes

# compute the 1D P(k) and save results ignoring the DC mode
Pk = np.histogram(k,bins=k_bins,weights=delta_k2)[0];  del delta_k2
Pk = Pk*BoxSize/modes  #Mpc/h
np.savetxt(f_1D_Pk,np.transpose([k_bin[1:],Pk[1:]]));  del modes

# compute 3D P(k) from 1D P(k) and results
k_3D  = 0.5*(k_bin[1:]+k_bin[:-1])
Pk_3D = -2.0*np.pi/k_3D*(Pk[1:]-Pk[:-1])/(k_bin[1:]-k_bin[:-1])
np.savetxt(f_3D_Pk,np.transpose([k_3D,Pk_3D]))

# final verbose
print 'time taken =',time.clock()-start_time,'seconds'


