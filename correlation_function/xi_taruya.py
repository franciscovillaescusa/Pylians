#This script computes the 2pt correlation function of total matter (CDM+NU)
#from an N-body snapshot using the Taruya et al. 2009 (0906.0507) estimator
import numpy as np
import readsnap
import CIC_library as CIC
import Power_spectrum_library as PSL
import redshift_space_library as RSL
import sys,os,imp

rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
#################################### INPUT ##################################
if len(sys.argv)>1:

    parameter_file = sys.argv[1]
    print '\nLoading the parameter file ',parameter_file
    if os.path.exists(parameter_file):
        parms = imp.load_source("name",parameter_file)
        globals().update(vars(parms))
    else:   print 'file does not exists!!!'

else:
    snapshot_fname = 'snapdir_003/snap_003'
    MAS            = 'CIC'
    dims           = 512
    bins_CF        = 'None'  #if 'None' bins_CF = dims/2+1
    do_RSD         = False
    axis           = 0
    fout           = 'CF_test_512.txt'
#############################################################################
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

#define the array containing the deltas
delta = np.zeros(dims3,dtype=np.float32)

#read the positions and masses of the CDM particles
pos  = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3  #Mpc/h
mass = readsnap.read_block(snapshot_fname,"MASS",parttype=1)*1e10 #Msun/h
if do_RSD:
    vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
print 'Omega_CDM = %.4f'%(np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit)

#if there are neutrinos read their positions and masses
if Nall[2]>0:
    pos_nu  = readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3  #Mpc/h
    mass_nu = readsnap.read_block(snapshot_fname,"MASS",parttype=2)*1e10 #Msun/h
    print 'Omega_NU  = %.4f'\
        %(np.sum(mass_nu,dtype=np.float64)/BoxSize**3/rho_crit)
    pos  = np.vstack([pos,pos_nu]);    del pos_nu
    mass = np.hstack([mass,mass_nu]);  del mass_nu
    if do_RSD:
        vel_nu = readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s
        vel    = np.vstack([vel,vel_nu]);  del vel_nu
print 'Omega_m   = %.4f'%(np.sum(mass,dtype=np.float64)/BoxSize**3/rho_crit)

if do_RSD:
    RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis)

#compute the mean mass in each cell
mean_mass = np.sum(mass,dtype=np.float64)*1.0/dims3

#compute the deltas
CIC.CIC_serial(pos,dims,BoxSize,delta,mass)
print '%.6e should be equal to\n%.6e'\
    %(np.sum(delta,dtype=np.float64),np.sum(mass,dtype=np.float64))
del pos,mass
delta = delta/mean_mass - 1.0
print '%.6e should be close to 0'%np.mean(delta,dtype=np.float64)

#compute the correlation function
r,xi = PSL.CF_Taruya(delta,dims,BoxSize,bins_CF,MAS)

#save results to file
np.savetxt(fout,np.transpose([r,xi]))
