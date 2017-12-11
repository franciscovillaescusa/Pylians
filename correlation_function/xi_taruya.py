#This script computes the 2pt correlation function of total matter (CDM+NU)
#from an N-body snapshot using the Taruya et al. 2009 (0906.0507) estimator
import numpy as np
import readsnap
import MAS_library as MASL
import Pk_library as PKL
import sys,os,imp

#################################### INPUT ##################################
if len(sys.argv)>1:

    parameter_file = sys.argv[1]
    print '\nLoading the parameter file ',parameter_file
    if os.path.exists(parameter_file):
        parms = imp.load_source("name",parameter_file)
        globals().update(vars(parms))
    else:   print 'file does not exists!!!'

else:
    snapshot_fname = '../snapdir_001/snap_001'
    MAS            = 'CIC'
    dims           = 1024
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

fout = 'CF_CDM_z=%.3f.txt'%redshift

# read the positions and masses of the CDM particles
pos = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3  #Mpc/h

# compute delta_CDM
delta = np.zeros((dims,dims,dims), dtype=np.float32)
MASL.MA(pos,delta,BoxSize,MAS)
print '%.6e should be equal to\n%.6e'\
    %(np.sum(delta,dtype=np.float64),len(pos))
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

#compute the correlation function
CF = PKL.Xi(delta, BoxSize, MAS, threads=8)

#save results to file
np.savetxt(fout,np.transpose([CF.r3D, CF.xi]))
