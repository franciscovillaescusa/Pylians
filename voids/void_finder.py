import numpy as np 
import sys,os,time
import void_library as VL
import readsnap
import MAS_library as MASL

################################# INPUT ######################################
snapshot_fname = '../snapdir_003/snap_003'

# density field parameters
dims = 768
MAS  = 'PCS'

# void finder parameters
threshold = -0.7
Rmax      = 45.0
Rmin      = 8.0
bins      = 50
threads   = 28
##############################################################################

# read snapshot head and obtain BoxSize, Omega_m and Omega_L
head     = readsnap.snapshot_header(snapshot_fname)
BoxSize  = head.boxsize/1e3  #Mpc/h                      
Nall     = head.nall
Masses   = head.massarr*1e10 #Msun/h              
Omega_m  = head.omega_m
Omega_l  = head.omega_l
redshift = head.redshift

# read particle positions
pos = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h

# compute density field
delta = np.zeros((dims, dims, dims), dtype=np.float32)
MASL.MA(pos, delta, BoxSize, MAS)
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

V = VL.void_finder(delta, BoxSize, threshold, Rmax, Rmin, bins, Omega_m, 
    threads, void_field=True)

# void properties
void_pos    = V.void_pos
void_radius = V.void_radius
void_mass   = V.void_mass
delta_void  = V.in_void  #array with 0. Cells belonging to voids have 1

# void mass function
Rvoid  = V.Rbins    # bins in radius
MFvoid = V.void_mf  # void mass function



