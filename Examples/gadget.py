import numpy as np
import readsnap,readfof
import MAS_library as MASL
import sys,os

#########################################################################
# read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head     = readsnap.snapshot_header(snapshot_fname)
BoxSize  = head.boxsize/1e3  #Mpc/h                      
Nall     = head.nall
Masses   = head.massarr*1e10 #Msun/h              
Omega_m  = head.omega_m
Omega_l  = head.omega_l
redshift = head.redshift
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#km/s/(Mpc/h)
h        = head.hubble
z        = '%.3f'%redshift
#########################################################################

#########################################################################
pos = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=1)     #km/s
#########################################################################

#########################################################################
# read positions and velocities of halos
FoF   = readfof.FoF_catalog(snapdir,snapnum,long_ids=False,
                            swap=False,SFR=False)
pos_h = FoF.GroupPos/1e3            #Mpc/h
mass  = FoF.GroupMass*1e10          #Msun/h
vel_h = FoF.GroupVel*(1.0+redshift) #km/s
indexes = np.where(mass>Mmin)[0]
pos_h = pos_h[indexes];  vel_h = vel_h[indexes];  del indexes
#########################################################################
