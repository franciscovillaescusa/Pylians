#This file contains several useful routines to be used when dealing with 
#halos, subhalos or FoF halos obtained through subfind

import numpy as np
import readsubf
import readfof

############### Routine to read SUBFIND CDM halos positions ###################
#returns the positions of the CDM halos (in Mpc/h) as read from the Subfind 
#groups. If only the positions of halos with masses in a given interval is 
#desired, use mass_interval=True and min_mass-max_mass 
#The groups masses are taken using the 200xmean criteria
def halo_positions(groups_fname,groups_number,mass_interval,
                   min_mass,max_mass):

    #read SUBFIND CDM halos information
    halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                   group_veldisp=True,masstab=True,
                                   long_ids=True,swap=False)
    Pos=halos.group_pos/1e3    #Mpc/h
    if mass_interval:
        a=halos.group_m_mean200>min_mass; b=halos.group_m_mean200<max_mass
        c=a*b; halos_indexes=np.where(c==True)[0]; del a,b,c,halos
        Pos=Pos[halos_indexes]

    return Pos
###############################################################################

############## Routine to read SUBFIND CDM subhalos positions #################
#returns the positions of the CDM subhalos (in Mpc/h) as read from the Subfind 
#subhalos. If only the positions of subhhalos with masses in a given interval 
#is desired, use mass_interval=True and min_mass-max_mass 
#if velocities=True returns also the subhalo velocities
#the velocities are physical, i.e. they dont need extra (sqrt(a)) factors
def subhalo_positions(groups_fname,groups_number,mass_interval,
                      min_mass,max_mass,velocities=False):

    #read SUBFIND CDM subhalos information
    halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                   group_veldisp=True,masstab=True,
                                   long_ids=True,swap=False)
    Pos=halos.sub_pos/1e3    #Mpc/h
    Vel=halos.sub_vel        #km/s ---> they are physical velocities!
    if mass_interval:
        a=halos.sub_mass>min_mass; b=halos.sub_mass<max_mass
        c=a*b; halos_indexes=np.where(c==True)[0]; del a,b,c,halos
        Pos=Pos[halos_indexes]
        Vel=Vel[halos_indexes]

    if velocities:
        return [Pos,Vel]
    else:
        return Pos
###############################################################################

################# Routine to read FoF CDM halos positions #####################
#returns the positions of the FoF halos (in Mpc/h) as read from the Subfind 
#FoF. If only the positions of halos with masses in a given interval is 
#desired, use mass_interval=True and min_mass-max_mass 
def FoF_halo_positions(groups_fname,groups_number,mass_interval,
                       min_mass,max_mass):

    #read FoF halos information
    halos=readfof.FoF_catalog(groups_fname,groups_number,
                              long_ids=True,swap=False)
    Pos=halos.GroupPos/1e3   #Mpc/h
    if mass_interval:
        a=halos.GroupMass>min_mass; b=halos.GroupMass<max_mass
        c=a*b; halos_indexes=np.where(c==True)[0]; del a,b,c,halos
        Pos=Pos[halos_indexes]

    return Pos
###############################################################################







################################### USAGE #####################################
## halos positions ##
"""
groups_fname='/home/villa/disksom2/1000Mpc_z=99/CDM'
groups_number=22
mass_interval=True
min_mass=2e3 #in units of 10^10 Msun/h
max_mass=2e5 #in units of 10^10 Msun/h

pos=halo_positions(groups_fname,groups_number,mass_interval,
                   min_mass,max_mass)

print len(pos)
print np.min(pos[:,0]),'< X <',np.max(pos[:,0])
print np.min(pos[:,1]),'< Y <',np.max(pos[:,1])
print np.min(pos[:,2]),'< Z <',np.max(pos[:,2])
"""

## subhalos positions ##
"""
groups_fname='/home/villa/disksom2/1000Mpc_z=99/CDM'
groups_number=22
mass_interval=True
min_mass=2e3 #in units of 10^10 Msun/h
max_mass=2e5 #in units of 10^10 Msun/h

pos=subhalo_positions(groups_fname,groups_number,mass_interval,
                      min_mass,max_mass)

print len(pos)
print np.min(pos[:,0]),'< X <',np.max(pos[:,0])
print np.min(pos[:,1]),'< Y <',np.max(pos[:,1])
print np.min(pos[:,2]),'< Z <',np.max(pos[:,2])
"""

## FoF halos positions ##
"""
groups_fname='/home/villa/disksom2/1000Mpc_z=99/CDM'
groups_number=22
mass_interval=True
min_mass=2e3 #in units of 10^10 Msun/h
max_mass=2e5 #in units of 10^10 Msun/h

pos=FoF_halo_positions(groups_fname,groups_number,mass_interval,
                       min_mass,max_mass)

print len(pos)
print np.min(pos[:,0]),'< X <',np.max(pos[:,0])
print np.min(pos[:,1]),'< Y <',np.max(pos[:,1])
print np.min(pos[:,2]),'< Z <',np.max(pos[:,2])
"""
