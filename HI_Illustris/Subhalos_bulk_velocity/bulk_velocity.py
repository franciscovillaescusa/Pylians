import numpy as np
import sys,os,h5py
import groupcat

# This routine computes the cosine of the angle between two vectors and the
# ratio between their moduli
def cosine_and_ratio(V1,V2):
    cos_a = np.dot(V1,V2)/np.sqrt(np.dot(V1,V1)*np.dot(V2,V2))
    ratio = np.sqrt(np.dot(V1,V1)/np.dot(V2,V2))

    return cos_a,ratio

############################## INPUT ########################################
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

snapshot_root = '%s/output/'%run

snapnum = 17
#############################################################################

z_dict = {99:0, 17:5}

FoF = groupcat.loadHalos(snapshot_root, snapnum, 
                         fields=['GroupLenType','GroupNsubs','GroupVel',
                                 'GroupFirstSub','GroupMass'])
halo_len   = FoF['GroupLenType'][:,0]  
Nsubs = FoF['GroupNsubs']
halo_vel = FoF['GroupVel']*(1.0+z_dict[snapnum])
first_sub = FoF['GroupFirstSub']
halo_mass = FoF['GroupMass']*1e10 #Msun/h

subhalos = groupcat.loadSubhalos(snapshot_root, snapnum, 
                                 fields=['SubhaloGrNr','SubhaloMass',
                                         'SubhaloVel'])
sub_FoF  = subhalos['SubhaloGrNr'][:]
sub_mass = subhalos['SubhaloMass'][:]*1e10 #Msun/h
sub_vel  = subhalos['SubhaloVel'][:]
del subhalos

number = 50000
1
Mass  = halo_mass[:number]
cos_a   = np.zeros(number)
ratio   = np.zeros(number)
cos_ac  = np.zeros(number)
ratio_c = np.zeros(number)

V = np.zeros(3, dtype=np.float64)
for halo_num in xrange(number):

    print '%03d'%(halo_num)

    indexes = np.where(sub_FoF==halo_num)[0]
    index = np.where(indexes==first_sub[halo_num])
    indexes = np.delete(indexes,index)
    
    Mtot = np.sum(sub_mass[indexes])

    for i in xrange(3):
        V[i] = np.sum(sub_vel[indexes,i]*sub_mass[indexes])/Mtot

    cos_a[halo_num], ratio[halo_num] = \
        cosine_and_ratio(V,halo_vel[halo_num])

    Vc = sub_vel[first_sub[halo_num]]

    cos_ac[halo_num], ratio_c[halo_num] =\
        cosine_and_ratio(Vc,halo_vel[halo_num])

np.savetxt('borrar.txt', np.transpose([Mass, cos_a, ratio, cos_ac, ratio_c]))



bins   = 30
M_bins = np.logspace(8, 15, bins+1)
M_mean = 10**(0.5*(np.log10(M_bins[1:]) + np.log10(M_bins[:-1])))

cos_a_mean  = np.zeros(bins, dtype=np.float64) 
cos_a_std   = np.zeros(bins, dtype=np.float64) 
cos_ac_mean = np.zeros(bins, dtype=np.float64) 
cos_ac_std  = np.zeros(bins, dtype=np.float64) 

ratio_mean = np.zeros(bins, dtype=np.float64) 
ratio_std  = np.zeros(bins, dtype=np.float64) 
ratio_c_mean = np.zeros(bins, dtype=np.float64) 
ratio_c_std  = np.zeros(bins, dtype=np.float64) 
for i in xrange(bins):
    indexes = np.where((Mass>=M_bins[i]) & (Mass<M_bins[i+1]))[0]
    if len(indexes)==0:  continue

    cos_a_mean[i] = np.mean(cos_a[indexes])
    cos_a_std[i]  = np.std(cos_a[indexes])
    cos_ac_mean[i] = np.mean(cos_ac[indexes])
    cos_ac_std[i]  = np.std(cos_ac[indexes])

    ratio_mean[i] = np.mean(ratio[indexes])
    ratio_std[i]  = np.std(ratio[indexes])
    ratio_c_mean[i] = np.mean(ratio_c[indexes])
    ratio_c_std[i]  = np.std(ratio_c[indexes])
    
np.savetxt('cos_a_mean_std_z=%d.txt'%round(z_dict[snapnum]), 
           np.transpose([M_mean, cos_a_mean, cos_a_std]))
np.savetxt('cos_ac_mean_std_z=%d.txt'%round(z_dict[snapnum]), 
           np.transpose([M_mean, cos_ac_mean, cos_ac_std]))
np.savetxt('ratio_mean_std_z=%d.txt'%round(z_dict[snapnum]), 
           np.transpose([M_mean, ratio_mean, ratio_std]))
np.savetxt('ratio_c_mean_std_z=%d.txt'%round(z_dict[snapnum]), 
           np.transpose([M_mean, ratio_c_mean, ratio_c_std]))
