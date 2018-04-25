import numpy as np
import sys,os,glob,h5py
import groupcat
import MAS_library as MASL
import Pk_library as PKL


################################ INPUT ######################################
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

BoxSize = 75.0 #Mpc/h
MAS  = 'NGP'
dims = 64
#############################################################################

num = {0:99, 1:50, 2:33, 3:25, 4:21, 5:17}

snapshot_root = '%s/output/'%run

bins_histo = np.logspace(-3,3,100)
dbins      = bins_histo[1:] - bins_histo[:-1]
bins_mean  = 10**(0.5*(np.log10(bins_histo[1:]) + np.log10(bins_histo[:-1])))


for z in [0,1,2,3,4,5]:

    # read number of particles in halos and subhalos and number of subhalos
    halos = groupcat.loadHalos(snapshot_root, num[z],
                               fields=['GroupMass','GroupPos'])
    halo_mass = halos['GroupMass'][:]*1e10 #Msun/h
    halo_pos  = halos['GroupPos'][:]/1e3   #Mpc/h
    del halos

    if z==0:
        alpha, M0, Mmin = 0.24, 4.3e10, 2e12
    elif z==1:
        alpha, M0, Mmin = 0.53, 1.5e10, 6e11
    elif z==2:
        alpha, M0, Mmin = 0.60, 1.3e10, 3.6e11
    elif z==3:
        alpha, M0, Mmin = 0.76, 2.9e9, 6.7e10
    elif z==4:
        alpha, M0, Mmin = 0.79, 1.4e9, 2.1e10
    elif z==5:
        alpha, M0, Mmin = 0.74, 1.9e9, 2e10
    else: raise Exception('wrong redshift')

    M_HI = M0*(halo_mass/Mmin)**alpha*np.exp(-(Mmin/halo_mass)**(0.35))
 
    delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)
    MASL.MA(halo_pos,delta_HI,BoxSize,MAS,W=M_HI)
    delta_HI /= np.mean(delta_HI, dtype=np.float64)
    delta_HI = np.ravel(delta_HI)
    
    histo = np.histogram(delta_HI, bins_histo)[0]
    histo = histo/(dims**3*dbins)
    np.savetxt('pdf_HI_%d_z=%d.txt'%(dims,z), np.transpose([bins_mean, histo]))


    delta_h = np.zeros((dims,dims,dims), dtype=np.float32)
    MASL.MA(halo_pos,delta_h,BoxSize,MAS,W=halo_mass)
    delta_h /= np.mean(delta_h, dtype=np.float64)
    delta_h = np.ravel(delta_h)
    
    histo = np.histogram(delta_h, bins_histo)[0]
    histo = histo/(dims**3*dbins)
    np.savetxt('pdf_h_%d_z=%d.txt'%(dims,z), np.transpose([bins_mean, histo]))

