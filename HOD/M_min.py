#This code computes M_min as a function of M1 and alpha

import numpy as np
import readsnap
import readsubf
import sys
import os


################################## INPUT #######################################
#snapshot_fname='/home/villa/disksom2/som_simulations/500Mpc_z=99_YB/CDM/snapdir_004/snap_004'
#groups_fname='/home/villa/disksom2/som_simulations/500Mpc_z=99_YB/CDM'
#groups_number=4

#snapshot_fname='/home/villa/disksom2/old_simulations/CDM/snapdir_017/snap_017'
#groups_fname='/home/villa/disksom2/old_simulations/CDM'
#groups_number=17

snapshot_fname='/home/villa/disksom2/500Mpc_z=99/CDM/snapdir_003/snap_003'
groups_fname='/home/villa/disksom2/500Mpc_z=99/CDM'
groups_number=3

#### HALO CATALOGUE PARAMETERS ####
mass_criteria='m200' #'t200' 'm200' or 'c200'
min_mass=3e10 #Msun/h
max_mass=2e15 #Msun/h

### HOD PARAMETERS ###
fiducial_density=0.00111 #mean number density for galaxies with Mr<-21
M1_min=8.0e13;     M1_max=1.4e14;   M1_bins=40
alpha_min=1.00;  alpha_max=1.80;  alpha_bins=40

#### PARAMETERS ####
BoxSize=500.0 #Mpc/h

f_out='borrar2.dat'
################################################################################



g=open(f_out,'w')
M1_array=np.linspace(M1_min, M1_max, M1_bins)
alpha_array=np.linspace(alpha_min, alpha_max, alpha_bins)

#read CDM halos information
halos=readsubf.subfind_catalog(groups_fname,groups_number,
                               group_veldisp=True,masstab=True,
                               long_ids=True,swap=False)
if mass_criteria=='t200':
    halos_mass=halos.group_m_tophat200*1e10   #masses in Msun/h
    halos_radius=halos.group_r_tophat200      #radius in kpc/h
elif mass_criteria=='m200':
    halos_mass=halos.group_m_mean200*1e10     #masses in Msun/h
    halos_radius=halos.group_r_mean200        #radius in kpc/h
elif mass_criteria=='c200':    
    halos_mass=halos.group_m_crit200*1e10     #masses in Msun/h
    halos_radius=halos.group_r_crit200        #radius in kpc/h
else:
    print 'bad mass_criteria'
    sys.exit()
halos_pos=halos.group_pos
halos_len=halos.group_len
halos_offset=halos.group_offset
halos_indexes=np.where((halos_mass>min_mass) & (halos_mass<max_mass))[0]
del halos
    
print ' '
print 'total halos found=',len(halos_pos)
print 'halos number density=',len(halos_pos)/BoxSize**3

#keep only the halos in the given mass range 
halo_mass=halos_mass[halos_indexes]
halo_pos=halos_pos[halos_indexes]
halo_radius=halos_radius[halos_indexes]
halo_len=halos_len[halos_indexes]
halo_offset=halos_offset[halos_indexes]
del halos_indexes

if np.any(halo_len==[]):
    print 'something bad'


thres=1e-10
for M1 in M1_array:
    for alpha in alpha_array:

        ##### COMPUTE Mmin GIVEN M1 & alpha #####
        i=0; max_iterations=20 #maximum number of iterations
        Mmin1=min_mass; Mmin2=max_mass
        while (i<max_iterations):
            Mmin=0.5*(Mmin1+Mmin2) #estimation of the HOD parameter Mmin

            total_galaxies=0
            inside=np.where(halo_mass>Mmin)[0]
            mass=halo_mass[inside] #only halos with M>Mmin have central/satellites

            total_galaxies=mass.shape[0]+np.sum((mass/M1)**alpha)
            mean_density=total_galaxies*1.0/BoxSize**3

            #print 'number of central galaxies=',mass.shape[0]
            #print 'number of satellite galaxies=',np.sum((mass/M1)**alpha)

            if (np.absolute((mean_density-fiducial_density)/fiducial_density)<thres):
                i=max_iterations
            elif (mean_density>fiducial_density):
                Mmin1=Mmin
            else:
                Mmin2=Mmin
            i+=1

        print ' '
        print 'alpha : M1 =',alpha,M1
        print 'Mmin=',Mmin
        print 'average number of galaxies=',total_galaxies
        print 'average galaxy density=',mean_density
        print 'number of satellites=',np.sum((mass/M1)**alpha)
        g.write(str(M1)+' '+str(alpha)+' '+str(Mmin)+' '+str(total_galaxies)+' '+str(mass.shape[0])+' '+str(np.sum((mass/M1)**alpha))+' '+str(mean_density)+'\n')
g.close()
