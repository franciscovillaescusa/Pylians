#DESCRIPTION: 
#This code computes the auto- (cross-) power spectrum (PS) and the quadrupole 
#of many different objects:

# 'CDM' -------------- only CDM
# 'NU' --------------- only neutrinos
# 'DM-NU' ------------ CDM+NU-NU
# 'DM' --------------- CDM+NU, CDM alone. NU alone and CDM-NU cross

# 'halos' ------------ SUBFIND halos
# 'subhalos' --------- SUBFIND subhalos
# 'FoF_halos' -------- FoF halos

# 'DM-halos' --------- CDM+NU - SUBFIND halos
# 'DM-subhalos' ------ CDM+NU - SUBFIND subhalos
# 'DM-FoF_halos' ----- CDM+NU - FoF halos

# 'CDM-halos' -------- CDM - SUBFIND halos
# 'CDM-subhalos' ----- CDM - SUBFIND subhalos
# 'CDM-FoF_halos' ---- CDM - FoF halos

# 'quadrupole-CDM' --- quadrupole CDM only

#VARIABLES:
#snapshot_fname --> name of the N-body snapshot
#groups_fname ----> folder containing the SUBFIND/FoF information
#groups_number ---> number of the group folder
#obj -------------> type of object over which compute the PS. Select from above
#mass_interval ---> Select halos by min/max mass: True/False.  
#min_mass --------> min mass of the dark matter halos (in 1e10 Msun/h units)
#max_mass --------> max mass of the dark matter halos (in 1e10 Msun/h units)
#do_RSD ----------> compute the PS in redshift-space: True/False
#axis ------------> axis along which compute the PS: 0, 1 or 2
#dims ------------> number of points per dimension to use in the grid
#f_out -----------> name of the output file

#USAGE:
#copy this file in the folder containing the snapshot/SUBFIND/FoF file. 
#Set the variables and type python Pk.py

#VERSION HISTORY:
#Version 2.3
#Computes the quadrupole

#Version 2.2
#includes the posibility of measuring the power spectrums in redshift-space
#along a given axis

#Version 2.1
#use the halo library to read the positions of the halos, subhalos and FoF


import numpy as np
import readsnap
import Power_spectrum_library as PSL
import halos_library as HL
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

    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=sa[3]
    obj=sa[4]

    mass_interval=bool(int(sa[5]))
    min_mass=float(sa[6]); max_mass=float(sa[7])

    do_RSD=bool(int(sa[8])); axis=int(sa[9])

    dims=int(sa[10])
    f_out=sa[11]
else:
    snapshot_fname='/home/villa/disksom2/ICTP/CDM/0/snapdir_022/snap_022'
    groups_fname='/home/villa/disksom2/ICTP/CDM/0/'
    groups_number=22 

    obj='CDM' 

    mass_interval=True
    min_mass=1.6e3
    max_mass=3.0e5

    do_RSD=False #do redshift-space distortions----True or False
    axis=0 #axis along which the RSD are computed: 0-X, 1-Y, 2-Z

    dims=512
    f_out = 'borrar.dat'
###############################################################################

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc


####################### CDM, NU, CDM-NU and DM=CDM+NU ########################
if obj=='DM':
    #read CDM and NU positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    Pos2=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s
        RSD(Pos2[:,axis],Vel[:,axis],Hubble,redshift); del Vel

    #computes OmegaCDM and OmegaNU
    OmegaCDM = Nall[1]*Masses[1]/(BoxSize**3*rho_crit)
    OmegaNU  = Nall[2]*Masses[2]/(BoxSize**3*rho_crit)
    print 'OmegaCDM=',OmegaCDM; print 'OmegaNU= ',OmegaNU
    print 'OmegaDM= ',OmegaCDM+OmegaNU

    #computes CDM & NU P(k), CDM-NU cross-P(k) and DM P(k)
    A=PSL.power_spectrum_full_analysis(Pos1,Pos2,OmegaCDM,OmegaNU,dims,BoxSize)
    k=A.k
    Pk_CDM=A.Pk1; dPk_CDM=A.dPk1
    Pk_NU=A.Pk2; dPk_NU=A.dPk2
    Pk_CDM_NU=A.Pk12
    Pk_DM=A.Pk
    print np.min(A.check),'< diff <',np.max(A.check)

    #write P(k) files
    f1=open(f_out+'_CDM','w'); f2=open(f_out+'_NU','w')
    f3=open(f_out+'_CDM-NU','w'); f4=open(f_out+'_DM','w')
    for i in range(len(k)):
        f1.write(str(k[i])+' '+str(Pk_CDM[i])+' '+str(dPk_CDM[i])+'\n')
        f2.write(str(k[i])+' '+str(Pk_NU[i])+' '+str(dPk_NU[i])+'\n')
        f3.write(str(k[i])+' '+str(Pk_CDM_NU[i])+'\n')
        f4.write(str(k[i])+' '+str(Pk_DM[i])+'\n')
    f1.close(); f2.close(); f3.close(); f4.close()


################################ CDM ONLY #################################

elif obj=='CDM':
    #read CDM positions
    Pos=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    
    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos[:,axis],Vel[:,axis],Hubble,redshift); del Vel

    #computes P(k)
    Pk=PSL.power_spectrum(Pos,dims,BoxSize,shoot_noise_correction=False)
    
    #write CDM P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+str(Pk[2][i])+'\n')
    f.close()

############################## NEUTRINOS ONLY ###############################

elif obj=='NU':
    #read NU positions
    Pos=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s
        RSD(Pos[:,axis],Vel[:,axis],Hubble,redshift); del Vel

    #computes P(k)
    Pk=PSL.power_spectrum(Pos,dims,BoxSize,shoot_noise_correction=False)

    #write NU P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+str(Pk[2][i])+'\n')
    f.close()

################################## SO HALOS #################################

elif obj=='halos':
    #read SUBFIND CDM halo positions
    Pos=HL.halo_positions(groups_fname,groups_number,
                          mass_interval,min_mass,max_mass) #Mpc/h

    #move to redshift-space 
    if do_RSD:
        print 'option not implemented for SO halos'

    #computes P(k)
    Pk=PSL.power_spectrum(Pos,dims,BoxSize,shoot_noise_correction=True)

    #write halos P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+str(Pk[2][i])+'\n')
    f.close()

################################## SUBHALOS #################################

elif obj=='subhalos':

    #read CDM halo positions (and velocities for RSD)
    if do_RSD:
        [Pos,Vel]=HL.subhalo_positions(groups_fname,groups_number,
                                       mass_interval,min_mass,max_mass,
                                       velocities=True) #Mpc/h
    else:
        Pos=HL.subhalo_positions(groups_fname,groups_number,
                                 mass_interval,min_mass,max_mass,
                                 velocities=False) #Mpc/h
    #move to redshift-space
    if do_RSD:
        RSD(Pos[:,axis],Vel[:,axis],Hubble,redshift); del Vel

    #computes P(k)
    Pk=PSL.power_spectrum(Pos,dims,BoxSize,shoot_noise_correction=True)

    #write subhalos P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+str(Pk[2][i])+'\n')
    f.close()

################################ FoF HALOS #################################

elif obj=='FoF_halos':
    #read FoF CDM halo positions
    Pos=HL.FoF_halo_positions(groups_fname,groups_number,mass_interval,
                              min_mass,max_mass) #Mpc/h

    #move to redshift-space 
    if do_RSD:
        print 'option not implemented for FoF halos'

    #computes P(k)
    Pk=PSL.power_spectrum(Pos,dims,BoxSize,shoot_noise_correction=True)

    #write halos P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+str(Pk[2][i])+'\n')
    f.close()

############################### DM-SO_HALOS ################################

elif obj=='DM-halos':
    #read CDM and NU positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    Pos2=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h    

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s
        RSD(Pos2[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        print 'option not implemented for SO halos'

    #computes OmegaCDM and OmegaNU
    OmegaCDM = Nall[1]*Masses[1]/(BoxSize**3*rho_crit)
    OmegaNU  = Nall[2]*Masses[2]/(BoxSize**3*rho_crit)
    print 'OmegaCDM=',OmegaCDM; print 'OmegaNU= ',OmegaNU
    print 'OmegaDM= ',OmegaCDM+OmegaNU

    #read CDM halo positions
    Pos3=HL.halo_positions(groups_fname,groups_number,
                           mass_interval,min_mass,max_mass) #Mpc/h

    #computes the DM-halos cross-P(k)
    Pk=PSL.cross_power_spectrum_DM(Pos1,Pos2,Pos3,OmegaCDM,OmegaNU,dims,BoxSize)
    
    #write cross-P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

############################### DM-SUBHALOS #################################

elif obj=='DM-subhalos':
    #read CDM and NU positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    Pos2=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h    

    #read CDM halo positions (and velocities for RSD)
    if do_RSD:
        [Pos3,Vel3]=HL.subhalo_positions(groups_fname,groups_number,
                                         mass_interval,min_mass,max_mass,
                                         velocities=True) #Mpc/h
    else:
        Pos3=HL.subhalo_positions(groups_fname,groups_number,
                                  mass_interval,min_mass,max_mass,
                                  velocities=False) #Mpc/h

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s
        RSD(Pos2[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        RSD(Pos3[:,axis],Vel3[:,axis],Hubble,redshift); del Vel3

    #computes OmegaCDM and OmegaNU
    OmegaCDM = Nall[1]*Masses[1]/(BoxSize**3*rho_crit)
    OmegaNU  = Nall[2]*Masses[2]/(BoxSize**3*rho_crit)
    print 'OmegaCDM=',OmegaCDM; print 'OmegaNU= ',OmegaNU
    print 'OmegaDM= ',OmegaCDM+OmegaNU


    #computes the DM-subhalos cross-P(k)
    Pk=PSL.cross_power_spectrum_DM(Pos1,Pos2,Pos3,OmegaCDM,OmegaNU,dims,BoxSize)
    
    #write cross-P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

############################### DM-FoF_HALOS ################################

elif obj=='DM-FoF_halos':
    #read CDM and NU positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    Pos2=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h    

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s
        RSD(Pos2[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        print 'option not implemented for FoF halos'

    #computes OmegaCDM and OmegaNU
    OmegaCDM = Nall[1]*Masses[1]/(BoxSize**3*rho_crit)
    OmegaNU  = Nall[2]*Masses[2]/(BoxSize**3*rho_crit)
    print 'OmegaCDM=',OmegaCDM; print 'OmegaNU= ',OmegaNU
    print 'OmegaDM= ',OmegaCDM+OmegaNU

    #read FoF CDM halo positions
    Pos3=HL.FoF_halo_positions(groups_fname,groups_number,mass_interval,
                               min_mass,max_mass) #Mpc/h

    #computes the DM-halos cross-P(k)
    Pk=PSL.cross_power_spectrum_DM(Pos1,Pos2,Pos3,OmegaCDM,OmegaNU,dims,BoxSize)
    
    #write cross-P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

############################# DM-NU = CDM+NU-NU ##############################

elif obj=='DM-NU':

    #read CDM and NU positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    Pos2=readsnap.read_block(snapshot_fname,"POS ",parttype=2)/1e3 #Mpc/h    

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s
        RSD(Pos2[:,axis],Vel[:,axis],Hubble,redshift); del Vel

    #computes OmegaCDM and OmegaNU
    OmegaCDM = Nall[1]*Masses[1]/(BoxSize**3*rho_crit)
    OmegaNU  = Nall[2]*Masses[2]/(BoxSize**3*rho_crit)
    print 'OmegaCDM=',OmegaCDM; print 'OmegaNU= ',OmegaNU
    print 'OmegaDM= ',OmegaCDM+OmegaNU
    
    #computes the DM-NU cross-P(k)
    Pk=PSL.cross_power_spectrum_DM(Pos1,Pos2,Pos2,OmegaCDM,OmegaNU,dims,BoxSize)
    
    #write cross-P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

############################### CDM-SO_HALOS ################################

elif obj=='CDM-halos':
    #read CDM positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel
        print 'option not implemented for SO halos'

    #read CDM halo positions
    Pos2=HL.halo_positions(groups_fname,groups_number,
                           mass_interval,min_mass,max_mass) #Mpc/h
    
    #computes the DM-halos cross-P(k)
    Pk=PSL.cross_power_spectrum(Pos1,Pos2,dims,BoxSize)
    
    #write cross-P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

############################### CDM-SUBHALOS ###############################

elif obj=='CDM-subhalos':
    #read CDM positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h

    #read CDM halo positions (and velocities for RSD)
    if do_RSD:
        [Pos2,Vel2]=HL.subhalo_positions(groups_fname,groups_number,
                                         mass_interval,min_mass,max_mass,
                                         velocities=True) #Mpc/h
    else:
        Pos2=HL.subhalo_positions(groups_fname,groups_number,
                                  mass_interval,min_mass,max_mass,
                                  velocities=False) #Mpc/h

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel

        RSD(Pos2[:,axis],Vel2[:,axis],Hubble,redshift); del Vel2
    
    #computes the DM-halos cross-P(k)
    Pk=PSL.cross_power_spectrum(Pos1,Pos2,dims,BoxSize)
    
    #write cross-P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

############################## CDM-FoF_HALOS ###############################

elif obj=='CDM-FoF_halos':
    #read CDM positions
    Pos1=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h

    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos1[:,axis],Vel[:,axis],Hubble,redshift); del Vel
        print 'option not implemented for FoF halos'

    #read FoF CDM halo positions
    Pos2=HL.FoF_halo_positions(groups_fname,groups_number,mass_interval,
                            min_mass,max_mass) #Mpc/h
    
    #computes the DM-halos cross-P(k)
    Pk=PSL.cross_power_spectrum(Pos1,Pos2,dims,BoxSize)
    
    #write cross-P(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

############################ QUADRUPOLE CDM ONLY #############################

elif obj=='quadrupole-CDM':
    #read CDM positions
    Pos=readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
    
    #move to redshift-space 
    if do_RSD:
        Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s
        RSD(Pos[:,axis],Vel[:,axis],Hubble,redshift); del Vel

    #computes P_2(k)
    ell=2
    Pk=PSL.multipole(Pos,dims,BoxSize,ell,shoot_noise_correction=False)
    
    #write CDM P_2(k) file
    f=open(f_out,'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+'\n')
    f.close()

#############################################################################

else:
    print 'Wrong option. Choose among:'
    print 'DM, CDM, NU or halos'
    sys.exit()

