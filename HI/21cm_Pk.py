import numpy as np
import readsnap
import CIC_library as CIC
import sys
import scipy.weave as wv
import Power_spectrum_library as PSL
import HI_library as HIL

#Pos is an array containing the positions of the particles along one axis
#Vel is an array containing the velocities of the particle along the above axis
def RSD(pos,vel,Hubble,redshift,axis):
    #transform coordinates to redshift space
    delta_y=(vel[:,axis]/Hubble)*(1.0+redshift)  #displacement in Mpc/h
    pos[:,axis]+=delta_y #add distorsion to position of particle in real-space
    del delta_y

    #take care of the boundary conditions
    beyond=np.where(pos[:,axis]>BoxSize)[0]; pos[beyond,axis]-=BoxSize
    beyond=np.where(pos[:,axis]<0.0)[0];     pos[beyond,axis]+=BoxSize
    del beyond

################################# UNITS #####################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

Mpc=3.0856e24 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams

pi=np.pi
#############################################################################

################################ INPUT ######################################
if len(sys.argv)>1:
    sa=sys.argv
    
    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=int(sa[3])
    method=sa[4]

    fac=float(sa[5]); HI_frac=float(sa[6]); Omega_HI_ref=float(sa[7])
    method_Bagla=int(sa[8]); long_ids_flag=bool(int(sa[9]))
    SFR_flag=bool(int(sa[10])); f_MF=sa[11]
    
    dims=int(sa[12]); f_out=sa[13]

    print '################# INFO ##############\n',sa

else:
    snapshot_fname='../Efective_model_60Mpc/snapdir_008/snap_008'
    groups_fname='../Efective_model_60Mpc/FoF_0.2'
    groups_number=8

    #'Dave','method_1','Bagla','Barnes','Paco','Nagamine'
    method='Bagla'

    #1.362889 (60 Mpc/h z=3) 1.436037 (30 Mpc/h z=3) 1.440990 (15 Mpc/h z=3)
    fac=1.440990 #factor to obtain <F> = <F>_obs from the Lya : only for Dave
    HI_frac=0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref=1e-3 #for method_1, Bagla and Paco and for computing x_HI
    method_Bagla=3 #only for Bagla
    long_ids_flag=False; SFR_flag=True #flags for reading the FoF file
    f_MF='../mass_function/ST_MF_z=3.dat' #file containing the mass function

    dims=512

    f_out='Pk_21cm_Bagla_60Mpc_z=3.dat2'

    #f_out=['Pk_21cm_Paco_120Mpc_X_z=3.dat',
    #       'Pk_21cm_Paco_120Mpc_Y_z=3.dat',
    #       'Pk_21cm_Paco_120Mpc_Z_z=3.dat',
    #       'Pk_21cm_Paco_120Mpc_z=3.dat']
#############################################################################
#div=3 #number of divisions in 1 dimension to compute P(k) on very small scales

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h=head.hubble

#find the total number of particles in the simulation
Ntotal=np.sum(Nall,dtype=np.uint64)
print 'Total number of particles in the simulation =',Ntotal

#sort the pos and vel array
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 #normalized
print 'sorting the POS array...'
pos_unsort=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort; del pos_unsort
print 'sorting the VEL array...'
vel_unsort=readsnap.read_block(snapshot_fname,"VEL ",parttype=-1) #km/s
vel=np.empty((Ntotal,3),dtype=np.float32); vel[ID_unsort]=vel_unsort; del vel_unsort
del ID_unsort

#find the IDs and HI masses of the particles to which HI has been assigned
if method=='Dave':
    [IDs,M_HI]=HIL.Dave_HI_assignment(snapshot_fname,HI_frac,fac)
elif method=='method_1': 
    [IDs,M_HI]=HIL.method_1_HI_assignment(snapshot_fname,HI_frac,Omega_HI_ref)
elif method=='Barnes':
    [IDs,M_HI]=HIL.Barnes_Haehnelt(snapshot_fname,groups_fname,
                                   groups_number,long_ids_flag,SFR_flag)
elif method=='Paco':
    [IDs,M_HI]=HIL.Paco_HI_assignment(snapshot_fname,groups_fname,
                                      groups_number,long_ids_flag,SFR_flag)
elif method=='Nagamine':
    [IDs,M_HI]=HIL.Nagamine_HI_assignment(snapshot_fname,
                                          correct_H2=False)
elif method=='Bagla':
    [IDs,M_HI]=HIL.Bagla_HI_assignment(snapshot_fname,groups_fname,
                                       groups_number,Omega_HI_ref,method_Bagla,
                                       f_MF,long_ids_flag,SFR_flag)
else:
    print 'Incorrect method selected!!!'; sys.exit()

#keep only the particles having HI masses
M_HI=M_HI[IDs]; pos=pos[IDs]; vel=vel[IDs]; del IDs

#mean HI mass per grid point
mean_M_HI=np.sum(M_HI,dtype=np.float64)/dims**3
print '< M_HI > = %e'%(mean_M_HI)
print 'Omega_HI = %e'%(mean_M_HI*dims**3/BoxSize**3/rho_crit)

#compute \delta T_b(z)---> prefactor to compute \delta T_b(x)
#note that when computing M_H we have to use the total Omega_B, not only the
#Hydrogen from the gas particles. Notice that the brigthness temperature excess
#will be computed as: delta_Tb = <delta_Tb> * M_HI/<M_HI>
#Therefore, the value of <M_HI> used to compute X_HI has to be the same of this
#used when computing M_HI/<M_HI>. We just take the <M_HI> of the simulation
Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b=Omega_m-Omega_cdm
X_HI=np.sum(M_HI,dtype=np.float64)/(0.76*Omega_b*rho_crit*BoxSize**3) #HI/H
mean_delta_Tb=23.44*(Omega_b*h**2/0.02)\
    *np.sqrt(0.15*(1.0+redshift)/(10.0*Omega_m*h**2))*X_HI #mK
print '\nOmega_CDM=',Omega_cdm; print 'Omega_B  =',Omega_b; 
print 'X_HI (simulation) =',X_HI; print 'mean_delta_Tb =',mean_delta_Tb,'mK'


Pk_axis=[]
for axis in range(0,3):

    print '\nComputing the 21 cm P(k) along axis:',axis

    #create a copy of the pos array
    pos_RSD=np.copy(pos)

    #do RSD along the axis
    RSD(pos_RSD,vel,Hubble,redshift,axis)

    #compute the value of M_HI in each grid point
    M_HI_grid=np.zeros(dims**3,dtype=np.float32)
    CIC.CIC_serial(pos_RSD,dims,BoxSize,M_HI_grid,M_HI); del pos_RSD
    print 'Omega_HI = %e'\
        %(np.sum(M_HI_grid,dtype=np.float64)/BoxSize**3/rho_crit)

    #we assume that Ts>>T_CMB
    delta_Tb=mean_delta_Tb*M_HI_grid/mean_M_HI
    #*Hubble/(Hubble+(1.0+redshift)*dVdr)
    print 'delta_Tb [mK] =',delta_Tb
    print '%f < delta_Tb [mK] < %f'%(np.min(delta_Tb),np.max(delta_Tb))

    #compute 21 cm P(k)
    Pk=PSL.power_spectrum_given_delta(delta_Tb,dims,BoxSize)
    Pk_axis.append(Pk[1])
    
    """#write P(k) to output file
    f=open(f_out[axis],'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+'\n')
    f.close()"""
Pk_axis=np.array(Pk_axis)

k=Pk[0]; f=open(f_out,'w')
for i in range(len(k)):
    f.write(str(k[i])+' '+str(np.mean(Pk_axis[:,i]))+'\n')
f.close()



