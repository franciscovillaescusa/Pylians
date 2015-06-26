import numpy as np
import readsnap
import sys
import cosmology_library as CL
import HI_library as HIL
import scipy.integrate as si
                

################################# UNITS #######################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

Mpc=3.0856e24 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams

pi=np.pi
###############################################################################

################################ INPUT ########################################
if len(sys.argv)>1:
    sa=sys.argv

    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=int(sa[3])
    method=sa[4]

    fac=float(sa[5]); HI_frac=float(sa[6]); Omega_HI_ref=float(sa[7])
    method_Bagla=int(sa[8]); long_ids_flag=bool(int(sa[9]))
    SFR_flag=bool(int(sa[10])); f_MF=sa[11]
    
    divisions=int(sa[12]); cells=int(sa[13]); threads=int(sa[14])
    f_HI_bins=int(sa[15])

    f_cd=sa[16]; f_out=sa[17]

    print '################# INFO ##############'
    print sa

else:
    #snapshot and halo catalogue
    snapshot_fname='../Efective_model_15Mpc/snapdir_008/snap_008'
    groups_fname='../Efective_model_15Mpc/FoF_0.2'
    groups_number=8
    
    #'Dave','method_1','Bagla','Barnes','Paco','Nagamine','Rahmati'
    method='Bagla'

    #1.362889 (60 Mpc/h z=3) 1.436037 (30 Mpc/h z=3) 1.440990 (15 Mpc/h z=3)
    #1.369705 (60 Mpc/h z=3 winds)
    fac=1.440990 #factor to obtain <F> = <F>_obs from the Lya : only for Dave
    HI_frac=0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref=1e-3 #for method_1 and Bagla
    method_Bagla=3 #only for Bagla
    long_ids_flag=False; SFR_flag=True #flags for reading the FoF file
    f_MF='../mass_function/ST_MF_z=3.dat' #file containing the mass function

    divisions=1 #number of divisions to do to the BoxSize to compute the N_HI

    cells=1000 #the LOS grid will have cells x cells elements 
    threads=20 #number of openmp threads
    f_HI_bins=300 #bins for computing f_HI
    
    #file with the projected N_HI values
    f_cd='N_HI_Bagla_15Mpc_subplot_z=3.dat' 
    f_out='f_HI_Dave_15Mpc_z=3.dat'
###############################################################################
TREECOOL_file='/home/villa/bias_HI/TREECOOL_bestfit_g1.3'

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
print 'Total number of particles in the simulation:',Ntotal

#sort the pos array
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 #normalized
pos_unsort=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort
del pos_unsort, ID_unsort

#sort the R array (note that only gas particles have an associated R)
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized
R_unsort=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3 #Mpc/h
R=np.zeros(Ntotal,dtype=np.float32); R[ID_unsort]=R_unsort
del R_unsort, ID_unsort

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
elif method=='Rahmati':
    [IDs,M_HI]=HIL.Rahmati_HI_assignment(snapshot_fname,fac,TREECOOL_file,
                                         Gamma_UVB=None,correct_H2=True)
elif method=='Bagla':
    [IDs,M_HI]=HIL.Bagla_HI_assignment(snapshot_fname,groups_fname,
                                       groups_number,Omega_HI_ref,method_Bagla,
                                       f_MF,long_ids_flag,SFR_flag)
else:
    print 'Incorrect method selected!!!'; sys.exit()

#keep only the particles having HI masses
M_HI=M_HI[IDs]; pos=pos[IDs]; R=R[IDs]; del IDs
X=pos[:,0]; Y=pos[:,1]; Z=pos[:,2]; del pos

#compute the value of Omega_HI
print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

#find the border size: to avoid problems with boundary conditions, we 
#restrict our LOS region to X,Y=[Border_size,BoxSize-Border_size]
Border_size=np.max(R) #Mpc/h
print 'Border size = %f Mpc/h'%Border_size

#compute the value of dX ----> dX/dz = H0*(1+z)^2/H(z)
dX=CL.absorption_distance(Omega_m,Omega_l,redshift,BoxSize/divisions)
print 'dX=%f\n'%dX

#keep only with the LOS that are not affected by boundary conditions
xy_min=int(Border_size*cells/BoxSize)
xy_max=int((BoxSize-Border_size)*cells/BoxSize)+1
indexes_los=np.empty((xy_max-xy_min+1)**2,dtype=np.int32)

offset=0; length=xy_max-xy_min+1; numbers=np.arange(xy_min,xy_max+1)
for i in xrange(xy_min,xy_max+1):
    indexes_los[offset:offset+length]=(cells*i+numbers); offset+=length
del numbers

#compute f_HI = #_of_absorbers / dN_HI / dX
log10_N_HI_min=16.0; log10_N_HI_max=23.0
bins_histo=np.logspace(log10_N_HI_min,log10_N_HI_max,f_HI_bins+1)
delta_bins_histo=bins_histo[1:]-bins_histo[:-1] #dN_HI
N_HI_histo=10**(0.5*(np.log10(bins_histo[:-1])+np.log10(bins_histo[1:])))
f_HI=np.zeros(f_HI_bins,dtype=np.float64)

total_los=len(indexes_los)*divisions
prefactor=(Msun/h)/mH/(Mpc/h/(1.0+redshift))**2
for i in range(divisions):

    z_min=BoxSize*i*1.0/divisions; z_max=BoxSize*(i+1)*1.0/divisions
    indexes=np.where((Z>=z_min) & (Z<z_max))[0]

    #make a subplot
    BoxSize=1.0; X_min=13.5; Y_min=9.0 #everything in Mpc/h
    indexes=np.where((X>X_min) & (X<X_min+BoxSize) & \
                     (Y>Y_min) & (Y<Y_min+BoxSize) & \
                     (Z>=z_min) & (Z<z_max))[0]
    X-=X_min; Y-=Y_min

    N_HI=HIL.NHI_los_sph(R[indexes],X[indexes],Y[indexes],M_HI[indexes],
                         BoxSize,cells,threads)*prefactor

    #write column density file
    f=open(f_cd,'w')
    for l in xrange(len(N_HI)):
        #y=l/cells*BoxSize*1.0/cells; x=l%cells*BoxSize*1.0/cells
        y=Y_min+l/cells*BoxSize*1.0/cells; x=X_min+l%cells*BoxSize*1.0/cells
        f.write(str(x)+' '+str(y)+' '+str(N_HI[l])+'\n')
    f.close()
    sys.exit()

    #take only the LOS not affected by the boundary conditions
    N_HI=N_HI[indexes_los]

    N_DLAs=len(np.where(N_HI>10**(20.3))[0]) #number of DLAs found 
    #compute dN/dX = number of DLAs per absorption distance
    print '\ndN/dX = %f'%(N_DLAs*1.0/len(N_HI)/dX) 
    print '%e < N_HI < %e\n'%(np.min(N_HI),np.max(N_HI))
    f_HI+=np.histogram(N_HI,bins=bins_histo)[0]/delta_bins_histo/(total_los*dX)
del R,X,Y,M_HI,indexes

#sanity check
print 'dN/dX = %f'%(HIL.incidence_rate(N_HI_histo,f_HI))

#write output file
f=open(f_out,'w')
for i in range(f_HI_bins):
    f.write(str(bins_histo[i])+' '+str(f_HI[i])+'\n')
f.close()

