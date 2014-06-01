import numpy as np
import cosmology_library as cl
import HI_library as HIL
import readsnap
import readfof
import CIC_library as CIC
import scipy.weave as wv
import scipy.fftpack
import time
import sys


#Pos is an array containing the positions of the particles along one axis
#Vel is an array containing the velocities of the particle along the above axis
def RSD(pos,vel,Hubble,redshift):
    #transform coordinates to redshift space
    delta_y=(vel/Hubble)*(1.0+redshift)  #displacement in Mpc/h
    pos+=delta_y #add distorsion to position of particle in real-space
    del delta_y

    #take care of the boundary conditions
    beyond=np.where(pos>BoxSize)[0]; pos[beyond]-=BoxSize
    beyond=np.where(pos<0.0)[0];     pos[beyond]+=BoxSize
    del beyond

################################# UNITS #####################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
Mpc=3.0856e24 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams
nu0=1420.0          #21-cm frequency in MHz
pi=np.pi
kB=1.38e3 #mJy*m^2/mK
c=3e8 #m/s
#############################################################################

################################### INPUT ###################################
if len(sys.argv)>1:
    sa=sys.argv

    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=int(sa[3])
    method=sa[4]

    fac=float(sa[5]); HI_frac=float(sa[6]); Omega_HI_ref=float(sa[7])
    method_Bagla=int(sa[8]); long_ids_flag=bool(int(sa[9]))
    SFR_flag=bool(int(sa[10])); f_MF=sa[11]
    
    ang_res=float(sa[12]); channel_width=float(sa[13]); axis=int(sa[14])
    dims=int(sa[15]); number_slice=int(sa[16])

    f_out=sa[17]

    print '################# INFO ##############'
    for element in sa:
        print element

    if number_slice<0:
        number_slice=None

else:
    snapshot_fname='../Efective_model_60Mpc/snapdir_013/snap_013'
    groups_fname='../Efective_model_60Mpc/FoF_0.2'
    groups_number=13
    
    #'Dave','method_1','Bagla','Barnes','Paco'
    method='Paco'

    #1.362889 (60 Mpc/h z=3) 1.436037 (30 Mpc/h z=3) 1.440990 (15 Mpc/h z=3)
    fac=1.362889 #factor to obtain <F> = <F>_obs from the Lya : only for Dave
    HI_frac=0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref=1e-3
    method_Bagla=3
    long_ids_flag=False; SFR_flag=True
    f_MF='../mass_function/ST_MF_z=2.4.dat'

    ang_res=2 #arc-minutes
    channel_width=0.500  #MHz
    axis=1 #axis along which make the redshift-space distortion

    dims=25

    number_slice=None #if none it will look the one with the highest peak flux

    f_out='RM_Paco_2arcmin_z=2.4.dat_25bins'
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
z=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+z)**3+Omega_l)  #km/s/(Mpc/h)
h=head.hubble

#comoving distance to redshift z and spatial resolution
r=cl.comoving_distance(z,Omega_m,Omega_l)
print '\nComoving distance to z=%2.3f : %4.2f Mpc/h'%(z,r)

#compute maximum/minimum frequencies of the channel and delta_r
print 'Observed frequency from z=%2.2f : %2.2f MHz'%(z,nu0/(1.0+z))
nu_min=nu0/(1.0+z)               #minimum frequency of the channel
nu_max=nu0/(1.0+z)-channel_width #maximum frequency of the channel
z_min=z; z_max=nu0/nu_max-1.0;
print 'Channel redshift interval: %1.4f < z < %1.4f'%(z_min,z_max)
print 'Channel frequency [%1.3f - %1.3f] MHz'%(nu_max,nu_min)
delta_r=cl.comoving_distance(z_max,Omega_m,Omega_l)-r
print 'delta_r channel = %2.2f Mpc/h'%delta_r
grid_res=(ang_res/60.0)*(pi/180.0)*r

#grid resolution
print '\nSpatial resolution = %2.3f Mpc/h'%grid_res

#find the total number of particles in the simulation
Ntotal=np.sum(Nall,dtype=np.uint64)
print '\nTotal number of particles in the simulation:',Ntotal

#sort the pos and vel array
ID_unsort =readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 #normalized
pos_unsort=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
vel_unsort=readsnap.read_block(snapshot_fname,"VEL ",parttype=-1)     #km/s
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort
vel=np.empty((Ntotal,3),dtype=np.float32); vel[ID_unsort]=vel_unsort
del pos_unsort, vel_unsort, ID_unsort

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

#just keep with the particles having HI masses
M_HI=M_HI[IDs]; pos=pos[IDs]; vel=vel[IDs]; del IDs

#compute the value of Omega_HI
print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

#compute \delta T_b(z)---> prefactor to compute \delta T_b(x)
#note that when computing M_H we have to use the total Omega_B, not only the
#Hydrogen from the gas particles
Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b=Omega_m-Omega_cdm
X_HI=np.sum(M_HI,dtype=np.float64)/(0.76*Omega_b*rho_crit*BoxSize**3)
mean_delta_Tb=\
    23.44*(Omega_b*h**2/0.02)*np.sqrt(0.15*(1.0+z)/(10.0*Omega_m*h**2))*X_HI #mK
print '\nOmega_CDM = %2.3f : Omega_B = %2.3f'%(Omega_cdm,Omega_b)
print 'X_HI = %2.4f ---> mean_delta_Tb = %2.4f mK \n'%(X_HI,mean_delta_Tb)

#do RSD along the axis
RSD(pos[:,axis],vel[:,axis],Hubble,z); del vel

#mean HI mass per grid cell. Note that the cells have a volume equal to:
#delta_r x BoxSize/dims x BoxSize/dims (Mpc/h)^3
mean_M_HI_grid=np.sum(M_HI,dtype=np.float64)/BoxSize**3*\
    (delta_r*(BoxSize/dims)**2)
print 'mean HI mass per grid cell = %e Msun/h / (Mpc/h)^3'%mean_M_HI_grid

#find the slice with the maximum value of the flux
print '[Zmin-Zmax] Mpc/h    <Inu> [mJy/beam]    peak flux [mJy/beam]'
peak_flux_max=0.0; slice_number=0; i=0
axis_min=0.0; axis_max=axis_min+delta_r 
while(axis_max<BoxSize):
    indexes=np.where((pos[:,axis]>axis_min) & (pos[:,axis]<axis_max))[0]
    pos2=np.empty((len(indexes),2),dtype=np.float32)
    pos2[:,0]=pos[indexes,(axis+1)%3]; pos2[:,1]=pos[indexes,(axis+2)%3]
    Inu_slice=HIL.peak_flux(pos2,M_HI[indexes],BoxSize,delta_r,z,dims,
                            mean_delta_Tb,mean_M_HI_grid,ang_res,grid_res,False)
    #substract the mean flux
    mean_flux=np.mean(Inu_slice,dtype=np.float64)
    Inu_slice=Inu_slice-mean_flux

    #compute the peak flux value
    peak_flux=np.max(Inu_slice)
    print ' [%6.2f - %6.2f] %15.3e %18.3e'\
        %(axis_min,axis_max,mean_flux,peak_flux)

    #keep with the Inu of the desired slice or this with the maximum peak flux
    if peak_flux>peak_flux_max:
        peak_flux_max=peak_flux; slice_number=i
        if number_slice==None:
            Inu=Inu_slice
    if i==number_slice:
        Inu=Inu_slice
    i+=1
    axis_min=axis_max; axis_max=axis_min+delta_r

#write some info about the selected slice
print '\nselected slice number =',number_slice
print 'edges of the selected slice ---> [%2.2f-%2.2f] Mpc/h'\
    %(slice_number*delta_r,(slice_number+1)*delta_r)
print 'peak flux from the slice = %1.3e mJy/beam'%(np.max(Inu))
print 'number of the slice with the highest peak flux =',slice_number
print 'peak flux from of slices = %1.3e mJy/beam'%peak_flux_max

#create an image
f=open(f_out,'w')
for i in range(dims**2):
    x=(BoxSize/dims)*(i/dims)
    y=(BoxSize/dims)*(i%dims)
    f.write(str(x)+' '+str(y)+' '+str(Inu[i])+'\n')
f.close()



