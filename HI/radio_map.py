import numpy as np
import cosmology_library as cl
import readsnap
import CIC_library as CIC
import scipy.weave as wv
import scipy.fftpack
import time
import sys

#This functions computes the value of |k| of each mode, and returns W(kR)
def smoothing(dims,BoxSize,R):
    smooth_array=np.empty(dims**2,dtype=np.float32)
    R=np.array([R]); BoxSize=np.array([BoxSize])

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int middle=dims/2;
       int i,j;
       float kR;

       for (long l=0;l<dims2;l++){
           i=l/dims;
           j=l%dims;

           i = (i>middle) ? i-dims : i;
           j = (j>middle) ? j-dims : j;

           kR=sqrt(i*i+j*j)*2.0*M_PI/BoxSize(0)*R(0);
           smooth_array(l)=exp(-kR*kR/2.0);
       } 
       printf("%f %f %f \\n",BoxSize(0),R(0),M_PI);
    """
    wv.inline(code,['dims','smooth_array','R','BoxSize'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return smooth_array

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

pi=np.pi

kB=1.38e3 #mJy*m^2/mK
c=3e8 #m/s
#############################################################################

################################### INPUT #####################################
snapshot_fname='Efective_model_60Mpc/Corrected_Snapshot/snapdir_008/snap_008'

ang_res=4.0 #arc-minutes 4.0

nu0=1420.0          #21 cm frequency MHz
channel_width=0.125 #MHz

dims=1024

axis=0

f_out='radio_map_60Mpc_0.5min.dat'
###############################################################################

## 1) READ THE PROPERTIES OF THE SNAPSHOT: BOXSIZE, Z, NALL .... ##
print '\nREADING SNAPSHOTS PROPERTIES'

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
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
print '\nComoving distance to z=%2.2f : %4.2f Mpc/h'%(z,r)

#compute maximum/minimum frequencies of the channel and delta_r
print 'Observed frequency from z=%2.2f : %2.1f MHz'%(z,nu0/(1.0+z))
nu_min=nu0/(1.0+z)               #minimum frequency of the channel
nu_max=nu0/(1.0+z)-channel_width #maximum frequency of the channel
z_min=z; z_max=nu0/nu_max-1.0;
print 'Channel redshift interval: %1.4f < z < %1.4f'%(z_min,z_max)
delta_r=cl.comoving_distance(z_max,Omega_m,Omega_l)-r
print 'delta_r channel = %2.2f Mpc/h'%delta_r
grid_res=(ang_res/60.0)*(pi/180.0)*r

#grid resolution
print '\nSpatial resolution = %2.3f Mpc/h'%grid_res

#read HI/H fractions and masses of the gas particles
nH0 =readsnap.read_block(snapshot_fname,"NH  ",parttype=0)      #HI/H
mass=readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h

#compute the HI mass in each gas particle
M_HI=0.76*nH0*mass
Omega_HI=np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit
print '\nOmega_HI = %e'%Omega_HI

#mean value of M_HI per grid point
mean_M_HI=np.sum(0.76*nH0*mass,dtype=np.float64)/dims**3; del nH0,mass
print 'Total HI mass = %e'%(np.sum(M_HI,dtype=np.float64))
print '< M_HI > = %e Msun/h'%(mean_M_HI)
print 'Omega_HI = %e'%(mean_M_HI*dims**3/BoxSize**3/rho_crit)

#compute \delta T_b(z)---> prefactor to compute \delta T_b(x)
#note that when computing M_H we have to use the total Omega_B, not only the
#Hydrogen from the gas particles
Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b=Omega_m-Omega_cdm
X_HI=np.sum(M_HI,dtype=np.float64)/(0.76*Omega_b*rho_crit*BoxSize**3)
mean_delta_Tb=23.44*(Omega_b*h**2/0.02)*np.sqrt(0.15*(1.0+z)/(10.0*Omega_m*h**2))*X_HI #mK
print '\nOmega_CDM=',Omega_cdm
print 'Omega_B  =',Omega_b
print 'X_HI =',X_HI
print 'mean_delta_Tb =',mean_delta_Tb,'mK\n'


#read positions, HI/H fractions and masses of the gas particles
pos=readsnap.read_block(snapshot_fname,"POS ",parttype=0)/1e3 #Mpc/h
vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=0)     #km/s

#do RSD along the axis
RSD(pos[:,axis],vel[:,axis],Hubble,z); del vel

#mean HI mass per grid cell. Note that the cells have a volume equal to:
#delta_r x BoxSize/dims x BoxSize/dims (Mpc/h)^3
mean_M_HI_grid=np.sum(M_HI,dtype=np.float64)/BoxSize**3*\
    (delta_r*(BoxSize/dims)**2)
print '\nmean HI mass per grid cell = %e Msun/h / (Mpc/h)^3'%(mean_M_HI_grid)

#take a slice of width delta_r
indexes=np.where(pos[:,0]<delta_r)[0]
pos=pos[indexes][:,1:3]; M_HI=M_HI[indexes]
print '\nLimits of the selected slice:'
print np.min(pos[:,0]),'< Y <',np.max(pos[:,0])
print np.min(pos[:,1]),'< Z <',np.max(pos[:,1])

#compute the value of M_HI in each grid point
M_HI_grid=np.zeros(dims**2,dtype=np.float32) 
CIC.CIC_serial_2D(pos,dims,BoxSize,M_HI_grid,M_HI) #compute surface densities
print '%e should be equal to %e'%(np.sum(M_HI,dtype=np.float64),np.sum(M_HI_grid,dtype=np.float64))
print 'Omega_HI (slice) = %e'%(np.sum(M_HI_grid,dtype=np.float64)/(BoxSize**2*delta_r)/rho_crit)

#compute delta_HI in the grid cells
delta_HI=M_HI_grid/mean_M_HI_grid-1.0
print np.min(delta_HI),'< delta_HI <',np.max(delta_HI)
print '<delta_HI> = %f'%np.mean(delta_HI,dtype=np.float64)

#we assume that Ts>>T_CMB
delta_Tb=mean_delta_Tb*(1.0+delta_HI)
#*Hubble/(Hubble+(1.0+redshift)*dVdr)
print np.min(delta_Tb),'< delta_Tb (mK) <',np.max(delta_Tb)
print '<delta_Tb> = %e (mK)'%(np.mean(delta_Tb,dtype=np.float64))

#create an image
"""
f=open('borrar_temperature.dat','w')
for i in range(dims**2):
    x=(BoxSize/dims)*(i/dims)
    y=(BoxSize/dims)*(i%dims)
    f.write(str(x)+' '+str(y)+' '+str(delta_Tb[i])+'\n')
f.close()
"""

#now compute specific intensity
prefactor=2.0*kB*(nu0*1e6/(1.0+z))**2/c**2 #mJy
print '\nprefactor=',prefactor
Inu=prefactor*delta_Tb
print 'sum of Inu = %e mJy/sr'%np.sum(Inu,dtype=np.float64)
print np.min(Inu),'< I_nu (mJy/sr) <',np.max(Inu)
print '<I_nu> = %e mJy/sr'%(np.mean(Inu,dtype=np.float64))
print mean_delta_Tb*prefactor
I_nu_mean=4.76/(pi/180.0)**2*h*Omega_HI*100.0/Hubble
print '<I_nu> = %e Jy/sr'%I_nu_mean

"""
#create an image
f=open('borrar.dat','w')
for i in range(dims**2):
    x=(BoxSize/dims)*(i/dims)
    y=(BoxSize/dims)*(i%dims)
    f.write(str(x)+' '+str(y)+' '+str(Inu[i])+'\n')
f.close()
"""

#compute the smooth array
print '\nSmoothing the field...'
smooth=smoothing(dims,BoxSize,grid_res)
print np.min(smooth),'< smoothing <',np.max(smooth)
smooth=np.reshape(smooth,(dims,dims))

#Fourier transform the intensity field
Inu=np.reshape(Inu,(dims,dims))
print 'Computing the FFT of the field...'; start_fft=time.clock()
Inu_k=scipy.fftpack.fftn(Inu,overwrite_x=True); del Inu
print 'done: time taken for computing the FFT=',time.clock()-start_fft

#smooth the density field
Inu_k*=smooth

#compute Fourier transform back to obtain Inu smoothed
print 'Computing the inverse FFT of the field...'; start_fft=time.clock()
Inu=scipy.fftpack.ifftn(Inu_k,overwrite_x=True); del Inu_k
print 'done: time taken for computing the FFT=',time.clock()-start_fft
Inu=np.ravel(Inu)
print np.min(Inu.imag),'< Inu.imag <',np.max(Inu.imag)

Inu=np.real(Inu) #just keep with the real part
print np.min(Inu),'< I_nu (mJy/sr) <',np.max(Inu)
print 'sum of Inu = %e mJy/sr'%np.sum(Inu,dtype=np.float64)
print 'points with Inu < 0.0  ---> ',len(np.where(Inu<0.0)[0])

#remove negative values
Inu[np.where(Inu<0.0)[0]]=0.0

#Move from Jy/sr to Jy/beam
Inu*=(ang_res/60.0*pi/180.0)**2
print 'peak flux = %e mJy/beam'%(np.max(Inu))

"""
#create an image
f=open(f_out,'w')
for i in range(dims**2):
    x=(BoxSize/dims)*(i/dims)
    y=(BoxSize/dims)*(i%dims)
    f.write(str(x)+' '+str(y)+' '+str(Inu[i])+'\n')
f.close()
"""
