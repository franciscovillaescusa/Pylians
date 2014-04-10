import numpy as np
import readsnap
import CIC_library as CIC
import sys
import scipy.weave as wv
import Power_spectrum_library as PSL

def derivative_4nd(field,axis,dims,BoxSize):

    if len(field)!=dims**3:
        print 'lengths are different!!!'
        sys.exit()

    deriv=np.empty(dims**3,dtype=np.float32)

    support = "#include <math.h>"
    code = """
       int dims2=dims*dims;
       int dims3=dims2*dims;
       int i,j,k,k_max2,k_max,k_min,k_min2;

       for (long l=0;l<dims3;l++){
         i=l/dims2; j=(l%dims2)/dims; k=(l%dims2)%dims;
         k_max2=dims2*i+dims*j+((k+2)%dims); 
         k_max =dims2*i+dims*j+((k+1)%dims); 
         k_min =dims2*i+dims*j+((k-1)%dims); 
         k_min2=dims2*i+dims*j+((k-2)%dims); 
         deriv(l)=-field(k_max2)+8.0*field(k_max)-8.0*field(k_min)+field(k_min2);
       }
    """
    wv.inline(code,['dims','field','axis','deriv'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    deriv/=(12.0*BoxSize/dims)

    return deriv

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
#############################################################################

################################ INPUT ######################################
snapshot_fname='Efective_model_30Mpc/Corrected_Snapshot/snapdir_008/snap_008'
groups_fname='Efective_model'
groups_number=8

dims=512

div=3 #number of divisions in 1 dimension to compute P(k) on very small scales

f_out=['Pk_21cm_30Mpc_X_z=3.dat_borrar',
       'Pk_21cm_30Mpc_Y_z=3.dat_borrar',
       'Pk_21cm_30Mpc_Z_z=3.dat_borrar',
       'Pk_21cm_30Mpc_z=3.dat_borrar']
#############################################################################


## 1) READ THE PROPERTIES OF THE SNAPSHOT: BOXSIZE, Z, NALL .... ##
print '\nREADING SNAPSHOTS PROPERTIES'

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h=head.hubble

#read HI/H fractions and masses of the gas particles
nH0 =readsnap.read_block(snapshot_fname,"NH  ",parttype=0)      #HI/H
mass=readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h

#compute the HI mass in each gas particle
M_HI=0.76*nH0*mass
print '\nOmega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

#mean value of M_HI per grid point
mean_M_HI=np.sum(0.76*nH0*mass,dtype=np.float64)/dims**3; del nH0,mass
print '< M_HI > = %e'%(mean_M_HI)
print 'Omega_HI = %e'%(mean_M_HI*dims**3/BoxSize**3/rho_crit)

#compute \delta T_b(z)---> prefactor to compute \delta T_b(x)
#note that when computing M_H we have to use the total Omega_B, not only the
#Hydrogen from the gas particles
Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
Omega_b=Omega_m-Omega_cdm
X_HI=np.sum(M_HI,dtype=np.float64)/(0.76*Omega_b*rho_crit*BoxSize**3)
mean_delta_Tb=23.44*(Omega_b*h**2/0.02)*np.sqrt(0.15*(1.0+redshift)/(10.0*Omega_m*h**2))*X_HI #mK
print '\nOmega_CDM=',Omega_cdm
print 'Omega_B  =',Omega_b
print 'X_HI =',X_HI
print 'mean_delta_Tb =',mean_delta_Tb,'mK'

"""
Pk_axis=[]
for axis in range(0,3):

    print '\nComputing the 21 cm P(k) along axis:',axis

    #read positions, HI/H fractions and masses of the gas particles
    pos=readsnap.read_block(snapshot_fname,"POS ",parttype=0)/1e3 #Mpc/h
    vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=0)     #km/s

    #do RSD along the axis
    RSD(pos[:,axis],vel[:,axis],Hubble,redshift); del vel

    #compute the value of M_HI in each grid point
    M_HI_grid=np.zeros(dims**3,dtype=np.float32)
    CIC.CIC_serial(pos,dims,BoxSize,M_HI_grid,M_HI); del pos
    print 'Omega_HI = %e'%(np.sum(M_HI_grid,dtype=np.float64)/BoxSize**3/rho_crit)

    #we assume that Ts>>T_CMB
    delta_Tb=mean_delta_Tb*M_HI_grid/mean_M_HI
    #*Hubble/(Hubble+(1.0+redshift)*dVdr)
    print delta_Tb
    print np.min(delta_Tb),np.max(delta_Tb)

    #compute 21 cm P(k)
    Pk=PSL.power_spectrum_given_delta(delta_Tb,dims,BoxSize)
    Pk_axis.append(Pk[1])
    
    #write P(k) to output file
    f=open(f_out[axis],'w')
    for i in range(len(Pk[0])):
        f.write(str(Pk[0][i])+' '+str(Pk[1][i])+' '+'\n')
    f.close()
Pk_axis=np.array(Pk_axis)

k=Pk[0]; f=open(f_out[3],'w')
for i in range(len(k)):
    f.write(str(k[i])+' '+str(np.mean(Pk_axis[:,i]))+'\n')
f.close()
"""

#divide the simulation box in subboxes and compute the power spectrum on them
print '\nComputing the P(k) on the subboxes'
axis=0

#read positions, HI/H fractions and masses of the gas particles
pos=readsnap.read_block(snapshot_fname,"POS ",parttype=0)/1e3 #Mpc/h
vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=0)     #km/s

#do RSD along the axis
RSD(pos[:,axis],vel[:,axis],Hubble,redshift); del vel

mean_M_HI=mean_M_HI/div**3
Pk_axis=[]; X=pos[:,0]; Y=pos[:,1]; Z=pos[:,2]
for i in range(div):
    x_min=i*BoxSize/div; x_max=(i+1)*BoxSize/div
    for j in range(div):
        y_min=j*BoxSize/div; y_max=(j+1)*BoxSize/div
        for k in range(div):
            z_min=k*BoxSize/div; z_max=(k+1)*BoxSize/div

            indexes=np.where((X>=x_min) & (X<x_max) & (Y>=y_min) & (Y<y_max) & (Z>=z_min) & (Z<z_max))[0]

            subpos=pos[indexes]
            subpos[:,0]-=x_min; subpos[:,1]-=y_min; subpos[:,2]-=z_min
            print '\n',i,j,k
            print '%2.2f < X < %2.2f'%(np.min(subpos[:,0]),np.max(subpos[:,0]))
            print '%2.2f < Y < %2.2f'%(np.min(subpos[:,1]),np.max(subpos[:,1]))
            print '%2.2f < Z < %2.2f'%(np.min(subpos[:,2]),np.max(subpos[:,2]))

            #compute the value of M_HI in each grid point
            M_HI_grid=np.zeros(dims**3,dtype=np.float32)
            CIC.CIC_serial(subpos,dims,BoxSize/div,M_HI_grid,M_HI[indexes])
            print 'Omega_HI = %e'%(np.sum(M_HI_grid,dtype=np.float64)/(BoxSize/div)**3/rho_crit)
            
            #we assume that Ts>>T_CMB
            delta_Tb=mean_delta_Tb*M_HI_grid/mean_M_HI
            #*Hubble/(Hubble+(1.0+redshift)*dVdr)
            print delta_Tb
            print np.min(delta_Tb),np.max(delta_Tb)

            #compute 21 cm P(k)
            Pk=PSL.power_spectrum_given_delta(delta_Tb,dims,BoxSize)
            Pk_axis.append(Pk[1])
    
            #write P(k) to output file
            f=open(f_out[0]+'_'+str(i)+str(j)+str(k),'w')
            for l in range(len(Pk[0])):
                f.write(str(Pk[0][l])+' '+str(Pk[1][l])+' '+'\n')
            f.close()
Pk_axis=np.array(Pk_axis)

k=Pk[0]; f=open(f_out[3]+'final','w')
for i in range(len(k)):
    f.write(str(k[i])+' '+str(np.mean(Pk_axis[:,i]))+'\n')
f.close()

"""
######################## COMPUTE THE dV/dt TERM ############################
#read the velocity of the gas particles
Vz=readsnap.read_block(snapshot_fname,"VEL ",parttype=0)[:,2] #km/s

#compute the values of Vz in the grid points
Vz_grid=np.zeros(dims**3,dtype=np.float32) 
CIC.CIC_serial(pos,dims,BoxSize,Vz_grid,Vz) #compute \Sum Vz
del Vz

#compute the density in the grid points
particles=np.zeros(dims**3,dtype=np.float32) 
CIC.CIC_serial(pos,dims,BoxSize,particles) #compute \Sum Vz
particles[np.where(particles==0)[0]]=1.0   #avoid divisions by 0

#compute the average velocity in the grid points
Vz_grid=Vz_grid/particles; del particles
print Vz_grid
print np.min(Vz_grid),np.max(Vz_grid)

#make the derivative 
dVdr=derivative_4nd(Vz_grid,2,dims,BoxSize)
print dVdr
print np.min(dVdr),np.max(dVdr)


"""
