import numpy as np
import readsnap
import readfof
import scipy.weave as wv
import scipy.integrate as si
import scipy.fftpack
import CIC_library as CIC
import sys
import time

############################# routines available ##############################
#Bagla_parameters
     #deriv_Bagla_parameters
#Bagla_HI_assignment
#method_1_HI_assignment
#Dave_HI_assignment
     #Iwdr
     #Volw
#Barnes_Haehelt_HI_assignment
#Paco_HI_assignment
#Nagamine_HI_assignment
#Rahmati_HI_assignment
#NHI_los_sph
     #I1
     #I2
     #Iwdl
#cross_section_halo
#cross_section_BH
     #column_density
     #deriv_column_density
     #analytic_integral
     #integral
     #deriv_HI_BH
#incidence_rate
     #deriv_incidence_rate
#HI_from_UVB
#particle_indexes
#peak_flux
     #smoothing

###############################################################################

################################# UNITS #####################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

yr=3.15576e7  #seconds
km=1e5        #cm
Mpc=3.0856e24 #cm
kpc=3.0856e21 #cm
Msun=1.989e33 #g
Ymass=0.24   #helium mass fraction
mH=1.6726e-24  #proton mass in grams
gamma=5.0/3.0  #ideal gas
kB=1.3806e-26  #gr (km/s)^2 K^{-1}
nu0=1420.0     #21-cm frequency in MHz

pi=np.pi
#############################################################################


###############################################################################
#This routine computes the values of the Bagla et al. 2009 parameters:
#f1, f2, f3, Mmin and Mmax: (see Eqs. 4,5 and 6 of Bagla et al. 2009)
#z ----------> redshift at which the values of the parameters are wanted
#h ----------> Hubble constant at z=0 in units of 100 km/s/Mpc
#Omega_HI ---> Value of Omega_HI to be reproduced
#f_MF -------> file containing the halo mass function (M, dn/dM)
#The routine returns [Mmin,Mmax,f1,f2,f3]
def Bagla_parameters(z,h,Omega_HI_ref,f_MF):

    #compute the values of Mmin and Mmax: see Bagla et al. 2009
    Mmax=1e10*(200.0/60.0)**3*((1.0+z)/4.0)**(-1.5)*h #Msun/h
    Mmin=1e10*(30.0/60.0)**3*((1.0+z)/4.0)**(-1.5)*h  #Msun/h

    print '\nM_min (z=%2.1f) = %e Msun/h'%(z,Mmin)
    print 'M_max (z=%2.1f) = %e Msun/h\n'%(z,Mmax)

    #read the mass function file
    M,MF=np.loadtxt(f_MF,unpack=True)

    #compute the values of f1, f2 and f3 (see Eqs. 4,5,6 of Bagla et al. 2009)
    #We have to use the fact that:  
    # Omega_HI = \int_0^infty dn/dM(M) M_HI(M) dM / rho_crit

    #compute the value of f1
    method=1; yinit=[0.0]; M_limits=[np.min(M),np.max(M)]
    I=si.odeint(deriv_Bagla_parameters,yinit,M_limits,
                args=(M,MF,Mmin,Mmax,method),
                rtol=1e-8,atol=1e-6,h0=1e6,mxstep=1000000)[1][0]
    f1=Omega_HI_ref/(I/rho_crit)
    print 'f1 = %e ----> Omega_HI = %e'%(f1,I*f1/rho_crit)

    #compute the value of f2
    method=2; yinit=[0.0]; M_limits=[np.min(M),np.max(M)]
    I=si.odeint(deriv_Bagla_parameters,yinit,M_limits,
                args=(M,MF,Mmin,Mmax,method),
                rtol=1e-8,atol=1e-6,h0=1e6,mxstep=1000000)[1][0]
    f2=Omega_HI_ref/(I/rho_crit)
    print 'f2 = %e ----> Omega_HI = %e'%(f2,I*f2/rho_crit)

    #compute the value of f3
    method=3; yinit=[0.0]; M_limits=[np.min(M),np.max(M)]
    I=si.odeint(deriv_Bagla_parameters,yinit,M_limits,
                args=(M,MF,Mmin,Mmax,method),
                rtol=1e-8,atol=1e-6,h0=1e6,mxstep=1000000)[1][0]
    f3=Omega_HI_ref/(I/rho_crit)
    print 'f3 = %e ----> Omega_HI = %e\n'%(f3,I*f3/rho_crit)

    return [Mmin,Mmax,f1,f2,f3]


def deriv_Bagla_parameters(y,x,M,MF,Mmin,Mmax,method):
    if method==1:
        if x>Mmin and x<Mmax:
            MF_I=10**(np.interp(np.log10(x),np.log10(M),np.log10(MF)))
        else:
            MF_I=0.0
        return [MF_I*x]

    elif method==2 or method==3:
        if x>Mmin:
            MF_I=10**(np.interp(np.log10(x),np.log10(M),np.log10(MF)))
        else:
            MF_I=0.0
        if method==2:
            return [MF_I*x/(1.0+(x/Mmax)**2)]
        else:
            return [MF_I*x/(1.0+(x/Mmax))]

    else:
        print 'wrong method!!!'; sys.exit()
###############################################################################

#This routine uses the Bagla et al. 2009 method to assign HI to the CDM (and
#baryon particles)
#snapshot_fname --> name of the N-body snapshot
#groups_fname ----> name of the folder containing the FoF/Subfind halos
#groups_number ---> number of the FoF/Subfind file to read
#Omega_HI_ref ----> value of Omega_HI that wants to be obtained
#method ----------> 1,2 or 3 (see Bagla et al. 2009 for the different methods)
#f_MF ------------> file containing the halo mass function (M, dn/dM)
#long_ids_flag ---> True if particle IDs are 64 bits. False otherwise
#SFR_flag --------> True for simulations with baryons particles. False otherwise
#obj_Bagla -------> object to assign the HI: 'ALL', 'CDM' or 'GAS'
#the routine returns the IDs of the particles to whom HI has been assigned
#and its HI masses. Note that the HI masses array has a length equal to the 
#total number of particles in the simulation
#If the positions of the particles to which HI has been assigned is wanted,
#one should first sort the positions and then use the IDs to select them
def Bagla_HI_assignment(snapshot_fname,groups_fname,groups_number,Omega_HI_ref,
                        method,f_MF,long_ids_flag,SFR_flag,obj_Bagla='GAS'):
                       

    #read snapshot header and obtain BoxSize, redshift and h
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Nall=head.nall
    redshift=head.redshift
    h=head.hubble

    #find the total number of particles in the simulation
    Ntotal=np.sum(Nall,dtype=np.int64)
    print 'Total number of particles in the simulation: %d\n'%Ntotal
    
    #read FoF halos information
    halos=readfof.FoF_catalog(groups_fname,groups_number,
                              long_ids=long_ids_flag,swap=False,SFR=SFR_flag)
    pos_FoF=halos.GroupPos/1e3   #Mpc/h
    M_FoF=halos.GroupMass*1e10   #Msun/h
    ID_FoF=halos.GroupIDs-1      #normalize IDs
    Len=halos.GroupLen           #number of particles in the halo
    Offset=halos.GroupOffset     #offset of the halo in the ID array
    del halos

    #some verbose
    print 'Number of FoF halos:',len(pos_FoF),len(M_FoF)
    print '%f < X [Mpc/h] < %f'%(np.min(pos_FoF[:,0]),np.max(pos_FoF[:,0]))
    print '%f < Y [Mpc/h] < %f'%(np.min(pos_FoF[:,1]),np.max(pos_FoF[:,1]))
    print '%f < Z [Mpc/h] < %f'%(np.min(pos_FoF[:,2]),np.max(pos_FoF[:,2]))
    print '%e < M [Msun/h] < %e\n'%(np.min(M_FoF),np.max(M_FoF))

    #find the values of Mmin,Mmax,f1,f2 and f3: see Bagla et al 2009
    [Mmin,Mmax,f1,f2,f3]=Bagla_parameters(redshift,h,Omega_HI_ref,f_MF)

    #read the mass function file
    M,MF=np.loadtxt(f_MF,unpack=True)

    #keep only with the halos that host HI
    if method==1:
        indexes=np.where((M_FoF>Mmin) & (M_FoF<Mmax))[0]
    else:
        indexes=np.where(M_FoF>Mmin)[0]
    M_FoF=M_FoF[indexes]; Len=Len[indexes]; Offset=Offset[indexes]; del indexes

    #f=open('borrar.dat','w')
    #for i in range(len(M_FoF)):
    #    f.write(str(pos_FoF[indexes][i,0])+' '+str(pos_FoF[indexes][i,1])+' '+\
    #                str(pos_FoF[indexes][i,2])+' '+str(M_FoF[i])+'\n')
    #f.close()

    #compute the total mass in halos hosting HI from the halo catalogue
    print 'Total contributing mass from the catalogue = %e Msun/h'\
        %(np.sum(M_FoF,dtype=np.float64))
    #compute the total mass in halos hosting HI from the halo mass function
    if method==1:
        M_limits=[Mmin,Mmax]; yinit=[0.0] 
        I=si.odeint(deriv_Bagla_parameters,yinit,M_limits,
                    args=(M,MF,Mmin,Mmax,1),
                    rtol=1e-8,atol=1e-6,h0=1e6,mxstep=1000000)[1][0]
    else:
        M_limits=[Mmin,np.max(M)]; yinit=[0.0] 
        I=si.odeint(deriv_Bagla_parameters,yinit,M_limits,
                    args=(M,MF,Mmin,np.max(M),1),
                    rtol=1e-8,atol=1e-6,h0=1e6,mxstep=1000000)[1][0]
    print 'Total contributing mass from the HMF       = %e Msun/h\n'\
        %(I*BoxSize**3)

    if obj_Bagla=='GAS':
        #sort the HI/H and the R array 
        #note that only gas particles have an associated R and HI/H
        ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1
        R_unsort=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3#Mpc/h
        nH0_unsort=readsnap.read_block(snapshot_fname,"NH  ",parttype=0)  #HI/H
        #sanity check: the radius of any gas particle has to be larger than 0!!!
        if np.min(R_unsort)<=0.0:
            print 'something wrong with the HSML radii'; sys.exit()
        R=np.zeros(Ntotal,dtype=np.float32); R[ID_unsort]=R_unsort
        nH0=np.zeros(Ntotal,dtype=np.float32); nH0[ID_unsort]=nH0_unsort
        del R_unsort, nH0_unsort, ID_unsort

        #keep only the IDs of gas particles within halos. The radius of CDM and 
        #star particles will be equal to 0. We use that fact to idenfify the gas
        #particles. We set the IDs of cdm or star particles to Ntotal.
        #Note that the normalized IDs goes from 0 to Ntotal-1. Since ID_FoF is
        #a uint array we can't use negative numbers
        flag_gas=R[ID_FoF]; indexes=np.where(flag_gas==0.0)[0]#; del R
        ID_FoF[indexes]=Ntotal
        #define the IDs array
        if long_ids_flag:
            IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint64)
        else:
            IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint32)
        print 'FoF groups contain %d gas particles'%(len(IDs))
        print 'FoF groups contain %d cdm+gas+star particles'%len(ID_FoF)
        del indexes

    elif obj_Bagla=='CDM':
        ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1
        Mass_unsort=readsnap.read_block(snapshot_fname,"MASS",parttype=-1)
        Mass=np.zeros(Ntotal,dtype=np.float32); Mass[ID_unsort]=Mass_unsort
        del Mass_unsort,ID_unsort

        Mass_CDM=head.massarr[1].astype(np.float32)
        flag_CDM=Mass[ID_FoF]; indexes=np.where(flag_CDM!=Mass_CDM)[0]
        ID_FoF[indexes]=Ntotal

        #define the IDs array
        if long_ids_flag:
            IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint64)
        else:
            IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint32)
        print 'FoF groups contain %d cdm particles'%(len(IDs))
        print 'FoF groups contain %d cdm+gas+star particles'%len(ID_FoF)
        del indexes,Mass_CDM,flag_CDM

    else:
        #define the IDs array
        if long_ids_flag:
            IDs=np.empty(len(ID_FoF),dtype=np.uint64)
        else:
            IDs=np.empty(len(ID_FoF),dtype=np.uint32)


    #loop over the halos containing HI and assign the HI to the gas particles
    M_HI=np.zeros(Ntotal,dtype=np.float32); No_gas_halos=0; IDs_offset=0
    for index in range(len(M_FoF)):

        #select the IDs of all particles belonging to the halo
        indexes=ID_FoF[Offset[index]:Offset[index]+Len[index]]

        if obj_Bagla=='GAS' or obj_Bagla=='CDM':
            #find the IDs of the gas particles belonging to the halo
            indexes=indexes[np.where(indexes!=Ntotal)[0]]

            if obj_Bagla=='GAS':
                #find the sph radii and HI/H fraction of the gas particles
                radii=R[indexes]; nH0_part=nH0[indexes]
            
        #fill the IDs array
        IDs[IDs_offset:IDs_offset+len(indexes)]=indexes
        IDs_offset+=len(indexes)

        #compute the total HI mass within the dark matter halo
        if method==1:
            M_HI_halo=f1*M_FoF[index]
        elif method==2:
            M_HI_halo=f2/(1.0+(M_FoF[index]/Mmax)**2)*M_FoF[index]
        else:
            M_HI_halo=f3/(1.0+(M_FoF[index]/Mmax))*M_FoF[index]

        #if there are gas particles assign the HI to them
        Num_gas=len(indexes)
        if Num_gas>0:
            M_HI[indexes]+=(M_HI_halo*1.0/Num_gas)
            #M_HI[indexes]+=M_HI_halo*(radii**2/nH0_part**2.5)\
            #    /np.sum(radii**2/nH0_part**2.5,dtype=np.float64)
            #M_HI[indexes]+=M_HI_halo*(nH0_part/radii**3)\
            #    /np.sum(nH0_part/radii**3,dtype=np.float64)
        else:
            No_gas_halos+=1
    print '\nNumber of halos with no gas particles=',No_gas_halos

    #just keep the IDs of the particles to which HI has been assigned
    IDs=IDs[0:IDs_offset]

    #compute the value of OmegaHI
    print 'Omega_HI (halos) = %e\n'\
        %(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return [IDs,M_HI]
###############################################################################

#This routine uses the method_1 to assign the HI to the gas particles. In short:
#the method consist in assign a value of HI / H equal to HI_frac (a parameter)
#to the particles with densities larger than rho_th and leave untouched the 
#value of HI / H for gas particles with densities smaller than rho_th. The value
#of rho_th is choosen such as Omega_HI = Omega_HI_ref (a parameter)
#snapshot_fname ---> name of the N-body snapshot
#HI_frac ----------> sets the value of HI / H for particles with rho>rho_th
#Omega_HI_ref -----> value of Omega_HI that wants to be assigned to particles
#the routine returns the IDs of the particles to whom HI has been assigned
#and its HI masses. Note that the HI masses array has a length equal to the 
#total number of particles in the simulation
#If the positions of the particles to which HI has been assigned is wanted,
#one should first sort the positions and then use the IDs to select them
def method_1_HI_assignment(snapshot_fname,HI_frac,Omega_HI_ref):

    #read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Nall=head.nall

    #read RHO (1e10Msun/h/(kpc/h)^3), NH0 and the masses of the gas particles
    rho =readsnap.read_block(snapshot_fname,"RHO ",parttype=0)
    nH0 =readsnap.read_block(snapshot_fname,"NH  ",parttype=0)
    mass=readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h

    mean_rho_b=np.sum(mass,dtype=np.float64)/BoxSize**3
    print 'baryons mean density = %e h^2 Msun/Mpc^3'\
        %(np.mean(rho,dtype=np.float64)*1e19)
    print '<rho_b> = %e h^2 Msun/Mpc^3'%(mean_rho_b)
    print 'Omega_gas = %f\n'%(mean_rho_b/rho_crit)

    #compute the value of rho_th by impossing that Omega_HI = Omega_HI_ref
    iterations=20; final=False; i=0
    rho_th_min=1e-10; rho_th_max=1e-2; tol=1e-3
    rho_th=10**(0.5*(np.log10(rho_th_min)+np.log10(rho_th_max)))
    while not(final):
        indexes_above=np.where(rho>rho_th)[0]
        indexes_below=np.where(rho<=rho_th)[0]
        MHI=np.sum(0.76*nH0[indexes_below]*mass[indexes_below],dtype=np.float64)
        MHI+=np.sum(0.76*HI_frac*mass[indexes_above],dtype=np.float64)

        Omega_HI=MHI/BoxSize**3/rho_crit
        print 'rho_th = %e ---> Omega_HI = %e' %(rho_th,Omega_HI)

        if (np.absolute((Omega_HI-Omega_HI_ref)/Omega_HI_ref)<tol) or\
                i>iterations:
            final=True
        else:
            i+=1
            if (Omega_HI<Omega_HI_ref):
                rho_th_max=rho_th
            else:
                rho_th_min=rho_th
            rho_th=10**(0.5*(np.log10(rho_th_min)+np.log10(rho_th_max)))
    print '\nrho_th = %e h^2 Msun/Mpc^3'%(rho_th*1e10/1e-9)
    print 'delta_th = rho_th / <rho_b> - 1 = %f\n'%(rho_th*1e19/mean_rho_b-1.0)
    del indexes_below,rho
    nH0[indexes_above]=HI_frac; del indexes_above

    #find the IDs and fill the array M_HI
    IDs=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized
    size_M_HI_array=np.sum(Nall,dtype=np.int64)
    M_HI=np.zeros(size_M_HI_array,dtype=np.float32)
    M_HI[IDs]=0.76*nH0*mass; del nH0,mass

    return [IDs,M_HI]
###############################################################################

#This routine uses the Dave et al. 2013 method to assign HI to the gas particles
#snapshot_fname ---> name of the N-body snapshot
#HI_frac ----------> sets the value of HI / H for self-shielded regions
#fac --------------> This is the factor to obtain <F> = <F>_obs from the Lya
#MaxSfrTimescale --> t_0^* in SH 2003 model. Gadget internal units
#PhysDensThresh ---> SF happens for densities higher than this. Physical units
#FactorSN ---------> parameter beta in SH 2003
#the routine returns the IDs of the particles to whom HI has been assigned
#and its HI masses. Note that the HI masses array has a length equal to the 
#total number of particles in the simulation
#If the positions of the particles to which HI has been assigned is wanted,
#one should first sort the positions and then use the IDs to select them
#If it is not clear which are the parameters MaxSfrTimescale, FactorSN and
#PhysDensThresh just set nH0=0.9 for star forming particles (good approximation)
def Dave_HI_assignment(snapshot_fname,HI_frac,fac,MaxSfrTimescale=1.5,
                       PhysDensThresh=0.0012805,FactorSN=0.1):

    #read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Nall=head.nall
    redshift=head.redshift
    h=head.hubble

    #find the total number of particles in the simulation
    Ntotal=np.sum(Nall,dtype=np.int64)
    print 'Total number of particles in the simulation: %d\n'%Ntotal

    #read the mass, SPH radius and HI/H fraction of the gas particles
    nH0=readsnap.read_block(snapshot_fname,"NH  ",parttype=0)*fac   #HI/H
    mass=readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h
    R=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3     #Mpc/h
    print np.min(R),'< R [Mpc/h] <',np.max(R),'\n'

    #find the star forming particles and their cold gas fraction
    #using an approximation
    """SFR=readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR
    indexes=np.where(SFR>0.0)[0]; nH0[indexes]=0.9; del indexes,SFR"""

    #find the star forming particles and their cold gas fraction
    SFR=readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR
    rho=readsnap.read_block(snapshot_fname,"RHO ",parttype=0) #1e10h^2Msun/Mpc^3
    t_star=MaxSfrTimescale*np.sqrt(PhysDensThresh/(rho*(1.0+redshift)**3))
    x=SFR*(kpc/km/yr)*t_star/(1.0-FactorSN)/mass #here mass in Msun/h
    print np.min(x),'< cold gas fraction <',np.max(x); del rho, t_star
    print '# of particles with cold gas fraction > 0 =',len(np.where(x>0.0)[0])
    indexes=np.where(SFR>0.0)[0]; del SFR; x=x[indexes]
    print np.min(x),'< cold gas fraction (SF particles) <',np.max(x)
    nH0[indexes]=x; del x #assign HI/H = x to star forming particles

    #compute the value of Omega_HI without any self-shielding post-processing
    print 'Omega_HI (0.76*m*nHI) = %e\n'\
        %(np.sum(0.76*mass*nH0,dtype=np.float64)/BoxSize**3/rho_crit)

    #N_HI = 0.76*f_HI*m/m_p * \int_r^h w(r,h)dr
    #we compute the quantity 10^(17.2)*h^2*m_p / (0.76*f_HI*m)
    #if that quantity is larger than 3/pi (see the notes) then there is not
    #enough HI in that particle to became self-shielded.
    #If it is smaller than 3/pi we compute the radius R_shield for which:
    # 10^(17.2) = 0.76*f_HI*m/m_p * \int_{R_shield}^h w(r,h)dr
    prefac = mH*(Mpc/h/(1.0+redshift))**2/(Msun/h)
    phi = prefac*10**(17.2)*R**2/(0.76*nH0*mass); del R 
    print '# of gas particles with N_HI > 10^17.2 = ',\
        len(np.where(phi<=3.0/pi)[0])

    #create a table with u=r/h and Iu=h^2*\int_r^h w(r,h)dr 
    u=np.linspace(1.0,0.0,1000)  #values of xp has to be sorted in np.interp
    Iu=np.zeros(1000)
    for i in range(1000):
        Iu[i]=Iwdr(u[i])

    #Now calculate u_shield = R_shield / h 
    u_shield=np.interp(phi,Iu,u)

    #create a table with u=r/h and V=\int_0^r w(r,h)d^3r
    u=np.linspace(0.0,1.0,1000)  #values of xp has to be sorted in np.interp
    V=np.zeros(1000)
    for i in range(1000):
        V[i]=Volw(u[i])
    Volume=np.interp(u_shield,u,V); del u_shield

    #Compute the HI mass in the region from r=0 to r=R_shield
    M1=0.76*Volume*mass*HI_frac

    #Compute the HI in the region from r=R_shield to r=h
    M2=0.76*(1.0-Volume)*mass*nH0; del Volume,mass,nH0

    #define the array HI_mass
    HI_mass=(M1+M2).astype(np.float32); del M1,M2

    #compute the pression of the gas particles
    rho=readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10#h^2Msun/kpc^3
    U=readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
    P=(gamma-1.0)*U*rho*(1.0+redshift)**3 #h^2 Msun/kpc^3*(km/s)^2
    P=h**2*Msun/kpc**3*P                  #gr/cm^3*(km/s)^2
    P/=kB                                 #K/cm^3
    nH=0.76*h**2*Msun/kpc**3/mH*rho*(1.0+redshift)**3
    del rho,U

    #assign H2 only to star forming particles
    SFR=readsnap.read_block(snapshot_fname,"SFR ",parttype=0)
    indexes=np.where(SFR>0.0)[0]; del SFR
    print '%e < P [K/cm^3] < %e'%(np.min(P[indexes]),np.max(P[indexes]))

    #compute the H2/HI fraction
    R_surf=(P/1.7e4)**0.8
    #R_surf=(P/3.5e4)**0.92
    #f_H2=1.0/(1.0+(35.0*(0.1/nH)**(gamma))**0.92)

    #compute the corrected HI mass taking into account the H2
    HI_mass[indexes]/=(1.0+R_surf[indexes]); del R_surf,indexes,nH
    #HI_mass[indexes]=HI_mass[indexes]*(1.0-f_H2[indexes]); del f_H2, indexes

    #define the M_HI array and fill it
    M_HI=np.zeros(Ntotal,dtype=np.float32)
    IDs=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized
    M_HI[IDs]=HI_mass; del HI_mass

    print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    """# This is for compute the HI mass in the halos
    #we create the array Star_mass that only contain the masses of the stars
    Star_mass=np.zeros(Ntotal,dtype=np.float32)
    Star_IDs=readsnap.read_block(snapshot_fname,"ID  ",parttype=4)-1 #normalized
    Star_mass[Star_IDs]=\
        readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10 #Msun/h
    del Star_IDs

    #read FoF halos information
    groups_fname='/home/villa/bias_HI/Efective_model_120Mpc/FoF_0.2'
    groups_number=8
    long_ids_flag=False; SFR_flag=True #flags for reading the FoF file
    halos=readfof.FoF_catalog(groups_fname,groups_number,
                              long_ids=long_ids_flag,swap=False,SFR=SFR_flag)
    pos_FoF=halos.GroupPos/1e3   #Mpc/h
    M_FoF=halos.GroupMass*1e10   #Msun/h
    ID_FoF=halos.GroupIDs-1      #normalize IDs
    Len=halos.GroupLen           #number of particles in the halo
    Offset=halos.GroupOffset     #offset of the halo in the ID array
    del halos

    #some verbose
    print 'Number of FoF halos:',len(pos_FoF),len(M_FoF)
    print '%f < X [Mpc/h] < %f'%(np.min(pos_FoF[:,0]),np.max(pos_FoF[:,0]))
    print '%f < Y [Mpc/h] < %f'%(np.min(pos_FoF[:,1]),np.max(pos_FoF[:,1]))
    print '%f < Z [Mpc/h] < %f'%(np.min(pos_FoF[:,2]),np.max(pos_FoF[:,2]))
    print '%e < M [Msun/h] < %e\n'%(np.min(M_FoF),np.max(M_FoF))

    #make a loop over the different FoF halos and populate with HI
    No_gas_halos=0; f=open('HI_mass_120Mpc_z=3.dat','w')
    for index in range(len(M_FoF)):

        indexes=ID_FoF[Offset[index]:Offset[index]+Len[index]]

        Num_gas=len(indexes)
        #Num_cdm_star=Len[index]-Num_gas
        #print Num_gas,Num_cdm_star
        if Num_gas>0:
            HI_mass=np.sum(M_HI[indexes],dtype=np.float64)
            Stellar_mass=np.sum(Star_mass[indexes],dtype=np.float64)
            f.write(str(M_FoF[index])+' '+str(HI_mass/M_FoF[index])+' '+\
                        str(Stellar_mass/M_FoF[index])+'\n')
        else:
            No_gas_halos+=1
    f.close(); print '\nNumber of halos with no gas particles=',No_gas_halos
    """

    return [IDs,M_HI]


#This functions returns h^2*\int_r^h w(r,h)dr: u=r/h
def Iwdr(u):
    if u<=0.5:
        return 3.0/pi-8.0/pi*(u-2.0*u**3+1.5*u**4)
    elif u<=1.0:
        return 4.0/pi*(1.0-u)**4
    else:
        print 'error: u=r/h can not be larger than 1'


#This function returns \int_0^r w(r,h) d^3r: u=r/h
def Volw(u):
    if u<0.5:
        return 32.0*(1.0/3.0*u**3-6.0/5.0*u**5+u**6)
    elif u<=1.0:
        return 16.0/15.0*u**3*(36.0*u**2-10.0*u**3-45.0*u+20.0)-1.0/15.0
    else:
        print 'error: u=r/h can not be larger than 1'
###############################################################################

#This routine implements the Barnes & Haehnelt 2014 method to assign the HI
#to dark matter halos
#snapshot_fname ---> name of the N-body snapshot
#groups_fname -----> name of the folder containing the FoF/Subfind halos
#groups_number ----> number of the FoF/Subfind file to read
#long_ids_flag ---> True if particle IDs are 64 bits. False otherwise
#SFR_flag --------> True for simulations with baryons particles. False otherwise
def Barnes_Haehnelt(snapshot_fname,groups_fname,groups_number,long_ids_flag,
                    SFR_flag):
    
    #read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Masses=head.massarr*1e10 #Msun/h
    Omega_m=head.omega_m
    Omega_l=head.omega_l
    Nall=head.nall
    redshift=head.redshift
    h=head.hubble

    #find the total number of particles in the simulation
    Ntotal=np.sum(Nall,dtype=np.int64)
    print 'Total number of particles in the simulation: %d\n'%Ntotal

    #compute the value of Omega_b
    Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
    Omega_b=Omega_m-Omega_cdm
    print 'Omega_cdm =',Omega_cdm; print 'Omega_b   =',Omega_b
    print 'Omega_m   =',Omega_m,'\n'

    #find Barnes & Haehnelt parameters
    c0=25.0
    f_HI=0.37
    f_H=0.76*Omega_b/Omega_m
    alpha_e=3.0
    V0=90.0 #km/s

    #compute the value of delta_c followign Bryan & Norman 1998
    Omega_m_z=Omega_m*(1.0+redshift)**3/(Omega_m*(1.0+redshift)**3+Omega_l)
    x=Omega_m_z-1.0
    delta_c=18.0*pi**2+82.0*x-39.0*x**2
    print 'x = %f  delta_c = %f\n'%(x,delta_c); del x

    #read FoF halos information
    halos=readfof.FoF_catalog(groups_fname,groups_number,
                              long_ids=long_ids_flag,swap=False,SFR=SFR_flag)
    pos_FoF=halos.GroupPos/1e3   #Mpc/h
    M_FoF=halos.GroupMass*1e10   #Msun/h
    ID_FoF=halos.GroupIDs-1      #normalize IDs
    Len=halos.GroupLen           #number of particles in the halo
    Offset=halos.GroupOffset     #offset of the halo in the ID array
    del halos

    #some verbose
    print 'Number of FoF halos:',len(pos_FoF),len(M_FoF)
    print '%f < X [Mpc/h] < %f'%(np.min(pos_FoF[:,0]),np.max(pos_FoF[:,0]))
    print '%f < Y [Mpc/h] < %f'%(np.min(pos_FoF[:,1]),np.max(pos_FoF[:,1]))
    print '%f < Z [Mpc/h] < %f'%(np.min(pos_FoF[:,2]),np.max(pos_FoF[:,2]))
    print '%e < M [Msun/h] < %e\n'%(np.min(M_FoF),np.max(M_FoF))

    #compute the circular velocities of the FoF halos. Here we assume they are
    #spherical and use the spherical collapse model. See Barnes & Haehnelt 2014
    V=96.6*(delta_c*Omega_m/24.4)**(1.0/6.0)*np.sqrt((1.0+redshift)/3.3)*\
        (M_FoF/1e11)**(1.0/3.0) #km/s

    #compute the HI mass associated to each halo and the value of Omega_HI
    M_HI_halo=f_HI*f_H*np.exp(-(V0/V)**alpha_e)*M_FoF

    #write a file with M_HI_halo/M_halo vs M_halo
    #f=open('borrar.dat','w')
    #for i in range(len(M_HI_halo)):
    #    f.write(str(M_FoF[i])+' '+str(M_HI_halo[i]/M_FoF[i])+'\n')
    #f.close()

    print 'Omega_HI = %e\n'%(np.sum(M_HI_halo,dtype=np.float64)/BoxSize**3/
                             rho_crit)

    #sort the R array (note that only gas particles have an associated R)
    ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1
    R_unsort=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3 #Mpc/h
    nH0_unsort=readsnap.read_block(snapshot_fname,"NH  ",parttype=0)   #HI/H
    #sanity check: the radius of any gas particle has to be larger than 0!!!
    if np.min(R_unsort)<=0.0:
        print 'something wrong with the HSML radii'
        sys.exit()
    R=np.zeros(Ntotal,dtype=np.float32); R[ID_unsort]=R_unsort
    nH0=np.zeros(Ntotal,dtype=np.float32); nH0[ID_unsort]=nH0_unsort
    del R_unsort, nH0_unsort, ID_unsort

    #keep only the IDs of gas particles within halos. The radius of CDM and 
    #star particles will be equal to 0. We use that fact to idenfify the gas
    #particles. We set the IDs of cdm or star particles to Ntotal. Note that 
    #the normalized IDs goes from 0 to Ntotal-1. Since ID_FoF is a uint array
    #we can't use negative numbers
    flag_gas=R[ID_FoF]; indexes=np.where(flag_gas==0.0)[0]
    ID_FoF[indexes]=Ntotal
    #define the IDs array
    if long_ids_flag:
        IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint64)
    else:
        IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint32)
    print 'FoF groups contain %d gas particles'%len(IDs)
    print 'FoF groups contain %d cdm+gas+star particles'%len(ID_FoF)
    del indexes

    #make a loop over the different FoF halos and populate with HI
    No_gas_halos=0; M_HI=np.zeros(Ntotal,dtype=np.float32); IDs_offset=0
    for index in range(len(M_FoF)):

        indexes=ID_FoF[Offset[index]:Offset[index]+Len[index]]

        #find how many gas particles there are in the FoF group
        indexes=indexes[np.where(indexes!=Ntotal)[0]]

        #find the sph radii and HI/H fraction of the gas particles
        radii=R[indexes]; nH0_part=nH0[indexes]

        #fill the IDs array
        IDs[IDs_offset:IDs_offset+len(indexes)]=indexes
        IDs_offset+=len(indexes)

        Num_gas=len(indexes)
        #Num_cdm_star=Len[index]-Num_gas
        #print Num_gas,Num_cdm_star
        if Num_gas>0:
            #M_HI[indexes]+=M_HI_halo[index]/Num_gas
            #M_HI[indexes]+=M_HI_halo[index]*(radii**1)\
            #    /np.sum(radii**1,dtype=np.float64)
            #M_HI[indexes]+=M_HI_halo[index]*(radii**2/nH0_part**1)\
            #    /np.sum(radii**2/nH0_part**1,dtype=np.float64)
            M_HI[indexes]+=M_HI_halo[index]*(radii**2/nH0_part**2.5)\
                /np.sum(radii**2/nH0_part**2.5,dtype=np.float64)
            #M_HI[indexes]+=M_HI_halo[index]*(radii**4)\
            #    /np.sum(radii**4,dtype=np.float64)
        else:
            No_gas_halos+=1
    print '\nNumber of halos with no gas particles=',No_gas_halos

    #compute again the value of Omega_HI
    print 'Omega_HI = %e\n'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return [IDs,M_HI]

###############################################################################

#This routine implements the Paco method to assign the HI to dark matter halos
#snapshot_fname ---> name of the N-body snapshot
#groups_fname -----> name of the folder containing the FoF/Subfind halos
#groups_number ----> number of the FoF/Subfind file to read
#long_ids_flag ----> True if particle IDs are 64 bits. False otherwise
#SFR_flag ---------> True for hydro simulations. False otherwise
#Omega_cdm --------> Value of Omega_CDM. For hydro simulations set to None
#Omega_b ----------> Value of Omega_B.   For hydro simulations set to None
#Paco_object ------> Type of particle to assign HI: 'GAS' or 'pureCDM'
#                    if 'GAS' it will only assign HI to the gas particles
#                    and each particle within the same halo will have a 
#                    HI mass depending on its properties. If pureCDM it will
#                    distribute the HI equally within all the particles
#                    belonging to the dark matter halo
def Paco_HI_assignment(snapshot_fname,groups_fname,groups_number,long_ids_flag,
                       SFR_flag,Omega_cdm=None,Omega_b=None,Paco_object='GAS'):
    
    #read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Masses=head.massarr*1e10 #Msun/h
    Omega_m=head.omega_m
    Omega_l=head.omega_l
    Nall=head.nall
    redshift=head.redshift
    h=head.hubble

    #find the total number of particles in the simulation
    Ntotal=np.sum(Nall,dtype=np.int64)
    print 'Total number of particles in the simulation: %d\n'%Ntotal

    #compute the value of Omega_b
    if Omega_cdm==None:
        Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
    if  Omega_b==None:
        Omega_b=Omega_m-Omega_cdm
    print 'Omega_cdm =',Omega_cdm; print 'Omega_b   =',Omega_b
    print 'Omega_m   =',Omega_m,'\n'

    #Paco parameters of the function M_HI(M)
    #c0=2.7       #3.4 (z=2.4)   : 3.2 (z=3.0)  : 2.7 (z=4.0)
    #f_HI=0.34    #0.19 (z=2.4) : 0.222 (z=3.0)  : 0.34 (z=4.0)
    z_t    = [2.4,  3.0,   4.0]
    c0_t   = [3.4,  3.2,   2.7]
    f_HI_t = [0.19, 0.222, 0.34]

    c0=np.interp(redshift,z_t,c0_t); f_HI=np.interp(redshift,z_t,f_HI_t)
    f_H=0.76*Omega_b/Omega_m
    alpha_e=3.0
    V0=37.0 #km/s
    print 'c0(z=%2.2f) = %2.2f : f_HI(z=%2.2f) = %2.2f'\
        %(redshift,c0,redshift,f_HI)

    #compute the value of delta_c followign Bryan & Norman 1998
    Omega_m_z=Omega_m*(1.0+redshift)**3/(Omega_m*(1.0+redshift)**3+Omega_l)
    x=Omega_m_z-1.0
    delta_c=18.0*pi**2+82.0*x-39.0*x**2
    print 'x = %f  delta_c = %f\n'%(x,delta_c); del x

    #read FoF halos information
    halos=readfof.FoF_catalog(groups_fname,groups_number,
                              long_ids=long_ids_flag,swap=False,SFR=SFR_flag)
    pos_FoF=halos.GroupPos/1e3   #Mpc/h
    M_FoF=halos.GroupMass*1e10   #Msun/h
    ID_FoF=halos.GroupIDs-1      #normalize IDs
    Len=halos.GroupLen           #number of particles in the halo
    Offset=halos.GroupOffset     #offset of the halo in the ID array
    del halos

    #some verbose
    print 'Number of FoF halos:',len(pos_FoF),len(M_FoF)
    print '%f < X [Mpc/h] < %f'%(np.min(pos_FoF[:,0]),np.max(pos_FoF[:,0]))
    print '%f < Y [Mpc/h] < %f'%(np.min(pos_FoF[:,1]),np.max(pos_FoF[:,1]))
    print '%f < Z [Mpc/h] < %f'%(np.min(pos_FoF[:,2]),np.max(pos_FoF[:,2]))
    print '%e < M [Msun/h] < %e\n'%(np.min(M_FoF),np.max(M_FoF))

    #compute the circular velocities of the FoF halos. Here we assume they are
    #spherical and use the spherical collapse model. See Barnes & Haehnelt 2014
    V=96.6*(delta_c*Omega_m/24.4)**(1.0/6.0)*np.sqrt((1.0+redshift)/3.3)*\
        (M_FoF/1e11)**(1.0/3.0) #km/s

    #compute the HI mass associated to each halo and the value of Omega_HI
    M_HI_halo=f_HI*f_H*np.exp(-(V0/V)**alpha_e)*M_FoF

    #write a file with M_HI_halo/M_halo vs M_halo
    #f=open('borrar.dat','w')
    #for i in range(len(M_HI_halo)):
    #    f.write(str(M_FoF[i])+' '+str(M_HI_halo[i]/M_FoF[i])+'\n')
    #f.close()

    print 'Omega_HI = %e\n'%(np.sum(M_HI_halo,dtype=np.float64)/BoxSize**3/
                             rho_crit)

    if Paco_object=='GAS':

        #sort the R array (note that only gas particles have an associated R)
        ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1
        R_unsort=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3#Mpc/h
        nH0_unsort=readsnap.read_block(snapshot_fname,"NH  ",parttype=0)   #HI/H
        #sanity check: the radius of any gas particle has to be larger than 0!!!
        if np.min(R_unsort)<=0.0:
            print 'something wrong with the HSML radii'; sys.exit()
        R=np.zeros(Ntotal,dtype=np.float32); R[ID_unsort]=R_unsort
        nH0=np.zeros(Ntotal,dtype=np.float32); nH0[ID_unsort]=nH0_unsort
        del R_unsort, nH0_unsort, ID_unsort

        #keep only the IDs of gas particles within halos. The radius of CDM and 
        #star particles will be equal to 0. We use that fact to idenfify the gas
        #particles. We set the IDs of cdm or star particles to Ntotal. Note 
        #that  the normalized IDs goes from 0 to Ntotal-1. Since ID_FoF is a 
        #uint array we can't use negative numbers
        flag_gas=R[ID_FoF]; indexes=np.where(flag_gas==0.0)[0]
        ID_FoF[indexes]=Ntotal
        #define the IDs array
        if long_ids_flag:
            IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint64)
        else:
            IDs=np.empty(len(ID_FoF)-len(indexes),dtype=np.uint32)
        print 'FoF groups contain %d gas particles'%len(IDs)
        print 'FoF groups contain %d cdm+gas+star particles'%len(ID_FoF)
        del indexes

    else:
        
        #define the IDs array
        if long_ids_flag:
            IDs=np.empty(len(ID_FoF),dtype=np.uint64)
        else:
            IDs=np.empty(len(ID_FoF),dtype=np.uint32)



    #make a loop over the different FoF halos and populate with HI
    No_gas_halos=0; M_HI=np.zeros(Ntotal,dtype=np.float32); IDs_offset=0
    for index in range(len(M_FoF)):

        indexes=ID_FoF[Offset[index]:Offset[index]+Len[index]]

        if Paco_object=='GAS':
            #find how many gas particles there are in the FoF group
            indexes=indexes[np.where(indexes!=Ntotal)[0]]

            #find the sph radii and HI/H fraction of the gas particles
            radii=R[indexes]; nH0_part=nH0[indexes]

        #fill the IDs array
        IDs[IDs_offset:IDs_offset+len(indexes)]=indexes
        IDs_offset+=len(indexes)

        Num_gas=len(indexes)
        if Num_gas>0:
            if Paco_object=='GAS':
                M_HI[indexes]+=M_HI_halo[index]*(nH0_part**0.17*radii**0.35)\
                    /np.sum(nH0_part**0.17*radii**0.35,dtype=np.float64)
            else:
                M_HI[indexes]+=M_HI_halo[index]/Num_gas
        else:
            No_gas_halos+=1
    print '\nNumber of halos with no gas particles=',No_gas_halos

    #compute again the value of Omega_HI
    print 'Omega_HI = %e\n'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return [IDs,M_HI]

###############################################################################

#This routine implements the Nagamine HI assignment
#snapshot_fname -------> name of the N-body snapshot
#FactorSN -------------> Fraction of stars converted into supernove
#MaxSfrTimescale ------> t_0^star from the Springel & Hernquist 2003 model
#PhysDensThresh -------> density threshold above which star formation begin
#correct_H2 -----------> correct the HI abundance taking into account H2
def Nagamine_HI_assignment(snapshot_fname,FactorSN,MaxSfrTimescale,
                           PhysDensThresh,correct_H2=True):

    #read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head=readsnap.snapshot_header(snapshot_fname)
    Nall=head.nall
    BoxSize=head.boxsize/1e3 #Mpc/h
    redshift=head.redshift
    h=head.hubble

    #find the total number of particles in the simulation
    Ntotal=np.sum(Nall,dtype=np.int64)
    print 'Total number of particles in the simulation: %d\n'%Ntotal

    #read the HI array from the snapshot
    #HI_mass=readsnap.read_block(snapshot_fname,"MHI ",parttype=0)*1e10 #Msun/h
    
    #read the blocks RHO, SFR and MASS
    #density units: 1e10 (Msun/h)/(kpc/h)^3
    rho=readsnap.read_block(snapshot_fname,"RHO ",parttype=0) 
    SFR=readsnap.read_block(snapshot_fname,"SFR ",parttype=0) 
    mass=readsnap.read_block(snapshot_fname,"MASS",parttype=0) #1e10 Msun/h

    #compute the cold fraction using the SFR block
    t_star=MaxSfrTimescale*np.sqrt(PhysDensThresh/(rho*(1.0+redshift)**3))
    x=SFR*(kpc/km/yr)*t_star/(1.0-FactorSN)/(mass*1e10); del rho, SFR

    #read the HI/H array
    nH0=readsnap.read_block(snapshot_fname,"NH  ",parttype=0) #HI/H

    #define the HI_mass array and fill it
    HI_mass=np.empty(len(x),dtype=np.float32)

    #fill the HI_mass using the Lya particles
    HI_mass=0.76*mass*1e10*nH0; del nH0

    #fill the HI_mass using the star forming particles
    indexes=np.where(x>0.0)[0]
    HI_mass[indexes]=0.76*mass[indexes]*1e10*x[indexes]; del indexes, mass, x
    
    #correct for the presence of H2
    if correct_H2:
        #compute the pression of the gas particles
        #h^2Msun/kpc^3
        rho=readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10
        U=readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
        P=(gamma-1.0)*U*rho*(1.0+redshift)**3 #h^2 Msun/kpc^3*(km/s)^2
        P=h**2*Msun/kpc**3*P                  #gr/cm^3*(km/s)^2
        P/=kB                                 #K/cm^3
        del rho,U
 
        #assign H2 only to star forming particles
        SFR=readsnap.read_block(snapshot_fname,"SFR ",parttype=0)
        indexes=np.where(SFR>0.0)[0]; del SFR

        #compute the H2/HI fraction
        R_surf=(P/1.7e4)**0.8

        #compute the corrected HI mass taking into account the H2
        HI_mass[indexes]=HI_mass[indexes]/(1.0+R_surf[indexes]); del indexes

    #read the gas particles IDs
    IDs=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized

    #define the M_HI array 
    M_HI=np.zeros(Ntotal,dtype=np.float32)
    M_HI[IDs]=HI_mass; del HI_mass

    print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return [IDs,M_HI]

###############################################################################

#This routine implements the Nagamine HI assignment
#snapshot_fname -------> name of the N-body snapshot
#fac ------------------> factor to reproduce the mean Lya flux
#TREECOOL_file --------> TREECOOL file used in the N-body
#Gamma_UVB ------------> value of the UVB photoionization rate
#correct_H2 -----------> correct the HI/H fraction to account for H2
#if Gamma_UVB is set to None the value of the photoionization rate will be read
#from the TREECOOL file, otherwise it is used the Gamma_UVB value
def Rahmati_HI_assignment(snapshot_fname,fac,TREECOOL_file,Gamma_UVB=None,
                          correct_H2=False,IDs=None):

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

    #Rahmati et. al. 2013 self-shielding parameters (table A1)
    z_t       = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00])
    n0_t      = np.array([-2.94,-2.29,-2.06,-2.13,-2.23,-2.35]); n0_t=10**n0_t
    alpha_1_t = np.array([-3.98,-2.94,-2.22,-1.99,-2.05,-2.63])
    alpha_2_t = np.array([-1.09,-0.90,-1.09,-0.88,-0.75,-0.57])
    beta_t    = np.array([1.29, 1.21, 1.75, 1.72, 1.93, 1.77])
    f_t       = np.array([0.01, 0.03, 0.03, 0.04, 0.02, 0.01])

    #compute the self-shielding parameters at the redshift of the N-body
    n0      = np.interp(redshift,z_t,n0_t)
    alpha_1 = np.interp(redshift,z_t,alpha_1_t)
    alpha_2 = np.interp(redshift,z_t,alpha_2_t)
    beta    = np.interp(redshift,z_t,beta_t)
    f       = np.interp(redshift,z_t,f_t)
    print 'n0 = %e\nalpha_1 = %2.3f\nalpha_2 = %2.3f\nbeta = %2.3f\nf = %2.3f'\
        %(n0,alpha_1,alpha_2,beta,f)

    #find the total number of particles in the simulation
    Ntotal=np.sum(Nall,dtype=np.uint64)
    print 'Total number of particles in the simulation:',Ntotal

    #find the value of the photoionization rate
    if Gamma_UVB==None:
        data=np.loadtxt(TREECOOL_file); logz=data[:,0]; Gamma_UVB=data[:,1]
        Gamma_UVB=np.interp(np.log10(1.0+redshift),logz,Gamma_UVB); del data
        print 'Gamma_UVB(z=%2.2f) = %e s^{-1}'%(redshift,Gamma_UVB)
    
    #Correct to reproduce the Lya forest mean flux
    Gamma_UVB/=fac

    #compute the HI/H fraction 
    nH0=HI_from_UVB(snapshot_fname,Gamma_UVB,True,
                    f,n0,alpha_1,alpha_2,beta,SF_temperature=1e4)
    
    #read the gas particles IDs and masses
    if IDs==None:
        IDs=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized
    mass=readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h

    #create the array M_HI and fill it
    M_HI=np.zeros(Ntotal,dtype=np.float32)
    M_HI[IDs]=0.76*mass*nH0; del nH0,mass

    #correct for the presence of H2
    if correct_H2:
        #compute the pression of the gas particles
        #h^2Msun/kpc^3
        rho=readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10
        U=readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
        P=(gamma-1.0)*U*rho*(1.0+redshift)**3 #h^2 Msun/kpc^3*(km/s)^2
        P=h**2*Msun/kpc**3*P                  #gr/cm^3*(km/s)^2
        P/=kB                                 #K/cm^3
        del rho,U
 
        #assign H2 only to star forming particles
        SFR=np.zeros(Ntotal,dtype=np.float32)
        SFR[IDs]=readsnap.read_block(snapshot_fname,"SFR ",parttype=0)
        indexes=np.where(SFR>0.0)[0]; del SFR

        #compute the H2/HI fraction
        R_surf=np.zeros(Ntotal,dtype=np.float32)
        R_surf[IDs]=(P/1.7e4)**0.8
        #R_surf[IDs]=(P/3.5e4)**0.92
        #R_surf[IDs]=1.0/(1.0+(35.0*(0.1/nH/0.76)**(gamma))**0.92)

        #compute the corrected HI mass taking into account the H2
        M_HI[indexes]=M_HI[indexes]/(1.0+R_surf[indexes]); del indexes,R_surf
        #M_HI[indexes]=M_HI[indexes]*(1.0-R_surf[indexes]); del indexes,R_surf

    print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return [IDs,M_HI]

###############################################################################

#this routine computes the column density along line of sights (los) that cross
#the entire simulation box from z=0 to z=BoxSize. The los are place in a grid
#within the XY plane, and the number of those is given by cells x cells
#R ------------> array containing the SPH smoothing lenghts of the particles 
#X ------------> array containing the X-positions of the particles
#Y ------------> array containing the Y-positions of the particles
#M_HI ---------> array containing the HI masses of the particles
#cells --------> size of the los grid along one direction
#threads ------> number of threads to use in the computation (openmp threads)
def NHI_los_sph(R,X,Y,M_HI,BoxSize,cells,threads):

    #create a table with c I(c), where c=b/h and I=h^2*\int_0^l_max W(r,h) dl
    intervals=1001 #the values of c goes as 0.000, 0.001, 0.002, 0.003, .....
    c=np.linspace(0.0,1.0,intervals); Ic=np.empty(intervals,dtype=np.float32)
    for i in range(intervals):
        Ic[i]=Iwdl(c[i])

    #create the los column density array. Each element contains the column
    #density along a given line of sight
    column_density=np.zeros(cells*cells,dtype=np.float64)

    #scipy.weave gives problems with scalars: define a 1-element array 
    Box=np.array([BoxSize],dtype=np.float32) 
    cell=np.array([cells],dtype=np.int32)

    support = """
       #include <math.h>"
       #include <omp.h>
    """
    code = """
       omp_set_num_threads(threads);
       float R_gas, X_gas, Y_gas, HI_gas, c_gas, NHI;
       float x_min, x_max, y_min, y_max;
       int index_x_min, index_x_max, index_y_min, index_y_max;
       float BoxSize, dist, diff_x, diff_y;
       int cells, index_x, index_y, index_los, p;

       BoxSize=Box(0); cells=cell(0);

       #pragma omp parallel for private(R_gas,X_gas,Y_gas,HI_gas,x_min,x_max,index_x_min,index_x_max,y_min,y_max,index_y_min,index_y_max,index_x,index_y,index_los,diff_x,diff_y,dist,c_gas,p,NHI) shared(column_density) 
       for (int l=0;l<length;l++){
           R_gas=R(l); X_gas=X(l); Y_gas=Y(l); HI_gas=M_HI(l);

           x_min=X_gas-R_gas; x_max=X_gas+R_gas;
           index_x_min=(int)(cells*x_min/BoxSize);
           index_x_min = (index_x_min<0) ? 0 : index_x_min;
           index_x_max=(int)(cells*x_max/BoxSize)+1;
           index_x_max = (index_x_max>=cells) ? cells-1 : index_x_max;

           y_min=Y_gas-R_gas; y_max=Y_gas+R_gas;
           index_y_min=(int)(cells*y_min/BoxSize);
           index_y_min = (index_y_min<0) ? 0 : index_y_min;
           index_y_max=(int)(cells*y_max/BoxSize)+1;
           index_y_max = (index_y_max>=cells) ? cells-1 : index_y_max;

           for (index_x=index_x_min; index_x<=index_x_max; index_x++){
              for (index_y=index_y_min; index_y<=index_y_max; index_y++){

                  index_los=cells*index_y+index_x;

                  diff_x=BoxSize*index_x/cells-X_gas; 
                  diff_y=BoxSize*index_y/cells-Y_gas;
                  dist=sqrt(diff_x*diff_x+diff_y*diff_y);
                  c_gas=dist/R_gas;

                  if (c_gas<1.0){
                      p=(int)(c_gas*1000.0);
                      NHI=Ic(p)+(Ic(p+1)-Ic(p))/(c(p+1)-c(p))*(c_gas-c(p));
                      NHI*=2.0*HI_gas/(R_gas*R_gas);
                      #pragma omp atomic
                          column_density(index_los)+=NHI;
                  }                    
              }
           }

           if (l%1000000==0)
               printf("%d\\n",l);
       } 
    """

    length=len(R)
    wv.inline(code,['length','X','Y','R','M_HI','c','Ic','Box','cell',
                    'column_density','threads'],
              type_converters = wv.converters.blitz,
              extra_link_args=['-lgomp'],
              support_code = support,libraries = ['m','gomp'],
              extra_compile_args =['-O3 -fopenmp'])
    
    return column_density


#This functions returns \int (1-6u^2+6u^3)udu/sqrt(u^2-c^2):  c=b/h
def I1(u,c):
    if u<c:
        print 'u must be greater than c'; sys.exit()
    u2=u**2; c2=c**2; d=np.sqrt(u2-c2)
    if c>0.0:
        return 0.25*(d*(c2*(9.0*u-16.0)+6.0*u**3-8.0*u2+4.0)+\
                         9.0*c**4*np.log(2.0*(d+u)))
    else:
        return 0.25*(d*(6.0*u**3-8.0*u2+4.0))
                        

#This functions returns \int 2(1-u)^3udu/sqrt(u^2-c^2):  c=b/h
def I2(u,c):
    if u<c:
        print 'u must be greater than c'; sys.exit()
    u2=u**2; c2=c**2; d=np.sqrt(u2-c2); g=3.0*(c2+4.0)/8.0
    if c>0.0:
        return -2.0*(g*c2*np.log(2.0*(d+u))+d*(g*u-2.0*c2+u**3/4.0-u2-1.0))
    else:
        return -2.0*(d*(g*u+u**3/4.0-u2-1.0))

#Function used to compute the integral \int_0^l_max w(r,h) dl:   u=r/h:  c=b/h
#The integral is performed in r, with b^2+l^2=r^2 and therefore the limits in r
#are r=b(bottom) and r=h(top)
def Iwdl(c):
    if c>0.5:
        return 8.0*(I2(1.0,c)-I2(c,c))/pi
    else:
        return 8.0*(I2(1.0,c)-I2(0.5,c)+I1(0.5,c)-I1(c,c))/pi
###############################################################################

#This routine computes the cross section of a given halo using the Eq. 2 of
#Pontzen et al. 2008
#X_halo -------> x-coordinate of the center of the dark matter halo
#Y_halo -------> y-coordinate of the center of the dark matter halo
#R_halo -------> virial radius of the halo
#num_los ------> number of random los over which N_HI will be computed
#redshift -----> redshift at which the calculation is performed
#h ------------> Hubble constant at z=0 in units of 100 km/s/Mpc
#X_gas --------> array containing the x-coordinates of the gas particles
#Y_gas --------> array containing the y-coordinates of the gas particles
#R_gas --------> array containing the gas particles sph smoothing lengths
#M_HI_gas -----> array containing the M_HI of the gas particles
#threads ------> number of threads to use in the computation (openmp threads)
def cross_section_halo(X_halo,Y_halo,R_halo,num_los,redshift,h,
                       X_gas,Y_gas,R_gas,M_HI_gas,threads):

    #create a table with c I(c)
    intervals=1001 #the values of c goes as 0.000, 0.001, 0.002, 0.003, .....
    c=np.linspace(0.0,1.0,intervals)
    Ic=np.empty(intervals,dtype=np.float32)
    for i in range(intervals):
        Ic[i]=Iwdl(c[i])
    Ic[0]=Ic[1]

    #find the (x,y) coordinates of the random los
    R_los=np.sqrt(np.random.random(num_los))*R_halo
    theta_los=2.0*np.pi*np.random.random(num_los)
    X_los=X_halo+R_los*np.cos(theta_los) #x-coordinate of the random los
    Y_los=Y_halo+R_los*np.sin(theta_los) #y-coordinate of the random los
    del R_los,theta_los
    
    #create the array that will contain the column densities values
    N_HI=np.zeros(num_los,dtype=np.float32)

    support = """
       #include <math.h>"
       #include <omp.h>
    """
    code = """
       omp_set_num_threads(threads);
       float X,Y,dist_x,dist_y,dist;
       float c_gas, IcI;
       int p,i,l;

       #pragma omp parallel for private(l,X,Y,dist_x,dist_y,dist,c_gas,p,IcI) shared(N_HI) 
       for (i=0;i<num_los;i++){
          X=X_los(i); Y=Y_los(i); 
          for (l=0;l<length;l++){
               
              dist_x=(X_gas(l)-X); 
              dist_y=(Y_gas(l)-Y);
              dist=sqrt(dist_x*dist_x+dist_y*dist_y);

              c_gas=dist/R_gas(l);

              if (c_gas<1.0){
                   p=(int)(c_gas*1000.0);
                   IcI=Ic(p)+(Ic(p+1)-Ic(p))/(c(p+1)-c(p))*(c_gas-c(p));
                   #pragma omp atomic
                       N_HI(i)+=IcI*2.0*M_HI_gas(l)/(R_gas(l)*R_gas(l));
              }           
          }
       } 
    """

    length=len(X_gas)
    wv.inline(code,['X_los','Y_los','X_gas','Y_gas','R_gas','M_HI_gas',
                    'num_los','N_HI','c','Ic','threads','length'],             
              type_converters = wv.converters.blitz,
              extra_link_args=['-lgomp'],
              support_code = support,libraries = ['m','gomp'],
              extra_compile_args =['-O3 -fopenmp'])
    
    prefactor=(Msun/h)/mH/(Mpc/h/(1.0+redshift))**2
    N_HI*=prefactor

    n_DLA=len(np.where(N_HI>10**(20.3))[0])
    cross_section=np.pi*R_halo**2*n_DLA*1.0/num_los

    print 'n_DLA=',n_DLA
    #f=open('borrar.dat','w')
    #for i in xrange(len(X_los)):
    #    f.write(str(X_los[i])+' '+str(Y_los[i])+' '+str(N_HI[i])+'\n')
    #f.close()

    return cross_section
###############################################################################

#This function returns the cross section for a dark matter halo of mass M
#using the Barnes & Haehnelt 2014 method
#Omega_m ----------> value of the Omega_m cosmological parameter
#Omega_b ----------> value of the Omega_b cosmological parameter
#Omega_l ----------> value of the Omega_l cosmological parameter
#redshift ---------> redshift of the dark matter halo
#h ----------------> Hubble parameter at z=0 in units of 100 km/s/Mpc
#M ----------------> mass of the dark matter halo in Msun/h
#c0 ---------------> value of the Barnes & Haehnelt parameter c0
#f_HI -------------> value of the Barnes & Haehnelt parameter f_HI
#alpha_e ----------> value of the Barnes & Haehnelt parameter alpha_e
#V0c --------------> value of the Barnes & Haehnelt parameter V0c
#N_HI_ref ---------> value of N_HI that wants to be obtained
#verbose ----------> whether or not print some information in the screen
def cross_section_BH(Omega_m,Omega_b,Omega_l,redshift,h,M,c0,f_HI,alpha_e,V0c,
                     N_HI_ref=10**20.3,verbose=False):

    #compute the value of f_H
    f_H=0.76*Omega_b/Omega_m

    #compute the value of delta_c followign Bryan & Norman 1998
    Omega_m_z=Omega_m*(1.0+redshift)**3/(Omega_m*(1.0+redshift)**3+Omega_l)
    x=Omega_m_z-1.0
    delta_c=18.0*pi**2+82.0*x-39.0*x**2
    if verbose:
        print 'x = %f  delta_c = %f\n'%(x,delta_c); del x

    #compute the concentration of the HI halo (Eq. 6 of BH): c=Rv/Rs
    c=c0*(M/h/1e11)**(-0.109)*(4.0/(1.0+redshift))
    if verbose:
        print 'c = %f\n'%c

    #compute the virial radius of the dark matter halo (Eq. 4 of BH)
    Rv=46.1*(delta_c*Omega_m/24.4)**(-1.0/3.0)*(3.3/(1.0+redshift))\
        *(M/1e11)**(1.0/3.0) #kpc/h
    if verbose:
        print 'Rv = %f kpc/h'%(Rv)

    #compute the value of Rs
    Rs=Rv/c
    if verbose:
        print 'Rs = %f kpc/h \n'%(Rs)

    #compute the circular velocity of the halo (Eq. 3 of BH)
    Vc=96.6*(delta_c*Omega_m/24.4)**(1.0/6.0)*np.sqrt((1.0+redshift)/3.3)*\
        (M/1e11)**(1.0/3.0) #km/s
    if verbose:
        print 'Vc = %f km/s \n'%Vc

    #compute the HI mass in the dark matter halo (Eq. 2 of BH)
    M_HI=f_HI*f_H*np.exp(-(V0c/Vc)**alpha_e)*M
    if verbose:
        print 'M_halo = %e Msun/h'%(M)
        print 'M_HI   = %e Msun/h\n'%(M_HI)

    #compute the value of \rho_0 using the fact that:
    #M_HI = 4*pi*\int_0^Rv r^2dr rho_HI
    rho_0=M_HI/(4.0*pi*integral(c)*Rs**3) #h^2*Msun/kpc^3
    if verbose:
        print 'rho_0 = %e h^2 Msun/kpc^3\n'%(rho_0)

    #sanity check
    """
    M_HI_check=4.0*pi*rho_0*Rs**3*analytic_integral(c)
    if verbose:
        print '%e should be equal to %e\n'%(M_HI,M_HI_check)
    if M_HI>0.0:
        if np.absolute(M_HI-M_HI_check)/M_HI>1e-3:
            print 'error: different M_HI values:'
            print '%e %e\n'%(M_HI,M_HI_check)
            print 'relative difference = %e'%(np.absolute(M_HI-M_HI_check)/M_HI)
            sys.exit()
    else:
        if M_HI_check!=M_HI:
            print 'error: different M_HI values:'
            print '%e %e\n'%(M_HI,M_HI_check)
            sys.exit()
    """

    #find the radius R_b for which N_HI(b)=N_HI_ref
    #prefactor=(Msun/h)/mH/(kpc/h/(1.0+redshift))**2
    prefactor=(Msun/h)/mH/(kpc/h)**2 #working in physical space (not comoving!!)
    if M_HI>0.0:
        b_min=1e-10; b_max=1.0; iterations=50; final=False; i=0; tol=1e-4
        while (not(final)):
            b=10**(0.5*(np.log10(b_min)+np.log10(b_max)))*Rv
            N_HI=column_density(b,Rv,Rs,rho_0)*prefactor
        
            if (np.absolute(N_HI-N_HI_ref)/N_HI_ref)<tol or i>iterations:
                final=True
            else:
                if N_HI>N_HI_ref:
                    b_min=b/Rv
                else:
                    b_max=b/Rv
                i+=1
            if verbose:
                print'%e %e'%(b,N_HI)

        cross_section=pi*b**2
    else:
        cross_section=0.0

    if verbose:
        print 'cross_section = %e (kpc/h)^2'%(cross_section/h**2)

    return [cross_section,M_HI]


#this function computes 2*\int_0^l_max rho_HI(r) dl
#where r^2=l^2+b^2  and l_max=Rv^2-b^2
def column_density(b,Rv,Rs,rho_0):
    l_limits=[0.0,np.sqrt(Rv**2-b**2)]; yinit=[0.0]

    I=si.odeint(deriv_column_density,yinit,l_limits,args=(b,Rs,rho_0),
                rtol=1e-10,atol=1e-25,mxstep=10000000)[1][0]    
    return 2.0*I

def deriv_column_density(y,x,b,Rs,rho_0):
    r=np.sqrt(x**2+b**2)
    
    #return [rho_0/((r/Rs+3.0/4.0)*(r/Rs+1.0)**2)]
    #return [rho_0/((r/Rs+1e-5)**2*(r/Rs+1.0)**2)]
    return [rho_0/((r/Rs+0.06)**2*(r/Rs+1.0)**3)]

#This function returns \int_0^r_max x^2/[(x+3/4)*(x+1)^2] computed analytically
def analytic_integral(r):
    #value=9.0*np.log(r_max+3.0/4.0)-8.0*np.log(4*(1.0+r_max))+4.0/(1.0+r_max)
    #value-=(9.0*np.log(3.0/4.0)-8.0*np.log(4.0)+4.0)
    value = 9.0*np.log(4.0*r/3.0+1.0) - 4.0*r/(r+1.0) - 8.0*np.log(1.0+r)
    return value

#this function computes \int_0^r_max x^2/[(x+3/4)*(x+1)^2]
#In particular, it returns: 1.0/(rho_0*Rs**3) *4*pi*\int_0^Rv r^2dr rho_HI
def integral(r_max):
    r_limits=[0.0,r_max]; yinit=[0.0]
    
    I=si.odeint(deriv_HI_BH,yinit,r_limits,rtol=1e-9,atol=1e-9,
                 mxstep=10000000)[1][0]
    return I

def deriv_HI_BH(y,x):
    #return [x**2/((x+3.0/4.0)*(x+1.0)**2)]
    #return [x**2/((x+1e-5)**2*(x+1.0)**2)]
    return [x**2/((x+0.06)**2*(x+1.0)**3)]
###############################################################################

#This function computes the DLA incidence rate. It returns the incidence rate
#N_HI -------- > array containing the column densities
#f_HI ---------> array containing the column density distribution
def incidence_rate(N_HI,f_HI):
    indexes=np.where(f_HI>0.0)[0]; N_HI=N_HI[indexes]; f_HI=f_HI[indexes]
    del indexes; yinit=[0.0]; N_HI_limits=[10**(20.3),np.max(N_HI)]
    I=si.odeint(deriv_incidence_rate,yinit,N_HI_limits,rtol=1e-8,atol=1e-6,
                args=(N_HI,f_HI),mxstep=10000000)[1][0]
    return I

def deriv_incidence_rate(y,x,N_HI,f_HI):
    f_HI_I=10**(np.interp(np.log10(x),np.log10(N_HI),np.log10(f_HI)))
    return [f_HI_I]
###############################################################################

#This function compute the HI/H fraction given the photoionization rate and 
#the density of the gas particles
#snapshot_fname ------------> name of the N-body snapshot
#Gamma_UVB -----------------> value of the UVB photoionization rate
#self_shielding_correction -> apply (True) or not (False) the self-shielding
#f -------------------------> parameter of the Rahmati et al model (see eq. A1)
#n0 ------------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_1 -------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_2 -------------------> parameter of the Rahmati et al model (see eq. A1)
#beta ----------------------> parameter of the Rahmati et al model (see eq. A1)
#SF_temperature ------------> associate temperature of star forming particles
#If SF_temperature is set to None it will compute the temperature of the SF
#particles from their density and internal energy
def HI_from_UVB(snapshot_fname,Gamma_UVB,self_shielding_correction=False,
                f=0.0,n0=0.0,alpha_1=0.0,alpha_2=0.0,beta=0.0,
                SF_temperature=None):

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

    #read the density, electron fraction and internal energy
    #rho units: h^2 Msun / Mpc^3
    rho=readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10/1e-9 
    ne=readsnap.read_block(snapshot_fname,"NE  ",parttype=0) #electron fraction
    U=readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2

    #compute the mean molecular weight
    yhelium=(1.0-0.76)/(4.0*0.76) 
    mean_mol_weight=(1.0+4.0*yhelium)/(1.0+yhelium+ne); del ne

    #compute the temperature of the gas particles
    T=U*(gamma-1.0)*mH*mean_mol_weight/kB; del U, mean_mol_weight
    T=T.astype(np.float64)
 
    #for star forming particle assign T=10^4 K
    if SF_temperature!=None:
        SFR=readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR
        indexes=np.where(SFR>0.0)[0]; T[indexes]=SF_temperature
        del indexes,SFR; print '%e < T[K] < %e'%(np.min(T),np.max(T))


    #### Now compute the HI/H fraction following Rahmati et. al. 2013 ####
    #compute densities in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
    nH=0.76*h**2*Msun/Mpc**3/mH*rho*(1.0+redshift)**3; del rho
    nH=nH.astype(np.float64)

    #compute the case A recombination rate
    Lambda=315614.0/T
    alpha_A=1.269e-13*Lambda**(1.503)\
        /(1.0+(Lambda/0.522)**(0.47))**(1.923) #cm^3/s
    alpha_A=alpha_A.astype(np.float64)

    #Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
    Lambda_T=1.17e-10*np.sqrt(T)*np.exp(-157809.0/T)/(1.0+np.sqrt(T/1e5))#cm^3/s
    Lambda_T=Lambda_T.astype(np.float64)

    #compute the photoionization rate
    Gamma_phot=Gamma_UVB
    if self_shielding_correction:
        Gamma_phot*=\
            ((1.0-f)*(1.0+(nH/n0)**beta)**alpha_1 + f*(1.0+nH/n0)**alpha_2)

    #compute the coeficients A,B and C to calculate the HI/H fraction
    A = alpha_A + Lambda_T
    B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
    C = alpha_A

    #compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
    nH0=(B-np.sqrt(B**2-4.0*A*C))/(2.0*A); del nH
    nH0=nH0.astype(np.float32)

    return nH0

###############################################################################
#This routine returns the indexes of the particles residing within different
#enviroments:
#CDM particles within selected halos -------------> self.cdm_halos
#CDM particles within not selected halos ---------> self.cdm_halos2
#CDM particles in filaments (outside halos) ------> self.cdm_filaments
#gas particles within selected halos -------------> self.gas_halos
#gas particles within not selected halos ---------> self.gas_halos2
#gas particles in filaments (outside halos) ------> self.gas_filaments
#star particles within selected halos ------------> self.star_halos
#star particles within not selected halos --------> self.star_halos2
#star particles in filaments (outside halos) -----> self.star_filaments

#By selected halos we mean halos within a given mass interval. If all halos
#are wanted then set mass_interval=False and the IDs of particles within not
#selected halos will be an empty array

#snapshot_fname --> name of the N-body snapshot
#groups_fname ----> name of the folder containing the FoF/Subfind halos
#groups_number ---> number of the FoF/Subfind file to read
#long_ids_flag ---> True if particle IDs are 64 bits. False otherwise
#SFR_flag --------> True for simulations with baryons particles. False otherwise
#mass_interval ---> True if IDs of halos in a given mass interval wanted
#min_mass --------> interval minimim mass in Msun/h (if mass_interval=True)
#max_mass --------> interval maximum mass in Msun/h (if mass_interval=True)

class particle_indexes:
    def __init__(self,snapshot_fname,groups_fname,groups_number,long_ids_flag,
                 SFR_flag,mass_interval,min_mass,max_mass):

        #read snapshot head and obtain Nall
        head=readsnap.snapshot_header(snapshot_fname)
        Nall=head.nall

        #compute the total number of particles in the simulation
        Ntotal=np.sum(Nall,dtype=np.uint64); del head,Nall
        print 'Total number of particles in the simulation =',Ntotal

        #create an array with all elements equal to 0 and length equal to the 
        #total number of particles in the simulations. Note that since we will
        #assign negative numbers to Total_ID, we can not use dtype=np.uint8
        Total_ID=np.zeros(Ntotal,dtype=np.int8)

        #read IDs: we subtract 1 such as the IDs start at 0 and end in N-1
        #assign a value of 5 and 10 to the gas an star particles respectively
        #The possibilities are these:
        #-----------------------------------------------------------
        # ID[N]=10 --> star particle not residing in a halo
        # ID[N]=9 ---> star particle residing in a not selected halo
        # ID[N]=8 ---> star particle residing in a selected halo
        ID=readsnap.read_block(snapshot_fname,"ID  ",parttype=4)-1 #normalized
        Total_ID[ID]=10; del ID 
        #-----------------------------------------------------------
        # ID[N]=5 ---> gas particle not residing in a halo
        # ID[N]=4 ---> gas particle residing in a not selected halo
        # ID[N]=3 ---> gas particle residing in a selected halo
        ID=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized
        Total_ID[ID]=5; del ID 
        #-----------------------------------------------------------
        # ID[N]=0 ---> CDM particle not residing in a halo
        # ID[N]=-1 --> CDM particle residing in a not selected halo
        # ID[N]=-2 --> CDM particle residing in a selected halo
        #-----------------------------------------------------------

        #read FoF halos information
        halos=readfof.FoF_catalog(groups_fname,groups_number,
                                  long_ids=long_ids_flag,swap=False,
                                  SFR=SFR_flag)
        M_FoF=halos.GroupMass*1e10   #Msun/h
        ID_FoF=halos.GroupIDs-1      #normalize IDs
        Len=halos.GroupLen           #number of particles in the halo
        Offset=halos.GroupOffset     #offset of the halo in the ID array
        del halos

        #first: find the IDs of the particles in the filaments
        #here by filaments we refer to everything outside halos
        Total_ID[ID_FoF]-=1
        self.cdm_filaments=np.where(Total_ID==0)[0]
        self.gas_filaments=np.where(Total_ID==5)[0]
        self.star_filaments=np.where(Total_ID==10)[0]
        
        #keep only with the halos in a given mass interval
        if mass_interval:
            indexes=np.where((M_FoF>=min_mass) & (M_FoF<max_mass))[0]
            Len=Len[indexes]; Offset=Offset[indexes]; del indexes

            Total_len=np.sum(Len,dtype=np.int64); Offs=0
            if long_ids_flag:
                ID_FoF_sample=np.empty(Total_len,dtype=np.uint64)
            else:
                ID_FoF_sample=np.empty(Total_len,dtype=np.uint32)
            for i in xrange(len(Offset)):
                ID_FoF_sample[Offs:Offs+Len[i]]=\
                    ID_FoF[Offset[i]:Offset[i]+Len[i]]
                Offs+=Len[i]
            ID_FoF=ID_FoF_sample; del ID_FoF_sample

        #decrease by one unit the array Total_ID in the positions of the 
        #particles composing the FoF halo
        Total_ID[ID_FoF]-=1; del ID_FoF

        #second: find the IDs of particles residing in the selected halos
        self.cdm_halos=np.where(Total_ID==-2)[0]
        self.gas_halos=np.where(Total_ID==3)[0]
        self.star_halos=np.where(Total_ID==8)[0]

        #third: find the IDs of particles residing in not selected halos
        self.cdm_halos2=np.where(Total_ID==-1)[0]
        self.gas_halos2=np.where(Total_ID==4)[0]
        self.star_halos2=np.where(Total_ID==9)[0]
        del Total_ID

        #sanity check
        Total_IDs=len(self.cdm_halos)+len(self.cdm_halos2)+\
            len(self.cdm_filaments)+len(self.gas_halos)+len(self.gas_halos2)+\
            len(self.gas_filaments)+len(self.star_halos)+len(self.star_halos2)+\
            len(self.star_filaments)

        if Total_IDs!=Ntotal:
            print 'IDs not properly split!!!'
            sys.exit()
###############################################################################

#This function computes the peak flux given a distribution of HI
#pos ---------------------> positions of the particles in 2D (x,y)
#M_HI --------------------> HI mass of the particles
#BoxSize -----------------> size of the simulation box (same units as pos)
#delta_r -----------------> width of the slice (same units as pos and BoxSize)
#redshift ----------------> redshift
#dims --------------------> number of point in the grid along each direction
#mean_delta_Tb -----------> mean value of delta_Tb in mK
#mean_M_HI_grid ----------> mean HI mass in each grid cell
#ang_res -----------------> angular resolution in arcmin
#grid_res ----------------> spatial resolutin (same units as pos and BoxSize)
def peak_flux(pos,M_HI,BoxSize,delta_r,redshift,dims,mean_delta_Tb,
              mean_M_HI_grid,ang_res,grid_res,verbose=False):
              
    c=3e8 #light speed in m/s
    kB=1.38e3 #Boltzmann constant in mJy*m^2/mK

    #compute the value of M_HI in each grid point
    M_HI_grid=np.zeros(dims**2,dtype=np.float32) 
    CIC.CIC_serial_2D(pos,dims,BoxSize,M_HI_grid,M_HI) #find surface densities
    if verbose:
        print '%e should be equal to \n%e'\
            %(np.sum(M_HI,dtype=np.float64),np.sum(M_HI_grid,dtype=np.float64))
        print 'Omega_HI (slice) = %e'\
            %(np.sum(M_HI_grid,dtype=np.float64)/(BoxSize**2*delta_r)/rho_crit)

    #compute delta_HI in the grid cells
    delta_HI=M_HI_grid/mean_M_HI_grid-1.0
    if verbose:
        print '%f < delta_HI < %f'%(np.min(delta_HI),np.max(delta_HI))
        print '<delta_HI> = %f'%np.mean(delta_HI,dtype=np.float64)

    #we assume that Ts>>T_CMB
    delta_Tb=mean_delta_Tb*(1.0+delta_HI)
    #*Hubble/(Hubble+(1.0+redshift)*dVdr)
    if verbose:
        print '\n%f < delta_Tb (mK) < %f'%(np.min(delta_Tb),np.max(delta_Tb))
        print '<delta_Tb> = %e (mK)'%(np.mean(delta_Tb,dtype=np.float64))

    """#create an image
    f=open('borrar_temperature.dat','w')
    for i in range(dims**2):
        x=(BoxSize/dims)*(i/dims)
        y=(BoxSize/dims)*(i%dims)
        f.write(str(x)+' '+str(y)+' '+str(delta_Tb[i])+'\n')
    f.close()
    """

    #now compute specific intensity
    prefactor=2.0*kB*(nu0*1e6/(1.0+redshift))**2/c**2 #mJy
    Inu=prefactor*delta_Tb
    if verbose:
        print '\nprefactor=',prefactor
        print 'sum of Inu = %e mJy/sr'%np.sum(Inu,dtype=np.float64)
        print np.min(Inu),'< I_nu (mJy/sr) <',np.max(Inu)
        print '<I_nu> = %e mJy/sr'%(np.mean(Inu,dtype=np.float64))
        print '<I_nu>_universe = %e mJy/sr'%(mean_delta_Tb*prefactor)
    #I_nu_mean=4.76/(pi/180.0)**2*h*Omega_HI_ref*100.0/Hubble #Bharadwaj-Pandey
    #print '<I_nu> = %e Jy/sr'%I_nu_mean

    """#create an image
    f=open('borrar.dat','w')
    for i in range(dims**2):
        x=(BoxSize/dims)*(i/dims)
        y=(BoxSize/dims)*(i%dims)
        f.write(str(x)+' '+str(y)+' '+str(Inu[i])+'\n')
    f.close()"""

    #compute the smooth array
    if verbose:
        print '\nSmoothing the field...'
    smooth=smoothing(dims,BoxSize,grid_res)
    if verbose:
        print np.min(smooth),'< smoothing <',np.max(smooth)
    smooth=np.reshape(smooth,(dims,dims))

    #Fourier transform the intensity field
    Inu=np.reshape(Inu,(dims,dims))
    if verbose:
        print 'Computing the FFT of the field...'; start_fft=time.clock()
    Inu_k=scipy.fftpack.fftn(Inu,overwrite_x=True); del Inu
    if verbose:
        print 'done: time taken for computing the FFT=',time.clock()-start_fft

    #smooth the density field
    Inu_k*=smooth

    #compute Fourier transform back to obtain Inu smoothed
    if verbose:
        print 'Computing the inverse FFT of the field..'; start_fft=time.clock()
    Inu=scipy.fftpack.ifftn(Inu_k,overwrite_x=True); del Inu_k
    Inu=np.ravel(Inu)
    if verbose:
        print 'done: time taken for computing the FFT=',time.clock()-start_fft
        print np.min(Inu.imag),'< Inu.imag <',np.max(Inu.imag)

    Inu=np.real(Inu) #just keep with the real part
    if verbose:
        print np.min(Inu),'< I_nu (mJy/sr) <',np.max(Inu)
        print 'sum of Inu = %e mJy/sr'%np.sum(Inu,dtype=np.float64)
        print 'points with Inu < 0.0  ---> ',len(np.where(Inu<0.0)[0])
    
    #remove negative values
    Inu[np.where(Inu<0.0)[0]]=0.0

    #Move from mJy/sr to mJy/beam
    Inu*=(ang_res/60.0*pi/180.0)**2
    if verbose:
        print 'peak flux = %e mJy/beam'%(np.max(Inu))

    return Inu
    

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
    """
    wv.inline(code,['dims','smooth_array','R','BoxSize'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'],
              extra_compile_args =['-O3'])

    return smooth_array

###############################################################################




###############################################################################
################################### USAGE #####################################
###############################################################################

###### Bagla parameters ######
"""
z=3; h=0.7; Omega_HI=1e-3 
f_MF='/home/villa/bias_HI/mass_function/ST_MF_z=3.dat'

[Mmin,Mmax,f1,f2,f3]=Bagla_parameters(z,h,Omega_HI,f_MF)

print '\nM_min (z=%2.1f) = %e Msun/h'%(z,Mmin)
print 'M_max (z=%2.1f) = %e Msun/h\n'%(z,Mmax)
print 'f1 = %e'%f1
print 'f2 = %e'%f2
print 'f3 = %e'%f3
"""

###### Bagla_HI_assignment ######
"""
snapshot_fname='/home/villa/bias_HI/Efective_model_60Mpc/snapdir_008/snap_008'
groups_fname='/home/villa/bias_HI/Efective_model_60Mpc/FoF_0.2'
groups_number=8
method=3; Omega_HI=1e-3; 
f_MF='/home/villa/bias_HI/mass_function/ST_MF_z=3.dat' #mass function file
long_ids_flag=False; SFR_flag=True
[IDs,M_HI]=Bagla_HI_assignment(snapshot_fname,groups_fname,groups_number,
                              Omega_HI,method,f_MF,long_ids_flag,SFR_flag)

BoxSize=readsnap.snapshot_header(snapshot_fname).boxsize/1e3 #Mpc/h
print IDs
print np.min(IDs),'< IDs <',np.max(IDs)
print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)
"""

###### method 1 HI assignment ######
"""
snapshot_fname='/home/villa/bias_HI/Efective_model_60Mpc/snapdir_008/snap_008'
HI_frac=0.9
Omega_HI=1e-3

[IDs,M_HI]=method_1_HI_assignment(snapshot_fname,HI_frac,Omega_HI)

BoxSize=readsnap.snapshot_header(snapshot_fname).boxsize/1e3 #Mpc/h
print IDs
print np.min(IDs),'< IDs <',np.max(IDs)
print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)
"""

###### Dave HI assignment ######
"""
snapshot_fname='/home/villa/bias_HI/Efective_model_120Mpc/snapdir_008/snap_008'
HI_frac=0.95
#z=3 --> 1.246181 (120Mpc) 1.362889 (60 Mpc) 1.436037 (30 Mpc) 1.440990 (15 Mpc)
fac=1.246181

[IDs,M_HI]=Dave_HI_assignment(snapshot_fname,HI_frac,fac)

BoxSize=readsnap.snapshot_header(snapshot_fname).boxsize/1e3 #Mpc/h
print IDs
print np.min(IDs),'< IDs <',np.max(IDs)
print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)
"""

###### Barnes & Haehnelt HI assignment ######
"""
snapshot_fname='../Efective_model_60Mpc/snapdir_008/snap_008'
groups_fname='../Efective_model_60Mpc'
groups_number=8
long_ids_flag=False; SFR_flag=True

[IDs,M_HI]=Barnes_Haehnelt(snapshot_fname,groups_fname,groups_number,
                           long_ids_flag,SFR_flag)                     
"""

###### Nagamine HI assignment ######
"""
snapshot_fname='/home/villa/bias_HI/Efective_model_60Mpc/snapdir_008/snap_008'
FactorSN=0.1
MaxSfrTimescale=1.5 #Gadget internal time units
PhysDensThresh=0.0012805
[IDs,M_HI]=Nagamine_HI_assignment(snapshot_fname,FactorSN,MaxSfrTimescale,
                                  PhysDensThresh,correct_H2=False)
print M_HI[IDs]
HI_mass=readsnap.read_block(snapshot_fname,"MHI ",parttype=0)*1e10 #Msun/h
print HI_mass
print 'Omega_HI = %e'%(np.sum(HI_mass,dtype=np.float64)/60.0**3/rho_crit)
"""

###### NHI LOS SPH ######
"""
number=128**3
R=np.zeros(number,dtype=np.float32)
X=np.zeros(number,dtype=np.float32)
Y=np.zeros(number,dtype=np.float32)
M_HI=np.zeros(number,dtype=np.float32)
BoxSize=512.0 #Mpc/h
threads=5
cells=1000

column_density=NHI_los_sph(R,X,Y,M_HI,BoxSize,cells,threads)
print '%e < N_HI [cm^(-2)] < %e'%(np.min(column_density),np.max(column_density))
"""

###### HI_from_UVB ######
"""
snapshot_fname='/home/villa/bias_HI/Efective_model_60Mpc/snapdir_008/snap_008'
f_TREECOOL='/home/villa/bias_HI/Efective_model_60Mpc/FoF_0.2/TREECOOL_bestfit_g1.3' #TREECOOL file

#find the redshift of the snapshot
z=readsnap.snapshot_header(snapshot_fname).redshift 

#read the TREECOOL file
data=np.loadtxt(f_TREECOOL); logz=data[:,0]; Gamma_UVB=data[:,1]; del data

#find the value of Gamma_UVB at the redshift of interest
Gamma_UVB=np.interp(np.log10(z),logz,Gamma_UVB) 
print 'Gamma_UVB(z=%2.1f) = %e s^{-1}'%(z,Gamma_UVB)

nH0=HI_from_UVB(snapshot_fname,Gamma_UVB,self_shielding_correction=False,
                f=0.0,n0=0.0,alpha_1=0.0,alpha_2=0.0,beta=0.0,
                SF_temperature=None)
print nH0
print np.min(nH0),'< HI/H <',np.max(nH0)
print 'Sum HI/H = %e'%(np.sum(nH0,dtype=np.float64))

nH0_Gadget=readsnap.read_block(snapshot_fname,"NH  ",parttype=0) #HI/H
print nH0_Gadget
print np.min(nH0_Gadget),'< HI/H <',np.max(nH0_Gadget)
print 'Sum HI/H = %e'%(np.sum(nH0_Gadget,dtype=np.float64))

rho=readsnap.read_block(snapshot_fname,"RHO ",parttype=0) #1e10 h^2 Msun/kpc^3
import random

Gamma_UVB/=1.4
nH03=HI_from_UVB(snapshot_fname,Gamma_UVB,self_shielding_correction=False,
                f=0.0,n0=0.0,alpha_1=0.0,alpha_2=0.0,beta=0.0,
                SF_temperature=None)
print nH03
print np.min(nH03),'< HI/H <',np.max(nH03)
print 'Sum HI/H = %e'%(np.sum(nH03,dtype=np.float64))

indexes=np.arange(len(nH0)); indexes=random.sample(indexes,1000000)
np.savetxt('borrar.dat',np.transpose([rho[indexes],nH0[indexes],
                                      nH0_Gadget[indexes],nH03[indexes]]))

indexes=np.where(rho<2e-5)[0]
print 'Sum HI/H = %e'%(np.sum(nH0[indexes],dtype=np.float64))
print 'Sum HI/H = %e'%(np.sum(nH03[indexes],dtype=np.float64))
"""

###### particle indexes ######
"""
snapshot_fname='../Efective_model_60Mpc/snapdir_008/snap_008'
groups_fname='../Efective_model_60Mpc'
groups_number=8
long_ids_flag=False; SFR_flag=True
mass_interval=False
min_mass=0.0; max_mass=0.0
IDs=particle_indexes(snapshot_fname,groups_fname,groups_number,long_ids_flag,
                     SFR_flag,mass_interval,min_mass,max_mass)

print 'IDs of particles in not selected halos;'
print IDs.cdm_halos2
print IDs.gas_halos2
print IDs.star_halos2

print 'IDs of particles in selected halos'
print IDs.cdm_halos
print IDs.gas_halos
print IDs.star_halos

print 'IDs of particles in filaments'
print IDs.cdm_filaments
print IDs.gas_filaments
print IDs.star_filaments
"""

