import numpy as np
import readsnap
import readfof
import scipy.weave as wv
import sys

################################ routines available ###########################
#Bagla_HI_assignment
#method_1_HI_assignment
#Dave_HI_assignment
#NHI_los_sph
#Barnes_Haehelt_HI_assignment
#particle_indexes

###############################################################################

################################# UNITS #####################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

Mpc=3.0856e24 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams

pi=np.pi
#############################################################################

###############################################################################
#This routine uses the Bagla et al. 2009 method to assign HI to the CDM (and
#baryon particles)
#snapshot_fname --> name of the N-body snapshot
#groups_fname ----> name of the folder containing the FoF/Subfind halos
#groups_number ---> number of the FoF/Subfind file to read
#Omega_HI_ref ----> value of Omega_HI that wants to be obtained
#method ----------> 1,2 or 3 (see Bagla et al. 2009 for the different methods)
#long_ids_flag ---> True if particle IDs are 64 bits. False otherwise
#SFR_flag --------> True for simulations with baryons particles. False otherwise
#the routine returns the IDs of the particles to whom HI has been assigned
#and its HI masses. Note that the HI masses array has a length equal to the 
#total number of particles in the simulation
#If the positions of the particles to which HI has been assigned is wanted,
#one should first sort the positions and then use the IDs to select them
def Bagla_HI_assignment(snapshot_fname,groups_fname,groups_number,Omega_HI_ref,
                       method,long_ids_flag,SFR_flag):
                       

    #read snapshot header and obtain BoxSize, redshift and h
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Nall=head.nall
    redshift=head.redshift
    h=head.hubble
    
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
    print '\nNumber of FoF halos:',len(pos_FoF),len(M_FoF)
    print '%f < X [Mpc/h] < %f'%(np.min(pos_FoF[:,0]),np.max(pos_FoF[:,0]))
    print '%f < Y [Mpc/h] < %f'%(np.min(pos_FoF[:,1]),np.max(pos_FoF[:,1]))
    print '%f < Z [Mpc/h] < %f'%(np.min(pos_FoF[:,2]),np.max(pos_FoF[:,2]))
    print '%e < M [Msun/h] < %e\n'%(np.min(M_FoF),np.max(M_FoF))

    #only consider halos with Vcirc < 200 km/s: see Bagla et al 2009
    Mmax=1e10*(200.0/60.0)**3*((1.0+redshift)/4.0)**(-1.5)*h #Msun/h
    Mmin=1e10*(30.0/60.0)**3*((1.0+redshift)/4.0)**(-1.5)*h #Msun/h
    print 'Mmin=%e Msun/h: log10(Min)=%f'%(Mmin,np.log10(Mmin))
    print 'Mmax=%e Msun/h: log10(Max)=%f\n'%(Mmax,np.log10(Mmax))
    if method==1:
        indexes=np.where((M_FoF>Mmin) & (M_FoF<Mmax))[0]
        M_FoF_HI=M_FoF[indexes]; Len_HI=Len[indexes]; Offset_HI=Offset[indexes]
        del indexes
    else:
        M_FoF_HI=M_FoF; Len_HI=Len; Offset_HI=Offset

    #Find the value of the parameter f1, f2 or f3 (methods 1,2 and 3 of Bagla)
    iterations=20; final=False; i=0
    f1_min=1e-6; f1_max=1.0; tol=1e-3
    f1=10**(0.5*(np.log10(f1_min)+np.log10(f1_max)))
    while not(final):
        if method==1:
            Omega_HI=np.sum(f1*M_FoF_HI,
                            dtype=np.float64)
        elif method==2:
            Omega_HI=np.sum(f1/(1.0+(M_FoF_HI/Mmax)**2)*M_FoF_HI,
                            dtype=np.float64)
        elif method==3:
            Omega_HI=np.sum(f1/(1.0+(M_FoF_HI/Mmax))*M_FoF_HI,
                            dtype=np.float64)
        else:
            print 'Choose between method 1 (eq. 4), 2 (eq. 5) or 3 (eq. 6)'
            print 'of Bagla et al. 2009'
        Omega_HI=Omega_HI/BoxSize**3/rho_crit
        print 'f1=%e --> Omega_HI=%e'%(f1,Omega_HI)

        if (np.absolute((Omega_HI-Omega_HI_ref)/Omega_HI_ref)<tol) or i>iterations:
            final=True
        else:
            if (Omega_HI<Omega_HI_ref):
                f1_min=f1
            else:
                f1_max=f1
        f1=10**(0.5*(np.log10(f1_min)+np.log10(f1_max)))
        i+=1

    #define array M_HI that contains the HI masses of the particles and fill it
    size_M_HI_array=np.sum(Nall,dtype=np.int64)
    M_HI=np.zeros(size_M_HI_array,dtype=np.float32); Mass_tot=0.0 
    size_IDs_array=np.sum(Len_HI,dtype=np.int64); IDs_offset=0
    if long_ids_flag:
        IDs=np.empty(size_IDs_array,dtype=np.int64)
    else:
        IDs=np.empty(size_IDs_array,dtype=np.int32)

    for index in range(len(M_FoF_HI)):

        indexes=ID_FoF[Offset_HI[index]:Offset_HI[index]+Len_HI[index]]
        IDs[IDs_offset:IDs_offset+Len_HI[index]]=indexes
        IDs_offset+=Len_HI[index]

        if method==1:
            M_HI_halo=f1*M_FoF_HI[index]
        elif method==2:
            M_HI_halo=f1/(1.0+(M_FoF_HI[index]/Mmax)**2)*M_FoF_HI[index]
        else:
            M_HI_halo=f1/(1.0+M_FoF_HI[index]/Mmax)*M_FoF_HI[index]
        M_HI[indexes]+=(M_HI_halo*1.0/Len_HI[index])

    print '\nTotal HI mass in halos = %e Msun/h'%np.sum(M_HI,dtype=np.float64)
    print 'Omega_HI (halos) = %e\n'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

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
#the routine returns the IDs of the particles to whom HI has been assigned
#and its HI masses. Note that the HI masses array has a length equal to the 
#total number of particles in the simulation
#If the positions of the particles to which HI has been assigned is wanted,
#one should first sort the positions and then use the IDs to select them
def Dave_HI_assignment(snapshot_fname,HI_frac,fac):

    #read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head=readsnap.snapshot_header(snapshot_fname)
    BoxSize=head.boxsize/1e3 #Mpc/h
    Nall=head.nall
    redshift=head.redshift
    h=head.hubble

    #read RHO, NH0 and the masses of the baryon particles
    nH0=readsnap.read_block(snapshot_fname,"NH  ",parttype=0)*fac   #HI/H
    mass=readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h
    R=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3     #Mpc/h

    print np.min(R),'< R [Mpc/h] <',np.max(R),'\n'

    #compute the value of Omega_HI without any post-processing
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

    #define the array M_HI
    IDs=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized
    size_M_HI_array=np.sum(Nall,dtype=np.int64)
    M_HI=np.zeros(size_M_HI_array,dtype=np.float32)
    M_HI[IDs]=(M1+M2).astype(np.float32); del M1,M2

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
    Ntotal=np.sum(Nall,dtype=np.int32)
    print 'Total number of particles in the simulation: %d\n'%Ntotal

    #compute the value of Omega_b
    Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
    Omega_b=Omega_m-Omega_cdm
    print 'Omega_cdm =',Omega_cdm; print 'Omega_b   =',Omega_b
    print 'Omega_m   =',Omega_m,'\n'

    #find Barnes & Haehnelt parameters
    f_HI=0.35
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
            M_HI[indexes]+=M_HI_halo[index]*(radii**2/nH0_part**2)\
                /np.sum(radii**2/nH0_part**2,dtype=np.float64)
        else:
            No_gas_halos+=1
    print '\nNumber of halos with no gas particles=',No_gas_halos

    #compute again the value of Omega_HI
    print 'Omega_HI = %e\n'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return [IDs,M_HI]
                             
    

        #M_HI[indexes]+=(M_HI_halo*1.0/Len_HI[index])

###############################################################################

#this routine computes the column density along line of sights (los) that cross
#the entire simulation box from z=0 to z=BoxSize. The los are place in a grid
#within the XY plane, and the number of those is given by cells x cells
#R ------------> array containing the smoothing lenghts of the particles 
#X ------------> array containing the X-positions of the particles
#Y ------------> array containing the Y-positions of the particles
#M_HI ---------> array containing the HI masses of the particles
#cells --------> size of the los grid along one direction
#threads ------> number of threads to use in the computation (openmp threads)
def NHI_los_sph(R,X,Y,M_HI,BoxSize,cells,threads):

    #create a table with c I(c)
    intervals=1001 #the values of c goes as 0.000, 0.001, 0.002, 0.003, .....
    c=np.linspace(0.0,1.0,intervals)
    Ic=np.empty(intervals,dtype=np.float32)
    for i in range(intervals):
        Ic[i]=Iwdl(c[i])
    Ic[0]=Ic[1]

    #create the column density los array. Each element contains the column
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
    wv.inline(code,['R','length','X','Y','c','Ic','Box','cell','M_HI',
                    'column_density','threads'],
              type_converters = wv.converters.blitz,
              extra_link_args=['-lgomp'],
              support_code = support,libraries = ['m','gomp'],
              extra_compile_args =['-O3 -fopenmp'])
    
    return column_density


#This functions returns \int (1-6u^2+6u^3)udu/sqrt(u^2-c^2):  c=b/h
def I1(u,c):
    u2=u**2; c2=c**2; d=np.sqrt(u2-c2)
    return 0.25*(d*(c2*(9.0*u-16.0)+6.0*u**3-8.0*u2+4.0)+\
        9.0*c**4*np.log(2.0*(d+u)))

#This functions returns \int 2(1-u)^3udu/sqrt(u^2-c^2):  c=b/h
def I2(u,c):
    u2=u**2; c2=c**2; d=np.sqrt(u2-c2); g=3.0*(c2+4.0)/8.0
    return -2.0*(g*c2*np.log(2.0*(d+u))+d*(g*u-2.0*c2+u**3/4.0-u2-1.0))


#Function used to compute the integral \int_0^l_max w(r,h) dl:   u=r/h:  c=b/h
def Iwdl(c):
    if c>0.5:
        return 8.0*(I2(1.0,c)-I2(c,c))/pi
    else:
        return 8.0*(I2(1.0,c)-I2(0.5,c)+I1(0.5,c)-I1(c,c))/pi
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
        #total number of particles in the simulations
        Total_ID =np.zeros(Ntotal,dtype=np.int8)

        #read IDs: we subtract 1 such as the IDs start at 0 and end in N-1
        #The possibilities are these:
        # ID[N]=10 --> star particle not residing in a halo
        # ID[N]=9 ---> star particle residing in a not selected halo
        # ID[N]=8 ---> star particle residing in a selected halo
        # ID[N]=5 ---> gas particle not residing in a halo
        # ID[N]=4 ---> gas particle residing in a not selected halo
        # ID[N]=3 ---> gas particle residing in a selected halo
        # ID[N]=0 ---> CDM particle not residing in a halo
        # ID[N]=-1 --> CDM particle residing in a not selected halo
        # ID[N]=-2 --> CDM particle residing in a selected halo

        #assign a value of 5 and 10 to the gas an star particles respectively
        ID=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1 #normalized
        Total_ID[ID]=5; del ID 
        ID=readsnap.read_block(snapshot_fname,"ID  ",parttype=4)-1 #normalized
        Total_ID[ID]=10; del ID 

        #read FoF halos information
        halos=readfof.FoF_catalog(groups_fname,groups_number,
                                  long_ids=False,swap=False,SFR=True)
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
        
        #just keep with the halos in a given mass interval
        if mass_interval:
            indexes=np.where((M_FoF>min_mass) & (M_FoF<max_mass))[0]
            Len=Len[indexes]; Offset=Offset[indexes]; del indexes

            Total_len=np.sum(Len,dtype=np.int64); Offs=0
            ID_FoF_sample=np.empty(Total_len,dtype=np.int64)
            for i in range(len(Offset)):
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
################################### USAGE #####################################
###############################################################################

###### Bagla_HI_assignment ######
"""
snapshot_fname='/home/villa/bias_HI/Efective_model_60Mpc/snapdir_008/snap_008'
groups_fname='/home/villa/bias_HI/Efective_model_60Mpc'
groups_number=8
method=1; Omega_HI=1e-3
long_ids_flag=False; SFR_flag=True
[IDs,M_HI]=Bagla_HI_assigment(snapshot_fname,groups_fname,groups_number,
                              Omega_HI,method,long_ids_flag,SFR_flag)
                              
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
snapshot_fname='/home/villa/bias_HI/Efective_model_15Mpc/snapdir_008/snap_008'
HI_frac=0.95
fac=1.436037

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
