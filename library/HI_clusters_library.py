#This library is used for the HI & clusters project

import numpy as np
import readsnap
import sys,os 

################################# UNITS #####################################
rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3

yr    = 3.15576e7  #seconds
km    = 1e5        #cm
Mpc   = 3.0856e24  #cm
kpc   = 3.0856e21  #cm
Msun  = 1.989e33   #g
Ymass = 0.24       #helium mass fraction
mH    = 1.6726e-24 #proton mass in grams
gamma = 5.0/3.0    #ideal gas
kB    = 1.3806e-26 #gr (km/s)^2 K^{-1}
nu0   = 1420.0     #21-cm frequency in MHz

pi = np.pi
#############################################################################

################################ Rahmati ######################################
#Notice that in this implementation we set the temperature of the star-forming
#particles to 1e4 K, and use the Rahmati formula to compute the HI/H fraction.
#The routine computes and returns the HI/H fraction taking into account the
#HI self-shielding and the presence of molecular hydrogen.
#snapshot_fname ----------> file containing the snapshot
#TREECOOL_file -----------> file containing the TREECOOL file
#T_block -----------------> whether the snapshot contains the TEMP block or not
#Gamma_UVB ---------------> value of Gamma_UVB. None to read it from TREECOOL
#SF_temperature ----------> The value to set the temperature of the SF particles
#self_shielding_correction --> whether to correct HI fraction for self-shielding
#correct_H2 --------------> correct HI fraction for presence of H2:
#                           'BR': Blitz-Rosolowsky, 'THINGS' or False    
def Rahmati(snapshot_fname, TREECOOL_file, T_block=True, Gamma_UVB=None,
            SF_temperature=1e4, self_shielding_correction=True,
            correct_H2='BR'):

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L 
    head     = readsnap.snapshot_header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h   
    Nall     = head.nall
    Masses   = head.massarr*1e10 #Msun/h    
    Omega_m  = head.omega_m
    Omega_l  = head.omega_l
    redshift = head.redshift
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc 
    h        = head.hubble

    # read/compute the temperature of the gas particles
    if T_block:
        T = readsnap.read_block(snapshot_fname,"TEMP",parttype=0) #K
    else:
        U = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #km/s
        T = U/1.5/(1.380658e-16/(0.6*1.672631e-24)*1e-10)         #K
    T = T.astype(np.float64) #to increase precision use float64 variables
    print '%.3e < T[K] < %.3e'%(np.min(T),np.max(T))

    # read the density block: units of h^2 Msun/Mpc^3
    rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10/1e-9 
    print '%.3e < rho < %.3e'%(np.min(rho),np.max(rho))

    # Rahmati et. al. 2013 self-shielding parameters (table A1)
    z_t       = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00])
    n0_t      = np.array([-2.94,-2.29,-2.06,-2.13,-2.23,-2.35]); n0_t=10**n0_t
    alpha_1_t = np.array([-3.98,-2.94,-2.22,-1.99,-2.05,-2.63])
    alpha_2_t = np.array([-1.09,-0.90,-1.09,-0.88,-0.75,-0.57])
    beta_t    = np.array([1.29, 1.21, 1.75, 1.72, 1.93, 1.77])
    f_t       = np.array([0.01, 0.03, 0.03, 0.04, 0.02, 0.01])

    # compute the self-shielding parameters at the redshift of the N-body
    n0      = np.interp(redshift,z_t,n0_t)
    alpha_1 = np.interp(redshift,z_t,alpha_1_t)
    alpha_2 = np.interp(redshift,z_t,alpha_2_t)
    beta    = np.interp(redshift,z_t,beta_t)
    f       = np.interp(redshift,z_t,f_t)
    print 'n0 = %e\nalpha_1 = %2.3f\nalpha_2 = %2.3f\nbeta = %2.3f\nf = %2.3f'\
        %(n0,alpha_1,alpha_2,beta,f)

    # find the value of the photoionization rate
    if Gamma_UVB==None:
        data = np.loadtxt(TREECOOL_file);  logz = data[:,0]; 
        Gamma_UVB = data[:,1]
        Gamma_UVB = np.interp(np.log10(1.0+redshift),logz,Gamma_UVB);  del data
        print 'Gamma_UVB(z=%2.2f) = %e s^{-1}'%(redshift,Gamma_UVB)


    # for star forming particle assign T=10^4 K
    if SF_temperature!=None and T_block==True:
        SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR
        indexes = np.where(SFR>0.0)[0];  T[indexes] = SF_temperature
        del indexes,SFR; print '%.3e < T[K] < %.3e'%(np.min(T),np.max(T))
    

    #### Now compute the HI/H fraction following Rahmati et. al. 2013 ####
    # compute densities in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
    nH = 0.76*h**2*Msun/Mpc**3/mH*rho*(1.0+redshift)**3; #del rho
    nH = nH.astype(np.float64)

    #compute the case A recombination rate
    Lambda  = 315614.0/T
    alpha_A = 1.269e-13*Lambda**(1.503)\
        /(1.0+(Lambda/0.522)**(0.47))**(1.923) #cm^3/s
    alpha_A = alpha_A.astype(np.float64)

    #Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
    Lambda_T=1.17e-10*np.sqrt(T)*np.exp(-157809.0/T)/(1.0+np.sqrt(T/1e5))#cm^3/s
    Lambda_T=Lambda_T.astype(np.float64)

    #compute the photoionization rate
    Gamma_phot = Gamma_UVB
    if self_shielding_correction:
        Gamma_phot *=\
            ((1.0-f)*(1.0+(nH/n0)**beta)**alpha_1 + f*(1.0+nH/n0)**alpha_2)
    print 'Gamma_phot = ',Gamma_phot

    #compute the coeficients A,B and C to calculate the HI/H fraction
    A = alpha_A + Lambda_T
    B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
    C = alpha_A

    #compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
    nH0 = (B-np.sqrt(B**2-4.0*A*C))/(2.0*A); del nH
    nH0 = nH0.astype(np.float32)

    #correct for the presence of H2
    if correct_H2!=False:
        print 'correcting HI/H to account for the presence of H2...'
        #compute the pression of the gas particles
        #h^2Msun/kpc^3
        rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10
        U   = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
        P   = (gamma-1.0)*U*rho*(1.0+redshift)**3 #h^2 Msun/kpc^3*(km/s)^2
        P   = h**2*Msun/kpc**3*P                  #gr/cm^3*(km/s)^2
        P  /= kB                                 #K/cm^3
        del rho,U
 
        #assign H2 only to star forming particles
        SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0)
        indexes = np.where(SFR>0.0)[0]; del SFR

        #compute the H2/HI fraction
        R_surf = np.zeros(Nall[0],dtype=np.float32)
        if correct_H2   == 'BR':       R_surf = (P/3.5e4)**0.92
        elif correct_H2 == 'THINGS':   R_surf = (P/1.7e4)**0.8
        #R_surf[IDs]=1.0/(1.0+(35.0*(0.1/nH/0.76)**(gamma))**0.92)
        else:
            print 'bad choice of correct_H2!!!'; sys.exit()

        #compute the corrected HI mass taking into account the H2
        nH0[indexes]=nH0[indexes]/(1.0+R_surf[indexes]); del indexes,R_surf
        #M_HI[indexes]=M_HI[indexes]*(1.0-R_surf[indexes]); del indexes,R_surf

    return nH0
##############################################################################

##############################################################################
#This routine computes and returns the metallicity of each gas particle,
#in units of the solar metallicity
def gas_metallicity(snapshot_fname):

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    print 'finding the metallicity of the gas particles'
    head  = readsnap.snapshot_header(snapshot_fname)
    Nall  = head.nall
    files = head.filenum

    # read the masses of the gas particles in 1e10 Msun/h
    mass = readsnap.read_block(snapshot_fname,"MASS",parttype=0)

    # define the metallicity array
    metallicity = np.zeros(Nall[0],dtype=np.float64)

    offset = 0
    for filenum in xrange(files):
        if files==1:  fname      = snapshot_fname
        else:         fname      = snapshot_fname+'.'+str(filenum)
        
        npart = readsnap.snapshot_header(fname).npart  #particles in that file

        block_found = False;  EoF = False;  f = open(fname,'rb')
        f.seek(0,2); last_position = f.tell(); f.seek(0,0)
        while (not(block_found) or not(EoF)):
        
            # read the three first elements and the blocksize
            format_type = np.fromfile(f,dtype=np.int32,count=1)[0]
            block_name  = f.read(4)
            dummy       = np.fromfile(f,dtype=np.float64,count=1)[0] 
            blocksize1  = np.fromfile(f,dtype=np.int32,count=1)[0]
            
            if block_name=='Zs  ':
                Z = np.fromfile(f,dtype=np.float32,count=npart[0]*15)
                Z = np.reshape(Z,(npart[0],15))
            
                metal = np.sum(Z[:,1:],axis=1,dtype=np.float64)\
                    /mass[offset:offset+npart[0]]/0.02
                metallicity[offset:offset+npart[0]] = metal;  offset += npart[0]

                f.seek(npart[4]*15*4,1);  block_found = True

            else:   f.seek(blocksize1,1)

            # read the size of the block and check that is the same number
            blocksize2 = np.fromfile(f,dtype=np.int32,count=1)[0]

            if blocksize2!=blocksize1:
                print 'error!!!'; sys.exit()
        
            current_position = f.tell()
            if current_position == last_position:  EoF = True
        f.close()

    if offset!=Nall[0]:
        print 'Not all files read!!!'; sys.exit()
        
    return metallicity
##############################################################################

##############################################################################
#This routine computes the H2 fraction in the gas particles. Only H2 is assigned
#to star-forming particles. We use the KMT formalism, see also Dave et al. 2013
def H2_fraction(snapshot_fname):

    # read snapshot header
    print '\nREADING SNAPSHOTS PROPERTIES'
    head     = readsnap.snapshot_header(snapshot_fname)
    Nall     = head.nall
    redshift = head.redshift
    h        = head.hubble

    # find the metallicity of the gas particles
    Z = gas_metallicity(snapshot_fname)

    Msun    = 1.99e33   #g
    Mpc     = 3.0857e24 #cm
    sigma_d = Z*1e-21   #cm^2
    mu_H    = 2.3e-24   #g

    # read the density of the gas particles in h^2 Msun/Mpc^3 in proper units
    rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e19
    rho = rho*h**2*(1.0+redshift)**3  #Msun/Mpc^3
 
    # read the SPH radii of the gas particles in Mpc/h in proper units
    R = readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3
    R = (R/h)/(1.0+redshift)          #Mpc

    # compute the gas surface density in g/cm^2
    sigma = (rho*R)*(Msun/Mpc**2)     #g/cm^2
    del rho, R

    # read the star formation rate. Only star-forming particles will host H2
    SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0)
    indexes = np.where((SFR>0.0) & (Z>0))[0]
    
    #define the molecular hydrogen fraction array
    f_H2 = np.zeros(Nall[0],dtype=np.float64)

    # compute the value of tau_c, chi and s of the star-forming particles
    tau_c = sigma[indexes]*(sigma_d[indexes]/mu_H)
    chi   = 0.756*(1.0 + 3.1*Z[indexes]**0.365)
    s     = np.log(1.0 + 0.6*chi + 0.01*chi**2)/(0.6*tau_c)

    f_H2[indexes] = 1.0 - 0.75*s/(1.0 + 0.25*s);  del indexes, tau_c, chi, s
    f_H2[np.where(f_H2<0.0)[0]] = 0.0 #avoid negative values in the H2_fraction

    return f_H2
##############################################################################
