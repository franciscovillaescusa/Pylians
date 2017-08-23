import numpy as np 
import time,sys,os,h5py
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,log10,abs,exp,log
import readsnap, groupcat
import units_library as UL
import MAS_library as MASL


################################# UNITS #####################################
cdef double rho_crit = (UL.units()).rho_crit 

cdef double yr    = 3.15576e7   #seconds
cdef double km    = 1e5         #cm
cdef double Mpc   = 3.0856e24   #cm
cdef double kpc   = 3.0856e21   #cm
cdef double Msun  = 1.989e33    #g
cdef double Ymass = 0.24        #helium mass fraction
cdef double mH    = 1.6726e-24  #proton mass in grams
cdef double gamma = 5.0/3.0     #ideal gas
cdef double kB    = 1.3806e-26  #gr (km/s)^2 K^{-1}
cdef double nu0   = 1420.0      #21-cm frequency in MHz
cdef double muH   = 2.3e-24     #gr
cdef double pi    = np.pi 
#############################################################################

# This routine computes the self-shielding parameters of Rahmati et al.
# and the amplitude of the UV background at a given redshift together with
def Rahmati_parameters(redshift, TREECOOL_file, Gamma_UVB=None, fac=1.0,
                       verbose=False):

    # Rahmati et. al. 2013 self-shielding parameters (table A1)
    z_t       = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00])
    n0_t      = np.array([-2.94,-2.29,-2.06,-2.13,-2.23,-2.35]); n0_t=10**n0_t
    alpha_1_t = np.array([-3.98,-2.94,-2.22,-1.99,-2.05,-2.63])
    alpha_2_t = np.array([-1.09,-0.90,-1.09,-0.88,-0.75,-0.57])
    beta_t    = np.array([1.29, 1.21, 1.75, 1.72, 1.93, 1.77])
    f_t       = np.array([0.01, 0.03, 0.03, 0.04, 0.02, 0.01])

    # compute the self-shielding parameters at the redshift of the N-body
    n0      = np.interp(redshift, z_t, n0_t)
    alpha_1 = np.interp(redshift, z_t, alpha_1_t)
    alpha_2 = np.interp(redshift, z_t, alpha_2_t)
    beta    = np.interp(redshift, z_t, beta_t)
    f       = np.interp(redshift, z_t, f_t)
    if verbose:
        print 'n0 = %e\nalpha_1 = %2.3f\nalpha_2 = %2.3f\nbeta = %2.3f\n'\
            %(n0,alpha_1,alpha_2,beta) + 'f = %2.3f'%f
        
    # find the value of the photoionization rate
    if Gamma_UVB is None:
        data = np.loadtxt(TREECOOL_file); logz=data[:,0]; Gamma_UVB=data[:,1]
        Gamma_UVB=np.interp(np.log10(1.0+redshift),logz,Gamma_UVB); del data
        if verbose:  print 'Gamma_UVB(z=%2.2f) = %e s^{-1}'%(redshift,Gamma_UVB)
                     
    # Correct to reproduce the Lya forest mean flux
    Gamma_UVB /= fac

    return n0, alpha_1, alpha_2, beta, f, Gamma_UVB




# This routine computes the HI/H fraction of star-forming particles in the 
# Illustris-TNG simulation
# rho ---------------> star-forming part. densities in h^2 Msun/Mpc^3 (comoving)
# SPH ---------------> star-forming part radii in Mpc/h (comoving)
# metals ------------> star-forming part metallicities in solar units
# TREECOOL_file -----> file containing the UV strength
# Gamma -------------> value of the UVB if TREECOOL_file not provided
# fac ---------------> correction to the UVB to reproduce <F> of the Lya forest
# correct_H2 --------> whether to correct for H2
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def Rahmati_HI_Illustris(np.float32_t[:] rho, np.float32_t[:] SPH, 
                         np.float32_t[:] metals, redshift, h,
                         TREECOOL_file, Gamma=None, fac=1.0, 
                         correct_H2=False, verbose=False):
                         
    cdef long i, particles
    cdef float prefact, prefact2, nH, f_H2
    cdef double n0, alpha_1, alpha_2, beta, f, Gamma_UVB, chi, tau_c
    cdef double T, Lambda, alpha_A, Lambda_T, Gamma_phot, A, B, C, s
    cdef double sigma
    cdef np.float32_t[:] nH0

    # find the values of the self-shielding parameters and the UV background
    n0, alpha_1, alpha_2, beta, f, Gamma_UVB = \
        Rahmati_parameters(redshift, TREECOOL_file, Gamma, fac, verbose)
                      
    # find the number of star-forming particles and set their temperatures
    particles = rho.shape[0]
    T         = 1e4 #K

    # compute the case A recombination rate
    Lambda  = 315614.0/T
    alpha_A = 1.269e-13*pow(Lambda,1.503)\
        /pow(1.0 + pow((Lambda/0.522),0.47),1.923) #cm^3/s

    # Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
    Lambda_T = 1.17e-10*sqrt(T)*exp(-157809.0/T)/(1.0+sqrt(T/1e5)) #cm^3/s

    # prefactor to change densities from h^2 Msun/Mpc^3 to atoms/cm^{-3}
    prefact  = 0.76*h**2*Msun/Mpc**3/mH*(1.0+redshift)**3                   

    # prefactor to change surface densities from h Msun/Mpc^2 to g/cm^2
    prefact2 = (1.0+redshift)**2*h*Msun/Mpc**2 

    # define HI/H fraction
    nH0 = np.zeros(particles, dtype=np.float32)

    # do a loop over all particles
    for i in xrange(particles):

        # compute particle density in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
        nH = rho[i]*prefact
        
        # compute the photoionization rate
        Gamma_phot = Gamma_UVB
        Gamma_phot *= ((1.0 - f)*pow(1.0 + pow(nH/n0,beta), alpha_1) +\
                           f*pow(1.0 + nH/n0, alpha_2))

        # compute the coeficients A,B and C to calculate the HI/H fraction
        A = alpha_A + Lambda_T
        B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
        C = alpha_A

        # compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
        nH0[i] = (B-sqrt(B*B-4.0*A*C))/(2.0*A)

        ##### correct for H2 using KMT #####
        if correct_H2:
            chi   = 0.756*(1.0 + 3.1*metals[i]**0.365)           #dimensionless
            sigma = rho[i]*SPH[i]*prefact2                       #g/cm^2
            tau_c = sigma*(metals[i]*1e-21)/muH                  #dimensionless 
            s     = log(1.0 + 0.6*chi + 0.01*chi**2)/(0.6*tau_c) #dimensionless
            if s<2.0:  f_H2 = 1.0 - 0.75*s/(1.0 + 0.25*s)
            else:      f_H2 = 0.0
            if f_H2<0.0:  f_H2 = 0.0
            nH0[i] = nH0[i]*(1.0-f_H2)
            
    return np.asarray(nH0)

#########################################################################    
# This routine identifies the offsets for halos and galaxies
# and perform several checks
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef test(f_offset, snapshot_root, num):

    cdef int i, j, Number
    cdef long offset, offset_halo, particles
    cdef np.int64_t[:] offset_galaxies, 
    cdef np.int64_t[:] end_halos, end_all_galaxies
    cdef np.int64_t[:] offset_halos, offset_subhalos
    cdef np.int32_t[:] lens_halos, subhalos_num, lens_subhalos

    # read halos and subhalos offset
    f = h5py.File(f_offset, "r")
    offset_halos    = f['Group/SnapByType'][:,0]
    offset_subhalos = f['Subhalo/SnapByType'][:,0]
    f.close()

    # read number of particles in halos and subhalos and number of subhalos
    halos = groupcat.loadHalos(snapshot_root, num, 
                               fields=['GroupLenType','GroupNsubs'])
    lens_halos   = halos['GroupLenType'][:,0]  
    subhalos_num = halos['GroupNsubs']
    subhalos = groupcat.loadSubhalos(snapshot_root, num, 
                                     fields=['SubhaloLenType','SubhaloMass'])
    lens_subhalos = subhalos['SubhaloLenType'][:,0]

    # define the array containing the beginning and ends of the halos
    end_halos = np.zeros(lens_halos.shape[0], dtype=np.int64)

    # find the offsets of the halos
    particles = 0
    for i in xrange(lens_halos.shape[0]):
        if offset_halos[i]!=particles:
            raise Exception('Offset are wrong!!')
        particles += lens_halos[i]
        end_halos[i] = particles
    del offset_halos


    # define the array hosting the offset of galaxies
    end_all_galaxies = np.zeros(lens_halos.shape[0], dtype=np.int64)
    offset_galaxies  = np.zeros(lens_subhalos.shape[0], dtype=np.int64)

    # do a loop over all halos
    Number, offset, offset_halo = 0, 0, 0
    for i in xrange(lens_halos.shape[0]):

        offset = offset_halo
        end_all_galaxies[i] = offset

        # do a loop over all galaxies belonging to the halo
        for j in xrange(subhalos_num[i]):
            offset_galaxies[Number] = offset
            end_all_galaxies[i] += lens_subhalos[Number]
            offset += lens_subhalos[Number];  Number += 1

        if (end_all_galaxies[i]-offset_halo)>lens_halos[i]:
            raise Exception('More particles in galaxies than in halo!!!')

        offset_halo += lens_halos[i]

    # check that galaxy offsets are the same
    for i in xrange(offset_galaxies.shape[0]):
        if offset_galaxies[i]!=offset_subhalos[i]:
            raise Exception('Offset of subhalos are different!!!')

    return np.asarray(end_halos),np.asarray(end_all_galaxies)
#########################################################################    

# This routine computes the HI 
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef M_HI_counter(np.float32_t[:] NHI, np.float32_t[:] mass, 
                   np.float64_t[:] M_HI, np.float64_t[:] M_HI_gal, 
                   np.int64_t[:] end_halos, np.int64_t[:] end_all_galaxies, 
                   long Number, long start, long end, long end_gal,
                   long halo_num, done):

    cdef long j, particles, num_halos
    
    # find the number of particles to iterate over
    particles = NHI.shape[0]
    num_halos = M_HI.shape[0]

    # do a loop over all particles
    for j in xrange(particles):
        if Number>end:
            halo_num += 1
            if halo_num<num_halos:  
                start   = end
                end     = end_halos[halo_num]
                end_gal = end_all_galaxies[halo_num]
            else:
                done = True;  break

        # if particle is in galaxy add to M_HI_gal
        if Number>=start and Number<end_gal:
            M_HI_gal[halo_num] += 0.76*NHI[j]*mass[j]
                    
        M_HI[halo_num] += 0.76*NHI[j]*mass[j]
        Number += 1

    return Number, start, end, end_gal, halo_num, done
#######################################################################

#######################################################################
# This routine computes the HI mass within halos and the HI mass within
# halos that it is in galaxies
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef M_HI_halos_galaxies(np.int64_t[:] pars, done,
                          np.int32_t[:] halo_len, np.int32_t[:] gal_len,
                          np.int32_t[:] gal_in_halo,
                          np.float32_t[:] mass_ratio, np.float32_t[:] MHI,
                          np.float64_t[:] M_HI, np.float64_t[:] M_HI_gal):

    cdef long Number            = pars[0]
    cdef long start_h           = pars[1]
    cdef long end_h             = pars[2]
    cdef long start_g           = pars[3]
    cdef long end_g             = pars[4]
    cdef long halo_num          = pars[5]
    cdef long gal_num           = pars[6]
    cdef long gal_in_halo_local = pars[7]
    cdef long i, particles, num_halos, num_galaxies
    
    # find the number of particles to iterate over
    particles    = MHI.shape[0]
    num_halos    = M_HI.shape[0]
    num_galaxies = gal_len.shape[0]

    # do a loop over all particles
    for i in xrange(particles):

        # if particle belongs to new halo change variables
        if Number>=end_h:

            # accout for missing galaxies (should be galaxies with 0 particles)
            if Number>=end_g and gal_in_halo_local<gal_in_halo[halo_num]-1:
                gal_num += 1
                gal_in_halo_local += 1
                while gal_len[gal_num]==0 and \
                        gal_in_halo_local<gal_in_halo[halo_num]-1:
                    gal_num += 1
                    gal_in_halo_local += 1
            
            #if gal_num!=np.sum(gal_in_halo[:halo_num+1], dtype=np.int64)-1:
            #    print 'Numbers differ!!!!'
            #    print halo_num
            #    print gal_num
            #    print np.sum(gal_in_halo[:halo_num+1], dtype=np.int64)
                
            # update halo variables
            halo_num += 1
            if halo_num<num_halos:  
                start_h = end_h
                end_h   = end_h + halo_len[halo_num]
            else:
                # check that all galaxies have been counted
                if gal_num!=np.sum(gal_in_halo, dtype=np.int64)-1:
                    print 'gal_num  = %ld'%gal_num
                    print 'galaxies = %ld'%(np.sum(gal_in_halo, dtype=np.int64)-1)
                    raise Exception("Finished without counting all galaxies")
                done = True;  break

            # restart galaxy variables
            if gal_num<num_galaxies and gal_in_halo[halo_num]>0:
                gal_in_halo_local = 0
                gal_num += 1
                start_g = start_h
                end_g   = start_h + gal_len[gal_num]

        # if particle belongs to new galaxy change variables
        if Number>=end_g and gal_in_halo_local<gal_in_halo[halo_num]-1:
            gal_num += 1
            gal_in_halo_local += 1
            while gal_len[gal_num]==0 and \
                    gal_in_halo_local<gal_in_halo[halo_num]-1:
                gal_num += 1
                gal_in_halo_local += 1
            start_g = end_g
            end_g   = end_g + gal_len[gal_num]

        # if particle is inside galaxy add it to M_HI_gal
        if Number>=start_g and Number<end_g and mass_ratio[gal_num]>0.1:
            M_HI_gal[halo_num] += MHI[i]

        # for halos always add the HI mass
        M_HI[halo_num] += MHI[i]
        Number += 1

    pars[0], pars[1], pars[2]          = Number, start_h, end_h 
    pars[3], pars[4], pars[5], pars[6] = start_g, end_g, halo_num, gal_num
    pars[7]                            = gal_in_halo_local

    return done
    
###############################################################
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef HI_image(np.float32_t[:,:] pos, float x_min, float x_max, 
               float y_min, float y_max, float z_min, float z_max):

    start = time.time()
    cdef long i, particles
    cdef list indexes_list

    indexes_list = []
    particles = pos.shape[0]
    for i in xrange(particles):

        if pos[i,0]>=x_min and pos[i,0]<=x_max:
            if pos[i,1]>=y_min and pos[i,1]<=y_max:
                if pos[i,2]>=z_min and pos[i,2]<=z_max:
                    indexes_list.append(i)
                    
    print 'Time taken = %.2f seconds'%(time.time()-start)
    return indexes_list


# This routine implements reads Gadget gas output and correct HI/H fractions
# to account for self-shielding and H2
# snapshot_fname -------> name of the N-body snapshot
# fac ------------------> factor to reproduce the mean Lya flux
# TREECOOL_file --------> TREECOOL file used in the N-body
# Gamma_UVB ------------> value of the UVB photoionization rate
# correct_H2 -----------> correct the HI/H fraction to account for H2
# if Gamma_UVB is set to None the value of the photoionization rate will be read
# from the TREECOOL file, otherwise it is used the Gamma_UVB value
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def Rahmati_HI_assignment(snapshot_fname, fac, TREECOOL_file, Gamma_UVB=None,
                          correct_H2=False, IDs=None, verbose=False):

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    print '\nREADING SNAPSHOT PROPERTIES'
    head     = readsnap.snapshot_header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h
    Nall     = head.nall
    redshift = head.redshift

    # find the values of the self-shielding parameters and the UV background
    n0, alpha_1, alpha_2, beta, f, Gamma_UVB = \
        Rahmati_parameters(redshift, TREECOOL_file, Gamma_UVB, fac, verbose)

    #compute the HI/H fraction 
    nH0 = HI_from_UVB(snapshot_fname, Gamma_UVB, True, correct_H2,
                      f, n0, alpha_1, alpha_2, beta, SF_temperature=1e4)
    
    #read the gas masses
    mass = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h

    #create the array M_HI and fill it
    M_HI = 0.76*mass*nH0;  del nH0,mass
    print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)

    return M_HI



#This function compute the HI/H fraction given the photoionization rate and 
#the density of the gas particles
#snapshot_fname ------------> name of the N-body snapshot
#Gamma_UVB -----------------> value of the UVB photoionization rate
#self_shielding_correction -> apply (True) or not (False) the self-shielding
#correct_H2 ----------------> correct the HI masses to account for H2
#f -------------------------> parameter of the Rahmati et al model (see eq. A1)
#n0 ------------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_1 -------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_2 -------------------> parameter of the Rahmati et al model (see eq. A1)
#beta ----------------------> parameter of the Rahmati et al model (see eq. A1)
#SF_temperature ------------> associate temperature of star forming particles
#If SF_temperature is set to None it will compute the temperature of the SF
#particles from their density and internal energy
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def HI_from_UVB(snapshot_fname, double Gamma_UVB,
                self_shielding_correction=False,correct_H2=False,
                double f=0.0, double n0=0.0, double alpha_1=0.0,
                double alpha_2=0.0, double beta=0.0,
                double SF_temperature=0.0):

    cdef long gas_part,i 
    cdef double mean_mol_weight, yhelium, A, B, C, R_surf, P, prefact2
    cdef double nH, Lambda, alpha_A, Lambda_T, Gamma_phot, prefact, T
    cdef np.float32_t[:] rho,U,ne,SFR,nH0

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
    gas_part = Nall[0]

    # read density, internal energy, electron fraction and star formation rate
    # density units: h^2 Msun / Mpc^3
    yhelium = (1.0-0.76)/(4.0*0.76) 
    U   = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
    ne  = readsnap.read_block(snapshot_fname,"NE  ",parttype=0) #electron frac
    rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10/1e-9 
    SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR

    # define HI/H fraction
    nH0 = np.zeros(gas_part, dtype=np.float32)

    #### Now compute the HI/H fraction following Rahmati et. al. 2013 ####
    prefact  = 0.76*h**2*Msun/Mpc**3/mH*(1.0+redshift)**3 
    prefact2 = (gamma-1.0)*(1.0+redshift)**3*h**2*Msun/kpc**3/kB 

    print 'doing loop...';  start = time.clock()    
    for i in xrange(gas_part):

        # compute particle density in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
        nH = rho[i]*prefact

        # compute particle temperature
        if SFR[i]>0.0:
            T = SF_temperature
            P = prefact2*U[i]*rho[i]*1e-9  #K/cm^3
        else:
            mean_mol_weight = (1.0+4.0*yhelium)/(1.0+yhelium+ne[i])
            T = U[i]*(gamma-1.0)*mH*mean_mol_weight/kB
        
        #compute the case A recombination rate
        Lambda  = 315614.0/T
        alpha_A = 1.269e-13*pow(Lambda,1.503)\
                  /pow(1.0 + pow((Lambda/0.522),0.47),1.923) #cm^3/s

        #Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
        Lambda_T = 1.17e-10*sqrt(T)*\
                   exp(-157809.0/T)/(1.0+sqrt(T/1e5)) #cm^3/s

        #compute the photoionization rate
        Gamma_phot = Gamma_UVB
        if self_shielding_correction:
            Gamma_phot *= ((1.0 - f)*pow(1.0 + pow(nH/n0,beta), alpha_1) +\
                           f*pow(1.0 + nH/n0, alpha_2))

        #compute the coeficients A,B and C to calculate the HI/H fraction
        A = alpha_A + Lambda_T
        B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
        C = alpha_A

        #compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
        nH0[i] = (B-sqrt(B*B-4.0*A*C))/(2.0*A)

        # correct for the presence of H2
        if correct_H2 and SFR[i]>0.0:
            R_surf = (P/1.7e4)**0.8
            nH0[i] = nH0[i]/(1.0 + R_surf)

    print 'Time taken = %.3f s'%(time.clock()-start)
    return np.asarray(nH0)



