import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL
import HI_library as HIL

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
################################ INPUT ########################################
snapnums = np.array([99, 50, 33, 25, 21, 17])
#snapnums = np.array([25,21,17])
TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'

#run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

dims = 2048

MAS = 'NGP'
##############################################################################

# do a loop over the different redshifts
for snapnum in snapnums:

    # define the array hosting delta_HI and delta_m
    delta_m  = np.zeros((dims,dims,dims), dtype=np.float32)
    delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)

    # read header
    snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
    f = h5py.File(snapshot+'.0.hdf5', 'r')
    redshift = f['Header'].attrs[u'Redshift']
    BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
    filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
    Omega_m  = f['Header'].attrs[u'Omega0']
    Omega_L  = f['Header'].attrs[u'OmegaLambda']
    h        = f['Header'].attrs[u'HubbleParam']
    Masses   = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
    f.close()

    print 'Working with snapshot at redshift %.0f'%redshift

    # do a loop over all subfiles in a given snapshot
    MHI_total, M_total, start = 0.0, 0.0, time.time()
    for i in xrange(filenum):

        snapshot = '%s/output/snapdir_%03d/snap_%03d.%d.hdf5'\
            %(run,snapnum,snapnum,i)
        f = h5py.File(snapshot, 'r')

        # read positions, radii, densities, HI/H and masses of gas particles 
        pos  = (f['PartType0/Coordinates'][:]/1e3).astype(np.float32)
        MHI  = f['PartType0/NeutralHydrogenAbundance'][:]
        mass = f['PartType0/Masses'][:]*1e10  #Msun/h
        rho  = f['PartType0/Density'][:]*1e19 #(Msun/h)/(Mpc/h)^3
        SFR  = f['PartType0/StarFormationRate'][:]
        indexes = np.where(SFR>0.0)[0];  del SFR

        # find the metallicity of star-forming particles
        metals = f['PartType0/GFM_Metallicity'][:]
        metals = metals[indexes]/0.0127

        # find densities of star-forming particles: units of h^2 Msun/Mpc^3
        Volume = mass/rho                            #(Mpc/h)^3
        radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 
        rho    = rho[indexes]                        #h^2 Msun/Mpc^3
        Volume = Volume[indexes]                     #(Mpc/h)^3

        # find volume and radius of star-forming particles
        radii_SFR  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 
        
        # find HI/H fraction for star-forming particles
        MHI[indexes] = HIL.Rahmati_HI_Illustris(rho, radii_SFR, metals, 
                                                redshift, h, TREECOOL_file, 
                                                Gamma=None, fac=1, 
                                                correct_H2=True) #HI/H
        MHI *= (0.76*mass)
        MHI_total += np.sum(MHI, dtype=np.float64)

        ######## delta_HI ########
        MASL.MA(pos, delta_HI, BoxSize, MAS, MHI)


        ######### delta_m #########
        MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #gas
        M_total += np.sum(mass, dtype=np.float64)

        pos  = (f['PartType1/Coordinates'][:]/1e3).astype(np.float32)        
        mass = np.ones(pos.shape[0], dtype=np.float32)*Masses[1] #Msun/h
        MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #CDM
        M_total += np.sum(mass, dtype=np.float64)

        pos  = (f['PartType4/Coordinates'][:]/1e3).astype(np.float32)        
        mass = f['PartType4/Masses'][:]*1e10  #Msun/h
        MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #stars
        M_total += np.sum(mass, dtype=np.float64)

        pos  = (f['PartType5/Coordinates'][:]/1e3).astype(np.float32)        
        mass = f['PartType5/Masses'][:]*1e10  #Msun/h
        MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #stars
        M_total += np.sum(mass, dtype=np.float64)

        f.close()

        print '%03d -----> Omega_HI = %.3e  Omega_m = %.4f  : %6.0f s'\
            %(i,MHI_total/(BoxSize**3*rho_crit),M_total/(BoxSize**3*rho_crit),
              time.time()-start)


    f = h5py.File('fields_NGP_z=%.1f.hdf5'%redshift,'w')
    f.create_dataset('delta_HI', data=delta_HI)
    f.create_dataset('delta_m',  data=delta_m)
    f.close()
