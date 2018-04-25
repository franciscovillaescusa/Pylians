import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL
import HI_library as HIL
import redshift_space_library as RSL

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
################################ INPUT #######################################
#snapnums = np.array([99, 50, 33, 25, 21, 17])
#snapnums = np.array([17,21,25])
snapnums = np.array([33,25])

TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

dims = 2048
MAS  = 'CIC'
##############################################################################

# do a loop over the different redshifts
for snapnum in snapnums:

    for axis in [0,1,2]:

        # read header
        snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
        f            = h5py.File(snapshot+'.0.hdf5', 'r')
        scale_factor = f['Header'].attrs[u'Time']   
        redshift     = f['Header'].attrs[u'Redshift']
        BoxSize      = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
        filenum      = f['Header'].attrs[u'NumFilesPerSnapshot']
        Omega_m      = f['Header'].attrs[u'Omega0']
        Omega_L      = f['Header'].attrs[u'OmegaLambda']
        h            = f['Header'].attrs[u'HubbleParam']
        Masses       = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
        f.close()
        Hubble = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_L) #km/s/(Mpc/h)
        print 'Working with snapshot at z=%.0f and axis %d'%(redshift,axis)

        fout = 'fields_RS_%d_z=%.1f.hdf5'%(axis,redshift)
        if os.path.exists(fout):  continue

        # define the array hosting delta_HI and delta_m
        delta_m  = np.zeros((dims,dims,dims), dtype=np.float32)
        delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)

        # do a loop over all subfiles in a given snapshot
        MHI_total, M_total, start = 0.0, 0.0, time.time()
        for i in xrange(filenum):

            snapshot = '%s/output/snapdir_%03d/snap_%03d.%d.hdf5'\
                %(run,snapnum,snapnum,i)
            f = h5py.File(snapshot, 'r')

            # read pos, radii, densities, HI/H and masses of gas particles 
            pos  = (f['PartType0/Coordinates'][:]/1e3).astype(np.float32)
            vel  = f['PartType0/Velocities'][:]*np.sqrt(scale_factor)
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

            # move gas particles to redshift-space
            RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)

            ######## delta_HI ########
            MASL.MA(pos, delta_HI, BoxSize, MAS, MHI)


            ######### delta_m #########
            MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #gas
            M_total += np.sum(mass, dtype=np.float64)

            for ptype in [1,4,5]:

                prefix = 'PartType%d/'%ptype
                name1 = prefix+'Coordinates'
                name2 = prefix+'Velocities'
                name3 = prefix+'Masses'

                pos  = (f[name1][:]/1e3).astype(np.float32) #Mpc/h

                # move positions to redshift-space
                vel  = f[name2][:]*np.sqrt(scale_factor) #km/s
                RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, 
                                       redshift, axis)

                # read masses in Msun/h
                if ptype==1:
                    mass = np.ones(pos.shape[0], dtype=np.float32)*Masses[1]
                else:
                    mass = f[name3][:]*1e10  #Msun/h

                MASL.MA(pos, delta_m, BoxSize, MAS, mass)
                M_total += np.sum(mass, dtype=np.float64)

            f.close()

            print '%03d -----> Omega_HI = %.3e  Omega_m = %.4f  : %6.0f s'\
                %(i,MHI_total/(BoxSize**3*rho_crit),M_total/(BoxSize**3*rho_crit),
                  time.time()-start)


        f = h5py.File(fout,'w')
        f.create_dataset('delta_HI', data=delta_HI)
        f.create_dataset('delta_m',  data=delta_m)
        f.close()

