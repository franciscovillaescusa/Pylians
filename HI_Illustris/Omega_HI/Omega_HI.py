# This script computes the value of Omega_HI from Illustris snapshot
# It returns a file with z, Omega_HI, Omega_HI2, Omega_HI3, Omega_g
# Omega_HI and Omega_gas are the standard quantities
# The values of Omega_HI2 and Omega_HI3 save the values of Omega_HI
# when HI/H is set to 0 and 1 for star formation particles, respectively
from mpi4py import MPI
import numpy as np
import units_library as UL
import HI_library as HIL
import sys,os,glob,h5py

####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
################################ INPUT ########################################
snapnums = np.array([13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99])
TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'

#run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'
                 
fout = 'Omega_HI_75_1820.txt'
###############################################################################

# find the numbers each cpu will work on
array   = np.arange(0, len(snapnums))
numbers = np.where(array%nprocs==myrank)[0]
numbers = snapnums[numbers]


# do a loop over the different realizations
if len(numbers)>0:  g = open(fout[:-4]+'_%d.txt'%myrank, 'w')
for snapnum in numbers:

    # read header
    snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
    f = h5py.File(snapshot+'.0.hdf5', 'r')
    redshift = f['Header'].attrs[u'Redshift']
    BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
    filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
    Omega_m  = f['Header'].attrs[u'Omega0']
    Omega_L  = f['Header'].attrs[u'OmegaLambda']
    h        = f['Header'].attrs[u'HubbleParam']
    f.close()

    if myrank==0:
        print '\n'
        print 'BoxSize         = %.3f Mpc/h'%BoxSize
        print 'Number of files = %d'%filenum
        print 'Omega_m         = %.3f'%Omega_m
        print 'Omega_l         = %.3f'%Omega_L
        print 'redshift        = %.3f'%redshift


    Omega_HI, Omega_HI2, Omega_HI3, Omega_g = 0.0, 0.0, 0.0, 0.0
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
        f.close()

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

        Omega_HI += np.sum(MHI,  dtype=np.float64)
        Omega_g  += np.sum(mass, dtype=np.float64)

        if len(indexes)>0:
            MHI[indexes] = 0.0
            Omega_HI2 += np.sum(MHI, dtype=np.float64)

            MHI[indexes] = 0.76*mass[indexes]
            Omega_HI3 += np.sum(MHI, dtype=np.float64)

        print '\nz = %.3f ---> i = %03d'%(redshift,i)
        print 'Omega_HI2 = %.6e'%(Omega_HI2/(BoxSize**3*rho_crit))
        print 'Omega_HI  = %.6e'%(Omega_HI/(BoxSize**3*rho_crit))
        print 'Omega_HI3 = %.6e'%(Omega_HI3/(BoxSize**3*rho_crit))
        print 'Omega_g   = %.6e'%(Omega_g/(BoxSize**3*rho_crit))

    Omega_HI  = Omega_HI/(BoxSize**3*rho_crit)
    Omega_HI2 = Omega_HI2/(BoxSize**3*rho_crit)
    Omega_HI3 = Omega_HI3/(BoxSize**3*rho_crit)
    Omega_g   = Omega_g/(BoxSize**3*rho_crit)

    g.write('%.3f %.6e %.6e %.6e %.6e\n'\
                %(redshift, Omega_HI, Omega_HI2, Omega_HI3, Omega_g))
g.close()

comm.Barrier()



# joint partial files into a single file
if myrank==0:

    # read partial files and create big array with them
    files = glob.glob(fout[:-4]+'_*');  data = []
    for fin in files:
        for data_in in np.loadtxt(fin,unpack=False):
            data.append(data_in)
    data = np.array(data)

    # sort big array by redshift (column 0) and save results
    np.savetxt(fout, data[data[:,0].argsort()], fmt="%.6e")


    
