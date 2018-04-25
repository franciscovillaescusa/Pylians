from mpi4py import MPI
import numpy as np
import HI_library as HIL
import sys,os,h5py
import MAS_library as MASL
import units_library as UL


####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
################################ INPUT ########################################
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

snapnum = 17 #33 40 50 59 67 72 78 84 91 99

dims = 35000
###############################################################################


# find snapshot root
snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)

# read header
f = h5py.File(snapshot+'.0.hdf5', 'r')
redshift = f['Header'].attrs[u'Redshift']
BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
Omega_m  = f['Header'].attrs[u'Omega0']
Omega_L  = f['Header'].attrs[u'OmegaLambda']
h        = f['Header'].attrs[u'HubbleParam']
f.close()

# find the subfiles that each cpu will work over
numbers = np.where(np.arange(filenum)%nprocs==myrank)[0]

# define the array hosting the column densities
CD = np.zeros((dims, dims), dtype=np.float64)

# do a loop over all subfiles
for i in numbers:

    print 'Myrank = %02d ---> %03d'%(myrank,i)

    # find sub-snapshot and open it for reading
    snapshot = '%s/output/snapdir_%03d/snap_%03d.%d.hdf5'\
        %(run,snapnum,snapnum,i)
    f = h5py.File(snapshot, 'r')

    # read positions, radii, densities, HI/H and masses of gas particles 
    pos  = (f['PartType0/Coordinates'][:]/1e3).astype(np.float32)
    MHI  = f['PartType0/NeutralHydrogenAbundance'][:]
    mass = f['PartType0/Masses'][:]*1e10
    rho  = f['PartType0/Density'][:]*1e19
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
    MHI[indexes] = HIL.Rahmati_HI_Illustris(rho, radii_SFR, metals, redshift, 
                                            h, TREECOOL_file, Gamma=None,
                                            fac=1, correct_H2=True) #HI/H
    MHI *= (0.76*mass)

    # compute column density
    MASL.voronoi_RT_2D_periodic(CD, pos, MHI, radii, 0.0, 0.0, BoxSize)


# convert (Msun/h)/(Mpc/h)^2 to cm^{-2}
factor = h*(1.0+redshift)**2*\
    (UL.units().Msun_g)/(UL.units().mH_g)/(UL.units().Mpc_cm)**2
CD *= factor

# sum the contributions from the different cpus
if myrank==0:  CD_tot = np.zeros((dims,dims), dtype=np.float64)
else:          CD_tot = None
comm.Reduce(CD, CD_tot, op=MPI.SUM);  del CD

# compute CDDF and save results to file
if myrank==0:
    NHI = np.logspace(17, 24, 100)
    CDDF = np.histogram(CD_tot, bins=NHI)[0]
    dX = 100.0*(1.0+redshift)**2*BoxSize/3e5 #H0*(1+z)^2*comoving_distance/c
    dNHI = NHI[1:]-NHI[:-1]
    NHI_mean = 0.5*(NHI[1:] + NHI[:-1])
    CDDF = CDDF/(dX*dNHI*dims*dims)

    np.savetxt('CDDF_35000_z=%.3f.txt'%redshift, np.transpose([NHI_mean, CDDF]))





