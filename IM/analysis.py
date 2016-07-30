# This code computes the radial P(k) from the 21cm maps, store the results
# and also computes the mean and variance of the results among different 
# realizations

from mpi4py import MPI
import numpy as np
import sys,os,gc
import IM_library as IML


###### MPI DEFINITIONS ######                                         
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

################################## INPUT #####################################
nuTable = '../nuTable.txt'
f_3D_Pk = '../pk_eh_planck.txt'

# cosmological parameters
Omega_m = 0.315
Omega_L = 0.685
h       = 0.67

# 1-172(z=2.5), 173-344(z=1.6), 345-516(z=1.0), 517-688(z=0.55)
bin_mins     = [1,   173, 345, 517]
bin_maxs     = [172, 344, 516, 688]
realizations = 100

# survey characteristics
D            = 15.0    #radio-telescope diameter in meters
T_instrument = 25.0    #K
time_total   = 10000.0 #hours
fsky         = 0.5     #sky fraction to be sampled
n_dish       = 200     #number of antennae
nside        = 128   
#sub_pixels   = [4**4]  #[4**3,4**4,4**5]

# folders roots where results are stored
folders_name = 'results/'  #['results/', 'results/nw_']

#################### fit parameters ####################
# 3D power spectra with and without wiggles for the theoretical template
f_3D_Pkw  = '../simulations/run_pkEH_s1001/pk.txt'
f_3D_Pknw = '../simulations/run_pkEH_smooth_s1001/pk.txt'
k_bins    = 50   #number of k-bins in [kmin,kmax] for precomputed Pkw, Pknw
R_bins    = 100  #number of Pkw, Pknw between [Rmin,Rmax]

# k-range for the fit
models     = ['cosmo','noise','fg']    #'cosmo', 'noise', 'fg'
kmin, kmax = 0.02, 0.2  #h/Mpc

# MCMC parameters
nwalkers  = 100    #number of MCMC chains
chain_pts = 10000  #number of points in each MCMC chain
##############################################################################

# find the number of the realizations carried out by the cpu
numbers = np.where((np.arange(realizations)%nprocs)==myrank)[0]+1
"""
# do a loop over all realizations
for i in numbers:

    num = str(i).zfill(3)
    
    # define name of output folder and create it if does not exist
    f_out = folders_name + str(i)
    if not(os.path.exists(f_out)):  os.system('mkdir '+f_out)
    
    # root of the maps containing the cosmo, cosmo+noise and cosmo+fg+noise
    root_map = ['maps/' + num + '/cosmo_',
                'maps/' + num + '/cosmo+noise1_',
                'maps/' + num + '/cosmo+noise2_',
                'maps/' + num + '/map1_clean_',
                'maps/' + num + '/map2_clean_']

    # define the masks 
    fmask = ['/data/villa/Alonso/mask/no_mask_128.fits',
             '/data/villa/Alonso/mask/no_mask_128.fits']

    # do a loop over the different redshift bins
    for bin_min,bin_max in zip(bin_mins,bin_maxs):

        IML.clustering_1D_all(root_map,nuTable,D,Omega_m,Omega_L,h,
                              nside,bin_min,bin_max,T_instrument,
                              time_total,fsky,n_dish,fmask,f_out)

                #IML.clustering_1D(root_map,nuTable,D,Omega_m,Omega_L,h,
                #                  n_side_map,subpixels,bin_min,bin_max,
                #                  T_instrument,time_total,fsky,n_dish,
                #                  fmask,do_JK,folder_name)

comm.Barrier()


# master compute the mean and variance of the results
if myrank==0:

    # do a loop over the different redshift bins
    for bin_min,bin_max in zip(bin_mins,bin_maxs):
            
        # compute mean and variance of the different realizations
        IML.mean_variance_Pk(f_3D_Pk,nuTable,nside,bin_min,bin_max,
                             D,time_total,realizations,Omega_m,Omega_L,
                             h,folders_name,do_nw=False,do_fg=True)
"""

for model in models:
    for bin_min,bin_max in zip(bin_mins,bin_maxs):
        IML.fit_function(nuTable,bin_min,bin_max,model,kmin,kmax,k_bins,R_bins,
                         D,time_total,nside,Omega_m,Omega_L,h,f_3D_Pkw,
                         f_3D_Pknw,nwalkers,chain_pts,realizations)
        gc.collect()

                                     
                                     
