# This script reads an input CF or Pk file (from CAMB for instance) and computes
# the CF or Pk in the same bins and using the same modes as those measured from
# the simulations. This is to provide a more fair comparison between theory and
# results from simulations. It is relatively important for bins with few modes
import numpy as np
import Pk_library as PKL
#import Power_spectrum_library as PSL
import sys,os


################################## INPUT ####################################
obj_type    = 'Pk'  #choose among 'CF' or 'Pk'

##################### CF ####################
#input_file = '../correlation_function/CF_m_z=10.dat'
#f_out      = 'expected_CF.txt'
#############################################

##################### Pk ####################
input_files = ['CAMB_TABLES/ics_matterpow_0.dat',
               'CAMB_TABLES/ics_matterpow_0.5.dat',
               'CAMB_TABLES/ics_matterpow_1.dat',
               'CAMB_TABLES/ics_matterpow_2.dat',
               'CAMB_TABLES/ics_matterpow_5.dat',
               'CAMB_TABLES/ics_matterpow_9.dat',
               'CAMB_TABLES/ics_matterpow_20.dat',
               'CAMB_TABLES/ics_matterpow_99.dat']

f_outs      = ['expected_Pk_z=0.txt',
               'expected_Pk_z=0.5.txt',
               'expected_Pk_z=1.txt',
               'expected_Pk_z=2.txt',
               'expected_Pk_z=5.txt',
               'expected_Pk_z=9.txt',
               'expected_Pk_z=20.txt',
               'expected_Pk_z=99.txt']
#############################################

BoxSize = 1000.0 #Mpc/h
dims    = 1024
#############################################################################


if obj_type=='Pk':

    # do a loop over the different input files
    for input_file, f_out in zip(input_files, f_outs):

        # read input file
        k_in, Pk_in = np.loadtxt(input_file,unpack=True)

        # compute expected Pk
        k, Pk, Nmodes = PKL.expected(k_in, Pk_in, BoxSize, dims)

        # save results to file ignoring DC mode
        np.savetxt(f_out, np.transpose([k,Pk,Nmodes]))


# WARNING!!!! This is still the old version
# UPDATE!!!!!
elif obj_type=='CF':

    # compute the value of |k| in each cell of the grid
    [array,k] = PSL.CIC_correction(dims);  del array

    bins_CF = dims/2+1

    # compute the value of r in each point of the grid 
    d_grid = k*BoxSize/dims;  del k;  
    print np.min(d_grid),'< d <',np.max(d_grid)

    # define the bins in r and the value of r in them
    distances     = np.linspace(0.0, BoxSize/2.0, bins_CF)
    distances_bin = 0.5*(distances[:-1]+distances[1:])

    # compute the number of modes in each bin
    modes = np.histogram(d_grid, bins=distances)[0];  del d_grid

    # do a loop over the different input files
    for input_file,f_out in zip(input_files,f_outs):
    
        # read input file
        r_input,xi_input = np.loadtxt(input_file,unpack=True)

        # compute xi(r) in each point of the grid
        xi_delta = np.interp(d_grid, r_input, xi_input)

        # compute the value of the correlation function
        xi = np.histogram(d_grid, bins=distances, weights=xi_delta)[0]
        xi /= modes

        # save results to file
        np.savetxt(f_out,np.transpose([distances_bin,xi]))

else:
    print 'bad object type choice!!!';  sys.exit()

