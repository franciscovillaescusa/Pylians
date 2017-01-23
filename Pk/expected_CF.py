# This script reads an input CF or Pk file (from CAMB for instance) and computes
# the CF or Pk in the same bins and using the same modes as those measured from
# the simulations. This is to provide a more fair comparison between theory and
# results from simulations. It is relatively important for bins with few modes
import numpy as np
import Power_spectrum_library as PSL
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

# compute the value of |k| in each cell of the grid
[array,k] = PSL.CIC_correction(dims);  del array

if obj_type=='Pk':

    # here k is in dimensionless units: k=0,1,sqrt(2),sqrt(3),2....
    bins_r = int(np.sqrt(3*int(0.5*(dims+1))**2))+1
    k_N = np.pi*dims/BoxSize  #Nyquist frequency    

    # count modes
    count = PSL.lin_histogram(bins_r, 0.0, bins_r*1.0, k)

    # define k-binning and value of k in it and give physical units
    bins_k = np.linspace(0.0, bins_r, bins_r+1)
    k = k.astype(np.float64) #to avoid problems with np.histogram
    k_bin = np.histogram(k,bins_k,weights=k)[0]/count  #value of k in the k-bin
    k_bin = k_bin*2.0*np.pi/BoxSize

    # do a loop over the different input files
    for input_file,f_out in zip(input_files,f_outs):

        # read input file
        k_input,Pk_input = np.loadtxt(input_file,unpack=True)

        # compute the value of P(k) in each cell
        delta_k2 = np.interp(k*2.0*np.pi/BoxSize, k_input, Pk_input)

        # compute the P(k)=<delta_k^2>
        Pk = PSL.lin_histogram(bins_r, 0.0, bins_r*1.0, k, weights=delta_k2)
        Pk = Pk/count
        
        # save results to file ignoring DC mode
        np.savetxt(f_out,np.transpose([k_bin[1:],Pk[1:]]))

elif obj_type=='CF':

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

