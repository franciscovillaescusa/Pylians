import numpy as np
import Power_spectrum_library as PSL
import sys,os


################################## INPUT ####################################
#input_file = '../correlation_function/CF_m_z=10.dat'
#obj_type   = 'CF'  #choose among 'CF' or 'Pk'
#f_out      = 'expected_CF.txt'

input_file = '../CAMB_TABLES/ics_matterpow_0.dat'
obj_type   = 'Pk'  #choose among 'CF' or 'Pk'
f_out      = 'expected_Pk_z=0.txt'

BoxSize = 2000.0 #Mpc/h
dims    = 512
#############################################################################

# compute the value of |k| in each cell of the grid
[array,k] = PSL.CIC_correction(dims);  del array

if obj_type=='CF':

    # read input file
    r_input,xi_input = np.loadtxt(input_file,unpack=True)

    bins_CF = dims/2+1

    # compute the value of r in each point of the grid 
    d_grid = k*BoxSize/dims;  del k
    print np.min(d_grid),'< d <',np.max(d_grid)

    # define the array with the bins in r
    distances = np.linspace(0.0,BoxSize/2.0,bins_CF)

    # compute xi(r) in each point of the grid
    xi_delta = np.interp(d_grid,r_input,xi_input)

    xi    = np.histogram(d_grid,bins=distances,weights=xi_delta)[0]
    modes = np.histogram(d_grid,bins=distances)[0];  del d_grid
    xi    /= modes

    distances_bin = 0.5*(distances[:-1]+distances[1:])

    # save results to file
    np.savetxt(f_out,np.transpose([distances_bin,xi]))

elif obj_type=='Pk':

    bins_r=int(np.sqrt(3*int(0.5*(dims+1))**2))+1
    
    # read input file
    k_input,Pk_input = np.loadtxt(input_file,unpack=True)

    # count modes
    count = PSL.lin_histogram(bins_r,0.0,bins_r*1.0,k)

    # compute the value of P(k) in each cell
    delta_k2 = np.interp(k,k_input*BoxSize/(2.0*np.pi),Pk_input)

    #compute the P(k)=<delta_k^2>
    Pk = PSL.lin_histogram(bins_r,0.0,bins_r*1.0,k,weights=delta_k2)
    Pk = Pk/count

    #final processing
    bins_k = np.linspace(0.0,bins_r,bins_r+1)
    #compute the bins in k-space and give them physical units (h/Mpc), (h/kpc)
    k=k.astype(np.float64) #to avoid problems with np.histogram
    k=2.0*np.pi/BoxSize*np.histogram(k,bins_k,weights=k)[0]/count

    #ignore the first bin (fundamental frequency)
    k=k[1:]; Pk=Pk[1:]

    #keep only with modes below 1.1*k_Nyquist
    k_N=np.pi*dims/BoxSize; indexes=np.where(k<1.1*k_N)
    k=k[indexes]; Pk=Pk[indexes]; del indexes

    # save results to file
    np.savetxt(f_out,np.transpose([k,Pk]))

else:
    print 'bad object type choice!!!';  sys.exit()

