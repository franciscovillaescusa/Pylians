import numpy as np
import sys,os,h5py
import transformer as TR
import smoothing_library as SL

################################### INPUT ######################################
threads = 28

BoxSize = 75.0 #Mpc/h
grid    = 2048

Filter  = 'Top-Hat'
R       = 1.0  #Mpc/h
################################################################################


numbers = np.random.randint(0, grid**3, 64**3)

W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

for z in [0,1,2,3,4,5]:

    f = h5py.File('../HI_bias/fields_z=%.1f.hdf5'%z,'r')
    delta_HI = f['delta_HI'][:]
    delta_m  = f['delta_m'][:]
    f.close()

    print 'Omega_HI(z=%d) = %.4e'\
        %(z,np.sum(delta_HI, dtype=np.float64)/(BoxSize**3*2.775e11))
    print '%.2f < M_HI < %.2f'%(np.min(delta_HI), np.max(delta_HI))

    delta_HI_new = SL.field_smoothing(delta_HI, W_k, threads)
    delta_m_new  = SL.field_smoothing(delta_m,  W_k, threads)

    #delta_HI_new = TR.grid_reducer(delta_HI, dims)
    #delta_m_new  = TR.grid_reducer(delta_m,  dims)

    print 'Omega_HI(z=%d) = %.4e'\
        %(z,np.sum(delta_HI_new, dtype=np.float64)/(BoxSize**3*2.775e11))
    print '%.2f < M_HI < %.2f'%(np.min(delta_HI_new), np.max(delta_HI_new))

    delta_HI_new /= np.mean(delta_HI_new)
    delta_m_new /= np.mean(delta_m_new)

    delta_HI_new = np.ravel(delta_HI_new)
    delta_m_new  = np.ravel(delta_m_new)

    delta_HI_new = delta_HI_new[numbers]
    delta_m_new  = delta_m_new[numbers]

    np.savetxt('delta_HI_vs_delta_m_%.1f_z=%d.txt'%(R,z),
               np.transpose([delta_m_new, delta_HI_new]))

    print ' '
