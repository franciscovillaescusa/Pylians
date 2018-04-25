import numpy as np
import sys,os,h5py
import HI_library as HIL


TREECOOL_file = '/n/home09/fvillaescusa/Illustris/HI/TREECOOL_fg_dec11'
################################ INPUT ########################################
# run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'

snapnum = 21 #33, 25, 21
fout    = 'cross_section_new_z=4hdf5'

resolution = 750e-6 #Mpc/h

threshold = np.array([10**(20.0), 10**(20.3), 10**(21.0), 10**(21.5), 
                      10**(22.0), 10**(22.5), 10**(23.0)])
###############################################################################

# find offset_root and snapshot_root                             
snapshot_root = '%s/output/'%run

mass, M_HI, cross_section = HIL.DLAs_cross_section(snapshot_root, snapnum, 
	TREECOOL_file, resolution, threshold)

f = h5py.File(fout, 'w')
f.create_dataset('M',     data=mass)
f.create_dataset('M_HI',  data=M_HI)
f.create_dataset('sigma', data=cross_section)
f.create_dataset('N_HI',  data=threshold)
f.close()





