import numpy as np
import Pk_library as PKL

################################## INPUT ######################################
snapshot_fname = ['../ics',
                  '../snapdir_000/snap_000',
                  '../snapdir_001/snap_001',
                  '../snapdir_002/snap_002',
                  '../snapdir_003/snap_003']
                  
dims           = 1024
particle_type  = [1,2] #list with particle types. [-1] for total matter
cpus           = 14
###############################################################################

# do a loop over the different snapshots
for snapshot in snapshot_fname:

    ######## REAL-SPACE ########
    do_RSD = False;  axis = 0 
    PKL.Pk_Gadget(snapshot,dims,particle_type,do_RSD,axis,cpus)
                  

    ###### REDSHIFT-SPACE ######
    do_RSD = True
    for axis in [0,1,2]:
        PKL.Pk_Gadget(snapshot,dims,particle_type,do_RSD,axis,cpus)
                  



