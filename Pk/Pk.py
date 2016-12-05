import numpy as np
import Pk_library as PKL

################################## INPUT ######################################
snapshot_fname = ['../ics',
                  '../snapdir_000/snap_000',
                  '../snapdir_001/snap_001',
                  '../snapdir_002/snap_002',
                  '../snapdir_003/snap_003',
                  '../snapdir_004/snap_004',
                  '../snapdir_005/snap_005',
                  '../snapdir_006/snap_006',
                  '../snapdir_007/snap_007']
                  
dims           = 1024
particle_type  = [1,2]
do_RSD         = False
axis           = 0
hydro          = False
cpus           = 14
###############################################################################

# compute the P(k) of each snapshot
for snapshot in snapshot_fname:
    PKL.Pk_Gadget(snapshot,dims,particle_type,do_RSD,axis,hydro,cpus)
                  



