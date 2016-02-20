import numpy as np
import Power_spectrum_library as PSL
import sys


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
                  
dims           = 768
particle_type  = [1,2]
do_RSD         = False
axis           = 0
hydro          = False
###############################################################################

# compute the P(k) of each snapshot
for snapshot in snapshot_fname:
    PSL.Power_spectrum_snapshot(snapshot,dims,particle_type,
                                do_RSD,axis,hydro)



