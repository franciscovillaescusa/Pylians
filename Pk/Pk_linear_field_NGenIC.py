import numpy as np
import Pk_library as PKL

############################# INPUT ######################################
BoxSize = 1000.0 #size of the box in Mpc/h

# N-GenIC generated files
f_coordinates = 'Coordinates_ptype_0'
f_amplitudes  = 'Amplitudes_ptype_0'
f_phases      = 'Phases_ptype_0'     #no needed for the P(k)
##########################################################################

# compute the Pk of the density field
k,Pk,Nmodes = PKL.Pk_NGenIC_IC_field(f_coordinates, f_amplitudes, BoxSize) 
np.savetxt('Pk_linear_df_m_z=127.txt', np.transpose([k,Pk,Nmodes]))




