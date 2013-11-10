import numpy as np
import mass_function_library as MFL
import bias_library as BL

################################# INPUT ######################################
Omega_CDM=0.2208
Omega_B=0.05

M_min=1e10 #Msun/h
M_max=1e16 #Msun/h

bins=50

z=0.0

f_Pk_DM='../../../CAMB_TABLES/ics_matterpow_0.dat'
f_transfer='../../../CAMB_TABLES/ics_transfer_0.dat'

f_out='Crocce_MF_z=0.dat'

do_CDM=False
##############################################################################
OmegaM=Omega_CDM+Omega_B
M=np.logspace(np.log10(M_min),np.log10(M_max),bins+1)


if do_CDM:
    [k,Pk]=BL.CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)    
else:
    [k,Pk]=BL.DM_Pk(f_Pk_DM)

#dndM=MFL.ST_mass_function(k,Pk,OmegaM,None,None,None,M)[1]
#dndM=MFL.Tinker_mass_function(k,Pk,OmegaM,None,None,None,M)[1]
dndM=MFL.Crocce_mass_function(k,Pk,OmegaM,z,None,None,None,M)[1]
#dndM=MFL.Jenkins_mass_function(k,Pk,OmegaM,None,None,None,M)[1]
#dndM=MFL.Warren_mass_function(k,Pk,OmegaM,None,None,None,M)[1]

f=open(f_out,'w')
for i in range(len(M)):
    f.write(str(M[i])+' '+str(dndM[i])+'\n')
f.close()




