import numpy as np
import readsubf
import readfof
import mass_function_library as MFL
import bias_library as BL
import sys


################################## INPUT ####################################
groups_fname  = './'
groups_number = 22

f_Pk_DM    = './CAMB_TABLES/ics_matterpow_0.dat'
f_transfer = './CAMB_TABLES/ics_transfer_0.dat'

min_mass = None #if None it will take the minimum halo mass in the catalogue
max_mass = None #if None it will take the minimum halo mass in the catalogue
bins     = 25
BoxSize  = 1000.0 #Mpc/h

obj = 'FoF'  #choose between 'FoF' or 'halos_m200'

Omega_CDM = 0.2208
Omega_B   = 0.05

f_out = 'mass_function_z=0.dat'
#############################################################################
Omega_M = Omega_CDM+Omega_B

MFL.mass_function(groups_fname,groups_number,obj,BoxSize,bins,f_out,
                  min_mass,max_mass,long_ids_flag=True,SFR_flag=False)

"""
[k,Pk]=BL.DM_Pk(f_Pk_DM)
f_out='f_sigma_FoF_corrected_z=0.dat_DM'
MFL.mass_function_fsigma(groups_fname,groups_number,f_out,min_mass,max_mass,
                         bins,BoxSize,obj,Omega_M,k,Pk)

[k,Pk]=BL.CDM_Pk(f_Pk_DM,f_transfer,Omega_CDM,Omega_B)
f_out='f_sigma_FoF_corrected_z=0.dat'
MFL.mass_function_fsigma(groups_fname,groups_number,f_out,min_mass,max_mass,
                         bins,BoxSize,obj,Omega_M,k,Pk)
"""
