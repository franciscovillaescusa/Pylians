# This script reads the peculiar velocities of the neutrino particles and 
# computes his distribution: fraction of neutrinos within each velocity bin. 
# It also computes the expectation from Fermi-Dirac momentum distribution 

import numpy as np
import readsnap

#################################### INPUT ####################################
snapshot_fname = '../ics'
bins           = 100 #number of bins for the distribution

# parameters for the FD distribution
Mnu      = 0.6       #eV
h_planck = 6.582e-16 #eV*s
kB       = 8.617e-5  #eV/K
c        = 3e5       #km/s 
Tnu      = 1.95      #K
###############################################################################

########## fraction from simulation ###########
# read snapshot redshift and neutrino velocities
z   = readsnap.snapshot_header(snapshot_fname).redshift
vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=2) #km/s

# compute velocity modulus
V = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2);  del vel

# define the velocity intervals, their mean value and their widths
vel_min, vel_max = np.min(V),np.max(V)
if vel_min==0.0:  vel_min = 1e-3
vel_intervals = np.logspace(np.log10(vel_min),np.log10(vel_max),bins+1)
dV            = vel_intervals[1:] - vel_intervals[:-1]       #km/s
V_mean        = 0.5*(vel_intervals[1:] + vel_intervals[:-1]) #km/s

# compute the franction of neutrinos within each velocity bin
hist = (np.histogram(V,bins=vel_intervals)[0])*1.0/len(V) 
###############################################

######## fraction from FD distribution ########
# compute the neutrino mass and total number density of neutrinos
mnu    = Mnu/3.0 #eV
rho_nu = 112.0*(1.0 + z)**3  #neutrinos/cm^3
rho_nu = rho_nu*(1e5)**3     #neutrinos/km^3

# factors for the FD distribution
prefactor = 4.0*np.pi*2.0/(2.0*np.pi*h_planck)**3  #1/(eV^3*s^3)
prefactor = prefactor*(mnu**3/c**6)                #1/(km^3*(km/s)^3)
fact      = kB*(1.0+z)*Tnu*c/mnu                   #km/s

# compute the number density of neutrinos in the velocity bin dV
pdf = V_mean**2*dV/(np.exp(V_mean/fact)+1.0) #(km/s)^3
pdf = pdf*prefactor                                  #1/km^3

# compute the fraction of neutrinos in each velocity bin
pdf = pdf/rho_nu
###############################################

# save results to file
np.savetxt('Nu_vel_distribution_z=%.3f.txt'%z,
           np.transpose([V_mean,hist,pdf]))







