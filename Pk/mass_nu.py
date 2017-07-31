# This script computes the value of nu = delta_c/sigma given a linear power
# spectrum and value of Omega_cb. Mmin, Mmax and bins are optional arguments

import argparse
import numpy as np
import sys,os
import units_library as UL
import mass_function_library as MFL

parser = argparse.ArgumentParser()
parser.add_argument("Pk_file", help="computes sigma_8 value from file")
parser.add_argument("Omega_cb", type=float, help="value of Omega_cb")
parser.add_argument("--Mmin", default=1e9,  type=float, help="Mmin in Msun/h")
parser.add_argument("--Mmax", default=1e15, type=float, help="Mmax in Msun/h")
parser.add_argument("--bins", default=60,   type=int, help="number of bins")
args = parser.parse_args()

# find values of rho_crit, delta_c, Mmin, Mmax and bins
rho_crit = (UL.units()).rho_crit #h^2 Msun/Mpc^3
delta_c, Omega_cb = 1.686, args.Omega_cb
Mmin, Mmax, bins = args.Mmin, args.Mmax, args.bins
print 'Computing values of nu = delta_c/sigma(M)'
print 'for halos between Mmin = %.3e and Mmax = %.3e\n'%(args.Mmin,args.Mmax)

# read input file, compute Mass and R arrays
k,Pk = np.loadtxt(args.Pk_file,unpack=True)
Mass = np.logspace(np.log10(Mmin), np.log10(Mmax), bins)
R    = (3.0*Mass/(4.0*np.pi*rho_crit*Omega_cb))**(1.0/3.0)

print 'Mass [Msun/h]   nu = delta_c/sigma'
for i in xrange(bins):
    sigma = MFL.sigma(k,Pk,R[i])
    print '  %.3e         %.3f'%(Mass[i],delta_c/sigma)
