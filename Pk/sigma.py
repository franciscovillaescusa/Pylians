import argparse
import numpy as np
import mass_function_library as MFL

parser = argparse.ArgumentParser()
parser.add_argument("Pk_file", help="computes sigma_8 value from file")
args = parser.parse_args()

# read input file
k,Pk = np.loadtxt(args.Pk_file,unpack=True)

# compute value of sigma_8
s8 = MFL.sigma(k,Pk,8.0)
print 'Value of sigma_8 = %.5e'%s8
