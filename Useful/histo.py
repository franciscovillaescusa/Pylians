# This script reads a file and split the data in bins bins between the 
# minimum and maximum. In each bin it computes the number of elements, the sum
# of the y-value and the mean value of y.
# Usage: python histo.py file_in file_out bins
# options: --logx (for log-x bins), --column1 c1 (read c1 column as x)
# --column2 c2 (read c2 column as y)
import argparse
import numpy as np
import sys,os

################################# INPUT #################################
parser = argparse.ArgumentParser()
parser.add_argument('file_in',   help="name of input file")
parser.add_argument('file_out',  help="name of output file")
parser.add_argument('bins', type=int, help="number of bins to use")
parser.add_argument('--logx', action="store_true",  
                    help="use log-x bins: default linear")
parser.add_argument('-c1','--column1', type=int, 
                    help="column to use in the file for x")
parser.add_argument('-c2','--column2', type=int, 
                    help="column to use in the file for y")
args = parser.parse_args()
#########################################################################

# find the columns to read in the input file (default 0 & 1)
c1,c2 = args.column1, args.column2
if c1 is None:  c1 = 0
if c2 is None:  c2 = 1

# read data file
x, y = np.loadtxt(args.file_in, usecols=(c1,c2), unpack=True)

# find the x-bins
x_min, x_max = np.min(x), np.max(x)
if args.logx: 
    bins   = np.logspace(np.log10(x_min), np.log10(x_max), args.bins)
    mean_x = 10**(0.5*(np.log10(bins[1:]) + np.log10(bins[:-1])))
else:        
    bins   = np.linspace(x_min, x_max, args.bins)
    mean_x = 0.5*(bins[1:] + bins[:-1])
    
elements_x = np.histogram(x,bins=bins)[0] #number of elements in the x-bins
sum_y      = np.histogram(x,bins=bins,weights=y)[0] #sum of y in the x-bins
mean_y     = sum_y*1.0/elements_x #mean value of y in the x-bins
sum_y2     = np.histogram(x,bins=bins,weights=y**2)[0] #sum of y^2 in the x-bins
std_y      = np.sqrt(sum_y2*1.0/elements_x - mean_y**2)

# save results to file
np.savetxt(args.file_out, np.transpose([mean_x, mean_y, std_y,
                                        elements_x, sum_y]))

