import numpy as np 
import sys,os,h5py
import math
import scipy.optimize as SO


def func(x, M0, N_HI=10**20.3):
	if M0<4 or M0>15:
		return 0

	A = -1.85-M0
	beta = 0.85*np.log10(N_HI)-16.35
	return 10**A*x**0.82*(1.0-np.exp(-(x/10**M0)**beta))

################################ INPUT ########################################
#redshifts = [2, 3, 4]
redshifts = [2,3,4]
bins = 25
###############################################################################

g = open('results_new.txt','w')
for z in redshifts:

	print ' '

	f_in = 'cross_section_new_z=%d.hdf5'%z

	f     = h5py.File(f_in, 'r')
	M     = f['M'][:]
	sigma = f['sigma'][:]
	N_HIs = f['N_HI'][:]
	f.close()	

	# find mass bins and mean
	M_bins = np.logspace(np.log10(np.min(M)), np.log10(np.max(M))*1.0001, bins)
	M_mean = 0.5*(M_bins[1:] + M_bins[:-1])
	Number = np.histogram(M, bins=M_bins)[0]

	for i,N_HI in enumerate(N_HIs):

		# compute mean cross-section and standard deviation
		sigma_mean  = np.histogram(M, bins=M_bins, weights=sigma[:,i])[0]
		sigma_mean2 = np.histogram(M, bins=M_bins, weights=sigma[:,i]**2)[0]

		sigma_mean  = sigma_mean*1.0/Number
		sigma_mean2 = sigma_mean2*1.0/Number
		var         = np.sqrt(sigma_mean2 - sigma_mean**2)

		# save results to file
		fout     = 'mean_sigma_new_%.2e_z=%d.txt'%(N_HI,z)
		fout_fit = 'fit_sigma_new_%.2e_z=%d.txt'%(N_HI,z)

		np.savetxt(fout, np.transpose([M_mean, sigma_mean, var]))

		indexes = np.where(sigma_mean>1e-12)[0]

		# fit the data to function
		p0 = [10]
		popt, pcov = SO.curve_fit(lambda x, M0: func(x, M0, N_HI),
			M_mean[indexes], sigma_mean[indexes], p0, sigma=var[indexes])

		# compute chi^2
		chi2 = 0
		for j in xrange(indexes.shape[0]):
			index = indexes[j]
			pred = func(M_mean[index], popt[0], N_HI=N_HI)
			chi2 += (pred-sigma_mean[index])**2/var[index]**2
		chi2 = chi2*1.0/(len(indexes)-2.0)

		print popt, chi2


		Masses = np.logspace(np.log10(np.min(M/10.0)), np.log10(np.max(M*10.0)), 1000)
		f = open(fout_fit, 'w')
		for i in xrange(1000):
			f.write(str(Masses[i])+' '+\
					str(func(Masses[i], popt[0], N_HI=N_HI))+\
				'\n')
		f.close()

		g.write('%.4f \n'%(popt[0]))

	g.write('\n')
g.close()
