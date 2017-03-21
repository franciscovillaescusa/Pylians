import numpy as np
import camb
import sys,os


################################## INPUT ######################################
# neutrino parameters
hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
Mnu       = 0.60  #eV
Nnu       = 3  #number of massive neutrinos

# cosmological parameters
h       = 0.6711
Omega_c = 0.2685 - Mnu/(93.14*h**2)
Omega_b = 0.049
Omega_k = 0.0
tau     = None

# initial P(k) parameters
ns           = 0.9624
As           = 2.13e-9
pivot_scalar = 0.05
pivot_tensor = 0.05

# redshifts and k-range
redshifts    = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0] 
kmax         = 10.0
k_per_logint = 5
###############################################################################

Omega_cb = Omega_c + Omega_b

pars = camb.CAMBparams()

# set accuracy of the calculation
pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=2.0, 
                  lAccuracyBoost=2.0, HighAccuracyDefault=True, 
                  DoLateRadTruncation=True)

# set value of the cosmological parameters
pars.set_cosmology(H0=h*100.0, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, 
                   mnu=Mnu, omk=Omega_k, 
                   neutrino_hierarchy=hierarchy, 
                   num_massive_neutrinos = Nnu,
                   tau=tau)
                   
# set the value of the primordial power spectrum parameters
pars.InitPower.set_params(As=As, ns=ns, 
                          pivot_scalar=pivot_scalar, 
                          pivot_tensor=pivot_tensor)

# set redshifts, k-range and k-sampling
pars.set_matter_power(redshifts=redshifts, kmax=kmax, 
                      k_per_logint=k_per_logint)

# compute results
results = camb.get_results(pars)

# get raw matter power spectrum and transfer functions with strange k-binning
#k, zs, Pk = results.get_linear_matter_power_spectrum()
#Tk        = (results.get_matter_transfer_data()).transfer_data

# interpolate to get Pmm, Pcc...etc
k, zs, Pkmm = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=7, var2=7, 
                                                have_power_spectra=True, 
                                                params=None)

k, zs, Pkcc = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=2, var2=2, 
                                                have_power_spectra=True, 
                                                params=None)

k, zs, Pkbb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=3, var2=3, 
                                                have_power_spectra=True, 
                                                params=None)

k, zs, Pkcb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=2, var2=3, 
                                                have_power_spectra=True, 
                                                params=None)

Pkcb = (Omega_c**2*Pkcc + Omega_b**2*Pkbb +\
        2.0*Omega_b*Omega_c*Pkcb)/Omega_cb**2

k, zs, Pknn = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=6, var2=6, 
                                                have_power_spectra=True, 
                                                params=None)

print pars

# get sigma_8 and Hz in km/s/(kpc/h)
s8 = np.array(results.get_sigma8())
Hz = results.hubble_parameter(99.0)
print 'H(z=99)      = %.4f km/s/(kpc/h)'%(Hz/1e3/h)
print 'sigma_8(z=0) = %.4f'%s8[-1]


# do a loop over all redshifts
for i,z in enumerate(zs):

    fout1 = 'Pk_mm_z=%.3f.txt'%z
    fout2 = 'Pk_cc_z=%.3f.txt'%z
    fout3 = 'Pk_bb_z=%.3f.txt'%z
    fout4 = 'Pk_cb_z=%.3f.txt'%z
    fout5 = 'Pk_nn_z=%.3f.txt'%z

    np.savetxt(fout1,np.transpose([k,Pkmm[i,:]]))
    np.savetxt(fout2,np.transpose([k,Pkcc[i,:]]))
    np.savetxt(fout3,np.transpose([k,Pkbb[i,:]]))
    np.savetxt(fout4,np.transpose([k,Pkcb[i,:]]))
    np.savetxt(fout5,np.transpose([k,Pknn[i,:]]))


    #fout = 'Pk_trans_z=%.3f.txt'%z
    # notice that transfer functions have an inverted order:i=0 ==>z_max
    #np.savetxt(fout,np.transpose([Tk[0,:,i],Tk[1,:,i],Tk[2,:,i],Tk[3,:,i],
    #                               Tk[4,:,i],Tk[5,:,i],Tk[6,:,i]]))
