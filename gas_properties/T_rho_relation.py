import numpy as np
import readsnap

################################# UNITS #####################################
rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3
yr       = 3.15576e7     #seconds
km       = 1e5           #cm
Mpc      = 3.0856e24     #cm
kpc      = 3.0856e21     #cm
Msun     = 1.989e33      #g
Ymass    = 0.24          #helium mass fraction
mH       = 1.6726e-24    #proton mass in grams
gamma    = 5.0/3.0       #ideal gas
kB       = 1.3806e-26    #gr (km/s)^2 K^{-1}
nu0      = 1420.0        #21-cm frequency in MHz
pi       = np.pi
#############################################################################

################################### INPUT ###################################
#snapshot_fname = '../new_GR_CDM_60_512/snapdir_011/snap_011'
#f_out          = 'T-rho_GR_CDM_z=3.txt'
snapshot_fname = '../new_GR_1keV_60_512/snapdir_011/snap_011'
f_out          = 'T-rho_GR_1keV_z=3.txt'
#snapshot_fname = '../new_fR5_1keV_60_512/snapdir_011/snap_011'
#f_out          = 'T-rho_fR5_1keV_z=3.txt'

Omega_m = 0.304752
Omega_b = Omega_m - 0.25667

overdensity_min = np.log10(1e-2)
overdensity_max = np.log10(1e6)

T_min = np.log10(1e3)
T_max = np.log10(1e8)

bins_histo = 200
#############################################################################


# read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head     = readsnap.snapshot_header(snapshot_fname)
BoxSize  = head.boxsize/1e3 #Mpc/h
Nall     = head.nall
Masses   = head.massarr*1e10 #Msun/h
Omega_m  = head.omega_m
Omega_l  = head.omega_l
redshift = head.redshift
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h        = head.hubble

# read the density, electron fraction and internal energy
# rho units: h^2 Msun / Mpc^3
rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10/1e-9 
ne  = readsnap.read_block(snapshot_fname,"NE  ",parttype=0) #electron fraction
U   = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2

# compute the mean molecular weight
yhelium = (1.0-0.76)/(4.0*0.76) 
mean_mol_weight = (1.0+4.0*yhelium)/(1.0+yhelium+ne);  del ne

# compute the temperature of the gas particles
T = U*(gamma-1.0)*mH*mean_mol_weight/kB;  del U, mean_mol_weight
T = T.astype(np.float64)

print '%.3e < T[k] < %.3e'%(np.min(T),np.max(T))

mean_rho_b      = Omega_b*rho_crit
rho_overdensity = rho/mean_rho_b;  del rho
print '%.3e < rho_g/<rho_b> < %.3e'\
    %(np.min(rho_overdensity),np.max(rho_overdensity))


# make a 2d histogram
H,xedges,yedges = np.histogram2d(np.log10(rho_overdensity),np.log10(T),
                                 bins=bins_histo,
                  range=[[overdensity_min,overdensity_max],[T_min,T_max]])

# normalize the histogram
H = H*1.0/len(T)

# check that the sum should be 1
print '%.3f should be close to 1'%np.sum(H,dtype=np.float64)

x_values = 0.5*(xedges[1:]+xedges[:-1])
y_values = 0.5*(yedges[1:]+yedges[:-1])

f = open(f_out,'w')
for i in xrange(bins_histo):
    for j in xrange(bins_histo):
        f.write(str(x_values[i])+' '+str(y_values[j])+' '+str(H[i,j])+'\n')
f.close()
