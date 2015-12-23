import numpy as np
import readsnap
import CIC_library as CIC
import Power_spectrum_library as PSL
import redshift_space_library as RSL
import sys,os


# This routine takes a given snapshot and computes the total matter 
# power spectrum
def compute_Pk(snapshot_fname,dims,do_RSD,axis,hydro):

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

    z = '%.3f'%redshift
    f_out = 'Pk_m_z='+z+'.dat'

    # compute the values of Omega_CDM and Omega_B
    Omega_cdm = Nall[1]*Masses[1]/BoxSize**3/rho_crit
    Omega_nu  = Nall[2]*Masses[2]/BoxSize**3/rho_crit
    Omega_b   = Omega_m-Omega_cdm-Omega_nu
    print '\nOmega_CDM = %.4f\nOmega_B   = %0.4f\nOmega_NU  = %.4f'\
        %(Omega_cdm,Omega_b,Omega_nu)
    print 'Omega_M   = %.4f'%(Omega_m)

    # read the positions of all the particles
    pos = readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
    print '%.3f < X [Mpc/h] < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0]))
    print '%.3f < Y [Mpc/h] < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1]))
    print '%.3f < Z [Mpc/h] < %.3f\n'%(np.min(pos[:,2]),np.max(pos[:,2]))

    if do_RSD:
        print 'moving particles to redshift-space'
        # read the velocities of all the particles
        vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=-1) #km/s
        RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis);  del vel

    # read the masses of all the particles
    if not(hydro):
        Ntotal = np.sum(Nall,dtype=np.int64)   #compute the number of particles
        M = np.zeros(Ntotal,dtype=np.float32)  #define the mass array
        offset = 0
        for ptype in [0,1,2,3,4,5]:
            M[offset:offset+Nall[ptype]] = Masses[ptype];  offset += Nall[ptype]
    else:
        M = readsnap.read_block(snapshot_fname,"MASS",parttype=-1)*1e10 #Msun/h
    print '%.3e < M [Msun/h] < %.3e'%(np.min(M),np.max(M))
    print 'Omega_M = %.4f\n'%(np.sum(M,dtype=np.float64)/rho_crit/BoxSize**3)

    # compute the mean mass per grid cell
    mean_M = np.sum(M,dtype=np.float64)/dims**3

    # compute the mass within each grid cell
    delta = np.zeros(dims**3,dtype=np.float32)
    CIC.CIC_serial(pos,dims,BoxSize,delta,M); del pos
    print '%.6e should be equal to \n%.6e\n'\
        %(np.sum(M,dtype=np.float64),np.sum(delta,dtype=np.float64)); del M

    # compute the density constrast within each grid cell
    delta/=mean_M; delta-=1.0
    print '%.3e < delta < %.3e\n'%(np.min(delta),np.max(delta))

    # compute the P(k)
    Pk = PSL.power_spectrum_given_delta(delta,dims,BoxSize)

    # write P(k) to output file
    np.savetxt(f_out,np.transpose([Pk[0],Pk[1]]))



rho_crit=2.77536627e11 #h^2 Msun/Mpc^3
################################## INPUT ######################################
if len(sys.argv)>1:
    sa=sys.argv
    snapshot_fname=sa[1]; dims=int(sa[2]);
    do_RSD=bool(int(sa[3])); axis=int(sa[4])

else:
    snapshots = ['../snapdir_003/snap_003']
    dims      = 768
    do_RSD    = False
    axis      = 0
    hydro     = False   #whether sim is hydro or not (to read particle masses)
###############################################################################



for snapshot_fname in snapshots:
    compute_Pk(snapshot_fname,dims,do_RSD,axis,hydro)

