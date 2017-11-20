# This library is used to make images from the Illustris simulations
from mpi4py import MPI
import numpy as np
import snapshot as sn
import groupcat
import readsnapHDF5 as rs
import HI_library as HIL
import sys,os,glob,h5py,time
import MAS_library as MASL

####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

#######################################################################################
# This routine reads in parallel the whole Illustris snapshot and select the 
# particles within the input region
# snapshot_root -------> folder containing the snapdir folders
# snapnum -------------> number of the snapshot
# TREECOOL_file -------> location of TREECOOL file
# x_min, x_max --------> limits in the x-direction
# y_min, y_max --------> limits in the y-direction
# z_min, z_max --------> limits in the z-direction
# padding -------------> padding to get all particles are borders
# fout ----------------> name of output file (hdf5)
def Illustris_region(snapshot_root, snapnum, TREECOOL_file, x_min, x_max, 
                     y_min, y_max, z_min, z_max, padding, fout):


    # read snapshot and find number of subfiles
    snapshot = snapshot_root + 'snapdir_%03d/snap_%03d'%(snapnum,snapnum)
    header   = rs.snapshot_header(snapshot)
    nall     = header.nall
    redshift = header.redshift
    BoxSize  = header.boxsize/1e3 #Mpc/h
    filenum  = header.filenum
    Omega_m  = header.omega0
    Omega_L  = header.omegaL
    h        = header.hubble

    if myrank==0:
        print '\n'
        print 'BoxSize         = %.3f Mpc/h'%BoxSize
        print 'Number of files = %d'%filenum
        print 'Omega_m         = %.3f'%Omega_m
        print 'Omega_l         = %.3f'%Omega_L
        print 'redshift        = %.3f'%redshift

    # find the numbers each cpu will work on
    array   = np.arange(0, filenum)
    numbers = np.where(array%nprocs==myrank)[0]

    # do a loop over the different realizations
    particles = 0
    for i in numbers:

        snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(snapnum,snapnum,i)
        pos = rs.read_block(snapshot, 'POS ', parttype=0, verbose=False)/1e3
        pos = pos.astype(np.float32)

        # check if particles are in the region
        indexes_region = np.where((pos[:,0]>=x_min-padding) & (pos[:,0]<=x_max+padding) &\
                                  (pos[:,1]>=y_min-padding) & (pos[:,1]<=y_max+padding) &\
                                  (pos[:,2]>=z_min-padding) & (pos[:,2]<=z_max+padding))[0]

        # if particles are not in the region continue
        local_particles = indexes_region.shape[0]
        print 'Myrank = %d ---> num = %d ---> part = %ld'%(myrank,i,local_particles)
        if local_particles==0:  continue

        # find radii, HI and gas masses
        MHI  = rs.read_block(snapshot, 'NH  ', parttype=0, verbose=False)#HI/H
        mass = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10
        SFR  = rs.read_block(snapshot, 'SFR ', parttype=0, verbose=False)
        indexes = np.where(SFR>0.0)[0];  del SFR

        # find the metallicity of star-forming particles
        metals = rs.read_block(snapshot, 'GZ  ', parttype=0, verbose=False)
        metals = metals[indexes]/0.0127

        # find densities of star-forming particles: units of h^2 Msun/Mpc^3
        rho    = rs.read_block(snapshot, 'RHO ', parttype=0, verbose=False)*1e19
        Volume = mass/rho                            #(Mpc/h)^3
        radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 
        rho    = rho[indexes]                        #h^2 Msun/Mpc^3
        Volume = Volume[indexes]                     #(Mpc/h)^3

        # find volume and radius of star-forming particles
        radii_SFR  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 
            
        # find HI/H fraction for star-forming particles
        MHI[indexes] = HIL.Rahmati_HI_Illustris(rho, radii_SFR, metals, redshift, 
                                                h, TREECOOL_file, Gamma=None,
                                                fac=1, correct_H2=True) #HI/H
        MHI *= (0.76*mass)

        # select the particles belonging to the region
        pos   = pos[indexes_region]
        MHI   = MHI[indexes_region]
        radii = radii[indexes_region]
        mass  = mass[indexes_region]

        # write partial files        
        new_size = particles + local_particles    

        if particles==0:
            f = h5py.File(fout[:-5]+'_%d.hdf5'%myrank, 'w')
            f.create_dataset('pos',   data=pos,   maxshape=(None,3))
            f.create_dataset('M_HI',  data=MHI,   maxshape=(None,))
            f.create_dataset('radii', data=radii, maxshape=(None,))
            f.create_dataset('mass',  data=mass,  maxshape=(None,))
        else:
            f = h5py.File(fout[:-5]+'_%d.hdf5'%myrank, 'a')
            pos_f   = f['pos'];    pos_f.resize((new_size,3))
            M_HI_f  = f['M_HI'];   M_HI_f.resize((new_size,))
            radii_f = f['radii'];  radii_f.resize((new_size,))
            mass_f  = f['mass'];   mass_f.resize((new_size,))
            pos_f[particles:]   = pos
            M_HI_f[particles:]  = MHI
            radii_f[particles:] = radii
            mass_f[particles:]  = mass
        f.close()
        particles += local_particles
                
    comm.Barrier()

    # sum the particles found in each cpu
    All_particles = 0 
    All_particles = comm.reduce(particles, op=MPI.SUM, root=0)

    # Master will merge partial files into a file one
    if myrank==0:

        print 'Found %d particles'%All_particles
        f = h5py.File(fout,'w')
        
        f1 = h5py.File(fout[:-5]+'_0.hdf5','r')
        pos   = f1['pos'][:]
        M_HI  = f1['M_HI'][:]
        radii = f1['radii'][:]
        mass  = f1['mass'][:]
        f1.close()

        particles = mass.shape[0]
        pos_f   = f.create_dataset('pos',   data=pos,   maxshape=(None,3))
        M_HI_f  = f.create_dataset('M_HI',  data=M_HI,  maxshape=(None,))
        radii_f = f.create_dataset('radii', data=radii, maxshape=(None,))
        mass_f  = f.create_dataset('mass',  data=mass,  maxshape=(None,))

        for i in xrange(1,nprocs):
            f1 = h5py.File(fout[:-5]+'_%d.hdf5'%i,'r')
            pos   = f1['pos'][:]
            M_HI  = f1['M_HI'][:]
            radii = f1['radii'][:]
            mass  = f1['mass'][:]
            f1.close()
            
            size = mass.shape[0]
            
            pos_f.resize((particles+size,3));  pos_f[particles:] = pos
            M_HI_f.resize((particles+size,));  M_HI_f[particles:] = M_HI
            radii_f.resize((particles+size,)); radii_f[particles:] = radii
            mass_f.resize((particles+size,));  mass_f[particles:] = mass

            particles += size

        f.close()

        for i in xrange(nprocs):
            os.system('rm '+fout[:-5]+'_%d.hdf5'%i)
#######################################################################################    

#######################################################################################    
# This routine finds the positions, radii, HI and gas masses of the particles belonging
# to a give halo
def Illustris_halo(snapshot_root, snapnum, halo_number, TREECOOL_file, fout):

    # find snapshot name and read header
    snapshot = snapshot_root + 'snapdir_%03d/snap_%03d'%(snapnum, snapnum)
    header   = rs.snapshot_header(snapshot)
    redshift = header.redshift
    BoxSize  = header.boxsize/1e3 #Mpc/h
    filenum  = header.filenum
    Omega_m  = header.omega0
    Omega_L  = header.omegaL
    h        = header.hubble

    print '\nBoxSize         = %.1f Mpc/h'%BoxSize
    print 'Number of files = %d'%filenum
    print 'Omega_m         = %.3f'%Omega_m
    print 'Omega_l         = %.3f'%Omega_L
    print 'redshift        = %.3f'%redshift

    # read number of particles in halos and subhalos and number of subhalos
    halos = groupcat.loadHalos(snapshot_root, snapnum, 
            fields=['GroupLenType','GroupPos','GroupMass'])
    halo_len  = halos['GroupLenType'][:,0]  
    halo_pos  = halos['GroupPos']/1e3
    halo_mass = halos['GroupMass']*1e10
    del halos


    # find where the halo starts and ends in the file
    begin = np.sum(halo_len[:halo_number], dtype=np.int64)
    end   = begin + halo_len[halo_number]
    print begin,end

    # do a loop over all snapshot subfiles
    f = h5py.File(fout,'w')
    pos_f   = f.create_dataset('pos',   (0,3),  maxshape=(None,3))
    mass_f  = f.create_dataset('mass',  (0,),   maxshape=(None,))
    MHI_f   = f.create_dataset('M_HI',  (0,),   maxshape=(None,))
    radii_f = f.create_dataset('radii', (0,),   maxshape=(None,))

    begin_subfile, particles = 0, 0
    for i in xrange(filenum):

        # find subfile name and read the number of particles in it
        snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(snapnum, snapnum, i)
        header = rs.snapshot_header(snapshot)
        npart  = header.npart 

        end_subfile = begin_subfile + npart[0]

        # if all particles in the halo has been read exit loop
        if end<begin_subfile:  break

        # if the subfile does not contain any particle move to next subfile
        if begin>end_subfile:
            begin_subfile = end_subfile;  continue


        print 'Working with subfile %03d'%i
        pos  = rs.read_block(snapshot, 'POS ', parttype=0, verbose=False)/1e3
        pos  = pos.astype(np.float32)
        MHI  = rs.read_block(snapshot, 'NH  ', parttype=0, verbose=False)#HI/H
        mass = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10
        SFR  = rs.read_block(snapshot, 'SFR ', parttype=0, verbose=False)
        indexes = np.where(SFR>0.0)[0];  del SFR

        # find the metallicity of star-forming particles
        metals = rs.read_block(snapshot, 'GZ  ', parttype=0, verbose=False)
        metals = metals[indexes]/0.0127

        # find densities of star-forming particles: units of h^2 Msun/Mpc^3
        rho = rs.read_block(snapshot, 'RHO ', parttype=0, verbose=False)*1e19
        Volume = mass/rho                            #(Mpc/h)^3
        radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 

        # find density and radius of star-forming particles
        radii_SFR = radii[indexes]    
        rho       = rho[indexes]

        # find HI/H fraction for star-forming particles
        MHI[indexes] = HIL.Rahmati_HI_Illustris(rho, radii_SFR, metals, redshift, 
                                                h, TREECOOL_file, Gamma=None,
                                                fac=1, correct_H2=True) #HI/H
        MHI *= (0.76*mass)

        # find the indexes of current subfile that contribute to halo
        begin_array = begin - begin_subfile
        end_array   = begin_array + (end-begin)

        if end>end_subfile:
            end_array = end_subfile - begin_subfile
            begin     = end_subfile

        new_size = particles + (end_array - begin_array)

        pos_f.resize((new_size,3))
        pos_f[particles:] = pos[begin_array:end_array]

        mass_f.resize((new_size,))
        mass_f[particles:] = mass[begin_array:end_array]

        MHI_f.resize((new_size,))
        MHI_f[particles:] = MHI[begin_array:end_array]

        radii_f.resize((new_size,))
        radii_f[particles:] = radii[begin_array:end_array]

        particles = new_size
        begin_subfile = end_subfile


    f.close()
    print 'Halo mass = %.3e'%halo_mass[halo_number]
    print 'Halo pos  =',halo_pos[halo_number]
    print 'Number of particles in the halo = %ld'%particles


