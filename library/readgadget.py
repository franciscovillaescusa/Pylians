# This library is designed to be able to read Gadget format 1, format 2
# and hdf5 files
import numpy as np
import readsnap
import sys,os,h5py

# find snapshot name and format
def fname_format(snapshot):
    if os.path.exists(snapshot):
        if snapshot[-4:]=='hdf5':  filename, fformat = snapshot, 'hdf5'
        else:                      filename, fformat = snapshot, 'binary'
    elif os.path.exists(snapshot+'.0'):
        filename, fformat = snapshot+'.0', 'binary'
    elif os.path.exists(snapshot+'.hdf5'):
        filename, fformat = snapshot+'.hdf5', 'hdf5'
    elif os.path.exists(snapshot+'.0.hdf5'):
        filename, fformat = snapshot+'.0.hdf5', 'hdf5'
    else:  raise Exception('File not found!')
    return filename,fformat


# This class reads the header of the gadget file
class header:
    def __init__(self, snapshot):

        filename, fformat = fname_format(snapshot)

        if fformat=='hdf5':
            f             = h5py.File(filename, 'r')
            self.time     = f['Header'].attrs[u'Time']
            self.redshift = f['Header'].attrs[u'Redshift']
            self.boxsize  = f['Header'].attrs[u'BoxSize']
            self.filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
            self.omega_m  = f['Header'].attrs[u'Omega0']
            self.omega_l  = f['Header'].attrs[u'OmegaLambda']
            self.hubble   = f['Header'].attrs[u'HubbleParam']
            self.massarr  = f['Header'].attrs[u'MassTable']
            self.npart    = f['Header'].attrs[u'NumPart_ThisFile']
            self.nall     = f['Header'].attrs[u'NumPart_Total']
            self.cooling  = f['Header'].attrs[u'Flag_Cooling']
            self.format   = 'hdf5'
            f.close()

        else:        
            head = readsnap.snapshot_header(filename)
            self.time     = head.time
            self.redshift = head.redshift
            self.boxsize  = head.boxsize
            self.filenum  = head.filenum
            self.omega_m  = head.omega_m
            self.omega_l  = head.omega_l
            self.hubble   = head.hubble
            self.massarr  = head.massarr
            self.npart    = head.npart
            self.nall     = head.nall
            self.cooling  = head.cooling
            self.format   = head.format

        # km/s/(Mpc/h)
        self.Hubble = 100.0*np.sqrt(self.omega_m*(1.0+self.redshift)**3+self.omega_l)


# This function reads a field in the gadget file        
def read_field(snapshot, field, ptype):

    filename, fformat = fname_format(snapshot)
    head              = header(filename)

    if fformat=="binary":
        return readsnap.read_block(filename, field, parttype=ptype)
    else:
        prefix = 'PartType%d/'%ptype
        f = h5py.File(filename, 'r')
        if   field=="POS ":  suffix = "Coordinates"
        elif field=="MASS":  suffix = "Masses"
        elif field=="ID  ":  suffix = "ParticleIDs"
        elif field=="VEL ":  suffix = "Velocities"
        else: raise Exception('field not implemented in readgadget!')

        array = f[prefix+suffix][:];  f.close()
        if field=="VEL ":  array *= np.sqrt(head.time)

        if field=="POS " and array.dtype==np.float64:
            array = array.astype(np.float32)

        return array
