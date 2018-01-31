# Pylians

Pylians stands for **Py**thon **li**braries for the **a**nalysis of **n**umerical **s**imulations. They are a set of python libraries, written in python, cython and C, whose purposes is to facilitate the analysis of numerical simulations (both N-body and hydro). Among other things, they can be used to:

- Compute density fields
- Compute power spectra
- Compute bispectra
- Compute correlation functions
- Identify voids
- Populate halos with galaxies using an HOD
- Apply HI+H2 corrections to the output of hydrodynamic simulations
- Make 21cm maps
- Compute DLAs column density distribution functions
- Plot density fields and make movies

[Pylians](https://en.wikipedia.org/wiki/Nestor_(mythology)) were the native or inhabitant of the Homeric town of Pylos. 

## Requisites

- numpy
- scipy
- h5py
- pyfftw
- mpi4py
- cython
- openmp
 
We recommend installing the first packages with [anaconda](https://www.anaconda.com/download/?lang=en-us). 

## Installation

```python
cd library
python setup.py build
```

The compiled libraries and scripts are in build/lib.XXX, where XXX depends on your machine. E.g. build/lib.linux-x86_64-2.7 or build/lib.macosx-10.7-x86_64-2.7

Add that folder to your PYTHONPATH in ~/.bashrc

```sh
export PYTHONPATH=$PYTHONPATH:/home/villa/software/Fcodes/library/build/lib.linux-x86_64-2.7
```

## Usage
We provide some examples on how to use the library for different purposes.

#### Density field

Example on how to compute the density field of CDM from a Gadget snapshot using the cloud-in-cell (CIC) mass assignment scheme

```python
import numpy as np
import MAS_library as MASL

# input parameters
snapshot_fname = 'snapdir_010/snap_010'
grid           = 512 #density field will be computed on a grid with grid^3 cells
ptypes         = [1] #[0],[1],[2],[4] for gas, CDM, neutrinos and stars. Can be combined, e.g. [0,1] for CDM+gas
MAS            = 'CIC' #Choose among 'NGP','CIC','TSC','PCS' 
do_RSD         = False #whether compute the density field in real- or redshift-space
axis           = 0 #if do_RSD, axis (0, 1 or 2), along which apply redshift-space distortions

delta = MASL.density_field_gadget(snapshot_fname, ptypes, grid, MAS, do_RSD, axis)
```
At this point the array ```delta``` contains the number density field of CDM. To compute the overdensity:
```python
delta /= np.mean(delta, dtype=np.float64)  #compute the mean on a double to increase accuracy
delta -= 1.0
```
The ```density_field_gadget``` routine supports Gadget format 1, format 2 and hdf5 formats and deals with them internally.


To compute the density field from a set of particles:

```python
import numpy as np
import MAS_library as MASL

# input parameters
grid    = 512  #density field will be computed on a grid with grid^3 cells
BoxSize = 1000 #Mpc/h. Size of the periodic box
MAS     = 'CIC'

# define the array hosting the density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)

# read the particle positions
pos = np.loadtxt('myfile.txt') #Mpc/h 

# compute density field
MASL.MA(pos,delta,BoxSize,MAS)

delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0 #for overdensity
```

#### Power spectrum
Once you have computed the overdensity field, the different power spectra can be computed as
```python
import Pk_library as PKL

Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)
```
If the density field is in redshift-space, the ```axis``` variable will tell the code along which axis (0,1 or 2) compute the quadrupole and hexadecapole. In real-space the value of axis will not have any effect. The variable ```threads``` is the number of openmp threads to be used when computing the power spectrum. The above ```Pk``` variable is a class containing the 1D, 2D and 3D power spectra, that can be retrieved with
```
# 1D P(k)
k1D      = Pk.k1D      
Pk1D     = Pk.Pk1D     
Nmodes1D = Pk.Nmodes1D  

# 2D P(k)
kpar     = Pk.kpar    
kper     = Pk.kper
Pk2D     = Pk.Pk2D
Nmodes2D = Pk.Nmodes2D

# 3D P(k)
k      = Pk.k3D
Pk0    = Pk.Pk[:,0] #monopole
Pk2    = Pk.Pk[:,1] #quadrupole
Pk4    = Pk.Pk[:,2] #hexadecapole
Nmodes = Pk.Nmodes3D
```

If the density field has not been computed using NGP, CIC, TSC or PCS set ```MAS='None'```.

#### Cross-power spectrum
For multiple overdensity fields, the auto- and cross-power spectra can be computed as

```python
import Pk_library as PKL

Pk = PKL.XPk([delta1,delta2], BoxSize=1000, axis=0, MAS=['CIC','CIC'], threads=16)
```

where ```delta1``` and ```delta2``` are the two overdensity fields. The ```Pk``` variable is a class that contains all the following information

```python
# 1D P(k)
k1D      = Pk.k1D
Pk1D_1   = Pk.Pk1D[:,0] #field 1
Pk1D_2   = Pk.Pk1D[:,1] #field 2
Pk1D_X   = Pk.PkX1D[:,0] #field 1 - field 2 cross 1D P(k)
Nmodes1D = Pk.Nmodes1D

# 2D P(k)
kpar     = Pk.kpar
kper     = Pk.kper
Pk2D_1   = Pk.Pk2D[:,0]  #2D P(k) of field 1
Pk2D_2   = Pk.Pk2D[:,1]  #2D P(k) of field 2
Pk2D_X   = Pk.PkX2D[:,0] #2D cross-P(k) of fields 1 and 2
Nmodes2D = Pk.Nmodes2D

# 3D P(k)
k      = Pk.k3D
Pk0_1  = Pk.Pk[:,0,0]  #monopole of field 1
Pk0_2  = Pk.Pk[:,0,1]  #monopole of field 2
Pk2_1  = Pk.Pk[:,1,0]  #quadrupole of field 1
Pk2_2  = Pk.Pk[:,1,1]  #quadrupole of field 2
Pk4_1  = Pk.Pk[:,2,0]  #hexadecapole of field 1
Pk4_2  = Pk.Pk[:,2,1]  #hexadecapole of field 2
Pk0_X  = Pk.XPk[:,0,0] #monopole of 1-2 cross P(k)
Pk2_X  = Pk.XPk[:,1,0] #quadrupole of 1-2 cross P(k)
Pk4_X  = Pk.XPk[:,2,0] #hexadecapole of 1-2 cross P(k)
Nmodes = Pk.Nmodes3D
```

The ```XPk``` function can be used for more than two fields, e.g.
```python
Pk = PKL.XPk([delta1,delta2,delta3,delta4], BoxSize=1000, axis=0, MAS=['CIC','NGP','TSC','None'], threads=16)
```


## Contact

For comments, problems, bugs ... etc you can reach me at [fvillaescusa@flatironinstitute.org](mailto:fvillaescusa@flatironinstitute.org).