# Fcodes

The Fcodes are a set of python/cython/c libraries and scripts that can be used to analyze the output of numerical simulations (both N-body and hydro). They can be used to:

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

## Requisites

- numpy
- scipy
- pyfftw
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

Example on how to compute the density field of CDM from a Gadget snapshot

```python
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

If you are computing the density field from a set of particles:

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


## Contact

For comments, problems, bugs ... etc you can reach me at [fvillaescusa@flatironinstitute.org](mailto:fvillaescusa@flatironinstitute.org).


