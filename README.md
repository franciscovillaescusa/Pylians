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
export PYTHONPATH=$PYTHONPATH:$HOME/Pylians/library/build/lib.linux-x86_64-2.7
```

## Usage

The most updated documentation with examples can be found [here](https://github.com/franciscovillaescusa/Pylians/blob/master/documentation/Documentation.md).


## Contact

For comments, problems, bugs... etc you can reach me at [fvillaescusa@flatironinstitute.org](mailto:fvillaescusa@flatironinstitute.org).