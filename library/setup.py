from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

ext_modules = [
    Extension("MAS_library", ["MAS_library.pyx"],
              extra_compile_args=['-O3','-ffast-math','-mtune=native']),

    Extension("density_field_library",["density_field_library.pyx"]),

    Extension("HI_clusters_library",["HI_clusters_library.pyx"]),

    Extension("Pk_library.Pk_library", ["Pk_library/Pk_library.pyx"]),

    Extension("Pk_library.bispectrum_library", 
              ["Pk_library/bispectrum_library.pyx"]),

    Extension("redshift_space_library",["redshift_space_library.pyx"]),

    Extension("HI_library",["HI_library.pyx"]),

    #Extension("nearest_point_library",["nearest_point_library.pyx"]),
    #extra_link_args=['-O3']),
    #extra_compile_args=['-O3', '-fopenmp'],
    #extra_link_args=['-O3','-fopenmp']),
]


setup(
    name = 'Fcodes',
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    py_modules=['HOD_library','bias_library','CAMB_library','cosmology_library',
                'correlation_function_library','halos_library','IM_library',
                'mass_function_library','readfof','readsnap','readsnap2',
                'readsnapHDF5','readsnap_mpi','readsubf','routines',
                'units_library','map']
)




