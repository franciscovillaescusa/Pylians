from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


ext_modules = [
    Extension("MAS_library", ["MAS_library.pyx"]),
    Extension("density_field_library",["density_field_library.pyx"]),
    Extension("HI_clusters_library",["HI_clusters_library.pyx"]),
    Extension("MAS_library", ["MAS_library.pyx"]),
    Extension("Pk_library.Pk_library", ["Pk_library/Pk_library.pyx"]),
    Extension("redshift_space_library",["redshift_space_library.pyx"]),
    Extension("HI_library",["HI_library.pyx"])
    #Extension("nearest_point_library",["nearest_point_library.pyx"]),
              
              #extra_compile_args=['-O3']),
              #extra_link_args=['-O3']),
              #extra_compile_args=['-O3', '-fopenmp'],
              #extra_link_args=['-O3','-fopenmp']),
]


setup(
    name = 'Fcodes',
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)

#ext_modules = cythonize(["*.pyx","*/*.pyx"]),

