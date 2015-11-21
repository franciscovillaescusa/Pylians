from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("density_field_library.pyx")
)
