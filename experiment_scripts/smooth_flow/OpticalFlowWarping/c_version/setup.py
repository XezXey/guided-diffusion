from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonize and compiled
ext = Extension(name="optical_flow_warp", sources=["optical_flow_warp.pyx"])
setup(ext_modules=cythonize(ext))
