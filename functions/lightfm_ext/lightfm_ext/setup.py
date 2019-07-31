from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
      ext_modules=cythonize(
            Extension('lightfm_ext.fit_warp_ext', ["fit_warp_ext.pyx"], include_dirs=['.','..'])
      )
)