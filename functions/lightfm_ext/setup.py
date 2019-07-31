from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('lightfm_ext.fit_warp_ext', ['lightfm_ext/fit_warp_ext.pyx'])]

setup(name='lightfm_ext',
      version='0.1',
      description="An extension of Maceij Kula's LightFM package",
      author='Sean Tibbitts',
      author_email='sean.tibbitts@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      ext_modules=cythonize(extensions))