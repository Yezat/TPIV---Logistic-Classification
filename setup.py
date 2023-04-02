from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

print(numpy.get_include())
setup(
    name='sklearn_loss',
    ext_modules=cythonize(
        Extension(
            "xyz",
            sources=["sklearn_loss.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ),
    install_requires=["numpy"]
)
# usage python setup.py build_ext --inplace  