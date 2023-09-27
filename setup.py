from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

print(numpy.get_include())
setup(
    name='adversarial_loss_gradient',
    ext_modules=cythonize(
        Extension(
            "adversarial_loss_gradient",
            sources=["adversarial_loss_gradient.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ),
    install_requires=["numpy"]
)
# usage python setup.py build_ext --inplace  


# setup(
#     name='sklearn_loss',
#     ext_modules=cythonize(
#         Extension(
#             "sklearn_loss",
#             sources=["sklearn_loss.pyx"],
#             include_dirs=[numpy.get_include()]
#         )
#     ),
#     install_requires=["numpy"]
# )