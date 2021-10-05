

from setuptools import setup 

__version__ = "1.5.0"
__author__ = "Khaled Nakhleh"

description = "implementation for NeurWIN algorithm with its training and testing settings.\
The repository also contains benchmark code for evaluation purposes."

requirements = [
 "numpy==1.18.5",
 "matplotlib==3.3.0",
 "pandas==1.0.5",
 "torch==1.5.1",
 "graphviz==0.14.1",
 "gym==0.17.2",
 "scipy==1.4.1",
 "tensorflow==2.3.0"
]

setup(
      name = "NeurWIN",
      description = description,
      install_requires = requirements,
      version = __version__,
      author = __author__
      )