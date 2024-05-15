import os
from setuptools import find_packages, setup

gmri_directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

setup(name="gMRI",
      description="Greedy Minimal Rational Interpolation",
      author="Davide Pradovera",
      author_email="davidepradovera@gmail.com",
      version="1.0.2",
      license="GNU Library or Lesser General Public License (LGPL)",
      packages=find_packages(gmri_directory),
      zip_safe=False
      )
