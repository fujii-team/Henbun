#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re
import os
import sys

# load version form _version.py
VERSIONFILE = "Henbun/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='Henbun',
      version=verstr,
      author="Keisuke Fujii",
      author_email="fujii@me.kyoto-u.ac.jp",
      description=("Variational inference for big-data."),
      license="BSD 3-clause",
      keywords="machine-learning gaussian-processes kernels",
      url="http://github.com/fujii-team/Henbun",
      include_package_data=True,
      ext_modules=[],
      packages=["Henbun"],
      package_dir={'Henbun': 'Henbun'},
      py_modules=['Henbun.__init__'],
      test_suite='testing',
      #install_requires=['numpy>=1.9', 'scipy>=0.16', 'tensorflow>=0.9', 'GPflow>=0.3.0'],
      install_requires='''
                numpy>=1.9
                scipy>=0.16
                tensorflow>=0.11.0rc1''',
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
