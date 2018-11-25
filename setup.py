#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import setup

NAME = 'rfim2d'
VERSION = '0.1'
DESCRIPTION = 'Performs scaling collapses of 2D NE-RFIM simulation data'
URL = 'https://github.com/lxh3/rfim2d'
AUTHOR = 'L. X. Hayden'
EMAIL = 'lxh3@cornell.edu'
REQUIRES_PYTHON = '>=3.6.0'

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(name = NAME,
      version = VERSION,
      description = DESCRIPTION,
      long_description = long_description,
      url = URL,
      author = AUTHOR,
      author_email = EMAIL,
      python_requires=REQUIRES_PYTHON,
      license='MIT',
      packages=['rfim2d'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      package_data = {'' : ['*.pkl.gz']},
      zip_safe=False)
