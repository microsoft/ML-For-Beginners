#!/usr/bin/env python
"""A Nose plugin to support IPython doctests.
"""

from setuptools import setup

setup(name='IPython doctest plugin',
      version='0.1',
      author='The IPython Team',
      description = 'Nose plugin to load IPython-extended doctests',
      license = 'LGPL',
      py_modules = ['ipdoctest'],
      entry_points = {
        'nose.plugins.0.10': ['ipdoctest = ipdoctest:IPythonDoctest',
                              'extdoctest = ipdoctest:ExtensionDoctest',
                              ],
        },
      )
