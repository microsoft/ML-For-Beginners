# -*- coding: utf-8 -*-
"""
Package with factor rotation algorithms.

This file contains a Python version of the gradient projection rotation
algorithms (GPA) developed by Bernaards, C.A. and Jennrich, R.I.
The code is based on the Matlab version of the code developed Bernaards, C.A.
and Jennrich, R.I. and is ported and made available with permission of the
authors.

Additionally, several analytic rotation methods are implemented.

References
----------
[1] Bernaards, C.A. and Jennrich, R.I. (2005) Gradient Projection Algorithms and Software for Arbitrary Rotation Criteria in Factor Analysis. Educational and Psychological Measurement, 65 (5), 676-696.

[2] Jennrich, R.I. (2001). A simple general procedure for orthogonal rotation. Psychometrika, 66, 289-306.

[3] Jennrich, R.I. (2002). A simple general method for oblique rotation. Psychometrika, 67, 7-19.

[4] http://www.stat.ucla.edu/research/gpa/matlab.net

[5] http://www.stat.ucla.edu/research/gpa/GPderfree.txt
"""
from ._wrappers import rotate_factors

from ._analytic_rotation import target_rotation, procrustes, promax
from statsmodels.tools._testing import PytestTester

__all__ = ['rotate_factors', 'target_rotation', 'procrustes', 'promax',
           'test']

test = PytestTester()
