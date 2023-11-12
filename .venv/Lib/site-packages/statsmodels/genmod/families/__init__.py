"""
This module contains the one-parameter exponential families used
for fitting GLMs and GAMs.

These families are described in

   P. McCullagh and J. A. Nelder.  "Generalized linear models."
   Monographs on Statistics and Applied Probability.
   Chapman & Hall, London, 1983.

"""

from statsmodels.genmod.families import links
from .family import Gaussian, Family, Poisson, Gamma, \
    InverseGaussian, Binomial, NegativeBinomial, Tweedie
from statsmodels.tools._testing import PytestTester

__all__ = ['test', 'links', 'Family', 'Gamma', 'Gaussian', 'Poisson',
           'InverseGaussian', 'Binomial', 'NegativeBinomial', 'Tweedie']

test = PytestTester()
