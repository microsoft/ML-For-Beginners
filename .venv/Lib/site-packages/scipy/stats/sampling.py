"""
======================================================
Random Number Generators (:mod:`scipy.stats.sampling`)
======================================================

.. currentmodule:: scipy.stats.sampling

This module contains a collection of random number generators to sample
from univariate continuous and discrete distributions. It uses the
implementation of a C library called "UNU.RAN". The only exception is
RatioUniforms, which is a pure Python implementation of the
Ratio-of-Uniforms method.

Generators Wrapped
==================

For continuous distributions
----------------------------

.. autosummary::
   :toctree: generated/

   NumericalInverseHermite
   NumericalInversePolynomial
   TransformedDensityRejection
   SimpleRatioUniforms
   RatioUniforms

For discrete distributions
--------------------------

.. autosummary::
   :toctree: generated/

   DiscreteAliasUrn
   DiscreteGuideTable

Warnings / Errors used in :mod:`scipy.stats.sampling`
-----------------------------------------------------

.. autosummary::
   :toctree: generated/

   UNURANError


Generators for pre-defined distributions
========================================

To easily apply the above methods for some of the continuous distributions
in :mod:`scipy.stats`, the following functionality can be used:

.. autosummary::
   :toctree: generated/

   FastGeneratorInversion

"""
from ._sampling import FastGeneratorInversion, RatioUniforms  # noqa: F401
from ._unuran.unuran_wrapper import (  # noqa: F401
   TransformedDensityRejection,
   DiscreteAliasUrn,
   DiscreteGuideTable,
   NumericalInversePolynomial,
   NumericalInverseHermite,
   SimpleRatioUniforms,
   UNURANError
)
