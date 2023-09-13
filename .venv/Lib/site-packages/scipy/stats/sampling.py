"""
======================================================
Random Number Generators (:mod:`scipy.stats.sampling`)
======================================================

.. currentmodule:: scipy.stats.sampling

This module contains a collection of random number generators to sample
from univariate continuous and discrete distributions. It uses the
implementation of a C library called "UNU.RAN".

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
"""
from ._unuran.unuran_wrapper import (  # noqa: F401
    TransformedDensityRejection,
    DiscreteAliasUrn,
    DiscreteGuideTable,
    NumericalInversePolynomial,
    NumericalInverseHermite,
    SimpleRatioUniforms,
    UNURANError
)
