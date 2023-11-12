"""
Created on Mar 30, 2022 1:21:54 PM

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose


def _mover_confint(stat1, stat2, ci1, ci2, contrast="diff"):
    """

    References
    ----------

    .. [#] Krishnamoorthy, K., Jie Peng, and Dan Zhang. 2016. “Modified Large
       Sample Confidence Intervals for Poisson Distributions: Ratio, Weighted
       Average, and Product of Means.” Communications in Statistics - Theory
       and Methods 45 (1): 83–97. https://doi.org/10.1080/03610926.2013.821486.


    .. [#] Li, Yanhong, John J. Koval, Allan Donner, and G. Y. Zou. 2010.
       “Interval Estimation for the Area under the Receiver Operating
       Characteristic Curve When Data Are Subject to Error.” Statistics in
       Medicine 29 (24): 2521–31. https://doi.org/10.1002/sim.4015.

    .. [#] Zou, G. Y., and A. Donner. 2008. “Construction of Confidence Limits
       about Effect Measures: A General Approach.” Statistics in Medicine 27
       (10): 1693–1702. https://doi.org/10.1002/sim.3095.
    """

    if contrast == "diff":
        stat = stat1 - stat2
        low_half = np.sqrt((stat1 - ci1[0])**2 + (stat2 - ci2[1])**2)
        upp_half = np.sqrt((stat1 - ci1[1])**2 + (stat2 - ci2[0])**2)
        ci = (stat - low_half, stat + upp_half)

    elif contrast == "sum":
        stat = stat1 + stat2
        low_half = np.sqrt((stat1 - ci1[0])**2 + (stat2 - ci2[0])**2)
        upp_half = np.sqrt((stat1 - ci1[1])**2 + (stat2 - ci2[1])**2)
        ci = (stat - low_half, stat + upp_half)

    elif contrast == "ratio":
        # stat = stat1 / stat2
        prod = stat1 * stat2
        term1 = stat2**2 - (ci2[1] - stat2)**2
        term2 = stat2**2 - (ci2[0] - stat2)**2
        low_ = (prod -
                np.sqrt(prod**2 - term1 * (stat1**2 - (ci1[0] - stat1)**2))
                ) / term1
        upp_ = (prod +
                np.sqrt(prod**2 - term2 * (stat1**2 - (ci1[1] - stat1)**2))
                ) / term2

        # method 2 Li, Tang, Wong 2014
        low1, upp1 = ci1
        low2, upp2 = ci2
        term1 = upp2 * (2 * stat2 - upp2)
        term2 = low2 * (2 * stat2 - low2)
        low = (prod -
               np.sqrt(prod**2 - term1 * low1 * (2 * stat1 - low1))
               ) / term1
        upp = (prod +
               np.sqrt(prod**2 - term2 * upp1 * (2 * stat1 - upp1))
               ) / term2

        assert_allclose((low_, upp_), (low, upp), atol=1e-15, rtol=1e-10)

        ci = (low, upp)

    return ci


def _mover_confint_sum(stat, ci):

    stat_ = stat.sum(0)
    low_half = np.sqrt(np.sum((stat_ - ci[0])**2))
    upp_half = np.sqrt(np.sum((stat_ - ci[1])**2))
    ci = (stat - low_half, stat + upp_half)
    return ci
