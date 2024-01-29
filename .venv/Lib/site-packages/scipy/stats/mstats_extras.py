# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'compare_medians_ms',
    'hdquantiles', 'hdmedian', 'hdquantiles_sd',
    'idealfourths',
    'median_cihs','mjci','mquantiles_cimj',
    'rsh',
    'trimmed_mean_ci', 'ma', 'MaskedArray', 'mstats',
    'norm', 'beta', 't', 'binom'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mstats_extras",
                                   private_modules=["_mstats_extras"], all=__all__,
                                   attribute=name, correct_module="mstats")
