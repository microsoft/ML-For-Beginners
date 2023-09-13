# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

import warnings
from . import _mstats_extras


__all__ = [  # noqa: F822
    'compare_medians_ms',
    'hdquantiles', 'hdmedian', 'hdquantiles_sd',
    'idealfourths',
    'median_cihs','mjci','mquantiles_cimj',
    'rsh',
    'trimmed_mean_ci', 'float_', 'int_', 'ma', 'MaskedArray', 'mstats',
    'norm', 'beta', 't', 'binom'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.stats.mstats_extras is deprecated and has no attribute "
            f"{name}. Try looking in scipy.stats instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.stats` namespace, "
                  "the `scipy.stats.mstats_extras` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_mstats_extras, name)
