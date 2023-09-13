# This file is not meant for public use and will be removed in SciPy v2.0.0.


import warnings
from . import _biasedurn


__all__ = [  # noqa: F822
    '_PyFishersNCHypergeometric',
    '_PyWalleniusNCHypergeometric',
    '_PyStochasticLib3'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.stats.biasedurn is deprecated and has no attribute "
            f"{name}.")

    warnings.warn("the `scipy.stats.biasedurn` namespace is deprecated and "
                  "will be removed in SciPy v2.0.0.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_biasedurn, name)
