# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.

import warnings
from . import _qhull


__all__ = [  # noqa: F822
    'ConvexHull',
    'Delaunay',
    'HalfspaceIntersection',
    'QhullError',
    'Voronoi',
    'os',
    'sys',
    'tempfile',
    'threading',
    'tsearch',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.spatial.qhull is deprecated and has no attribute "
            f"{name}. Try looking in scipy.spatial instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.spatial` namespace, "
                  "the `scipy.spatial.qhull` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_qhull, name)
