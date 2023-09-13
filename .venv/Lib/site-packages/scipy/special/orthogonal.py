# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

import warnings
from . import _orthogonal


_polyfuns = ['legendre', 'chebyt', 'chebyu', 'chebyc', 'chebys',
             'jacobi', 'laguerre', 'genlaguerre', 'hermite',
             'hermitenorm', 'gegenbauer', 'sh_legendre', 'sh_chebyt',
             'sh_chebyu', 'sh_jacobi']

# Correspondence between new and old names of root functions
_rootfuns_map = {'roots_legendre': 'p_roots',
               'roots_chebyt': 't_roots',
               'roots_chebyu': 'u_roots',
               'roots_chebyc': 'c_roots',
               'roots_chebys': 's_roots',
               'roots_jacobi': 'j_roots',
               'roots_laguerre': 'l_roots',
               'roots_genlaguerre': 'la_roots',
               'roots_hermite': 'h_roots',
               'roots_hermitenorm': 'he_roots',
               'roots_gegenbauer': 'cg_roots',
               'roots_sh_legendre': 'ps_roots',
               'roots_sh_chebyt': 'ts_roots',
               'roots_sh_chebyu': 'us_roots',
               'roots_sh_jacobi': 'js_roots'}


__all__ = _polyfuns + list(_rootfuns_map.keys()) + [  # noqa: F822
    'exp', 'inf', 'floor', 'around', 'hstack', 'arange',
    'linalg', 'airy', 'orthopoly1d', 'newfun',
    'oldfun', 'p_roots', 't_roots', 'u_roots', 'c_roots', 's_roots',
    'j_roots', 'l_roots', 'la_roots', 'h_roots', 'he_roots', 'cg_roots',
    'ps_roots', 'ts_roots', 'us_roots', 'js_roots'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.special.orthogonal is deprecated and has no attribute "
            f"{name}. Try looking in scipy.special instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.special` namespace, "
                  "the `scipy.special.orthogonal` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_orthogonal, name)
