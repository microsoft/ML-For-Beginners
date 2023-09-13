# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

import warnings
from . import _basic
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
                      psi, hankel1, hankel2, yv, kv)


__all__ = [  # noqa: F822
    'ai_zeros',
    'assoc_laguerre',
    'bei_zeros',
    'beip_zeros',
    'ber_zeros',
    'bernoulli',
    'berp_zeros',
    'bi_zeros',
    'clpmn',
    'comb',
    'digamma',
    'diric',
    'erf_zeros',
    'euler',
    'factorial',
    'factorial2',
    'factorialk',
    'fresnel_zeros',
    'fresnelc_zeros',
    'fresnels_zeros',
    'gamma',
    'h1vp',
    'h2vp',
    'hankel1',
    'hankel2',
    'iv',
    'ivp',
    'jn_zeros',
    'jnjnp_zeros',
    'jnp_zeros',
    'jnyn_zeros',
    'jv',
    'jvp',
    'kei_zeros',
    'keip_zeros',
    'kelvin_zeros',
    'ker_zeros',
    'kerp_zeros',
    'kv',
    'kvp',
    'lmbda',
    'lpmn',
    'lpn',
    'lqmn',
    'lqn',
    'mathieu_a',
    'mathieu_b',
    'mathieu_even_coef',
    'mathieu_odd_coef',
    'obl_cv_seq',
    'pbdn_seq',
    'pbdv_seq',
    'pbvv_seq',
    'perm',
    'polygamma',
    'pro_cv_seq',
    'psi',
    'riccati_jn',
    'riccati_yn',
    'sinc',
    'y0_zeros',
    'y1_zeros',
    'y1p_zeros',
    'yn_zeros',
    'ynp_zeros',
    'yv',
    'yvp',
    'zeta'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.special.basic is deprecated and has no attribute "
            f"{name}. Try looking in scipy.special instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.special` namespace, "
                  "the `scipy.special.basic` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_basic, name)
