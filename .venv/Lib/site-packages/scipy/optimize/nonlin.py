# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

import warnings
from . import _nonlin


__all__ = [  # noqa: F822
    'Anderson',
    'BroydenFirst',
    'BroydenSecond',
    'DiagBroyden',
    'ExcitingMixing',
    'GenericBroyden',
    'InverseJacobian',
    'Jacobian',
    'KrylovJacobian',
    'LinAlgError',
    'LinearMixing',
    'LowRankMatrix',
    'NoConvergence',
    'TerminationCondition',
    'anderson',
    'asarray',
    'asjacobian',
    'broyden1',
    'broyden2',
    'diagbroyden',
    'dot',
    'excitingmixing',
    'get_blas_funcs',
    'inspect',
    'inv',
    'linearmixing',
    'maxnorm',
    'newton_krylov',
    'nonlin_solve',
    'norm',
    'qr',
    'scalar_search_armijo',
    'scalar_search_wolfe1',
    'scipy',
    'solve',
    'svd',
    'sys',
    'vdot',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.optimize.nonlin is deprecated and has no attribute "
            f"{name}. Try looking in scipy.optimize instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.optimize` namespace, "
                  "the `scipy.optimize.nonlin` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_nonlin, name)
