"""`uarray` provides functions for generating multimethods that dispatch to
multiple different backends

This should be imported, rather than `_uarray` so that an installed version could
be used instead, if available. This means that users can call
`uarray.set_backend` directly instead of going through SciPy.

"""


# Prefer an installed version of uarray, if available
try:
    import uarray as _uarray
except ImportError:
    _has_uarray = False
else:
    from scipy._lib._pep440 import Version as _Version

    _has_uarray = _Version(_uarray.__version__) >= _Version("0.8")
    del _uarray
    del _Version


if _has_uarray:
    from uarray import *  # noqa: F403
    from uarray import _Function
else:
    from ._uarray import *  # noqa: F403
    from ._uarray import _Function  # noqa: F401

del _has_uarray
