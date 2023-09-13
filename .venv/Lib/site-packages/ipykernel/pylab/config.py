"""Configurable for configuring the IPython inline backend

This module does not import anything from matplotlib.
"""

import warnings

from matplotlib_inline.config import *  # analysis: ignore # noqa F401

warnings.warn(
    "`ipykernel.pylab.config` is deprecated, directly use `matplotlib_inline.config`",
    DeprecationWarning,
    stacklevel=2,
)
