"""
Shim to maintain backwards compatibility with old IPython.consoleapp imports.
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from warnings import warn

warn("The `IPython.consoleapp` package has been deprecated since IPython 4.0."
     "You should import from jupyter_client.consoleapp instead.", stacklevel=2)

from jupyter_client.consoleapp import *
