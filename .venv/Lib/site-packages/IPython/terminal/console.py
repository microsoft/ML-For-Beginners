"""
Shim to maintain backwards compatibility with old IPython.terminal.console imports.
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import sys
from warnings import warn

from IPython.utils.shimmodule import ShimModule, ShimWarning

warn("The `IPython.terminal.console` package has been deprecated since IPython 4.0. "
     "You should import from jupyter_console instead.", ShimWarning)

# Unconditionally insert the shim into sys.modules so that further import calls
# trigger the custom attribute access above

sys.modules['IPython.terminal.console'] = ShimModule(
    src='IPython.terminal.console', mirror='jupyter_console')
