# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

"""An implementation of the Debug Adapter Protocol (DAP) for Python.

https://microsoft.github.io/debug-adapter-protocol/
"""

# debugpy stable public API consists solely of members of this module that are
# enumerated below.
__all__ = [  # noqa
    "__version__",
    "breakpoint",
    "configure",
    "connect",
    "debug_this_thread",
    "is_client_connected",
    "listen",
    "log_to",
    "trace_this_thread",
    "wait_for_client",
]

import sys

assert sys.version_info >= (3, 7), (
    "Python 3.6 and below is not supported by this version of debugpy; "
    "use debugpy 1.5.1 or earlier."
)


# Actual definitions are in a separate file to work around parsing issues causing
# SyntaxError on Python 2 and preventing the above version check from executing.
from debugpy.public_api import *  # noqa
from debugpy.public_api import __version__

del sys
