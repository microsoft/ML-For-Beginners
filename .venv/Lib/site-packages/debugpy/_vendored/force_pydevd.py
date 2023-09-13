# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

from importlib import import_module
import os
import warnings

from . import check_modules, prefix_matcher, preimport, vendored

# Ensure that pydevd is our vendored copy.
_unvendored, _ = check_modules('pydevd',
                               prefix_matcher('pydev', '_pydev'))
if _unvendored:
    _unvendored = sorted(_unvendored.values())
    msg = 'incompatible copy of pydevd already imported'
    # raise ImportError(msg)
    warnings.warn(msg + ':\n {}'.format('\n  '.join(_unvendored)))

# If debugpy logging is enabled, enable it for pydevd as well
if "DEBUGPY_LOG_DIR" in os.environ:
    os.environ[str("PYDEVD_DEBUG")] = str("True")
    os.environ[str("PYDEVD_DEBUG_FILE")] = os.environ["DEBUGPY_LOG_DIR"] + str("/debugpy.pydevd.log")

# Disable pydevd frame-eval optimizations only if unset, to allow opt-in.
if "PYDEVD_USE_FRAME_EVAL" not in os.environ:
    os.environ[str("PYDEVD_USE_FRAME_EVAL")] = str("NO")

# Constants must be set before importing any other pydevd module
# # due to heavy use of "from" in them.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    with vendored('pydevd'):
        pydevd_constants = import_module('_pydevd_bundle.pydevd_constants')
# We limit representation size in our representation provider when needed.
pydevd_constants.MAXIMUM_VARIABLE_REPRESENTATION_SIZE = 2 ** 32

# Now make sure all the top-level modules and packages in pydevd are
# loaded.  Any pydevd modules that aren't loaded at this point, will
# be loaded using their parent package's __path__ (i.e. one of the
# following).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    preimport('pydevd', [
        '_pydev_bundle',
        '_pydev_runfiles',
        '_pydevd_bundle',
        '_pydevd_frame_eval',
        'pydev_ipython',
        'pydevd_plugins',
        'pydevd',
    ])

# When pydevd is imported it sets the breakpoint behavior, but it needs to be
# overridden because by default pydevd will connect to the remote debugger using
# its own custom protocol rather than DAP.
import pydevd  # noqa
import debugpy  # noqa


def debugpy_breakpointhook():
    debugpy.breakpoint()


pydevd.install_breakpointhook(debugpy_breakpointhook)

# Ensure that pydevd uses JSON protocol
from _pydevd_bundle import pydevd_constants
from _pydevd_bundle import pydevd_defaults
pydevd_defaults.PydevdCustomization.DEFAULT_PROTOCOL = pydevd_constants.HTTP_JSON_PROTOCOL

# Enable some defaults related to debugpy such as sending a single notification when
# threads pause and stopping on any exception.
pydevd_defaults.PydevdCustomization.DEBUG_MODE = 'debugpy-dap'

# This is important when pydevd attaches automatically to a subprocess. In this case, we have to
# make sure that debugpy is properly put back in the game for users to be able to use it.
pydevd_defaults.PydevdCustomization.PREIMPORT = '%s;%s' % (
    os.path.dirname(os.path.dirname(debugpy.__file__)), 
    'debugpy._vendored.force_pydevd'
)
