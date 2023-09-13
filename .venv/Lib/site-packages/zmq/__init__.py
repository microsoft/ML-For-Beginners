"""Python bindings for 0MQ."""


# start delvewheel patch
def _delvewheel_init_patch_0_0_22():
    import os
    import sys
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pyzmq.libs'))
    if sys.version_info[:2] >= (3, 8):
        conda_workaround = os.path.exists(os.path.join(sys.base_prefix, 'conda-meta')) and (sys.version_info[:3] < (3, 8, 13) or (3, 9, 0) <= sys.version_info[:3] < (3, 9, 9))
        if conda_workaround:
            # backup the state of the environment variable CONDA_DLL_SEARCH_MODIFICATION_ENABLE
            conda_dll_search_modification_enable = os.environ.get('CONDA_DLL_SEARCH_MODIFICATION_ENABLE')
            os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'
        os.add_dll_directory(libs_dir)
        if conda_workaround:
            # restore the state of the environment variable CONDA_DLL_SEARCH_MODIFICATION_ENABLE
            if conda_dll_search_modification_enable is None:
                os.environ.pop('CONDA_DLL_SEARCH_MODIFICATION_ENABLE', None)
            else:
                os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = conda_dll_search_modification_enable
    else:
        from ctypes import WinDLL
        with open(os.path.join(libs_dir, '.load-order-pyzmq-25.1.1')) as file:
            load_order = file.read().split()
        for lib in load_order:
            WinDLL(os.path.join(libs_dir, lib))


_delvewheel_init_patch_0_0_22()
del _delvewheel_init_patch_0_0_22
# end delvewheel patch



# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

# load bundled libzmq, if there is one:

import os
import sys
from contextlib import contextmanager


def _load_libzmq():
    """load bundled libzmq if there is one"""
    import platform

    dlopen = hasattr(sys, 'getdlopenflags')  # unix-only
    # RTLD flags are added to os in Python 3
    # get values from os because ctypes values are WRONG on pypy
    PYPY = platform.python_implementation().lower() == 'pypy'

    if dlopen:
        import ctypes

        dlflags = sys.getdlopenflags()
        # set RTLD_GLOBAL, unset RTLD_LOCAL
        flags = ctypes.RTLD_GLOBAL | dlflags
        # ctypes.RTLD_LOCAL is 0 on pypy, which is *wrong*
        flags &= ~getattr(os, 'RTLD_LOCAL', 4)
        # pypy on darwin needs RTLD_LAZY for some reason
        if PYPY and sys.platform == 'darwin':
            flags |= getattr(os, 'RTLD_LAZY', 1)
            flags &= ~getattr(os, 'RTLD_NOW', 2)
        sys.setdlopenflags(flags)
    try:
        from . import libzmq
    except ImportError:
        # raise on failure to load if libzmq is present
        from importlib.util import find_spec

        if find_spec(".libzmq", "zmq"):
            # found libzmq, but failed to load it!
            # raise instead of silently moving on
            raise
    else:
        # store libzmq as zmq._libzmq for backward-compat
        globals()['_libzmq'] = libzmq
        if PYPY:
            # should already have been imported above, so reimporting is as cheap as checking
            import ctypes

            # some versions of pypy (5.3 < ? < 5.8) needs explicit CDLL load for some reason,
            # otherwise symbols won't be globally available
            # do this unconditionally because it should be harmless (?)
            ctypes.CDLL(libzmq.__file__, ctypes.RTLD_GLOBAL)
    finally:
        if dlopen:
            sys.setdlopenflags(dlflags)


_load_libzmq()


@contextmanager
def _libs_on_path():
    """context manager for libs directory on $PATH

    Works around mysterious issue where os.add_dll_directory
    does not resolve imports (conda-forge Python >= 3.8)
    """

    if not sys.platform.startswith("win"):
        yield
        return

    libs_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "pyzmq.libs",
        )
    )
    if not os.path.exists(libs_dir):
        # no bundled libs
        yield
        return

    path_before = os.environ.get("PATH")
    try:
        os.environ["PATH"] = os.pathsep.join([path_before or "", libs_dir])
        yield
    finally:
        if path_before is None:
            os.environ.pop("PATH")
        else:
            os.environ["PATH"] = path_before


# zmq top-level imports

# workaround for Windows
with _libs_on_path():
    from zmq import backend

from . import constants  # noqa
from .constants import *  # noqa
from zmq.backend import *  # noqa
from zmq import sugar
from zmq.sugar import *  # noqa


def get_includes():
    """Return a list of directories to include for linking against pyzmq with cython."""
    from os.path import abspath, dirname, exists, join, pardir

    base = dirname(__file__)
    parent = abspath(join(base, pardir))
    includes = [parent] + [join(parent, base, subdir) for subdir in ('utils',)]
    if exists(join(parent, base, 'include')):
        includes.append(join(parent, base, 'include'))
    return includes


def get_library_dirs():
    """Return a list of directories used to link against pyzmq's bundled libzmq."""
    from os.path import abspath, dirname, join, pardir

    base = dirname(__file__)
    parent = abspath(join(base, pardir))
    return [join(parent, base)]


COPY_THRESHOLD = 65536
DRAFT_API = backend.has("draft")

__all__ = (
    [
        'get_includes',
        'COPY_THRESHOLD',
        'DRAFT_API',
    ]
    + constants.__all__
    + sugar.__all__
    + backend.__all__
)