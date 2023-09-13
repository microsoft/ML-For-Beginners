"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck
#
# License: BSD 3 clause

import sys
from importlib import resources

import numpy as np
import scipy
import scipy.stats
import threadpoolctl

import sklearn

from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated

np_version = parse_version(np.__version__)
sp_version = parse_version(scipy.__version__)
sp_base_version = parse_version(sp_version.base_version)


try:
    from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2
except ImportError:  # SciPy < 1.8
    from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1  # type: ignore  # noqa


def _object_dtype_isnan(X):
    return X != X


# Rename the `method` kwarg to `interpolation` for NumPy < 1.22, because
# `interpolation` kwarg was deprecated in favor of `method` in NumPy >= 1.22.
def _percentile(a, q, *, method="linear", **kwargs):
    return np.percentile(a, q, interpolation=method, **kwargs)


if np_version < parse_version("1.22"):
    percentile = _percentile
else:  # >= 1.22
    from numpy import percentile  # type: ignore  # noqa


# compatibility fix for threadpoolctl >= 3.0.0
# since version 3 it's possible to setup a global threadpool controller to avoid
# looping through all loaded shared libraries each time.
# the global controller is created during the first call to threadpoolctl.
def _get_threadpool_controller():
    if not hasattr(threadpoolctl, "ThreadpoolController"):
        return None

    if not hasattr(sklearn, "_sklearn_threadpool_controller"):
        sklearn._sklearn_threadpool_controller = threadpoolctl.ThreadpoolController()

    return sklearn._sklearn_threadpool_controller


def threadpool_limits(limits=None, user_api=None):
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.limit(limits=limits, user_api=user_api)
    else:
        return threadpoolctl.threadpool_limits(limits=limits, user_api=user_api)


threadpool_limits.__doc__ = threadpoolctl.threadpool_limits.__doc__


def threadpool_info():
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.info()
    else:
        return threadpoolctl.threadpool_info()


threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__


@deprecated(
    "The function `delayed` has been moved from `sklearn.utils.fixes` to "
    "`sklearn.utils.parallel`. This import path will be removed in 1.5."
)
def delayed(function):
    from sklearn.utils.parallel import delayed

    return delayed(function)


# TODO: Remove when SciPy 1.11 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        mode = scipy.stats.mode(a, axis=axis, keepdims=True)
        if sp_version >= parse_version("1.10.999"):
            # scipy.stats.mode has changed returned array shape with axis=None
            # and keepdims=True, see https://github.com/scipy/scipy/pull/17561
            if axis is None:
                mode = np.ravel(mode)
        return mode
    return scipy.stats.mode(a, axis=axis)


###############################################################################
# Backport of Python 3.9's importlib.resources
# TODO: Remove when Python 3.9 is the minimum supported version


def _open_text(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(data_module).joinpath(data_file_name).open("r")
    else:
        return resources.open_text(data_module, data_file_name)


def _open_binary(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(data_module).joinpath(data_file_name).open("rb")
    else:
        return resources.open_binary(data_module, data_file_name)


def _read_text(descr_module, descr_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(descr_module).joinpath(descr_file_name).read_text()
    else:
        return resources.read_text(descr_module, descr_file_name)


def _path(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.as_file(resources.files(data_module).joinpath(data_file_name))
    else:
        return resources.path(data_module, data_file_name)


def _is_resource(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(data_module).joinpath(data_file_name).is_file()
    else:
        return resources.is_resource(data_module, data_file_name)


def _contents(data_module):
    if sys.version_info >= (3, 9):
        return (
            resource.name
            for resource in resources.files(data_module).iterdir()
            if resource.is_file()
        )
    else:
        return resources.contents(data_module)
