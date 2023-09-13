# encoding: utf-8
"""
Utilities for version comparison

It is a bit ridiculous that we need these.
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2013  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

from warnings import warn

warn(
    "The `IPython.utils.version` module has been deprecated since IPython 8.0.",
    DeprecationWarning,
)


def check_version(v, check):
    """check version string v >= check

    If dev/prerelease tags result in TypeError for string-number comparison,
    it is assumed that the dependency is satisfied.
    Users on dev branches are responsible for keeping their own packages up to date.
    """
    warn(
        "`check_version` function is deprecated as of IPython 8.0"
        "and will be removed in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )

    from distutils.version import LooseVersion

    try:
        return LooseVersion(v) >= LooseVersion(check)
    except TypeError:
        return True

