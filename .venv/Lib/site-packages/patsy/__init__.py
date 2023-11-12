# This file is part of Patsy
# Copyright (C) 2011-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

"""patsy is a Python package for describing statistical models and building
design matrices. It is closely inspired by the 'formula' mini-language used in
R and S."""

import sys

from patsy.version import __version__

# Do this first, to make it easy to check for warnings while testing:
import os
if os.environ.get("PATSY_FORCE_NO_WARNINGS"):
    import warnings
    warnings.filterwarnings("error", module="^patsy")
    del warnings
del os

import patsy.origin

class PatsyError(Exception):
    """This is the main error type raised by Patsy functions.

    In addition to the usual Python exception features, you can pass a second
    argument to this function specifying the origin of the error; this is
    included in any error message, and used to help the user locate errors
    arising from malformed formulas. This second argument should be an
    :class:`Origin` object, or else an arbitrary object with a ``.origin``
    attribute. (If it is neither of these things, then it will simply be
    ignored.)

    For ordinary display to the user with default formatting, use
    ``str(exc)``. If you want to do something cleverer, you can use the
    ``.message`` and ``.origin`` attributes directly. (The latter may be
    None.)
    """
    def __init__(self, message, origin=None):
        Exception.__init__(self, message)
        self.message = message
        self.origin = None
        self.set_origin(origin)

    def __str__(self):
        if self.origin is None:
            return self.message
        else:
            return ("%s\n%s"
                    % (self.message, self.origin.caretize(indent=4)))

    def set_origin(self, origin):
        # This is useful to modify an exception to add origin information as
        # it "passes by", without losing traceback information. (In Python 3
        # we can use the built-in exception wrapping stuff, but it will be
        # some time before we can count on that...)
        if self.origin is None:
            if hasattr(origin, "origin"):
                origin = origin.origin
            if not isinstance(origin, patsy.origin.Origin):
                origin = None
            self.origin = origin

__all__ = ["PatsyError"]

# We make a rich API available for explicit use. To see what exactly is
# exported, check each module's __all__, or import this module and look at its
# __all__.

def _reexport(mod):
    __all__.extend(mod.__all__)
    for var in mod.__all__:
        globals()[var] = getattr(mod, var)

# This used to have less copy-paste, but explicit import statements make
# packaging tools like py2exe and py2app happier. Sigh.
import patsy.highlevel
_reexport(patsy.highlevel)

import patsy.build
_reexport(patsy.build)

import patsy.constraint
_reexport(patsy.constraint)

import patsy.contrasts
_reexport(patsy.contrasts)

import patsy.desc
_reexport(patsy.desc)

import patsy.design_info
_reexport(patsy.design_info)

import patsy.eval
_reexport(patsy.eval)

import patsy.origin
_reexport(patsy.origin)

import patsy.state
_reexport(patsy.state)

import patsy.user_util
_reexport(patsy.user_util)

import patsy.missing
_reexport(patsy.missing)

import patsy.splines
_reexport(patsy.splines)

import patsy.mgcv_cubic_splines
_reexport(patsy.mgcv_cubic_splines)

# XX FIXME: we aren't exporting any of the explicit parsing interface
# yet. Need to figure out how to do that.
