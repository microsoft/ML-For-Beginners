"""
    getargspec excerpted from:

    sphinx.util.inspect
    ~~~~~~~~~~~~~~~~~~~
    Helpers for inspecting Python modules.
    :copyright: Copyright 2007-2015 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import inspect
from functools import partial

# Unmodified from sphinx below this line


def getargspec(func):
    """Like inspect.getargspec but supports functools.partial as well."""
    if inspect.ismethod(func):
        func = func.__func__
    if type(func) is partial:
        orig_func = func.func
        argspec = getargspec(orig_func)
        args = list(argspec[0])
        defaults = list(argspec[3] or ())
        kwoargs = list(argspec[4])
        kwodefs = dict(argspec[5] or {})
        if func.args:
            args = args[len(func.args) :]
        for arg in func.keywords or ():
            try:
                i = args.index(arg) - len(args)
                del args[i]
                try:
                    del defaults[i]
                except IndexError:
                    pass
            except ValueError:  # must be a kwonly arg
                i = kwoargs.index(arg)
                del kwoargs[i]
                del kwodefs[arg]
        return inspect.FullArgSpec(
            args, argspec[1], argspec[2], tuple(defaults), kwoargs, kwodefs, argspec[6]
        )
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    if not inspect.isfunction(func):
        raise TypeError("%r is not a Python function" % func)
    return inspect.getfullargspec(func)
