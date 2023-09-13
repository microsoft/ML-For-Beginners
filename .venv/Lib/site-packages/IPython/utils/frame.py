# encoding: utf-8
"""
Utilities for working with stack frames.
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import sys

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

def extract_vars(*names,**kw):
    """Extract a set of variables by name from another frame.

    Parameters
    ----------
    *names : str
        One or more variable names which will be extracted from the caller's
        frame.
    **kw : integer, optional
        How many frames in the stack to walk when looking for your variables.
        The default is 0, which will use the frame where the call was made.

    Examples
    --------
    ::

        In [2]: def func(x):
           ...:     y = 1
           ...:     print(sorted(extract_vars('x','y').items()))
           ...:

        In [3]: func('hello')
        [('x', 'hello'), ('y', 1)]
    """

    depth = kw.get('depth',0)

    callerNS = sys._getframe(depth+1).f_locals
    return dict((k,callerNS[k]) for k in names)


def extract_vars_above(*names):
    """Extract a set of variables by name from another frame.

    Similar to extractVars(), but with a specified depth of 1, so that names
    are extracted exactly from above the caller.

    This is simply a convenience function so that the very common case (for us)
    of skipping exactly 1 frame doesn't have to construct a special dict for
    keyword passing."""

    callerNS = sys._getframe(2).f_locals
    return dict((k,callerNS[k]) for k in names)


def debugx(expr,pre_msg=''):
    """Print the value of an expression from the caller's frame.

    Takes an expression, evaluates it in the caller's frame and prints both
    the given expression and the resulting value (as well as a debug mark
    indicating the name of the calling function.  The input must be of a form
    suitable for eval().

    An optional message can be passed, which will be prepended to the printed
    expr->value pair."""

    cf = sys._getframe(1)
    print('[DBG:%s] %s%s -> %r' % (cf.f_code.co_name,pre_msg,expr,
                                   eval(expr,cf.f_globals,cf.f_locals)))


# deactivate it by uncommenting the following line, which makes it a no-op
#def debugx(expr,pre_msg=''): pass

def extract_module_locals(depth=0):
    """Returns (module, locals) of the function `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)
