"""Minimal script to reproduce our nasty reference counting bug.

The problem is related to https://github.com/ipython/ipython/issues/141

The original fix for that appeared to work, but John D. Hunter found a
matplotlib example which, when run twice in a row, would break.  The problem
were references held by open figures to internals of Tkinter.

This code reproduces the problem that John saw, without matplotlib.

This script is meant to be called by other parts of the test suite that call it
via %run as if it were executed interactively by the user. As of 2011-05-29,
test_run.py calls it.
"""

#-----------------------------------------------------------------------------
# Module imports
#-----------------------------------------------------------------------------

from IPython import get_ipython

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# This needs to be here because nose and other test runners will import
# this module. Importing this module has potential side effects that we
# want to prevent.
if __name__ == '__main__':

    ip = get_ipython()

    if not '_refbug_cache' in ip.user_ns:
        ip.user_ns['_refbug_cache'] = []


    aglobal = 'Hello'
    def f():
        return aglobal

    cache = ip.user_ns['_refbug_cache']
    cache.append(f)

    def call_f():
        for func in cache:
            print('lowercased:',func().lower())
