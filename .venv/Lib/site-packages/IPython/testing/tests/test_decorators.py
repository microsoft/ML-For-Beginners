"""Tests for the decorators we've created for IPython.
"""

# Module imports
# Std lib
import inspect
import sys

# Our own
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest

#-----------------------------------------------------------------------------
# Utilities

# Note: copied from OInspect, kept here so the testing stuff doesn't create
# circular dependencies and is easier to reuse.
def getargspec(obj):
    """Get the names and default values of a function's arguments.

    A tuple of four things is returned: (args, varargs, varkw, defaults).
    'args' is a list of the argument names (it may contain nested lists).
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    'defaults' is an n-tuple of the default values of the last n arguments.

    Modified version of inspect.getargspec from the Python Standard
    Library."""

    if inspect.isfunction(obj):
        func_obj = obj
    elif inspect.ismethod(obj):
        func_obj = obj.__func__
    else:
        raise TypeError('arg is not a Python function')
    args, varargs, varkw = inspect.getargs(func_obj.__code__)
    return args, varargs, varkw, func_obj.__defaults__

#-----------------------------------------------------------------------------
# Testing functions

@dec.as_unittest
def trivial():
    """A trivial test"""
    pass


@dec.skip()
def test_deliberately_broken():
    """A deliberately broken test - we want to skip this one."""
    1/0

@dec.skip('Testing the skip decorator')
def test_deliberately_broken2():
    """Another deliberately broken test - we want to skip this one."""
    1/0


# Verify that we can correctly skip the doctest for a function at will, but
# that the docstring itself is NOT destroyed by the decorator.
@skip_doctest
def doctest_bad(x,y=1,**k):
    """A function whose doctest we need to skip.

    >>> 1+1
    3
    """
    print('x:',x)
    print('y:',y)
    print('k:',k)


def call_doctest_bad():
    """Check that we can still call the decorated functions.
    
    >>> doctest_bad(3,y=4)
    x: 3
    y: 4
    k: {}
    """
    pass


def test_skip_dt_decorator():
    """Doctest-skipping decorator should preserve the docstring.
    """
    # Careful: 'check' must be a *verbatim* copy of the doctest_bad docstring!
    check = """A function whose doctest we need to skip.

    >>> 1+1
    3
    """
    # Fetch the docstring from doctest_bad after decoration.
    val = doctest_bad.__doc__
    
    assert check == val, "doctest_bad docstrings don't match"


# Doctest skipping should work for class methods too
class FooClass(object):
    """FooClass

    Example:

    >>> 1+1
    2
    """

    @skip_doctest
    def __init__(self,x):
        """Make a FooClass.

        Example:

        >>> f = FooClass(3)
        junk
        """
        print('Making a FooClass.')
        self.x = x
        
    @skip_doctest
    def bar(self,y):
        """Example:

        >>> ff = FooClass(3)
        >>> ff.bar(0)
        boom!
        >>> 1/0
        bam!
        """
        return 1/y

    def baz(self,y):
        """Example:

        >>> ff2 = FooClass(3)
        Making a FooClass.
        >>> ff2.baz(3)
        True
        """
        return self.x==y


def test_skip_dt_decorator2():
    """Doctest-skipping decorator should preserve function signature.
    """
    # Hardcoded correct answer
    dtargs = (['x', 'y'], None, 'k', (1,))
    # Introspect out the value
    dtargsr = getargspec(doctest_bad)
    assert dtargsr==dtargs, \
           "Incorrectly reconstructed args for doctest_bad: %s" % (dtargsr,)


@dec.skip_linux
def test_linux():
    assert sys.platform.startswith("linux") is False, "This test can't run under linux"


@dec.skip_win32
def test_win32():
    assert sys.platform != "win32", "This test can't run under windows"


@dec.skip_osx
def test_osx():
    assert sys.platform != "darwin", "This test can't run under osx"
