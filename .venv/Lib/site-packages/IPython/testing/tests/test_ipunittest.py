"""Tests for IPython's test support utilities.

These are decorators that allow standalone functions and docstrings to be seen
as tests by unittest, replicating some of nose's functionality.  Additionally,
IPython-syntax docstrings can be auto-converted to '>>>' so that ipython
sessions can be copy-pasted as tests.

This file can be run as a script, and it will call unittest.main().  We must
check that it works with unittest as well as with nose...


Notes:

- Using nosetests --with-doctest --doctest-tests testfile.py
  will find docstrings as tests wherever they are, even in methods.  But
  if we use ipython syntax in the docstrings, they must be decorated with
  @ipdocstring.  This is OK for test-only code, but not for user-facing
  docstrings where we want to keep the ipython syntax.

- Using nosetests --with-doctest file.py
  also finds doctests if the file name doesn't have 'test' in it, because it is
  treated like a normal module.  But if nose treats the file like a test file,
  then for normal classes to be doctested the extra --doctest-tests is
  necessary.

- running this script with python (it has a __main__ section at the end) misses
  one docstring test, the one embedded in the Foo object method.  Since our
  approach relies on using decorators that create standalone TestCase
  instances, it can only be used for functions, not for methods of objects.
Authors
-------

- Fernando Perez <Fernando.Perez@berkeley.edu>
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2009-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from IPython.testing.ipunittest import ipdoctest, ipdocstring

#-----------------------------------------------------------------------------
# Test classes and functions
#-----------------------------------------------------------------------------
@ipdoctest
def simple_dt():
    """
    >>> print(1+1)
    2
    """


@ipdoctest
def ipdt_flush():
    """
In [20]: print(1)
1

In [26]: for i in range(4):
   ....:     print(i)
   ....:     
   ....: 
0
1
2
3

In [27]: 3+4
Out[27]: 7
"""


@ipdoctest
def ipdt_indented_test():
    """
    In [20]: print(1)
    1

    In [26]: for i in range(4):
       ....:     print(i)
       ....:     
       ....: 
    0
    1
    2
    3

    In [27]: 3+4
    Out[27]: 7
    """


class Foo(object):
    """For methods, the normal decorator doesn't work.

    But rewriting the docstring with ip2py does, *but only if using nose
    --with-doctest*.  Do we want to have that as a dependency?
    """

    @ipdocstring
    def ipdt_method(self):
        """
        In [20]: print(1)
        1

        In [26]: for i in range(4):
           ....:     print(i)
           ....:     
           ....: 
        0
        1
        2
        3

        In [27]: 3+4
        Out[27]: 7
        """

    def normaldt_method(self):
        """
        >>> print(1+1)
        2
        """
