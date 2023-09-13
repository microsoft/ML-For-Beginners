"""Simple example using doctests.

This file just contains doctests both using plain python and IPython prompts.
All tests should be loaded by Pytest.
"""

def pyfunc():
    """Some pure python tests...

    >>> pyfunc()
    'pyfunc'

    >>> import os

    >>> 2+3
    5

    >>> for i in range(3):
    ...     print(i, end=' ')
    ...     print(i+1, end=' ')
    ...
    0 1 1 2 2 3 
    """
    return 'pyfunc'


def ipyfunc():
    """Some IPython tests...

    In [1]: ipyfunc()
    Out[1]: 'ipyfunc'

    In [2]: import os

    In [3]: 2+3
    Out[3]: 5

    In [4]: for i in range(3):
       ...:     print(i, end=' ')
       ...:     print(i+1, end=' ')
       ...:
    Out[4]: 0 1 1 2 2 3
    """
    return "ipyfunc"
