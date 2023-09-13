"""Simple example using doctests.

This file just contains doctests both using plain python and IPython prompts.
All tests should be loaded by nose.
"""

import os


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

def ipfunc():
    """Some ipython tests...

    In [1]: import os

    In [3]: 2+3
    Out[3]: 5

    In [26]: for i in range(3):
       ....:     print(i, end=' ')
       ....:     print(i+1, end=' ')
       ....:
    0 1 1 2 2 3


    It's OK to use '_' for the last result, but do NOT try to use IPython's
    numbered history of _NN outputs, since those won't exist under the
    doctest environment:

    In [7]: 'hi'
    Out[7]: 'hi'

    In [8]: print(repr(_))
    'hi'

    In [7]: 3+4
    Out[7]: 7

    In [8]: _+3
    Out[8]: 10

    In [9]: ipfunc()
    Out[9]: 'ipfunc'
    """
    return "ipfunc"


def ipos():
    """Examples that access the operating system work:

    In [1]: !echo hello
    hello

    In [2]: !echo hello > /tmp/foo_iptest

    In [3]: !cat /tmp/foo_iptest
    hello

    In [4]: rm -f /tmp/foo_iptest
    """
    pass


ipos.__skip_doctest__ = os.name == "nt"


def ranfunc():
    """A function with some random output.

       Normal examples are verified as usual:
       >>> 1+3
       4

       But if you put '# random' in the output, it is ignored:
       >>> 1+3
       junk goes here...  # random

       >>> 1+2
       again,  anything goes #random
       if multiline, the random mark is only needed once.

       >>> 1+2
       You can also put the random marker at the end:
       # random

       >>> 1+2
       # random
       .. or at the beginning.

       More correct input is properly verified:
       >>> ranfunc()
       'ranfunc'
    """
    return 'ranfunc'


def random_all():
    """A function where we ignore the output of ALL examples.

    Examples:

      # all-random

      This mark tells the testing machinery that all subsequent examples should
      be treated as random (ignoring their output).  They are still executed,
      so if a they raise an error, it will be detected as such, but their
      output is completely ignored.

      >>> 1+3
      junk goes here...

      >>> 1+3
      klasdfj;

      >>> 1+2
      again,  anything goes
      blah...
    """
    pass

def iprand():
    """Some ipython tests with random output.

    In [7]: 3+4
    Out[7]: 7

    In [8]: print('hello')
    world  # random

    In [9]: iprand()
    Out[9]: 'iprand'
    """
    return 'iprand'

def iprand_all():
    """Some ipython tests with fully random output.

    # all-random
    
    In [7]: 1
    Out[7]: 99

    In [8]: print('hello')
    world

    In [9]: iprand_all()
    Out[9]: 'junk'
    """
    return 'iprand_all'
