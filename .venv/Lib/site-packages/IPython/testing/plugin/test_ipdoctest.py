"""Tests for the ipdoctest machinery itself.

Note: in a file named test_X, functions whose only test is their docstring (as
a doctest) and which have no test functionality of their own, should be called
'doctest_foo' instead of 'test_foo', otherwise they get double-counted (the
empty function call is counted as a test, which just inflates tests numbers
artificially).
"""

def doctest_simple():
    """ipdoctest must handle simple inputs
    
    In [1]: 1
    Out[1]: 1

    In [2]: print(1)
    1
    """

def doctest_multiline1():
    """The ipdoctest machinery must handle multiline examples gracefully.

    In [2]: for i in range(4):
       ...:     print(i)
       ...:      
    0
    1
    2
    3
    """

def doctest_multiline2():
    """Multiline examples that define functions and print output.

    In [7]: def f(x):
       ...:     return x+1
       ...: 

    In [8]: f(1)
    Out[8]: 2

    In [9]: def g(x):
       ...:     print('x is:',x)
       ...:      

    In [10]: g(1)
    x is: 1

    In [11]: g('hello')
    x is: hello
    """


def doctest_multiline3():
    """Multiline examples with blank lines.

    In [12]: def h(x):
       ....:     if x>1:
       ....:         return x**2
       ....:     # To leave a blank line in the input, you must mark it
       ....:     # with a comment character:
       ....:     #
       ....:     # otherwise the doctest parser gets confused.
       ....:     else:
       ....:         return -1
       ....:      

    In [13]: h(5)
    Out[13]: 25

    In [14]: h(1)
    Out[14]: -1

    In [15]: h(0)
    Out[15]: -1
   """


def doctest_builtin_underscore():
    """Defining builtins._ should not break anything outside the doctest
    while also should be working as expected inside the doctest.

    In [1]: import builtins

    In [2]: builtins._ = 42

    In [3]: builtins._
    Out[3]: 42

    In [4]: _
    Out[4]: 42
    """
