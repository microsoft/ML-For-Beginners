"""These kinds of tests are less than ideal, but at least they run.

This was an old test that was being run interactively in the top-level tests/
directory, which we are removing.  For now putting this here ensures at least
we do run the test, though ultimately this functionality should all be tested
with better-isolated tests that don't rely on the global instance in iptest.
"""
from IPython.core.splitinput import LineInfo
from IPython.core.prefilter import AutocallChecker


def doctest_autocall():
    """
    In [1]: def f1(a,b,c):
       ...:     return a+b+c
       ...:

    In [2]: def f2(a):
       ...:     return a + a
       ...:

    In [3]: def r(x):
       ...:     return True
       ...:

    In [4]: ;f2 a b c
    Out[4]: 'a b ca b c'

    In [5]: assert _ == "a b ca b c"

    In [6]: ,f1 a b c
    Out[6]: 'abc'

    In [7]: assert _ == 'abc'

    In [8]: print(_)
    abc

    In [9]: /f1 1,2,3
    Out[9]: 6

    In [10]: assert _ == 6

    In [11]: /f2 4
    Out[11]: 8

    In [12]: assert _ == 8

    In [12]: del f1, f2

    In [13]: ,r a
    Out[13]: True

    In [14]: assert _ == True

    In [15]: r'a'
    Out[15]: 'a'

    In [16]: assert _ == 'a'
    """


def test_autocall_should_ignore_raw_strings():
    line_info = LineInfo("r'a'")
    pm = ip.prefilter_manager
    ac = AutocallChecker(shell=pm.shell, prefilter_manager=pm, config=pm.config)
    assert ac.check(line_info) is None
