"""Tests for the key interactiveshell module, where the main ipython class is defined.
"""

import stack_data
import sys

SV_VERSION = tuple([int(x) for x in stack_data.__version__.split(".")[0:2]])


def test_reset():
    """reset must clear most namespaces."""

    # Check that reset runs without error
    ip.reset()

    # Once we've reset it (to clear of any junk that might have been there from
    # other tests, we can count how many variables are in the user's namespace
    nvars_user_ns = len(ip.user_ns)
    nvars_hidden = len(ip.user_ns_hidden)

    # Now add a few variables to user_ns, and check that reset clears them
    ip.user_ns['x'] = 1
    ip.user_ns['y'] = 1
    ip.reset()
    
    # Finally, check that all namespaces have only as many variables as we
    # expect to find in them:
    assert len(ip.user_ns) == nvars_user_ns
    assert len(ip.user_ns_hidden) == nvars_hidden


# Tests for reporting of exceptions in various modes, handling of SystemExit,
# and %tb functionality.  This is really a mix of testing ultraTB and interactiveshell.

def doctest_tb_plain():
    """
    In [18]: xmode plain
    Exception reporting mode: Plain

    In [19]: run simpleerr.py
    Traceback (most recent call last):
      File ...:...
        bar(mode)
      File ...:... in bar
        div0()
      File ...:... in div0
        x/y
    ZeroDivisionError: ...
    """


def doctest_tb_context():
    """
    In [3]: xmode context
    Exception reporting mode: Context

    In [4]: run simpleerr.py
    ---------------------------------------------------------------------------
    ZeroDivisionError                         Traceback (most recent call last)
    <BLANKLINE>
    ...
         30     except IndexError:
         31         mode = 'div'
    ---> 33     bar(mode)
    <BLANKLINE>
    ... in bar(mode)
         15     "bar"
         16     if mode=='div':
    ---> 17         div0()
         18     elif mode=='exit':
         19         try:
    <BLANKLINE>
    ... in div0()
          6     x = 1
          7     y = 0
    ----> 8     x/y
    <BLANKLINE>
    ZeroDivisionError: ..."""


def doctest_tb_verbose():
    """
    In [5]: xmode verbose
    Exception reporting mode: Verbose

    In [6]: run simpleerr.py
    ---------------------------------------------------------------------------
    ZeroDivisionError                         Traceback (most recent call last)
    <BLANKLINE>
    ...
         30     except IndexError:
         31         mode = 'div'
    ---> 33     bar(mode)
            mode = 'div'
    <BLANKLINE>
    ... in bar(mode='div')
         15     "bar"
         16     if mode=='div':
    ---> 17         div0()
         18     elif mode=='exit':
         19         try:
    <BLANKLINE>
    ... in div0()
          6     x = 1
          7     y = 0
    ----> 8     x/y
            x = 1
            y = 0
    <BLANKLINE>
    ZeroDivisionError: ...
    """


def doctest_tb_sysexit():
    """
    In [17]: %xmode plain
    Exception reporting mode: Plain

    In [18]: %run simpleerr.py exit
    An exception has occurred, use %tb to see the full traceback.
    SystemExit: (1, 'Mode = exit')

    In [19]: %run simpleerr.py exit 2
    An exception has occurred, use %tb to see the full traceback.
    SystemExit: (2, 'Mode = exit')

    In [20]: %tb
    Traceback (most recent call last):
      File ...:... in execfile
        exec(compiler(f.read(), fname, "exec"), glob, loc)
      File ...:...
        bar(mode)
      File ...:... in bar
        sysexit(stat, mode)
      File ...:... in sysexit
        raise SystemExit(stat, f"Mode = {mode}")
    SystemExit: (2, 'Mode = exit')

    In [21]: %xmode context
    Exception reporting mode: Context

    In [22]: %tb
    ---------------------------------------------------------------------------
    SystemExit                                Traceback (most recent call last)
    File ..., in execfile(fname, glob, loc, compiler)
         ... with open(fname, "rb") as f:
         ...     compiler = compiler or compile
    ---> ...     exec(compiler(f.read(), fname, "exec"), glob, loc)
    ...
         30     except IndexError:
         31         mode = 'div'
    ---> 33     bar(mode)
    <BLANKLINE>
    ...bar(mode)
         21         except:
         22             stat = 1
    ---> 23         sysexit(stat, mode)
         24     else:
         25         raise ValueError('Unknown mode')
    <BLANKLINE>
    ...sysexit(stat, mode)
         10 def sysexit(stat, mode):
    ---> 11     raise SystemExit(stat, f"Mode = {mode}")
    <BLANKLINE>
    SystemExit: (2, 'Mode = exit')
    """


if SV_VERSION < (0, 6):

    def doctest_tb_sysexit_verbose_stack_data_05():
        """
        In [18]: %run simpleerr.py exit
        An exception has occurred, use %tb to see the full traceback.
        SystemExit: (1, 'Mode = exit')

        In [19]: %run simpleerr.py exit 2
        An exception has occurred, use %tb to see the full traceback.
        SystemExit: (2, 'Mode = exit')

        In [23]: %xmode verbose
        Exception reporting mode: Verbose

        In [24]: %tb
        ---------------------------------------------------------------------------
        SystemExit                                Traceback (most recent call last)
        <BLANKLINE>
        ...
            30     except IndexError:
            31         mode = 'div'
        ---> 33     bar(mode)
                mode = 'exit'
        <BLANKLINE>
        ... in bar(mode='exit')
            ...     except:
            ...         stat = 1
        ---> ...     sysexit(stat, mode)
                mode = 'exit'
                stat = 2
            ...     else:
            ...         raise ValueError('Unknown mode')
        <BLANKLINE>
        ... in sysexit(stat=2, mode='exit')
            10 def sysexit(stat, mode):
        ---> 11     raise SystemExit(stat, f"Mode = {mode}")
                stat = 2
        <BLANKLINE>
        SystemExit: (2, 'Mode = exit')
        """


def test_run_cell():
    import textwrap

    ip.run_cell("a = 10\na+=1")
    ip.run_cell("assert a == 11\nassert 1")

    assert ip.user_ns["a"] == 11
    complex = textwrap.dedent(
        """
    if 1:
        print "hello"
        if 1:
            print "world"
        
    if 2:
        print "foo"

    if 3:
        print "bar"

    if 4:
        print "bar"
    
    """
    )
    # Simply verifies that this kind of input is run
    ip.run_cell(complex)
    

def test_db():
    """Test the internal database used for variable persistence."""
    ip.db["__unittest_"] = 12
    assert ip.db["__unittest_"] == 12
    del ip.db["__unittest_"]
    assert "__unittest_" not in ip.db
