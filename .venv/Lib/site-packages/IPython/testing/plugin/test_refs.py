"""Some simple tests for the plugin while running scripts.
"""
# Module imports
# Std lib
import inspect

# Our own

#-----------------------------------------------------------------------------
# Testing functions

def test_trivial():
    """A trivial passing test."""
    pass

def doctest_run():
    """Test running a trivial script.

    In [13]: run simplevars.py
    x is: 1
    """

def doctest_runvars():
    """Test that variables defined in scripts get loaded correctly via %run.

    In [13]: run simplevars.py
    x is: 1

    In [14]: x
    Out[14]: 1
    """

def doctest_ivars():
    """Test that variables defined interactively are picked up.
    In [5]: zz=1

    In [6]: zz
    Out[6]: 1
    """
