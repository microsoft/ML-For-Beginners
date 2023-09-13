"""Experimental code for cleaner support of IPython syntax with unittest.

In IPython up until 0.10, we've used very hacked up nose machinery for running
tests with IPython special syntax, and this has proved to be extremely slow.
This module provides decorators to try a different approach, stemming from a
conversation Brian and I (FP) had about this problem Sept/09.

The goal is to be able to easily write simple functions that can be seen by
unittest as tests, and ultimately for these to support doctests with full
IPython syntax.  Nose already offers this based on naming conventions and our
hackish plugins, but we are seeking to move away from nose dependencies if
possible.

This module follows a different approach, based on decorators.

- A decorator called @ipdoctest can mark any function as having a docstring
  that should be viewed as a doctest, but after syntax conversion.

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

# Stdlib
import re
import unittest
from doctest import DocTestFinder, DocTestRunner, TestResults
from IPython.terminal.interactiveshell import InteractiveShell

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

def count_failures(runner):
    """Count number of failures in a doctest runner.

    Code modeled after the summarize() method in doctest.
    """
    return [TestResults(f, t) for f, t in runner._name2ft.values() if f > 0 ]


class IPython2PythonConverter(object):
    """Convert IPython 'syntax' to valid Python.

    Eventually this code may grow to be the full IPython syntax conversion
    implementation, but for now it only does prompt conversion."""
    
    def __init__(self):
        self.rps1 = re.compile(r'In\ \[\d+\]: ')
        self.rps2 = re.compile(r'\ \ \ \.\.\.+: ')
        self.rout = re.compile(r'Out\[\d+\]: \s*?\n?')
        self.pyps1 = '>>> '
        self.pyps2 = '... '
        self.rpyps1 = re.compile (r'(\s*%s)(.*)$' % self.pyps1)
        self.rpyps2 = re.compile (r'(\s*%s)(.*)$' % self.pyps2)

    def __call__(self, ds):
        """Convert IPython prompts to python ones in a string."""
        from . import globalipapp

        pyps1 = '>>> '
        pyps2 = '... '
        pyout = ''

        dnew = ds
        dnew = self.rps1.sub(pyps1, dnew)
        dnew = self.rps2.sub(pyps2, dnew)
        dnew = self.rout.sub(pyout, dnew)
        ip = InteractiveShell.instance()

        # Convert input IPython source into valid Python.
        out = []
        newline = out.append
        for line in dnew.splitlines():

            mps1 = self.rpyps1.match(line)
            if mps1 is not None:
                prompt, text = mps1.groups()
                newline(prompt+ip.prefilter(text, False))
                continue

            mps2 = self.rpyps2.match(line)
            if mps2 is not None:
                prompt, text = mps2.groups()
                newline(prompt+ip.prefilter(text, True))
                continue
            
            newline(line)
        newline('')  # ensure a closing newline, needed by doctest
        #print "PYSRC:", '\n'.join(out)  # dbg
        return '\n'.join(out)

    #return dnew


class Doc2UnitTester(object):
    """Class whose instances act as a decorator for docstring testing.

    In practice we're only likely to need one instance ever, made below (though
    no attempt is made at turning it into a singleton, there is no need for
    that).
    """
    def __init__(self, verbose=False):
        """New decorator.

        Parameters
        ----------

        verbose : boolean, optional (False)
          Passed to the doctest finder and runner to control verbosity.
        """
        self.verbose = verbose
        # We can reuse the same finder for all instances
        self.finder = DocTestFinder(verbose=verbose, recurse=False)

    def __call__(self, func):
        """Use as a decorator: doctest a function's docstring as a unittest.
        
        This version runs normal doctests, but the idea is to make it later run
        ipython syntax instead."""

        # Capture the enclosing instance with a different name, so the new
        # class below can see it without confusion regarding its own 'self'
        # that will point to the test instance at runtime
        d2u = self

        # Rewrite the function's docstring to have python syntax
        if func.__doc__ is not None:
            func.__doc__ = ip2py(func.__doc__)

        # Now, create a tester object that is a real unittest instance, so
        # normal unittest machinery (or Nose, or Trial) can find it.
        class Tester(unittest.TestCase):
            def test(self):
                # Make a new runner per function to be tested
                runner = DocTestRunner(verbose=d2u.verbose)
                for the_test in d2u.finder.find(func, func.__name__):
                    runner.run(the_test)
                failed = count_failures(runner)
                if failed:
                    # Since we only looked at a single function's docstring,
                    # failed should contain at most one item.  More than that
                    # is a case we can't handle and should error out on
                    if len(failed) > 1:
                        err = "Invalid number of test results: %s" % failed
                        raise ValueError(err)
                    # Report a normal failure.
                    self.fail('failed doctests: %s' % str(failed[0]))
                    
        # Rename it so test reports have the original signature.
        Tester.__name__ = func.__name__
        return Tester


def ipdocstring(func):
    """Change the function docstring via ip2py.
    """
    if func.__doc__ is not None:
        func.__doc__ = ip2py(func.__doc__)
    return func

        
# Make an instance of the classes for public use
ipdoctest = Doc2UnitTester()
ip2py = IPython2PythonConverter()
