#-----------------------------------------------------------------------------
# Copyright (C) 2010-2011, IPython Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import argparse
import sys

from IPython.core.magic_arguments import (argument, argument_group, kwds,
    magic_arguments, parse_argstring, real_name)


@magic_arguments()
@argument('-f', '--foo', help="an argument")
def magic_foo1(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo1, args)


@magic_arguments()
def magic_foo2(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo2, args)


@magic_arguments()
@argument('-f', '--foo', help="an argument")
@argument_group('Group')
@argument('-b', '--bar', help="a grouped argument")
@argument_group('Second Group')
@argument('-z', '--baz', help="another grouped argument")
def magic_foo3(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo3, args)


@magic_arguments()
@kwds(argument_default=argparse.SUPPRESS)
@argument('-f', '--foo', help="an argument")
def magic_foo4(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo4, args)


@magic_arguments('frobnicate')
@argument('-f', '--foo', help="an argument")
def magic_foo5(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo5, args)


@magic_arguments()
@argument('-f', '--foo', help="an argument")
def magic_magic_foo(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_magic_foo, args)


@magic_arguments()
@argument('-f', '--foo', help="an argument")
def foo(self, args):
    """ A docstring.
    """
    return parse_argstring(foo, args)


def test_magic_arguments():
    # “optional arguments” was replaced with “options” in argparse help
    # https://docs.python.org/3/whatsnew/3.10.html#argparse
    # https://bugs.python.org/issue9694
    options = "optional arguments" if sys.version_info < (3, 10) else "options"

    assert (
        magic_foo1.__doc__
        == f"::\n\n  %foo1 [-f FOO]\n\n A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n"
    )
    assert getattr(magic_foo1, "argcmd_name", None) == None
    assert real_name(magic_foo1) == "foo1"
    assert magic_foo1(None, "") == argparse.Namespace(foo=None)
    assert hasattr(magic_foo1, "has_arguments")

    assert magic_foo2.__doc__ == "::\n\n  %foo2\n\n A docstring.\n"
    assert getattr(magic_foo2, "argcmd_name", None) == None
    assert real_name(magic_foo2) == "foo2"
    assert magic_foo2(None, "") == argparse.Namespace()
    assert hasattr(magic_foo2, "has_arguments")

    assert (
        magic_foo3.__doc__
        == f"::\n\n  %foo3 [-f FOO] [-b BAR] [-z BAZ]\n\n A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n\nGroup:\n  -b BAR, --bar BAR  a grouped argument\n\nSecond Group:\n  -z BAZ, --baz BAZ  another grouped argument\n"
    )
    assert getattr(magic_foo3, "argcmd_name", None) == None
    assert real_name(magic_foo3) == "foo3"
    assert magic_foo3(None, "") == argparse.Namespace(bar=None, baz=None, foo=None)
    assert hasattr(magic_foo3, "has_arguments")

    assert (
        magic_foo4.__doc__
        == f"::\n\n  %foo4 [-f FOO]\n\n A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n"
    )
    assert getattr(magic_foo4, "argcmd_name", None) == None
    assert real_name(magic_foo4) == "foo4"
    assert magic_foo4(None, "") == argparse.Namespace()
    assert hasattr(magic_foo4, "has_arguments")

    assert (
        magic_foo5.__doc__
        == f"::\n\n  %frobnicate [-f FOO]\n\n A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n"
    )
    assert getattr(magic_foo5, "argcmd_name", None) == "frobnicate"
    assert real_name(magic_foo5) == "frobnicate"
    assert magic_foo5(None, "") == argparse.Namespace(foo=None)
    assert hasattr(magic_foo5, "has_arguments")

    assert (
        magic_magic_foo.__doc__
        == f"::\n\n  %magic_foo [-f FOO]\n\n A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n"
    )
    assert getattr(magic_magic_foo, "argcmd_name", None) == None
    assert real_name(magic_magic_foo) == "magic_foo"
    assert magic_magic_foo(None, "") == argparse.Namespace(foo=None)
    assert hasattr(magic_magic_foo, "has_arguments")

    assert (
        foo.__doc__
        == f"::\n\n  %foo [-f FOO]\n\n A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n"
    )
    assert getattr(foo, "argcmd_name", None) == None
    assert real_name(foo) == "foo"
    assert foo(None, "") == argparse.Namespace(foo=None)
    assert hasattr(foo, "has_arguments")
