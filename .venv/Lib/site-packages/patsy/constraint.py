# This file is part of Patsy
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Interpreting linear constraints like "2*x1 + x2 = 0"

from __future__ import print_function

# These are made available in the patsy.* namespace
__all__ = ["LinearConstraint"]

import re
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
                        repr_pretty_delegate, repr_pretty_impl,
                        no_pickling, assert_no_pickling)
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test


class LinearConstraint(object):
    """A linear constraint in matrix form.

    This object represents a linear constraint of the form `Ax = b`.

    Usually you won't be constructing these by hand, but instead get them as
    the return value from :meth:`DesignInfo.linear_constraint`.

    .. attribute:: coefs

       A 2-dimensional ndarray with float dtype, representing `A`.

    .. attribute:: constants

       A 2-dimensional single-column ndarray with float dtype, representing
       `b`.

    .. attribute:: variable_names

       A list of strings giving the names of the variables being
       constrained. (Used only for consistency checking.)
    """
    def __init__(self, variable_names, coefs, constants=None):
        self.variable_names = list(variable_names)
        self.coefs = np.atleast_2d(np.asarray(coefs, dtype=float))
        if constants is None:
            constants = np.zeros(self.coefs.shape[0], dtype=float)
        constants = np.asarray(constants, dtype=float)
        self.constants = atleast_2d_column_default(constants)
        if self.constants.ndim != 2 or self.constants.shape[1] != 1:
            raise ValueError("constants is not (convertible to) a column matrix")
        if self.coefs.ndim != 2 or self.coefs.shape[1] != len(variable_names):
            raise ValueError("wrong shape for coefs")
        if self.coefs.shape[0] == 0:
            raise ValueError("must have at least one row in constraint matrix")
        if self.coefs.shape[0] != self.constants.shape[0]:
            raise ValueError("shape mismatch between coefs and constants")

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        return repr_pretty_impl(p, self,
                                [self.variable_names, self.coefs, self.constants])

    __getstate__ = no_pickling

    @classmethod
    def combine(cls, constraints):
        """Create a new LinearConstraint by ANDing together several existing
        LinearConstraints.

        :arg constraints: An iterable of LinearConstraint objects. Their
          :attr:`variable_names` attributes must all match.
        :returns: A new LinearConstraint object.
        """
        if not constraints:
            raise ValueError("no constraints specified")
        variable_names = constraints[0].variable_names
        for constraint in constraints:
            if constraint.variable_names != variable_names:
                raise ValueError("variable names don't match")
        coefs = np.vstack([c.coefs for c in constraints])
        constants = np.vstack([c.constants for c in constraints])
        return cls(variable_names, coefs, constants)

def test_LinearConstraint():
    try:
        from numpy.testing import assert_equal
    except ImportError:
        from numpy.testing.utils import assert_equal
    lc = LinearConstraint(["foo", "bar"], [1, 1])
    assert lc.variable_names == ["foo", "bar"]
    assert_equal(lc.coefs, [[1, 1]])
    assert_equal(lc.constants, [[0]])

    lc = LinearConstraint(["foo", "bar"], [[1, 1], [2, 3]], [10, 20])
    assert_equal(lc.coefs, [[1, 1], [2, 3]])
    assert_equal(lc.constants, [[10], [20]])

    assert lc.coefs.dtype == np.dtype(float)
    assert lc.constants.dtype == np.dtype(float)


    # statsmodels wants to be able to create degenerate constraints like this,
    # see:
    #     https://github.com/pydata/patsy/issues/89
    # We used to forbid it, but I guess it's harmless, so why not.
    lc = LinearConstraint(["a"], [[0]])
    assert_equal(lc.coefs, [[0]])

    import pytest
    pytest.raises(ValueError, LinearConstraint, ["a"], [[1, 2]])
    pytest.raises(ValueError, LinearConstraint, ["a"], [[[1]]])
    pytest.raises(ValueError, LinearConstraint, ["a"], [[1, 2]], [3, 4])
    pytest.raises(ValueError, LinearConstraint, ["a", "b"], [[1, 2]], [3, 4])
    pytest.raises(ValueError, LinearConstraint, ["a"], [[1]], [[]])
    pytest.raises(ValueError, LinearConstraint, ["a", "b"], [])
    pytest.raises(ValueError, LinearConstraint, ["a", "b"],
                  np.zeros((0, 2)))

    assert_no_pickling(lc)

def test_LinearConstraint_combine():
    comb = LinearConstraint.combine([LinearConstraint(["a", "b"], [1, 0]),
                                     LinearConstraint(["a", "b"], [0, 1], [1])])
    assert comb.variable_names == ["a", "b"]
    try:
        from numpy.testing import assert_equal
    except ImportError:
        from numpy.testing.utils import assert_equal
    assert_equal(comb.coefs, [[1, 0], [0, 1]])
    assert_equal(comb.constants, [[0], [1]])

    import pytest
    pytest.raises(ValueError, LinearConstraint.combine, [])
    pytest.raises(ValueError, LinearConstraint.combine,
                  [LinearConstraint(["a"], [1]), LinearConstraint(["b"], [1])])


_ops = [
    Operator(",", 2, -100),

    Operator("=", 2, 0),

    Operator("+", 1, 100),
    Operator("-", 1, 100),
    Operator("+", 2, 100),
    Operator("-", 2, 100),

    Operator("*", 2, 200),
    Operator("/", 2, 200),
    ]

_atomic = ["NUMBER", "VARIABLE"]

def _token_maker(type, string):
    def make_token(scanner, token_string):
        if type == "__OP__":
            actual_type = token_string
        else:
            actual_type = type
        return Token(actual_type,
                     Origin(string, *scanner.match.span()),
                     token_string)
    return make_token

def _tokenize_constraint(string, variable_names):
    lparen_re = r"\("
    rparen_re = r"\)"
    op_re = "|".join([re.escape(op.token_type) for op in _ops])
    num_re = r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
    whitespace_re = r"\s+"

    # Prefer long matches:
    variable_names = sorted(variable_names, key=len, reverse=True)
    variable_re = "|".join([re.escape(n) for n in variable_names])

    lexicon = [
        (lparen_re, _token_maker(Token.LPAREN, string)),
        (rparen_re, _token_maker(Token.RPAREN, string)),
        (op_re, _token_maker("__OP__", string)),
        (variable_re, _token_maker("VARIABLE", string)),
        (num_re, _token_maker("NUMBER", string)),
        (whitespace_re, None),
        ]

    scanner = re.Scanner(lexicon)
    tokens, leftover = scanner.scan(string)
    if leftover:
        offset = len(string) - len(leftover)
        raise PatsyError("unrecognized token in constraint",
                            Origin(string, offset, offset + 1))

    return tokens

def test__tokenize_constraint():
    code = "2 * (a + b) = q"
    tokens = _tokenize_constraint(code, ["a", "b", "q"])
    expecteds = [("NUMBER", 0, 1, "2"),
                 ("*", 2, 3, "*"),
                 (Token.LPAREN, 4, 5, "("),
                 ("VARIABLE", 5, 6, "a"),
                 ("+", 7, 8, "+"),
                 ("VARIABLE", 9, 10, "b"),
                 (Token.RPAREN, 10, 11, ")"),
                 ("=", 12, 13, "="),
                 ("VARIABLE", 14, 15, "q")]
    for got, expected in zip(tokens, expecteds):
        assert isinstance(got, Token)
        assert got.type == expected[0]
        assert got.origin == Origin(code, expected[1], expected[2])
        assert got.extra == expected[3]

    import pytest
    pytest.raises(PatsyError, _tokenize_constraint, "1 + @b", ["b"])
    # Shouldn't raise an error:
    _tokenize_constraint("1 + @b", ["@b"])

    # Check we aren't confused by names which are proper prefixes of other
    # names:
    for names in (["a", "aa"], ["aa", "a"]):
        tokens = _tokenize_constraint("a aa a", names)
        assert len(tokens) == 3
        assert [t.extra for t in tokens] == ["a", "aa", "a"]

    # Check that embedding ops and numbers inside a variable name works
    tokens = _tokenize_constraint("2 * a[1,1],", ["a[1,1]"])
    assert len(tokens) == 4
    assert [t.type for t in tokens] == ["NUMBER", "*", "VARIABLE", ","]
    assert [t.extra for t in tokens] == ["2", "*", "a[1,1]", ","]

def parse_constraint(string, variable_names):
    return infix_parse(_tokenize_constraint(string, variable_names),
                       _ops, _atomic)

class _EvalConstraint(object):
    def __init__(self, variable_names):
        self._variable_names = variable_names
        self._N = len(variable_names)

        self._dispatch = {
            ("VARIABLE", 0): self._eval_variable,
            ("NUMBER", 0): self._eval_number,
            ("+", 1): self._eval_unary_plus,
            ("-", 1): self._eval_unary_minus,
            ("+", 2): self._eval_binary_plus,
            ("-", 2): self._eval_binary_minus,
            ("*", 2): self._eval_binary_multiply,
            ("/", 2): self._eval_binary_div,
            ("=", 2): self._eval_binary_eq,
            (",", 2): self._eval_binary_comma,
            }

    # General scheme: there are 2 types we deal with:
    #   - linear combinations ("lincomb"s) of variables and constants,
    #     represented as ndarrays with size N+1
    #     The last entry is the constant, so [10, 20, 30] means 10x + 20y +
    #     30.
    #   - LinearConstraint objects

    def is_constant(self, coefs):
        return np.all(coefs[:self._N] == 0)

    def _eval_variable(self, tree):
        var = tree.token.extra
        coefs = np.zeros((self._N + 1,), dtype=float)
        coefs[self._variable_names.index(var)] = 1
        return coefs

    def _eval_number(self, tree):
        coefs = np.zeros((self._N + 1,), dtype=float)
        coefs[-1] = float(tree.token.extra)
        return coefs

    def _eval_unary_plus(self, tree):
        return self.eval(tree.args[0])

    def _eval_unary_minus(self, tree):
        return -1 * self.eval(tree.args[0])

    def _eval_binary_plus(self, tree):
        return self.eval(tree.args[0]) + self.eval(tree.args[1])

    def _eval_binary_minus(self, tree):
        return self.eval(tree.args[0]) - self.eval(tree.args[1])

    def _eval_binary_div(self, tree):
        left = self.eval(tree.args[0])
        right = self.eval(tree.args[1])
        if not self.is_constant(right):
            raise PatsyError("Can't divide by a variable in a linear "
                                "constraint", tree.args[1])
        return left / right[-1]

    def _eval_binary_multiply(self, tree):
        left = self.eval(tree.args[0])
        right = self.eval(tree.args[1])
        if self.is_constant(left):
            return left[-1] * right
        elif self.is_constant(right):
            return left * right[-1]
        else:
            raise PatsyError("Can't multiply one variable by another "
                                "in a linear constraint", tree)

    def _eval_binary_eq(self, tree):
        # Handle "a1 = a2 = a3", which is parsed as "(a1 = a2) = a3"
        args = list(tree.args)
        constraints = []
        for i, arg in enumerate(args):
            if arg.type == "=":
                constraints.append(self.eval(arg, constraint=True))
                # make our left argument be their right argument, or
                # vice-versa
                args[i] = arg.args[1 - i]
        left = self.eval(args[0])
        right = self.eval(args[1])
        coefs = left[:self._N] - right[:self._N]
        if np.all(coefs == 0):
            raise PatsyError("no variables appear in constraint", tree)
        constant = -left[-1] + right[-1]
        constraint = LinearConstraint(self._variable_names, coefs, constant)
        constraints.append(constraint)
        return LinearConstraint.combine(constraints)

    def _eval_binary_comma(self, tree):
        left = self.eval(tree.args[0], constraint=True)
        right = self.eval(tree.args[1], constraint=True)
        return LinearConstraint.combine([left, right])

    def eval(self, tree, constraint=False):
        key = (tree.type, len(tree.args))
        assert key in self._dispatch
        val = self._dispatch[key](tree)
        if constraint:
            # Force it to be a constraint
            if isinstance(val, LinearConstraint):
                return val
            else:
                assert val.size == self._N + 1
                if np.all(val[:self._N] == 0):
                    raise PatsyError("term is constant, with no variables",
                                        tree)
                return LinearConstraint(self._variable_names,
                                        val[:self._N],
                                        -val[-1])
        else:
            # Force it to *not* be a constraint
            if isinstance(val, LinearConstraint):
                raise PatsyError("unexpected constraint object", tree)
            return val

def linear_constraint(constraint_like, variable_names):
    """This is the internal interface implementing
    DesignInfo.linear_constraint, see there for docs."""
    if isinstance(constraint_like, LinearConstraint):
        if constraint_like.variable_names != variable_names:
            raise ValueError("LinearConstraint has wrong variable_names "
                             "(got %r, expected %r)"
                             % (constraint_like.variable_names,
                                variable_names))
        return constraint_like

    if isinstance(constraint_like, Mapping):
        # Simple conjunction-of-equality constraints can be specified as
        # dicts. {"x": 1, "y": 2} -> tests x = 1 and y = 2. Keys can be
        # either variable names, or variable indices.
        coefs = np.zeros((len(constraint_like), len(variable_names)),
                         dtype=float)
        constants = np.zeros(len(constraint_like))
        used = set()
        for i, (name, value) in enumerate(six.iteritems(constraint_like)):
            if name in variable_names:
                idx = variable_names.index(name)
            elif isinstance(name, six.integer_types):
                idx = name
            else:
                raise ValueError("unrecognized variable name/index %r"
                                 % (name,))
            if idx in used:
                raise ValueError("duplicated constraint on %r"
                                 % (variable_names[idx],))
            used.add(idx)
            coefs[i, idx] = 1
            constants[i] = value
        return LinearConstraint(variable_names, coefs, constants)

    if isinstance(constraint_like, str):
        constraint_like = [constraint_like]
        # fall-through

    if (isinstance(constraint_like, list)
        and constraint_like
        and isinstance(constraint_like[0], str)):
        constraints = []
        for code in constraint_like:
            if not isinstance(code, str):
                raise ValueError("expected a string, not %r" % (code,))
            tree = parse_constraint(code, variable_names)
            evaluator = _EvalConstraint(variable_names)
            constraints.append(evaluator.eval(tree, constraint=True))
        return LinearConstraint.combine(constraints)

    if isinstance(constraint_like, tuple):
        if len(constraint_like) != 2:
            raise ValueError("constraint tuple must have length 2")
        coef, constants = constraint_like
        return LinearConstraint(variable_names, coef, constants)

    # assume a raw ndarray
    coefs = np.asarray(constraint_like, dtype=float)
    return LinearConstraint(variable_names, coefs)


def _check_lincon(input, varnames, coefs, constants):
    try:
        from numpy.testing import assert_equal
    except ImportError:
        from numpy.testing.utils import assert_equal
    got = linear_constraint(input, varnames)
    print("got", got)
    expected = LinearConstraint(varnames, coefs, constants)
    print("expected", expected)
    assert_equal(got.variable_names, expected.variable_names)
    assert_equal(got.coefs, expected.coefs)
    assert_equal(got.constants, expected.constants)
    assert_equal(got.coefs.dtype, np.dtype(float))
    assert_equal(got.constants.dtype, np.dtype(float))


def test_linear_constraint():
    import pytest
    from patsy.compat import OrderedDict
    t = _check_lincon

    t(LinearConstraint(["a", "b"], [2, 3]), ["a", "b"], [[2, 3]], [[0]])
    pytest.raises(ValueError, linear_constraint,
                  LinearConstraint(["b", "a"], [2, 3]),
                  ["a", "b"])

    t({"a": 2}, ["a", "b"], [[1, 0]], [[2]])
    t(OrderedDict([("a", 2), ("b", 3)]),
      ["a", "b"], [[1, 0], [0, 1]], [[2], [3]])
    t(OrderedDict([("a", 2), ("b", 3)]),
      ["b", "a"], [[0, 1], [1, 0]], [[2], [3]])

    t({0: 2}, ["a", "b"], [[1, 0]], [[2]])
    t(OrderedDict([(0, 2), (1, 3)]), ["a", "b"], [[1, 0], [0, 1]], [[2], [3]])

    t(OrderedDict([("a", 2), (1, 3)]),
      ["a", "b"], [[1, 0], [0, 1]], [[2], [3]])

    pytest.raises(ValueError, linear_constraint, {"q": 1}, ["a", "b"])
    pytest.raises(ValueError, linear_constraint, {"a": 1, 0: 2}, ["a", "b"])

    t(np.array([2, 3]), ["a", "b"], [[2, 3]], [[0]])
    t(np.array([[2, 3], [4, 5]]), ["a", "b"], [[2, 3], [4, 5]], [[0], [0]])

    t("a = 2", ["a", "b"], [[1, 0]], [[2]])
    t("a - 2", ["a", "b"], [[1, 0]], [[2]])
    t("a + 1 = 3", ["a", "b"], [[1, 0]], [[2]])
    t("a + b = 3", ["a", "b"], [[1, 1]], [[3]])
    t("a = 2, b = 3", ["a", "b"], [[1, 0], [0, 1]], [[2], [3]])
    t("b = 3, a = 2", ["a", "b"], [[0, 1], [1, 0]], [[3], [2]])

    t(["a = 2", "b = 3"], ["a", "b"], [[1, 0], [0, 1]], [[2], [3]])

    pytest.raises(ValueError, linear_constraint, ["a", {"b": 0}], ["a", "b"])

    # Actual evaluator tests
    t("2 * (a + b/3) + b + 2*3/4 = 1 + 2*3", ["a", "b"],
      [[2, 2.0/3 + 1]], [[7 - 6.0/4]])
    t("+2 * -a", ["a", "b"], [[-2, 0]], [[0]])
    t("a - b, a + b = 2", ["a", "b"], [[1, -1], [1, 1]], [[0], [2]])
    t("a = 1, a = 2, a = 3", ["a", "b"],
      [[1, 0], [1, 0], [1, 0]], [[1], [2], [3]])
    t("a * 2", ["a", "b"], [[2, 0]], [[0]])
    t("-a = 1", ["a", "b"], [[-1, 0]], [[1]])
    t("(2 + a - a) * b", ["a", "b"], [[0, 2]], [[0]])

    t("a = 1 = b", ["a", "b"], [[1, 0], [0, -1]], [[1], [-1]])
    t("a = (1 = b)", ["a", "b"], [[0, -1], [1, 0]], [[-1], [1]])
    t("a = 1, a = b = c", ["a", "b", "c"],
      [[1, 0, 0], [1, -1, 0], [0, 1, -1]], [[1], [0], [0]])

    # One should never do this of course, but test that it works anyway...
    t("a + 1 = 2", ["a", "a + 1"], [[0, 1]], [[2]])

    t(([10, 20], [30]), ["a", "b"], [[10, 20]], [[30]])
    t(([[10, 20], [20, 40]], [[30], [35]]), ["a", "b"],
      [[10, 20], [20, 40]], [[30], [35]])
    # wrong-length tuple
    pytest.raises(ValueError, linear_constraint,
                  ([1, 0], [0], [0]), ["a", "b"])
    pytest.raises(ValueError, linear_constraint, ([1, 0],), ["a", "b"])

    t([10, 20], ["a", "b"], [[10, 20]], [[0]])
    t([[10, 20], [20, 40]], ["a", "b"], [[10, 20], [20, 40]], [[0], [0]])
    t(np.array([10, 20]), ["a", "b"], [[10, 20]], [[0]])
    t(np.array([[10, 20], [20, 40]]), ["a", "b"],
      [[10, 20], [20, 40]], [[0], [0]])

    # unknown object type
    pytest.raises(ValueError, linear_constraint, None, ["a", "b"])


_parse_eval_error_tests = [
    # Bad token
    "a + <f>oo",
    # No pure constant equalities
    "a = 1, <1 = 1>, b = 1",
    "a = 1, <b * 2 - b + (-2/2 * b)>",
    "a = 1, <1>, b = 2",
    "a = 1, <2 * b = b + b>, c",
    # No non-linearities
    "a + <a * b> + c",
    "a + 2 / <b> + c",
    # Constraints are not numbers
    "a = 1, 2 * <(a = b)>, c",
    "a = 1, a + <(a = b)>, c",
    "a = 1, <(a, b)> + 2, c",
]


def test_eval_errors():
    def doit(bad_code):
        return linear_constraint(bad_code, ["a", "b", "c"])
    _parsing_error_test(doit, _parse_eval_error_tests)
