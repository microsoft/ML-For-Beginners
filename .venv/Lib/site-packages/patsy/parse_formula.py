 # This file is part of Patsy
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file defines a parser for a simple language based on S/R "formulas"
# (which are described in sections 2.3 and 2.4 in Chambers & Hastie, 1992). It
# uses the machinery in patsy.parse_core to do the heavy-lifting -- its
# biggest job is to handle tokenization.

from __future__ import print_function

__all__ = ["parse_formula"]

# The Python tokenizer
import tokenize

import six
from six.moves import cStringIO as StringIO

from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter

_atomic_token_types = ["PYTHON_EXPR", "ZERO", "ONE", "NUMBER"]

def _is_a(f, v):
    try:
        f(v)
    except ValueError:
        return False
    else:
        return True

# Helper function for _tokenize_formula:
def _read_python_expr(it, end_tokens):
    # Read out a full python expression, stopping when we hit an
    # unnested end token.
    pytypes = []
    token_strings = []
    origins = []
    bracket_level = 0
    for pytype, token_string, origin in it:
        assert bracket_level >= 0
        if bracket_level == 0 and token_string in end_tokens:
            it.push_back((pytype, token_string, origin))
            break
        if token_string in ("(", "[", "{"):
            bracket_level += 1
        if token_string in (")", "]", "}"):
            bracket_level -= 1
        if bracket_level < 0:
            raise PatsyError("unmatched close bracket", origin)
        pytypes.append(pytype)
        token_strings.append(token_string)
        origins.append(origin)
    # Either we found an end_token, or we hit the end of the string
    if bracket_level == 0:
        expr_text = pretty_untokenize(zip(pytypes, token_strings))
        if expr_text == "0":
            token_type = "ZERO"
        elif expr_text == "1":
            token_type = "ONE"
        elif _is_a(int, expr_text) or _is_a(float, expr_text):
            token_type = "NUMBER"
        else:
            token_type = "PYTHON_EXPR"
        return Token(token_type, Origin.combine(origins), extra=expr_text)
    else:
        raise PatsyError("unclosed bracket in embedded Python "
                            "expression",
                            Origin.combine(origins))

def _tokenize_formula(code, operator_strings):
    assert "(" not in operator_strings
    assert ")" not in operator_strings
    magic_token_types = {"(": Token.LPAREN,
                         ")": Token.RPAREN,
                         }
    for operator_string in operator_strings:
        magic_token_types[operator_string] = operator_string
    # Once we enter a Python expression, a ( does not end it, but any other
    # "magic" token does:
    end_tokens = set(magic_token_types)
    end_tokens.remove("(")

    it = PushbackAdapter(python_tokenize(code))
    for pytype, token_string, origin in it:
        if token_string in magic_token_types:
            yield Token(magic_token_types[token_string], origin)
        else:
            it.push_back((pytype, token_string, origin))
            yield _read_python_expr(it, end_tokens)

def test__tokenize_formula():
    code = "y ~ a + (foo(b,c +   2)) + -1 + 0 + 10"
    tokens = list(_tokenize_formula(code, ["+", "-", "~"]))
    expecteds = [("PYTHON_EXPR", Origin(code, 0, 1), "y"),
                 ("~", Origin(code, 2, 3), None),
                 ("PYTHON_EXPR", Origin(code, 4, 5), "a"),
                 ("+", Origin(code, 6, 7), None),
                 (Token.LPAREN, Origin(code, 8, 9), None),
                 ("PYTHON_EXPR", Origin(code, 9, 23), "foo(b, c + 2)"),
                 (Token.RPAREN, Origin(code, 23, 24), None),
                 ("+", Origin(code, 25, 26), None),
                 ("-", Origin(code, 27, 28), None),
                 ("ONE", Origin(code, 28, 29), "1"),
                 ("+", Origin(code, 30, 31), None),
                 ("ZERO", Origin(code, 32, 33), "0"),
                 ("+", Origin(code, 34, 35), None),
                 ("NUMBER", Origin(code, 36, 38), "10"),
                 ]
    for got, expected in zip(tokens, expecteds):
        assert isinstance(got, Token)
        assert got.type == expected[0]
        assert got.origin == expected[1]
        assert got.extra == expected[2]

_unary_tilde = Operator("~", 1, -100)
_default_ops = [
    _unary_tilde,
    Operator("~", 2, -100),

    Operator("+", 2, 100),
    Operator("-", 2, 100),
    Operator("*", 2, 200),
    Operator("/", 2, 200),
    Operator(":", 2, 300),
    Operator("**", 2, 500),

    Operator("+", 1, 100),
    Operator("-", 1, 100),
]

def parse_formula(code, extra_operators=[]):
    if not code.strip():
        code = "~ 1"

    for op in extra_operators:
        if op.precedence < 0:
            raise ValueError("all operators must have precedence >= 0")

    operators = _default_ops + extra_operators
    operator_strings = [op.token_type for op in operators]
    tree = infix_parse(_tokenize_formula(code, operator_strings),
                       operators,
                       _atomic_token_types)
    if not isinstance(tree, ParseNode) or tree.type != "~":
        tree = ParseNode("~", None, [tree], tree.origin)
    return tree

#############

_parser_tests = {
    "": ["~", "1"],
    " ": ["~", "1"],
    " \n ": ["~", "1"],

    "1": ["~", "1"],
    "a": ["~", "a"],
    "a ~ b": ["~", "a", "b"],

    "(a ~ b)": ["~", "a", "b"],
    "a ~ ((((b))))": ["~", "a", "b"],
    "a ~ ((((+b))))": ["~", "a", ["+", "b"]],

    "a + b + c": ["~", ["+", ["+", "a", "b"], "c"]],
    "a + (b ~ c) + d": ["~", ["+", ["+", "a", ["~", "b", "c"]], "d"]],

    "a + np.log(a, base=10)": ["~", ["+", "a", "np.log(a, base=10)"]],
    # Note different spacing:
    "a + np . log(a , base = 10)": ["~", ["+", "a", "np.log(a, base=10)"]],

    # Check precedence
    "a + b ~ c * d": ["~", ["+", "a", "b"], ["*", "c", "d"]],
    "a + b * c": ["~", ["+", "a", ["*", "b", "c"]]],
    "-a**2": ["~", ["-", ["**", "a", "2"]]],
    "-a:b": ["~", ["-", [":", "a", "b"]]],
    "a + b:c": ["~", ["+", "a", [":", "b", "c"]]],
    "(a + b):c": ["~", [":", ["+", "a", "b"], "c"]],
    "a*b:c": ["~", ["*", "a", [":", "b", "c"]]],

    "a+b / c": ["~", ["+", "a", ["/", "b", "c"]]],
    "~ a": ["~", "a"],

    "-1": ["~", ["-", "1"]],
    }

def _compare_trees(got, expected):
    assert isinstance(got, ParseNode)
    if got.args:
        assert got.type == expected[0]
        for arg, expected_arg in zip(got.args, expected[1:]):
            _compare_trees(arg, expected_arg)
    else:
        assert got.type in _atomic_token_types
        assert got.token.extra == expected

def _do_parse_test(test_cases, extra_operators):
    for code, expected in six.iteritems(test_cases):
        actual = parse_formula(code, extra_operators=extra_operators)
        print(repr(code), repr(expected))
        print(actual)
        _compare_trees(actual, expected)

def test_parse_formula():
    _do_parse_test(_parser_tests, [])

def test_parse_origin():
    tree = parse_formula("a ~ b + c")
    assert tree.origin == Origin("a ~ b + c", 0, 9)
    assert tree.token.origin == Origin("a ~ b + c", 2, 3)
    assert tree.args[0].origin == Origin("a ~ b + c", 0, 1)
    assert tree.args[1].origin == Origin("a ~ b + c", 4, 9)
    assert tree.args[1].token.origin == Origin("a ~ b + c", 6, 7)
    assert tree.args[1].args[0].origin == Origin("a ~ b + c", 4, 5)
    assert tree.args[1].args[1].origin == Origin("a ~ b + c", 8, 9)

# <> mark off where the error should be reported:
_parser_error_tests = [
    "a <+>",
    "a + <(>",

    "a + b <# asdf>",

    "<)>",
    "a + <)>",
    "<*> a",
    "a + <*>",

    "a + <foo[bar>",
    "a + <foo{bar>",
    "a + <foo(bar>",

    "a + <[bar>",
    "a + <{bar>",

    "a + <{bar[]>",

    "a + foo<]>bar",
    "a + foo[]<]>bar",
    "a + foo{}<}>bar",
    "a + foo<)>bar",

    "a + b<)>",
    "(a) <.>",

    "<(>a + b",

    "a +< >'foo", # Not the best placement for the error
]

# Split out so it can also be used by tests of the evaluator (which also
# raises PatsyError's)
def _parsing_error_test(parse_fn, error_descs): # pragma: no cover
    for error_desc in error_descs:
        letters = []
        start = None
        end = None
        for letter in error_desc:
            if letter == "<":
                start = len(letters)
            elif letter == ">":
                end = len(letters)
            else:
                letters.append(letter)
        bad_code = "".join(letters)
        assert start is not None and end is not None
        print(error_desc)
        print(repr(bad_code), start, end)
        try:
            parse_fn(bad_code)
        except PatsyError as e:
            print(e)
            assert e.origin.code == bad_code
            assert e.origin.start in (0, start)
            assert e.origin.end in (end, len(bad_code))
        else:
            assert False, "parser failed to report an error!"

def test_parse_errors(extra_operators=[]):
    def parse_fn(code):
        return parse_formula(code, extra_operators=extra_operators)
    _parsing_error_test(parse_fn, _parser_error_tests)

_extra_op_parser_tests = {
    "a | b": ["~", ["|", "a", "b"]],
    "a * b|c": ["~", ["*", "a", ["|", "b", "c"]]],
    }

def test_parse_extra_op():
    extra_operators = [Operator("|", 2, 250)]
    _do_parse_test(_parser_tests,
                   extra_operators=extra_operators)
    _do_parse_test(_extra_op_parser_tests,
                   extra_operators=extra_operators)
    test_parse_errors(extra_operators=extra_operators)
