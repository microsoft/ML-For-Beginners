# This file is part of Patsy
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file implements a simple "shunting yard algorithm" parser for infix
# languages with parentheses. It is used as the core of our parser for
# formulas, but is generic enough to be used for other purposes as well
# (e.g. parsing linear constraints). It just builds a parse tree; semantics
# are somebody else's problem.
#
# Plus it spends energy on tracking where each item in the parse tree comes
# from, to allow high-quality error reporting.
#
# You are expected to provide an collection of Operators, a collection of
# atomic types, and an iterator that provides Tokens. Each Operator should
# have a unique token_type (which is an arbitrary Python object), and each
# Token should have a matching token_type, or one of the special types
# Token.LPAREN, Token.RPAREN. Each Token is required to have a valid Origin
# attached, for error reporting.

# XX: still seriously consider putting the magic intercept handling into the
# tokenizer. we'd still need separate term-sets that get pasted together by ~
# to create the modeldesc, though... heck maybe we should just have a
# modeldesc be 1-or-more termsets, with the convention that if it's 1, then
# it's a rhs, and if it's 2, it's (lhs, rhs), and otherwise you're on your
# own. Test: would this be useful for multiple-group log-linear models,
# maybe? Answer: Perhaps. outcome ~ x1 + x2 ~ group. But lots of other
# plausible, maybe better ways to write this -- (outcome | group) ~ x1 + x2?
# "outcome ~ x1 + x2", group="group"? etc.

from __future__ import print_function

__all__ = ["Token", "ParseNode", "Operator", "parse"]

from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
                        no_pickling, assert_no_pickling)

class _UniqueValue(object):
    def __init__(self, print_as):
        self._print_as = print_as

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._print_as)

    __getstate__ = no_pickling

class Token(object):
    """A token with possible payload.

    .. attribute:: type

       An arbitrary object indicating the type of this token. Should be
      :term:`hashable`, but otherwise it can be whatever you like.
    """
    LPAREN = _UniqueValue("LPAREN")
    RPAREN = _UniqueValue("RPAREN")

    def __init__(self, type, origin, extra=None):
        self.type = type
        self.origin = origin
        self.extra = extra

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        kwargs = []
        if self.extra is not None:
            kwargs = [("extra", self.extra)]
        return repr_pretty_impl(p, self, [self.type, self.origin], kwargs)

    __getstate__ = no_pickling

class ParseNode(object):
    def __init__(self, type, token, args, origin):
        self.type = type
        self.token = token
        self.args = args
        self.origin = origin

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        return repr_pretty_impl(p, self, [self.type, self.token, self.args])

    __getstate__ = no_pickling

class Operator(object):
    def __init__(self, token_type, arity, precedence):
        self.token_type = token_type
        self.arity = arity
        self.precedence = precedence

    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__,
                                   self.token_type, self.arity, self.precedence)

    __getstate__ = no_pickling

class _StackOperator(object):
    def __init__(self, op, token):
        self.op = op
        self.token = token

    __getstate__ = no_pickling

_open_paren = Operator(Token.LPAREN, -1, -9999999)

class _ParseContext(object):
    def __init__(self, unary_ops, binary_ops, atomic_types, trace):
        self.op_stack = []
        self.noun_stack = []
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.atomic_types = atomic_types
        self.trace = trace

    __getstate__ = no_pickling

def _read_noun_context(token, c):
    if token.type == Token.LPAREN:
        if c.trace:
            print("Pushing open-paren")
        c.op_stack.append(_StackOperator(_open_paren, token))
        return True
    elif token.type in c.unary_ops:
        if c.trace:
            print("Pushing unary op %r" % (token.type,))
        c.op_stack.append(_StackOperator(c.unary_ops[token.type], token))
        return True
    elif token.type in c.atomic_types:
        if c.trace:
            print("Pushing noun %r (%r)" % (token.type, token.extra))
        c.noun_stack.append(ParseNode(token.type, token, [],
                                      token.origin))
        return False
    else:
        raise PatsyError("expected a noun, not '%s'"
                            % (token.origin.relevant_code(),),
                            token)

def _run_op(c):
    assert c.op_stack
    stackop = c.op_stack.pop()
    args = []
    for i in range(stackop.op.arity):
        args.append(c.noun_stack.pop())
    args.reverse()
    if c.trace:
        print("Reducing %r (%r)" % (stackop.op.token_type, args))
    node = ParseNode(stackop.op.token_type, stackop.token, args,
                     Origin.combine([stackop.token] + args))
    c.noun_stack.append(node)

def _read_op_context(token, c):
    if token.type == Token.RPAREN:
        if c.trace:
            print("Found close-paren")
        while c.op_stack and c.op_stack[-1].op.token_type != Token.LPAREN:
            _run_op(c)
        if not c.op_stack:
            raise PatsyError("missing '(' or extra ')'", token)
        assert c.op_stack[-1].op.token_type == Token.LPAREN
        # Expand the origin of the item on top of the noun stack to include
        # the open and close parens:
        combined = Origin.combine([c.op_stack[-1].token,
                                   c.noun_stack[-1].token,
                                   token])
        c.noun_stack[-1].origin = combined
        # Pop the open-paren
        c.op_stack.pop()
        return False
    elif token.type in c.binary_ops:
        if c.trace:
            print("Found binary operator %r" % (token.type))
        stackop = _StackOperator(c.binary_ops[token.type], token)
        while (c.op_stack
               and stackop.op.precedence <= c.op_stack[-1].op.precedence):
            _run_op(c)
        if c.trace:
            print("Pushing binary operator %r" % (token.type))
        c.op_stack.append(stackop)
        return True
    else:
        raise PatsyError("expected an operator, not '%s'"
                            % (token.origin.relevant_code(),),
                            token)

def infix_parse(tokens, operators, atomic_types, trace=False):
    token_source = iter(tokens)

    unary_ops = {}
    binary_ops = {}
    for op in operators:
        assert op.precedence > _open_paren.precedence
        if op.arity == 1:
            unary_ops[op.token_type] = op
        elif op.arity == 2:
            binary_ops[op.token_type] = op
        else:
            raise ValueError("operators must be unary or binary")

    c = _ParseContext(unary_ops, binary_ops, atomic_types, trace)

    # This is an implementation of Dijkstra's shunting yard algorithm:
    #   http://en.wikipedia.org/wiki/Shunting_yard_algorithm
    #   http://www.engr.mun.ca/~theo/Misc/exp_parsing.htm

    want_noun = True
    for token in token_source:
        if c.trace:
            print("Reading next token (want_noun=%r)" % (want_noun,))
        if want_noun:
            want_noun = _read_noun_context(token, c)
        else:
            want_noun = _read_op_context(token, c)
    if c.trace:
        print("End of token stream")

    if want_noun:
        raise PatsyError("expected a noun, but instead the expression ended",
                            c.op_stack[-1].token.origin)

    while c.op_stack:
        if c.op_stack[-1].op.token_type == Token.LPAREN:
            raise PatsyError("Unmatched '('", c.op_stack[-1].token)
        _run_op(c)

    assert len(c.noun_stack) == 1
    return c.noun_stack.pop()

# Much more thorough tests in parse_formula.py, this is just a smoke test:
def test_infix_parse():
    ops = [Operator("+", 2, 10),
           Operator("*", 2, 20),
           Operator("-", 1, 30)]
    atomic = ["ATOM1", "ATOM2"]
    # a + -b * (c + d)
    mock_origin = Origin("asdf", 2, 3)
    tokens = [Token("ATOM1", mock_origin, "a"),
              Token("+", mock_origin, "+"),
              Token("-", mock_origin, "-"),
              Token("ATOM2", mock_origin, "b"),
              Token("*", mock_origin, "*"),
              Token(Token.LPAREN, mock_origin, "("),
              Token("ATOM1", mock_origin, "c"),
              Token("+", mock_origin, "+"),
              Token("ATOM2", mock_origin, "d"),
              Token(Token.RPAREN, mock_origin, ")")]
    tree = infix_parse(tokens, ops, atomic)
    def te(tree, type, extra):
        assert tree.type == type
        assert tree.token.extra == extra
    te(tree, "+", "+")
    te(tree.args[0], "ATOM1", "a")
    assert tree.args[0].args == []
    te(tree.args[1], "*", "*")
    te(tree.args[1].args[0], "-", "-")
    assert len(tree.args[1].args[0].args) == 1
    te(tree.args[1].args[0].args[0], "ATOM2", "b")
    te(tree.args[1].args[1], "+", "+")
    te(tree.args[1].args[1].args[0], "ATOM1", "c")
    te(tree.args[1].args[1].args[1], "ATOM2", "d")

    import pytest
    # No ternary ops
    pytest.raises(ValueError,
                  infix_parse, [], [Operator("+", 3, 10)], ["ATOMIC"])

    # smoke test just to make sure there are no egregious bugs in 'trace'
    infix_parse(tokens, ops, atomic, trace=True)
