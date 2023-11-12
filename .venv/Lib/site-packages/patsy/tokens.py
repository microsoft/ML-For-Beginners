# This file is part of Patsy
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Utilities for dealing with Python code at the token level.
#
# Includes:
#   a "pretty printer" that converts a sequence of tokens back into a
#       readable, white-space normalized string.
#   a utility function to replace calls to global functions with calls to
#       other functions

import tokenize
from six.moves import cStringIO as StringIO

from patsy import PatsyError
from patsy.origin import Origin

__all__ = ["python_tokenize", "pretty_untokenize",
           "normalize_token_spacing"]

# A convenience wrapper around tokenize.generate_tokens. yields tuples
#   (tokenize type, token string, origin object)
def python_tokenize(code):
    # Since formulas can only contain Python expressions, and Python
    # expressions cannot meaningfully contain newlines, we'll just remove all
    # the newlines up front to avoid any complications:
    code = code.replace("\n", " ").strip()
    it = tokenize.generate_tokens(StringIO(code).readline)
    try:
        for (pytype, string, (_, start), (_, end), code) in it:
            if pytype == tokenize.ENDMARKER:
                break
            origin = Origin(code, start, end)
            assert pytype != tokenize.NL
            if pytype == tokenize.NEWLINE:
                assert string == ""
                continue
            if pytype == tokenize.ERRORTOKEN:
                raise PatsyError("error tokenizing input "
                                 "(maybe an unclosed string?)",
                                 origin)
            if pytype == tokenize.COMMENT:
                raise PatsyError("comments are not allowed", origin)
            yield (pytype, string, origin)
        else: # pragma: no cover
            raise ValueError("stream ended without ENDMARKER?!?")
    except tokenize.TokenError as e:
        # TokenError is raised iff the tokenizer thinks that there is
        # some sort of multi-line construct in progress (e.g., an
        # unclosed parentheses, which in Python lets a virtual line
        # continue past the end of the physical line), and it hits the
        # end of the source text. We have our own error handling for
        # such cases, so just treat this as an end-of-stream.
        #
        # Just in case someone adds some other error case:
        assert e.args[0].startswith("EOF in multi-line")
        return

def test_python_tokenize():
    code = "a + (foo * -1)"
    tokens = list(python_tokenize(code))
    expected = [(tokenize.NAME, "a", Origin(code, 0, 1)),
                (tokenize.OP, "+", Origin(code, 2, 3)),
                (tokenize.OP, "(", Origin(code, 4, 5)),
                (tokenize.NAME, "foo", Origin(code, 5, 8)),
                (tokenize.OP, "*", Origin(code, 9, 10)),
                (tokenize.OP, "-", Origin(code, 11, 12)),
                (tokenize.NUMBER, "1", Origin(code, 12, 13)),
                (tokenize.OP, ")", Origin(code, 13, 14))]
    assert tokens == expected

    code2 = "a + (b"
    tokens2 = list(python_tokenize(code2))
    expected2 = [(tokenize.NAME, "a", Origin(code2, 0, 1)),
                 (tokenize.OP, "+", Origin(code2, 2, 3)),
                 (tokenize.OP, "(", Origin(code2, 4, 5)),
                 (tokenize.NAME, "b", Origin(code2, 5, 6))]
    assert tokens2 == expected2

    import pytest
    pytest.raises(PatsyError, list, python_tokenize("a b # c"))

    import pytest
    pytest.raises(PatsyError, list, python_tokenize("a b \"c"))

_python_space_both = (list("+-*/%&^|<>")
                      + ["==", "<>", "!=", "<=", ">=",
                         "<<", ">>", "**", "//"])
_python_space_before = (_python_space_both
                        + ["!", "~"])
_python_space_after = (_python_space_both
                       + [",", ":"])

def pretty_untokenize(typed_tokens):
    text = []
    prev_was_space_delim = False
    prev_wants_space = False
    prev_was_open_paren_or_comma = False
    prev_was_object_like = False
    brackets = []
    for token_type, token in typed_tokens:
        assert token_type not in (tokenize.INDENT, tokenize.DEDENT,
                                  tokenize.NL)
        if token_type == tokenize.NEWLINE:
            continue
        if token_type == tokenize.ENDMARKER:
            continue
        if token_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
            if prev_wants_space or prev_was_space_delim:
                text.append(" ")
            text.append(token)
            prev_wants_space = False
            prev_was_space_delim = True
        else:
            if token in ("(", "[", "{"):
                brackets.append(token)
            elif brackets and token in (")", "]", "}"):
                brackets.pop()
            this_wants_space_before = (token in _python_space_before)
            this_wants_space_after = (token in _python_space_after)
            # Special case for slice syntax: foo[:10]
            # Otherwise ":" is spaced after, like: "{1: ...}", "if a: ..."
            if token == ":" and brackets and brackets[-1] == "[":
                this_wants_space_after = False
            # Special case for foo(*args), foo(a, *args):
            if token in ("*", "**") and prev_was_open_paren_or_comma:
                this_wants_space_before = False
                this_wants_space_after = False
            # Special case for "a = foo(b=1)":
            if token == "=" and not brackets:
                this_wants_space_before = True
                this_wants_space_after = True
            # Special case for unary -, +. Our heuristic is that if we see the
            # + or - after something that looks like an object (a NAME,
            # NUMBER, STRING, or close paren) then it is probably binary,
            # otherwise it is probably unary.
            if token in ("+", "-") and not prev_was_object_like:
                this_wants_space_before = False
                this_wants_space_after = False
            if prev_wants_space or this_wants_space_before:
                text.append(" ")
            text.append(token)
            prev_wants_space = this_wants_space_after
            prev_was_space_delim = False
        if (token_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING)
            or token == ")"):
            prev_was_object_like = True
        else:
            prev_was_object_like = False
        prev_was_open_paren_or_comma = token in ("(", ",")
    return "".join(text)

def normalize_token_spacing(code):
    tokens = [(t[0], t[1])
              for t in tokenize.generate_tokens(StringIO(code).readline)]
    return pretty_untokenize(tokens)

def test_pretty_untokenize_and_normalize_token_spacing():
    assert normalize_token_spacing("1 + 1") == "1 + 1"
    assert normalize_token_spacing("1+1") == "1 + 1"
    assert normalize_token_spacing("1*(2+3**2)") == "1 * (2 + 3 ** 2)"
    assert normalize_token_spacing("a and b") == "a and b"
    assert normalize_token_spacing("foo(a=bar.baz[1:])") == "foo(a=bar.baz[1:])"
    assert normalize_token_spacing("""{"hi":foo[:]}""") == """{"hi": foo[:]}"""
    assert normalize_token_spacing("""'a' "b" 'c'""") == """'a' "b" 'c'"""
    assert normalize_token_spacing('"""a""" is 1 or 2==3') == '"""a""" is 1 or 2 == 3'
    assert normalize_token_spacing("foo ( * args )") == "foo(*args)"
    assert normalize_token_spacing("foo ( a * args )") == "foo(a * args)"
    assert normalize_token_spacing("foo ( ** args )") == "foo(**args)"
    assert normalize_token_spacing("foo ( a ** args )") == "foo(a ** args)"
    assert normalize_token_spacing("foo (1, * args )") == "foo(1, *args)"
    assert normalize_token_spacing("foo (1, a * args )") == "foo(1, a * args)"
    assert normalize_token_spacing("foo (1, ** args )") == "foo(1, **args)"
    assert normalize_token_spacing("foo (1, a ** args )") == "foo(1, a ** args)"

    assert normalize_token_spacing("a=foo(b = 1)") == "a = foo(b=1)"

    assert normalize_token_spacing("foo(+ 10, bar = - 1)") == "foo(+10, bar=-1)"
    assert normalize_token_spacing("1 + +10 + -1 - 5") == "1 + +10 + -1 - 5"
