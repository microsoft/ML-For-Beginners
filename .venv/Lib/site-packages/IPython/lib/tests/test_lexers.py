"""Test lexers module"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from unittest import TestCase
from pygments import __version__ as pygments_version
from pygments.token import Token
from pygments.lexers import BashLexer

from .. import lexers

pyg214 = tuple(int(x) for x in pygments_version.split(".")[:2]) >= (2, 14)


class TestLexers(TestCase):
    """Collection of lexers tests"""
    def setUp(self):
        self.lexer = lexers.IPythonLexer()
        self.bash_lexer = BashLexer()

    def testIPythonLexer(self):
        fragment = '!echo $HOME\n'
        bash_tokens = [
            (Token.Operator, '!'),
        ]
        bash_tokens.extend(self.bash_lexer.get_tokens(fragment[1:]))
        ipylex_token = list(self.lexer.get_tokens(fragment))
        assert bash_tokens[:-1] == ipylex_token[:-1]

        fragment_2 = "!" + fragment
        tokens_2 = [
            (Token.Operator, '!!'),
        ] + bash_tokens[1:]
        assert tokens_2[:-1] == list(self.lexer.get_tokens(fragment_2))[:-1]

        fragment_2 = '\t %%!\n' + fragment[1:]
        tokens_2 = [
            (Token.Text, '\t '),
            (Token.Operator, '%%!'),
            (Token.Text, '\n'),
        ] + bash_tokens[1:]
        assert tokens_2 == list(self.lexer.get_tokens(fragment_2))

        fragment_2 = 'x = ' + fragment
        tokens_2 = [
            (Token.Name, 'x'),
            (Token.Text, ' '),
            (Token.Operator, '='),
            (Token.Text, ' '),
        ] + bash_tokens
        assert tokens_2[:-1] == list(self.lexer.get_tokens(fragment_2))[:-1]

        fragment_2 = 'x, = ' + fragment
        tokens_2 = [
            (Token.Name, 'x'),
            (Token.Punctuation, ','),
            (Token.Text, ' '),
            (Token.Operator, '='),
            (Token.Text, ' '),
        ] + bash_tokens
        assert tokens_2[:-1] == list(self.lexer.get_tokens(fragment_2))[:-1]

        fragment_2 = 'x, = %sx ' + fragment[1:]
        tokens_2 = [
            (Token.Name, 'x'),
            (Token.Punctuation, ','),
            (Token.Text, ' '),
            (Token.Operator, '='),
            (Token.Text, ' '),
            (Token.Operator, '%'),
            (Token.Keyword, 'sx'),
            (Token.Text, ' '),
        ] + bash_tokens[1:]
        if tokens_2[7] == (Token.Text, " ") and pyg214:  # pygments 2.14+
            tokens_2[7] = (Token.Text.Whitespace, " ")
        assert tokens_2[:-1] == list(self.lexer.get_tokens(fragment_2))[:-1]

        fragment_2 = 'f = %R function () {}\n'
        tokens_2 = [
            (Token.Name, 'f'),
            (Token.Text, ' '),
            (Token.Operator, '='),
            (Token.Text, ' '),
            (Token.Operator, '%'),
            (Token.Keyword, 'R'),
            (Token.Text, ' function () {}\n'),
        ]
        assert tokens_2 == list(self.lexer.get_tokens(fragment_2))

        fragment_2 = '\t%%xyz\n$foo\n'
        tokens_2 = [
            (Token.Text, '\t'),
            (Token.Operator, '%%'),
            (Token.Keyword, 'xyz'),
            (Token.Text, '\n$foo\n'),
        ]
        assert tokens_2 == list(self.lexer.get_tokens(fragment_2))

        fragment_2 = '%system?\n'
        tokens_2 = [
            (Token.Operator, '%'),
            (Token.Keyword, 'system'),
            (Token.Operator, '?'),
            (Token.Text, '\n'),
        ]
        assert tokens_2[:-1] == list(self.lexer.get_tokens(fragment_2))[:-1]

        fragment_2 = 'x != y\n'
        tokens_2 = [
            (Token.Name, 'x'),
            (Token.Text, ' '),
            (Token.Operator, '!='),
            (Token.Text, ' '),
            (Token.Name, 'y'),
            (Token.Text, '\n'),
        ]
        assert tokens_2[:-1] == list(self.lexer.get_tokens(fragment_2))[:-1]

        fragment_2 = ' ?math.sin\n'
        tokens_2 = [
            (Token.Text, ' '),
            (Token.Operator, '?'),
            (Token.Text, 'math.sin'),
            (Token.Text, '\n'),
        ]
        assert tokens_2[:-1] == list(self.lexer.get_tokens(fragment_2))[:-1]

        fragment = ' *int*?\n'
        tokens = [
            (Token.Text, ' *int*'),
            (Token.Operator, '?'),
            (Token.Text, '\n'),
        ]
        assert tokens == list(self.lexer.get_tokens(fragment))

        fragment = '%%writefile -a foo.py\nif a == b:\n    pass'
        tokens = [
            (Token.Operator, '%%writefile'),
            (Token.Text, ' -a foo.py\n'),
            (Token.Keyword, 'if'),
            (Token.Text, ' '),
            (Token.Name, 'a'),
            (Token.Text, ' '),
            (Token.Operator, '=='),
            (Token.Text, ' '),
            (Token.Name, 'b'),
            (Token.Punctuation, ':'),
            (Token.Text, '\n'),
            (Token.Text, '    '),
            (Token.Keyword, 'pass'),
            (Token.Text, '\n'),
        ]
        if tokens[10] == (Token.Text, "\n") and pyg214:  # pygments 2.14+
            tokens[10] = (Token.Text.Whitespace, "\n")
        assert tokens[:-1] == list(self.lexer.get_tokens(fragment))[:-1]

        fragment = '%%timeit\nmath.sin(0)'
        tokens = [
            (Token.Operator, '%%timeit\n'),
            (Token.Name, 'math'),
            (Token.Operator, '.'),
            (Token.Name, 'sin'),
            (Token.Punctuation, '('),
            (Token.Literal.Number.Integer, '0'),
            (Token.Punctuation, ')'),
            (Token.Text, '\n'),
        ]

        fragment = '%%HTML\n<div>foo</div>'
        tokens = [
            (Token.Operator, '%%HTML'),
            (Token.Text, '\n'),
            (Token.Punctuation, '<'),
            (Token.Name.Tag, 'div'),
            (Token.Punctuation, '>'),
            (Token.Text, 'foo'),
            (Token.Punctuation, '<'),
            (Token.Punctuation, '/'),
            (Token.Name.Tag, 'div'),
            (Token.Punctuation, '>'),
            (Token.Text, '\n'),
        ]
        assert tokens == list(self.lexer.get_tokens(fragment))
