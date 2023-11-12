import unittest

import nltk
from nltk.grammar import CFG


class ChomskyNormalFormForCFGTest(unittest.TestCase):
    def test_simple(self):
        grammar = CFG.fromstring(
            """
          S -> NP VP
          PP -> P NP
          NP -> Det N | NP PP P
          VP -> V NP | VP PP
          VP -> Det
          Det -> 'a' | 'the'
          N -> 'dog' | 'cat'
          V -> 'chased' | 'sat'
          P -> 'on' | 'in'
        """
        )
        self.assertFalse(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())
        grammar = grammar.chomsky_normal_form(flexible=True)
        self.assertTrue(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())

        grammar2 = CFG.fromstring(
            """
          S -> NP VP
          NP -> VP N P
          VP -> P
          N -> 'dog' | 'cat'
          P -> 'on' | 'in'
        """
        )
        self.assertFalse(grammar2.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar2.is_chomsky_normal_form())
        grammar2 = grammar2.chomsky_normal_form()
        self.assertTrue(grammar2.is_flexible_chomsky_normal_form())
        self.assertTrue(grammar2.is_chomsky_normal_form())

    def test_complex(self):
        grammar = nltk.data.load("grammars/large_grammars/atis.cfg")
        self.assertFalse(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())
        grammar = grammar.chomsky_normal_form(flexible=True)
        self.assertTrue(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())
