import unittest

from nltk import RegexpParser


class TestChunkRule(unittest.TestCase):
    def test_tag_pattern2re_pattern_quantifier(self):
        """Test for bug https://github.com/nltk/nltk/issues/1597

        Ensures that curly bracket quantifiers can be used inside a chunk rule.
        This type of quantifier has been used for the supplementary example
        in https://www.nltk.org/book/ch07.html#exploring-text-corpora.
        """
        sent = [
            ("The", "AT"),
            ("September-October", "NP"),
            ("term", "NN"),
            ("jury", "NN"),
            ("had", "HVD"),
            ("been", "BEN"),
            ("charged", "VBN"),
            ("by", "IN"),
            ("Fulton", "NP-TL"),
            ("Superior", "JJ-TL"),
            ("Court", "NN-TL"),
            ("Judge", "NN-TL"),
            ("Durwood", "NP"),
            ("Pye", "NP"),
            ("to", "TO"),
            ("investigate", "VB"),
            ("reports", "NNS"),
            ("of", "IN"),
            ("possible", "JJ"),
            ("``", "``"),
            ("irregularities", "NNS"),
            ("''", "''"),
            ("in", "IN"),
            ("the", "AT"),
            ("hard-fought", "JJ"),
            ("primary", "NN"),
            ("which", "WDT"),
            ("was", "BEDZ"),
            ("won", "VBN"),
            ("by", "IN"),
            ("Mayor-nominate", "NN-TL"),
            ("Ivan", "NP"),
            ("Allen", "NP"),
            ("Jr.", "NP"),
            (".", "."),
        ]  # source: brown corpus
        cp = RegexpParser("CHUNK: {<N.*>{4,}}")
        tree = cp.parse(sent)
        assert (
            tree.pformat()
            == """(S
  The/AT
  September-October/NP
  term/NN
  jury/NN
  had/HVD
  been/BEN
  charged/VBN
  by/IN
  Fulton/NP-TL
  Superior/JJ-TL
  (CHUNK Court/NN-TL Judge/NN-TL Durwood/NP Pye/NP)
  to/TO
  investigate/VB
  reports/NNS
  of/IN
  possible/JJ
  ``/``
  irregularities/NNS
  ''/''
  in/IN
  the/AT
  hard-fought/JJ
  primary/NN
  which/WDT
  was/BEDZ
  won/VBN
  by/IN
  (CHUNK Mayor-nominate/NN-TL Ivan/NP Allen/NP Jr./NP)
  ./.)"""
        )
