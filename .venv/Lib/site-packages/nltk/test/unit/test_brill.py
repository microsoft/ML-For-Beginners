"""
Tests for Brill tagger.
"""

import unittest

from nltk.corpus import treebank
from nltk.tag import UnigramTagger, brill, brill_trainer
from nltk.tbl import demo


class TestBrill(unittest.TestCase):
    def test_pos_template(self):
        train_sents = treebank.tagged_sents()[:1000]
        tagger = UnigramTagger(train_sents)
        trainer = brill_trainer.BrillTaggerTrainer(
            tagger, [brill.Template(brill.Pos([-1]))]
        )
        brill_tagger = trainer.train(train_sents)
        # Example from https://github.com/nltk/nltk/issues/769
        result = brill_tagger.tag("This is a foo bar sentence".split())
        expected = [
            ("This", "DT"),
            ("is", "VBZ"),
            ("a", "DT"),
            ("foo", None),
            ("bar", "NN"),
            ("sentence", None),
        ]
        self.assertEqual(result, expected)

    @unittest.skip("Should be tested in __main__ of nltk.tbl.demo")
    def test_brill_demo(self):
        demo()
