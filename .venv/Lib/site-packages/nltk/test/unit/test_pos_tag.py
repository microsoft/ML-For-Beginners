"""
Tests for nltk.pos_tag
"""


import unittest

from nltk import pos_tag, word_tokenize


class TestPosTag(unittest.TestCase):
    def test_pos_tag_eng(self):
        text = "John's big idea isn't all that bad."
        expected_tagged = [
            ("John", "NNP"),
            ("'s", "POS"),
            ("big", "JJ"),
            ("idea", "NN"),
            ("is", "VBZ"),
            ("n't", "RB"),
            ("all", "PDT"),
            ("that", "DT"),
            ("bad", "JJ"),
            (".", "."),
        ]
        assert pos_tag(word_tokenize(text)) == expected_tagged

    def test_pos_tag_eng_universal(self):
        text = "John's big idea isn't all that bad."
        expected_tagged = [
            ("John", "NOUN"),
            ("'s", "PRT"),
            ("big", "ADJ"),
            ("idea", "NOUN"),
            ("is", "VERB"),
            ("n't", "ADV"),
            ("all", "DET"),
            ("that", "DET"),
            ("bad", "ADJ"),
            (".", "."),
        ]
        assert pos_tag(word_tokenize(text), tagset="universal") == expected_tagged

    def test_pos_tag_rus(self):
        text = "Илья оторопел и дважды перечитал бумажку."
        expected_tagged = [
            ("Илья", "S"),
            ("оторопел", "V"),
            ("и", "CONJ"),
            ("дважды", "ADV"),
            ("перечитал", "V"),
            ("бумажку", "S"),
            (".", "NONLEX"),
        ]
        assert pos_tag(word_tokenize(text), lang="rus") == expected_tagged

    def test_pos_tag_rus_universal(self):
        text = "Илья оторопел и дважды перечитал бумажку."
        expected_tagged = [
            ("Илья", "NOUN"),
            ("оторопел", "VERB"),
            ("и", "CONJ"),
            ("дважды", "ADV"),
            ("перечитал", "VERB"),
            ("бумажку", "NOUN"),
            (".", "."),
        ]
        assert (
            pos_tag(word_tokenize(text), tagset="universal", lang="rus")
            == expected_tagged
        )

    def test_pos_tag_unknown_lang(self):
        text = "모르겠 습니 다"
        self.assertRaises(NotImplementedError, pos_tag, word_tokenize(text), lang="kor")
        # Test for default kwarg, `lang=None`
        self.assertRaises(NotImplementedError, pos_tag, word_tokenize(text), lang=None)

    def test_unspecified_lang(self):
        # Tries to force the lang='eng' option.
        text = "모르겠 습니 다"
        expected_but_wrong = [("모르겠", "JJ"), ("습니", "NNP"), ("다", "NN")]
        assert pos_tag(word_tokenize(text)) == expected_but_wrong
