"""
Mock test for Stanford CoreNLP wrappers.
"""

from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from nltk.parse import corenlp
from nltk.tree import Tree


def setup_module(module):
    global server

    try:
        server = corenlp.CoreNLPServer(port=9000)
    except LookupError:
        pytest.skip("Could not instantiate CoreNLPServer.")

    try:
        server.start()
    except corenlp.CoreNLPServerError as e:
        pytest.skip(
            "Skipping CoreNLP tests because the server could not be started. "
            "Make sure that the 9000 port is free. "
            "{}".format(e.strerror)
        )


def teardown_module(module):
    server.stop()


class TestTokenizerAPI(TestCase):
    def test_tokenize(self):
        corenlp_tokenizer = corenlp.CoreNLPParser()

        api_return_value = {
            "sentences": [
                {
                    "index": 0,
                    "tokens": [
                        {
                            "after": " ",
                            "before": "",
                            "characterOffsetBegin": 0,
                            "characterOffsetEnd": 4,
                            "index": 1,
                            "originalText": "Good",
                            "word": "Good",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 5,
                            "characterOffsetEnd": 12,
                            "index": 2,
                            "originalText": "muffins",
                            "word": "muffins",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 13,
                            "characterOffsetEnd": 17,
                            "index": 3,
                            "originalText": "cost",
                            "word": "cost",
                        },
                        {
                            "after": "",
                            "before": " ",
                            "characterOffsetBegin": 18,
                            "characterOffsetEnd": 19,
                            "index": 4,
                            "originalText": "$",
                            "word": "$",
                        },
                        {
                            "after": "\n",
                            "before": "",
                            "characterOffsetBegin": 19,
                            "characterOffsetEnd": 23,
                            "index": 5,
                            "originalText": "3.88",
                            "word": "3.88",
                        },
                        {
                            "after": " ",
                            "before": "\n",
                            "characterOffsetBegin": 24,
                            "characterOffsetEnd": 26,
                            "index": 6,
                            "originalText": "in",
                            "word": "in",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 27,
                            "characterOffsetEnd": 30,
                            "index": 7,
                            "originalText": "New",
                            "word": "New",
                        },
                        {
                            "after": "",
                            "before": " ",
                            "characterOffsetBegin": 31,
                            "characterOffsetEnd": 35,
                            "index": 8,
                            "originalText": "York",
                            "word": "York",
                        },
                        {
                            "after": "  ",
                            "before": "",
                            "characterOffsetBegin": 35,
                            "characterOffsetEnd": 36,
                            "index": 9,
                            "originalText": ".",
                            "word": ".",
                        },
                    ],
                },
                {
                    "index": 1,
                    "tokens": [
                        {
                            "after": " ",
                            "before": "  ",
                            "characterOffsetBegin": 38,
                            "characterOffsetEnd": 44,
                            "index": 1,
                            "originalText": "Please",
                            "word": "Please",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 45,
                            "characterOffsetEnd": 48,
                            "index": 2,
                            "originalText": "buy",
                            "word": "buy",
                        },
                        {
                            "after": "\n",
                            "before": " ",
                            "characterOffsetBegin": 49,
                            "characterOffsetEnd": 51,
                            "index": 3,
                            "originalText": "me",
                            "word": "me",
                        },
                        {
                            "after": " ",
                            "before": "\n",
                            "characterOffsetBegin": 52,
                            "characterOffsetEnd": 55,
                            "index": 4,
                            "originalText": "two",
                            "word": "two",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 56,
                            "characterOffsetEnd": 58,
                            "index": 5,
                            "originalText": "of",
                            "word": "of",
                        },
                        {
                            "after": "",
                            "before": " ",
                            "characterOffsetBegin": 59,
                            "characterOffsetEnd": 63,
                            "index": 6,
                            "originalText": "them",
                            "word": "them",
                        },
                        {
                            "after": "\n",
                            "before": "",
                            "characterOffsetBegin": 63,
                            "characterOffsetEnd": 64,
                            "index": 7,
                            "originalText": ".",
                            "word": ".",
                        },
                    ],
                },
                {
                    "index": 2,
                    "tokens": [
                        {
                            "after": "",
                            "before": "\n",
                            "characterOffsetBegin": 65,
                            "characterOffsetEnd": 71,
                            "index": 1,
                            "originalText": "Thanks",
                            "word": "Thanks",
                        },
                        {
                            "after": "",
                            "before": "",
                            "characterOffsetBegin": 71,
                            "characterOffsetEnd": 72,
                            "index": 2,
                            "originalText": ".",
                            "word": ".",
                        },
                    ],
                },
            ]
        }
        corenlp_tokenizer.api_call = MagicMock(return_value=api_return_value)

        input_string = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."

        expected_output = [
            "Good",
            "muffins",
            "cost",
            "$",
            "3.88",
            "in",
            "New",
            "York",
            ".",
            "Please",
            "buy",
            "me",
            "two",
            "of",
            "them",
            ".",
            "Thanks",
            ".",
        ]

        tokenized_output = list(corenlp_tokenizer.tokenize(input_string))

        corenlp_tokenizer.api_call.assert_called_once_with(
            "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.",
            properties={"annotators": "tokenize,ssplit"},
        )
        self.assertEqual(expected_output, tokenized_output)


class TestTaggerAPI(TestCase):
    def test_pos_tagger(self):
        corenlp_tagger = corenlp.CoreNLPParser(tagtype="pos")

        api_return_value = {
            "sentences": [
                {
                    "basicDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 1,
                            "dependentGloss": "What",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "cop",
                            "dependent": 2,
                            "dependentGloss": "is",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                        {
                            "dep": "det",
                            "dependent": 3,
                            "dependentGloss": "the",
                            "governor": 4,
                            "governorGloss": "airspeed",
                        },
                        {
                            "dep": "nsubj",
                            "dependent": 4,
                            "dependentGloss": "airspeed",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                        {
                            "dep": "case",
                            "dependent": 5,
                            "dependentGloss": "of",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "det",
                            "dependent": 6,
                            "dependentGloss": "an",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "compound",
                            "dependent": 7,
                            "dependentGloss": "unladen",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "nmod",
                            "dependent": 8,
                            "dependentGloss": "swallow",
                            "governor": 4,
                            "governorGloss": "airspeed",
                        },
                        {
                            "dep": "punct",
                            "dependent": 9,
                            "dependentGloss": "?",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                    ],
                    "enhancedDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 1,
                            "dependentGloss": "What",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "cop",
                            "dependent": 2,
                            "dependentGloss": "is",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                        {
                            "dep": "det",
                            "dependent": 3,
                            "dependentGloss": "the",
                            "governor": 4,
                            "governorGloss": "airspeed",
                        },
                        {
                            "dep": "nsubj",
                            "dependent": 4,
                            "dependentGloss": "airspeed",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                        {
                            "dep": "case",
                            "dependent": 5,
                            "dependentGloss": "of",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "det",
                            "dependent": 6,
                            "dependentGloss": "an",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "compound",
                            "dependent": 7,
                            "dependentGloss": "unladen",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "nmod:of",
                            "dependent": 8,
                            "dependentGloss": "swallow",
                            "governor": 4,
                            "governorGloss": "airspeed",
                        },
                        {
                            "dep": "punct",
                            "dependent": 9,
                            "dependentGloss": "?",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                    ],
                    "enhancedPlusPlusDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 1,
                            "dependentGloss": "What",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "cop",
                            "dependent": 2,
                            "dependentGloss": "is",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                        {
                            "dep": "det",
                            "dependent": 3,
                            "dependentGloss": "the",
                            "governor": 4,
                            "governorGloss": "airspeed",
                        },
                        {
                            "dep": "nsubj",
                            "dependent": 4,
                            "dependentGloss": "airspeed",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                        {
                            "dep": "case",
                            "dependent": 5,
                            "dependentGloss": "of",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "det",
                            "dependent": 6,
                            "dependentGloss": "an",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "compound",
                            "dependent": 7,
                            "dependentGloss": "unladen",
                            "governor": 8,
                            "governorGloss": "swallow",
                        },
                        {
                            "dep": "nmod:of",
                            "dependent": 8,
                            "dependentGloss": "swallow",
                            "governor": 4,
                            "governorGloss": "airspeed",
                        },
                        {
                            "dep": "punct",
                            "dependent": 9,
                            "dependentGloss": "?",
                            "governor": 1,
                            "governorGloss": "What",
                        },
                    ],
                    "index": 0,
                    "parse": "(ROOT\n  (SBARQ\n    (WHNP (WP What))\n    (SQ (VBZ is)\n      (NP\n        (NP (DT the) (NN airspeed))\n        (PP (IN of)\n          (NP (DT an) (NN unladen) (NN swallow)))))\n    (. ?)))",
                    "tokens": [
                        {
                            "after": " ",
                            "before": "",
                            "characterOffsetBegin": 0,
                            "characterOffsetEnd": 4,
                            "index": 1,
                            "lemma": "what",
                            "originalText": "What",
                            "pos": "WP",
                            "word": "What",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 5,
                            "characterOffsetEnd": 7,
                            "index": 2,
                            "lemma": "be",
                            "originalText": "is",
                            "pos": "VBZ",
                            "word": "is",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 8,
                            "characterOffsetEnd": 11,
                            "index": 3,
                            "lemma": "the",
                            "originalText": "the",
                            "pos": "DT",
                            "word": "the",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 12,
                            "characterOffsetEnd": 20,
                            "index": 4,
                            "lemma": "airspeed",
                            "originalText": "airspeed",
                            "pos": "NN",
                            "word": "airspeed",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 21,
                            "characterOffsetEnd": 23,
                            "index": 5,
                            "lemma": "of",
                            "originalText": "of",
                            "pos": "IN",
                            "word": "of",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 24,
                            "characterOffsetEnd": 26,
                            "index": 6,
                            "lemma": "a",
                            "originalText": "an",
                            "pos": "DT",
                            "word": "an",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 27,
                            "characterOffsetEnd": 34,
                            "index": 7,
                            "lemma": "unladen",
                            "originalText": "unladen",
                            "pos": "JJ",
                            "word": "unladen",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 35,
                            "characterOffsetEnd": 42,
                            "index": 8,
                            "lemma": "swallow",
                            "originalText": "swallow",
                            "pos": "VB",
                            "word": "swallow",
                        },
                        {
                            "after": "",
                            "before": " ",
                            "characterOffsetBegin": 43,
                            "characterOffsetEnd": 44,
                            "index": 9,
                            "lemma": "?",
                            "originalText": "?",
                            "pos": ".",
                            "word": "?",
                        },
                    ],
                }
            ]
        }
        corenlp_tagger.api_call = MagicMock(return_value=api_return_value)

        input_tokens = "What is the airspeed of an unladen swallow ?".split()
        expected_output = [
            ("What", "WP"),
            ("is", "VBZ"),
            ("the", "DT"),
            ("airspeed", "NN"),
            ("of", "IN"),
            ("an", "DT"),
            ("unladen", "JJ"),
            ("swallow", "VB"),
            ("?", "."),
        ]
        tagged_output = corenlp_tagger.tag(input_tokens)

        corenlp_tagger.api_call.assert_called_once_with(
            "What is the airspeed of an unladen swallow ?",
            properties={
                "ssplit.isOneSentence": "true",
                "annotators": "tokenize,ssplit,pos",
            },
        )
        self.assertEqual(expected_output, tagged_output)

    def test_ner_tagger(self):
        corenlp_tagger = corenlp.CoreNLPParser(tagtype="ner")

        api_return_value = {
            "sentences": [
                {
                    "index": 0,
                    "tokens": [
                        {
                            "after": " ",
                            "before": "",
                            "characterOffsetBegin": 0,
                            "characterOffsetEnd": 4,
                            "index": 1,
                            "lemma": "Rami",
                            "ner": "PERSON",
                            "originalText": "Rami",
                            "pos": "NNP",
                            "word": "Rami",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 5,
                            "characterOffsetEnd": 8,
                            "index": 2,
                            "lemma": "Eid",
                            "ner": "PERSON",
                            "originalText": "Eid",
                            "pos": "NNP",
                            "word": "Eid",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 9,
                            "characterOffsetEnd": 11,
                            "index": 3,
                            "lemma": "be",
                            "ner": "O",
                            "originalText": "is",
                            "pos": "VBZ",
                            "word": "is",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 12,
                            "characterOffsetEnd": 20,
                            "index": 4,
                            "lemma": "study",
                            "ner": "O",
                            "originalText": "studying",
                            "pos": "VBG",
                            "word": "studying",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 21,
                            "characterOffsetEnd": 23,
                            "index": 5,
                            "lemma": "at",
                            "ner": "O",
                            "originalText": "at",
                            "pos": "IN",
                            "word": "at",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 24,
                            "characterOffsetEnd": 29,
                            "index": 6,
                            "lemma": "Stony",
                            "ner": "ORGANIZATION",
                            "originalText": "Stony",
                            "pos": "NNP",
                            "word": "Stony",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 30,
                            "characterOffsetEnd": 35,
                            "index": 7,
                            "lemma": "Brook",
                            "ner": "ORGANIZATION",
                            "originalText": "Brook",
                            "pos": "NNP",
                            "word": "Brook",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 36,
                            "characterOffsetEnd": 46,
                            "index": 8,
                            "lemma": "University",
                            "ner": "ORGANIZATION",
                            "originalText": "University",
                            "pos": "NNP",
                            "word": "University",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 47,
                            "characterOffsetEnd": 49,
                            "index": 9,
                            "lemma": "in",
                            "ner": "O",
                            "originalText": "in",
                            "pos": "IN",
                            "word": "in",
                        },
                        {
                            "after": "",
                            "before": " ",
                            "characterOffsetBegin": 50,
                            "characterOffsetEnd": 52,
                            "index": 10,
                            "lemma": "NY",
                            "ner": "O",
                            "originalText": "NY",
                            "pos": "NNP",
                            "word": "NY",
                        },
                    ],
                }
            ]
        }

        corenlp_tagger.api_call = MagicMock(return_value=api_return_value)

        input_tokens = "Rami Eid is studying at Stony Brook University in NY".split()
        expected_output = [
            ("Rami", "PERSON"),
            ("Eid", "PERSON"),
            ("is", "O"),
            ("studying", "O"),
            ("at", "O"),
            ("Stony", "ORGANIZATION"),
            ("Brook", "ORGANIZATION"),
            ("University", "ORGANIZATION"),
            ("in", "O"),
            ("NY", "O"),
        ]
        tagged_output = corenlp_tagger.tag(input_tokens)

        corenlp_tagger.api_call.assert_called_once_with(
            "Rami Eid is studying at Stony Brook University in NY",
            properties={
                "ssplit.isOneSentence": "true",
                "annotators": "tokenize,ssplit,ner",
            },
        )
        self.assertEqual(expected_output, tagged_output)

    def test_unexpected_tagtype(self):
        with self.assertRaises(ValueError):
            corenlp_tagger = corenlp.CoreNLPParser(tagtype="test")


class TestParserAPI(TestCase):
    def test_parse(self):
        corenlp_parser = corenlp.CoreNLPParser()

        api_return_value = {
            "sentences": [
                {
                    "basicDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 4,
                            "dependentGloss": "fox",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "det",
                            "dependent": 1,
                            "dependentGloss": "The",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 2,
                            "dependentGloss": "quick",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 3,
                            "dependentGloss": "brown",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "dep",
                            "dependent": 5,
                            "dependentGloss": "jumps",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "case",
                            "dependent": 6,
                            "dependentGloss": "over",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "det",
                            "dependent": 7,
                            "dependentGloss": "the",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "amod",
                            "dependent": 8,
                            "dependentGloss": "lazy",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "nmod",
                            "dependent": 9,
                            "dependentGloss": "dog",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                    ],
                    "enhancedDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 4,
                            "dependentGloss": "fox",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "det",
                            "dependent": 1,
                            "dependentGloss": "The",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 2,
                            "dependentGloss": "quick",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 3,
                            "dependentGloss": "brown",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "dep",
                            "dependent": 5,
                            "dependentGloss": "jumps",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "case",
                            "dependent": 6,
                            "dependentGloss": "over",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "det",
                            "dependent": 7,
                            "dependentGloss": "the",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "amod",
                            "dependent": 8,
                            "dependentGloss": "lazy",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "nmod:over",
                            "dependent": 9,
                            "dependentGloss": "dog",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                    ],
                    "enhancedPlusPlusDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 4,
                            "dependentGloss": "fox",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "det",
                            "dependent": 1,
                            "dependentGloss": "The",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 2,
                            "dependentGloss": "quick",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 3,
                            "dependentGloss": "brown",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "dep",
                            "dependent": 5,
                            "dependentGloss": "jumps",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "case",
                            "dependent": 6,
                            "dependentGloss": "over",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "det",
                            "dependent": 7,
                            "dependentGloss": "the",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "amod",
                            "dependent": 8,
                            "dependentGloss": "lazy",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "nmod:over",
                            "dependent": 9,
                            "dependentGloss": "dog",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                    ],
                    "index": 0,
                    "parse": "(ROOT\n  (NP\n    (NP (DT The) (JJ quick) (JJ brown) (NN fox))\n    (NP\n      (NP (NNS jumps))\n      (PP (IN over)\n        (NP (DT the) (JJ lazy) (NN dog))))))",
                    "tokens": [
                        {
                            "after": " ",
                            "before": "",
                            "characterOffsetBegin": 0,
                            "characterOffsetEnd": 3,
                            "index": 1,
                            "lemma": "the",
                            "originalText": "The",
                            "pos": "DT",
                            "word": "The",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 4,
                            "characterOffsetEnd": 9,
                            "index": 2,
                            "lemma": "quick",
                            "originalText": "quick",
                            "pos": "JJ",
                            "word": "quick",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 10,
                            "characterOffsetEnd": 15,
                            "index": 3,
                            "lemma": "brown",
                            "originalText": "brown",
                            "pos": "JJ",
                            "word": "brown",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 16,
                            "characterOffsetEnd": 19,
                            "index": 4,
                            "lemma": "fox",
                            "originalText": "fox",
                            "pos": "NN",
                            "word": "fox",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 20,
                            "characterOffsetEnd": 25,
                            "index": 5,
                            "lemma": "jump",
                            "originalText": "jumps",
                            "pos": "VBZ",
                            "word": "jumps",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 26,
                            "characterOffsetEnd": 30,
                            "index": 6,
                            "lemma": "over",
                            "originalText": "over",
                            "pos": "IN",
                            "word": "over",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 31,
                            "characterOffsetEnd": 34,
                            "index": 7,
                            "lemma": "the",
                            "originalText": "the",
                            "pos": "DT",
                            "word": "the",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 35,
                            "characterOffsetEnd": 39,
                            "index": 8,
                            "lemma": "lazy",
                            "originalText": "lazy",
                            "pos": "JJ",
                            "word": "lazy",
                        },
                        {
                            "after": "",
                            "before": " ",
                            "characterOffsetBegin": 40,
                            "characterOffsetEnd": 43,
                            "index": 9,
                            "lemma": "dog",
                            "originalText": "dog",
                            "pos": "NN",
                            "word": "dog",
                        },
                    ],
                }
            ]
        }

        corenlp_parser.api_call = MagicMock(return_value=api_return_value)

        input_string = "The quick brown fox jumps over the lazy dog".split()
        expected_output = Tree(
            "ROOT",
            [
                Tree(
                    "NP",
                    [
                        Tree(
                            "NP",
                            [
                                Tree("DT", ["The"]),
                                Tree("JJ", ["quick"]),
                                Tree("JJ", ["brown"]),
                                Tree("NN", ["fox"]),
                            ],
                        ),
                        Tree(
                            "NP",
                            [
                                Tree("NP", [Tree("NNS", ["jumps"])]),
                                Tree(
                                    "PP",
                                    [
                                        Tree("IN", ["over"]),
                                        Tree(
                                            "NP",
                                            [
                                                Tree("DT", ["the"]),
                                                Tree("JJ", ["lazy"]),
                                                Tree("NN", ["dog"]),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )

        parsed_data = next(corenlp_parser.parse(input_string))

        corenlp_parser.api_call.assert_called_once_with(
            "The quick brown fox jumps over the lazy dog",
            properties={"ssplit.eolonly": "true"},
        )
        self.assertEqual(expected_output, parsed_data)

    def test_dependency_parser(self):
        corenlp_parser = corenlp.CoreNLPDependencyParser()

        api_return_value = {
            "sentences": [
                {
                    "basicDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 5,
                            "dependentGloss": "jumps",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "det",
                            "dependent": 1,
                            "dependentGloss": "The",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 2,
                            "dependentGloss": "quick",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 3,
                            "dependentGloss": "brown",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "nsubj",
                            "dependent": 4,
                            "dependentGloss": "fox",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                        {
                            "dep": "case",
                            "dependent": 6,
                            "dependentGloss": "over",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "det",
                            "dependent": 7,
                            "dependentGloss": "the",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "amod",
                            "dependent": 8,
                            "dependentGloss": "lazy",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "nmod",
                            "dependent": 9,
                            "dependentGloss": "dog",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                    ],
                    "enhancedDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 5,
                            "dependentGloss": "jumps",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "det",
                            "dependent": 1,
                            "dependentGloss": "The",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 2,
                            "dependentGloss": "quick",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 3,
                            "dependentGloss": "brown",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "nsubj",
                            "dependent": 4,
                            "dependentGloss": "fox",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                        {
                            "dep": "case",
                            "dependent": 6,
                            "dependentGloss": "over",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "det",
                            "dependent": 7,
                            "dependentGloss": "the",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "amod",
                            "dependent": 8,
                            "dependentGloss": "lazy",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "nmod:over",
                            "dependent": 9,
                            "dependentGloss": "dog",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                    ],
                    "enhancedPlusPlusDependencies": [
                        {
                            "dep": "ROOT",
                            "dependent": 5,
                            "dependentGloss": "jumps",
                            "governor": 0,
                            "governorGloss": "ROOT",
                        },
                        {
                            "dep": "det",
                            "dependent": 1,
                            "dependentGloss": "The",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 2,
                            "dependentGloss": "quick",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "amod",
                            "dependent": 3,
                            "dependentGloss": "brown",
                            "governor": 4,
                            "governorGloss": "fox",
                        },
                        {
                            "dep": "nsubj",
                            "dependent": 4,
                            "dependentGloss": "fox",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                        {
                            "dep": "case",
                            "dependent": 6,
                            "dependentGloss": "over",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "det",
                            "dependent": 7,
                            "dependentGloss": "the",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "amod",
                            "dependent": 8,
                            "dependentGloss": "lazy",
                            "governor": 9,
                            "governorGloss": "dog",
                        },
                        {
                            "dep": "nmod:over",
                            "dependent": 9,
                            "dependentGloss": "dog",
                            "governor": 5,
                            "governorGloss": "jumps",
                        },
                    ],
                    "index": 0,
                    "tokens": [
                        {
                            "after": " ",
                            "before": "",
                            "characterOffsetBegin": 0,
                            "characterOffsetEnd": 3,
                            "index": 1,
                            "lemma": "the",
                            "originalText": "The",
                            "pos": "DT",
                            "word": "The",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 4,
                            "characterOffsetEnd": 9,
                            "index": 2,
                            "lemma": "quick",
                            "originalText": "quick",
                            "pos": "JJ",
                            "word": "quick",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 10,
                            "characterOffsetEnd": 15,
                            "index": 3,
                            "lemma": "brown",
                            "originalText": "brown",
                            "pos": "JJ",
                            "word": "brown",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 16,
                            "characterOffsetEnd": 19,
                            "index": 4,
                            "lemma": "fox",
                            "originalText": "fox",
                            "pos": "NN",
                            "word": "fox",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 20,
                            "characterOffsetEnd": 25,
                            "index": 5,
                            "lemma": "jump",
                            "originalText": "jumps",
                            "pos": "VBZ",
                            "word": "jumps",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 26,
                            "characterOffsetEnd": 30,
                            "index": 6,
                            "lemma": "over",
                            "originalText": "over",
                            "pos": "IN",
                            "word": "over",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 31,
                            "characterOffsetEnd": 34,
                            "index": 7,
                            "lemma": "the",
                            "originalText": "the",
                            "pos": "DT",
                            "word": "the",
                        },
                        {
                            "after": " ",
                            "before": " ",
                            "characterOffsetBegin": 35,
                            "characterOffsetEnd": 39,
                            "index": 8,
                            "lemma": "lazy",
                            "originalText": "lazy",
                            "pos": "JJ",
                            "word": "lazy",
                        },
                        {
                            "after": "",
                            "before": " ",
                            "characterOffsetBegin": 40,
                            "characterOffsetEnd": 43,
                            "index": 9,
                            "lemma": "dog",
                            "originalText": "dog",
                            "pos": "NN",
                            "word": "dog",
                        },
                    ],
                }
            ]
        }

        corenlp_parser.api_call = MagicMock(return_value=api_return_value)

        input_string = "The quick brown fox jumps over the lazy dog".split()
        expected_output = Tree(
            "jumps",
            [
                Tree("fox", ["The", "quick", "brown"]),
                Tree("dog", ["over", "the", "lazy"]),
            ],
        )

        parsed_data = next(corenlp_parser.parse(input_string))

        corenlp_parser.api_call.assert_called_once_with(
            "The quick brown fox jumps over the lazy dog",
            properties={"ssplit.eolonly": "true"},
        )
        self.assertEqual(expected_output, parsed_data.tree())
