"""
Unit tests for Senna
"""

import unittest
from os import environ, path, sep

from nltk.classify import Senna
from nltk.tag import SennaChunkTagger, SennaNERTagger, SennaTagger

# Set Senna executable path for tests if it is not specified as an environment variable
if "SENNA" in environ:
    SENNA_EXECUTABLE_PATH = path.normpath(environ["SENNA"]) + sep
else:
    SENNA_EXECUTABLE_PATH = "/usr/share/senna-v3.0"

senna_is_installed = path.exists(SENNA_EXECUTABLE_PATH)


@unittest.skipUnless(senna_is_installed, "Requires Senna executable")
class TestSennaPipeline(unittest.TestCase):
    """Unittest for nltk.classify.senna"""

    def test_senna_pipeline(self):
        """Senna pipeline interface"""

        pipeline = Senna(SENNA_EXECUTABLE_PATH, ["pos", "chk", "ner"])
        sent = "Dusseldorf is an international business center".split()
        result = [
            (token["word"], token["chk"], token["ner"], token["pos"])
            for token in pipeline.tag(sent)
        ]
        expected = [
            ("Dusseldorf", "B-NP", "B-LOC", "NNP"),
            ("is", "B-VP", "O", "VBZ"),
            ("an", "B-NP", "O", "DT"),
            ("international", "I-NP", "O", "JJ"),
            ("business", "I-NP", "O", "NN"),
            ("center", "I-NP", "O", "NN"),
        ]
        self.assertEqual(result, expected)


@unittest.skipUnless(senna_is_installed, "Requires Senna executable")
class TestSennaTagger(unittest.TestCase):
    """Unittest for nltk.tag.senna"""

    def test_senna_tagger(self):
        tagger = SennaTagger(SENNA_EXECUTABLE_PATH)
        result = tagger.tag("What is the airspeed of an unladen swallow ?".split())
        expected = [
            ("What", "WP"),
            ("is", "VBZ"),
            ("the", "DT"),
            ("airspeed", "NN"),
            ("of", "IN"),
            ("an", "DT"),
            ("unladen", "NN"),
            ("swallow", "NN"),
            ("?", "."),
        ]
        self.assertEqual(result, expected)

    def test_senna_chunk_tagger(self):
        chktagger = SennaChunkTagger(SENNA_EXECUTABLE_PATH)
        result_1 = chktagger.tag("What is the airspeed of an unladen swallow ?".split())
        expected_1 = [
            ("What", "B-NP"),
            ("is", "B-VP"),
            ("the", "B-NP"),
            ("airspeed", "I-NP"),
            ("of", "B-PP"),
            ("an", "B-NP"),
            ("unladen", "I-NP"),
            ("swallow", "I-NP"),
            ("?", "O"),
        ]

        result_2 = list(chktagger.bio_to_chunks(result_1, chunk_type="NP"))
        expected_2 = [
            ("What", "0"),
            ("the airspeed", "2-3"),
            ("an unladen swallow", "5-6-7"),
        ]
        self.assertEqual(result_1, expected_1)
        self.assertEqual(result_2, expected_2)

    def test_senna_ner_tagger(self):
        nertagger = SennaNERTagger(SENNA_EXECUTABLE_PATH)
        result_1 = nertagger.tag("Shakespeare theatre was in London .".split())
        expected_1 = [
            ("Shakespeare", "B-PER"),
            ("theatre", "O"),
            ("was", "O"),
            ("in", "O"),
            ("London", "B-LOC"),
            (".", "O"),
        ]

        result_2 = nertagger.tag("UN headquarters are in NY , USA .".split())
        expected_2 = [
            ("UN", "B-ORG"),
            ("headquarters", "O"),
            ("are", "O"),
            ("in", "O"),
            ("NY", "B-LOC"),
            (",", "O"),
            ("USA", "B-LOC"),
            (".", "O"),
        ]
        self.assertEqual(result_1, expected_1)
        self.assertEqual(result_2, expected_2)
