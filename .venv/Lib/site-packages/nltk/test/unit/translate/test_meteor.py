import unittest

from nltk.translate.meteor_score import meteor_score


class TestMETEOR(unittest.TestCase):
    reference = [["this", "is", "a", "test"], ["this", "is" "test"]]
    candidate = ["THIS", "Is", "a", "tEST"]

    def test_meteor(self):
        score = meteor_score(self.reference, self.candidate, preprocess=str.lower)
        assert score == 0.9921875

    def test_reference_type_check(self):
        str_reference = [" ".join(ref) for ref in self.reference]
        self.assertRaises(TypeError, meteor_score, str_reference, self.candidate)

    def test_candidate_type_check(self):
        str_candidate = " ".join(self.candidate)
        self.assertRaises(TypeError, meteor_score, self.reference, str_candidate)
