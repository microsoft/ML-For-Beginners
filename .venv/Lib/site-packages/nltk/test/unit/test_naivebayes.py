import unittest

from nltk.classify.naivebayes import NaiveBayesClassifier


class NaiveBayesClassifierTest(unittest.TestCase):
    def test_simple(self):
        training_features = [
            ({"nice": True, "good": True}, "positive"),
            ({"bad": True, "mean": True}, "negative"),
        ]

        classifier = NaiveBayesClassifier.train(training_features)

        result = classifier.prob_classify({"nice": True})
        self.assertTrue(result.prob("positive") > result.prob("negative"))
        self.assertEqual(result.max(), "positive")

        result = classifier.prob_classify({"bad": True})
        self.assertTrue(result.prob("positive") < result.prob("negative"))
        self.assertEqual(result.max(), "negative")
