import unittest

import pytest

from nltk import ConditionalFreqDist, tokenize


class TestEmptyCondFreq(unittest.TestCase):
    def test_tabulate(self):
        empty = ConditionalFreqDist()
        self.assertEqual(empty.conditions(), [])
        with pytest.raises(ValueError):
            empty.tabulate(conditions="BUG")  # nonexistent keys shouldn't be added
        self.assertEqual(empty.conditions(), [])

    def test_plot(self):
        empty = ConditionalFreqDist()
        self.assertEqual(empty.conditions(), [])
        empty.plot(conditions=["BUG"])  # nonexistent keys shouldn't be added
        self.assertEqual(empty.conditions(), [])

    def test_increment(self):
        # make sure that we can still mutate cfd normally
        text = "cow cat mouse cat tiger"
        cfd = ConditionalFreqDist()

        # create cfd with word length as condition
        for word in tokenize.word_tokenize(text):
            condition = len(word)
            cfd[condition][word] += 1

        self.assertEqual(cfd.conditions(), [3, 5])

        # incrementing previously unseen key is still possible
        cfd[2]["hi"] += 1
        self.assertCountEqual(cfd.conditions(), [3, 5, 2])  # new condition added
        self.assertEqual(
            cfd[2]["hi"], 1
        )  # key's frequency incremented from 0 (unseen) to 1
