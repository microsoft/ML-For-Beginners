"""
Corpus View Regression Tests
"""
import unittest

import nltk.data
from nltk.corpus.reader.util import (
    StreamBackedCorpusView,
    read_line_block,
    read_whitespace_block,
)


class TestCorpusViews(unittest.TestCase):

    linetok = nltk.LineTokenizer(blanklines="keep")
    names = [
        "corpora/inaugural/README",  # A very short file (160 chars)
        "corpora/inaugural/1793-Washington.txt",  # A relatively short file (791 chars)
        "corpora/inaugural/1909-Taft.txt",  # A longer file (32k chars)
    ]

    def data(self):
        for name in self.names:
            f = nltk.data.find(name)
            with f.open() as fp:
                file_data = fp.read().decode("utf8")
            yield f, file_data

    def test_correct_values(self):
        # Check that corpus views produce the correct sequence of values.

        for f, file_data in self.data():
            v = StreamBackedCorpusView(f, read_whitespace_block)
            self.assertEqual(list(v), file_data.split())

            v = StreamBackedCorpusView(f, read_line_block)
            self.assertEqual(list(v), self.linetok.tokenize(file_data))

    def test_correct_length(self):
        # Check that the corpus views report the correct lengths:

        for f, file_data in self.data():
            v = StreamBackedCorpusView(f, read_whitespace_block)
            self.assertEqual(len(v), len(file_data.split()))

            v = StreamBackedCorpusView(f, read_line_block)
            self.assertEqual(len(v), len(self.linetok.tokenize(file_data)))
