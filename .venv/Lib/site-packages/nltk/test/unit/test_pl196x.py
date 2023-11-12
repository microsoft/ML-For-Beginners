import unittest

import nltk
from nltk.corpus.reader import pl196x


class TestCorpusViews(unittest.TestCase):
    def test_corpus_reader(self):
        pl196x_dir = nltk.data.find("corpora/pl196x")
        pl = pl196x.Pl196xCorpusReader(
            pl196x_dir, r".*\.xml", textids="textids.txt", cat_file="cats.txt"
        )
        pl.tagged_words(fileids=pl.fileids(), categories="cats.txt")
