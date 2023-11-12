# Natural Language Toolkit: Product Reviews Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Pierpaolo Pantone <24alsecondo@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
CorpusReader for reviews corpora (syntax based on Customer Review Corpus).

Customer Review Corpus information
==================================

Annotated by: Minqing Hu and Bing Liu, 2004.
    Department of Computer Science
    University of Illinois at Chicago

Contact: Bing Liu, liub@cs.uic.edu
        https://www.cs.uic.edu/~liub

Distributed with permission.

The "product_reviews_1" and "product_reviews_2" datasets respectively contain
annotated customer reviews of 5 and 9 products from amazon.com.

Related papers:

- Minqing Hu and Bing Liu. "Mining and summarizing customer reviews".
    Proceedings of the ACM SIGKDD International Conference on Knowledge
    Discovery & Data Mining (KDD-04), 2004.

- Minqing Hu and Bing Liu. "Mining Opinion Features in Customer Reviews".
    Proceedings of Nineteeth National Conference on Artificial Intelligence
    (AAAI-2004), 2004.

- Xiaowen Ding, Bing Liu and Philip S. Yu. "A Holistic Lexicon-Based Appraoch to
    Opinion Mining." Proceedings of First ACM International Conference on Web
    Search and Data Mining (WSDM-2008), Feb 11-12, 2008, Stanford University,
    Stanford, California, USA.

Symbols used in the annotated reviews:

    :[t]: the title of the review: Each [t] tag starts a review.
    :xxxx[+|-n]: xxxx is a product feature.
    :[+n]: Positive opinion, n is the opinion strength: 3 strongest, and 1 weakest.
           Note that the strength is quite subjective.
           You may want ignore it, but only considering + and -
    :[-n]: Negative opinion
    :##:   start of each sentence. Each line is a sentence.
    :[u]:  feature not appeared in the sentence.
    :[p]:  feature not appeared in the sentence. Pronoun resolution is needed.
    :[s]:  suggestion or recommendation.
    :[cc]: comparison with a competing product from a different brand.
    :[cs]: comparison with a competing product from the same brand.

Note: Some of the files (e.g. "ipod.txt", "Canon PowerShot SD500.txt") do not
    provide separation between different reviews. This is due to the fact that
    the dataset was specifically designed for aspect/feature-based sentiment
    analysis, for which sentence-level annotation is sufficient. For document-
    level classification and analysis, this peculiarity should be taken into
    consideration.
"""

import re

from nltk.corpus.reader.api import *
from nltk.tokenize import *

TITLE = re.compile(r"^\[t\](.*)$")  # [t] Title
FEATURES = re.compile(
    r"((?:(?:\w+\s)+)?\w+)\[((?:\+|\-)\d)\]"
)  # find 'feature' in feature[+3]
NOTES = re.compile(r"\[(?!t)(p|u|s|cc|cs)\]")  # find 'p' in camera[+2][p]
SENT = re.compile(r"##(.*)$")  # find tokenized sentence


class Review:
    """
    A Review is the main block of a ReviewsCorpusReader.
    """

    def __init__(self, title=None, review_lines=None):
        """
        :param title: the title of the review.
        :param review_lines: the list of the ReviewLines that belong to the Review.
        """
        self.title = title
        if review_lines is None:
            self.review_lines = []
        else:
            self.review_lines = review_lines

    def add_line(self, review_line):
        """
        Add a line (ReviewLine) to the review.

        :param review_line: a ReviewLine instance that belongs to the Review.
        """
        assert isinstance(review_line, ReviewLine)
        self.review_lines.append(review_line)

    def features(self):
        """
        Return a list of features in the review. Each feature is a tuple made of
        the specific item feature and the opinion strength about that feature.

        :return: all features of the review as a list of tuples (feat, score).
        :rtype: list(tuple)
        """
        features = []
        for review_line in self.review_lines:
            features.extend(review_line.features)
        return features

    def sents(self):
        """
        Return all tokenized sentences in the review.

        :return: all sentences of the review as lists of tokens.
        :rtype: list(list(str))
        """
        return [review_line.sent for review_line in self.review_lines]

    def __repr__(self):
        return 'Review(title="{}", review_lines={})'.format(
            self.title, self.review_lines
        )


class ReviewLine:
    """
    A ReviewLine represents a sentence of the review, together with (optional)
    annotations of its features and notes about the reviewed item.
    """

    def __init__(self, sent, features=None, notes=None):
        self.sent = sent
        if features is None:
            self.features = []
        else:
            self.features = features

        if notes is None:
            self.notes = []
        else:
            self.notes = notes

    def __repr__(self):
        return "ReviewLine(features={}, notes={}, sent={})".format(
            self.features, self.notes, self.sent
        )


class ReviewsCorpusReader(CorpusReader):
    """
    Reader for the Customer Review Data dataset by Hu, Liu (2004).
    Note: we are not applying any sentence tokenization at the moment, just word
    tokenization.

        >>> from nltk.corpus import product_reviews_1
        >>> camera_reviews = product_reviews_1.reviews('Canon_G3.txt')
        >>> review = camera_reviews[0]
        >>> review.sents()[0] # doctest: +NORMALIZE_WHITESPACE
        ['i', 'recently', 'purchased', 'the', 'canon', 'powershot', 'g3', 'and', 'am',
        'extremely', 'satisfied', 'with', 'the', 'purchase', '.']
        >>> review.features() # doctest: +NORMALIZE_WHITESPACE
        [('canon powershot g3', '+3'), ('use', '+2'), ('picture', '+2'),
        ('picture quality', '+1'), ('picture quality', '+1'), ('camera', '+2'),
        ('use', '+2'), ('feature', '+1'), ('picture quality', '+3'), ('use', '+1'),
        ('option', '+1')]

    We can also reach the same information directly from the stream:

        >>> product_reviews_1.features('Canon_G3.txt')
        [('canon powershot g3', '+3'), ('use', '+2'), ...]

    We can compute stats for specific product features:

        >>> n_reviews = len([(feat,score) for (feat,score) in product_reviews_1.features('Canon_G3.txt') if feat=='picture'])
        >>> tot = sum([int(score) for (feat,score) in product_reviews_1.features('Canon_G3.txt') if feat=='picture'])
        >>> mean = tot / n_reviews
        >>> print(n_reviews, tot, mean)
        15 24 1.6
    """

    CorpusView = StreamBackedCorpusView

    def __init__(
        self, root, fileids, word_tokenizer=WordPunctTokenizer(), encoding="utf8"
    ):
        """
        :param root: The root directory for the corpus.
        :param fileids: a list or regexp specifying the fileids in the corpus.
        :param word_tokenizer: a tokenizer for breaking sentences or paragraphs
            into words. Default: `WordPunctTokenizer`
        :param encoding: the encoding that should be used to read the corpus.
        """

        CorpusReader.__init__(self, root, fileids, encoding)
        self._word_tokenizer = word_tokenizer
        self._readme = "README.txt"

    def features(self, fileids=None):
        """
        Return a list of features. Each feature is a tuple made of the specific
        item feature and the opinion strength about that feature.

        :param fileids: a list or regexp specifying the ids of the files whose
            features have to be returned.
        :return: all features for the item(s) in the given file(s).
        :rtype: list(tuple)
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat(
            [
                self.CorpusView(fileid, self._read_features, encoding=enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def reviews(self, fileids=None):
        """
        Return all the reviews as a list of Review objects. If `fileids` is
        specified, return all the reviews from each of the specified files.

        :param fileids: a list or regexp specifying the ids of the files whose
            reviews have to be returned.
        :return: the given file(s) as a list of reviews.
        """
        if fileids is None:
            fileids = self._fileids
        return concat(
            [
                self.CorpusView(fileid, self._read_review_block, encoding=enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def sents(self, fileids=None):
        """
        Return all sentences in the corpus or in the specified files.

        :param fileids: a list or regexp specifying the ids of the files whose
            sentences have to be returned.
        :return: the given file(s) as a list of sentences, each encoded as a
            list of word strings.
        :rtype: list(list(str))
        """
        return concat(
            [
                self.CorpusView(path, self._read_sent_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def words(self, fileids=None):
        """
        Return all words and punctuation symbols in the corpus or in the specified
        files.

        :param fileids: a list or regexp specifying the ids of the files whose
            words have to be returned.
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        return concat(
            [
                self.CorpusView(path, self._read_word_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def _read_features(self, stream):
        features = []
        for i in range(20):
            line = stream.readline()
            if not line:
                return features
            features.extend(re.findall(FEATURES, line))
        return features

    def _read_review_block(self, stream):
        while True:
            line = stream.readline()
            if not line:
                return []  # end of file.
            title_match = re.match(TITLE, line)
            if title_match:
                review = Review(
                    title=title_match.group(1).strip()
                )  # We create a new review
                break

        # Scan until we find another line matching the regexp, or EOF.
        while True:
            oldpos = stream.tell()
            line = stream.readline()
            # End of file:
            if not line:
                return [review]
            # Start of a new review: backup to just before it starts, and
            # return the review we've already collected.
            if re.match(TITLE, line):
                stream.seek(oldpos)
                return [review]
            # Anything else is part of the review line.
            feats = re.findall(FEATURES, line)
            notes = re.findall(NOTES, line)
            sent = re.findall(SENT, line)
            if sent:
                sent = self._word_tokenizer.tokenize(sent[0])
            review_line = ReviewLine(sent=sent, features=feats, notes=notes)
            review.add_line(review_line)

    def _read_sent_block(self, stream):
        sents = []
        for review in self._read_review_block(stream):
            sents.extend([sent for sent in review.sents()])
        return sents

    def _read_word_block(self, stream):
        words = []
        for i in range(20):  # Read 20 lines at a time.
            line = stream.readline()
            sent = re.findall(SENT, line)
            if sent:
                words.extend(self._word_tokenizer.tokenize(sent[0]))
        return words
