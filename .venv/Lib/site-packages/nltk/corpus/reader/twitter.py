# Natural Language Toolkit: Twitter Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A reader for corpora that consist of Tweets. It is assumed that the Tweets
have been serialised into line-delimited JSON.
"""

import json
import os

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, ZipFilePathPointer, concat
from nltk.tokenize import TweetTokenizer


class TwitterCorpusReader(CorpusReader):
    r"""
    Reader for corpora that consist of Tweets represented as a list of line-delimited JSON.

    Individual Tweets can be tokenized using the default tokenizer, or by a
    custom tokenizer specified as a parameter to the constructor.

    Construct a new Tweet corpus reader for a set of documents
    located at the given root directory.

    If you made your own tweet collection in a directory called
    `twitter-files`, then you can initialise the reader as::

        from nltk.corpus import TwitterCorpusReader
        reader = TwitterCorpusReader(root='/path/to/twitter-files', '.*\.json')

    However, the recommended approach is to set the relevant directory as the
    value of the environmental variable `TWITTER`, and then invoke the reader
    as follows::

       root = os.environ['TWITTER']
       reader = TwitterCorpusReader(root, '.*\.json')

    If you want to work directly with the raw Tweets, the `json` library can
    be used::

       import json
       for tweet in reader.docs():
           print(json.dumps(tweet, indent=1, sort_keys=True))

    """

    CorpusView = StreamBackedCorpusView
    """
    The corpus view class used by this reader.
    """

    def __init__(
        self, root, fileids=None, word_tokenizer=TweetTokenizer(), encoding="utf8"
    ):
        """
        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        :param word_tokenizer: Tokenizer for breaking the text of Tweets into
            smaller units, including but not limited to words.
        """
        CorpusReader.__init__(self, root, fileids, encoding)

        for path in self.abspaths(self._fileids):
            if isinstance(path, ZipFilePathPointer):
                pass
            elif os.path.getsize(path) == 0:
                raise ValueError(f"File {path} is empty")
        """Check that all user-created corpus files are non-empty."""

        self._word_tokenizer = word_tokenizer

    def docs(self, fileids=None):
        """
        Returns the full Tweet objects, as specified by `Twitter
        documentation on Tweets
        <https://dev.twitter.com/docs/platform-objects/tweets>`_

        :return: the given file(s) as a list of dictionaries deserialised
            from JSON.
        :rtype: list(dict)
        """
        return concat(
            [
                self.CorpusView(path, self._read_tweets, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def strings(self, fileids=None):
        """
        Returns only the text content of Tweets in the file(s)

        :return: the given file(s) as a list of Tweets.
        :rtype: list(str)
        """
        fulltweets = self.docs(fileids)
        tweets = []
        for jsono in fulltweets:
            try:
                text = jsono["text"]
                if isinstance(text, bytes):
                    text = text.decode(self.encoding)
                tweets.append(text)
            except KeyError:
                pass
        return tweets

    def tokenized(self, fileids=None):
        """
        :return: the given file(s) as a list of the text content of Tweets as
            as a list of words, screenanames, hashtags, URLs and punctuation symbols.

        :rtype: list(list(str))
        """
        tweets = self.strings(fileids)
        tokenizer = self._word_tokenizer
        return [tokenizer.tokenize(t) for t in tweets]

    def _read_tweets(self, stream):
        """
        Assumes that each line in ``stream`` is a JSON-serialised object.
        """
        tweets = []
        for i in range(10):
            line = stream.readline()
            if not line:
                return tweets
            tweet = json.loads(line)
            tweets.append(tweet)
        return tweets
