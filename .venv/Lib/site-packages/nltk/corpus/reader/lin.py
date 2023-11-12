# Natural Language Toolkit: Lin's Thesaurus
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Dan Blanchard <dblanchard@ets.org>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.txt

import re
from collections import defaultdict
from functools import reduce

from nltk.corpus.reader import CorpusReader


class LinThesaurusCorpusReader(CorpusReader):
    """Wrapper for the LISP-formatted thesauruses distributed by Dekang Lin."""

    # Compiled regular expression for extracting the key from the first line of each
    # thesaurus entry
    _key_re = re.compile(r'\("?([^"]+)"? \(desc [0-9.]+\).+')

    @staticmethod
    def __defaultdict_factory():
        """Factory for creating defaultdict of defaultdict(dict)s"""
        return defaultdict(dict)

    def __init__(self, root, badscore=0.0):
        """
        Initialize the thesaurus.

        :param root: root directory containing thesaurus LISP files
        :type root: C{string}
        :param badscore: the score to give to words which do not appear in each other's sets of synonyms
        :type badscore: C{float}
        """

        super().__init__(root, r"sim[A-Z]\.lsp")
        self._thesaurus = defaultdict(LinThesaurusCorpusReader.__defaultdict_factory)
        self._badscore = badscore
        for path, encoding, fileid in self.abspaths(
            include_encoding=True, include_fileid=True
        ):
            with open(path) as lin_file:
                first = True
                for line in lin_file:
                    line = line.strip()
                    # Start of entry
                    if first:
                        key = LinThesaurusCorpusReader._key_re.sub(r"\1", line)
                        first = False
                    # End of entry
                    elif line == "))":
                        first = True
                    # Lines with pairs of ngrams and scores
                    else:
                        split_line = line.split("\t")
                        if len(split_line) == 2:
                            ngram, score = split_line
                            self._thesaurus[fileid][key][ngram.strip('"')] = float(
                                score
                            )

    def similarity(self, ngram1, ngram2, fileid=None):
        """
        Returns the similarity score for two ngrams.

        :param ngram1: first ngram to compare
        :type ngram1: C{string}
        :param ngram2: second ngram to compare
        :type ngram2: C{string}
        :param fileid: thesaurus fileid to search in. If None, search all fileids.
        :type fileid: C{string}
        :return: If fileid is specified, just the score for the two ngrams; otherwise,
                 list of tuples of fileids and scores.
        """
        # Entries don't contain themselves, so make sure similarity between item and itself is 1.0
        if ngram1 == ngram2:
            if fileid:
                return 1.0
            else:
                return [(fid, 1.0) for fid in self._fileids]
        else:
            if fileid:
                return (
                    self._thesaurus[fileid][ngram1][ngram2]
                    if ngram2 in self._thesaurus[fileid][ngram1]
                    else self._badscore
                )
            else:
                return [
                    (
                        fid,
                        (
                            self._thesaurus[fid][ngram1][ngram2]
                            if ngram2 in self._thesaurus[fid][ngram1]
                            else self._badscore
                        ),
                    )
                    for fid in self._fileids
                ]

    def scored_synonyms(self, ngram, fileid=None):
        """
        Returns a list of scored synonyms (tuples of synonyms and scores) for the current ngram

        :param ngram: ngram to lookup
        :type ngram: C{string}
        :param fileid: thesaurus fileid to search in. If None, search all fileids.
        :type fileid: C{string}
        :return: If fileid is specified, list of tuples of scores and synonyms; otherwise,
                 list of tuples of fileids and lists, where inner lists consist of tuples of
                 scores and synonyms.
        """
        if fileid:
            return self._thesaurus[fileid][ngram].items()
        else:
            return [
                (fileid, self._thesaurus[fileid][ngram].items())
                for fileid in self._fileids
            ]

    def synonyms(self, ngram, fileid=None):
        """
        Returns a list of synonyms for the current ngram.

        :param ngram: ngram to lookup
        :type ngram: C{string}
        :param fileid: thesaurus fileid to search in. If None, search all fileids.
        :type fileid: C{string}
        :return: If fileid is specified, list of synonyms; otherwise, list of tuples of fileids and
                 lists, where inner lists contain synonyms.
        """
        if fileid:
            return self._thesaurus[fileid][ngram].keys()
        else:
            return [
                (fileid, self._thesaurus[fileid][ngram].keys())
                for fileid in self._fileids
            ]

    def __contains__(self, ngram):
        """
        Determines whether or not the given ngram is in the thesaurus.

        :param ngram: ngram to lookup
        :type ngram: C{string}
        :return: whether the given ngram is in the thesaurus.
        """
        return reduce(
            lambda accum, fileid: accum or (ngram in self._thesaurus[fileid]),
            self._fileids,
            False,
        )


######################################################################
# Demo
######################################################################


def demo():
    from nltk.corpus import lin_thesaurus as thes

    word1 = "business"
    word2 = "enterprise"
    print("Getting synonyms for " + word1)
    print(thes.synonyms(word1))

    print("Getting scored synonyms for " + word1)
    print(thes.scored_synonyms(word1))

    print("Getting synonyms from simN.lsp (noun subsection) for " + word1)
    print(thes.synonyms(word1, fileid="simN.lsp"))

    print("Getting synonyms from simN.lsp (noun subsection) for " + word1)
    print(thes.synonyms(word1, fileid="simN.lsp"))

    print(f"Similarity score for {word1} and {word2}:")
    print(thes.similarity(word1, word2))


if __name__ == "__main__":
    demo()
