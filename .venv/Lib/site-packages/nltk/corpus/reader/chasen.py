#
# Copyright (C) 2001-2023 NLTK Project
# Author: Masato Hagiwara <hagisan@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import sys

from nltk.corpus.reader import util
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *


class ChasenCorpusReader(CorpusReader):
    def __init__(self, root, fileids, encoding="utf8", sent_splitter=None):
        self._sent_splitter = sent_splitter
        CorpusReader.__init__(self, root, fileids, encoding)

    def words(self, fileids=None):
        return concat(
            [
                ChasenCorpusView(fileid, enc, False, False, False, self._sent_splitter)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def tagged_words(self, fileids=None):
        return concat(
            [
                ChasenCorpusView(fileid, enc, True, False, False, self._sent_splitter)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def sents(self, fileids=None):
        return concat(
            [
                ChasenCorpusView(fileid, enc, False, True, False, self._sent_splitter)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def tagged_sents(self, fileids=None):
        return concat(
            [
                ChasenCorpusView(fileid, enc, True, True, False, self._sent_splitter)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def paras(self, fileids=None):
        return concat(
            [
                ChasenCorpusView(fileid, enc, False, True, True, self._sent_splitter)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def tagged_paras(self, fileids=None):
        return concat(
            [
                ChasenCorpusView(fileid, enc, True, True, True, self._sent_splitter)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )


class ChasenCorpusView(StreamBackedCorpusView):
    """
    A specialized corpus view for ChasenReader. Similar to ``TaggedCorpusView``,
    but this'll use fixed sets of word and sentence tokenizer.
    """

    def __init__(
        self,
        corpus_file,
        encoding,
        tagged,
        group_by_sent,
        group_by_para,
        sent_splitter=None,
    ):
        self._tagged = tagged
        self._group_by_sent = group_by_sent
        self._group_by_para = group_by_para
        self._sent_splitter = sent_splitter
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        """Reads one paragraph at a time."""
        block = []
        for para_str in read_regexp_block(stream, r".", r"^EOS\n"):

            para = []

            sent = []
            for line in para_str.splitlines():

                _eos = line.strip() == "EOS"
                _cells = line.split("\t")
                w = (_cells[0], "\t".join(_cells[1:]))
                if not _eos:
                    sent.append(w)

                if _eos or (self._sent_splitter and self._sent_splitter(w)):
                    if not self._tagged:
                        sent = [w for (w, t) in sent]
                    if self._group_by_sent:
                        para.append(sent)
                    else:
                        para.extend(sent)
                    sent = []

            if len(sent) > 0:
                if not self._tagged:
                    sent = [w for (w, t) in sent]

                if self._group_by_sent:
                    para.append(sent)
                else:
                    para.extend(sent)

            if self._group_by_para:
                block.append(para)
            else:
                block.extend(para)

        return block


def demo():

    import nltk
    from nltk.corpus.util import LazyCorpusLoader

    jeita = LazyCorpusLoader("jeita", ChasenCorpusReader, r".*chasen", encoding="utf-8")
    print("/".join(jeita.words()[22100:22140]))

    print(
        "\nEOS\n".join(
            "\n".join("{}/{}".format(w[0], w[1].split("\t")[2]) for w in sent)
            for sent in jeita.tagged_sents()[2170:2173]
        )
    )


def test():

    from nltk.corpus.util import LazyCorpusLoader

    jeita = LazyCorpusLoader("jeita", ChasenCorpusReader, r".*chasen", encoding="utf-8")

    assert isinstance(jeita.tagged_words()[0][1], str)


if __name__ == "__main__":
    demo()
    test()
