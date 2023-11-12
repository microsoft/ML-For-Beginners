#! /usr/bin/env python
# KNB Corpus reader
# Copyright (C) 2001-2023 NLTK Project
# Author: Masato Hagiwara <hagisan@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

# For more information, see http://lilyx.net/pages/nltkjapanesecorpus.html

import re

from nltk.corpus.reader.api import CorpusReader, SyntaxCorpusReader
from nltk.corpus.reader.util import (
    FileSystemPathPointer,
    find_corpus_fileids,
    read_blankline_block,
)
from nltk.parse import DependencyGraph

# default function to convert morphlist to str for tree representation
_morphs2str_default = lambda morphs: "/".join(m[0] for m in morphs if m[0] != "EOS")


class KNBCorpusReader(SyntaxCorpusReader):
    """
    This class implements:
      - ``__init__``, which specifies the location of the corpus
        and a method for detecting the sentence blocks in corpus files.
      - ``_read_block``, which reads a block from the input stream.
      - ``_word``, which takes a block and returns a list of list of words.
      - ``_tag``, which takes a block and returns a list of list of tagged
        words.
      - ``_parse``, which takes a block and returns a list of parsed
        sentences.

    The structure of tagged words:
      tagged_word = (word(str), tags(tuple))
      tags = (surface, reading, lemma, pos1, posid1, pos2, posid2, pos3, posid3, others ...)

    Usage example

    >>> from nltk.corpus.util import LazyCorpusLoader
    >>> knbc = LazyCorpusLoader(
    ...     'knbc/corpus1',
    ...     KNBCorpusReader,
    ...     r'.*/KN.*',
    ...     encoding='euc-jp',
    ... )

    >>> len(knbc.sents()[0])
    9

    """

    def __init__(self, root, fileids, encoding="utf8", morphs2str=_morphs2str_default):
        """
        Initialize KNBCorpusReader
        morphs2str is a function to convert morphlist to str for tree representation
        for _parse()
        """
        SyntaxCorpusReader.__init__(self, root, fileids, encoding)
        self.morphs2str = morphs2str

    def _read_block(self, stream):
        # blocks are split by blankline (or EOF) - default
        return read_blankline_block(stream)

    def _word(self, t):
        res = []
        for line in t.splitlines():
            # ignore the Bunsets headers
            if not re.match(r"EOS|\*|\#|\+", line):
                cells = line.strip().split(" ")
                res.append(cells[0])

        return res

    # ignores tagset argument
    def _tag(self, t, tagset=None):
        res = []
        for line in t.splitlines():
            # ignore the Bunsets headers
            if not re.match(r"EOS|\*|\#|\+", line):
                cells = line.strip().split(" ")
                # convert cells to morph tuples
                res.append((cells[0], " ".join(cells[1:])))

        return res

    def _parse(self, t):
        dg = DependencyGraph()
        i = 0
        for line in t.splitlines():
            if line[0] in "*+":
                # start of bunsetsu or tag

                cells = line.strip().split(" ", 3)
                m = re.match(r"([\-0-9]*)([ADIP])", cells[1])

                assert m is not None

                node = dg.nodes[i]
                node.update({"address": i, "rel": m.group(2), "word": []})

                dep_parent = int(m.group(1))

                if dep_parent == -1:
                    dg.root = node
                else:
                    dg.nodes[dep_parent]["deps"].append(i)

                i += 1
            elif line[0] != "#":
                # normal morph
                cells = line.strip().split(" ")
                # convert cells to morph tuples
                morph = cells[0], " ".join(cells[1:])
                dg.nodes[i - 1]["word"].append(morph)

        if self.morphs2str:
            for node in dg.nodes.values():
                node["word"] = self.morphs2str(node["word"])

        return dg.tree()


######################################################################
# Demo
######################################################################


def demo():

    import nltk
    from nltk.corpus.util import LazyCorpusLoader

    root = nltk.data.find("corpora/knbc/corpus1")
    fileids = [
        f
        for f in find_corpus_fileids(FileSystemPathPointer(root), ".*")
        if re.search(r"\d\-\d\-[\d]+\-[\d]+", f)
    ]

    def _knbc_fileids_sort(x):
        cells = x.split("-")
        return (cells[0], int(cells[1]), int(cells[2]), int(cells[3]))

    knbc = LazyCorpusLoader(
        "knbc/corpus1",
        KNBCorpusReader,
        sorted(fileids, key=_knbc_fileids_sort),
        encoding="euc-jp",
    )

    print(knbc.fileids()[:10])
    print("".join(knbc.words()[:100]))

    print("\n\n".join(str(tree) for tree in knbc.parsed_sents()[:2]))

    knbc.morphs2str = lambda morphs: "/".join(
        "{}({})".format(m[0], m[1].split(" ")[2]) for m in morphs if m[0] != "EOS"
    ).encode("utf-8")

    print("\n\n".join("%s" % tree for tree in knbc.parsed_sents()[:2]))

    print(
        "\n".join(
            " ".join("{}/{}".format(w[0], w[1].split(" ")[2]) for w in sent)
            for sent in knbc.tagged_sents()[0:2]
        )
    )


def test():

    from nltk.corpus.util import LazyCorpusLoader

    knbc = LazyCorpusLoader(
        "knbc/corpus1", KNBCorpusReader, r".*/KN.*", encoding="euc-jp"
    )
    assert isinstance(knbc.words()[0], str)
    assert isinstance(knbc.sents()[0][0], str)
    assert isinstance(knbc.tagged_words()[0], tuple)
    assert isinstance(knbc.tagged_sents()[0][0], tuple)


if __name__ == "__main__":
    demo()
