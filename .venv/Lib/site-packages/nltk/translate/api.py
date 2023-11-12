# Natural Language Toolkit: API for alignment and translation objects
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Will Zhang <wilzzha@gmail.com>
#         Guan Gui <ggui@student.unimelb.edu.au>
#         Steven Bird <stevenbird1@gmail.com>
#         Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import subprocess
from collections import namedtuple


class AlignedSent:
    """
    Return an aligned sentence object, which encapsulates two sentences
    along with an ``Alignment`` between them.

    Typically used in machine translation to represent a sentence and
    its translation.

        >>> from nltk.translate import AlignedSent, Alignment
        >>> algnsent = AlignedSent(['klein', 'ist', 'das', 'Haus'],
        ...     ['the', 'house', 'is', 'small'], Alignment.fromstring('0-3 1-2 2-0 3-1'))
        >>> algnsent.words
        ['klein', 'ist', 'das', 'Haus']
        >>> algnsent.mots
        ['the', 'house', 'is', 'small']
        >>> algnsent.alignment
        Alignment([(0, 3), (1, 2), (2, 0), (3, 1)])
        >>> from nltk.corpus import comtrans
        >>> print(comtrans.aligned_sents()[54])
        <AlignedSent: 'Weshalb also sollten...' -> 'So why should EU arm...'>
        >>> print(comtrans.aligned_sents()[54].alignment)
        0-0 0-1 1-0 2-2 3-4 3-5 4-7 5-8 6-3 7-9 8-9 9-10 9-11 10-12 11-6 12-6 13-13

    :param words: Words in the target language sentence
    :type words: list(str)
    :param mots: Words in the source language sentence
    :type mots: list(str)
    :param alignment: Word-level alignments between ``words`` and ``mots``.
        Each alignment is represented as a 2-tuple (words_index, mots_index).
    :type alignment: Alignment
    """

    def __init__(self, words, mots, alignment=None):
        self._words = words
        self._mots = mots
        if alignment is None:
            self.alignment = Alignment([])
        else:
            assert type(alignment) is Alignment
            self.alignment = alignment

    @property
    def words(self):
        return self._words

    @property
    def mots(self):
        return self._mots

    def _get_alignment(self):
        return self._alignment

    def _set_alignment(self, alignment):
        _check_alignment(len(self.words), len(self.mots), alignment)
        self._alignment = alignment

    alignment = property(_get_alignment, _set_alignment)

    def __repr__(self):
        """
        Return a string representation for this ``AlignedSent``.

        :rtype: str
        """
        words = "[%s]" % (", ".join("'%s'" % w for w in self._words))
        mots = "[%s]" % (", ".join("'%s'" % w for w in self._mots))

        return f"AlignedSent({words}, {mots}, {self._alignment!r})"

    def _to_dot(self):
        """
        Dot representation of the aligned sentence
        """
        s = "graph align {\n"
        s += "node[shape=plaintext]\n"

        # Declare node
        for w in self._words:
            s += f'"{w}_source" [label="{w}"] \n'

        for w in self._mots:
            s += f'"{w}_target" [label="{w}"] \n'

        # Alignment
        for u, v in self._alignment:
            s += f'"{self._words[u]}_source" -- "{self._mots[v]}_target" \n'

        # Connect the source words
        for i in range(len(self._words) - 1):
            s += '"{}_source" -- "{}_source" [style=invis]\n'.format(
                self._words[i],
                self._words[i + 1],
            )

        # Connect the target words
        for i in range(len(self._mots) - 1):
            s += '"{}_target" -- "{}_target" [style=invis]\n'.format(
                self._mots[i],
                self._mots[i + 1],
            )

        # Put it in the same rank
        s += "{rank = same; %s}\n" % (" ".join('"%s_source"' % w for w in self._words))
        s += "{rank = same; %s}\n" % (" ".join('"%s_target"' % w for w in self._mots))

        s += "}"

        return s

    def _repr_svg_(self):
        """
        Ipython magic : show SVG representation of this ``AlignedSent``.
        """
        dot_string = self._to_dot().encode("utf8")
        output_format = "svg"
        try:
            process = subprocess.Popen(
                ["dot", "-T%s" % output_format],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            raise Exception("Cannot find the dot binary from Graphviz package") from e
        out, err = process.communicate(dot_string)

        return out.decode("utf8")

    def __str__(self):
        """
        Return a human-readable string representation for this ``AlignedSent``.

        :rtype: str
        """
        source = " ".join(self._words)[:20] + "..."
        target = " ".join(self._mots)[:20] + "..."
        return f"<AlignedSent: '{source}' -> '{target}'>"

    def invert(self):
        """
        Return the aligned sentence pair, reversing the directionality

        :rtype: AlignedSent
        """
        return AlignedSent(self._mots, self._words, self._alignment.invert())


class Alignment(frozenset):
    """
    A storage class for representing alignment between two sequences, s1, s2.
    In general, an alignment is a set of tuples of the form (i, j, ...)
    representing an alignment between the i-th element of s1 and the
    j-th element of s2.  Tuples are extensible (they might contain
    additional data, such as a boolean to indicate sure vs possible alignments).

        >>> from nltk.translate import Alignment
        >>> a = Alignment([(0, 0), (0, 1), (1, 2), (2, 2)])
        >>> a.invert()
        Alignment([(0, 0), (1, 0), (2, 1), (2, 2)])
        >>> print(a.invert())
        0-0 1-0 2-1 2-2
        >>> a[0]
        [(0, 1), (0, 0)]
        >>> a.invert()[2]
        [(2, 1), (2, 2)]
        >>> b = Alignment([(0, 0), (0, 1)])
        >>> b.issubset(a)
        True
        >>> c = Alignment.fromstring('0-0 0-1')
        >>> b == c
        True
    """

    def __new__(cls, pairs):
        self = frozenset.__new__(cls, pairs)
        self._len = max(p[0] for p in self) if self != frozenset([]) else 0
        self._index = None
        return self

    @classmethod
    def fromstring(cls, s):
        """
        Read a giza-formatted string and return an Alignment object.

            >>> Alignment.fromstring('0-0 2-1 9-2 21-3 10-4 7-5')
            Alignment([(0, 0), (2, 1), (7, 5), (9, 2), (10, 4), (21, 3)])

        :type s: str
        :param s: the positional alignments in giza format
        :rtype: Alignment
        :return: An Alignment object corresponding to the string representation ``s``.
        """

        return Alignment([_giza2pair(a) for a in s.split()])

    def __getitem__(self, key):
        """
        Look up the alignments that map from a given index or slice.
        """
        if not self._index:
            self._build_index()
        return self._index.__getitem__(key)

    def invert(self):
        """
        Return an Alignment object, being the inverted mapping.
        """
        return Alignment(((p[1], p[0]) + p[2:]) for p in self)

    def range(self, positions=None):
        """
        Work out the range of the mapping from the given positions.
        If no positions are specified, compute the range of the entire mapping.
        """
        image = set()
        if not self._index:
            self._build_index()
        if not positions:
            positions = list(range(len(self._index)))
        for p in positions:
            image.update(f for _, f in self._index[p])
        return sorted(image)

    def __repr__(self):
        """
        Produce a Giza-formatted string representing the alignment.
        """
        return "Alignment(%r)" % sorted(self)

    def __str__(self):
        """
        Produce a Giza-formatted string representing the alignment.
        """
        return " ".join("%d-%d" % p[:2] for p in sorted(self))

    def _build_index(self):
        """
        Build a list self._index such that self._index[i] is a list
        of the alignments originating from word i.
        """
        self._index = [[] for _ in range(self._len + 1)]
        for p in self:
            self._index[p[0]].append(p)


def _giza2pair(pair_string):
    i, j = pair_string.split("-")
    return int(i), int(j)


def _naacl2pair(pair_string):
    i, j, p = pair_string.split("-")
    return int(i), int(j)


def _check_alignment(num_words, num_mots, alignment):
    """
    Check whether the alignments are legal.

    :param num_words: the number of source language words
    :type num_words: int
    :param num_mots: the number of target language words
    :type num_mots: int
    :param alignment: alignment to be checked
    :type alignment: Alignment
    :raise IndexError: if alignment falls outside the sentence
    """

    assert type(alignment) is Alignment

    if not all(0 <= pair[0] < num_words for pair in alignment):
        raise IndexError("Alignment is outside boundary of words")
    if not all(pair[1] is None or 0 <= pair[1] < num_mots for pair in alignment):
        raise IndexError("Alignment is outside boundary of mots")


PhraseTableEntry = namedtuple("PhraseTableEntry", ["trg_phrase", "log_prob"])


class PhraseTable:
    """
    In-memory store of translations for a given phrase, and the log
    probability of the those translations
    """

    def __init__(self):
        self.src_phrases = dict()

    def translations_for(self, src_phrase):
        """
        Get the translations for a source language phrase

        :param src_phrase: Source language phrase of interest
        :type src_phrase: tuple(str)

        :return: A list of target language phrases that are translations
            of ``src_phrase``, ordered in decreasing order of
            likelihood. Each list element is a tuple of the target
            phrase and its log probability.
        :rtype: list(PhraseTableEntry)
        """
        return self.src_phrases[src_phrase]

    def add(self, src_phrase, trg_phrase, log_prob):
        """
        :type src_phrase: tuple(str)
        :type trg_phrase: tuple(str)

        :param log_prob: Log probability that given ``src_phrase``,
            ``trg_phrase`` is its translation
        :type log_prob: float
        """
        entry = PhraseTableEntry(trg_phrase=trg_phrase, log_prob=log_prob)
        if src_phrase not in self.src_phrases:
            self.src_phrases[src_phrase] = []
        self.src_phrases[src_phrase].append(entry)
        self.src_phrases[src_phrase].sort(key=lambda e: e.log_prob, reverse=True)

    def __contains__(self, src_phrase):
        return src_phrase in self.src_phrases
