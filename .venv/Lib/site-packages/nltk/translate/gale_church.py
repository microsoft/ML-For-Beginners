# Natural Language Toolkit: Gale-Church Aligner
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Torsten Marek <marek@ifi.uzh.ch>
# Contributor: Cassidy Laidlaw, Liling Tan
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""

A port of the Gale-Church Aligner.

Gale & Church (1993), A Program for Aligning Sentences in Bilingual Corpora.
https://aclweb.org/anthology/J93-1004.pdf

"""

import math

try:
    from norm import logsf as norm_logsf
    from scipy.stats import norm
except ImportError:

    def erfcc(x):
        """Complementary error function."""
        z = abs(x)
        t = 1 / (1 + 0.5 * z)
        r = t * math.exp(
            -z * z
            - 1.26551223
            + t
            * (
                1.00002368
                + t
                * (
                    0.37409196
                    + t
                    * (
                        0.09678418
                        + t
                        * (
                            -0.18628806
                            + t
                            * (
                                0.27886807
                                + t
                                * (
                                    -1.13520398
                                    + t
                                    * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                                )
                            )
                        )
                    )
                )
            )
        )
        if x >= 0.0:
            return r
        else:
            return 2.0 - r

    def norm_cdf(x):
        """Return the area under the normal distribution from M{-∞..x}."""
        return 1 - 0.5 * erfcc(x / math.sqrt(2))

    def norm_logsf(x):
        try:
            return math.log(1 - norm_cdf(x))
        except ValueError:
            return float("-inf")


LOG2 = math.log(2)


class LanguageIndependent:
    # These are the language-independent probabilities and parameters
    # given in Gale & Church

    # for the computation, l_1 is always the language with less characters
    PRIORS = {
        (1, 0): 0.0099,
        (0, 1): 0.0099,
        (1, 1): 0.89,
        (2, 1): 0.089,
        (1, 2): 0.089,
        (2, 2): 0.011,
    }

    AVERAGE_CHARACTERS = 1
    VARIANCE_CHARACTERS = 6.8


def trace(backlinks, source_sents_lens, target_sents_lens):
    """
    Traverse the alignment cost from the tracebacks and retrieves
    appropriate sentence pairs.

    :param backlinks: A dictionary where the key is the alignment points and value is the cost (referencing the LanguageIndependent.PRIORS)
    :type backlinks: dict
    :param source_sents_lens: A list of target sentences' lengths
    :type source_sents_lens: list(int)
    :param target_sents_lens: A list of target sentences' lengths
    :type target_sents_lens: list(int)
    """
    links = []
    position = (len(source_sents_lens), len(target_sents_lens))
    while position != (0, 0) and all(p >= 0 for p in position):
        try:
            s, t = backlinks[position]
        except TypeError:
            position = (position[0] - 1, position[1] - 1)
            continue
        for i in range(s):
            for j in range(t):
                links.append((position[0] - i - 1, position[1] - j - 1))
        position = (position[0] - s, position[1] - t)

    return links[::-1]


def align_log_prob(i, j, source_sents, target_sents, alignment, params):
    """Returns the log probability of the two sentences C{source_sents[i]}, C{target_sents[j]}
    being aligned with a specific C{alignment}.

    @param i: The offset of the source sentence.
    @param j: The offset of the target sentence.
    @param source_sents: The list of source sentence lengths.
    @param target_sents: The list of target sentence lengths.
    @param alignment: The alignment type, a tuple of two integers.
    @param params: The sentence alignment parameters.

    @returns: The log probability of a specific alignment between the two sentences, given the parameters.
    """
    l_s = sum(source_sents[i - offset - 1] for offset in range(alignment[0]))
    l_t = sum(target_sents[j - offset - 1] for offset in range(alignment[1]))
    try:
        # actually, the paper says l_s * params.VARIANCE_CHARACTERS, this is based on the C
        # reference implementation. With l_s in the denominator, insertions are impossible.
        m = (l_s + l_t / params.AVERAGE_CHARACTERS) / 2
        delta = (l_s * params.AVERAGE_CHARACTERS - l_t) / math.sqrt(
            m * params.VARIANCE_CHARACTERS
        )
    except ZeroDivisionError:
        return float("-inf")

    return -(LOG2 + norm_logsf(abs(delta)) + math.log(params.PRIORS[alignment]))


def align_blocks(source_sents_lens, target_sents_lens, params=LanguageIndependent):
    """Return the sentence alignment of two text blocks (usually paragraphs).

        >>> align_blocks([5,5,5], [7,7,7])
        [(0, 0), (1, 1), (2, 2)]
        >>> align_blocks([10,5,5], [12,20])
        [(0, 0), (1, 1), (2, 1)]
        >>> align_blocks([12,20], [10,5,5])
        [(0, 0), (1, 1), (1, 2)]
        >>> align_blocks([10,2,10,10,2,10], [12,3,20,3,12])
        [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 4)]

    @param source_sents_lens: The list of source sentence lengths.
    @param target_sents_lens: The list of target sentence lengths.
    @param params: the sentence alignment parameters.
    @return: The sentence alignments, a list of index pairs.
    """

    alignment_types = list(params.PRIORS.keys())

    # there are always three rows in the history (with the last of them being filled)
    D = [[]]

    backlinks = {}

    for i in range(len(source_sents_lens) + 1):
        for j in range(len(target_sents_lens) + 1):
            min_dist = float("inf")
            min_align = None
            for a in alignment_types:
                prev_i = -1 - a[0]
                prev_j = j - a[1]
                if prev_i < -len(D) or prev_j < 0:
                    continue
                p = D[prev_i][prev_j] + align_log_prob(
                    i, j, source_sents_lens, target_sents_lens, a, params
                )
                if p < min_dist:
                    min_dist = p
                    min_align = a

            if min_dist == float("inf"):
                min_dist = 0

            backlinks[(i, j)] = min_align
            D[-1].append(min_dist)

        if len(D) > 2:
            D.pop(0)
        D.append([])

    return trace(backlinks, source_sents_lens, target_sents_lens)


def align_texts(source_blocks, target_blocks, params=LanguageIndependent):
    """Creates the sentence alignment of two texts.

    Texts can consist of several blocks. Block boundaries cannot be crossed by sentence
    alignment links.

    Each block consists of a list that contains the lengths (in characters) of the sentences
    in this block.

    @param source_blocks: The list of blocks in the source text.
    @param target_blocks: The list of blocks in the target text.
    @param params: the sentence alignment parameters.

    @returns: A list of sentence alignment lists
    """
    if len(source_blocks) != len(target_blocks):
        raise ValueError(
            "Source and target texts do not have the same number of blocks."
        )

    return [
        align_blocks(source_block, target_block, params)
        for source_block, target_block in zip(source_blocks, target_blocks)
    ]


# File I/O functions; may belong in a corpus reader


def split_at(it, split_value):
    """Splits an iterator C{it} at values of C{split_value}.

    Each instance of C{split_value} is swallowed. The iterator produces
    subiterators which need to be consumed fully before the next subiterator
    can be used.
    """

    def _chunk_iterator(first):
        v = first
        while v != split_value:
            yield v
            v = it.next()

    while True:
        yield _chunk_iterator(it.next())


def parse_token_stream(stream, soft_delimiter, hard_delimiter):
    """Parses a stream of tokens and splits it into sentences (using C{soft_delimiter} tokens)
    and blocks (using C{hard_delimiter} tokens) for use with the L{align_texts} function.
    """
    return [
        [
            sum(len(token) for token in sentence_it)
            for sentence_it in split_at(block_it, soft_delimiter)
        ]
        for block_it in split_at(stream, hard_delimiter)
    ]
