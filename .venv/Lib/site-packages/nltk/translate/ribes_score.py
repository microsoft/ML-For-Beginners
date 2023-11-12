# Natural Language Toolkit: RIBES Score
#
# Copyright (C) 2001-2023 NLTK Project
# Contributors: Katsuhito Sudoh, Liling Tan, Kasramvd, J.F.Sebastian
#               Mark Byers, ekhumoro, P. Ortiz
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
""" RIBES score implementation """

import math
from itertools import islice

from nltk.util import choose, ngrams


def sentence_ribes(references, hypothesis, alpha=0.25, beta=0.10):
    """
    The RIBES (Rank-based Intuitive Bilingual Evaluation Score) from
    Hideki Isozaki, Tsutomu Hirao, Kevin Duh, Katsuhito Sudoh and
    Hajime Tsukada. 2010. "Automatic Evaluation of Translation Quality for
    Distant Language Pairs". In Proceedings of EMNLP.
    https://www.aclweb.org/anthology/D/D10/D10-1092.pdf

    The generic RIBES scores used in shared task, e.g. Workshop for
    Asian Translation (WAT) uses the following RIBES calculations:

        RIBES = kendall_tau * (alpha**p1) * (beta**bp)

    Please note that this re-implementation differs from the official
    RIBES implementation and though it emulates the results as describe
    in the original paper, there are further optimization implemented
    in the official RIBES script.

    Users are encouraged to use the official RIBES script instead of this
    implementation when evaluating your machine translation system. Refer
    to https://www.kecl.ntt.co.jp/icl/lirg/ribes/ for the official script.

    :param references: a list of reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param alpha: hyperparameter used as a prior for the unigram precision.
    :type alpha: float
    :param beta: hyperparameter used as a prior for the brevity penalty.
    :type beta: float
    :return: The best ribes score from one of the references.
    :rtype: float
    """
    best_ribes = -1.0
    # Calculates RIBES for each reference and returns the best score.
    for reference in references:
        # Collects the *worder* from the ranked correlation alignments.
        worder = word_rank_alignment(reference, hypothesis)
        nkt = kendall_tau(worder)

        # Calculates the brevity penalty
        bp = min(1.0, math.exp(1.0 - len(reference) / len(hypothesis)))

        # Calculates the unigram precision, *p1*
        p1 = len(worder) / len(hypothesis)

        _ribes = nkt * (p1**alpha) * (bp**beta)

        if _ribes > best_ribes:  # Keeps the best score.
            best_ribes = _ribes

    return best_ribes


def corpus_ribes(list_of_references, hypotheses, alpha=0.25, beta=0.10):
    """
    This function "calculates RIBES for a system output (hypothesis) with
    multiple references, and returns "best" score among multi-references and
    individual scores. The scores are corpus-wise, i.e., averaged by the number
    of sentences." (c.f. RIBES version 1.03.1 code).

    Different from BLEU's micro-average precision, RIBES calculates the
    macro-average precision by averaging the best RIBES score for each pair of
    hypothesis and its corresponding references

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> round(corpus_ribes(list_of_references, hypotheses),4)
    0.3597

    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param alpha: hyperparameter used as a prior for the unigram precision.
    :type alpha: float
    :param beta: hyperparameter used as a prior for the brevity penalty.
    :type beta: float
    :return: The best ribes score from one of the references.
    :rtype: float
    """
    corpus_best_ribes = 0.0
    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        corpus_best_ribes += sentence_ribes(references, hypothesis, alpha, beta)
    return corpus_best_ribes / len(hypotheses)


def position_of_ngram(ngram, sentence):
    """
    This function returns the position of the first instance of the ngram
    appearing in a sentence.

    Note that one could also use string as follows but the code is a little
    convoluted with type casting back and forth:

        char_pos = ' '.join(sent)[:' '.join(sent).index(' '.join(ngram))]
        word_pos = char_pos.count(' ')

    Another way to conceive this is:

        return next(i for i, ng in enumerate(ngrams(sentence, len(ngram)))
                    if ng == ngram)

    :param ngram: The ngram that needs to be searched
    :type ngram: tuple
    :param sentence: The list of tokens to search from.
    :type sentence: list(str)
    """
    # Iterates through the ngrams in sentence.
    for i, sublist in enumerate(ngrams(sentence, len(ngram))):
        # Returns the index of the word when ngram matches.
        if ngram == sublist:
            return i


def word_rank_alignment(reference, hypothesis, character_based=False):
    """
    This is the word rank alignment algorithm described in the paper to produce
    the *worder* list, i.e. a list of word indices of the hypothesis word orders
    w.r.t. the list of reference words.

    Below is (H0, R0) example from the Isozaki et al. 2010 paper,
    note the examples are indexed from 1 but the results here are indexed from 0:

        >>> ref = str('he was interested in world history because he '
        ... 'read the book').split()
        >>> hyp = str('he read the book because he was interested in world '
        ... 'history').split()
        >>> word_rank_alignment(ref, hyp)
        [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]

    The (H1, R1) example from the paper, note the 0th index:

        >>> ref = 'John hit Bob yesterday'.split()
        >>> hyp = 'Bob hit John yesterday'.split()
        >>> word_rank_alignment(ref, hyp)
        [2, 1, 0, 3]

    Here is the (H2, R2) example from the paper, note the 0th index here too:

        >>> ref = 'the boy read the book'.split()
        >>> hyp = 'the book was read by the boy'.split()
        >>> word_rank_alignment(ref, hyp)
        [3, 4, 2, 0, 1]

    :param reference: a reference sentence
    :type reference: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    """
    worder = []
    hyp_len = len(hypothesis)
    # Stores a list of possible ngrams from the reference sentence.
    # This is used for matching context window later in the algorithm.
    ref_ngrams = []
    hyp_ngrams = []
    for n in range(1, len(reference) + 1):
        for ng in ngrams(reference, n):
            ref_ngrams.append(ng)
        for ng in ngrams(hypothesis, n):
            hyp_ngrams.append(ng)
    for i, h_word in enumerate(hypothesis):
        # If word is not in the reference, continue.
        if h_word not in reference:
            continue
        # If we can determine one-to-one word correspondence for unigrams that
        # only appear once in both the reference and hypothesis.
        elif hypothesis.count(h_word) == reference.count(h_word) == 1:
            worder.append(reference.index(h_word))
        else:
            max_window_size = max(i, hyp_len - i + 1)
            for window in range(1, max_window_size):
                if i + window < hyp_len:  # If searching the right context is possible.
                    # Retrieve the right context window.
                    right_context_ngram = tuple(islice(hypothesis, i, i + window + 1))
                    num_times_in_ref = ref_ngrams.count(right_context_ngram)
                    num_times_in_hyp = hyp_ngrams.count(right_context_ngram)
                    # If ngram appears only once in both ref and hyp.
                    if num_times_in_ref == num_times_in_hyp == 1:
                        # Find the position of ngram that matched the reference.
                        pos = position_of_ngram(right_context_ngram, reference)
                        worder.append(pos)  # Add the positions of the ngram.
                        break
                if window <= i:  # If searching the left context is possible.
                    # Retrieve the left context window.
                    left_context_ngram = tuple(islice(hypothesis, i - window, i + 1))
                    num_times_in_ref = ref_ngrams.count(left_context_ngram)
                    num_times_in_hyp = hyp_ngrams.count(left_context_ngram)
                    if num_times_in_ref == num_times_in_hyp == 1:
                        # Find the position of ngram that matched the reference.
                        pos = position_of_ngram(left_context_ngram, reference)
                        # Add the positions of the ngram.
                        worder.append(pos + len(left_context_ngram) - 1)
                        break
    return worder


def find_increasing_sequences(worder):
    """
    Given the *worder* list, this function groups monotonic +1 sequences.

        >>> worder = [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]
        >>> list(find_increasing_sequences(worder))
        [(7, 8, 9, 10), (0, 1, 2, 3, 4, 5)]

    :param worder: The worder list output from word_rank_alignment
    :param type: list(int)
    """
    items = iter(worder)
    a, b = None, next(items, None)
    result = [b]
    while b is not None:
        a, b = b, next(items, None)
        if b is not None and a + 1 == b:
            result.append(b)
        else:
            if len(result) > 1:
                yield tuple(result)
            result = [b]


def kendall_tau(worder, normalize=True):
    """
    Calculates the Kendall's Tau correlation coefficient given the *worder*
    list of word alignments from word_rank_alignment(), using the formula:

        tau = 2 * num_increasing_pairs / num_possible_pairs -1

    Note that the no. of increasing pairs can be discontinuous in the *worder*
    list and each each increasing sequence can be tabulated as choose(len(seq), 2)
    no. of increasing pairs, e.g.

        >>> worder = [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]
        >>> number_possible_pairs = choose(len(worder), 2)
        >>> round(kendall_tau(worder, normalize=False),3)
        -0.236
        >>> round(kendall_tau(worder),3)
        0.382

    :param worder: The worder list output from word_rank_alignment
    :type worder: list(int)
    :param normalize: Flag to indicate normalization to between 0.0 and 1.0.
    :type normalize: boolean
    :return: The Kendall's Tau correlation coefficient.
    :rtype: float
    """
    worder_len = len(worder)
    # With worder_len < 2, `choose(worder_len, 2)` will be 0.
    # As we divide by this, it will give a ZeroDivisionError.
    # To avoid this, we can just return the lowest possible score.
    if worder_len < 2:
        tau = -1
    else:
        # Extract the groups of increasing/monotonic sequences.
        increasing_sequences = find_increasing_sequences(worder)
        # Calculate no. of increasing_pairs in *worder* list.
        num_increasing_pairs = sum(choose(len(seq), 2) for seq in increasing_sequences)
        # Calculate no. of possible pairs.
        num_possible_pairs = choose(worder_len, 2)
        # Kendall's Tau computation.
        tau = 2 * num_increasing_pairs / num_possible_pairs - 1
    if normalize:  # If normalized, the tau output falls between 0.0 to 1.0
        return (tau + 1) / 2
    else:  # Otherwise, the tau outputs falls between -1.0 to +1.0
        return tau


def spearman_rho(worder, normalize=True):
    """
    Calculates the Spearman's Rho correlation coefficient given the *worder*
    list of word alignment from word_rank_alignment(), using the formula:

        rho = 1 - sum(d**2) / choose(len(worder)+1, 3)

    Given that d is the sum of difference between the *worder* list of indices
    and the original word indices from the reference sentence.

    Using the (H0,R0) and (H5, R5) example from the paper

        >>> worder =  [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]
        >>> round(spearman_rho(worder, normalize=False), 3)
        -0.591
        >>> round(spearman_rho(worder), 3)
        0.205

    :param worder: The worder list output from word_rank_alignment
    :param type: list(int)
    """
    worder_len = len(worder)
    sum_d_square = sum((wi - i) ** 2 for wi, i in zip(worder, range(worder_len)))
    rho = 1 - sum_d_square / choose(worder_len + 1, 3)

    if normalize:  # If normalized, the rho output falls between 0.0 to 1.0
        return (rho + 1) / 2
    else:  # Otherwise, the rho outputs falls between -1.0 to +1.0
        return rho
