# Natural Language Toolkit: NIST Score
#
# Copyright (C) 2001-2023 NLTK Project
# Authors:
# Contributors:
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""NIST score implementation."""

import fractions
import math
from collections import Counter

from nltk.util import ngrams


def sentence_nist(references, hypothesis, n=5):
    """
    Calculate NIST score from
    George Doddington. 2002. "Automatic evaluation of machine translation quality
    using n-gram co-occurrence statistics." Proceedings of HLT.
    Morgan Kaufmann Publishers Inc. https://dl.acm.org/citation.cfm?id=1289189.1289273

    DARPA commissioned NIST to develop an MT evaluation facility based on the BLEU
    score. The official script used by NIST to compute BLEU and NIST score is
    mteval-14.pl. The main differences are:

     - BLEU uses geometric mean of the ngram overlaps, NIST uses arithmetic mean.
     - NIST has a different brevity penalty
     - NIST score from mteval-14.pl has a self-contained tokenizer

    Note: The mteval-14.pl includes a smoothing function for BLEU score that is NOT
          used in the NIST score computation.

    >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...               'ensures', 'that', 'the', 'military', 'always',
    ...               'obeys', 'the', 'commands', 'of', 'the', 'party']

    >>> hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
    ...               'forever', 'hearing', 'the', 'activity', 'guidebook',
    ...               'that', 'party', 'direct']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...               'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...               'heed', 'Party', 'commands']

    >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...               'guarantees', 'the', 'military', 'forces', 'always',
    ...               'being', 'under', 'the', 'command', 'of', 'the',
    ...               'Party']

    >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...               'army', 'always', 'to', 'heed', 'the', 'directions',
    ...               'of', 'the', 'party']

    >>> sentence_nist([reference1, reference2, reference3], hypothesis1) # doctest: +ELLIPSIS
    3.3709...

    >>> sentence_nist([reference1, reference2, reference3], hypothesis2) # doctest: +ELLIPSIS
    1.4619...

    :param references: reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param n: highest n-gram order
    :type n: int
    """
    return corpus_nist([references], [hypothesis], n)


def corpus_nist(list_of_references, hypotheses, n=5):
    """
    Calculate a single corpus-level NIST score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param n: highest n-gram order
    :type n: int
    """
    # Before proceeding to compute NIST, perform sanity checks.
    assert len(list_of_references) == len(
        hypotheses
    ), "The number of hypotheses and their reference(s) should be the same"

    # Collect the ngram coounts from the reference sentences.
    ngram_freq = Counter()
    total_reference_words = 0
    for (
        references
    ) in list_of_references:  # For each source sent, there's a list of reference sents.
        for reference in references:
            # For each order of ngram, count the ngram occurrences.
            for i in range(1, n + 1):
                ngram_freq.update(ngrams(reference, i))
            total_reference_words += len(reference)

    # Compute the information weights based on the reference sentences.
    # Eqn 2 in Doddington (2002):
    # Info(w_1 ... w_n) = log_2 [ (# of occurrences of w_1 ... w_n-1) / (# of occurrences of w_1 ... w_n) ]
    information_weights = {}
    for _ngram in ngram_freq:  # w_1 ... w_n
        _mgram = _ngram[:-1]  #  w_1 ... w_n-1
        # From https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v13a.pl#L546
        # it's computed as such:
        #     denominator = ngram_freq[_mgram] if _mgram and _mgram in ngram_freq else denominator = total_reference_words
        #     information_weights[_ngram] = -1 * math.log(ngram_freq[_ngram]/denominator) / math.log(2)
        #
        # Mathematically, it's equivalent to the our implementation:
        if _mgram and _mgram in ngram_freq:
            numerator = ngram_freq[_mgram]
        else:
            numerator = total_reference_words
        information_weights[_ngram] = math.log(numerator / ngram_freq[_ngram], 2)

    # Micro-average.
    nist_precision_numerator_per_ngram = Counter()
    nist_precision_denominator_per_ngram = Counter()
    l_ref, l_sys = 0, 0
    # For each order of ngram.
    for i in range(1, n + 1):
        # Iterate through each hypothesis and their corresponding references.
        for references, hypothesis in zip(list_of_references, hypotheses):
            hyp_len = len(hypothesis)

            # Find reference with the best NIST score.
            nist_score_per_ref = []
            for reference in references:
                _ref_len = len(reference)
                # Counter of ngrams in hypothesis.
                hyp_ngrams = (
                    Counter(ngrams(hypothesis, i))
                    if len(hypothesis) >= i
                    else Counter()
                )
                ref_ngrams = (
                    Counter(ngrams(reference, i)) if len(reference) >= i else Counter()
                )
                ngram_overlaps = hyp_ngrams & ref_ngrams
                # Precision part of the score in Eqn 3
                _numerator = sum(
                    information_weights[_ngram] * count
                    for _ngram, count in ngram_overlaps.items()
                )
                _denominator = sum(hyp_ngrams.values())
                _precision = 0 if _denominator == 0 else _numerator / _denominator
                nist_score_per_ref.append(
                    (_precision, _numerator, _denominator, _ref_len)
                )
            # Best reference.
            precision, numerator, denominator, ref_len = max(nist_score_per_ref)
            nist_precision_numerator_per_ngram[i] += numerator
            nist_precision_denominator_per_ngram[i] += denominator
            l_ref += ref_len
            l_sys += hyp_len

    # Final NIST micro-average mean aggregation.
    nist_precision = 0
    for i in nist_precision_numerator_per_ngram:
        precision = (
            nist_precision_numerator_per_ngram[i]
            / nist_precision_denominator_per_ngram[i]
        )
        nist_precision += precision
    # Eqn 3 in Doddington(2002)
    return nist_precision * nist_length_penalty(l_ref, l_sys)


def nist_length_penalty(ref_len, hyp_len):
    """
    Calculates the NIST length penalty, from Eq. 3 in Doddington (2002)

        penalty = exp( beta * log( min( len(hyp)/len(ref) , 1.0 )))

    where,

        `beta` is chosen to make the brevity penalty factor = 0.5 when the
        no. of words in the system output (hyp) is 2/3 of the average
        no. of words in the reference translation (ref)

    The NIST penalty is different from BLEU's such that it minimize the impact
    of the score of small variations in the length of a translation.
    See Fig. 4 in  Doddington (2002)
    """
    ratio = hyp_len / ref_len
    if 0 < ratio < 1:
        ratio_x, score_x = 1.5, 0.5
        beta = math.log(score_x) / math.log(ratio_x) ** 2
        return math.exp(beta * math.log(ratio) ** 2)
    else:  # ratio <= 0 or ratio >= 1
        return max(min(ratio, 1.0), 0.0)
