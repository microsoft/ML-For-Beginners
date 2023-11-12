# Natural Language Toolkit: ChrF score
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Maja Popovic
# Contributors: Liling Tan, Aleš Tamchyna (Memsource)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

""" ChrF score implementation """
import re
from collections import Counter, defaultdict

from nltk.util import ngrams


def sentence_chrf(
    reference, hypothesis, min_len=1, max_len=6, beta=3.0, ignore_whitespace=True
):
    """
    Calculates the sentence level CHRF (Character n-gram F-score) described in
     - Maja Popovic. 2015. CHRF: Character n-gram F-score for Automatic MT Evaluation.
       In Proceedings of the 10th Workshop on Machine Translation.
       https://www.statmt.org/wmt15/pdf/WMT49.pdf
     - Maja Popovic. 2016. CHRF Deconstructed: β Parameters and n-gram Weights.
       In Proceedings of the 1st Conference on Machine Translation.
       https://www.statmt.org/wmt16/pdf/W16-2341.pdf

    This implementation of CHRF only supports a single reference at the moment.

    For details not reported in the paper, consult Maja Popovic's original
    implementation: https://github.com/m-popovic/chrF

    The code should output results equivalent to running CHRF++ with the
    following options: -nw 0 -b 3

    An example from the original BLEU paper
    https://www.aclweb.org/anthology/P02-1040.pdf

        >>> ref1 = str('It is a guide to action that ensures that the military '
        ...            'will forever heed Party commands').split()
        >>> hyp1 = str('It is a guide to action which ensures that the military '
        ...            'always obeys the commands of the party').split()
        >>> hyp2 = str('It is to insure the troops forever hearing the activity '
        ...            'guidebook that party direct').split()
        >>> sentence_chrf(ref1, hyp1) # doctest: +ELLIPSIS
        0.6349...
        >>> sentence_chrf(ref1, hyp2) # doctest: +ELLIPSIS
        0.3330...

    The infamous "the the the ... " example

        >>> ref = 'the cat is on the mat'.split()
        >>> hyp = 'the the the the the the the'.split()
        >>> sentence_chrf(ref, hyp)  # doctest: +ELLIPSIS
        0.1468...

    An example to show that this function allows users to use strings instead of
    tokens, i.e. list(str) as inputs.

        >>> ref1 = str('It is a guide to action that ensures that the military '
        ...            'will forever heed Party commands')
        >>> hyp1 = str('It is a guide to action which ensures that the military '
        ...            'always obeys the commands of the party')
        >>> sentence_chrf(ref1, hyp1) # doctest: +ELLIPSIS
        0.6349...
        >>> type(ref1) == type(hyp1) == str
        True
        >>> sentence_chrf(ref1.split(), hyp1.split()) # doctest: +ELLIPSIS
        0.6349...

    To skip the unigrams and only use 2- to 3-grams:

        >>> sentence_chrf(ref1, hyp1, min_len=2, max_len=3) # doctest: +ELLIPSIS
        0.6617...

    :param references: reference sentence
    :type references: list(str) / str
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str) / str
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :param beta: the parameter to assign more importance to recall over precision
    :type beta: float
    :param ignore_whitespace: ignore whitespace characters in scoring
    :type ignore_whitespace: bool
    :return: the sentence level CHRF score.
    :rtype: float
    """
    return corpus_chrf(
        [reference],
        [hypothesis],
        min_len,
        max_len,
        beta=beta,
        ignore_whitespace=ignore_whitespace,
    )


def _preprocess(sent, ignore_whitespace):
    if type(sent) != str:
        # turn list of tokens into a string
        sent = " ".join(sent)

    if ignore_whitespace:
        sent = re.sub(r"\s+", "", sent)
    return sent


def chrf_precision_recall_fscore_support(
    reference, hypothesis, n, beta=3.0, epsilon=1e-16
):
    """
    This function computes the precision, recall and fscore from the ngram
    overlaps. It returns the `support` which is the true positive score.

    By underspecifying the input type, the function will be agnostic as to how
    it computes the ngrams and simply take the whichever element in the list;
    it could be either token or character.

    :param reference: The reference sentence.
    :type reference: list
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: list
    :param n: Extract up to the n-th order ngrams
    :type n: int
    :param beta: The parameter to assign more importance to recall over precision.
    :type beta: float
    :param epsilon: The fallback value if the hypothesis or reference is empty.
    :type epsilon: float
    :return: Returns the precision, recall and f-score and support (true positive).
    :rtype: tuple(float)
    """
    ref_ngrams = Counter(ngrams(reference, n))
    hyp_ngrams = Counter(ngrams(hypothesis, n))

    # calculate the number of ngram matches
    overlap_ngrams = ref_ngrams & hyp_ngrams
    tp = sum(overlap_ngrams.values())  # True positives.
    tpfp = sum(hyp_ngrams.values())  # True positives + False positives.
    tpfn = sum(ref_ngrams.values())  # True positives + False negatives.

    try:
        prec = tp / tpfp  # precision
        rec = tp / tpfn  # recall
        factor = beta**2
        fscore = (1 + factor) * (prec * rec) / (factor * prec + rec)
    except ZeroDivisionError:
        prec = rec = fscore = epsilon
    return prec, rec, fscore, tp


def corpus_chrf(
    references, hypotheses, min_len=1, max_len=6, beta=3.0, ignore_whitespace=True
):
    """
    Calculates the corpus level CHRF (Character n-gram F-score), it is the
    macro-averaged value of the sentence/segment level CHRF score.

    This implementation of CHRF only supports a single reference at the moment.

        >>> ref1 = str('It is a guide to action that ensures that the military '
        ...            'will forever heed Party commands').split()
        >>> ref2 = str('It is the guiding principle which guarantees the military '
        ...            'forces always being under the command of the Party').split()
        >>>
        >>> hyp1 = str('It is a guide to action which ensures that the military '
        ...            'always obeys the commands of the party').split()
        >>> hyp2 = str('It is to insure the troops forever hearing the activity '
        ...            'guidebook that party direct')
        >>> corpus_chrf([ref1, ref2, ref1, ref2], [hyp1, hyp2, hyp2, hyp1]) # doctest: +ELLIPSIS
        0.3910...

    :param references: a corpus of list of reference sentences, w.r.t. hypotheses
    :type references: list(list(str))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :param beta: the parameter to assign more importance to recall over precision
    :type beta: float
    :param ignore_whitespace: ignore whitespace characters in scoring
    :type ignore_whitespace: bool
    :return: the sentence level CHRF score.
    :rtype: float
    """

    assert len(references) == len(
        hypotheses
    ), "The number of hypotheses and their references should be the same"
    num_sents = len(hypotheses)

    # Keep f-scores for each n-gram order separate
    ngram_fscores = defaultdict(lambda: list())

    # Iterate through each hypothesis and their corresponding references.
    for reference, hypothesis in zip(references, hypotheses):

        # preprocess both reference and hypothesis
        reference = _preprocess(reference, ignore_whitespace)
        hypothesis = _preprocess(hypothesis, ignore_whitespace)

        # Calculate f-scores for each sentence and for each n-gram order
        # separately.
        for n in range(min_len, max_len + 1):
            # Compute the precision, recall, fscore and support.
            prec, rec, fscore, tp = chrf_precision_recall_fscore_support(
                reference, hypothesis, n, beta=beta
            )
            ngram_fscores[n].append(fscore)

    # how many n-gram sizes
    num_ngram_sizes = len(ngram_fscores)

    # sum of f-scores over all sentences for each n-gram order
    total_scores = [sum(fscores) for n, fscores in ngram_fscores.items()]

    # macro-average over n-gram orders and over all sentences
    return (sum(total_scores) / num_ngram_sizes) / num_sents
