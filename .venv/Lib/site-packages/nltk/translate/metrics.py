# Natural Language Toolkit: Translation metrics
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Will Zhang <wilzzha@gmail.com>
#         Guan Gui <ggui@student.unimelb.edu.au>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


def alignment_error_rate(reference, hypothesis, possible=None):
    """
    Return the Alignment Error Rate (AER) of an alignment
    with respect to a "gold standard" reference alignment.
    Return an error rate between 0.0 (perfect alignment) and 1.0 (no
    alignment).

        >>> from nltk.translate import Alignment
        >>> ref = Alignment([(0, 0), (1, 1), (2, 2)])
        >>> test = Alignment([(0, 0), (1, 2), (2, 1)])
        >>> alignment_error_rate(ref, test) # doctest: +ELLIPSIS
        0.6666666666666667

    :type reference: Alignment
    :param reference: A gold standard alignment (sure alignments)
    :type hypothesis: Alignment
    :param hypothesis: A hypothesis alignment (aka. candidate alignments)
    :type possible: Alignment or None
    :param possible: A gold standard reference of possible alignments
        (defaults to *reference* if None)
    :rtype: float or None
    """

    if possible is None:
        possible = reference
    else:
        assert reference.issubset(possible)  # sanity check

    return 1.0 - (len(hypothesis & reference) + len(hypothesis & possible)) / float(
        len(hypothesis) + len(reference)
    )
