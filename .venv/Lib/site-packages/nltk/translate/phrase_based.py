# Natural Language Toolkit: Phrase Extraction Algorithm
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Liling Tan, Fredrik Hedman, Petra Barancikova
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


def extract(
    f_start,
    f_end,
    e_start,
    e_end,
    alignment,
    f_aligned,
    srctext,
    trgtext,
    srclen,
    trglen,
    max_phrase_length,
):
    """
    This function checks for alignment point consistency and extracts
    phrases using the chunk of consistent phrases.

    A phrase pair (e, f ) is consistent with an alignment A if and only if:

    (i) No English words in the phrase pair are aligned to words outside it.

           ∀e i ∈ e, (e i , f j ) ∈ A ⇒ f j ∈ f

    (ii) No Foreign words in the phrase pair are aligned to words outside it.

            ∀f j ∈ f , (e i , f j ) ∈ A ⇒ e i ∈ e

    (iii) The phrase pair contains at least one alignment point.

            ∃e i ∈ e  ̄ , f j ∈ f  ̄ s.t. (e i , f j ) ∈ A

    :type f_start: int
    :param f_start: Starting index of the possible foreign language phrases
    :type f_end: int
    :param f_end: End index of the possible foreign language phrases
    :type e_start: int
    :param e_start: Starting index of the possible source language phrases
    :type e_end: int
    :param e_end: End index of the possible source language phrases
    :type srctext: list
    :param srctext: The source language tokens, a list of string.
    :type trgtext: list
    :param trgtext: The target language tokens, a list of string.
    :type srclen: int
    :param srclen: The number of tokens in the source language tokens.
    :type trglen: int
    :param trglen: The number of tokens in the target language tokens.
    """

    if f_end < 0:  # 0-based indexing.
        return {}
    # Check if alignment points are consistent.
    for e, f in alignment:
        if (f_start <= f <= f_end) and (e < e_start or e > e_end):
            return {}

    # Add phrase pairs (incl. additional unaligned f)
    phrases = set()
    fs = f_start
    while True:
        fe = min(f_end, f_start + max_phrase_length - 1)
        while True:
            # add phrase pair ([e_start, e_end], [fs, fe]) to set E
            # Need to +1 in range  to include the end-point.
            src_phrase = " ".join(srctext[e_start : e_end + 1])
            trg_phrase = " ".join(trgtext[fs : fe + 1])
            # Include more data for later ordering.
            phrases.add(((e_start, e_end + 1), (fs, fe + 1), src_phrase, trg_phrase))
            fe += 1
            if fe in f_aligned or fe >= trglen:
                break
        fs -= 1
        if fs in f_aligned or fs < 0:
            break
    return phrases


def phrase_extraction(srctext, trgtext, alignment, max_phrase_length=0):
    """
    Phrase extraction algorithm extracts all consistent phrase pairs from
    a word-aligned sentence pair.

    The idea is to loop over all possible source language (e) phrases and find
    the minimal foreign phrase (f) that matches each of them. Matching is done
    by identifying all alignment points for the source phrase and finding the
    shortest foreign phrase that includes all the foreign counterparts for the
    source words.

    In short, a phrase alignment has to
    (a) contain all alignment points for all covered words
    (b) contain at least one alignment point

    >>> srctext = "michael assumes that he will stay in the house"
    >>> trgtext = "michael geht davon aus , dass er im haus bleibt"
    >>> alignment = [(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9),
    ... (5,9), (6,7), (7,7), (8,8)]
    >>> phrases = phrase_extraction(srctext, trgtext, alignment)
    >>> for i in sorted(phrases):
    ...    print(i)
    ...
    ((0, 1), (0, 1), 'michael', 'michael')
    ((0, 2), (0, 4), 'michael assumes', 'michael geht davon aus')
    ((0, 2), (0, 5), 'michael assumes', 'michael geht davon aus ,')
    ((0, 3), (0, 6), 'michael assumes that', 'michael geht davon aus , dass')
    ((0, 4), (0, 7), 'michael assumes that he', 'michael geht davon aus , dass er')
    ((0, 9), (0, 10), 'michael assumes that he will stay in the house', 'michael geht davon aus , dass er im haus bleibt')
    ((1, 2), (1, 4), 'assumes', 'geht davon aus')
    ((1, 2), (1, 5), 'assumes', 'geht davon aus ,')
    ((1, 3), (1, 6), 'assumes that', 'geht davon aus , dass')
    ((1, 4), (1, 7), 'assumes that he', 'geht davon aus , dass er')
    ((1, 9), (1, 10), 'assumes that he will stay in the house', 'geht davon aus , dass er im haus bleibt')
    ((2, 3), (4, 6), 'that', ', dass')
    ((2, 3), (5, 6), 'that', 'dass')
    ((2, 4), (4, 7), 'that he', ', dass er')
    ((2, 4), (5, 7), 'that he', 'dass er')
    ((2, 9), (4, 10), 'that he will stay in the house', ', dass er im haus bleibt')
    ((2, 9), (5, 10), 'that he will stay in the house', 'dass er im haus bleibt')
    ((3, 4), (6, 7), 'he', 'er')
    ((3, 9), (6, 10), 'he will stay in the house', 'er im haus bleibt')
    ((4, 6), (9, 10), 'will stay', 'bleibt')
    ((4, 9), (7, 10), 'will stay in the house', 'im haus bleibt')
    ((6, 8), (7, 8), 'in the', 'im')
    ((6, 9), (7, 9), 'in the house', 'im haus')
    ((8, 9), (8, 9), 'house', 'haus')

    :type srctext: str
    :param srctext: The sentence string from the source language.
    :type trgtext: str
    :param trgtext: The sentence string from the target language.
    :type alignment: list(tuple)
    :param alignment: The word alignment outputs as list of tuples, where
        the first elements of tuples are the source words' indices and
        second elements are the target words' indices. This is also the output
        format of nltk.translate.ibm1
    :rtype: list(tuple)
    :return: A list of tuples, each element in a list is a phrase and each
        phrase is a tuple made up of (i) its source location, (ii) its target
        location, (iii) the source phrase and (iii) the target phrase. The phrase
        list of tuples represents all the possible phrases extracted from the
        word alignments.
    :type max_phrase_length: int
    :param max_phrase_length: maximal phrase length, if 0 or not specified
        it is set to a length of the longer sentence (srctext or trgtext).
    """

    srctext = srctext.split()  # e
    trgtext = trgtext.split()  # f
    srclen = len(srctext)  # len(e)
    trglen = len(trgtext)  # len(f)
    # Keeps an index of which source/target words that are aligned.
    f_aligned = [j for _, j in alignment]
    max_phrase_length = max_phrase_length or max(srclen, trglen)

    # set of phrase pairs BP
    bp = set()

    for e_start in range(srclen):
        max_idx = min(srclen, e_start + max_phrase_length)
        for e_end in range(e_start, max_idx):
            # // find the minimally matching foreign phrase
            # (f start , f end ) = ( length(f), 0 )
            # f_start ∈ [0, len(f) - 1]; f_end ∈ [0, len(f) - 1]
            f_start, f_end = trglen - 1, -1  #  0-based indexing

            for e, f in alignment:
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            # add extract (f start , f end , e start , e end ) to set BP
            phrases = extract(
                f_start,
                f_end,
                e_start,
                e_end,
                alignment,
                f_aligned,
                srctext,
                trgtext,
                srclen,
                trglen,
                max_phrase_length,
            )
            if phrases:
                bp.update(phrases)
    return bp
