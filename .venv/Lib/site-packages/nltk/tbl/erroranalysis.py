# Natural Language Toolkit: Transformation-based learning
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Marcus Uneson <marcus.uneson@gmail.com>
#   based on previous (nltk2) version by
#   Christopher Maloof, Edward Loper, Steven Bird
# URL: <https://www.nltk.org/>
# For license information, see  LICENSE.TXT

# returns a list of errors in string format


def error_list(train_sents, test_sents):
    """
    Returns a list of human-readable strings indicating the errors in the
    given tagging of the corpus.

    :param train_sents: The correct tagging of the corpus
    :type train_sents: list(tuple)
    :param test_sents: The tagged corpus
    :type test_sents: list(tuple)
    """
    hdr = ("%25s | %s | %s\n" + "-" * 26 + "+" + "-" * 24 + "+" + "-" * 26) % (
        "left context",
        "word/test->gold".center(22),
        "right context",
    )
    errors = [hdr]
    for (train_sent, test_sent) in zip(train_sents, test_sents):
        for wordnum, (word, train_pos) in enumerate(train_sent):
            test_pos = test_sent[wordnum][1]
            if train_pos != test_pos:
                left = " ".join("%s/%s" % w for w in train_sent[:wordnum])
                right = " ".join("%s/%s" % w for w in train_sent[wordnum + 1 :])
                mid = f"{word}/{test_pos}->{train_pos}"
                errors.append(f"{left[-25:]:>25} | {mid.center(22)} | {right[:25]}")

    return errors
