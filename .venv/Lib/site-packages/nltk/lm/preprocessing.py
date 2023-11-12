# Natural Language Toolkit: Language Model Unit Tests
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
from functools import partial
from itertools import chain

from nltk.util import everygrams, pad_sequence

flatten = chain.from_iterable
pad_both_ends = partial(
    pad_sequence,
    pad_left=True,
    left_pad_symbol="<s>",
    pad_right=True,
    right_pad_symbol="</s>",
)
pad_both_ends.__doc__ = """Pads both ends of a sentence to length specified by ngram order.

    Following convention <s> pads the start of sentence </s> pads its end.
    """


def padded_everygrams(order, sentence):
    """Helper with some useful defaults.

    Applies pad_both_ends to sentence and follows it up with everygrams.
    """
    return everygrams(list(pad_both_ends(sentence, n=order)), max_len=order)


def padded_everygram_pipeline(order, text):
    """Default preprocessing for a sequence of sentences.

    Creates two iterators:

    - sentences padded and turned into sequences of `nltk.util.everygrams`
    - sentences padded as above and chained together for a flat stream of words

    :param order: Largest ngram length produced by `everygrams`.
    :param text: Text to iterate over. Expected to be an iterable of sentences.
    :type text: Iterable[Iterable[str]]
    :return: iterator over text as ngrams, iterator over text as vocabulary data
    """
    padding_fn = partial(pad_both_ends, n=order)
    return (
        (everygrams(list(padding_fn(sent)), max_len=order) for sent in text),
        flatten(map(padding_fn, text)),
    )
