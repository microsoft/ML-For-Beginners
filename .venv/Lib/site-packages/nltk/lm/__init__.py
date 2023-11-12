# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/
# For license information, see LICENSE.TXT
"""
NLTK Language Modeling Module.
------------------------------

Currently this module covers only ngram language models, but it should be easy
to extend to neural models.


Preparing Data
==============

Before we train our ngram models it is necessary to make sure the data we put in
them is in the right format.
Let's say we have a text that is a list of sentences, where each sentence is
a list of strings. For simplicity we just consider a text consisting of
characters instead of words.

    >>> text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]

If we want to train a bigram model, we need to turn this text into bigrams.
Here's what the first sentence of our text would look like if we use a function
from NLTK for this.

    >>> from nltk.util import bigrams
    >>> list(bigrams(text[0]))
    [('a', 'b'), ('b', 'c')]

Notice how "b" occurs both as the first and second member of different bigrams
but "a" and "c" don't? Wouldn't it be nice to somehow indicate how often sentences
start with "a" and end with "c"?
A standard way to deal with this is to add special "padding" symbols to the
sentence before splitting it into ngrams.
Fortunately, NLTK also has a function for that, let's see what it does to the
first sentence.

    >>> from nltk.util import pad_sequence
    >>> list(pad_sequence(text[0],
    ... pad_left=True,
    ... left_pad_symbol="<s>",
    ... pad_right=True,
    ... right_pad_symbol="</s>",
    ... n=2))
    ['<s>', 'a', 'b', 'c', '</s>']

Note the `n` argument, that tells the function we need padding for bigrams.
Now, passing all these parameters every time is tedious and in most cases they
can be safely assumed as defaults anyway.
Thus our module provides a convenience function that has all these arguments
already set while the other arguments remain the same as for `pad_sequence`.

    >>> from nltk.lm.preprocessing import pad_both_ends
    >>> list(pad_both_ends(text[0], n=2))
    ['<s>', 'a', 'b', 'c', '</s>']

Combining the two parts discussed so far we get the following preparation steps
for one sentence.

    >>> list(bigrams(pad_both_ends(text[0], n=2)))
    [('<s>', 'a'), ('a', 'b'), ('b', 'c'), ('c', '</s>')]

To make our model more robust we could also train it on unigrams (single words)
as well as bigrams, its main source of information.
NLTK once again helpfully provides a function called `everygrams`.
While not the most efficient, it is conceptually simple.


    >>> from nltk.util import everygrams
    >>> padded_bigrams = list(pad_both_ends(text[0], n=2))
    >>> list(everygrams(padded_bigrams, max_len=2))
    [('<s>',), ('<s>', 'a'), ('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',), ('c', '</s>'), ('</s>',)]

We are almost ready to start counting ngrams, just one more step left.
During training and evaluation our model will rely on a vocabulary that
defines which words are "known" to the model.
To create this vocabulary we need to pad our sentences (just like for counting
ngrams) and then combine the sentences into one flat stream of words.

    >>> from nltk.lm.preprocessing import flatten
    >>> list(flatten(pad_both_ends(sent, n=2) for sent in text))
    ['<s>', 'a', 'b', 'c', '</s>', '<s>', 'a', 'c', 'd', 'c', 'e', 'f', '</s>']

In most cases we want to use the same text as the source for both vocabulary
and ngram counts.
Now that we understand what this means for our preprocessing, we can simply import
a function that does everything for us.

    >>> from nltk.lm.preprocessing import padded_everygram_pipeline
    >>> train, vocab = padded_everygram_pipeline(2, text)

So as to avoid re-creating the text in memory, both `train` and `vocab` are lazy
iterators. They are evaluated on demand at training time.


Training
========
Having prepared our data we are ready to start training a model.
As a simple example, let us train a Maximum Likelihood Estimator (MLE).
We only need to specify the highest ngram order to instantiate it.

    >>> from nltk.lm import MLE
    >>> lm = MLE(2)

This automatically creates an empty vocabulary...

    >>> len(lm.vocab)
    0

... which gets filled as we fit the model.

    >>> lm.fit(train, vocab)
    >>> print(lm.vocab)
    <Vocabulary with cutoff=1 unk_label='<UNK>' and 9 items>
    >>> len(lm.vocab)
    9

The vocabulary helps us handle words that have not occurred during training.

    >>> lm.vocab.lookup(text[0])
    ('a', 'b', 'c')
    >>> lm.vocab.lookup(["aliens", "from", "Mars"])
    ('<UNK>', '<UNK>', '<UNK>')

Moreover, in some cases we want to ignore words that we did see during training
but that didn't occur frequently enough, to provide us useful information.
You can tell the vocabulary to ignore such words.
To find out how that works, check out the docs for the `Vocabulary` class.


Using a Trained Model
=====================
When it comes to ngram models the training boils down to counting up the ngrams
from the training corpus.

    >>> print(lm.counts)
    <NgramCounter with 2 ngram orders and 24 ngrams>

This provides a convenient interface to access counts for unigrams...

    >>> lm.counts['a']
    2

...and bigrams (in this case "a b")

    >>> lm.counts[['a']]['b']
    1

And so on. However, the real purpose of training a language model is to have it
score how probable words are in certain contexts.
This being MLE, the model returns the item's relative frequency as its score.

    >>> lm.score("a")
    0.15384615384615385

Items that are not seen during training are mapped to the vocabulary's
"unknown label" token. This is "<UNK>" by default.

    >>> lm.score("<UNK>") == lm.score("aliens")
    True

Here's how you get the score for a word given some preceding context.
For example we want to know what is the chance that "b" is preceded by "a".

    >>> lm.score("b", ["a"])
    0.5

To avoid underflow when working with many small score values it makes sense to
take their logarithm.
For convenience this can be done with the `logscore` method.

    >>> lm.logscore("a")
    -2.700439718141092

Building on this method, we can also evaluate our model's cross-entropy and
perplexity with respect to sequences of ngrams.

    >>> test = [('a', 'b'), ('c', 'd')]
    >>> lm.entropy(test)
    1.292481250360578
    >>> lm.perplexity(test)
    2.449489742783178

It is advisable to preprocess your test text exactly the same way as you did
the training text.

One cool feature of ngram models is that they can be used to generate text.

    >>> lm.generate(1, random_seed=3)
    '<s>'
    >>> lm.generate(5, random_seed=3)
    ['<s>', 'a', 'b', 'c', 'd']

Provide `random_seed` if you want to consistently reproduce the same text all
other things being equal. Here we are using it to test the examples.

You can also condition your generation on some preceding text with the `context`
argument.

    >>> lm.generate(5, text_seed=['c'], random_seed=3)
    ['</s>', 'c', 'd', 'c', 'd']

Note that an ngram model is restricted in how much preceding context it can
take into account. For example, a trigram model can only condition its output
on 2 preceding words. If you pass in a 4-word context, the first two words
will be ignored.
"""

from nltk.lm.counter import NgramCounter
from nltk.lm.models import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,
    Laplace,
    Lidstone,
    StupidBackoff,
    WittenBellInterpolated,
)
from nltk.lm.vocabulary import Vocabulary

__all__ = [
    "Vocabulary",
    "NgramCounter",
    "MLE",
    "Lidstone",
    "Laplace",
    "WittenBellInterpolated",
    "KneserNeyInterpolated",
    "AbsoluteDiscountingInterpolated",
    "StupidBackoff",
]
