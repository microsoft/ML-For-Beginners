# Natural Language Toolkit: Chunkers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Classes and interfaces for identifying non-overlapping linguistic
groups (such as base noun phrases) in unrestricted text.  This task is
called "chunk parsing" or "chunking", and the identified groups are
called "chunks".  The chunked text is represented using a shallow
tree called a "chunk structure."  A chunk structure is a tree
containing tokens and chunks, where each chunk is a subtree containing
only tokens.  For example, the chunk structure for base noun phrase
chunks in the sentence "I saw the big dog on the hill" is::

  (SENTENCE:
    (NP: <I>)
    <saw>
    (NP: <the> <big> <dog>)
    <on>
    (NP: <the> <hill>))

To convert a chunk structure back to a list of tokens, simply use the
chunk structure's ``leaves()`` method.

This module defines ``ChunkParserI``, a standard interface for
chunking texts; and ``RegexpChunkParser``, a regular-expression based
implementation of that interface. It also defines ``ChunkScore``, a
utility class for scoring chunk parsers.

RegexpChunkParser
=================

``RegexpChunkParser`` is an implementation of the chunk parser interface
that uses regular-expressions over tags to chunk a text.  Its
``parse()`` method first constructs a ``ChunkString``, which encodes a
particular chunking of the input text.  Initially, nothing is
chunked.  ``parse.RegexpChunkParser`` then applies a sequence of
``RegexpChunkRule`` rules to the ``ChunkString``, each of which modifies
the chunking that it encodes.  Finally, the ``ChunkString`` is
transformed back into a chunk structure, which is returned.

``RegexpChunkParser`` can only be used to chunk a single kind of phrase.
For example, you can use an ``RegexpChunkParser`` to chunk the noun
phrases in a text, or the verb phrases in a text; but you can not
use it to simultaneously chunk both noun phrases and verb phrases in
the same text.  (This is a limitation of ``RegexpChunkParser``, not of
chunk parsers in general.)

RegexpChunkRules
----------------

A ``RegexpChunkRule`` is a transformational rule that updates the
chunking of a text by modifying its ``ChunkString``.  Each
``RegexpChunkRule`` defines the ``apply()`` method, which modifies
the chunking encoded by a ``ChunkString``.  The
``RegexpChunkRule`` class itself can be used to implement any
transformational rule based on regular expressions.  There are
also a number of subclasses, which can be used to implement
simpler types of rules:

    - ``ChunkRule`` chunks anything that matches a given regular
      expression.
    - ``StripRule`` strips anything that matches a given regular
      expression.
    - ``UnChunkRule`` will un-chunk any chunk that matches a given
      regular expression.
    - ``MergeRule`` can be used to merge two contiguous chunks.
    - ``SplitRule`` can be used to split a single chunk into two
      smaller chunks.
    - ``ExpandLeftRule`` will expand a chunk to incorporate new
      unchunked material on the left.
    - ``ExpandRightRule`` will expand a chunk to incorporate new
      unchunked material on the right.

Tag Patterns
~~~~~~~~~~~~

A ``RegexpChunkRule`` uses a modified version of regular
expression patterns, called "tag patterns".  Tag patterns are
used to match sequences of tags.  Examples of tag patterns are::

     r'(<DT>|<JJ>|<NN>)+'
     r'<NN>+'
     r'<NN.*>'

The differences between regular expression patterns and tag
patterns are:

    - In tag patterns, ``'<'`` and ``'>'`` act as parentheses; so
      ``'<NN>+'`` matches one or more repetitions of ``'<NN>'``, not
      ``'<NN'`` followed by one or more repetitions of ``'>'``.
    - Whitespace in tag patterns is ignored.  So
      ``'<DT> | <NN>'`` is equivalent to ``'<DT>|<NN>'``
    - In tag patterns, ``'.'`` is equivalent to ``'[^{}<>]'``; so
      ``'<NN.*>'`` matches any single tag starting with ``'NN'``.

The function ``tag_pattern2re_pattern`` can be used to transform
a tag pattern to an equivalent regular expression pattern.

Efficiency
----------

Preliminary tests indicate that ``RegexpChunkParser`` can chunk at a
rate of about 300 tokens/second, with a moderately complex rule set.

There may be problems if ``RegexpChunkParser`` is used with more than
5,000 tokens at a time.  In particular, evaluation of some regular
expressions may cause the Python regular expression engine to
exceed its maximum recursion depth.  We have attempted to minimize
these problems, but it is impossible to avoid them completely.  We
therefore recommend that you apply the chunk parser to a single
sentence at a time.

Emacs Tip
---------

If you evaluate the following elisp expression in emacs, it will
colorize a ``ChunkString`` when you use an interactive python shell
with emacs or xemacs ("C-c !")::

    (let ()
      (defconst comint-mode-font-lock-keywords
        '(("<[^>]+>" 0 'font-lock-reference-face)
          ("[{}]" 0 'font-lock-function-name-face)))
      (add-hook 'comint-mode-hook (lambda () (turn-on-font-lock))))

You can evaluate this code by copying it to a temporary buffer,
placing the cursor after the last close parenthesis, and typing
"``C-x C-e``".  You should evaluate it before running the interactive
session.  The change will last until you close emacs.

Unresolved Issues
-----------------

If we use the ``re`` module for regular expressions, Python's
regular expression engine generates "maximum recursion depth
exceeded" errors when processing very large texts, even for
regular expressions that should not require any recursion.  We
therefore use the ``pre`` module instead.  But note that ``pre``
does not include Unicode support, so this module will not work
with unicode strings.  Note also that ``pre`` regular expressions
are not quite as advanced as ``re`` ones (e.g., no leftward
zero-length assertions).

:type CHUNK_TAG_PATTERN: regexp
:var CHUNK_TAG_PATTERN: A regular expression to test whether a tag
     pattern is valid.
"""

from nltk.chunk.api import ChunkParserI
from nltk.chunk.regexp import RegexpChunkParser, RegexpParser
from nltk.chunk.util import (
    ChunkScore,
    accuracy,
    conllstr2tree,
    conlltags2tree,
    ieerstr2tree,
    tagstr2tree,
    tree2conllstr,
    tree2conlltags,
)
from nltk.data import load

# Standard treebank POS tagger
_BINARY_NE_CHUNKER = "chunkers/maxent_ne_chunker/english_ace_binary.pickle"
_MULTICLASS_NE_CHUNKER = "chunkers/maxent_ne_chunker/english_ace_multiclass.pickle"


def ne_chunk(tagged_tokens, binary=False):
    """
    Use NLTK's currently recommended named entity chunker to
    chunk the given list of tagged tokens.
    """
    if binary:
        chunker_pickle = _BINARY_NE_CHUNKER
    else:
        chunker_pickle = _MULTICLASS_NE_CHUNKER
    chunker = load(chunker_pickle)
    return chunker.parse(tagged_tokens)


def ne_chunk_sents(tagged_sentences, binary=False):
    """
    Use NLTK's currently recommended named entity chunker to chunk the
    given list of tagged sentences, each consisting of a list of tagged tokens.
    """
    if binary:
        chunker_pickle = _BINARY_NE_CHUNKER
    else:
        chunker_pickle = _MULTICLASS_NE_CHUNKER
    chunker = load(chunker_pickle)
    return chunker.parse_sents(tagged_sentences)
