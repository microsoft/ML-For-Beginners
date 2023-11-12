# Natural Language Toolkit: Parser Utility Functions
#
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Tom Aarsen <>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


"""
Utility functions for parsers.
"""

from nltk.data import load
from nltk.grammar import CFG, PCFG, FeatureGrammar
from nltk.parse.chart import Chart, ChartParser
from nltk.parse.featurechart import FeatureChart, FeatureChartParser
from nltk.parse.pchart import InsideChartParser


def load_parser(
    grammar_url, trace=0, parser=None, chart_class=None, beam_size=0, **load_args
):
    """
    Load a grammar from a file, and build a parser based on that grammar.
    The parser depends on the grammar format, and might also depend
    on properties of the grammar itself.

    The following grammar formats are currently supported:
      - ``'cfg'``  (CFGs: ``CFG``)
      - ``'pcfg'`` (probabilistic CFGs: ``PCFG``)
      - ``'fcfg'`` (feature-based CFGs: ``FeatureGrammar``)

    :type grammar_url: str
    :param grammar_url: A URL specifying where the grammar is located.
        The default protocol is ``"nltk:"``, which searches for the file
        in the the NLTK data package.
    :type trace: int
    :param trace: The level of tracing that should be used when
        parsing a text.  ``0`` will generate no tracing output;
        and higher numbers will produce more verbose tracing output.
    :param parser: The class used for parsing; should be ``ChartParser``
        or a subclass.
        If None, the class depends on the grammar format.
    :param chart_class: The class used for storing the chart;
        should be ``Chart`` or a subclass.
        Only used for CFGs and feature CFGs.
        If None, the chart class depends on the grammar format.
    :type beam_size: int
    :param beam_size: The maximum length for the parser's edge queue.
        Only used for probabilistic CFGs.
    :param load_args: Keyword parameters used when loading the grammar.
        See ``data.load`` for more information.
    """
    grammar = load(grammar_url, **load_args)
    if not isinstance(grammar, CFG):
        raise ValueError("The grammar must be a CFG, " "or a subclass thereof.")
    if isinstance(grammar, PCFG):
        if parser is None:
            parser = InsideChartParser
        return parser(grammar, trace=trace, beam_size=beam_size)

    elif isinstance(grammar, FeatureGrammar):
        if parser is None:
            parser = FeatureChartParser
        if chart_class is None:
            chart_class = FeatureChart
        return parser(grammar, trace=trace, chart_class=chart_class)

    else:  # Plain CFG.
        if parser is None:
            parser = ChartParser
        if chart_class is None:
            chart_class = Chart
        return parser(grammar, trace=trace, chart_class=chart_class)


def taggedsent_to_conll(sentence):
    """
    A module to convert a single POS tagged sentence into CONLL format.

    >>> from nltk import word_tokenize, pos_tag
    >>> text = "This is a foobar sentence."
    >>> for line in taggedsent_to_conll(pos_tag(word_tokenize(text))): # doctest: +NORMALIZE_WHITESPACE
    ... 	print(line, end="")
        1	This	_	DT	DT	_	0	a	_	_
        2	is	_	VBZ	VBZ	_	0	a	_	_
        3	a	_	DT	DT	_	0	a	_	_
        4	foobar	_	JJ	JJ	_	0	a	_	_
        5	sentence	_	NN	NN	_	0	a	_	_
        6	.		_	.	.	_	0	a	_	_

    :param sentence: A single input sentence to parse
    :type sentence: list(tuple(str, str))
    :rtype: iter(str)
    :return: a generator yielding a single sentence in CONLL format.
    """
    for (i, (word, tag)) in enumerate(sentence, start=1):
        input_str = [str(i), word, "_", tag, tag, "_", "0", "a", "_", "_"]
        input_str = "\t".join(input_str) + "\n"
        yield input_str


def taggedsents_to_conll(sentences):
    """
    A module to convert the a POS tagged document stream
    (i.e. list of list of tuples, a list of sentences) and yield lines
    in CONLL format. This module yields one line per word and two newlines
    for end of sentence.

    >>> from nltk import word_tokenize, sent_tokenize, pos_tag
    >>> text = "This is a foobar sentence. Is that right?"
    >>> sentences = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(text)]
    >>> for line in taggedsents_to_conll(sentences): # doctest: +NORMALIZE_WHITESPACE
    ...     if line:
    ...         print(line, end="")
    1	This	_	DT	DT	_	0	a	_	_
    2	is	_	VBZ	VBZ	_	0	a	_	_
    3	a	_	DT	DT	_	0	a	_	_
    4	foobar	_	JJ	JJ	_	0	a	_	_
    5	sentence	_	NN	NN	_	0	a	_	_
    6	.		_	.	.	_	0	a	_	_
    <BLANKLINE>
    <BLANKLINE>
    1	Is	_	VBZ	VBZ	_	0	a	_	_
    2	that	_	IN	IN	_	0	a	_	_
    3	right	_	NN	NN	_	0	a	_	_
    4	?	_	.	.	_	0	a	_	_
    <BLANKLINE>
    <BLANKLINE>

    :param sentences: Input sentences to parse
    :type sentence: list(list(tuple(str, str)))
    :rtype: iter(str)
    :return: a generator yielding sentences in CONLL format.
    """
    for sentence in sentences:
        yield from taggedsent_to_conll(sentence)
        yield "\n\n"


######################################################################
# { Test Suites
######################################################################


class TestGrammar:
    """
    Unit tests for  CFG.
    """

    def __init__(self, grammar, suite, accept=None, reject=None):
        self.test_grammar = grammar

        self.cp = load_parser(grammar, trace=0)
        self.suite = suite
        self._accept = accept
        self._reject = reject

    def run(self, show_trees=False):
        """
        Sentences in the test suite are divided into two classes:

        - grammatical (``accept``) and
        - ungrammatical (``reject``).

        If a sentence should parse according to the grammar, the value of
        ``trees`` will be a non-empty list. If a sentence should be rejected
        according to the grammar, then the value of ``trees`` will be None.
        """
        for test in self.suite:
            print(test["doc"] + ":", end=" ")
            for key in ["accept", "reject"]:
                for sent in test[key]:
                    tokens = sent.split()
                    trees = list(self.cp.parse(tokens))
                    if show_trees and trees:
                        print()
                        print(sent)
                        for tree in trees:
                            print(tree)
                    if key == "accept":
                        if trees == []:
                            raise ValueError("Sentence '%s' failed to parse'" % sent)
                        else:
                            accepted = True
                    else:
                        if trees:
                            raise ValueError("Sentence '%s' received a parse'" % sent)
                        else:
                            rejected = True
            if accepted and rejected:
                print("All tests passed!")


def extract_test_sentences(string, comment_chars="#%;", encoding=None):
    """
    Parses a string with one test sentence per line.
    Lines can optionally begin with:

    - a bool, saying if the sentence is grammatical or not, or
    - an int, giving the number of parse trees is should have,

    The result information is followed by a colon, and then the sentence.
    Empty lines and lines beginning with a comment char are ignored.

    :return: a list of tuple of sentences and expected results,
        where a sentence is a list of str,
        and a result is None, or bool, or int

    :param comment_chars: ``str`` of possible comment characters.
    :param encoding: the encoding of the string, if it is binary
    """
    if encoding is not None:
        string = string.decode(encoding)
    sentences = []
    for sentence in string.split("\n"):
        if sentence == "" or sentence[0] in comment_chars:
            continue
        split_info = sentence.split(":", 1)
        result = None
        if len(split_info) == 2:
            if split_info[0] in ["True", "true", "False", "false"]:
                result = split_info[0] in ["True", "true"]
                sentence = split_info[1]
            else:
                result = int(split_info[0])
                sentence = split_info[1]
        tokens = sentence.split()
        if tokens == []:
            continue
        sentences += [(tokens, result)]
    return sentences
