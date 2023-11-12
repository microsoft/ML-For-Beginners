# Natural Language Toolkit: Interface to BLLIP Parser
#
# Author: David McClosky <dmcc@bigasterisk.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from nltk.parse.api import ParserI
from nltk.tree import Tree

"""
Interface for parsing with BLLIP Parser. Requires the Python
bllipparser module. BllipParser objects can be constructed with the
``BllipParser.from_unified_model_dir`` class method or manually using the
``BllipParser`` constructor. The former is generally easier if you have
a BLLIP Parser unified model directory -- a basic model can be obtained
from NLTK's downloader. More unified parsing models can be obtained with
BLLIP Parser's ModelFetcher (run ``python -m bllipparser.ModelFetcher``
or see docs for ``bllipparser.ModelFetcher.download_and_install_model``).

Basic usage::

    # download and install a basic unified parsing model (Wall Street Journal)
    # sudo python -m nltk.downloader bllip_wsj_no_aux

    >>> from nltk.data import find
    >>> model_dir = find('models/bllip_wsj_no_aux').path
    >>> bllip = BllipParser.from_unified_model_dir(model_dir)

    # 1-best parsing
    >>> sentence1 = 'British left waffles on Falklands .'.split()
    >>> top_parse = bllip.parse_one(sentence1)
    >>> print(top_parse)
    (S1
      (S
        (NP (JJ British) (NN left))
        (VP (VBZ waffles) (PP (IN on) (NP (NNP Falklands))))
        (. .)))

    # n-best parsing
    >>> sentence2 = 'Time flies'.split()
    >>> all_parses = bllip.parse_all(sentence2)
    >>> print(len(all_parses))
    50
    >>> print(all_parses[0])
    (S1 (S (NP (NNP Time)) (VP (VBZ flies))))

    # incorporating external tagging constraints (None means unconstrained tag)
    >>> constrained1 = bllip.tagged_parse([('Time', 'VB'), ('flies', 'NNS')])
    >>> print(next(constrained1))
    (S1 (NP (VB Time) (NNS flies)))
    >>> constrained2 = bllip.tagged_parse([('Time', 'NN'), ('flies', None)])
    >>> print(next(constrained2))
    (S1 (NP (NN Time) (VBZ flies)))

References
----------

- Charniak, Eugene. "A maximum-entropy-inspired parser." Proceedings of
  the 1st North American chapter of the Association for Computational
  Linguistics conference. Association for Computational Linguistics,
  2000.

- Charniak, Eugene, and Mark Johnson. "Coarse-to-fine n-best parsing
  and MaxEnt discriminative reranking." Proceedings of the 43rd Annual
  Meeting on Association for Computational Linguistics. Association
  for Computational Linguistics, 2005.

Known issues
------------

Note that BLLIP Parser is not currently threadsafe. Since this module
uses a SWIG interface, it is potentially unsafe to create multiple
``BllipParser`` objects in the same process. BLLIP Parser currently
has issues with non-ASCII text and will raise an error if given any.

See https://pypi.python.org/pypi/bllipparser/ for more information
on BLLIP Parser's Python interface.
"""

__all__ = ["BllipParser"]

# this block allows this module to be imported even if bllipparser isn't
# available
try:
    from bllipparser import RerankingParser
    from bllipparser.RerankingParser import get_unified_model_parameters

    def _ensure_bllip_import_or_error():
        pass

except ImportError as ie:

    def _ensure_bllip_import_or_error(ie=ie):
        raise ImportError("Couldn't import bllipparser module: %s" % ie)


def _ensure_ascii(words):
    try:
        for i, word in enumerate(words):
            word.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(
            f"Token {i} ({word!r}) is non-ASCII. BLLIP Parser "
            "currently doesn't support non-ASCII inputs."
        ) from e


def _scored_parse_to_nltk_tree(scored_parse):
    return Tree.fromstring(str(scored_parse.ptb_parse))


class BllipParser(ParserI):
    """
    Interface for parsing with BLLIP Parser. BllipParser objects can be
    constructed with the ``BllipParser.from_unified_model_dir`` class
    method or manually using the ``BllipParser`` constructor.
    """

    def __init__(
        self,
        parser_model=None,
        reranker_features=None,
        reranker_weights=None,
        parser_options=None,
        reranker_options=None,
    ):
        """
        Load a BLLIP Parser model from scratch. You'll typically want to
        use the ``from_unified_model_dir()`` class method to construct
        this object.

        :param parser_model: Path to parser model directory
        :type parser_model: str

        :param reranker_features: Path the reranker model's features file
        :type reranker_features: str

        :param reranker_weights: Path the reranker model's weights file
        :type reranker_weights: str

        :param parser_options: optional dictionary of parser options, see
            ``bllipparser.RerankingParser.RerankingParser.load_parser_options()``
            for more information.
        :type parser_options: dict(str)

        :param reranker_options: optional
            dictionary of reranker options, see
            ``bllipparser.RerankingParser.RerankingParser.load_reranker_model()``
            for more information.
        :type reranker_options: dict(str)
        """
        _ensure_bllip_import_or_error()

        parser_options = parser_options or {}
        reranker_options = reranker_options or {}

        self.rrp = RerankingParser()
        self.rrp.load_parser_model(parser_model, **parser_options)
        if reranker_features and reranker_weights:
            self.rrp.load_reranker_model(
                features_filename=reranker_features,
                weights_filename=reranker_weights,
                **reranker_options,
            )

    def parse(self, sentence):
        """
        Use BLLIP Parser to parse a sentence. Takes a sentence as a list
        of words; it will be automatically tagged with this BLLIP Parser
        instance's tagger.

        :return: An iterator that generates parse trees for the sentence
            from most likely to least likely.

        :param sentence: The sentence to be parsed
        :type sentence: list(str)
        :rtype: iter(Tree)
        """
        _ensure_ascii(sentence)
        nbest_list = self.rrp.parse(sentence)
        for scored_parse in nbest_list:
            yield _scored_parse_to_nltk_tree(scored_parse)

    def tagged_parse(self, word_and_tag_pairs):
        """
        Use BLLIP to parse a sentence. Takes a sentence as a list of
        (word, tag) tuples; the sentence must have already been tokenized
        and tagged. BLLIP will attempt to use the tags provided but may
        use others if it can't come up with a complete parse subject
        to those constraints. You may also specify a tag as ``None``
        to leave a token's tag unconstrained.

        :return: An iterator that generates parse trees for the sentence
            from most likely to least likely.

        :param sentence: Input sentence to parse as (word, tag) pairs
        :type sentence: list(tuple(str, str))
        :rtype: iter(Tree)
        """
        words = []
        tag_map = {}
        for i, (word, tag) in enumerate(word_and_tag_pairs):
            words.append(word)
            if tag is not None:
                tag_map[i] = tag

        _ensure_ascii(words)
        nbest_list = self.rrp.parse_tagged(words, tag_map)
        for scored_parse in nbest_list:
            yield _scored_parse_to_nltk_tree(scored_parse)

    @classmethod
    def from_unified_model_dir(
        cls, model_dir, parser_options=None, reranker_options=None
    ):
        """
        Create a ``BllipParser`` object from a unified parsing model
        directory. Unified parsing model directories are a standardized
        way of storing BLLIP parser and reranker models together on disk.
        See ``bllipparser.RerankingParser.get_unified_model_parameters()``
        for more information about unified model directories.

        :return: A ``BllipParser`` object using the parser and reranker
            models in the model directory.

        :param model_dir: Path to the unified model directory.
        :type model_dir: str
        :param parser_options: optional dictionary of parser options, see
            ``bllipparser.RerankingParser.RerankingParser.load_parser_options()``
            for more information.
        :type parser_options: dict(str)
        :param reranker_options: optional dictionary of reranker options, see
            ``bllipparser.RerankingParser.RerankingParser.load_reranker_model()``
            for more information.
        :type reranker_options: dict(str)
        :rtype: BllipParser
        """
        (
            parser_model_dir,
            reranker_features_filename,
            reranker_weights_filename,
        ) = get_unified_model_parameters(model_dir)
        return cls(
            parser_model_dir,
            reranker_features_filename,
            reranker_weights_filename,
            parser_options,
            reranker_options,
        )


def demo():
    """This assumes the Python module bllipparser is installed."""

    # download and install a basic unified parsing model (Wall Street Journal)
    # sudo python -m nltk.downloader bllip_wsj_no_aux

    from nltk.data import find

    model_dir = find("models/bllip_wsj_no_aux").path

    print("Loading BLLIP Parsing models...")
    # the easiest way to get started is to use a unified model
    bllip = BllipParser.from_unified_model_dir(model_dir)
    print("Done.")

    sentence1 = "British left waffles on Falklands .".split()
    sentence2 = "I saw the man with the telescope .".split()
    # this sentence is known to fail under the WSJ parsing model
    fail1 = "# ! ? : -".split()
    for sentence in (sentence1, sentence2, fail1):
        print("Sentence: %r" % " ".join(sentence))
        try:
            tree = next(bllip.parse(sentence))
            print(tree)
        except StopIteration:
            print("(parse failed)")

    # n-best parsing demo
    for i, parse in enumerate(bllip.parse(sentence1)):
        print("parse %d:\n%s" % (i, parse))

    # using external POS tag constraints
    print(
        "forcing 'tree' to be 'NN':",
        next(bllip.tagged_parse([("A", None), ("tree", "NN")])),
    )
    print(
        "forcing 'A' to be 'DT' and 'tree' to be 'NNP':",
        next(bllip.tagged_parse([("A", "DT"), ("tree", "NNP")])),
    )
    # constraints don't have to make sense... (though on more complicated
    # sentences, they may cause the parse to fail)
    print(
        "forcing 'A' to be 'NNP':",
        next(bllip.tagged_parse([("A", "NNP"), ("tree", None)])),
    )
