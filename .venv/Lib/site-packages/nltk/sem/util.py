# Natural Language Toolkit: Semantic Interpretation
#
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Utility functions for batch-processing sentences: parsing and
extraction of the semantic representation of the root node of the the
syntax tree, followed by evaluation of the semantic representation in
a first-order model.
"""

import codecs

from nltk.sem import evaluate

##############################################################
## Utility functions for connecting parse output to semantics
##############################################################


def parse_sents(inputs, grammar, trace=0):
    """
    Convert input sentences into syntactic trees.

    :param inputs: sentences to be parsed
    :type inputs: list(str)
    :param grammar: ``FeatureGrammar`` or name of feature-based grammar
    :type grammar: nltk.grammar.FeatureGrammar
    :rtype: list(nltk.tree.Tree) or dict(list(str)): list(Tree)
    :return: a mapping from input sentences to a list of ``Tree`` instances.
    """
    # put imports here to avoid circult dependencies
    from nltk.grammar import FeatureGrammar
    from nltk.parse import FeatureChartParser, load_parser

    if isinstance(grammar, FeatureGrammar):
        cp = FeatureChartParser(grammar)
    else:
        cp = load_parser(grammar, trace=trace)
    parses = []
    for sent in inputs:
        tokens = sent.split()  # use a tokenizer?
        syntrees = list(cp.parse(tokens))
        parses.append(syntrees)
    return parses


def root_semrep(syntree, semkey="SEM"):
    """
    Find the semantic representation at the root of a tree.

    :param syntree: a parse ``Tree``
    :param semkey: the feature label to use for the root semantics in the tree
    :return: the semantic representation at the root of a ``Tree``
    :rtype: sem.Expression
    """
    from nltk.grammar import FeatStructNonterminal

    node = syntree.label()
    assert isinstance(node, FeatStructNonterminal)
    try:
        return node[semkey]
    except KeyError:
        print(node, end=" ")
        print("has no specification for the feature %s" % semkey)
    raise


def interpret_sents(inputs, grammar, semkey="SEM", trace=0):
    """
    Add the semantic representation to each syntactic parse tree
    of each input sentence.

    :param inputs: a list of sentences
    :type inputs: list(str)
    :param grammar: ``FeatureGrammar`` or name of feature-based grammar
    :type grammar: nltk.grammar.FeatureGrammar
    :return: a mapping from sentences to lists of pairs (parse-tree, semantic-representations)
    :rtype: list(list(tuple(nltk.tree.Tree, nltk.sem.logic.ConstantExpression)))
    """
    return [
        [(syn, root_semrep(syn, semkey)) for syn in syntrees]
        for syntrees in parse_sents(inputs, grammar, trace=trace)
    ]


def evaluate_sents(inputs, grammar, model, assignment, trace=0):
    """
    Add the truth-in-a-model value to each semantic representation
    for each syntactic parse of each input sentences.

    :param inputs: a list of sentences
    :type inputs: list(str)
    :param grammar: ``FeatureGrammar`` or name of feature-based grammar
    :type grammar: nltk.grammar.FeatureGrammar
    :return: a mapping from sentences to lists of triples (parse-tree, semantic-representations, evaluation-in-model)
    :rtype: list(list(tuple(nltk.tree.Tree, nltk.sem.logic.ConstantExpression, bool or dict(str): bool)))
    """
    return [
        [
            (syn, sem, model.evaluate("%s" % sem, assignment, trace=trace))
            for (syn, sem) in interpretations
        ]
        for interpretations in interpret_sents(inputs, grammar)
    ]


def demo_model0():
    global m0, g0
    # Initialize a valuation of non-logical constants."""
    v = [
        ("john", "b1"),
        ("mary", "g1"),
        ("suzie", "g2"),
        ("fido", "d1"),
        ("tess", "d2"),
        ("noosa", "n"),
        ("girl", {"g1", "g2"}),
        ("boy", {"b1", "b2"}),
        ("dog", {"d1", "d2"}),
        ("bark", {"d1", "d2"}),
        ("walk", {"b1", "g2", "d1"}),
        ("chase", {("b1", "g1"), ("b2", "g1"), ("g1", "d1"), ("g2", "d2")}),
        (
            "see",
            {("b1", "g1"), ("b2", "d2"), ("g1", "b1"), ("d2", "b1"), ("g2", "n")},
        ),
        ("in", {("b1", "n"), ("b2", "n"), ("d2", "n")}),
        ("with", {("b1", "g1"), ("g1", "b1"), ("d1", "b1"), ("b1", "d1")}),
    ]
    # Read in the data from ``v``
    val = evaluate.Valuation(v)
    # Bind ``dom`` to the ``domain`` property of ``val``
    dom = val.domain
    # Initialize a model with parameters ``dom`` and ``val``.
    m0 = evaluate.Model(dom, val)
    # Initialize a variable assignment with parameter ``dom``
    g0 = evaluate.Assignment(dom)


def read_sents(filename, encoding="utf8"):
    with codecs.open(filename, "r", encoding) as fp:
        sents = [l.rstrip() for l in fp]

    # get rid of blank lines
    sents = [l for l in sents if len(l) > 0]
    sents = [l for l in sents if not l[0] == "#"]
    return sents


def demo_legacy_grammar():
    """
    Check that interpret_sents() is compatible with legacy grammars that use
    a lowercase 'sem' feature.

    Define 'test.fcfg' to be the following

    """
    from nltk.grammar import FeatureGrammar

    g = FeatureGrammar.fromstring(
        """
    % start S
    S[sem=<hello>] -> 'hello'
    """
    )
    print("Reading grammar: %s" % g)
    print("*" * 20)
    for reading in interpret_sents(["hello"], g, semkey="sem"):
        syn, sem = reading[0]
        print()
        print("output: ", sem)


def demo():
    import sys
    from optparse import OptionParser

    description = """
    Parse and evaluate some sentences.
    """

    opts = OptionParser(description=description)

    opts.set_defaults(
        evaluate=True,
        beta=True,
        syntrace=0,
        semtrace=0,
        demo="default",
        grammar="",
        sentences="",
    )

    opts.add_option(
        "-d",
        "--demo",
        dest="demo",
        help="choose demo D; omit this for the default demo, or specify 'chat80'",
        metavar="D",
    )
    opts.add_option(
        "-g", "--gram", dest="grammar", help="read in grammar G", metavar="G"
    )
    opts.add_option(
        "-m",
        "--model",
        dest="model",
        help="import model M (omit '.py' suffix)",
        metavar="M",
    )
    opts.add_option(
        "-s",
        "--sentences",
        dest="sentences",
        help="read in a file of test sentences S",
        metavar="S",
    )
    opts.add_option(
        "-e",
        "--no-eval",
        action="store_false",
        dest="evaluate",
        help="just do a syntactic analysis",
    )
    opts.add_option(
        "-b",
        "--no-beta-reduction",
        action="store_false",
        dest="beta",
        help="don't carry out beta-reduction",
    )
    opts.add_option(
        "-t",
        "--syntrace",
        action="count",
        dest="syntrace",
        help="set syntactic tracing on; requires '-e' option",
    )
    opts.add_option(
        "-T",
        "--semtrace",
        action="count",
        dest="semtrace",
        help="set semantic tracing on",
    )

    (options, args) = opts.parse_args()

    SPACER = "-" * 30

    demo_model0()

    sents = [
        "Fido sees a boy with Mary",
        "John sees Mary",
        "every girl chases a dog",
        "every boy chases a girl",
        "John walks with a girl in Noosa",
        "who walks",
    ]

    gramfile = "grammars/sample_grammars/sem2.fcfg"

    if options.sentences:
        sentsfile = options.sentences
    if options.grammar:
        gramfile = options.grammar
    if options.model:
        exec("import %s as model" % options.model)

    if sents is None:
        sents = read_sents(sentsfile)

    # Set model and assignment
    model = m0
    g = g0

    if options.evaluate:
        evaluations = evaluate_sents(sents, gramfile, model, g, trace=options.semtrace)
    else:
        semreps = interpret_sents(sents, gramfile, trace=options.syntrace)

    for i, sent in enumerate(sents):
        n = 1
        print("\nSentence: %s" % sent)
        print(SPACER)
        if options.evaluate:

            for (syntree, semrep, value) in evaluations[i]:
                if isinstance(value, dict):
                    value = set(value.keys())
                print("%d:  %s" % (n, semrep))
                print(value)
                n += 1
        else:

            for (syntree, semrep) in semreps[i]:
                print("%d:  %s" % (n, semrep))
                n += 1


if __name__ == "__main__":
    demo()
    demo_legacy_grammar()
