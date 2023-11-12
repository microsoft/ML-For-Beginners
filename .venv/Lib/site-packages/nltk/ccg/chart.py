# Natural Language Toolkit: Combinatory Categorial Grammar
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Graeme Gange <ggange@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
The lexicon is constructed by calling
``lexicon.fromstring(<lexicon string>)``.

In order to construct a parser, you also need a rule set.
The standard English rules are provided in chart as
``chart.DefaultRuleSet``.

The parser can then be constructed by calling, for example:
``parser = chart.CCGChartParser(<lexicon>, <ruleset>)``

Parsing is then performed by running
``parser.parse(<sentence>.split())``.

While this returns a list of trees, the default representation
of the produced trees is not very enlightening, particularly
given that it uses the same tree class as the CFG parsers.
It is probably better to call:
``chart.printCCGDerivation(<parse tree extracted from list>)``
which should print a nice representation of the derivation.

This entire process is shown far more clearly in the demonstration:
python chart.py
"""

import itertools

from nltk.ccg.combinator import *
from nltk.ccg.combinator import (
    BackwardApplication,
    BackwardBx,
    BackwardComposition,
    BackwardSx,
    BackwardT,
    ForwardApplication,
    ForwardComposition,
    ForwardSubstitution,
    ForwardT,
)
from nltk.ccg.lexicon import Token, fromstring
from nltk.ccg.logic import *
from nltk.parse import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.sem.logic import *
from nltk.tree import Tree


# Based on the EdgeI class from NLTK.
# A number of the properties of the EdgeI interface don't
# transfer well to CCGs, however.
class CCGEdge(EdgeI):
    def __init__(self, span, categ, rule):
        self._span = span
        self._categ = categ
        self._rule = rule
        self._comparison_key = (span, categ, rule)

    # Accessors
    def lhs(self):
        return self._categ

    def span(self):
        return self._span

    def start(self):
        return self._span[0]

    def end(self):
        return self._span[1]

    def length(self):
        return self._span[1] - self.span[0]

    def rhs(self):
        return ()

    def dot(self):
        return 0

    def is_complete(self):
        return True

    def is_incomplete(self):
        return False

    def nextsym(self):
        return None

    def categ(self):
        return self._categ

    def rule(self):
        return self._rule


class CCGLeafEdge(EdgeI):
    """
    Class representing leaf edges in a CCG derivation.
    """

    def __init__(self, pos, token, leaf):
        self._pos = pos
        self._token = token
        self._leaf = leaf
        self._comparison_key = (pos, token.categ(), leaf)

    # Accessors
    def lhs(self):
        return self._token.categ()

    def span(self):
        return (self._pos, self._pos + 1)

    def start(self):
        return self._pos

    def end(self):
        return self._pos + 1

    def length(self):
        return 1

    def rhs(self):
        return self._leaf

    def dot(self):
        return 0

    def is_complete(self):
        return True

    def is_incomplete(self):
        return False

    def nextsym(self):
        return None

    def token(self):
        return self._token

    def categ(self):
        return self._token.categ()

    def leaf(self):
        return self._leaf


class BinaryCombinatorRule(AbstractChartRule):
    """
    Class implementing application of a binary combinator to a chart.
    Takes the directed combinator to apply.
    """

    NUMEDGES = 2

    def __init__(self, combinator):
        self._combinator = combinator

    # Apply a combinator
    def apply(self, chart, grammar, left_edge, right_edge):
        # The left & right edges must be touching.
        if not (left_edge.end() == right_edge.start()):
            return

        # Check if the two edges are permitted to combine.
        # If so, generate the corresponding edge.
        if self._combinator.can_combine(left_edge.categ(), right_edge.categ()):
            for res in self._combinator.combine(left_edge.categ(), right_edge.categ()):
                new_edge = CCGEdge(
                    span=(left_edge.start(), right_edge.end()),
                    categ=res,
                    rule=self._combinator,
                )
                if chart.insert(new_edge, (left_edge, right_edge)):
                    yield new_edge

    # The representation of the combinator (for printing derivations)
    def __str__(self):
        return "%s" % self._combinator


# Type-raising must be handled slightly differently to the other rules, as the
# resulting rules only span a single edge, rather than both edges.


class ForwardTypeRaiseRule(AbstractChartRule):
    """
    Class for applying forward type raising
    """

    NUMEDGES = 2

    def __init__(self):
        self._combinator = ForwardT

    def apply(self, chart, grammar, left_edge, right_edge):
        if not (left_edge.end() == right_edge.start()):
            return

        for res in self._combinator.combine(left_edge.categ(), right_edge.categ()):
            new_edge = CCGEdge(span=left_edge.span(), categ=res, rule=self._combinator)
            if chart.insert(new_edge, (left_edge,)):
                yield new_edge

    def __str__(self):
        return "%s" % self._combinator


class BackwardTypeRaiseRule(AbstractChartRule):
    """
    Class for applying backward type raising.
    """

    NUMEDGES = 2

    def __init__(self):
        self._combinator = BackwardT

    def apply(self, chart, grammar, left_edge, right_edge):
        if not (left_edge.end() == right_edge.start()):
            return

        for res in self._combinator.combine(left_edge.categ(), right_edge.categ()):
            new_edge = CCGEdge(span=right_edge.span(), categ=res, rule=self._combinator)
            if chart.insert(new_edge, (right_edge,)):
                yield new_edge

    def __str__(self):
        return "%s" % self._combinator


# Common sets of combinators used for English derivations.
ApplicationRuleSet = [
    BinaryCombinatorRule(ForwardApplication),
    BinaryCombinatorRule(BackwardApplication),
]
CompositionRuleSet = [
    BinaryCombinatorRule(ForwardComposition),
    BinaryCombinatorRule(BackwardComposition),
    BinaryCombinatorRule(BackwardBx),
]
SubstitutionRuleSet = [
    BinaryCombinatorRule(ForwardSubstitution),
    BinaryCombinatorRule(BackwardSx),
]
TypeRaiseRuleSet = [ForwardTypeRaiseRule(), BackwardTypeRaiseRule()]

# The standard English rule set.
DefaultRuleSet = (
    ApplicationRuleSet + CompositionRuleSet + SubstitutionRuleSet + TypeRaiseRuleSet
)


class CCGChartParser(ParserI):
    """
    Chart parser for CCGs.
    Based largely on the ChartParser class from NLTK.
    """

    def __init__(self, lexicon, rules, trace=0):
        self._lexicon = lexicon
        self._rules = rules
        self._trace = trace

    def lexicon(self):
        return self._lexicon

    # Implements the CYK algorithm
    def parse(self, tokens):
        tokens = list(tokens)
        chart = CCGChart(list(tokens))
        lex = self._lexicon

        # Initialize leaf edges.
        for index in range(chart.num_leaves()):
            for token in lex.categories(chart.leaf(index)):
                new_edge = CCGLeafEdge(index, token, chart.leaf(index))
                chart.insert(new_edge, ())

        # Select a span for the new edges
        for span in range(2, chart.num_leaves() + 1):
            for start in range(0, chart.num_leaves() - span + 1):
                # Try all possible pairs of edges that could generate
                # an edge for that span
                for part in range(1, span):
                    lstart = start
                    mid = start + part
                    rend = start + span

                    for left in chart.select(span=(lstart, mid)):
                        for right in chart.select(span=(mid, rend)):
                            # Generate all possible combinations of the two edges
                            for rule in self._rules:
                                edges_added_by_rule = 0
                                for newedge in rule.apply(chart, lex, left, right):
                                    edges_added_by_rule += 1

        # Output the resulting parses
        return chart.parses(lex.start())


class CCGChart(Chart):
    def __init__(self, tokens):
        Chart.__init__(self, tokens)

    # Constructs the trees for a given parse. Unfortnunately, the parse trees need to be
    # constructed slightly differently to those in the default Chart class, so it has to
    # be reimplemented
    def _trees(self, edge, complete, memo, tree_class):
        assert complete, "CCGChart cannot build incomplete trees"

        if edge in memo:
            return memo[edge]

        if isinstance(edge, CCGLeafEdge):
            word = tree_class(edge.token(), [self._tokens[edge.start()]])
            leaf = tree_class((edge.token(), "Leaf"), [word])
            memo[edge] = [leaf]
            return [leaf]

        memo[edge] = []
        trees = []

        for cpl in self.child_pointer_lists(edge):
            child_choices = [self._trees(cp, complete, memo, tree_class) for cp in cpl]
            for children in itertools.product(*child_choices):
                lhs = (
                    Token(
                        self._tokens[edge.start() : edge.end()],
                        edge.lhs(),
                        compute_semantics(children, edge),
                    ),
                    str(edge.rule()),
                )
                trees.append(tree_class(lhs, children))

        memo[edge] = trees
        return trees


def compute_semantics(children, edge):
    if children[0].label()[0].semantics() is None:
        return None

    if len(children) == 2:
        if isinstance(edge.rule(), BackwardCombinator):
            children = [children[1], children[0]]

        combinator = edge.rule()._combinator
        function = children[0].label()[0].semantics()
        argument = children[1].label()[0].semantics()

        if isinstance(combinator, UndirectedFunctionApplication):
            return compute_function_semantics(function, argument)
        elif isinstance(combinator, UndirectedComposition):
            return compute_composition_semantics(function, argument)
        elif isinstance(combinator, UndirectedSubstitution):
            return compute_substitution_semantics(function, argument)
        else:
            raise AssertionError("Unsupported combinator '" + combinator + "'")
    else:
        return compute_type_raised_semantics(children[0].label()[0].semantics())


# --------
# Displaying derivations
# --------
def printCCGDerivation(tree):
    # Get the leaves and initial categories
    leafcats = tree.pos()
    leafstr = ""
    catstr = ""

    # Construct a string with both the leaf word and corresponding
    # category aligned.
    for (leaf, cat) in leafcats:
        str_cat = "%s" % cat
        nextlen = 2 + max(len(leaf), len(str_cat))
        lcatlen = (nextlen - len(str_cat)) // 2
        rcatlen = lcatlen + (nextlen - len(str_cat)) % 2
        catstr += " " * lcatlen + str_cat + " " * rcatlen
        lleaflen = (nextlen - len(leaf)) // 2
        rleaflen = lleaflen + (nextlen - len(leaf)) % 2
        leafstr += " " * lleaflen + leaf + " " * rleaflen
    print(leafstr.rstrip())
    print(catstr.rstrip())

    # Display the derivation steps
    printCCGTree(0, tree)


# Prints the sequence of derivation steps.
def printCCGTree(lwidth, tree):
    rwidth = lwidth

    # Is a leaf (word).
    # Increment the span by the space occupied by the leaf.
    if not isinstance(tree, Tree):
        return 2 + lwidth + len(tree)

    # Find the width of the current derivation step
    for child in tree:
        rwidth = max(rwidth, printCCGTree(rwidth, child))

    # Is a leaf node.
    # Don't print anything, but account for the space occupied.
    if not isinstance(tree.label(), tuple):
        return max(
            rwidth, 2 + lwidth + len("%s" % tree.label()), 2 + lwidth + len(tree[0])
        )

    (token, op) = tree.label()

    if op == "Leaf":
        return rwidth

    # Pad to the left with spaces, followed by a sequence of '-'
    # and the derivation rule.
    print(lwidth * " " + (rwidth - lwidth) * "-" + "%s" % op)
    # Print the resulting category on a new line.
    str_res = "%s" % (token.categ())
    if token.semantics() is not None:
        str_res += " {" + str(token.semantics()) + "}"
    respadlen = (rwidth - lwidth - len(str_res)) // 2 + lwidth
    print(respadlen * " " + str_res)
    return rwidth


### Demonstration code

# Construct the lexicon
lex = fromstring(
    """
    :- S, NP, N, VP    # Primitive categories, S is the target primitive

    Det :: NP/N         # Family of words
    Pro :: NP
    TV :: VP/NP
    Modal :: (S\\NP)/VP # Backslashes need to be escaped

    I => Pro             # Word -> Category mapping
    you => Pro

    the => Det

    # Variables have the special keyword 'var'
    # '.' prevents permutation
    # ',' prevents composition
    and => var\\.,var/.,var

    which => (N\\N)/(S/NP)

    will => Modal # Categories can be either explicit, or families.
    might => Modal

    cook => TV
    eat => TV

    mushrooms => N
    parsnips => N
    bacon => N
    """
)


def demo():
    parser = CCGChartParser(lex, DefaultRuleSet)
    for parse in parser.parse("I might cook and eat the bacon".split()):
        printCCGDerivation(parse)


if __name__ == "__main__":
    demo()
