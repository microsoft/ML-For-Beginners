# Natural Language Toolkit: Probabilistic Chart Parsers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Classes and interfaces for associating probabilities with tree
structures that represent the internal organization of a text.  The
probabilistic parser module defines ``BottomUpProbabilisticChartParser``.

``BottomUpProbabilisticChartParser`` is an abstract class that implements
a bottom-up chart parser for ``PCFG`` grammars.  It maintains a queue of edges,
and adds them to the chart one at a time.  The ordering of this queue
is based on the probabilities associated with the edges, allowing the
parser to expand more likely edges before less likely ones.  Each
subclass implements a different queue ordering, producing different
search strategies.  Currently the following subclasses are defined:

  - ``InsideChartParser`` searches edges in decreasing order of
    their trees' inside probabilities.
  - ``RandomChartParser`` searches edges in random order.
  - ``LongestChartParser`` searches edges in decreasing order of their
    location's length.

The ``BottomUpProbabilisticChartParser`` constructor has an optional
argument beam_size.  If non-zero, this controls the size of the beam
(aka the edge queue).  This option is most useful with InsideChartParser.
"""

##//////////////////////////////////////////////////////
##  Bottom-Up PCFG Chart Parser
##//////////////////////////////////////////////////////

# [XX] This might not be implemented quite right -- it would be better
# to associate probabilities with child pointer lists.

import random
from functools import reduce

from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree


# Probabilistic edges
class ProbabilisticLeafEdge(LeafEdge):
    def prob(self):
        return 1.0


class ProbabilisticTreeEdge(TreeEdge):
    def __init__(self, prob, *args, **kwargs):
        TreeEdge.__init__(self, *args, **kwargs)
        self._prob = prob
        # two edges with different probabilities are not equal.
        self._comparison_key = (self._comparison_key, prob)

    def prob(self):
        return self._prob

    @staticmethod
    def from_production(production, index, p):
        return ProbabilisticTreeEdge(
            p, (index, index), production.lhs(), production.rhs(), 0
        )


# Rules using probabilistic edges
class ProbabilisticBottomUpInitRule(AbstractChartRule):
    NUM_EDGES = 0

    def apply(self, chart, grammar):
        for index in range(chart.num_leaves()):
            new_edge = ProbabilisticLeafEdge(chart.leaf(index), index)
            if chart.insert(new_edge, ()):
                yield new_edge


class ProbabilisticBottomUpPredictRule(AbstractChartRule):
    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return
        for prod in grammar.productions():
            if edge.lhs() == prod.rhs()[0]:
                new_edge = ProbabilisticTreeEdge.from_production(
                    prod, edge.start(), prod.prob()
                )
                if chart.insert(new_edge, ()):
                    yield new_edge


class ProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES = 2

    def apply(self, chart, grammar, left_edge, right_edge):
        # Make sure the rule is applicable.
        if not (
            left_edge.end() == right_edge.start()
            and left_edge.nextsym() == right_edge.lhs()
            and left_edge.is_incomplete()
            and right_edge.is_complete()
        ):
            return

        # Construct the new edge.
        p = left_edge.prob() * right_edge.prob()
        new_edge = ProbabilisticTreeEdge(
            p,
            span=(left_edge.start(), right_edge.end()),
            lhs=left_edge.lhs(),
            rhs=left_edge.rhs(),
            dot=left_edge.dot() + 1,
        )

        # Add it to the chart, with appropriate child pointers.
        changed_chart = False
        for cpl1 in chart.child_pointer_lists(left_edge):
            if chart.insert(new_edge, cpl1 + (right_edge,)):
                changed_chart = True

        # If we changed the chart, then generate the edge.
        if changed_chart:
            yield new_edge


class SingleEdgeProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES = 1

    _fundamental_rule = ProbabilisticFundamentalRule()

    def apply(self, chart, grammar, edge1):
        fr = self._fundamental_rule
        if edge1.is_incomplete():
            # edge1 = left_edge; edge2 = right_edge
            for edge2 in chart.select(
                start=edge1.end(), is_complete=True, lhs=edge1.nextsym()
            ):
                yield from fr.apply(chart, grammar, edge1, edge2)
        else:
            # edge2 = left_edge; edge1 = right_edge
            for edge2 in chart.select(
                end=edge1.start(), is_complete=False, nextsym=edge1.lhs()
            ):
                yield from fr.apply(chart, grammar, edge2, edge1)

    def __str__(self):
        return "Fundamental Rule"


class BottomUpProbabilisticChartParser(ParserI):
    """
    An abstract bottom-up parser for ``PCFG`` grammars that uses a ``Chart`` to
    record partial results.  ``BottomUpProbabilisticChartParser`` maintains
    a queue of edges that can be added to the chart.  This queue is
    initialized with edges for each token in the text that is being
    parsed.  ``BottomUpProbabilisticChartParser`` inserts these edges into
    the chart one at a time, starting with the most likely edges, and
    proceeding to less likely edges.  For each edge that is added to
    the chart, it may become possible to insert additional edges into
    the chart; these are added to the queue.  This process continues
    until enough complete parses have been generated, or until the
    queue is empty.

    The sorting order for the queue is not specified by
    ``BottomUpProbabilisticChartParser``.  Different sorting orders will
    result in different search strategies.  The sorting order for the
    queue is defined by the method ``sort_queue``; subclasses are required
    to provide a definition for this method.

    :type _grammar: PCFG
    :ivar _grammar: The grammar used to parse sentences.
    :type _trace: int
    :ivar _trace: The level of tracing output that should be generated
        when parsing a text.
    """

    def __init__(self, grammar, beam_size=0, trace=0):
        """
        Create a new ``BottomUpProbabilisticChartParser``, that uses
        ``grammar`` to parse texts.

        :type grammar: PCFG
        :param grammar: The grammar used to parse texts.
        :type beam_size: int
        :param beam_size: The maximum length for the parser's edge queue.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        """
        if not isinstance(grammar, PCFG):
            raise ValueError("The grammar must be probabilistic PCFG")
        self._grammar = grammar
        self.beam_size = beam_size
        self._trace = trace

    def grammar(self):
        return self._grammar

    def trace(self, trace=2):
        """
        Set the level of tracing output that should be generated when
        parsing a text.

        :type trace: int
        :param trace: The trace level.  A trace level of ``0`` will
            generate no tracing output; and higher trace levels will
            produce more verbose tracing output.
        :rtype: None
        """
        self._trace = trace

    # TODO: change this to conform more with the standard ChartParser
    def parse(self, tokens):
        self._grammar.check_coverage(tokens)
        chart = Chart(list(tokens))
        grammar = self._grammar

        # Chart parser rules.
        bu_init = ProbabilisticBottomUpInitRule()
        bu = ProbabilisticBottomUpPredictRule()
        fr = SingleEdgeProbabilisticFundamentalRule()

        # Our queue
        queue = []

        # Initialize the chart.
        for edge in bu_init.apply(chart, grammar):
            if self._trace > 1:
                print(
                    "  %-50s [%s]"
                    % (chart.pretty_format_edge(edge, width=2), edge.prob())
                )
            queue.append(edge)

        while len(queue) > 0:
            # Re-sort the queue.
            self.sort_queue(queue, chart)

            # Prune the queue to the correct size if a beam was defined
            if self.beam_size:
                self._prune(queue, chart)

            # Get the best edge.
            edge = queue.pop()
            if self._trace > 0:
                print(
                    "  %-50s [%s]"
                    % (chart.pretty_format_edge(edge, width=2), edge.prob())
                )

            # Apply BU & FR to it.
            queue.extend(bu.apply(chart, grammar, edge))
            queue.extend(fr.apply(chart, grammar, edge))

        # Get a list of complete parses.
        parses = list(chart.parses(grammar.start(), ProbabilisticTree))

        # Assign probabilities to the trees.
        prod_probs = {}
        for prod in grammar.productions():
            prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
        for parse in parses:
            self._setprob(parse, prod_probs)

        # Sort by probability
        parses.sort(reverse=True, key=lambda tree: tree.prob())

        return iter(parses)

    def _setprob(self, tree, prod_probs):
        if tree.prob() is not None:
            return

        # Get the prob of the CFG production.
        lhs = Nonterminal(tree.label())
        rhs = []
        for child in tree:
            if isinstance(child, Tree):
                rhs.append(Nonterminal(child.label()))
            else:
                rhs.append(child)
        prob = prod_probs[lhs, tuple(rhs)]

        # Get the probs of children.
        for child in tree:
            if isinstance(child, Tree):
                self._setprob(child, prod_probs)
                prob *= child.prob()

        tree.set_prob(prob)

    def sort_queue(self, queue, chart):
        """
        Sort the given queue of ``Edge`` objects, placing the edge that should
        be tried first at the beginning of the queue.  This method
        will be called after each ``Edge`` is added to the queue.

        :param queue: The queue of ``Edge`` objects to sort.  Each edge in
            this queue is an edge that could be added to the chart by
            the fundamental rule; but that has not yet been added.
        :type queue: list(Edge)
        :param chart: The chart being used to parse the text.  This
            chart can be used to provide extra information for sorting
            the queue.
        :type chart: Chart
        :rtype: None
        """
        raise NotImplementedError()

    def _prune(self, queue, chart):
        """Discard items in the queue if the queue is longer than the beam."""
        if len(queue) > self.beam_size:
            split = len(queue) - self.beam_size
            if self._trace > 2:
                for edge in queue[:split]:
                    print("  %-50s [DISCARDED]" % chart.pretty_format_edge(edge, 2))
            del queue[:split]


class InsideChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in descending
    order of the inside probabilities of their trees.  The "inside
    probability" of a tree is simply the
    probability of the entire tree, ignoring its context.  In
    particular, the inside probability of a tree generated by
    production *p* with children *c[1], c[2], ..., c[n]* is
    *P(p)P(c[1])P(c[2])...P(c[n])*; and the inside
    probability of a token is 1 if it is present in the text, and 0 if
    it is absent.

    This sorting order results in a type of lowest-cost-first search
    strategy.
    """

    # Inherit constructor.
    def sort_queue(self, queue, chart):
        """
        Sort the given queue of edges, in descending order of the
        inside probabilities of the edges' trees.

        :param queue: The queue of ``Edge`` objects to sort.  Each edge in
            this queue is an edge that could be added to the chart by
            the fundamental rule; but that has not yet been added.
        :type queue: list(Edge)
        :param chart: The chart being used to parse the text.  This
            chart can be used to provide extra information for sorting
            the queue.
        :type chart: Chart
        :rtype: None
        """
        queue.sort(key=lambda edge: edge.prob())


# Eventually, this will become some sort of inside-outside parser:
# class InsideOutsideParser(BottomUpProbabilisticChartParser):
#     def __init__(self, grammar, trace=0):
#         # Inherit docs.
#         BottomUpProbabilisticChartParser.__init__(self, grammar, trace)
#
#         # Find the best path from S to each nonterminal
#         bestp = {}
#         for production in grammar.productions(): bestp[production.lhs()]=0
#         bestp[grammar.start()] = 1.0
#
#         for i in range(len(grammar.productions())):
#             for production in grammar.productions():
#                 lhs = production.lhs()
#                 for elt in production.rhs():
#                     bestp[elt] = max(bestp[lhs]*production.prob(),
#                                      bestp.get(elt,0))
#
#         self._bestp = bestp
#         for (k,v) in self._bestp.items(): print(k,v)
#
#     def _sortkey(self, edge):
#         return edge.structure()[PROB] * self._bestp[edge.lhs()]
#
#     def sort_queue(self, queue, chart):
#         queue.sort(key=self._sortkey)


class RandomChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in random order.
    This sorting order results in a random search strategy.
    """

    # Inherit constructor
    def sort_queue(self, queue, chart):
        i = random.randint(0, len(queue) - 1)
        (queue[-1], queue[i]) = (queue[i], queue[-1])


class UnsortedChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in whatever order.
    """

    # Inherit constructor
    def sort_queue(self, queue, chart):
        return


class LongestChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries longer edges before
    shorter ones.  This sorting order results in a type of best-first
    search strategy.
    """

    # Inherit constructor
    def sort_queue(self, queue, chart):
        queue.sort(key=lambda edge: edge.length())


##//////////////////////////////////////////////////////
##  Test Code
##//////////////////////////////////////////////////////


def demo(choice=None, draw_parses=None, print_parses=None):
    """
    A demonstration of the probabilistic parsers.  The user is
    prompted to select which demo to run, and how many parses should
    be found; and then each parser is run on the same demo, and a
    summary of the results are displayed.
    """
    import sys
    import time

    from nltk import tokenize
    from nltk.parse import pchart

    # Define two demos.  Each demo has a sentence and a grammar.
    toy_pcfg1 = PCFG.fromstring(
        """
    S -> NP VP [1.0]
    NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
    Det -> 'the' [0.8] | 'my' [0.2]
    N -> 'man' [0.5] | 'telescope' [0.5]
    VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
    V -> 'ate' [0.35] | 'saw' [0.65]
    PP -> P NP [1.0]
    P -> 'with' [0.61] | 'under' [0.39]
    """
    )

    toy_pcfg2 = PCFG.fromstring(
        """
    S    -> NP VP         [1.0]
    VP   -> V NP          [.59]
    VP   -> V             [.40]
    VP   -> VP PP         [.01]
    NP   -> Det N         [.41]
    NP   -> Name          [.28]
    NP   -> NP PP         [.31]
    PP   -> P NP          [1.0]
    V    -> 'saw'         [.21]
    V    -> 'ate'         [.51]
    V    -> 'ran'         [.28]
    N    -> 'boy'         [.11]
    N    -> 'cookie'      [.12]
    N    -> 'table'       [.13]
    N    -> 'telescope'   [.14]
    N    -> 'hill'        [.5]
    Name -> 'Jack'        [.52]
    Name -> 'Bob'         [.48]
    P    -> 'with'        [.61]
    P    -> 'under'       [.39]
    Det  -> 'the'         [.41]
    Det  -> 'a'           [.31]
    Det  -> 'my'          [.28]
    """
    )

    demos = [
        ("I saw John with my telescope", toy_pcfg1),
        ("the boy saw Jack with Bob under the table with a telescope", toy_pcfg2),
    ]

    if choice is None:
        # Ask the user which demo they want to use.
        print()
        for i in range(len(demos)):
            print(f"{i + 1:>3}: {demos[i][0]}")
            print("     %r" % demos[i][1])
            print()
        print("Which demo (%d-%d)? " % (1, len(demos)), end=" ")
        choice = int(sys.stdin.readline().strip()) - 1
    try:
        sent, grammar = demos[choice]
    except:
        print("Bad sentence number")
        return

    # Tokenize the sentence.
    tokens = sent.split()

    # Define a list of parsers.  We'll use all parsers.
    parsers = [
        pchart.InsideChartParser(grammar),
        pchart.RandomChartParser(grammar),
        pchart.UnsortedChartParser(grammar),
        pchart.LongestChartParser(grammar),
        pchart.InsideChartParser(grammar, beam_size=len(tokens) + 1),  # was BeamParser
    ]

    # Run the parsers on the tokenized sentence.
    times = []
    average_p = []
    num_parses = []
    all_parses = {}
    for parser in parsers:
        print(f"\ns: {sent}\nparser: {parser}\ngrammar: {grammar}")
        parser.trace(3)
        t = time.time()
        parses = list(parser.parse(tokens))
        times.append(time.time() - t)
        p = reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses) if parses else 0
        average_p.append(p)
        num_parses.append(len(parses))
        for p in parses:
            all_parses[p.freeze()] = 1

    # Print some summary statistics
    print()
    print("       Parser      Beam | Time (secs)   # Parses   Average P(parse)")
    print("------------------------+------------------------------------------")
    for i in range(len(parsers)):
        print(
            "%18s %4d |%11.4f%11d%19.14f"
            % (
                parsers[i].__class__.__name__,
                parsers[i].beam_size,
                times[i],
                num_parses[i],
                average_p[i],
            )
        )
    parses = all_parses.keys()
    if parses:
        p = reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses)
    else:
        p = 0
    print("------------------------+------------------------------------------")
    print("%18s      |%11s%11d%19.14f" % ("(All Parses)", "n/a", len(parses), p))

    if draw_parses is None:
        # Ask the user if we should draw the parses.
        print()
        print("Draw parses (y/n)? ", end=" ")
        draw_parses = sys.stdin.readline().strip().lower().startswith("y")
    if draw_parses:
        from nltk.draw.tree import draw_trees

        print("  please wait...")
        draw_trees(*parses)

    if print_parses is None:
        # Ask the user if we should print the parses.
        print()
        print("Print parses (y/n)? ", end=" ")
        print_parses = sys.stdin.readline().strip().lower().startswith("y")
    if print_parses:
        for parse in parses:
            print(parse)


if __name__ == "__main__":
    demo()
