# Natural Language Toolkit: An Incremental Earley Chart Parser
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Peter Ljunglöf <peter.ljunglof@heatherleaf.se>
#         Rob Speer <rspeer@mit.edu>
#         Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Jean Mark Gawron <gawron@mail.sdsu.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Data classes and parser implementations for *incremental* chart
parsers, which use dynamic programming to efficiently parse a text.
A "chart parser" derives parse trees for a text by iteratively adding
\"edges\" to a \"chart\".  Each "edge" represents a hypothesis about the tree
structure for a subsequence of the text.  The "chart" is a
\"blackboard\" for composing and combining these hypotheses.

A parser is "incremental", if it guarantees that for all i, j where i < j,
all edges ending at i are built before any edges ending at j.
This is appealing for, say, speech recognizer hypothesis filtering.

The main parser class is ``EarleyChartParser``, which is a top-down
algorithm, originally formulated by Jay Earley (1970).
"""

from time import perf_counter

from nltk.parse.chart import (
    BottomUpPredictCombineRule,
    BottomUpPredictRule,
    CachedTopDownPredictRule,
    Chart,
    ChartParser,
    EdgeI,
    EmptyPredictRule,
    FilteredBottomUpPredictCombineRule,
    FilteredSingleEdgeFundamentalRule,
    LeafEdge,
    LeafInitRule,
    SingleEdgeFundamentalRule,
    TopDownInitRule,
)
from nltk.parse.featurechart import (
    FeatureBottomUpPredictCombineRule,
    FeatureBottomUpPredictRule,
    FeatureChart,
    FeatureChartParser,
    FeatureEmptyPredictRule,
    FeatureSingleEdgeFundamentalRule,
    FeatureTopDownInitRule,
    FeatureTopDownPredictRule,
)

# ////////////////////////////////////////////////////////////
# Incremental Chart
# ////////////////////////////////////////////////////////////


class IncrementalChart(Chart):
    def initialize(self):
        # A sequence of edge lists contained in this chart.
        self._edgelists = tuple([] for x in self._positions())

        # The set of child pointer lists associated with each edge.
        self._edge_to_cpls = {}

        # Indexes mapping attribute values to lists of edges
        # (used by select()).
        self._indexes = {}

    def edges(self):
        return list(self.iteredges())

    def iteredges(self):
        return (edge for edgelist in self._edgelists for edge in edgelist)

    def select(self, end, **restrictions):
        edgelist = self._edgelists[end]

        # If there are no restrictions, then return all edges.
        if restrictions == {}:
            return iter(edgelist)

        # Find the index corresponding to the given restrictions.
        restr_keys = sorted(restrictions.keys())
        restr_keys = tuple(restr_keys)

        # If it doesn't exist, then create it.
        if restr_keys not in self._indexes:
            self._add_index(restr_keys)

        vals = tuple(restrictions[key] for key in restr_keys)
        return iter(self._indexes[restr_keys][end].get(vals, []))

    def _add_index(self, restr_keys):
        # Make sure it's a valid index.
        for key in restr_keys:
            if not hasattr(EdgeI, key):
                raise ValueError("Bad restriction: %s" % key)

        # Create the index.
        index = self._indexes[restr_keys] = tuple({} for x in self._positions())

        # Add all existing edges to the index.
        for end, edgelist in enumerate(self._edgelists):
            this_index = index[end]
            for edge in edgelist:
                vals = tuple(getattr(edge, key)() for key in restr_keys)
                this_index.setdefault(vals, []).append(edge)

    def _register_with_indexes(self, edge):
        end = edge.end()
        for (restr_keys, index) in self._indexes.items():
            vals = tuple(getattr(edge, key)() for key in restr_keys)
            index[end].setdefault(vals, []).append(edge)

    def _append_edge(self, edge):
        self._edgelists[edge.end()].append(edge)

    def _positions(self):
        return range(self.num_leaves() + 1)


class FeatureIncrementalChart(IncrementalChart, FeatureChart):
    def select(self, end, **restrictions):
        edgelist = self._edgelists[end]

        # If there are no restrictions, then return all edges.
        if restrictions == {}:
            return iter(edgelist)

        # Find the index corresponding to the given restrictions.
        restr_keys = sorted(restrictions.keys())
        restr_keys = tuple(restr_keys)

        # If it doesn't exist, then create it.
        if restr_keys not in self._indexes:
            self._add_index(restr_keys)

        vals = tuple(
            self._get_type_if_possible(restrictions[key]) for key in restr_keys
        )
        return iter(self._indexes[restr_keys][end].get(vals, []))

    def _add_index(self, restr_keys):
        # Make sure it's a valid index.
        for key in restr_keys:
            if not hasattr(EdgeI, key):
                raise ValueError("Bad restriction: %s" % key)

        # Create the index.
        index = self._indexes[restr_keys] = tuple({} for x in self._positions())

        # Add all existing edges to the index.
        for end, edgelist in enumerate(self._edgelists):
            this_index = index[end]
            for edge in edgelist:
                vals = tuple(
                    self._get_type_if_possible(getattr(edge, key)())
                    for key in restr_keys
                )
                this_index.setdefault(vals, []).append(edge)

    def _register_with_indexes(self, edge):
        end = edge.end()
        for (restr_keys, index) in self._indexes.items():
            vals = tuple(
                self._get_type_if_possible(getattr(edge, key)()) for key in restr_keys
            )
            index[end].setdefault(vals, []).append(edge)


# ////////////////////////////////////////////////////////////
# Incremental CFG Rules
# ////////////////////////////////////////////////////////////


class CompleteFundamentalRule(SingleEdgeFundamentalRule):
    def _apply_incomplete(self, chart, grammar, left_edge):
        end = left_edge.end()
        # When the chart is incremental, we only have to look for
        # empty complete edges here.
        for right_edge in chart.select(
            start=end, end=end, is_complete=True, lhs=left_edge.nextsym()
        ):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge


class CompleterRule(CompleteFundamentalRule):
    _fundamental_rule = CompleteFundamentalRule()

    def apply(self, chart, grammar, edge):
        if not isinstance(edge, LeafEdge):
            yield from self._fundamental_rule.apply(chart, grammar, edge)


class ScannerRule(CompleteFundamentalRule):
    _fundamental_rule = CompleteFundamentalRule()

    def apply(self, chart, grammar, edge):
        if isinstance(edge, LeafEdge):
            yield from self._fundamental_rule.apply(chart, grammar, edge)


class PredictorRule(CachedTopDownPredictRule):
    pass


class FilteredCompleteFundamentalRule(FilteredSingleEdgeFundamentalRule):
    def apply(self, chart, grammar, edge):
        # Since the Filtered rule only works for grammars without empty productions,
        # we only have to bother with complete edges here.
        if edge.is_complete():
            yield from self._apply_complete(chart, grammar, edge)


# ////////////////////////////////////////////////////////////
# Incremental FCFG Rules
# ////////////////////////////////////////////////////////////


class FeatureCompleteFundamentalRule(FeatureSingleEdgeFundamentalRule):
    def _apply_incomplete(self, chart, grammar, left_edge):
        fr = self._fundamental_rule
        end = left_edge.end()
        # When the chart is incremental, we only have to look for
        # empty complete edges here.
        for right_edge in chart.select(
            start=end, end=end, is_complete=True, lhs=left_edge.nextsym()
        ):
            yield from fr.apply(chart, grammar, left_edge, right_edge)


class FeatureCompleterRule(CompleterRule):
    _fundamental_rule = FeatureCompleteFundamentalRule()


class FeatureScannerRule(ScannerRule):
    _fundamental_rule = FeatureCompleteFundamentalRule()


class FeaturePredictorRule(FeatureTopDownPredictRule):
    pass


# ////////////////////////////////////////////////////////////
# Incremental CFG Chart Parsers
# ////////////////////////////////////////////////////////////

EARLEY_STRATEGY = [
    LeafInitRule(),
    TopDownInitRule(),
    CompleterRule(),
    ScannerRule(),
    PredictorRule(),
]
TD_INCREMENTAL_STRATEGY = [
    LeafInitRule(),
    TopDownInitRule(),
    CachedTopDownPredictRule(),
    CompleteFundamentalRule(),
]
BU_INCREMENTAL_STRATEGY = [
    LeafInitRule(),
    EmptyPredictRule(),
    BottomUpPredictRule(),
    CompleteFundamentalRule(),
]
BU_LC_INCREMENTAL_STRATEGY = [
    LeafInitRule(),
    EmptyPredictRule(),
    BottomUpPredictCombineRule(),
    CompleteFundamentalRule(),
]

LC_INCREMENTAL_STRATEGY = [
    LeafInitRule(),
    FilteredBottomUpPredictCombineRule(),
    FilteredCompleteFundamentalRule(),
]


class IncrementalChartParser(ChartParser):
    """
    An *incremental* chart parser implementing Jay Earley's
    parsing algorithm:

    | For each index end in [0, 1, ..., N]:
    |   For each edge such that edge.end = end:
    |     If edge is incomplete and edge.next is not a part of speech:
    |       Apply PredictorRule to edge
    |     If edge is incomplete and edge.next is a part of speech:
    |       Apply ScannerRule to edge
    |     If edge is complete:
    |       Apply CompleterRule to edge
    | Return any complete parses in the chart
    """

    def __init__(
        self,
        grammar,
        strategy=BU_LC_INCREMENTAL_STRATEGY,
        trace=0,
        trace_chart_width=50,
        chart_class=IncrementalChart,
    ):
        """
        Create a new Earley chart parser, that uses ``grammar`` to
        parse texts.

        :type grammar: CFG
        :param grammar: The grammar used to parse texts.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        :type trace_chart_width: int
        :param trace_chart_width: The default total width reserved for
            the chart in trace output.  The remainder of each line will
            be used to display edges.
        :param chart_class: The class that should be used to create
            the charts used by this parser.
        """
        self._grammar = grammar
        self._trace = trace
        self._trace_chart_width = trace_chart_width
        self._chart_class = chart_class

        self._axioms = []
        self._inference_rules = []
        for rule in strategy:
            if rule.NUM_EDGES == 0:
                self._axioms.append(rule)
            elif rule.NUM_EDGES == 1:
                self._inference_rules.append(rule)
            else:
                raise ValueError(
                    "Incremental inference rules must have " "NUM_EDGES == 0 or 1"
                )

    def chart_parse(self, tokens, trace=None):
        if trace is None:
            trace = self._trace
        trace_new_edges = self._trace_new_edges

        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        chart = self._chart_class(tokens)
        grammar = self._grammar

        # Width, for printing trace edges.
        trace_edge_width = self._trace_chart_width // (chart.num_leaves() + 1)
        if trace:
            print(chart.pretty_format_leaves(trace_edge_width))

        for axiom in self._axioms:
            new_edges = list(axiom.apply(chart, grammar))
            trace_new_edges(chart, axiom, new_edges, trace, trace_edge_width)

        inference_rules = self._inference_rules
        for end in range(chart.num_leaves() + 1):
            if trace > 1:
                print("\n* Processing queue:", end, "\n")
            agenda = list(chart.select(end=end))
            while agenda:
                edge = agenda.pop()
                for rule in inference_rules:
                    new_edges = list(rule.apply(chart, grammar, edge))
                    trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)
                    for new_edge in new_edges:
                        if new_edge.end() == end:
                            agenda.append(new_edge)

        return chart


class EarleyChartParser(IncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(self, grammar, EARLEY_STRATEGY, **parser_args)


class IncrementalTopDownChartParser(IncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(
            self, grammar, TD_INCREMENTAL_STRATEGY, **parser_args
        )


class IncrementalBottomUpChartParser(IncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(
            self, grammar, BU_INCREMENTAL_STRATEGY, **parser_args
        )


class IncrementalBottomUpLeftCornerChartParser(IncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(
            self, grammar, BU_LC_INCREMENTAL_STRATEGY, **parser_args
        )


class IncrementalLeftCornerChartParser(IncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        if not grammar.is_nonempty():
            raise ValueError(
                "IncrementalLeftCornerParser only works for grammars "
                "without empty productions."
            )
        IncrementalChartParser.__init__(
            self, grammar, LC_INCREMENTAL_STRATEGY, **parser_args
        )


# ////////////////////////////////////////////////////////////
# Incremental FCFG Chart Parsers
# ////////////////////////////////////////////////////////////

EARLEY_FEATURE_STRATEGY = [
    LeafInitRule(),
    FeatureTopDownInitRule(),
    FeatureCompleterRule(),
    FeatureScannerRule(),
    FeaturePredictorRule(),
]
TD_INCREMENTAL_FEATURE_STRATEGY = [
    LeafInitRule(),
    FeatureTopDownInitRule(),
    FeatureTopDownPredictRule(),
    FeatureCompleteFundamentalRule(),
]
BU_INCREMENTAL_FEATURE_STRATEGY = [
    LeafInitRule(),
    FeatureEmptyPredictRule(),
    FeatureBottomUpPredictRule(),
    FeatureCompleteFundamentalRule(),
]
BU_LC_INCREMENTAL_FEATURE_STRATEGY = [
    LeafInitRule(),
    FeatureEmptyPredictRule(),
    FeatureBottomUpPredictCombineRule(),
    FeatureCompleteFundamentalRule(),
]


class FeatureIncrementalChartParser(IncrementalChartParser, FeatureChartParser):
    def __init__(
        self,
        grammar,
        strategy=BU_LC_INCREMENTAL_FEATURE_STRATEGY,
        trace_chart_width=20,
        chart_class=FeatureIncrementalChart,
        **parser_args
    ):
        IncrementalChartParser.__init__(
            self,
            grammar,
            strategy=strategy,
            trace_chart_width=trace_chart_width,
            chart_class=chart_class,
            **parser_args
        )


class FeatureEarleyChartParser(FeatureIncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        FeatureIncrementalChartParser.__init__(
            self, grammar, EARLEY_FEATURE_STRATEGY, **parser_args
        )


class FeatureIncrementalTopDownChartParser(FeatureIncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        FeatureIncrementalChartParser.__init__(
            self, grammar, TD_INCREMENTAL_FEATURE_STRATEGY, **parser_args
        )


class FeatureIncrementalBottomUpChartParser(FeatureIncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        FeatureIncrementalChartParser.__init__(
            self, grammar, BU_INCREMENTAL_FEATURE_STRATEGY, **parser_args
        )


class FeatureIncrementalBottomUpLeftCornerChartParser(FeatureIncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        FeatureIncrementalChartParser.__init__(
            self, grammar, BU_LC_INCREMENTAL_FEATURE_STRATEGY, **parser_args
        )


# ////////////////////////////////////////////////////////////
# Demonstration
# ////////////////////////////////////////////////////////////


def demo(
    print_times=True,
    print_grammar=False,
    print_trees=True,
    trace=2,
    sent="I saw John with a dog with my cookie",
    numparses=5,
):
    """
    A demonstration of the Earley parsers.
    """
    import sys
    import time

    from nltk.parse.chart import demo_grammar

    # The grammar for ChartParser and SteppingChartParser:
    grammar = demo_grammar()
    if print_grammar:
        print("* Grammar")
        print(grammar)

    # Tokenize the sample sentence.
    print("* Sentence:")
    print(sent)
    tokens = sent.split()
    print(tokens)
    print()

    # Do the parsing.
    earley = EarleyChartParser(grammar, trace=trace)
    t = perf_counter()
    chart = earley.chart_parse(tokens)
    parses = list(chart.parses(grammar.start()))
    t = perf_counter() - t

    # Print results.
    if numparses:
        assert len(parses) == numparses, "Not all parses found"
    if print_trees:
        for tree in parses:
            print(tree)
    else:
        print("Nr trees:", len(parses))
    if print_times:
        print("Time:", t)


if __name__ == "__main__":
    demo()
