# Natural Language Toolkit: Dependency Grammars
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Jason Narad <jason.narad@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

from collections import defaultdict
from functools import total_ordering
from itertools import chain

from nltk.grammar import (
    DependencyGrammar,
    DependencyProduction,
    ProbabilisticDependencyGrammar,
)
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph

#################################################################
# Dependency Span
#################################################################


@total_ordering
class DependencySpan:
    """
    A contiguous span over some part of the input string representing
    dependency (head -> modifier) relationships amongst words.  An atomic
    span corresponds to only one word so it isn't a 'span' in the conventional
    sense, as its _start_index = _end_index = _head_index for concatenation
    purposes.  All other spans are assumed to have arcs between all nodes
    within the start and end indexes of the span, and one head index corresponding
    to the head word for the entire span.  This is the same as the root node if
    the dependency structure were depicted as a graph.
    """

    def __init__(self, start_index, end_index, head_index, arcs, tags):
        self._start_index = start_index
        self._end_index = end_index
        self._head_index = head_index
        self._arcs = arcs
        self._tags = tags
        self._comparison_key = (start_index, end_index, head_index, tuple(arcs))
        self._hash = hash(self._comparison_key)

    def head_index(self):
        """
        :return: An value indexing the head of the entire ``DependencySpan``.
        :rtype: int
        """
        return self._head_index

    def __repr__(self):
        """
        :return: A concise string representatino of the ``DependencySpan``.
        :rtype: str.
        """
        return "Span %d-%d; Head Index: %d" % (
            self._start_index,
            self._end_index,
            self._head_index,
        )

    def __str__(self):
        """
        :return: A verbose string representation of the ``DependencySpan``.
        :rtype: str
        """
        str = "Span %d-%d; Head Index: %d" % (
            self._start_index,
            self._end_index,
            self._head_index,
        )
        for i in range(len(self._arcs)):
            str += "\n%d <- %d, %s" % (i, self._arcs[i], self._tags[i])
        return str

    def __eq__(self, other):
        return (
            type(self) == type(other) and self._comparison_key == other._comparison_key
        )

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, DependencySpan):
            raise_unorderable_types("<", self, other)
        return self._comparison_key < other._comparison_key

    def __hash__(self):
        """
        :return: The hash value of this ``DependencySpan``.
        """
        return self._hash


#################################################################
# Chart Cell
#################################################################


class ChartCell:
    """
    A cell from the parse chart formed when performing the CYK algorithm.
    Each cell keeps track of its x and y coordinates (though this will probably
    be discarded), and a list of spans serving as the cell's entries.
    """

    def __init__(self, x, y):
        """
        :param x: This cell's x coordinate.
        :type x: int.
        :param y: This cell's y coordinate.
        :type y: int.
        """
        self._x = x
        self._y = y
        self._entries = set()

    def add(self, span):
        """
        Appends the given span to the list of spans
        representing the chart cell's entries.

        :param span: The span to add.
        :type span: DependencySpan
        """
        self._entries.add(span)

    def __str__(self):
        """
        :return: A verbose string representation of this ``ChartCell``.
        :rtype: str.
        """
        return "CC[%d,%d]: %s" % (self._x, self._y, self._entries)

    def __repr__(self):
        """
        :return: A concise string representation of this ``ChartCell``.
        :rtype: str.
        """
        return "%s" % self


#################################################################
# Parsing  with Dependency Grammars
#################################################################


class ProjectiveDependencyParser:
    """
    A projective, rule-based, dependency parser.  A ProjectiveDependencyParser
    is created with a DependencyGrammar, a set of productions specifying
    word-to-word dependency relations.  The parse() method will then
    return the set of all parses, in tree representation, for a given input
    sequence of tokens.  Each parse must meet the requirements of the both
    the grammar and the projectivity constraint which specifies that the
    branches of the dependency tree are not allowed to cross.  Alternatively,
    this can be understood as stating that each parent node and its children
    in the parse tree form a continuous substring of the input sequence.
    """

    def __init__(self, dependency_grammar):
        """
        Create a new ProjectiveDependencyParser, from a word-to-word
        dependency grammar ``DependencyGrammar``.

        :param dependency_grammar: A word-to-word relation dependencygrammar.
        :type dependency_grammar: DependencyGrammar
        """
        self._grammar = dependency_grammar

    def parse(self, tokens):
        """
        Performs a projective dependency parse on the list of tokens using
        a chart-based, span-concatenation algorithm similar to Eisner (1996).

        :param tokens: The list of input tokens.
        :type tokens: list(str)
        :return: An iterator over parse trees.
        :rtype: iter(Tree)
        """
        self._tokens = list(tokens)
        chart = []
        for i in range(0, len(self._tokens) + 1):
            chart.append([])
            for j in range(0, len(self._tokens) + 1):
                chart[i].append(ChartCell(i, j))
                if i == j + 1:
                    chart[i][j].add(DependencySpan(i - 1, i, i - 1, [-1], ["null"]))

        for i in range(1, len(self._tokens) + 1):
            for j in range(i - 2, -1, -1):
                for k in range(i - 1, j, -1):
                    for span1 in chart[k][j]._entries:
                        for span2 in chart[i][k]._entries:
                            for newspan in self.concatenate(span1, span2):
                                chart[i][j].add(newspan)

        for parse in chart[len(self._tokens)][0]._entries:
            conll_format = ""
            #            malt_format = ""
            for i in range(len(tokens)):
                #                malt_format += '%s\t%s\t%d\t%s\n' % (tokens[i], 'null', parse._arcs[i] + 1, 'null')
                # conll_format += '\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' % (i+1, tokens[i], tokens[i], 'null', 'null', 'null', parse._arcs[i] + 1, 'null', '-', '-')
                # Modify to comply with the new Dependency Graph requirement (at least must have an root elements)
                conll_format += "\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n" % (
                    i + 1,
                    tokens[i],
                    tokens[i],
                    "null",
                    "null",
                    "null",
                    parse._arcs[i] + 1,
                    "ROOT",
                    "-",
                    "-",
                )
            dg = DependencyGraph(conll_format)
            #           if self.meets_arity(dg):
            yield dg.tree()

    def concatenate(self, span1, span2):
        """
        Concatenates the two spans in whichever way possible.  This
        includes rightward concatenation (from the leftmost word of the
        leftmost span to the rightmost word of the rightmost span) and
        leftward concatenation (vice-versa) between adjacent spans.  Unlike
        Eisner's presentation of span concatenation, these spans do not
        share or pivot on a particular word/word-index.

        :return: A list of new spans formed through concatenation.
        :rtype: list(DependencySpan)
        """
        spans = []
        if span1._start_index == span2._start_index:
            print("Error: Mismatched spans - replace this with thrown error")
        if span1._start_index > span2._start_index:
            temp_span = span1
            span1 = span2
            span2 = temp_span
        # adjacent rightward covered concatenation
        new_arcs = span1._arcs + span2._arcs
        new_tags = span1._tags + span2._tags
        if self._grammar.contains(
            self._tokens[span1._head_index], self._tokens[span2._head_index]
        ):
            #           print('Performing rightward cover %d to %d' % (span1._head_index, span2._head_index))
            new_arcs[span2._head_index - span1._start_index] = span1._head_index
            spans.append(
                DependencySpan(
                    span1._start_index,
                    span2._end_index,
                    span1._head_index,
                    new_arcs,
                    new_tags,
                )
            )
        # adjacent leftward covered concatenation
        new_arcs = span1._arcs + span2._arcs
        if self._grammar.contains(
            self._tokens[span2._head_index], self._tokens[span1._head_index]
        ):
            #           print('performing leftward cover %d to %d' % (span2._head_index, span1._head_index))
            new_arcs[span1._head_index - span1._start_index] = span2._head_index
            spans.append(
                DependencySpan(
                    span1._start_index,
                    span2._end_index,
                    span2._head_index,
                    new_arcs,
                    new_tags,
                )
            )
        return spans


#################################################################
# Parsing  with Probabilistic Dependency Grammars
#################################################################


class ProbabilisticProjectiveDependencyParser:
    """A probabilistic, projective dependency parser.

    This parser returns the most probable projective parse derived from the
    probabilistic dependency grammar derived from the train() method.  The
    probabilistic model is an implementation of Eisner's (1996) Model C, which
    conditions on head-word, head-tag, child-word, and child-tag.  The decoding
    uses a bottom-up chart-based span concatenation algorithm that's identical
    to the one utilized by the rule-based projective parser.

    Usage example

    >>> from nltk.parse.dependencygraph import conll_data2

    >>> graphs = [
    ... DependencyGraph(entry) for entry in conll_data2.split('\\n\\n') if entry
    ... ]

    >>> ppdp = ProbabilisticProjectiveDependencyParser()
    >>> ppdp.train(graphs)

    >>> sent = ['Cathy', 'zag', 'hen', 'wild', 'zwaaien', '.']
    >>> list(ppdp.parse(sent))
    [Tree('zag', ['Cathy', 'hen', Tree('zwaaien', ['wild', '.'])])]

    """

    def __init__(self):
        """
        Create a new probabilistic dependency parser.  No additional
        operations are necessary.
        """

    def parse(self, tokens):
        """
        Parses the list of tokens subject to the projectivity constraint
        and the productions in the parser's grammar.  This uses a method
        similar to the span-concatenation algorithm defined in Eisner (1996).
        It returns the most probable parse derived from the parser's
        probabilistic dependency grammar.
        """
        self._tokens = list(tokens)
        chart = []
        for i in range(0, len(self._tokens) + 1):
            chart.append([])
            for j in range(0, len(self._tokens) + 1):
                chart[i].append(ChartCell(i, j))
                if i == j + 1:
                    if tokens[i - 1] in self._grammar._tags:
                        for tag in self._grammar._tags[tokens[i - 1]]:
                            chart[i][j].add(
                                DependencySpan(i - 1, i, i - 1, [-1], [tag])
                            )
                    else:
                        print(
                            "No tag found for input token '%s', parse is impossible."
                            % tokens[i - 1]
                        )
                        return []
        for i in range(1, len(self._tokens) + 1):
            for j in range(i - 2, -1, -1):
                for k in range(i - 1, j, -1):
                    for span1 in chart[k][j]._entries:
                        for span2 in chart[i][k]._entries:
                            for newspan in self.concatenate(span1, span2):
                                chart[i][j].add(newspan)
        trees = []
        max_parse = None
        max_score = 0
        for parse in chart[len(self._tokens)][0]._entries:
            conll_format = ""
            malt_format = ""
            for i in range(len(tokens)):
                malt_format += "%s\t%s\t%d\t%s\n" % (
                    tokens[i],
                    "null",
                    parse._arcs[i] + 1,
                    "null",
                )
                # conll_format += '\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' % (i+1, tokens[i], tokens[i], parse._tags[i], parse._tags[i], 'null', parse._arcs[i] + 1, 'null', '-', '-')
                # Modify to comply with recent change in dependency graph such that there must be a ROOT element.
                conll_format += "\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n" % (
                    i + 1,
                    tokens[i],
                    tokens[i],
                    parse._tags[i],
                    parse._tags[i],
                    "null",
                    parse._arcs[i] + 1,
                    "ROOT",
                    "-",
                    "-",
                )
            dg = DependencyGraph(conll_format)
            score = self.compute_prob(dg)
            trees.append((score, dg.tree()))
        trees.sort()
        return (tree for (score, tree) in trees)

    def concatenate(self, span1, span2):
        """
        Concatenates the two spans in whichever way possible.  This
        includes rightward concatenation (from the leftmost word of the
        leftmost span to the rightmost word of the rightmost span) and
        leftward concatenation (vice-versa) between adjacent spans.  Unlike
        Eisner's presentation of span concatenation, these spans do not
        share or pivot on a particular word/word-index.

        :return: A list of new spans formed through concatenation.
        :rtype: list(DependencySpan)
        """
        spans = []
        if span1._start_index == span2._start_index:
            print("Error: Mismatched spans - replace this with thrown error")
        if span1._start_index > span2._start_index:
            temp_span = span1
            span1 = span2
            span2 = temp_span
        # adjacent rightward covered concatenation
        new_arcs = span1._arcs + span2._arcs
        new_tags = span1._tags + span2._tags
        if self._grammar.contains(
            self._tokens[span1._head_index], self._tokens[span2._head_index]
        ):
            new_arcs[span2._head_index - span1._start_index] = span1._head_index
            spans.append(
                DependencySpan(
                    span1._start_index,
                    span2._end_index,
                    span1._head_index,
                    new_arcs,
                    new_tags,
                )
            )
        # adjacent leftward covered concatenation
        new_arcs = span1._arcs + span2._arcs
        new_tags = span1._tags + span2._tags
        if self._grammar.contains(
            self._tokens[span2._head_index], self._tokens[span1._head_index]
        ):
            new_arcs[span1._head_index - span1._start_index] = span2._head_index
            spans.append(
                DependencySpan(
                    span1._start_index,
                    span2._end_index,
                    span2._head_index,
                    new_arcs,
                    new_tags,
                )
            )
        return spans

    def train(self, graphs):
        """
        Trains a ProbabilisticDependencyGrammar based on the list of input
        DependencyGraphs.  This model is an implementation of Eisner's (1996)
        Model C, which derives its statistics from head-word, head-tag,
        child-word, and child-tag relationships.

        :param graphs: A list of dependency graphs to train from.
        :type: list(DependencyGraph)
        """
        productions = []
        events = defaultdict(int)
        tags = {}
        for dg in graphs:
            for node_index in range(1, len(dg.nodes)):
                # children = dg.nodes[node_index]['deps']
                children = list(
                    chain.from_iterable(dg.nodes[node_index]["deps"].values())
                )

                nr_left_children = dg.left_children(node_index)
                nr_right_children = dg.right_children(node_index)
                nr_children = nr_left_children + nr_right_children
                for child_index in range(
                    0 - (nr_left_children + 1), nr_right_children + 2
                ):
                    head_word = dg.nodes[node_index]["word"]
                    head_tag = dg.nodes[node_index]["tag"]
                    if head_word in tags:
                        tags[head_word].add(head_tag)
                    else:
                        tags[head_word] = {head_tag}
                    child = "STOP"
                    child_tag = "STOP"
                    prev_word = "START"
                    prev_tag = "START"
                    if child_index < 0:
                        array_index = child_index + nr_left_children
                        if array_index >= 0:
                            child = dg.nodes[children[array_index]]["word"]
                            child_tag = dg.nodes[children[array_index]]["tag"]
                        if child_index != -1:
                            prev_word = dg.nodes[children[array_index + 1]]["word"]
                            prev_tag = dg.nodes[children[array_index + 1]]["tag"]
                        if child != "STOP":
                            productions.append(DependencyProduction(head_word, [child]))
                        head_event = "(head ({} {}) (mods ({}, {}, {}) left))".format(
                            child,
                            child_tag,
                            prev_tag,
                            head_word,
                            head_tag,
                        )
                        mod_event = "(mods ({}, {}, {}) left))".format(
                            prev_tag,
                            head_word,
                            head_tag,
                        )
                        events[head_event] += 1
                        events[mod_event] += 1
                    elif child_index > 0:
                        array_index = child_index + nr_left_children - 1
                        if array_index < nr_children:
                            child = dg.nodes[children[array_index]]["word"]
                            child_tag = dg.nodes[children[array_index]]["tag"]
                        if child_index != 1:
                            prev_word = dg.nodes[children[array_index - 1]]["word"]
                            prev_tag = dg.nodes[children[array_index - 1]]["tag"]
                        if child != "STOP":
                            productions.append(DependencyProduction(head_word, [child]))
                        head_event = "(head ({} {}) (mods ({}, {}, {}) right))".format(
                            child,
                            child_tag,
                            prev_tag,
                            head_word,
                            head_tag,
                        )
                        mod_event = "(mods ({}, {}, {}) right))".format(
                            prev_tag,
                            head_word,
                            head_tag,
                        )
                        events[head_event] += 1
                        events[mod_event] += 1
        self._grammar = ProbabilisticDependencyGrammar(productions, events, tags)

    def compute_prob(self, dg):
        """
        Computes the probability of a dependency graph based
        on the parser's probability model (defined by the parser's
        statistical dependency grammar).

        :param dg: A dependency graph to score.
        :type dg: DependencyGraph
        :return: The probability of the dependency graph.
        :rtype: int
        """
        prob = 1.0
        for node_index in range(1, len(dg.nodes)):
            # children = dg.nodes[node_index]['deps']
            children = list(chain.from_iterable(dg.nodes[node_index]["deps"].values()))

            nr_left_children = dg.left_children(node_index)
            nr_right_children = dg.right_children(node_index)
            nr_children = nr_left_children + nr_right_children
            for child_index in range(0 - (nr_left_children + 1), nr_right_children + 2):
                head_word = dg.nodes[node_index]["word"]
                head_tag = dg.nodes[node_index]["tag"]
                child = "STOP"
                child_tag = "STOP"
                prev_word = "START"
                prev_tag = "START"
                if child_index < 0:
                    array_index = child_index + nr_left_children
                    if array_index >= 0:
                        child = dg.nodes[children[array_index]]["word"]
                        child_tag = dg.nodes[children[array_index]]["tag"]
                    if child_index != -1:
                        prev_word = dg.nodes[children[array_index + 1]]["word"]
                        prev_tag = dg.nodes[children[array_index + 1]]["tag"]
                    head_event = "(head ({} {}) (mods ({}, {}, {}) left))".format(
                        child,
                        child_tag,
                        prev_tag,
                        head_word,
                        head_tag,
                    )
                    mod_event = "(mods ({}, {}, {}) left))".format(
                        prev_tag,
                        head_word,
                        head_tag,
                    )
                    h_count = self._grammar._events[head_event]
                    m_count = self._grammar._events[mod_event]

                    # If the grammar is not covered
                    if m_count != 0:
                        prob *= h_count / m_count
                    else:
                        prob = 0.00000001  # Very small number

                elif child_index > 0:
                    array_index = child_index + nr_left_children - 1
                    if array_index < nr_children:
                        child = dg.nodes[children[array_index]]["word"]
                        child_tag = dg.nodes[children[array_index]]["tag"]
                    if child_index != 1:
                        prev_word = dg.nodes[children[array_index - 1]]["word"]
                        prev_tag = dg.nodes[children[array_index - 1]]["tag"]
                    head_event = "(head ({} {}) (mods ({}, {}, {}) right))".format(
                        child,
                        child_tag,
                        prev_tag,
                        head_word,
                        head_tag,
                    )
                    mod_event = "(mods ({}, {}, {}) right))".format(
                        prev_tag,
                        head_word,
                        head_tag,
                    )
                    h_count = self._grammar._events[head_event]
                    m_count = self._grammar._events[mod_event]

                    if m_count != 0:
                        prob *= h_count / m_count
                    else:
                        prob = 0.00000001  # Very small number

        return prob


#################################################################
# Demos
#################################################################


def demo():
    projective_rule_parse_demo()
    #    arity_parse_demo()
    projective_prob_parse_demo()


def projective_rule_parse_demo():
    """
    A demonstration showing the creation and use of a
    ``DependencyGrammar`` to perform a projective dependency
    parse.
    """
    grammar = DependencyGrammar.fromstring(
        """
    'scratch' -> 'cats' | 'walls'
    'walls' -> 'the'
    'cats' -> 'the'
    """
    )
    print(grammar)
    pdp = ProjectiveDependencyParser(grammar)
    trees = pdp.parse(["the", "cats", "scratch", "the", "walls"])
    for tree in trees:
        print(tree)


def arity_parse_demo():
    """
    A demonstration showing the creation of a ``DependencyGrammar``
    in which a specific number of modifiers is listed for a given
    head.  This can further constrain the number of possible parses
    created by a ``ProjectiveDependencyParser``.
    """
    print()
    print("A grammar with no arity constraints. Each DependencyProduction")
    print("specifies a relationship between one head word and only one")
    print("modifier word.")
    grammar = DependencyGrammar.fromstring(
        """
    'fell' -> 'price' | 'stock'
    'price' -> 'of' | 'the'
    'of' -> 'stock'
    'stock' -> 'the'
    """
    )
    print(grammar)

    print()
    print("For the sentence 'The price of the stock fell', this grammar")
    print("will produce the following three parses:")
    pdp = ProjectiveDependencyParser(grammar)
    trees = pdp.parse(["the", "price", "of", "the", "stock", "fell"])
    for tree in trees:
        print(tree)

    print()
    print("By contrast, the following grammar contains a ")
    print("DependencyProduction that specifies a relationship")
    print("between a single head word, 'price', and two modifier")
    print("words, 'of' and 'the'.")
    grammar = DependencyGrammar.fromstring(
        """
    'fell' -> 'price' | 'stock'
    'price' -> 'of' 'the'
    'of' -> 'stock'
    'stock' -> 'the'
    """
    )
    print(grammar)

    print()
    print(
        "This constrains the number of possible parses to just one:"
    )  # unimplemented, soon to replace
    pdp = ProjectiveDependencyParser(grammar)
    trees = pdp.parse(["the", "price", "of", "the", "stock", "fell"])
    for tree in trees:
        print(tree)


def projective_prob_parse_demo():
    """
    A demo showing the training and use of a projective
    dependency parser.
    """
    from nltk.parse.dependencygraph import conll_data2

    graphs = [DependencyGraph(entry) for entry in conll_data2.split("\n\n") if entry]
    ppdp = ProbabilisticProjectiveDependencyParser()
    print("Training Probabilistic Projective Dependency Parser...")
    ppdp.train(graphs)

    sent = ["Cathy", "zag", "hen", "wild", "zwaaien", "."]
    print("Parsing '", " ".join(sent), "'...")
    print("Parse:")
    for tree in ppdp.parse(sent):
        print(tree)


if __name__ == "__main__":
    demo()
