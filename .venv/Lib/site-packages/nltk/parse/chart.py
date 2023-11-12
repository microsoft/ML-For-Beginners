# Natural Language Toolkit: A Chart Parser
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Jean Mark Gawron <gawron@mail.sdsu.edu>
#         Peter Ljunglöf <peter.ljunglof@heatherleaf.se>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Data classes and parser implementations for "chart parsers", which
use dynamic programming to efficiently parse a text.  A chart
parser derives parse trees for a text by iteratively adding "edges"
to a "chart."  Each edge represents a hypothesis about the tree
structure for a subsequence of the text.  The chart is a
"blackboard" for composing and combining these hypotheses.

When a chart parser begins parsing a text, it creates a new (empty)
chart, spanning the text.  It then incrementally adds new edges to the
chart.  A set of "chart rules" specifies the conditions under which
new edges should be added to the chart.  Once the chart reaches a
stage where none of the chart rules adds any new edges, parsing is
complete.

Charts are encoded with the ``Chart`` class, and edges are encoded with
the ``TreeEdge`` and ``LeafEdge`` classes.  The chart parser module
defines three chart parsers:

  - ``ChartParser`` is a simple and flexible chart parser.  Given a
    set of chart rules, it will apply those rules to the chart until
    no more edges are added.

  - ``SteppingChartParser`` is a subclass of ``ChartParser`` that can
    be used to step through the parsing process.
"""

import itertools
import re
import warnings
from functools import total_ordering

from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict

########################################################################
##  Edges
########################################################################


@total_ordering
class EdgeI:
    """
    A hypothesis about the structure of part of a sentence.
    Each edge records the fact that a structure is (partially)
    consistent with the sentence.  An edge contains:

    - A span, indicating what part of the sentence is
      consistent with the hypothesized structure.
    - A left-hand side, specifying what kind of structure is
      hypothesized.
    - A right-hand side, specifying the contents of the
      hypothesized structure.
    - A dot position, indicating how much of the hypothesized
      structure is consistent with the sentence.

    Every edge is either complete or incomplete:

    - An edge is complete if its structure is fully consistent
      with the sentence.
    - An edge is incomplete if its structure is partially
      consistent with the sentence.  For every incomplete edge, the
      span specifies a possible prefix for the edge's structure.

    There are two kinds of edge:

    - A ``TreeEdge`` records which trees have been found to
      be (partially) consistent with the text.
    - A ``LeafEdge`` records the tokens occurring in the text.

    The ``EdgeI`` interface provides a common interface to both types
    of edge, allowing chart parsers to treat them in a uniform manner.
    """

    def __init__(self):
        if self.__class__ == EdgeI:
            raise TypeError("Edge is an abstract interface")

    # ////////////////////////////////////////////////////////////
    # Span
    # ////////////////////////////////////////////////////////////

    def span(self):
        """
        Return a tuple ``(s, e)``, where ``tokens[s:e]`` is the
        portion of the sentence that is consistent with this
        edge's structure.

        :rtype: tuple(int, int)
        """
        raise NotImplementedError()

    def start(self):
        """
        Return the start index of this edge's span.

        :rtype: int
        """
        raise NotImplementedError()

    def end(self):
        """
        Return the end index of this edge's span.

        :rtype: int
        """
        raise NotImplementedError()

    def length(self):
        """
        Return the length of this edge's span.

        :rtype: int
        """
        raise NotImplementedError()

    # ////////////////////////////////////////////////////////////
    # Left Hand Side
    # ////////////////////////////////////////////////////////////

    def lhs(self):
        """
        Return this edge's left-hand side, which specifies what kind
        of structure is hypothesized by this edge.

        :see: ``TreeEdge`` and ``LeafEdge`` for a description of
            the left-hand side values for each edge type.
        """
        raise NotImplementedError()

    # ////////////////////////////////////////////////////////////
    # Right Hand Side
    # ////////////////////////////////////////////////////////////

    def rhs(self):
        """
        Return this edge's right-hand side, which specifies
        the content of the structure hypothesized by this edge.

        :see: ``TreeEdge`` and ``LeafEdge`` for a description of
            the right-hand side values for each edge type.
        """
        raise NotImplementedError()

    def dot(self):
        """
        Return this edge's dot position, which indicates how much of
        the hypothesized structure is consistent with the
        sentence.  In particular, ``self.rhs[:dot]`` is consistent
        with ``tokens[self.start():self.end()]``.

        :rtype: int
        """
        raise NotImplementedError()

    def nextsym(self):
        """
        Return the element of this edge's right-hand side that
        immediately follows its dot.

        :rtype: Nonterminal or terminal or None
        """
        raise NotImplementedError()

    def is_complete(self):
        """
        Return True if this edge's structure is fully consistent
        with the text.

        :rtype: bool
        """
        raise NotImplementedError()

    def is_incomplete(self):
        """
        Return True if this edge's structure is partially consistent
        with the text.

        :rtype: bool
        """
        raise NotImplementedError()

    # ////////////////////////////////////////////////////////////
    # Comparisons & hashing
    # ////////////////////////////////////////////////////////////

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and self._comparison_key == other._comparison_key
        )

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, EdgeI):
            raise_unorderable_types("<", self, other)
        if self.__class__ is other.__class__:
            return self._comparison_key < other._comparison_key
        else:
            return self.__class__.__name__ < other.__class__.__name__

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self._comparison_key)
            return self._hash


class TreeEdge(EdgeI):
    """
    An edge that records the fact that a tree is (partially)
    consistent with the sentence.  A tree edge consists of:

    - A span, indicating what part of the sentence is
      consistent with the hypothesized tree.
    - A left-hand side, specifying the hypothesized tree's node
      value.
    - A right-hand side, specifying the hypothesized tree's
      children.  Each element of the right-hand side is either a
      terminal, specifying a token with that terminal as its leaf
      value; or a nonterminal, specifying a subtree with that
      nonterminal's symbol as its node value.
    - A dot position, indicating which children are consistent
      with part of the sentence.  In particular, if ``dot`` is the
      dot position, ``rhs`` is the right-hand size, ``(start,end)``
      is the span, and ``sentence`` is the list of tokens in the
      sentence, then ``tokens[start:end]`` can be spanned by the
      children specified by ``rhs[:dot]``.

    For more information about edges, see the ``EdgeI`` interface.
    """

    def __init__(self, span, lhs, rhs, dot=0):
        """
        Construct a new ``TreeEdge``.

        :type span: tuple(int, int)
        :param span: A tuple ``(s, e)``, where ``tokens[s:e]`` is the
            portion of the sentence that is consistent with the new
            edge's structure.
        :type lhs: Nonterminal
        :param lhs: The new edge's left-hand side, specifying the
            hypothesized tree's node value.
        :type rhs: list(Nonterminal and str)
        :param rhs: The new edge's right-hand side, specifying the
            hypothesized tree's children.
        :type dot: int
        :param dot: The position of the new edge's dot.  This position
            specifies what prefix of the production's right hand side
            is consistent with the text.  In particular, if
            ``sentence`` is the list of tokens in the sentence, then
            ``okens[span[0]:span[1]]`` can be spanned by the
            children specified by ``rhs[:dot]``.
        """
        self._span = span
        self._lhs = lhs
        rhs = tuple(rhs)
        self._rhs = rhs
        self._dot = dot
        self._comparison_key = (span, lhs, rhs, dot)

    @staticmethod
    def from_production(production, index):
        """
        Return a new ``TreeEdge`` formed from the given production.
        The new edge's left-hand side and right-hand side will
        be taken from ``production``; its span will be
        ``(index,index)``; and its dot position will be ``0``.

        :rtype: TreeEdge
        """
        return TreeEdge(
            span=(index, index), lhs=production.lhs(), rhs=production.rhs(), dot=0
        )

    def move_dot_forward(self, new_end):
        """
        Return a new ``TreeEdge`` formed from this edge.
        The new edge's dot position is increased by ``1``,
        and its end index will be replaced by ``new_end``.

        :param new_end: The new end index.
        :type new_end: int
        :rtype: TreeEdge
        """
        return TreeEdge(
            span=(self._span[0], new_end),
            lhs=self._lhs,
            rhs=self._rhs,
            dot=self._dot + 1,
        )

    # Accessors
    def lhs(self):
        return self._lhs

    def span(self):
        return self._span

    def start(self):
        return self._span[0]

    def end(self):
        return self._span[1]

    def length(self):
        return self._span[1] - self._span[0]

    def rhs(self):
        return self._rhs

    def dot(self):
        return self._dot

    def is_complete(self):
        return self._dot == len(self._rhs)

    def is_incomplete(self):
        return self._dot != len(self._rhs)

    def nextsym(self):
        if self._dot >= len(self._rhs):
            return None
        else:
            return self._rhs[self._dot]

    # String representation
    def __str__(self):
        str = f"[{self._span[0]}:{self._span[1]}] "
        str += "%-2r ->" % (self._lhs,)

        for i in range(len(self._rhs)):
            if i == self._dot:
                str += " *"
            str += " %s" % repr(self._rhs[i])
        if len(self._rhs) == self._dot:
            str += " *"
        return str

    def __repr__(self):
        return "[Edge: %s]" % self


class LeafEdge(EdgeI):
    """
    An edge that records the fact that a leaf value is consistent with
    a word in the sentence.  A leaf edge consists of:

    - An index, indicating the position of the word.
    - A leaf, specifying the word's content.

    A leaf edge's left-hand side is its leaf value, and its right hand
    side is ``()``.  Its span is ``[index, index+1]``, and its dot
    position is ``0``.
    """

    def __init__(self, leaf, index):
        """
        Construct a new ``LeafEdge``.

        :param leaf: The new edge's leaf value, specifying the word
            that is recorded by this edge.
        :param index: The new edge's index, specifying the position of
            the word that is recorded by this edge.
        """
        self._leaf = leaf
        self._index = index
        self._comparison_key = (leaf, index)

    # Accessors
    def lhs(self):
        return self._leaf

    def span(self):
        return (self._index, self._index + 1)

    def start(self):
        return self._index

    def end(self):
        return self._index + 1

    def length(self):
        return 1

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

    # String representations
    def __str__(self):
        return f"[{self._index}:{self._index + 1}] {repr(self._leaf)}"

    def __repr__(self):
        return "[Edge: %s]" % (self)


########################################################################
##  Chart
########################################################################


class Chart:
    """
    A blackboard for hypotheses about the syntactic constituents of a
    sentence.  A chart contains a set of edges, and each edge encodes
    a single hypothesis about the structure of some portion of the
    sentence.

    The ``select`` method can be used to select a specific collection
    of edges.  For example ``chart.select(is_complete=True, start=0)``
    yields all complete edges whose start indices are 0.  To ensure
    the efficiency of these selection operations, ``Chart`` dynamically
    creates and maintains an index for each set of attributes that
    have been selected on.

    In order to reconstruct the trees that are represented by an edge,
    the chart associates each edge with a set of child pointer lists.
    A child pointer list is a list of the edges that license an
    edge's right-hand side.

    :ivar _tokens: The sentence that the chart covers.
    :ivar _num_leaves: The number of tokens.
    :ivar _edges: A list of the edges in the chart
    :ivar _edge_to_cpls: A dictionary mapping each edge to a set
        of child pointer lists that are associated with that edge.
    :ivar _indexes: A dictionary mapping tuples of edge attributes
        to indices, where each index maps the corresponding edge
        attribute values to lists of edges.
    """

    def __init__(self, tokens):
        """
        Construct a new chart. The chart is initialized with the
        leaf edges corresponding to the terminal leaves.

        :type tokens: list
        :param tokens: The sentence that this chart will be used to parse.
        """
        # Record the sentence token and the sentence length.
        self._tokens = tuple(tokens)
        self._num_leaves = len(self._tokens)

        # Initialise the chart.
        self.initialize()

    def initialize(self):
        """
        Clear the chart.
        """
        # A list of edges contained in this chart.
        self._edges = []

        # The set of child pointer lists associated with each edge.
        self._edge_to_cpls = {}

        # Indexes mapping attribute values to lists of edges
        # (used by select()).
        self._indexes = {}

    # ////////////////////////////////////////////////////////////
    # Sentence Access
    # ////////////////////////////////////////////////////////////

    def num_leaves(self):
        """
        Return the number of words in this chart's sentence.

        :rtype: int
        """
        return self._num_leaves

    def leaf(self, index):
        """
        Return the leaf value of the word at the given index.

        :rtype: str
        """
        return self._tokens[index]

    def leaves(self):
        """
        Return a list of the leaf values of each word in the
        chart's sentence.

        :rtype: list(str)
        """
        return self._tokens

    # ////////////////////////////////////////////////////////////
    # Edge access
    # ////////////////////////////////////////////////////////////

    def edges(self):
        """
        Return a list of all edges in this chart.  New edges
        that are added to the chart after the call to edges()
        will *not* be contained in this list.

        :rtype: list(EdgeI)
        :see: ``iteredges``, ``select``
        """
        return self._edges[:]

    def iteredges(self):
        """
        Return an iterator over the edges in this chart.  It is
        not guaranteed that new edges which are added to the
        chart before the iterator is exhausted will also be generated.

        :rtype: iter(EdgeI)
        :see: ``edges``, ``select``
        """
        return iter(self._edges)

    # Iterating over the chart yields its edges.
    __iter__ = iteredges

    def num_edges(self):
        """
        Return the number of edges contained in this chart.

        :rtype: int
        """
        return len(self._edge_to_cpls)

    def select(self, **restrictions):
        """
        Return an iterator over the edges in this chart.  Any
        new edges that are added to the chart before the iterator
        is exahusted will also be generated.  ``restrictions``
        can be used to restrict the set of edges that will be
        generated.

        :param span: Only generate edges ``e`` where ``e.span()==span``
        :param start: Only generate edges ``e`` where ``e.start()==start``
        :param end: Only generate edges ``e`` where ``e.end()==end``
        :param length: Only generate edges ``e`` where ``e.length()==length``
        :param lhs: Only generate edges ``e`` where ``e.lhs()==lhs``
        :param rhs: Only generate edges ``e`` where ``e.rhs()==rhs``
        :param nextsym: Only generate edges ``e`` where
            ``e.nextsym()==nextsym``
        :param dot: Only generate edges ``e`` where ``e.dot()==dot``
        :param is_complete: Only generate edges ``e`` where
            ``e.is_complete()==is_complete``
        :param is_incomplete: Only generate edges ``e`` where
            ``e.is_incomplete()==is_incomplete``
        :rtype: iter(EdgeI)
        """
        # If there are no restrictions, then return all edges.
        if restrictions == {}:
            return iter(self._edges)

        # Find the index corresponding to the given restrictions.
        restr_keys = sorted(restrictions.keys())
        restr_keys = tuple(restr_keys)

        # If it doesn't exist, then create it.
        if restr_keys not in self._indexes:
            self._add_index(restr_keys)

        vals = tuple(restrictions[key] for key in restr_keys)
        return iter(self._indexes[restr_keys].get(vals, []))

    def _add_index(self, restr_keys):
        """
        A helper function for ``select``, which creates a new index for
        a given set of attributes (aka restriction keys).
        """
        # Make sure it's a valid index.
        for key in restr_keys:
            if not hasattr(EdgeI, key):
                raise ValueError("Bad restriction: %s" % key)

        # Create the index.
        index = self._indexes[restr_keys] = {}

        # Add all existing edges to the index.
        for edge in self._edges:
            vals = tuple(getattr(edge, key)() for key in restr_keys)
            index.setdefault(vals, []).append(edge)

    def _register_with_indexes(self, edge):
        """
        A helper function for ``insert``, which registers the new
        edge with all existing indexes.
        """
        for (restr_keys, index) in self._indexes.items():
            vals = tuple(getattr(edge, key)() for key in restr_keys)
            index.setdefault(vals, []).append(edge)

    # ////////////////////////////////////////////////////////////
    # Edge Insertion
    # ////////////////////////////////////////////////////////////

    def insert_with_backpointer(self, new_edge, previous_edge, child_edge):
        """
        Add a new edge to the chart, using a pointer to the previous edge.
        """
        cpls = self.child_pointer_lists(previous_edge)
        new_cpls = [cpl + (child_edge,) for cpl in cpls]
        return self.insert(new_edge, *new_cpls)

    def insert(self, edge, *child_pointer_lists):
        """
        Add a new edge to the chart, and return True if this operation
        modified the chart.  In particular, return true iff the chart
        did not already contain ``edge``, or if it did not already associate
        ``child_pointer_lists`` with ``edge``.

        :type edge: EdgeI
        :param edge: The new edge
        :type child_pointer_lists: sequence of tuple(EdgeI)
        :param child_pointer_lists: A sequence of lists of the edges that
            were used to form this edge.  This list is used to reconstruct
            the trees (or partial trees) that are associated with ``edge``.
        :rtype: bool
        """
        # Is it a new edge?
        if edge not in self._edge_to_cpls:
            # Add it to the list of edges.
            self._append_edge(edge)
            # Register with indexes.
            self._register_with_indexes(edge)

        # Get the set of child pointer lists for this edge.
        cpls = self._edge_to_cpls.setdefault(edge, OrderedDict())
        chart_was_modified = False
        for child_pointer_list in child_pointer_lists:
            child_pointer_list = tuple(child_pointer_list)
            if child_pointer_list not in cpls:
                # It's a new CPL; register it, and return true.
                cpls[child_pointer_list] = True
                chart_was_modified = True
        return chart_was_modified

    def _append_edge(self, edge):
        self._edges.append(edge)

    # ////////////////////////////////////////////////////////////
    # Tree extraction & child pointer lists
    # ////////////////////////////////////////////////////////////

    def parses(self, root, tree_class=Tree):
        """
        Return an iterator of the complete tree structures that span
        the entire chart, and whose root node is ``root``.
        """
        for edge in self.select(start=0, end=self._num_leaves, lhs=root):
            yield from self.trees(edge, tree_class=tree_class, complete=True)

    def trees(self, edge, tree_class=Tree, complete=False):
        """
        Return an iterator of the tree structures that are associated
        with ``edge``.

        If ``edge`` is incomplete, then the unexpanded children will be
        encoded as childless subtrees, whose node value is the
        corresponding terminal or nonterminal.

        :rtype: list(Tree)
        :note: If two trees share a common subtree, then the same
            Tree may be used to encode that subtree in
            both trees.  If you need to eliminate this subtree
            sharing, then create a deep copy of each tree.
        """
        return iter(self._trees(edge, complete, memo={}, tree_class=tree_class))

    def _trees(self, edge, complete, memo, tree_class):
        """
        A helper function for ``trees``.

        :param memo: A dictionary used to record the trees that we've
            generated for each edge, so that when we see an edge more
            than once, we can reuse the same trees.
        """
        # If we've seen this edge before, then reuse our old answer.
        if edge in memo:
            return memo[edge]

        # when we're reading trees off the chart, don't use incomplete edges
        if complete and edge.is_incomplete():
            return []

        # Leaf edges.
        if isinstance(edge, LeafEdge):
            leaf = self._tokens[edge.start()]
            memo[edge] = [leaf]
            return [leaf]

        # Until we're done computing the trees for edge, set
        # memo[edge] to be empty.  This has the effect of filtering
        # out any cyclic trees (i.e., trees that contain themselves as
        # descendants), because if we reach this edge via a cycle,
        # then it will appear that the edge doesn't generate any trees.
        memo[edge] = []
        trees = []
        lhs = edge.lhs().symbol()

        # Each child pointer list can be used to form trees.
        for cpl in self.child_pointer_lists(edge):
            # Get the set of child choices for each child pointer.
            # child_choices[i] is the set of choices for the tree's
            # ith child.
            child_choices = [self._trees(cp, complete, memo, tree_class) for cp in cpl]

            # For each combination of children, add a tree.
            for children in itertools.product(*child_choices):
                trees.append(tree_class(lhs, children))

        # If the edge is incomplete, then extend it with "partial trees":
        if edge.is_incomplete():
            unexpanded = [tree_class(elt, []) for elt in edge.rhs()[edge.dot() :]]
            for tree in trees:
                tree.extend(unexpanded)

        # Update the memoization dictionary.
        memo[edge] = trees

        # Return the list of trees.
        return trees

    def child_pointer_lists(self, edge):
        """
        Return the set of child pointer lists for the given edge.
        Each child pointer list is a list of edges that have
        been used to form this edge.

        :rtype: list(list(EdgeI))
        """
        # Make a copy, in case they modify it.
        return self._edge_to_cpls.get(edge, {}).keys()

    # ////////////////////////////////////////////////////////////
    # Display
    # ////////////////////////////////////////////////////////////
    def pretty_format_edge(self, edge, width=None):
        """
        Return a pretty-printed string representation of a given edge
        in this chart.

        :rtype: str
        :param width: The number of characters allotted to each
            index in the sentence.
        """
        if width is None:
            width = 50 // (self.num_leaves() + 1)
        (start, end) = (edge.start(), edge.end())

        str = "|" + ("." + " " * (width - 1)) * start

        # Zero-width edges are "#" if complete, ">" if incomplete
        if start == end:
            if edge.is_complete():
                str += "#"
            else:
                str += ">"

        # Spanning complete edges are "[===]"; Other edges are
        # "[---]" if complete, "[--->" if incomplete
        elif edge.is_complete() and edge.span() == (0, self._num_leaves):
            str += "[" + ("=" * width) * (end - start - 1) + "=" * (width - 1) + "]"
        elif edge.is_complete():
            str += "[" + ("-" * width) * (end - start - 1) + "-" * (width - 1) + "]"
        else:
            str += "[" + ("-" * width) * (end - start - 1) + "-" * (width - 1) + ">"

        str += (" " * (width - 1) + ".") * (self._num_leaves - end)
        return str + "| %s" % edge

    def pretty_format_leaves(self, width=None):
        """
        Return a pretty-printed string representation of this
        chart's leaves.  This string can be used as a header
        for calls to ``pretty_format_edge``.
        """
        if width is None:
            width = 50 // (self.num_leaves() + 1)

        if self._tokens is not None and width > 1:
            header = "|."
            for tok in self._tokens:
                header += tok[: width - 1].center(width - 1) + "."
            header += "|"
        else:
            header = ""

        return header

    def pretty_format(self, width=None):
        """
        Return a pretty-printed string representation of this chart.

        :param width: The number of characters allotted to each
            index in the sentence.
        :rtype: str
        """
        if width is None:
            width = 50 // (self.num_leaves() + 1)
        # sort edges: primary key=length, secondary key=start index.
        # (and filter out the token edges)
        edges = sorted((e.length(), e.start(), e) for e in self)
        edges = [e for (_, _, e) in edges]

        return (
            self.pretty_format_leaves(width)
            + "\n"
            + "\n".join(self.pretty_format_edge(edge, width) for edge in edges)
        )

    # ////////////////////////////////////////////////////////////
    # Display: Dot (AT&T Graphviz)
    # ////////////////////////////////////////////////////////////

    def dot_digraph(self):
        # Header
        s = "digraph nltk_chart {\n"
        # s += '  size="5,5";\n'
        s += "  rankdir=LR;\n"
        s += "  node [height=0.1,width=0.1];\n"
        s += '  node [style=filled, color="lightgray"];\n'

        # Set up the nodes
        for y in range(self.num_edges(), -1, -1):
            if y == 0:
                s += '  node [style=filled, color="black"];\n'
            for x in range(self.num_leaves() + 1):
                if y == 0 or (
                    x <= self._edges[y - 1].start() or x >= self._edges[y - 1].end()
                ):
                    s += '  %04d.%04d [label=""];\n' % (x, y)

        # Add a spacer
        s += "  x [style=invis]; x->0000.0000 [style=invis];\n"

        # Declare ranks.
        for x in range(self.num_leaves() + 1):
            s += "  {rank=same;"
            for y in range(self.num_edges() + 1):
                if y == 0 or (
                    x <= self._edges[y - 1].start() or x >= self._edges[y - 1].end()
                ):
                    s += " %04d.%04d" % (x, y)
            s += "}\n"

        # Add the leaves
        s += "  edge [style=invis, weight=100];\n"
        s += "  node [shape=plaintext]\n"
        s += "  0000.0000"
        for x in range(self.num_leaves()):
            s += "->%s->%04d.0000" % (self.leaf(x), x + 1)
        s += ";\n\n"

        # Add the edges
        s += "  edge [style=solid, weight=1];\n"
        for y, edge in enumerate(self):
            for x in range(edge.start()):
                s += '  %04d.%04d -> %04d.%04d [style="invis"];\n' % (
                    x,
                    y + 1,
                    x + 1,
                    y + 1,
                )
            s += '  %04d.%04d -> %04d.%04d [label="%s"];\n' % (
                edge.start(),
                y + 1,
                edge.end(),
                y + 1,
                edge,
            )
            for x in range(edge.end(), self.num_leaves()):
                s += '  %04d.%04d -> %04d.%04d [style="invis"];\n' % (
                    x,
                    y + 1,
                    x + 1,
                    y + 1,
                )
        s += "}\n"
        return s


########################################################################
##  Chart Rules
########################################################################


class ChartRuleI:
    """
    A rule that specifies what new edges are licensed by any given set
    of existing edges.  Each chart rule expects a fixed number of
    edges, as indicated by the class variable ``NUM_EDGES``.  In
    particular:

    - A chart rule with ``NUM_EDGES=0`` specifies what new edges are
      licensed, regardless of existing edges.
    - A chart rule with ``NUM_EDGES=1`` specifies what new edges are
      licensed by a single existing edge.
    - A chart rule with ``NUM_EDGES=2`` specifies what new edges are
      licensed by a pair of existing edges.

    :type NUM_EDGES: int
    :cvar NUM_EDGES: The number of existing edges that this rule uses
        to license new edges.  Typically, this number ranges from zero
        to two.
    """

    def apply(self, chart, grammar, *edges):
        """
        Return a generator that will add edges licensed by this rule
        and the given edges to the chart, one at a time.  Each
        time the generator is resumed, it will either add a new
        edge and yield that edge; or return.

        :type edges: list(EdgeI)
        :param edges: A set of existing edges.  The number of edges
            that should be passed to ``apply()`` is specified by the
            ``NUM_EDGES`` class variable.
        :rtype: iter(EdgeI)
        """
        raise NotImplementedError()

    def apply_everywhere(self, chart, grammar):
        """
        Return a generator that will add all edges licensed by
        this rule, given the edges that are currently in the
        chart, one at a time.  Each time the generator is resumed,
        it will either add a new edge and yield that edge; or return.

        :rtype: iter(EdgeI)
        """
        raise NotImplementedError()


class AbstractChartRule(ChartRuleI):
    """
    An abstract base class for chart rules.  ``AbstractChartRule``
    provides:

    - A default implementation for ``apply``.
    - A default implementation for ``apply_everywhere``,
      (Currently, this implementation assumes that ``NUM_EDGES <= 3``.)
    - A default implementation for ``__str__``, which returns a
      name based on the rule's class name.
    """

    # Subclasses must define apply.
    def apply(self, chart, grammar, *edges):
        raise NotImplementedError()

    # Default: loop through the given number of edges, and call
    # self.apply() for each set of edges.
    def apply_everywhere(self, chart, grammar):
        if self.NUM_EDGES == 0:
            yield from self.apply(chart, grammar)

        elif self.NUM_EDGES == 1:
            for e1 in chart:
                yield from self.apply(chart, grammar, e1)

        elif self.NUM_EDGES == 2:
            for e1 in chart:
                for e2 in chart:
                    yield from self.apply(chart, grammar, e1, e2)

        elif self.NUM_EDGES == 3:
            for e1 in chart:
                for e2 in chart:
                    for e3 in chart:
                        yield from self.apply(chart, grammar, e1, e2, e3)

        else:
            raise AssertionError("NUM_EDGES>3 is not currently supported")

    # Default: return a name based on the class name.
    def __str__(self):
        # Add spaces between InitialCapsWords.
        return re.sub("([a-z])([A-Z])", r"\1 \2", self.__class__.__name__)


# ////////////////////////////////////////////////////////////
# Fundamental Rule
# ////////////////////////////////////////////////////////////


class FundamentalRule(AbstractChartRule):
    r"""
    A rule that joins two adjacent edges to form a single combined
    edge.  In particular, this rule specifies that any pair of edges

    - ``[A -> alpha \* B beta][i:j]``
    - ``[B -> gamma \*][j:k]``

    licenses the edge:

    - ``[A -> alpha B * beta][i:j]``
    """

    NUM_EDGES = 2

    def apply(self, chart, grammar, left_edge, right_edge):
        # Make sure the rule is applicable.
        if not (
            left_edge.is_incomplete()
            and right_edge.is_complete()
            and left_edge.end() == right_edge.start()
            and left_edge.nextsym() == right_edge.lhs()
        ):
            return

        # Construct the new edge.
        new_edge = left_edge.move_dot_forward(right_edge.end())

        # Insert it into the chart.
        if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
            yield new_edge


class SingleEdgeFundamentalRule(FundamentalRule):
    r"""
    A rule that joins a given edge with adjacent edges in the chart,
    to form combined edges.  In particular, this rule specifies that
    either of the edges:

    - ``[A -> alpha \* B beta][i:j]``
    - ``[B -> gamma \*][j:k]``

    licenses the edge:

    - ``[A -> alpha B * beta][i:j]``

    if the other edge is already in the chart.

    :note: This is basically ``FundamentalRule``, with one edge left
        unspecified.
    """

    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            yield from self._apply_incomplete(chart, grammar, edge)
        else:
            yield from self._apply_complete(chart, grammar, edge)

    def _apply_complete(self, chart, grammar, right_edge):
        for left_edge in chart.select(
            end=right_edge.start(), is_complete=False, nextsym=right_edge.lhs()
        ):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge

    def _apply_incomplete(self, chart, grammar, left_edge):
        for right_edge in chart.select(
            start=left_edge.end(), is_complete=True, lhs=left_edge.nextsym()
        ):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge


# ////////////////////////////////////////////////////////////
# Inserting Terminal Leafs
# ////////////////////////////////////////////////////////////


class LeafInitRule(AbstractChartRule):
    NUM_EDGES = 0

    def apply(self, chart, grammar):
        for index in range(chart.num_leaves()):
            new_edge = LeafEdge(chart.leaf(index), index)
            if chart.insert(new_edge, ()):
                yield new_edge


# ////////////////////////////////////////////////////////////
# Top-Down Prediction
# ////////////////////////////////////////////////////////////


class TopDownInitRule(AbstractChartRule):
    r"""
    A rule licensing edges corresponding to the grammar productions for
    the grammar's start symbol.  In particular, this rule specifies that
    ``[S -> \* alpha][0:i]`` is licensed for each grammar production
    ``S -> alpha``, where ``S`` is the grammar's start symbol.
    """

    NUM_EDGES = 0

    def apply(self, chart, grammar):
        for prod in grammar.productions(lhs=grammar.start()):
            new_edge = TreeEdge.from_production(prod, 0)
            if chart.insert(new_edge, ()):
                yield new_edge


class TopDownPredictRule(AbstractChartRule):
    r"""
    A rule licensing edges corresponding to the grammar productions
    for the nonterminal following an incomplete edge's dot.  In
    particular, this rule specifies that
    ``[A -> alpha \* B beta][i:j]`` licenses the edge
    ``[B -> \* gamma][j:j]`` for each grammar production ``B -> gamma``.

    :note: This rule corresponds to the Predictor Rule in Earley parsing.
    """

    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_complete():
            return
        for prod in grammar.productions(lhs=edge.nextsym()):
            new_edge = TreeEdge.from_production(prod, edge.end())
            if chart.insert(new_edge, ()):
                yield new_edge


class CachedTopDownPredictRule(TopDownPredictRule):
    r"""
    A cached version of ``TopDownPredictRule``.  After the first time
    this rule is applied to an edge with a given ``end`` and ``next``,
    it will not generate any more edges for edges with that ``end`` and
    ``next``.

    If ``chart`` or ``grammar`` are changed, then the cache is flushed.
    """

    def __init__(self):
        TopDownPredictRule.__init__(self)
        self._done = {}

    def apply(self, chart, grammar, edge):
        if edge.is_complete():
            return
        nextsym, index = edge.nextsym(), edge.end()
        if not is_nonterminal(nextsym):
            return

        # If we've already applied this rule to an edge with the same
        # next & end, and the chart & grammar have not changed, then
        # just return (no new edges to add).
        done = self._done.get((nextsym, index), (None, None))
        if done[0] is chart and done[1] is grammar:
            return

        # Add all the edges indicated by the top down expand rule.
        for prod in grammar.productions(lhs=nextsym):
            # If the left corner in the predicted production is
            # leaf, it must match with the input.
            if prod.rhs():
                first = prod.rhs()[0]
                if is_terminal(first):
                    if index >= chart.num_leaves() or first != chart.leaf(index):
                        continue

            new_edge = TreeEdge.from_production(prod, index)
            if chart.insert(new_edge, ()):
                yield new_edge

        # Record the fact that we've applied this rule.
        self._done[nextsym, index] = (chart, grammar)


# ////////////////////////////////////////////////////////////
# Bottom-Up Prediction
# ////////////////////////////////////////////////////////////


class BottomUpPredictRule(AbstractChartRule):
    r"""
    A rule licensing any edge corresponding to a production whose
    right-hand side begins with a complete edge's left-hand side.  In
    particular, this rule specifies that ``[A -> alpha \*]`` licenses
    the edge ``[B -> \* A beta]`` for each grammar production ``B -> A beta``.
    """

    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return
        for prod in grammar.productions(rhs=edge.lhs()):
            new_edge = TreeEdge.from_production(prod, edge.start())
            if chart.insert(new_edge, ()):
                yield new_edge


class BottomUpPredictCombineRule(BottomUpPredictRule):
    r"""
    A rule licensing any edge corresponding to a production whose
    right-hand side begins with a complete edge's left-hand side.  In
    particular, this rule specifies that ``[A -> alpha \*]``
    licenses the edge ``[B -> A \* beta]`` for each grammar
    production ``B -> A beta``.

    :note: This is like ``BottomUpPredictRule``, but it also applies
        the ``FundamentalRule`` to the resulting edge.
    """

    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return
        for prod in grammar.productions(rhs=edge.lhs()):
            new_edge = TreeEdge(edge.span(), prod.lhs(), prod.rhs(), 1)
            if chart.insert(new_edge, (edge,)):
                yield new_edge


class EmptyPredictRule(AbstractChartRule):
    """
    A rule that inserts all empty productions as passive edges,
    in every position in the chart.
    """

    NUM_EDGES = 0

    def apply(self, chart, grammar):
        for prod in grammar.productions(empty=True):
            for index in range(chart.num_leaves() + 1):
                new_edge = TreeEdge.from_production(prod, index)
                if chart.insert(new_edge, ()):
                    yield new_edge


########################################################################
##  Filtered Bottom Up
########################################################################


class FilteredSingleEdgeFundamentalRule(SingleEdgeFundamentalRule):
    def _apply_complete(self, chart, grammar, right_edge):
        end = right_edge.end()
        nexttoken = end < chart.num_leaves() and chart.leaf(end)
        for left_edge in chart.select(
            end=right_edge.start(), is_complete=False, nextsym=right_edge.lhs()
        ):
            if _bottomup_filter(grammar, nexttoken, left_edge.rhs(), left_edge.dot()):
                new_edge = left_edge.move_dot_forward(right_edge.end())
                if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                    yield new_edge

    def _apply_incomplete(self, chart, grammar, left_edge):
        for right_edge in chart.select(
            start=left_edge.end(), is_complete=True, lhs=left_edge.nextsym()
        ):
            end = right_edge.end()
            nexttoken = end < chart.num_leaves() and chart.leaf(end)
            if _bottomup_filter(grammar, nexttoken, left_edge.rhs(), left_edge.dot()):
                new_edge = left_edge.move_dot_forward(right_edge.end())
                if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                    yield new_edge


class FilteredBottomUpPredictCombineRule(BottomUpPredictCombineRule):
    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return

        end = edge.end()
        nexttoken = end < chart.num_leaves() and chart.leaf(end)
        for prod in grammar.productions(rhs=edge.lhs()):
            if _bottomup_filter(grammar, nexttoken, prod.rhs()):
                new_edge = TreeEdge(edge.span(), prod.lhs(), prod.rhs(), 1)
                if chart.insert(new_edge, (edge,)):
                    yield new_edge


def _bottomup_filter(grammar, nexttoken, rhs, dot=0):
    if len(rhs) <= dot + 1:
        return True
    _next = rhs[dot + 1]
    if is_terminal(_next):
        return nexttoken == _next
    else:
        return grammar.is_leftcorner(_next, nexttoken)


########################################################################
##  Generic Chart Parser
########################################################################

TD_STRATEGY = [
    LeafInitRule(),
    TopDownInitRule(),
    CachedTopDownPredictRule(),
    SingleEdgeFundamentalRule(),
]
BU_STRATEGY = [
    LeafInitRule(),
    EmptyPredictRule(),
    BottomUpPredictRule(),
    SingleEdgeFundamentalRule(),
]
BU_LC_STRATEGY = [
    LeafInitRule(),
    EmptyPredictRule(),
    BottomUpPredictCombineRule(),
    SingleEdgeFundamentalRule(),
]

LC_STRATEGY = [
    LeafInitRule(),
    FilteredBottomUpPredictCombineRule(),
    FilteredSingleEdgeFundamentalRule(),
]


class ChartParser(ParserI):
    """
    A generic chart parser.  A "strategy", or list of
    ``ChartRuleI`` instances, is used to decide what edges to add to
    the chart.  In particular, ``ChartParser`` uses the following
    algorithm to parse texts:

    | Until no new edges are added:
    |   For each *rule* in *strategy*:
    |     Apply *rule* to any applicable edges in the chart.
    | Return any complete parses in the chart
    """

    def __init__(
        self,
        grammar,
        strategy=BU_LC_STRATEGY,
        trace=0,
        trace_chart_width=50,
        use_agenda=True,
        chart_class=Chart,
    ):
        """
        Create a new chart parser, that uses ``grammar`` to parse
        texts.

        :type grammar: CFG
        :param grammar: The grammar used to parse texts.
        :type strategy: list(ChartRuleI)
        :param strategy: A list of rules that should be used to decide
            what edges to add to the chart (top-down strategy by default).
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        :type trace_chart_width: int
        :param trace_chart_width: The default total width reserved for
            the chart in trace output.  The remainder of each line will
            be used to display edges.
        :type use_agenda: bool
        :param use_agenda: Use an optimized agenda-based algorithm,
            if possible.
        :param chart_class: The class that should be used to create
            the parse charts.
        """
        self._grammar = grammar
        self._strategy = strategy
        self._trace = trace
        self._trace_chart_width = trace_chart_width
        # If the strategy only consists of axioms (NUM_EDGES==0) and
        # inference rules (NUM_EDGES==1), we can use an agenda-based algorithm:
        self._use_agenda = use_agenda
        self._chart_class = chart_class

        self._axioms = []
        self._inference_rules = []
        for rule in strategy:
            if rule.NUM_EDGES == 0:
                self._axioms.append(rule)
            elif rule.NUM_EDGES == 1:
                self._inference_rules.append(rule)
            else:
                self._use_agenda = False

    def grammar(self):
        return self._grammar

    def _trace_new_edges(self, chart, rule, new_edges, trace, edge_width):
        if not trace:
            return
        print_rule_header = trace > 1
        for edge in new_edges:
            if print_rule_header:
                print("%s:" % rule)
                print_rule_header = False
            print(chart.pretty_format_edge(edge, edge_width))

    def chart_parse(self, tokens, trace=None):
        """
        Return the final parse ``Chart`` from which all possible
        parse trees can be extracted.

        :param tokens: The sentence to be parsed
        :type tokens: list(str)
        :rtype: Chart
        """
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

        if self._use_agenda:
            # Use an agenda-based algorithm.
            for axiom in self._axioms:
                new_edges = list(axiom.apply(chart, grammar))
                trace_new_edges(chart, axiom, new_edges, trace, trace_edge_width)

            inference_rules = self._inference_rules
            agenda = chart.edges()
            # We reverse the initial agenda, since it is a stack
            # but chart.edges() functions as a queue.
            agenda.reverse()
            while agenda:
                edge = agenda.pop()
                for rule in inference_rules:
                    new_edges = list(rule.apply(chart, grammar, edge))
                    if trace:
                        trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)
                    agenda += new_edges

        else:
            # Do not use an agenda-based algorithm.
            edges_added = True
            while edges_added:
                edges_added = False
                for rule in self._strategy:
                    new_edges = list(rule.apply_everywhere(chart, grammar))
                    edges_added = len(new_edges)
                    trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)

        # Return the final chart.
        return chart

    def parse(self, tokens, tree_class=Tree):
        chart = self.chart_parse(tokens)
        return iter(chart.parses(self._grammar.start(), tree_class=tree_class))


class TopDownChartParser(ChartParser):
    """
    A ``ChartParser`` using a top-down parsing strategy.
    See ``ChartParser`` for more information.
    """

    def __init__(self, grammar, **parser_args):
        ChartParser.__init__(self, grammar, TD_STRATEGY, **parser_args)


class BottomUpChartParser(ChartParser):
    """
    A ``ChartParser`` using a bottom-up parsing strategy.
    See ``ChartParser`` for more information.
    """

    def __init__(self, grammar, **parser_args):
        if isinstance(grammar, PCFG):
            warnings.warn(
                "BottomUpChartParser only works for CFG, "
                "use BottomUpProbabilisticChartParser instead",
                category=DeprecationWarning,
            )
        ChartParser.__init__(self, grammar, BU_STRATEGY, **parser_args)


class BottomUpLeftCornerChartParser(ChartParser):
    """
    A ``ChartParser`` using a bottom-up left-corner parsing strategy.
    This strategy is often more efficient than standard bottom-up.
    See ``ChartParser`` for more information.
    """

    def __init__(self, grammar, **parser_args):
        ChartParser.__init__(self, grammar, BU_LC_STRATEGY, **parser_args)


class LeftCornerChartParser(ChartParser):
    def __init__(self, grammar, **parser_args):
        if not grammar.is_nonempty():
            raise ValueError(
                "LeftCornerParser only works for grammars " "without empty productions."
            )
        ChartParser.__init__(self, grammar, LC_STRATEGY, **parser_args)


########################################################################
##  Stepping Chart Parser
########################################################################


class SteppingChartParser(ChartParser):
    """
    A ``ChartParser`` that allows you to step through the parsing
    process, adding a single edge at a time.  It also allows you to
    change the parser's strategy or grammar midway through parsing a
    text.

    The ``initialize`` method is used to start parsing a text.  ``step``
    adds a single edge to the chart.  ``set_strategy`` changes the
    strategy used by the chart parser.  ``parses`` returns the set of
    parses that has been found by the chart parser.

    :ivar _restart: Records whether the parser's strategy, grammar,
        or chart has been changed.  If so, then ``step`` must restart
        the parsing algorithm.
    """

    def __init__(self, grammar, strategy=[], trace=0):
        self._chart = None
        self._current_chartrule = None
        self._restart = False
        ChartParser.__init__(self, grammar, strategy, trace)

    # ////////////////////////////////////////////////////////////
    # Initialization
    # ////////////////////////////////////////////////////////////

    def initialize(self, tokens):
        "Begin parsing the given tokens."
        self._chart = Chart(list(tokens))
        self._restart = True

    # ////////////////////////////////////////////////////////////
    # Stepping
    # ////////////////////////////////////////////////////////////

    def step(self):
        """
        Return a generator that adds edges to the chart, one at a
        time.  Each time the generator is resumed, it adds a single
        edge and yields that edge.  If no more edges can be added,
        then it yields None.

        If the parser's strategy, grammar, or chart is changed, then
        the generator will continue adding edges using the new
        strategy, grammar, or chart.

        Note that this generator never terminates, since the grammar
        or strategy might be changed to values that would add new
        edges.  Instead, it yields None when no more edges can be
        added with the current strategy and grammar.
        """
        if self._chart is None:
            raise ValueError("Parser must be initialized first")
        while True:
            self._restart = False
            w = 50 // (self._chart.num_leaves() + 1)

            for e in self._parse():
                if self._trace > 1:
                    print(self._current_chartrule)
                if self._trace > 0:
                    print(self._chart.pretty_format_edge(e, w))
                yield e
                if self._restart:
                    break
            else:
                yield None  # No more edges.

    def _parse(self):
        """
        A generator that implements the actual parsing algorithm.
        ``step`` iterates through this generator, and restarts it
        whenever the parser's strategy, grammar, or chart is modified.
        """
        chart = self._chart
        grammar = self._grammar
        edges_added = 1
        while edges_added > 0:
            edges_added = 0
            for rule in self._strategy:
                self._current_chartrule = rule
                for e in rule.apply_everywhere(chart, grammar):
                    edges_added += 1
                    yield e

    # ////////////////////////////////////////////////////////////
    # Accessors
    # ////////////////////////////////////////////////////////////

    def strategy(self):
        "Return the strategy used by this parser."
        return self._strategy

    def grammar(self):
        "Return the grammar used by this parser."
        return self._grammar

    def chart(self):
        "Return the chart that is used by this parser."
        return self._chart

    def current_chartrule(self):
        "Return the chart rule used to generate the most recent edge."
        return self._current_chartrule

    def parses(self, tree_class=Tree):
        "Return the parse trees currently contained in the chart."
        return self._chart.parses(self._grammar.start(), tree_class)

    # ////////////////////////////////////////////////////////////
    # Parser modification
    # ////////////////////////////////////////////////////////////

    def set_strategy(self, strategy):
        """
        Change the strategy that the parser uses to decide which edges
        to add to the chart.

        :type strategy: list(ChartRuleI)
        :param strategy: A list of rules that should be used to decide
            what edges to add to the chart.
        """
        if strategy == self._strategy:
            return
        self._strategy = strategy[:]  # Make a copy.
        self._restart = True

    def set_grammar(self, grammar):
        "Change the grammar used by the parser."
        if grammar is self._grammar:
            return
        self._grammar = grammar
        self._restart = True

    def set_chart(self, chart):
        "Load a given chart into the chart parser."
        if chart is self._chart:
            return
        self._chart = chart
        self._restart = True

    # ////////////////////////////////////////////////////////////
    # Standard parser methods
    # ////////////////////////////////////////////////////////////

    def parse(self, tokens, tree_class=Tree):
        tokens = list(tokens)
        self._grammar.check_coverage(tokens)

        # Initialize ourselves.
        self.initialize(tokens)

        # Step until no more edges are generated.
        for e in self.step():
            if e is None:
                break

        # Return an iterator of complete parses.
        return self.parses(tree_class=tree_class)


########################################################################
##  Demo Code
########################################################################


def demo_grammar():
    from nltk.grammar import CFG

    return CFG.fromstring(
        """
S  -> NP VP
PP -> "with" NP
NP -> NP PP
VP -> VP PP
VP -> Verb NP
VP -> Verb
NP -> Det Noun
NP -> "John"
NP -> "I"
Det -> "the"
Det -> "my"
Det -> "a"
Noun -> "dog"
Noun -> "cookie"
Verb -> "ate"
Verb -> "saw"
Prep -> "with"
Prep -> "under"
"""
    )


def demo(
    choice=None,
    print_times=True,
    print_grammar=False,
    print_trees=True,
    trace=2,
    sent="I saw John with a dog with my cookie",
    numparses=5,
):
    """
    A demonstration of the chart parsers.
    """
    import sys
    import time

    from nltk import CFG, Production, nonterminals

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

    # Ask the user which parser to test,
    # if the parser wasn't provided as an argument
    if choice is None:
        print("  1: Top-down chart parser")
        print("  2: Bottom-up chart parser")
        print("  3: Bottom-up left-corner chart parser")
        print("  4: Left-corner chart parser with bottom-up filter")
        print("  5: Stepping chart parser (alternating top-down & bottom-up)")
        print("  6: All parsers")
        print("\nWhich parser (1-6)? ", end=" ")
        choice = sys.stdin.readline().strip()
        print()

    choice = str(choice)
    if choice not in "123456":
        print("Bad parser number")
        return

    # Keep track of how long each parser takes.
    times = {}

    strategies = {
        "1": ("Top-down", TD_STRATEGY),
        "2": ("Bottom-up", BU_STRATEGY),
        "3": ("Bottom-up left-corner", BU_LC_STRATEGY),
        "4": ("Filtered left-corner", LC_STRATEGY),
    }
    choices = []
    if choice in strategies:
        choices = [choice]
    if choice == "6":
        choices = "1234"

    # Run the requested chart parser(s), except the stepping parser.
    for strategy in choices:
        print("* Strategy: " + strategies[strategy][0])
        print()
        cp = ChartParser(grammar, strategies[strategy][1], trace=trace)
        t = time.time()
        chart = cp.chart_parse(tokens)
        parses = list(chart.parses(grammar.start()))

        times[strategies[strategy][0]] = time.time() - t
        print("Nr edges in chart:", len(chart.edges()))
        if numparses:
            assert len(parses) == numparses, "Not all parses found"
        if print_trees:
            for tree in parses:
                print(tree)
        else:
            print("Nr trees:", len(parses))
        print()

    # Run the stepping parser, if requested.
    if choice in "56":
        print("* Strategy: Stepping (top-down vs bottom-up)")
        print()
        t = time.time()
        cp = SteppingChartParser(grammar, trace=trace)
        cp.initialize(tokens)
        for i in range(5):
            print("*** SWITCH TO TOP DOWN")
            cp.set_strategy(TD_STRATEGY)
            for j, e in enumerate(cp.step()):
                if j > 20 or e is None:
                    break
            print("*** SWITCH TO BOTTOM UP")
            cp.set_strategy(BU_STRATEGY)
            for j, e in enumerate(cp.step()):
                if j > 20 or e is None:
                    break
        times["Stepping"] = time.time() - t
        print("Nr edges in chart:", len(cp.chart().edges()))
        if numparses:
            assert len(list(cp.parses())) == numparses, "Not all parses found"
        if print_trees:
            for tree in cp.parses():
                print(tree)
        else:
            print("Nr trees:", len(list(cp.parses())))
        print()

    # Print the times of all parsers:
    if not (print_times and times):
        return
    print("* Parsing times")
    print()
    maxlen = max(len(key) for key in times)
    format = "%" + repr(maxlen) + "s parser: %6.3fsec"
    times_items = times.items()
    for (parser, t) in sorted(times_items, key=lambda a: a[1]):
        print(format % (parser, t))


if __name__ == "__main__":
    demo()
