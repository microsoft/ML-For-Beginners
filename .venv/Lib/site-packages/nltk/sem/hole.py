# Natural Language Toolkit: Logic
#
# Author:     Peter Wang
# Updated by: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
An implementation of the Hole Semantics model, following Blackburn and Bos,
Representation and Inference for Natural Language (CSLI, 2005).

The semantic representations are built by the grammar hole.fcfg.
This module contains driver code to read in sentences and parse them
according to a hole semantics grammar.

After parsing, the semantic representation is in the form of an underspecified
representation that is not easy to read.  We use a "plugging" algorithm to
convert that representation into first-order logic formulas.
"""

from functools import reduce

from nltk.parse import load_parser
from nltk.sem.logic import (
    AllExpression,
    AndExpression,
    ApplicationExpression,
    ExistsExpression,
    IffExpression,
    ImpExpression,
    LambdaExpression,
    NegatedExpression,
    OrExpression,
)
from nltk.sem.skolemize import skolemize

# Note that in this code there may be multiple types of trees being referred to:
#
# 1. parse trees
# 2. the underspecified representation
# 3. first-order logic formula trees
# 4. the search space when plugging (search tree)
#


class Constants:
    ALL = "ALL"
    EXISTS = "EXISTS"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    IMP = "IMP"
    IFF = "IFF"
    PRED = "PRED"
    LEQ = "LEQ"
    HOLE = "HOLE"
    LABEL = "LABEL"

    MAP = {
        ALL: lambda v, e: AllExpression(v.variable, e),
        EXISTS: lambda v, e: ExistsExpression(v.variable, e),
        NOT: NegatedExpression,
        AND: AndExpression,
        OR: OrExpression,
        IMP: ImpExpression,
        IFF: IffExpression,
        PRED: ApplicationExpression,
    }


class HoleSemantics:
    """
    This class holds the broken-down components of a hole semantics, i.e. it
    extracts the holes, labels, logic formula fragments and constraints out of
    a big conjunction of such as produced by the hole semantics grammar.  It
    then provides some operations on the semantics dealing with holes, labels
    and finding legal ways to plug holes with labels.
    """

    def __init__(self, usr):
        """
        Constructor.  `usr' is a ``sem.Expression`` representing an
        Underspecified Representation Structure (USR).  A USR has the following
        special predicates:
        ALL(l,v,n),
        EXISTS(l,v,n),
        AND(l,n,n),
        OR(l,n,n),
        IMP(l,n,n),
        IFF(l,n,n),
        PRED(l,v,n,v[,v]*) where the brackets and star indicate zero or more repetitions,
        LEQ(n,n),
        HOLE(n),
        LABEL(n)
        where l is the label of the node described by the predicate, n is either
        a label or a hole, and v is a variable.
        """
        self.holes = set()
        self.labels = set()
        self.fragments = {}  # mapping of label -> formula fragment
        self.constraints = set()  # set of Constraints
        self._break_down(usr)
        self.top_most_labels = self._find_top_most_labels()
        self.top_hole = self._find_top_hole()

    def is_node(self, x):
        """
        Return true if x is a node (label or hole) in this semantic
        representation.
        """
        return x in (self.labels | self.holes)

    def _break_down(self, usr):
        """
        Extract holes, labels, formula fragments and constraints from the hole
        semantics underspecified representation (USR).
        """
        if isinstance(usr, AndExpression):
            self._break_down(usr.first)
            self._break_down(usr.second)
        elif isinstance(usr, ApplicationExpression):
            func, args = usr.uncurry()
            if func.variable.name == Constants.LEQ:
                self.constraints.add(Constraint(args[0], args[1]))
            elif func.variable.name == Constants.HOLE:
                self.holes.add(args[0])
            elif func.variable.name == Constants.LABEL:
                self.labels.add(args[0])
            else:
                label = args[0]
                assert label not in self.fragments
                self.fragments[label] = (func, args[1:])
        else:
            raise ValueError(usr.label())

    def _find_top_nodes(self, node_list):
        top_nodes = node_list.copy()
        for f in self.fragments.values():
            # the label is the first argument of the predicate
            args = f[1]
            for arg in args:
                if arg in node_list:
                    top_nodes.discard(arg)
        return top_nodes

    def _find_top_most_labels(self):
        """
        Return the set of labels which are not referenced directly as part of
        another formula fragment.  These will be the top-most labels for the
        subtree that they are part of.
        """
        return self._find_top_nodes(self.labels)

    def _find_top_hole(self):
        """
        Return the hole that will be the top of the formula tree.
        """
        top_holes = self._find_top_nodes(self.holes)
        assert len(top_holes) == 1  # it must be unique
        return top_holes.pop()

    def pluggings(self):
        """
        Calculate and return all the legal pluggings (mappings of labels to
        holes) of this semantics given the constraints.
        """
        record = []
        self._plug_nodes([(self.top_hole, [])], self.top_most_labels, {}, record)
        return record

    def _plug_nodes(self, queue, potential_labels, plug_acc, record):
        """
        Plug the nodes in `queue' with the labels in `potential_labels'.

        Each element of `queue' is a tuple of the node to plug and the list of
        ancestor holes from the root of the graph to that node.

        `potential_labels' is a set of the labels which are still available for
        plugging.

        `plug_acc' is the incomplete mapping of holes to labels made on the
        current branch of the search tree so far.

        `record' is a list of all the complete pluggings that we have found in
        total so far.  It is the only parameter that is destructively updated.
        """
        if queue != []:
            (node, ancestors) = queue[0]
            if node in self.holes:
                # The node is a hole, try to plug it.
                self._plug_hole(
                    node, ancestors, queue[1:], potential_labels, plug_acc, record
                )
            else:
                assert node in self.labels
                # The node is a label.  Replace it in the queue by the holes and
                # labels in the formula fragment named by that label.
                args = self.fragments[node][1]
                head = [(a, ancestors) for a in args if self.is_node(a)]
                self._plug_nodes(head + queue[1:], potential_labels, plug_acc, record)
        else:
            raise Exception("queue empty")

    def _plug_hole(self, hole, ancestors0, queue, potential_labels0, plug_acc0, record):
        """
        Try all possible ways of plugging a single hole.
        See _plug_nodes for the meanings of the parameters.
        """
        # Add the current hole we're trying to plug into the list of ancestors.
        assert hole not in ancestors0
        ancestors = [hole] + ancestors0

        # Try each potential label in this hole in turn.
        for l in potential_labels0:
            # Is the label valid in this hole?
            if self._violates_constraints(l, ancestors):
                continue

            plug_acc = plug_acc0.copy()
            plug_acc[hole] = l
            potential_labels = potential_labels0.copy()
            potential_labels.remove(l)

            if len(potential_labels) == 0:
                # No more potential labels.  That must mean all the holes have
                # been filled so we have found a legal plugging so remember it.
                #
                # Note that the queue might not be empty because there might
                # be labels on there that point to formula fragments with
                # no holes in them.  _sanity_check_plugging will make sure
                # all holes are filled.
                self._sanity_check_plugging(plug_acc, self.top_hole, [])
                record.append(plug_acc)
            else:
                # Recursively try to fill in the rest of the holes in the
                # queue.  The label we just plugged into the hole could have
                # holes of its own so at the end of the queue.  Putting it on
                # the end of the queue gives us a breadth-first search, so that
                # all the holes at level i of the formula tree are filled
                # before filling level i+1.
                # A depth-first search would work as well since the trees must
                # be finite but the bookkeeping would be harder.
                self._plug_nodes(
                    queue + [(l, ancestors)], potential_labels, plug_acc, record
                )

    def _violates_constraints(self, label, ancestors):
        """
        Return True if the `label' cannot be placed underneath the holes given
        by the set `ancestors' because it would violate the constraints imposed
        on it.
        """
        for c in self.constraints:
            if c.lhs == label:
                if c.rhs not in ancestors:
                    return True
        return False

    def _sanity_check_plugging(self, plugging, node, ancestors):
        """
        Make sure that a given plugging is legal.  We recursively go through
        each node and make sure that no constraints are violated.
        We also check that all holes have been filled.
        """
        if node in self.holes:
            ancestors = [node] + ancestors
            label = plugging[node]
        else:
            label = node
        assert label in self.labels
        for c in self.constraints:
            if c.lhs == label:
                assert c.rhs in ancestors
        args = self.fragments[label][1]
        for arg in args:
            if self.is_node(arg):
                self._sanity_check_plugging(plugging, arg, [label] + ancestors)

    def formula_tree(self, plugging):
        """
        Return the first-order logic formula tree for this underspecified
        representation using the plugging given.
        """
        return self._formula_tree(plugging, self.top_hole)

    def _formula_tree(self, plugging, node):
        if node in plugging:
            return self._formula_tree(plugging, plugging[node])
        elif node in self.fragments:
            pred, args = self.fragments[node]
            children = [self._formula_tree(plugging, arg) for arg in args]
            return reduce(Constants.MAP[pred.variable.name], children)
        else:
            return node


class Constraint:
    """
    This class represents a constraint of the form (L =< N),
    where L is a label and N is a node (a label or a hole).
    """

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return f"({self.lhs} < {self.rhs})"


def hole_readings(sentence, grammar_filename=None, verbose=False):
    if not grammar_filename:
        grammar_filename = "grammars/sample_grammars/hole.fcfg"

    if verbose:
        print("Reading grammar file", grammar_filename)

    parser = load_parser(grammar_filename)

    # Parse the sentence.
    tokens = sentence.split()
    trees = list(parser.parse(tokens))
    if verbose:
        print("Got %d different parses" % len(trees))

    all_readings = []
    for tree in trees:
        # Get the semantic feature from the top of the parse tree.
        sem = tree.label()["SEM"].simplify()

        # Print the raw semantic representation.
        if verbose:
            print("Raw:       ", sem)

        # Skolemize away all quantifiers.  All variables become unique.
        while isinstance(sem, LambdaExpression):
            sem = sem.term
        skolemized = skolemize(sem)

        if verbose:
            print("Skolemized:", skolemized)

        # Break the hole semantics representation down into its components
        # i.e. holes, labels, formula fragments and constraints.
        hole_sem = HoleSemantics(skolemized)

        # Maybe show the details of the semantic representation.
        if verbose:
            print("Holes:       ", hole_sem.holes)
            print("Labels:      ", hole_sem.labels)
            print("Constraints: ", hole_sem.constraints)
            print("Top hole:    ", hole_sem.top_hole)
            print("Top labels:  ", hole_sem.top_most_labels)
            print("Fragments:")
            for l, f in hole_sem.fragments.items():
                print(f"\t{l}: {f}")

        # Find all the possible ways to plug the formulas together.
        pluggings = hole_sem.pluggings()

        # Build FOL formula trees using the pluggings.
        readings = list(map(hole_sem.formula_tree, pluggings))

        # Print out the formulas in a textual format.
        if verbose:
            for i, r in enumerate(readings):
                print()
                print("%d. %s" % (i, r))
            print()

        all_readings.extend(readings)

    return all_readings


if __name__ == "__main__":
    for r in hole_readings("a dog barks"):
        print(r)
    print()
    for r in hole_readings("every girl chases a dog"):
        print(r)
