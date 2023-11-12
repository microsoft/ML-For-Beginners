# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
#         Tom Aarsen <>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
NLTK Tree Package

This package may be used for representing hierarchical language
structures, such as syntax trees and morphological trees.
"""

# TODO: add LabelledTree (can be used for dependency trees)

from nltk.tree.immutable import (
    ImmutableMultiParentedTree,
    ImmutableParentedTree,
    ImmutableProbabilisticTree,
    ImmutableTree,
)
from nltk.tree.parented import MultiParentedTree, ParentedTree
from nltk.tree.parsing import bracket_parse, sinica_parse
from nltk.tree.prettyprinter import TreePrettyPrinter
from nltk.tree.probabilistic import ProbabilisticTree
from nltk.tree.transforms import (
    chomsky_normal_form,
    collapse_unary,
    un_chomsky_normal_form,
)
from nltk.tree.tree import Tree

__all__ = [
    "ImmutableMultiParentedTree",
    "ImmutableParentedTree",
    "ImmutableProbabilisticTree",
    "ImmutableTree",
    "MultiParentedTree",
    "ParentedTree",
    "bracket_parse",
    "sinica_parse",
    "TreePrettyPrinter",
    "ProbabilisticTree",
    "chomsky_normal_form",
    "collapse_unary",
    "un_chomsky_normal_form",
    "Tree",
]
