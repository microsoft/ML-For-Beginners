# Natural Language Toolkit: ASCII visualization of NLTK trees
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Andreas van Cranenburgh <A.W.vanCranenburgh@uva.nl>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Pretty-printing of discontinuous trees.
Adapted from the disco-dop project, by Andreas van Cranenburgh.
https://github.com/andreasvc/disco-dop

Interesting reference (not used for this code):
T. Eschbach et al., Orth. Hypergraph Drawing, Journal of
Graph Algorithms and Applications, 10(2) 141--157 (2006)149.
https://jgaa.info/accepted/2006/EschbachGuentherBecker2006.10.2.pdf
"""

from nltk.internals import Deprecated
from nltk.tree.prettyprinter import TreePrettyPrinter as TPP


class TreePrettyPrinter(Deprecated, TPP):
    """Import `TreePrettyPrinter` using `from nltk.tree import TreePrettyPrinter` instead."""


__all__ = ["TreePrettyPrinter"]
