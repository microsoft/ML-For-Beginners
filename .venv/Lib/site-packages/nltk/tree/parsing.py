# Natural Language Toolkit: Text Trees
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
#         Tom Aarsen <>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import re

from nltk.tree.tree import Tree

######################################################################
## Parsing
######################################################################


def bracket_parse(s):
    """
    Use Tree.read(s, remove_empty_top_bracketing=True) instead.
    """
    raise NameError("Use Tree.read(s, remove_empty_top_bracketing=True) instead.")


def sinica_parse(s):
    """
    Parse a Sinica Treebank string and return a tree.  Trees are represented as nested brackettings,
    as shown in the following example (X represents a Chinese character):
    S(goal:NP(Head:Nep:XX)|theme:NP(Head:Nhaa:X)|quantity:Dab:X|Head:VL2:X)#0(PERIODCATEGORY)

    :return: A tree corresponding to the string representation.
    :rtype: Tree
    :param s: The string to be converted
    :type s: str
    """
    tokens = re.split(r"([()| ])", s)
    for i in range(len(tokens)):
        if tokens[i] == "(":
            tokens[i - 1], tokens[i] = (
                tokens[i],
                tokens[i - 1],
            )  # pull nonterminal inside parens
        elif ":" in tokens[i]:
            fields = tokens[i].split(":")
            if len(fields) == 2:  # non-terminal
                tokens[i] = fields[1]
            else:
                tokens[i] = "(" + fields[-2] + " " + fields[-1] + ")"
        elif tokens[i] == "|":
            tokens[i] = ""

    treebank_string = " ".join(tokens)
    return Tree.fromstring(treebank_string, remove_empty_top_bracketing=True)


#    s = re.sub(r'^#[^\s]*\s', '', s)  # remove leading identifier
#    s = re.sub(r'\w+:', '', s)       # remove role tags

#    return s

__all__ = [
    "bracket_parse",
    "sinica_parse",
]
