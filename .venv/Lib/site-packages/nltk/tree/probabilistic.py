# Natural Language Toolkit: Text Trees
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
#         Tom Aarsen <>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


from nltk.internals import raise_unorderable_types
from nltk.probability import ProbabilisticMixIn
from nltk.tree.immutable import ImmutableProbabilisticTree
from nltk.tree.tree import Tree

######################################################################
## Probabilistic trees
######################################################################


class ProbabilisticTree(Tree, ProbabilisticMixIn):
    def __init__(self, node, children=None, **prob_kwargs):
        Tree.__init__(self, node, children)
        ProbabilisticMixIn.__init__(self, **prob_kwargs)

    # We have to patch up these methods to make them work right:
    def _frozen_class(self):
        return ImmutableProbabilisticTree

    def __repr__(self):
        return f"{Tree.__repr__(self)} (p={self.prob()!r})"

    def __str__(self):
        return f"{self.pformat(margin=60)} (p={self.prob():.6g})"

    def copy(self, deep=False):
        if not deep:
            return type(self)(self._label, self, prob=self.prob())
        else:
            return type(self).convert(self)

    @classmethod
    def convert(cls, val):
        if isinstance(val, Tree):
            children = [cls.convert(child) for child in val]
            if isinstance(val, ProbabilisticMixIn):
                return cls(val._label, children, prob=val.prob())
            else:
                return cls(val._label, children, prob=1.0)
        else:
            return val

    def __eq__(self, other):
        return self.__class__ is other.__class__ and (
            self._label,
            list(self),
            self.prob(),
        ) == (other._label, list(other), other.prob())

    def __lt__(self, other):
        if not isinstance(other, Tree):
            raise_unorderable_types("<", self, other)
        if self.__class__ is other.__class__:
            return (self._label, list(self), self.prob()) < (
                other._label,
                list(other),
                other.prob(),
            )
        else:
            return self.__class__.__name__ < other.__class__.__name__


__all__ = ["ProbabilisticTree"]
