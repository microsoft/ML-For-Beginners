# Natural Language Toolkit: Text Trees
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
#         Tom Aarsen <>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from nltk.probability import ProbabilisticMixIn
from nltk.tree.parented import MultiParentedTree, ParentedTree
from nltk.tree.tree import Tree


class ImmutableTree(Tree):
    def __init__(self, node, children=None):
        super().__init__(node, children)
        # Precompute our hash value.  This ensures that we're really
        # immutable.  It also means we only have to calculate it once.
        try:
            self._hash = hash((self._label, tuple(self)))
        except (TypeError, ValueError) as e:
            raise ValueError(
                "%s: node value and children " "must be immutable" % type(self).__name__
            ) from e

    def __setitem__(self, index, value):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __setslice__(self, i, j, value):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __delitem__(self, index):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __delslice__(self, i, j):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __iadd__(self, other):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __imul__(self, other):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def append(self, v):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def extend(self, v):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def pop(self, v=None):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def remove(self, v):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def reverse(self):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def sort(self):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __hash__(self):
        return self._hash

    def set_label(self, value):
        """
        Set the node label.  This will only succeed the first time the
        node label is set, which should occur in ImmutableTree.__init__().
        """
        if hasattr(self, "_label"):
            raise ValueError("%s may not be modified" % type(self).__name__)
        self._label = value


class ImmutableProbabilisticTree(ImmutableTree, ProbabilisticMixIn):
    def __init__(self, node, children=None, **prob_kwargs):
        ImmutableTree.__init__(self, node, children)
        ProbabilisticMixIn.__init__(self, **prob_kwargs)
        self._hash = hash((self._label, tuple(self), self.prob()))

    # We have to patch up these methods to make them work right:
    def _frozen_class(self):
        return ImmutableProbabilisticTree

    def __repr__(self):
        return f"{Tree.__repr__(self)} [{self.prob()}]"

    def __str__(self):
        return f"{self.pformat(margin=60)} [{self.prob()}]"

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


class ImmutableParentedTree(ImmutableTree, ParentedTree):
    pass


class ImmutableMultiParentedTree(ImmutableTree, MultiParentedTree):
    pass


__all__ = [
    "ImmutableProbabilisticTree",
    "ImmutableTree",
    "ImmutableParentedTree",
    "ImmutableMultiParentedTree",
]
