# Natural Language Toolkit: Transformation-based learning
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Marcus Uneson <marcus.uneson@gmail.com>
#   based on previous (nltk2) version by
#   Christopher Maloof, Edward Loper, Steven Bird
# URL: <https://www.nltk.org/>
# For license information, see  LICENSE.TXT

import itertools as it
from abc import ABCMeta, abstractmethod

from nltk.tbl.feature import Feature
from nltk.tbl.rule import Rule


class BrillTemplateI(metaclass=ABCMeta):
    """
    An interface for generating lists of transformational rules that
    apply at given sentence positions.  ``BrillTemplateI`` is used by
    ``Brill`` training algorithms to generate candidate rules.
    """

    @abstractmethod
    def applicable_rules(self, tokens, i, correctTag):
        """
        Return a list of the transformational rules that would correct
        the ``i``-th subtoken's tag in the given token.  In particular,
        return a list of zero or more rules that would change
        ``tokens[i][1]`` to ``correctTag``, if applied to ``token[i]``.

        If the ``i``-th token already has the correct tag (i.e., if
        ``tagged_tokens[i][1] == correctTag``), then
        ``applicable_rules()`` should return the empty list.

        :param tokens: The tagged tokens being tagged.
        :type tokens: list(tuple)
        :param i: The index of the token whose tag should be corrected.
        :type i: int
        :param correctTag: The correct tag for the ``i``-th token.
        :type correctTag: any
        :rtype: list(BrillRule)
        """

    @abstractmethod
    def get_neighborhood(self, token, index):
        """
        Returns the set of indices *i* such that
        ``applicable_rules(token, i, ...)`` depends on the value of
        the *index*th token of *token*.

        This method is used by the "fast" Brill tagger trainer.

        :param token: The tokens being tagged.
        :type token: list(tuple)
        :param index: The index whose neighborhood should be returned.
        :type index: int
        :rtype: set
        """


class Template(BrillTemplateI):
    """
    A tbl Template that generates a list of L{Rule}s that apply at a given sentence
    position.  In particular, each C{Template} is parameterized by a list of
    independent features (a combination of a specific
    property to extract and a list C{L} of relative positions at which to extract
    it) and generates all Rules that:

      - use the given features, each at its own independent position; and
      - are applicable to the given token.
    """

    ALLTEMPLATES = []
    # record a unique id of form "001", for each template created
    # _ids = it.count(0)

    def __init__(self, *features):

        """
        Construct a Template for generating Rules.

        Takes a list of Features. A C{Feature} is a combination
        of a specific property and its relative positions and should be
        a subclass of L{nltk.tbl.feature.Feature}.

        An alternative calling convention (kept for backwards compatibility,
        but less expressive as it only permits one feature type) is
        Template(Feature, (start1, end1), (start2, end2), ...)
        In new code, that would be better written
        Template(Feature(start1, end1), Feature(start2, end2), ...)

        For instance, importing some features

        >>> from nltk.tbl.template import Template
        >>> from nltk.tag.brill import Word, Pos

        Create some features

        >>> wfeat1, wfeat2, pfeat = (Word([-1]), Word([1,2]), Pos([-2,-1]))

        Create a single-feature template

        >>> Template(wfeat1)
        Template(Word([-1]))

        Or a two-feature one

        >>> Template(wfeat1, wfeat2)
        Template(Word([-1]),Word([1, 2]))

        Or a three-feature one with two different feature types

        >>> Template(wfeat1, wfeat2, pfeat)
        Template(Word([-1]),Word([1, 2]),Pos([-2, -1]))

        deprecated api: Feature subclass, followed by list of (start,end) pairs
        (permits only a single Feature)

        >>> Template(Word, (-2,-1), (0,0))
        Template(Word([-2, -1]),Word([0]))

        Incorrect specification raises TypeError

        >>> Template(Word, (-2,-1), Pos, (0,0))
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "nltk/tag/tbl/template.py", line 143, in __init__
            raise TypeError(
        TypeError: expected either Feature1(args), Feature2(args), ... or Feature, (start1, end1), (start2, end2), ...

        :type features: list of Features
        :param features: the features to build this Template on
        """
        # determine the calling form: either
        # Template(Feature, args1, [args2, ...)]
        # Template(Feature1(args),  Feature2(args), ...)
        if all(isinstance(f, Feature) for f in features):
            self._features = features
        elif issubclass(features[0], Feature) and all(
            isinstance(a, tuple) for a in features[1:]
        ):
            self._features = [features[0](*tp) for tp in features[1:]]
        else:
            raise TypeError(
                "expected either Feature1(args), Feature2(args), ... or Feature, (start1, end1), (start2, end2), ..."
            )
        self.id = f"{len(self.ALLTEMPLATES):03d}"
        self.ALLTEMPLATES.append(self)

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ",".join([str(f) for f in self._features]),
        )

    def applicable_rules(self, tokens, index, correct_tag):
        if tokens[index][1] == correct_tag:
            return []

        # For each of this Template's features, find the conditions
        # that are applicable for the given token.
        # Then, generate one Rule for each combination of features
        # (the crossproduct of the conditions).

        applicable_conditions = self._applicable_conditions(tokens, index)
        xs = list(it.product(*applicable_conditions))
        return [Rule(self.id, tokens[index][1], correct_tag, tuple(x)) for x in xs]

    def _applicable_conditions(self, tokens, index):
        """
        :returns: A set of all conditions for rules
        that are applicable to C{tokens[index]}.
        """
        conditions = []

        for feature in self._features:
            conditions.append([])
            for pos in feature.positions:
                if not (0 <= index + pos < len(tokens)):
                    continue
                value = feature.extract_property(tokens, index + pos)
                conditions[-1].append((feature, value))
        return conditions

    def get_neighborhood(self, tokens, index):
        # inherit docs from BrillTemplateI

        # applicable_rules(tokens, index, ...) depends on index.
        neighborhood = {index}  # set literal for python 2.7+

        # applicable_rules(tokens, i, ...) depends on index if
        # i+start < index <= i+end.

        allpositions = [0] + [p for feat in self._features for p in feat.positions]
        start, end = min(allpositions), max(allpositions)
        s = max(0, index + (-end))
        e = min(index + (-start) + 1, len(tokens))
        for i in range(s, e):
            neighborhood.add(i)
        return neighborhood

    @classmethod
    def expand(cls, featurelists, combinations=None, skipintersecting=True):

        """
        Factory method to mass generate Templates from a list L of lists of  Features.

        #With combinations=(k1, k2), the function will in all possible ways choose k1 ... k2
        #of the sublists in L; it will output all Templates formed by the Cartesian product
        #of this selection, with duplicates and other semantically equivalent
        #forms removed. Default for combinations is (1, len(L)).

        The feature lists may have been specified
        manually, or generated from Feature.expand(). For instance,

        >>> from nltk.tbl.template import Template
        >>> from nltk.tag.brill import Word, Pos

        #creating some features
        >>> (wd_0, wd_01) = (Word([0]), Word([0,1]))

        >>> (pos_m2, pos_m33) = (Pos([-2]), Pos([3-2,-1,0,1,2,3]))

        >>> list(Template.expand([[wd_0], [pos_m2]]))
        [Template(Word([0])), Template(Pos([-2])), Template(Pos([-2]),Word([0]))]

        >>> list(Template.expand([[wd_0, wd_01], [pos_m2]]))
        [Template(Word([0])), Template(Word([0, 1])), Template(Pos([-2])), Template(Pos([-2]),Word([0])), Template(Pos([-2]),Word([0, 1]))]

        #note: with Feature.expand(), it is very easy to generate more templates
        #than your system can handle -- for instance,
        >>> wordtpls = Word.expand([-2,-1,0,1], [1,2], excludezero=False)
        >>> len(wordtpls)
        7

        >>> postpls = Pos.expand([-3,-2,-1,0,1,2], [1,2,3], excludezero=True)
        >>> len(postpls)
        9

        #and now the Cartesian product of all non-empty combinations of two wordtpls and
        #two postpls, with semantic equivalents removed
        >>> templates = list(Template.expand([wordtpls, wordtpls, postpls, postpls]))
        >>> len(templates)
        713


          will return a list of eight templates
              Template(Word([0])),
              Template(Word([0, 1])),
              Template(Pos([-2])),
              Template(Pos([-1])),
              Template(Pos([-2]),Word([0])),
              Template(Pos([-1]),Word([0])),
              Template(Pos([-2]),Word([0, 1])),
              Template(Pos([-1]),Word([0, 1]))]


        #Templates where one feature is a subset of another, such as
        #Template(Word([0,1]), Word([1]), will not appear in the output.
        #By default, this non-subset constraint is tightened to disjointness:
        #Templates of type Template(Word([0,1]), Word([1,2]) will also be filtered out.
        #With skipintersecting=False, then such Templates are allowed

        WARNING: this method makes it very easy to fill all your memory when training
        generated templates on any real-world corpus

        :param featurelists: lists of Features, whose Cartesian product will return a set of Templates
        :type featurelists: list of (list of Features)
        :param combinations: given n featurelists: if combinations=k, all generated Templates will have
                k features; if combinations=(k1,k2) they will have k1..k2 features; if None, defaults to 1..n
        :type combinations: None, int, or (int, int)
        :param skipintersecting: if True, do not output intersecting Templates (non-disjoint positions for some feature)
        :type skipintersecting: bool
        :returns: generator of Templates

        """

        def nonempty_powerset(xs):  # xs is a list
            # itertools docnonempty_powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

            # find the correct tuple given combinations, one of {None, k, (k1,k2)}
            k = combinations  # for brevity
            combrange = (
                (1, len(xs) + 1)
                if k is None
                else (k, k + 1)  # n over 1 .. n over n (all non-empty combinations)
                if isinstance(k, int)
                else (k[0], k[1] + 1)  # n over k (only
            )  # n over k1, n over k1+1... n over k2
            return it.chain.from_iterable(
                it.combinations(xs, r) for r in range(*combrange)
            )

        seentemplates = set()
        for picks in nonempty_powerset(featurelists):
            for pick in it.product(*picks):
                if any(
                    i != j and x.issuperset(y)
                    for (i, x) in enumerate(pick)
                    for (j, y) in enumerate(pick)
                ):
                    continue
                if skipintersecting and any(
                    i != j and x.intersects(y)
                    for (i, x) in enumerate(pick)
                    for (j, y) in enumerate(pick)
                ):
                    continue
                thistemplate = cls(*sorted(pick))
                strpick = str(thistemplate)
                #!!FIXME --this is hackish
                if strpick in seentemplates:  # already added
                    cls._poptemplate()
                    continue
                seentemplates.add(strpick)
                yield thistemplate

    @classmethod
    def _cleartemplates(cls):
        cls.ALLTEMPLATES = []

    @classmethod
    def _poptemplate(cls):
        return cls.ALLTEMPLATES.pop() if cls.ALLTEMPLATES else None
