# Natural Language Toolkit: Transformation-based learning
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Marcus Uneson <marcus.uneson@gmail.com>
#   based on previous (nltk2) version by
#   Christopher Maloof, Edward Loper, Steven Bird
# URL: <https://www.nltk.org/>
# For license information, see  LICENSE.TXT

from abc import ABCMeta, abstractmethod

from nltk import jsontags


######################################################################
# Tag Rules
######################################################################
class TagRule(metaclass=ABCMeta):
    """
    An interface for tag transformations on a tagged corpus, as
    performed by tbl taggers.  Each transformation finds all tokens
    in the corpus that are tagged with a specific original tag and
    satisfy a specific condition, and replaces their tags with a
    replacement tag.  For any given transformation, the original
    tag, replacement tag, and condition are fixed.  Conditions may
    depend on the token under consideration, as well as any other
    tokens in the corpus.

    Tag rules must be comparable and hashable.
    """

    def __init__(self, original_tag, replacement_tag):

        self.original_tag = original_tag
        """The tag which this TagRule may cause to be replaced."""

        self.replacement_tag = replacement_tag
        """The tag with which this TagRule may replace another tag."""

    def apply(self, tokens, positions=None):
        """
        Apply this rule at every position in positions where it
        applies to the given sentence.  I.e., for each position p
        in *positions*, if *tokens[p]* is tagged with this rule's
        original tag, and satisfies this rule's condition, then set
        its tag to be this rule's replacement tag.

        :param tokens: The tagged sentence
        :type tokens: list(tuple(str, str))
        :type positions: list(int)
        :param positions: The positions where the transformation is to
            be tried.  If not specified, try it at all positions.
        :return: The indices of tokens whose tags were changed by this
            rule.
        :rtype: int
        """
        if positions is None:
            positions = list(range(len(tokens)))

        # Determine the indices at which this rule applies.
        change = [i for i in positions if self.applies(tokens, i)]

        # Make the changes.  Note: this must be done in a separate
        # step from finding applicable locations, since we don't want
        # the rule to interact with itself.
        for i in change:
            tokens[i] = (tokens[i][0], self.replacement_tag)

        return change

    @abstractmethod
    def applies(self, tokens, index):
        """
        :return: True if the rule would change the tag of
            ``tokens[index]``, False otherwise
        :rtype: bool
        :param tokens: A tagged sentence
        :type tokens: list(str)
        :param index: The index to check
        :type index: int
        """

    # Rules must be comparable and hashable for the algorithm to work
    def __eq__(self, other):
        raise TypeError("Rules must implement __eq__()")

    def __ne__(self, other):
        raise TypeError("Rules must implement __ne__()")

    def __hash__(self):
        raise TypeError("Rules must implement __hash__()")


@jsontags.register_tag
class Rule(TagRule):
    """
    A Rule checks the current corpus position for a certain set of conditions;
    if they are all fulfilled, the Rule is triggered, meaning that it
    will change tag A to tag B. For other tags than A, nothing happens.

    The conditions are parameters to the Rule instance. Each condition is a feature-value pair,
    with a set of positions to check for the value of the corresponding feature.
    Conceptually, the positions are joined by logical OR, and the feature set by logical AND.

    More formally, the Rule is then applicable to the M{n}th token iff:

      - The M{n}th token is tagged with the Rule's original tag; and
      - For each (Feature(positions), M{value}) tuple:

        - The value of Feature of at least one token in {n+p for p in positions}
          is M{value}.
    """

    json_tag = "nltk.tbl.Rule"

    def __init__(self, templateid, original_tag, replacement_tag, conditions):
        """
        Construct a new Rule that changes a token's tag from
        C{original_tag} to C{replacement_tag} if all of the properties
        specified in C{conditions} hold.

        :param templateid: the template id (a zero-padded string, '001' etc,
            so it will sort nicely)
        :type templateid: string

        :param conditions: A list of Feature(positions),
            each of which specifies that the property (computed by
            Feature.extract_property()) of at least one
            token in M{n} + p in positions is C{value}.
        :type conditions: C{iterable} of C{Feature}

        """
        TagRule.__init__(self, original_tag, replacement_tag)
        self._conditions = conditions
        self.templateid = templateid

    def encode_json_obj(self):
        return {
            "templateid": self.templateid,
            "original": self.original_tag,
            "replacement": self.replacement_tag,
            "conditions": self._conditions,
        }

    @classmethod
    def decode_json_obj(cls, obj):
        return cls(
            obj["templateid"],
            obj["original"],
            obj["replacement"],
            tuple(tuple(feat) for feat in obj["conditions"]),
        )

    def applies(self, tokens, index):
        # Inherit docs from TagRule

        # Does the given token have this Rule's "original tag"?
        if tokens[index][1] != self.original_tag:
            return False

        # Check to make sure that every condition holds.
        for (feature, val) in self._conditions:

            # Look for *any* token that satisfies the condition.
            for pos in feature.positions:
                if not (0 <= index + pos < len(tokens)):
                    continue
                if feature.extract_property(tokens, index + pos) == val:
                    break
            else:
                # No token satisfied the condition; return false.
                return False

        # Every condition checked out, so the Rule is applicable.
        return True

    def __eq__(self, other):
        return self is other or (
            other is not None
            and other.__class__ == self.__class__
            and self.original_tag == other.original_tag
            and self.replacement_tag == other.replacement_tag
            and self._conditions == other._conditions
        )

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):

        # Cache our hash value (justified by profiling.)
        try:
            return self.__hash
        except AttributeError:
            self.__hash = hash(repr(self))
            return self.__hash

    def __repr__(self):
        # Cache the repr (justified by profiling -- this is used as
        # a sort key when deterministic=True.)
        try:
            return self.__repr
        except AttributeError:
            self.__repr = "{}('{}', {}, {}, [{}])".format(
                self.__class__.__name__,
                self.templateid,
                repr(self.original_tag),
                repr(self.replacement_tag),
                # list(self._conditions) would be simpler but will not generate
                # the same Rule.__repr__ in python 2 and 3 and thus break some tests
                ", ".join(f"({f},{repr(v)})" for (f, v) in self._conditions),
            )

            return self.__repr

    def __str__(self):
        def _condition_to_logic(feature, value):
            """
            Return a compact, predicate-logic styled string representation
            of the given condition.
            """
            return "{}:{}@[{}]".format(
                feature.PROPERTY_NAME,
                value,
                ",".join(str(w) for w in feature.positions),
            )

        conditions = " & ".join(
            [_condition_to_logic(f, v) for (f, v) in self._conditions]
        )
        s = f"{self.original_tag}->{self.replacement_tag} if {conditions}"

        return s

    def format(self, fmt):
        """
        Return a string representation of this rule.

        >>> from nltk.tbl.rule import Rule
        >>> from nltk.tag.brill import Pos

        >>> r = Rule("23", "VB", "NN", [(Pos([-2,-1]), 'DT')])

        r.format("str") == str(r)
        True
        >>> r.format("str")
        'VB->NN if Pos:DT@[-2,-1]'

        r.format("repr") == repr(r)
        True
        >>> r.format("repr")
        "Rule('23', 'VB', 'NN', [(Pos([-2, -1]),'DT')])"

        >>> r.format("verbose")
        'VB -> NN if the Pos of words i-2...i-1 is "DT"'

        >>> r.format("not_found")
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "nltk/tbl/rule.py", line 256, in format
            raise ValueError("unknown rule format spec: {0}".format(fmt))
        ValueError: unknown rule format spec: not_found
        >>>

        :param fmt: format specification
        :type fmt: str
        :return: string representation
        :rtype: str
        """
        if fmt == "str":
            return self.__str__()
        elif fmt == "repr":
            return self.__repr__()
        elif fmt == "verbose":
            return self._verbose_format()
        else:
            raise ValueError(f"unknown rule format spec: {fmt}")

    def _verbose_format(self):
        """
        Return a wordy, human-readable string representation
        of the given rule.

        Not sure how useful this is.
        """

        def condition_to_str(feature, value):
            return 'the {} of {} is "{}"'.format(
                feature.PROPERTY_NAME,
                range_to_str(feature.positions),
                value,
            )

        def range_to_str(positions):
            if len(positions) == 1:
                p = positions[0]
                if p == 0:
                    return "this word"
                if p == -1:
                    return "the preceding word"
                elif p == 1:
                    return "the following word"
                elif p < 0:
                    return "word i-%d" % -p
                elif p > 0:
                    return "word i+%d" % p
            else:
                # for complete compatibility with the wordy format of nltk2
                mx = max(positions)
                mn = min(positions)
                if mx - mn == len(positions) - 1:
                    return "words i%+d...i%+d" % (mn, mx)
                else:
                    return "words {{{}}}".format(
                        ",".join("i%+d" % d for d in positions)
                    )

        replacement = f"{self.original_tag} -> {self.replacement_tag}"
        conditions = (" if " if self._conditions else "") + ", and ".join(
            condition_to_str(f, v) for (f, v) in self._conditions
        )
        return replacement + conditions
