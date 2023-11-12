# Natural Language Toolkit: Transformation-based learning
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Marcus Uneson <marcus.uneson@gmail.com>
#   based on previous (nltk2) version by
#   Christopher Maloof, Edward Loper, Steven Bird
# URL: <https://www.nltk.org/>
# For license information, see  LICENSE.TXT

from collections import Counter, defaultdict

from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template

######################################################################
# Brill Templates
######################################################################


@jsontags.register_tag
class Word(Feature):
    """
    Feature which examines the text (word) of nearby tokens.
    """

    json_tag = "nltk.tag.brill.Word"

    @staticmethod
    def extract_property(tokens, index):
        """@return: The given token's text."""
        return tokens[index][0]


@jsontags.register_tag
class Pos(Feature):
    """
    Feature which examines the tags of nearby tokens.
    """

    json_tag = "nltk.tag.brill.Pos"

    @staticmethod
    def extract_property(tokens, index):
        """@return: The given token's tag."""
        return tokens[index][1]


def nltkdemo18():
    """
    Return 18 templates, from the original nltk demo, in multi-feature syntax
    """
    return [
        Template(Pos([-1])),
        Template(Pos([1])),
        Template(Pos([-2])),
        Template(Pos([2])),
        Template(Pos([-2, -1])),
        Template(Pos([1, 2])),
        Template(Pos([-3, -2, -1])),
        Template(Pos([1, 2, 3])),
        Template(Pos([-1]), Pos([1])),
        Template(Word([-1])),
        Template(Word([1])),
        Template(Word([-2])),
        Template(Word([2])),
        Template(Word([-2, -1])),
        Template(Word([1, 2])),
        Template(Word([-3, -2, -1])),
        Template(Word([1, 2, 3])),
        Template(Word([-1]), Word([1])),
    ]


def nltkdemo18plus():
    """
    Return 18 templates, from the original nltk demo, and additionally a few
    multi-feature ones (the motivation is easy comparison with nltkdemo18)
    """
    return nltkdemo18() + [
        Template(Word([-1]), Pos([1])),
        Template(Pos([-1]), Word([1])),
        Template(Word([-1]), Word([0]), Pos([1])),
        Template(Pos([-1]), Word([0]), Word([1])),
        Template(Pos([-1]), Word([0]), Pos([1])),
    ]


def fntbl37():
    """
    Return 37 templates taken from the postagging task of the
    fntbl distribution https://www.cs.jhu.edu/~rflorian/fntbl/
    (37 is after excluding a handful which do not condition on Pos[0];
    fntbl can do that but the current nltk implementation cannot.)
    """
    return [
        Template(Word([0]), Word([1]), Word([2])),
        Template(Word([-1]), Word([0]), Word([1])),
        Template(Word([0]), Word([-1])),
        Template(Word([0]), Word([1])),
        Template(Word([0]), Word([2])),
        Template(Word([0]), Word([-2])),
        Template(Word([1, 2])),
        Template(Word([-2, -1])),
        Template(Word([1, 2, 3])),
        Template(Word([-3, -2, -1])),
        Template(Word([0]), Pos([2])),
        Template(Word([0]), Pos([-2])),
        Template(Word([0]), Pos([1])),
        Template(Word([0]), Pos([-1])),
        Template(Word([0])),
        Template(Word([-2])),
        Template(Word([2])),
        Template(Word([1])),
        Template(Word([-1])),
        Template(Pos([-1]), Pos([1])),
        Template(Pos([1]), Pos([2])),
        Template(Pos([-1]), Pos([-2])),
        Template(Pos([1])),
        Template(Pos([-1])),
        Template(Pos([-2])),
        Template(Pos([2])),
        Template(Pos([1, 2, 3])),
        Template(Pos([1, 2])),
        Template(Pos([-3, -2, -1])),
        Template(Pos([-2, -1])),
        Template(Pos([1]), Word([0]), Word([1])),
        Template(Pos([1]), Word([0]), Word([-1])),
        Template(Pos([-1]), Word([-1]), Word([0])),
        Template(Pos([-1]), Word([0]), Word([1])),
        Template(Pos([-2]), Pos([-1])),
        Template(Pos([1]), Pos([2])),
        Template(Pos([1]), Pos([2]), Word([1])),
    ]


def brill24():
    """
    Return 24 templates of the seminal TBL paper, Brill (1995)
    """
    return [
        Template(Pos([-1])),
        Template(Pos([1])),
        Template(Pos([-2])),
        Template(Pos([2])),
        Template(Pos([-2, -1])),
        Template(Pos([1, 2])),
        Template(Pos([-3, -2, -1])),
        Template(Pos([1, 2, 3])),
        Template(Pos([-1]), Pos([1])),
        Template(Pos([-2]), Pos([-1])),
        Template(Pos([1]), Pos([2])),
        Template(Word([-1])),
        Template(Word([1])),
        Template(Word([-2])),
        Template(Word([2])),
        Template(Word([-2, -1])),
        Template(Word([1, 2])),
        Template(Word([-1, 0])),
        Template(Word([0, 1])),
        Template(Word([0])),
        Template(Word([-1]), Pos([-1])),
        Template(Word([1]), Pos([1])),
        Template(Word([0]), Word([-1]), Pos([-1])),
        Template(Word([0]), Word([1]), Pos([1])),
    ]


def describe_template_sets():
    """
    Print the available template sets in this demo, with a short description"
    """
    import inspect
    import sys

    # a bit of magic to get all functions in this module
    templatesets = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    for (name, obj) in templatesets:
        if name == "describe_template_sets":
            continue
        print(name, obj.__doc__, "\n")


######################################################################
# The Brill Tagger
######################################################################


@jsontags.register_tag
class BrillTagger(TaggerI):
    """
    Brill's transformational rule-based tagger.  Brill taggers use an
    initial tagger (such as ``tag.DefaultTagger``) to assign an initial
    tag sequence to a text; and then apply an ordered list of
    transformational rules to correct the tags of individual tokens.
    These transformation rules are specified by the ``TagRule``
    interface.

    Brill taggers can be created directly, from an initial tagger and
    a list of transformational rules; but more often, Brill taggers
    are created by learning rules from a training corpus, using one
    of the TaggerTrainers available.
    """

    json_tag = "nltk.tag.BrillTagger"

    def __init__(self, initial_tagger, rules, training_stats=None):
        """
        :param initial_tagger: The initial tagger
        :type initial_tagger: TaggerI

        :param rules: An ordered list of transformation rules that
            should be used to correct the initial tagging.
        :type rules: list(TagRule)

        :param training_stats: A dictionary of statistics collected
            during training, for possible later use
        :type training_stats: dict

        """
        self._initial_tagger = initial_tagger
        self._rules = tuple(rules)
        self._training_stats = training_stats

    def encode_json_obj(self):
        return self._initial_tagger, self._rules, self._training_stats

    @classmethod
    def decode_json_obj(cls, obj):
        _initial_tagger, _rules, _training_stats = obj
        return cls(_initial_tagger, _rules, _training_stats)

    def rules(self):
        """
        Return the ordered list of  transformation rules that this tagger has learnt

        :return: the ordered list of transformation rules that correct the initial tagging
        :rtype: list of Rules
        """
        return self._rules

    def train_stats(self, statistic=None):
        """
        Return a named statistic collected during training, or a dictionary of all
        available statistics if no name given

        :param statistic: name of statistic
        :type statistic: str
        :return: some statistic collected during training of this tagger
        :rtype: any (but usually a number)
        """
        if statistic is None:
            return self._training_stats
        else:
            return self._training_stats.get(statistic)

    def tag(self, tokens):
        # Inherit documentation from TaggerI

        # Run the initial tagger.
        tagged_tokens = self._initial_tagger.tag(tokens)

        # Create a dictionary that maps each tag to a list of the
        # indices of tokens that have that tag.
        tag_to_positions = defaultdict(set)
        for i, (token, tag) in enumerate(tagged_tokens):
            tag_to_positions[tag].add(i)

        # Apply each rule, in order.  Only try to apply rules at
        # positions that have the desired original tag.
        for rule in self._rules:
            # Find the positions where it might apply
            positions = tag_to_positions.get(rule.original_tag, [])
            # Apply the rule at those positions.
            changed = rule.apply(tagged_tokens, positions)
            # Update tag_to_positions with the positions of tags that
            # were modified.
            for i in changed:
                tag_to_positions[rule.original_tag].remove(i)
                tag_to_positions[rule.replacement_tag].add(i)

        return tagged_tokens

    def print_template_statistics(self, test_stats=None, printunused=True):
        """
        Print a list of all templates, ranked according to efficiency.

        If test_stats is available, the templates are ranked according to their
        relative contribution (summed for all rules created from a given template,
        weighted by score) to the performance on the test set. If no test_stats, then
        statistics collected during training are used instead. There is also
        an unweighted measure (just counting the rules). This is less informative,
        though, as many low-score rules will appear towards end of training.

        :param test_stats: dictionary of statistics collected during testing
        :type test_stats: dict of str -> any (but usually numbers)
        :param printunused: if True, print a list of all unused templates
        :type printunused: bool
        :return: None
        :rtype: None
        """
        tids = [r.templateid for r in self._rules]
        train_stats = self.train_stats()

        trainscores = train_stats["rulescores"]
        assert len(trainscores) == len(
            tids
        ), "corrupt statistics: " "{} train scores for {} rules".format(
            trainscores, tids
        )
        template_counts = Counter(tids)
        weighted_traincounts = Counter()
        for (tid, score) in zip(tids, trainscores):
            weighted_traincounts[tid] += score
        tottrainscores = sum(trainscores)

        # det_tplsort() is for deterministic sorting;
        # the otherwise convenient Counter.most_common() unfortunately
        # does not break ties deterministically
        # between python versions and will break cross-version tests
        def det_tplsort(tpl_value):
            return (tpl_value[1], repr(tpl_value[0]))

        def print_train_stats():
            print(
                "TEMPLATE STATISTICS (TRAIN)  {} templates, {} rules)".format(
                    len(template_counts), len(tids)
                )
            )
            print(
                "TRAIN ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} "
                "final: {finalerrors:5d} {finalacc:.4f}".format(**train_stats)
            )
            head = "#ID | Score (train) |  #Rules     | Template"
            print(head, "\n", "-" * len(head), sep="")
            train_tplscores = sorted(
                weighted_traincounts.items(), key=det_tplsort, reverse=True
            )
            for (tid, trainscore) in train_tplscores:
                s = "{} | {:5d}   {:5.3f} |{:4d}   {:.3f} | {}".format(
                    tid,
                    trainscore,
                    trainscore / tottrainscores,
                    template_counts[tid],
                    template_counts[tid] / len(tids),
                    Template.ALLTEMPLATES[int(tid)],
                )
                print(s)

        def print_testtrain_stats():
            testscores = test_stats["rulescores"]
            print(
                "TEMPLATE STATISTICS (TEST AND TRAIN) ({} templates, {} rules)".format(
                    len(template_counts), len(tids)
                )
            )
            print(
                "TEST  ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} "
                "final: {finalerrors:5d} {finalacc:.4f} ".format(**test_stats)
            )
            print(
                "TRAIN ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} "
                "final: {finalerrors:5d} {finalacc:.4f} ".format(**train_stats)
            )
            weighted_testcounts = Counter()
            for (tid, score) in zip(tids, testscores):
                weighted_testcounts[tid] += score
            tottestscores = sum(testscores)
            head = "#ID | Score (test) | Score (train) |  #Rules     | Template"
            print(head, "\n", "-" * len(head), sep="")
            test_tplscores = sorted(
                weighted_testcounts.items(), key=det_tplsort, reverse=True
            )
            for (tid, testscore) in test_tplscores:
                s = "{:s} |{:5d}  {:6.3f} |  {:4d}   {:.3f} |{:4d}   {:.3f} | {:s}".format(
                    tid,
                    testscore,
                    testscore / tottestscores,
                    weighted_traincounts[tid],
                    weighted_traincounts[tid] / tottrainscores,
                    template_counts[tid],
                    template_counts[tid] / len(tids),
                    Template.ALLTEMPLATES[int(tid)],
                )
                print(s)

        def print_unused_templates():
            usedtpls = {int(tid) for tid in tids}
            unused = [
                (tid, tpl)
                for (tid, tpl) in enumerate(Template.ALLTEMPLATES)
                if tid not in usedtpls
            ]
            print(f"UNUSED TEMPLATES ({len(unused)})")

            for (tid, tpl) in unused:
                print(f"{tid:03d} {str(tpl):s}")

        if test_stats is None:
            print_train_stats()
        else:
            print_testtrain_stats()
        print()
        if printunused:
            print_unused_templates()
        print()

    def batch_tag_incremental(self, sequences, gold):
        """
        Tags by applying each rule to the entire corpus (rather than all rules to a
        single sequence). The point is to collect statistics on the test set for
        individual rules.

        NOTE: This is inefficient (does not build any index, so will traverse the entire
        corpus N times for N rules) -- usually you would not care about statistics for
        individual rules and thus use batch_tag() instead

        :param sequences: lists of token sequences (sentences, in some applications) to be tagged
        :type sequences: list of list of strings
        :param gold: the gold standard
        :type gold: list of list of strings
        :returns: tuple of (tagged_sequences, ordered list of rule scores (one for each rule))
        """

        def counterrors(xs):
            return sum(t[1] != g[1] for pair in zip(xs, gold) for (t, g) in zip(*pair))

        testing_stats = {}
        testing_stats["tokencount"] = sum(len(t) for t in sequences)
        testing_stats["sequencecount"] = len(sequences)
        tagged_tokenses = [self._initial_tagger.tag(tokens) for tokens in sequences]
        testing_stats["initialerrors"] = counterrors(tagged_tokenses)
        testing_stats["initialacc"] = (
            1 - testing_stats["initialerrors"] / testing_stats["tokencount"]
        )
        # Apply each rule to the entire corpus, in order
        errors = [testing_stats["initialerrors"]]
        for rule in self._rules:
            for tagged_tokens in tagged_tokenses:
                rule.apply(tagged_tokens)
            errors.append(counterrors(tagged_tokenses))
        testing_stats["rulescores"] = [
            err0 - err1 for (err0, err1) in zip(errors, errors[1:])
        ]
        testing_stats["finalerrors"] = errors[-1]
        testing_stats["finalacc"] = (
            1 - testing_stats["finalerrors"] / testing_stats["tokencount"]
        )
        return (tagged_tokenses, testing_stats)
