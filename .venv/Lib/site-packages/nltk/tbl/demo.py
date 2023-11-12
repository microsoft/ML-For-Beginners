# Natural Language Toolkit: Transformation-based learning
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Marcus Uneson <marcus.uneson@gmail.com>
#   based on previous (nltk2) version by
#   Christopher Maloof, Edward Loper, Steven Bird
# URL: <https://www.nltk.org/>
# For license information, see  LICENSE.TXT

import os
import pickle
import random
import time

from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list


def demo():
    """
    Run a demo with defaults. See source comments for details,
    or docstrings of any of the more specific demo_* functions.
    """
    postag()


def demo_repr_rule_format():
    """
    Exemplify repr(Rule) (see also str(Rule) and Rule.format("verbose"))
    """
    postag(ruleformat="repr")


def demo_str_rule_format():
    """
    Exemplify repr(Rule) (see also str(Rule) and Rule.format("verbose"))
    """
    postag(ruleformat="str")


def demo_verbose_rule_format():
    """
    Exemplify Rule.format("verbose")
    """
    postag(ruleformat="verbose")


def demo_multiposition_feature():
    """
    The feature/s of a template takes a list of positions
    relative to the current word where the feature should be
    looked for, conceptually joined by logical OR. For instance,
    Pos([-1, 1]), given a value V, will hold whenever V is found
    one step to the left and/or one step to the right.

    For contiguous ranges, a 2-arg form giving inclusive end
    points can also be used: Pos(-3, -1) is the same as the arg
    below.
    """
    postag(templates=[Template(Pos([-3, -2, -1]))])


def demo_multifeature_template():
    """
    Templates can have more than a single feature.
    """
    postag(templates=[Template(Word([0]), Pos([-2, -1]))])


def demo_template_statistics():
    """
    Show aggregate statistics per template. Little used templates are
    candidates for deletion, much used templates may possibly be refined.

    Deleting unused templates is mostly about saving time and/or space:
    training is basically O(T) in the number of templates T
    (also in terms of memory usage, which often will be the limiting factor).
    """
    postag(incremental_stats=True, template_stats=True)


def demo_generated_templates():
    """
    Template.expand and Feature.expand are class methods facilitating
    generating large amounts of templates. See their documentation for
    details.

    Note: training with 500 templates can easily fill all available
    even on relatively small corpora
    """
    wordtpls = Word.expand([-1, 0, 1], [1, 2], excludezero=False)
    tagtpls = Pos.expand([-2, -1, 0, 1], [1, 2], excludezero=True)
    templates = list(Template.expand([wordtpls, tagtpls], combinations=(1, 3)))
    print(
        "Generated {} templates for transformation-based learning".format(
            len(templates)
        )
    )
    postag(templates=templates, incremental_stats=True, template_stats=True)


def demo_learning_curve():
    """
    Plot a learning curve -- the contribution on tagging accuracy of
    the individual rules.
    Note: requires matplotlib
    """
    postag(
        incremental_stats=True,
        separate_baseline_data=True,
        learning_curve_output="learningcurve.png",
    )


def demo_error_analysis():
    """
    Writes a file with context for each erroneous word after tagging testing data
    """
    postag(error_output="errors.txt")


def demo_serialize_tagger():
    """
    Serializes the learned tagger to a file in pickle format; reloads it
    and validates the process.
    """
    postag(serialize_output="tagger.pcl")


def demo_high_accuracy_rules():
    """
    Discard rules with low accuracy. This may hurt performance a bit,
    but will often produce rules which are more interesting read to a human.
    """
    postag(num_sents=3000, min_acc=0.96, min_score=10)


def postag(
    templates=None,
    tagged_data=None,
    num_sents=1000,
    max_rules=300,
    min_score=3,
    min_acc=None,
    train=0.8,
    trace=3,
    randomize=False,
    ruleformat="str",
    incremental_stats=False,
    template_stats=False,
    error_output=None,
    serialize_output=None,
    learning_curve_output=None,
    learning_curve_take=300,
    baseline_backoff_tagger=None,
    separate_baseline_data=False,
    cache_baseline_tagger=None,
):
    """
    Brill Tagger Demonstration
    :param templates: how many sentences of training and testing data to use
    :type templates: list of Template

    :param tagged_data: maximum number of rule instances to create
    :type tagged_data: C{int}

    :param num_sents: how many sentences of training and testing data to use
    :type num_sents: C{int}

    :param max_rules: maximum number of rule instances to create
    :type max_rules: C{int}

    :param min_score: the minimum score for a rule in order for it to be considered
    :type min_score: C{int}

    :param min_acc: the minimum score for a rule in order for it to be considered
    :type min_acc: C{float}

    :param train: the fraction of the the corpus to be used for training (1=all)
    :type train: C{float}

    :param trace: the level of diagnostic tracing output to produce (0-4)
    :type trace: C{int}

    :param randomize: whether the training data should be a random subset of the corpus
    :type randomize: C{bool}

    :param ruleformat: rule output format, one of "str", "repr", "verbose"
    :type ruleformat: C{str}

    :param incremental_stats: if true, will tag incrementally and collect stats for each rule (rather slow)
    :type incremental_stats: C{bool}

    :param template_stats: if true, will print per-template statistics collected in training and (optionally) testing
    :type template_stats: C{bool}

    :param error_output: the file where errors will be saved
    :type error_output: C{string}

    :param serialize_output: the file where the learned tbl tagger will be saved
    :type serialize_output: C{string}

    :param learning_curve_output: filename of plot of learning curve(s) (train and also test, if available)
    :type learning_curve_output: C{string}

    :param learning_curve_take: how many rules plotted
    :type learning_curve_take: C{int}

    :param baseline_backoff_tagger: the file where rules will be saved
    :type baseline_backoff_tagger: tagger

    :param separate_baseline_data: use a fraction of the training data exclusively for training baseline
    :type separate_baseline_data: C{bool}

    :param cache_baseline_tagger: cache baseline tagger to this file (only interesting as a temporary workaround to get
                                  deterministic output from the baseline unigram tagger between python versions)
    :type cache_baseline_tagger: C{string}


    Note on separate_baseline_data: if True, reuse training data both for baseline and rule learner. This
    is fast and fine for a demo, but is likely to generalize worse on unseen data.
    Also cannot be sensibly used for learning curves on training data (the baseline will be artificially high).
    """

    # defaults
    baseline_backoff_tagger = baseline_backoff_tagger or REGEXP_TAGGER
    if templates is None:
        from nltk.tag.brill import brill24, describe_template_sets

        # some pre-built template sets taken from typical systems or publications are
        # available. Print a list with describe_template_sets()
        # for instance:
        templates = brill24()
    (training_data, baseline_data, gold_data, testing_data) = _demo_prepare_data(
        tagged_data, train, num_sents, randomize, separate_baseline_data
    )

    # creating (or reloading from cache) a baseline tagger (unigram tagger)
    # this is just a mechanism for getting deterministic output from the baseline between
    # python versions
    if cache_baseline_tagger:
        if not os.path.exists(cache_baseline_tagger):
            baseline_tagger = UnigramTagger(
                baseline_data, backoff=baseline_backoff_tagger
            )
            with open(cache_baseline_tagger, "w") as print_rules:
                pickle.dump(baseline_tagger, print_rules)
            print(
                "Trained baseline tagger, pickled it to {}".format(
                    cache_baseline_tagger
                )
            )
        with open(cache_baseline_tagger) as print_rules:
            baseline_tagger = pickle.load(print_rules)
            print(f"Reloaded pickled tagger from {cache_baseline_tagger}")
    else:
        baseline_tagger = UnigramTagger(baseline_data, backoff=baseline_backoff_tagger)
        print("Trained baseline tagger")
    if gold_data:
        print(
            "    Accuracy on test set: {:0.4f}".format(
                baseline_tagger.accuracy(gold_data)
            )
        )

    # creating a Brill tagger
    tbrill = time.time()
    trainer = BrillTaggerTrainer(
        baseline_tagger, templates, trace, ruleformat=ruleformat
    )
    print("Training tbl tagger...")
    brill_tagger = trainer.train(training_data, max_rules, min_score, min_acc)
    print(f"Trained tbl tagger in {time.time() - tbrill:0.2f} seconds")
    if gold_data:
        print("    Accuracy on test set: %.4f" % brill_tagger.accuracy(gold_data))

    # printing the learned rules, if learned silently
    if trace == 1:
        print("\nLearned rules: ")
        for (ruleno, rule) in enumerate(brill_tagger.rules(), 1):
            print(f"{ruleno:4d} {rule.format(ruleformat):s}")

    # printing template statistics (optionally including comparison with the training data)
    # note: if not separate_baseline_data, then baseline accuracy will be artificially high
    if incremental_stats:
        print(
            "Incrementally tagging the test data, collecting individual rule statistics"
        )
        (taggedtest, teststats) = brill_tagger.batch_tag_incremental(
            testing_data, gold_data
        )
        print("    Rule statistics collected")
        if not separate_baseline_data:
            print(
                "WARNING: train_stats asked for separate_baseline_data=True; the baseline "
                "will be artificially high"
            )
        trainstats = brill_tagger.train_stats()
        if template_stats:
            brill_tagger.print_template_statistics(teststats)
        if learning_curve_output:
            _demo_plot(
                learning_curve_output, teststats, trainstats, take=learning_curve_take
            )
            print(f"Wrote plot of learning curve to {learning_curve_output}")
    else:
        print("Tagging the test data")
        taggedtest = brill_tagger.tag_sents(testing_data)
        if template_stats:
            brill_tagger.print_template_statistics()

    # writing error analysis to file
    if error_output is not None:
        with open(error_output, "w") as f:
            f.write("Errors for Brill Tagger %r\n\n" % serialize_output)
            f.write("\n".join(error_list(gold_data, taggedtest)).encode("utf-8") + "\n")
        print(f"Wrote tagger errors including context to {error_output}")

    # serializing the tagger to a pickle file and reloading (just to see it works)
    if serialize_output is not None:
        taggedtest = brill_tagger.tag_sents(testing_data)
        with open(serialize_output, "w") as print_rules:
            pickle.dump(brill_tagger, print_rules)
        print(f"Wrote pickled tagger to {serialize_output}")
        with open(serialize_output) as print_rules:
            brill_tagger_reloaded = pickle.load(print_rules)
        print(f"Reloaded pickled tagger from {serialize_output}")
        taggedtest_reloaded = brill_tagger.tag_sents(testing_data)
        if taggedtest == taggedtest_reloaded:
            print("Reloaded tagger tried on test set, results identical")
        else:
            print("PROBLEM: Reloaded tagger gave different results on test set")


def _demo_prepare_data(
    tagged_data, train, num_sents, randomize, separate_baseline_data
):
    # train is the proportion of data used in training; the rest is reserved
    # for testing.
    if tagged_data is None:
        print("Loading tagged data from treebank... ")
        tagged_data = treebank.tagged_sents()
    if num_sents is None or len(tagged_data) <= num_sents:
        num_sents = len(tagged_data)
    if randomize:
        random.seed(len(tagged_data))
        random.shuffle(tagged_data)
    cutoff = int(num_sents * train)
    training_data = tagged_data[:cutoff]
    gold_data = tagged_data[cutoff:num_sents]
    testing_data = [[t[0] for t in sent] for sent in gold_data]
    if not separate_baseline_data:
        baseline_data = training_data
    else:
        bl_cutoff = len(training_data) // 3
        (baseline_data, training_data) = (
            training_data[:bl_cutoff],
            training_data[bl_cutoff:],
        )
    (trainseqs, traintokens) = corpus_size(training_data)
    (testseqs, testtokens) = corpus_size(testing_data)
    (bltrainseqs, bltraintokens) = corpus_size(baseline_data)
    print(f"Read testing data ({testseqs:d} sents/{testtokens:d} wds)")
    print(f"Read training data ({trainseqs:d} sents/{traintokens:d} wds)")
    print(
        "Read baseline data ({:d} sents/{:d} wds) {:s}".format(
            bltrainseqs,
            bltraintokens,
            "" if separate_baseline_data else "[reused the training set]",
        )
    )
    return (training_data, baseline_data, gold_data, testing_data)


def _demo_plot(learning_curve_output, teststats, trainstats=None, take=None):
    testcurve = [teststats["initialerrors"]]
    for rulescore in teststats["rulescores"]:
        testcurve.append(testcurve[-1] - rulescore)
    testcurve = [1 - x / teststats["tokencount"] for x in testcurve[:take]]

    traincurve = [trainstats["initialerrors"]]
    for rulescore in trainstats["rulescores"]:
        traincurve.append(traincurve[-1] - rulescore)
    traincurve = [1 - x / trainstats["tokencount"] for x in traincurve[:take]]

    import matplotlib.pyplot as plt

    r = list(range(len(testcurve)))
    plt.plot(r, testcurve, r, traincurve)
    plt.axis([None, None, None, 1.0])
    plt.savefig(learning_curve_output)


NN_CD_TAGGER = RegexpTagger([(r"^-?[0-9]+(\.[0-9]+)?$", "CD"), (r".*", "NN")])

REGEXP_TAGGER = RegexpTagger(
    [
        (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),  # cardinal numbers
        (r"(The|the|A|a|An|an)$", "AT"),  # articles
        (r".*able$", "JJ"),  # adjectives
        (r".*ness$", "NN"),  # nouns formed from adjectives
        (r".*ly$", "RB"),  # adverbs
        (r".*s$", "NNS"),  # plural nouns
        (r".*ing$", "VBG"),  # gerunds
        (r".*ed$", "VBD"),  # past tense verbs
        (r".*", "NN"),  # nouns (default)
    ]
)


def corpus_size(seqs):
    return (len(seqs), sum(len(x) for x in seqs))


if __name__ == "__main__":
    demo_learning_curve()
