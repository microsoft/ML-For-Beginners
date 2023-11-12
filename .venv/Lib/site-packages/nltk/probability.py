# Natural Language Toolkit: Probability and Statistics
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (additions)
#         Trevor Cohn <tacohn@cs.mu.oz.au> (additions)
#         Peter Ljunglöf <peter.ljunglof@heatherleaf.se> (additions)
#         Liang Dong <ldong@clemson.edu> (additions)
#         Geoffrey Sampson <sampson@cantab.net> (additions)
#         Ilia Kurenkov <ilia.kurenkov@gmail.com> (additions)
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Classes for representing and processing probabilistic information.

The ``FreqDist`` class is used to encode "frequency distributions",
which count the number of times that each outcome of an experiment
occurs.

The ``ProbDistI`` class defines a standard interface for "probability
distributions", which encode the probability of each outcome for an
experiment.  There are two types of probability distribution:

  - "derived probability distributions" are created from frequency
    distributions.  They attempt to model the probability distribution
    that generated the frequency distribution.
  - "analytic probability distributions" are created directly from
    parameters (such as variance).

The ``ConditionalFreqDist`` class and ``ConditionalProbDistI`` interface
are used to encode conditional distributions.  Conditional probability
distributions can be derived or analytic; but currently the only
implementation of the ``ConditionalProbDistI`` interface is
``ConditionalProbDist``, a derived distribution.

"""

import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce

from nltk.internals import raise_unorderable_types

_NINF = float("-1e300")

##//////////////////////////////////////////////////////
##  Frequency Distributions
##//////////////////////////////////////////////////////


class FreqDist(Counter):
    """
    A frequency distribution for the outcomes of an experiment.  A
    frequency distribution records the number of times each outcome of
    an experiment has occurred.  For example, a frequency distribution
    could be used to record the frequency of each word type in a
    document.  Formally, a frequency distribution can be defined as a
    function mapping from each sample to the number of times that
    sample occurred as an outcome.

    Frequency distributions are generally constructed by running a
    number of experiments, and incrementing the count for a sample
    every time it is an outcome of an experiment.  For example, the
    following code will produce a frequency distribution that encodes
    how often each word occurs in a text:

        >>> from nltk.tokenize import word_tokenize
        >>> from nltk.probability import FreqDist
        >>> sent = 'This is an example sentence'
        >>> fdist = FreqDist()
        >>> for word in word_tokenize(sent):
        ...    fdist[word.lower()] += 1

    An equivalent way to do this is with the initializer:

        >>> fdist = FreqDist(word.lower() for word in word_tokenize(sent))

    """

    def __init__(self, samples=None):
        """
        Construct a new frequency distribution.  If ``samples`` is
        given, then the frequency distribution will be initialized
        with the count of each object in ``samples``; otherwise, it
        will be initialized to be empty.

        In particular, ``FreqDist()`` returns an empty frequency
        distribution; and ``FreqDist(samples)`` first creates an empty
        frequency distribution, and then calls ``update`` with the
        list ``samples``.

        :param samples: The samples to initialize the frequency
            distribution with.
        :type samples: Sequence
        """
        Counter.__init__(self, samples)

        # Cached number of samples in this FreqDist
        self._N = None

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this FreqDist.  For the number of unique
        sample values (or bins) with counts greater than zero, use
        ``FreqDist.B()``.

        :rtype: int
        """
        if self._N is None:
            # Not already cached, or cache has been invalidated
            self._N = sum(self.values())
        return self._N

    def __setitem__(self, key, val):
        """
        Override ``Counter.__setitem__()`` to invalidate the cached N
        """
        self._N = None
        super().__setitem__(key, val)

    def __delitem__(self, key):
        """
        Override ``Counter.__delitem__()`` to invalidate the cached N
        """
        self._N = None
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        """
        Override ``Counter.update()`` to invalidate the cached N
        """
        self._N = None
        super().update(*args, **kwargs)

    def setdefault(self, key, val):
        """
        Override ``Counter.setdefault()`` to invalidate the cached N
        """
        self._N = None
        super().setdefault(key, val)

    def B(self):
        """
        Return the total number of sample values (or "bins") that
        have counts greater than zero.  For the total
        number of sample outcomes recorded, use ``FreqDist.N()``.
        (FreqDist.B() is the same as len(FreqDist).)

        :rtype: int
        """
        return len(self)

    def hapaxes(self):
        """
        Return a list of all samples that occur once (hapax legomena)

        :rtype: list
        """
        return [item for item in self if self[item] == 1]

    def Nr(self, r, bins=None):
        return self.r_Nr(bins)[r]

    def r_Nr(self, bins=None):
        """
        Return the dictionary mapping r to Nr, the number of samples with frequency r, where Nr > 0.

        :type bins: int
        :param bins: The number of possible sample outcomes.  ``bins``
            is used to calculate Nr(0).  In particular, Nr(0) is
            ``bins-self.B()``.  If ``bins`` is not specified, it
            defaults to ``self.B()`` (so Nr(0) will be 0).
        :rtype: int
        """

        _r_Nr = defaultdict(int)
        for count in self.values():
            _r_Nr[count] += 1

        # Special case for Nr[0]:
        _r_Nr[0] = bins - self.B() if bins is not None else 0

        return _r_Nr

    def _cumulative_frequencies(self, samples):
        """
        Return the cumulative frequencies of the specified samples.
        If no samples are specified, all counts are returned, starting
        with the largest.

        :param samples: the samples whose frequencies should be returned.
        :type samples: any
        :rtype: list(float)
        """
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf

    # slightly odd nomenclature freq() if FreqDist does counts and ProbDist does probs,
    # here, freq() does probs
    def freq(self, sample):
        """
        Return the frequency of a given sample.  The frequency of a
        sample is defined as the count of that sample divided by the
        total number of sample outcomes that have been recorded by
        this FreqDist.  The count of a sample is defined as the
        number of times that sample outcome was recorded by this
        FreqDist.  Frequencies are always real numbers in the range
        [0, 1].

        :param sample: the sample whose frequency
               should be returned.
        :type sample: any
        :rtype: float
        """
        n = self.N()
        if n == 0:
            return 0
        return self[sample] / n

    def max(self):
        """
        Return the sample with the greatest number of outcomes in this
        frequency distribution.  If two or more samples have the same
        number of outcomes, return one of them; which sample is
        returned is undefined.  If no outcomes have occurred in this
        frequency distribution, return None.

        :return: The sample with the maximum number of outcomes in this
                frequency distribution.
        :rtype: any or None
        """
        if len(self) == 0:
            raise ValueError(
                "A FreqDist must have at least one sample before max is defined."
            )
        return self.most_common(1)[0][0]

    def plot(
        self, *args, title="", cumulative=False, percents=False, show=True, **kwargs
    ):
        """
        Plot samples from the frequency distribution
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.  For a cumulative plot, specify cumulative=True. Additional
        ``**kwargs`` are passed to matplotlib's plot function.
        (Requires Matplotlib to be installed.)

        :param title: The title for the graph.
        :type title: str
        :param cumulative: Whether the plot is cumulative. (default = False)
        :type cumulative: bool
        :param percents: Whether the plot uses percents instead of counts. (default = False)
        :type percents: bool
        :param show: Whether to show the plot, or only return the ax.
        :type show: bool
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ValueError(
                "The plot function requires matplotlib to be installed."
                "See https://matplotlib.org/"
            ) from e

        if len(args) == 0:
            args = [len(self)]
        samples = [item for item, _ in self.most_common(*args)]

        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
            ylabel = "Cumulative "
        else:
            freqs = [self[sample] for sample in samples]
            ylabel = ""

        if percents:
            freqs = [f / self.N() * 100 for f in freqs]
            ylabel += "Percents"
        else:
            ylabel += "Counts"

        ax = plt.gca()
        ax.grid(True, color="silver")

        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 2
        if title:
            ax.set_title(title)

        ax.plot(freqs, **kwargs)
        ax.set_xticks(range(len(samples)))
        ax.set_xticklabels([str(s) for s in samples], rotation=90)
        ax.set_xlabel("Samples")
        ax.set_ylabel(ylabel)

        if show:
            plt.show()

        return ax

    def tabulate(self, *args, **kwargs):
        """
        Tabulate the given samples from the frequency distribution (cumulative),
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.

        :param samples: The samples to plot (default is all samples)
        :type samples: list
        :param cumulative: A flag to specify whether the freqs are cumulative (default = False)
        :type title: bool
        """
        if len(args) == 0:
            args = [len(self)]
        samples = _get_kwarg(
            kwargs, "samples", [item for item, _ in self.most_common(*args)]
        )

        cumulative = _get_kwarg(kwargs, "cumulative", False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
        else:
            freqs = [self[sample] for sample in samples]
        # percents = [f * 100 for f in freqs]  only in ProbDist?

        width = max(len(f"{s}") for s in samples)
        width = max(width, max(len("%d" % f) for f in freqs))

        for i in range(len(samples)):
            print("%*s" % (width, samples[i]), end=" ")
        print()
        for i in range(len(samples)):
            print("%*d" % (width, freqs[i]), end=" ")
        print()

    def copy(self):
        """
        Create a copy of this frequency distribution.

        :rtype: FreqDist
        """
        return self.__class__(self)

    # Mathematical operatiors

    def __add__(self, other):
        """
        Add counts from two counters.

        >>> FreqDist('abbb') + FreqDist('bcc')
        FreqDist({'b': 4, 'c': 2, 'a': 1})

        """
        return self.__class__(super().__add__(other))

    def __sub__(self, other):
        """
        Subtract count, but keep only results with positive counts.

        >>> FreqDist('abbbc') - FreqDist('bccd')
        FreqDist({'b': 2, 'a': 1})

        """
        return self.__class__(super().__sub__(other))

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.

        >>> FreqDist('abbb') | FreqDist('bcc')
        FreqDist({'b': 3, 'c': 2, 'a': 1})

        """
        return self.__class__(super().__or__(other))

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.

        >>> FreqDist('abbb') & FreqDist('bcc')
        FreqDist({'b': 1})

        """
        return self.__class__(super().__and__(other))

    def __le__(self, other):
        """
        Returns True if this frequency distribution is a subset of the other
        and for no key the value exceeds the value of the same key from
        the other frequency distribution.

        The <= operator forms partial order and satisfying the axioms
        reflexivity, antisymmetry and transitivity.

        >>> FreqDist('a') <= FreqDist('a')
        True
        >>> a = FreqDist('abc')
        >>> b = FreqDist('aabc')
        >>> (a <= b, b <= a)
        (True, False)
        >>> FreqDist('a') <= FreqDist('abcd')
        True
        >>> FreqDist('abc') <= FreqDist('xyz')
        False
        >>> FreqDist('xyz') <= FreqDist('abc')
        False
        >>> c = FreqDist('a')
        >>> d = FreqDist('aa')
        >>> e = FreqDist('aaa')
        >>> c <= d and d <= e and c <= e
        True
        """
        if not isinstance(other, FreqDist):
            raise_unorderable_types("<=", self, other)
        return set(self).issubset(other) and all(
            self[key] <= other[key] for key in self
        )

    def __ge__(self, other):
        if not isinstance(other, FreqDist):
            raise_unorderable_types(">=", self, other)
        return set(self).issuperset(other) and all(
            self[key] >= other[key] for key in other
        )

    __lt__ = lambda self, other: self <= other and not self == other
    __gt__ = lambda self, other: self >= other and not self == other

    def __repr__(self):
        """
        Return a string representation of this FreqDist.

        :rtype: string
        """
        return self.pformat()

    def pprint(self, maxlen=10, stream=None):
        """
        Print a string representation of this FreqDist to 'stream'

        :param maxlen: The maximum number of items to print
        :type maxlen: int
        :param stream: The stream to print to. stdout by default
        """
        print(self.pformat(maxlen=maxlen), file=stream)

    def pformat(self, maxlen=10):
        """
        Return a string representation of this FreqDist.

        :param maxlen: The maximum number of items to display
        :type maxlen: int
        :rtype: string
        """
        items = ["{!r}: {!r}".format(*item) for item in self.most_common(maxlen)]
        if len(self) > maxlen:
            items.append("...")
        return "FreqDist({{{0}}})".format(", ".join(items))

    def __str__(self):
        """
        Return a string representation of this FreqDist.

        :rtype: string
        """
        return "<FreqDist with %d samples and %d outcomes>" % (len(self), self.N())

    def __iter__(self):
        """
        Return an iterator which yields tokens ordered by frequency.

        :rtype: iterator
        """
        for token, _ in self.most_common(self.B()):
            yield token


##//////////////////////////////////////////////////////
##  Probability Distributions
##//////////////////////////////////////////////////////


class ProbDistI(metaclass=ABCMeta):
    """
    A probability distribution for the outcomes of an experiment.  A
    probability distribution specifies how likely it is that an
    experiment will have any given outcome.  For example, a
    probability distribution could be used to predict the probability
    that a token in a document will have a given type.  Formally, a
    probability distribution can be defined as a function mapping from
    samples to nonnegative real numbers, such that the sum of every
    number in the function's range is 1.0.  A ``ProbDist`` is often
    used to model the probability distribution of the experiment used
    to generate a frequency distribution.
    """

    SUM_TO_ONE = True
    """True if the probabilities of the samples in this probability
       distribution will always sum to one."""

    @abstractmethod
    def __init__(self):
        """
        Classes inheriting from ProbDistI should implement __init__.
        """

    @abstractmethod
    def prob(self, sample):
        """
        Return the probability for a given sample.  Probabilities
        are always real numbers in the range [0, 1].

        :param sample: The sample whose probability
               should be returned.
        :type sample: any
        :rtype: float
        """

    def logprob(self, sample):
        """
        Return the base 2 logarithm of the probability for a given sample.

        :param sample: The sample whose probability
               should be returned.
        :type sample: any
        :rtype: float
        """
        # Default definition, in terms of prob()
        p = self.prob(sample)
        return math.log(p, 2) if p != 0 else _NINF

    @abstractmethod
    def max(self):
        """
        Return the sample with the greatest probability.  If two or
        more samples have the same probability, return one of them;
        which sample is returned is undefined.

        :rtype: any
        """

    @abstractmethod
    def samples(self):
        """
        Return a list of all samples that have nonzero probabilities.
        Use ``prob`` to find the probability of each sample.

        :rtype: list
        """

    # cf self.SUM_TO_ONE
    def discount(self):
        """
        Return the ratio by which counts are discounted on average: c*/c

        :rtype: float
        """
        return 0.0

    # Subclasses should define more efficient implementations of this,
    # where possible.
    def generate(self):
        """
        Return a randomly selected sample from this probability distribution.
        The probability of returning each sample ``samp`` is equal to
        ``self.prob(samp)``.
        """
        p = random.random()
        p_init = p
        for sample in self.samples():
            p -= self.prob(sample)
            if p <= 0:
                return sample
        # allow for some rounding error:
        if p < 0.0001:
            return sample
        # we *should* never get here
        if self.SUM_TO_ONE:
            warnings.warn(
                "Probability distribution %r sums to %r; generate()"
                " is returning an arbitrary sample." % (self, p_init - p)
            )
        return random.choice(list(self.samples()))


class UniformProbDist(ProbDistI):
    """
    A probability distribution that assigns equal probability to each
    sample in a given set; and a zero probability to all other
    samples.
    """

    def __init__(self, samples):
        """
        Construct a new uniform probability distribution, that assigns
        equal probability to each sample in ``samples``.

        :param samples: The samples that should be given uniform
            probability.
        :type samples: list
        :raise ValueError: If ``samples`` is empty.
        """
        if len(samples) == 0:
            raise ValueError(
                "A Uniform probability distribution must " + "have at least one sample."
            )
        self._sampleset = set(samples)
        self._prob = 1.0 / len(self._sampleset)
        self._samples = list(self._sampleset)

    def prob(self, sample):
        return self._prob if sample in self._sampleset else 0

    def max(self):
        return self._samples[0]

    def samples(self):
        return self._samples

    def __repr__(self):
        return "<UniformProbDist with %d samples>" % len(self._sampleset)


class RandomProbDist(ProbDistI):
    """
    Generates a random probability distribution whereby each sample
    will be between 0 and 1 with equal probability (uniform random distribution.
    Also called a continuous uniform distribution).
    """

    def __init__(self, samples):
        if len(samples) == 0:
            raise ValueError(
                "A probability distribution must " + "have at least one sample."
            )
        self._probs = self.unirand(samples)
        self._samples = list(self._probs.keys())

    @classmethod
    def unirand(cls, samples):
        """
        The key function that creates a randomized initial distribution
        that still sums to 1. Set as a dictionary of prob values so that
        it can still be passed to MutableProbDist and called with identical
        syntax to UniformProbDist
        """
        samples = set(samples)
        randrow = [random.random() for i in range(len(samples))]
        total = sum(randrow)
        for i, x in enumerate(randrow):
            randrow[i] = x / total

        total = sum(randrow)
        if total != 1:
            # this difference, if present, is so small (near NINF) that it
            # can be subtracted from any element without risking probs not (0 1)
            randrow[-1] -= total - 1

        return {s: randrow[i] for i, s in enumerate(samples)}

    def max(self):
        if not hasattr(self, "_max"):
            self._max = max((p, v) for (v, p) in self._probs.items())[1]
        return self._max

    def prob(self, sample):
        return self._probs.get(sample, 0)

    def samples(self):
        return self._samples

    def __repr__(self):
        return "<RandomUniformProbDist with %d samples>" % len(self._probs)


class DictionaryProbDist(ProbDistI):
    """
    A probability distribution whose probabilities are directly
    specified by a given dictionary.  The given dictionary maps
    samples to probabilities.
    """

    def __init__(self, prob_dict=None, log=False, normalize=False):
        """
        Construct a new probability distribution from the given
        dictionary, which maps values to probabilities (or to log
        probabilities, if ``log`` is true).  If ``normalize`` is
        true, then the probability values are scaled by a constant
        factor such that they sum to 1.

        If called without arguments, the resulting probability
        distribution assigns zero probability to all values.
        """

        self._prob_dict = prob_dict.copy() if prob_dict is not None else {}
        self._log = log

        # Normalize the distribution, if requested.
        if normalize:
            if len(prob_dict) == 0:
                raise ValueError(
                    "A DictionaryProbDist must have at least one sample "
                    + "before it can be normalized."
                )
            if log:
                value_sum = sum_logs(list(self._prob_dict.values()))
                if value_sum <= _NINF:
                    logp = math.log(1.0 / len(prob_dict), 2)
                    for x in prob_dict:
                        self._prob_dict[x] = logp
                else:
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] -= value_sum
            else:
                value_sum = sum(self._prob_dict.values())
                if value_sum == 0:
                    p = 1.0 / len(prob_dict)
                    for x in prob_dict:
                        self._prob_dict[x] = p
                else:
                    norm_factor = 1.0 / value_sum
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] *= norm_factor

    def prob(self, sample):
        if self._log:
            return 2 ** (self._prob_dict[sample]) if sample in self._prob_dict else 0
        else:
            return self._prob_dict.get(sample, 0)

    def logprob(self, sample):
        if self._log:
            return self._prob_dict.get(sample, _NINF)
        else:
            if sample not in self._prob_dict:
                return _NINF
            elif self._prob_dict[sample] == 0:
                return _NINF
            else:
                return math.log(self._prob_dict[sample], 2)

    def max(self):
        if not hasattr(self, "_max"):
            self._max = max((p, v) for (v, p) in self._prob_dict.items())[1]
        return self._max

    def samples(self):
        return self._prob_dict.keys()

    def __repr__(self):
        return "<ProbDist with %d samples>" % len(self._prob_dict)


class MLEProbDist(ProbDistI):
    """
    The maximum likelihood estimate for the probability distribution
    of the experiment used to generate a frequency distribution.  The
    "maximum likelihood estimate" approximates the probability of
    each sample as the frequency of that sample in the frequency
    distribution.
    """

    def __init__(self, freqdist, bins=None):
        """
        Use the maximum likelihood estimate to create a probability
        distribution for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        """
        self._freqdist = freqdist

    def freqdist(self):
        """
        Return the frequency distribution that this probability
        distribution is based on.

        :rtype: FreqDist
        """
        return self._freqdist

    def prob(self, sample):
        return self._freqdist.freq(sample)

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def __repr__(self):
        """
        :rtype: str
        :return: A string representation of this ``ProbDist``.
        """
        return "<MLEProbDist based on %d samples>" % self._freqdist.N()


class LidstoneProbDist(ProbDistI):
    """
    The Lidstone estimate for the probability distribution of the
    experiment used to generate a frequency distribution.  The
    "Lidstone estimate" is parameterized by a real number *gamma*,
    which typically ranges from 0 to 1.  The Lidstone estimate
    approximates the probability of a sample with count *c* from an
    experiment with *N* outcomes and *B* bins as
    ``c+gamma)/(N+B*gamma)``.  This is equivalent to adding
    *gamma* to the count for each bin, and taking the maximum
    likelihood estimate of the resulting frequency distribution.
    """

    SUM_TO_ONE = False

    def __init__(self, freqdist, gamma, bins=None):
        """
        Use the Lidstone estimate to create a probability distribution
        for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        :type gamma: float
        :param gamma: A real number used to parameterize the
            estimate.  The Lidstone estimate is equivalent to adding
            *gamma* to the count for each bin, and taking the
            maximum likelihood estimate of the resulting frequency
            distribution.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        if (bins == 0) or (bins is None and freqdist.N() == 0):
            name = self.__class__.__name__[:-8]
            raise ValueError(
                "A %s probability distribution " % name + "must have at least one bin."
            )
        if (bins is not None) and (bins < freqdist.B()):
            name = self.__class__.__name__[:-8]
            raise ValueError(
                "\nThe number of bins in a %s distribution " % name
                + "(%d) must be greater than or equal to\n" % bins
                + "the number of bins in the FreqDist used "
                + "to create it (%d)." % freqdist.B()
            )

        self._freqdist = freqdist
        self._gamma = float(gamma)
        self._N = self._freqdist.N()

        if bins is None:
            bins = freqdist.B()
        self._bins = bins

        self._divisor = self._N + bins * gamma
        if self._divisor == 0.0:
            # In extreme cases we force the probability to be 0,
            # which it will be, since the count will be 0:
            self._gamma = 0
            self._divisor = 1

    def freqdist(self):
        """
        Return the frequency distribution that this probability
        distribution is based on.

        :rtype: FreqDist
        """
        return self._freqdist

    def prob(self, sample):
        c = self._freqdist[sample]
        return (c + self._gamma) / self._divisor

    def max(self):
        # For Lidstone distributions, probability is monotonic with
        # frequency, so the most probable sample is the one that
        # occurs most frequently.
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def discount(self):
        gb = self._gamma * self._bins
        return gb / (self._N + gb)

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return "<LidstoneProbDist based on %d samples>" % self._freqdist.N()


class LaplaceProbDist(LidstoneProbDist):
    """
    The Laplace estimate for the probability distribution of the
    experiment used to generate a frequency distribution.  The
    "Laplace estimate" approximates the probability of a sample with
    count *c* from an experiment with *N* outcomes and *B* bins as
    *(c+1)/(N+B)*.  This is equivalent to adding one to the count for
    each bin, and taking the maximum likelihood estimate of the
    resulting frequency distribution.
    """

    def __init__(self, freqdist, bins=None):
        """
        Use the Laplace estimate to create a probability distribution
        for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        LidstoneProbDist.__init__(self, freqdist, 1, bins)

    def __repr__(self):
        """
        :rtype: str
        :return: A string representation of this ``ProbDist``.
        """
        return "<LaplaceProbDist based on %d samples>" % self._freqdist.N()


class ELEProbDist(LidstoneProbDist):
    """
    The expected likelihood estimate for the probability distribution
    of the experiment used to generate a frequency distribution.  The
    "expected likelihood estimate" approximates the probability of a
    sample with count *c* from an experiment with *N* outcomes and
    *B* bins as *(c+0.5)/(N+B/2)*.  This is equivalent to adding 0.5
    to the count for each bin, and taking the maximum likelihood
    estimate of the resulting frequency distribution.
    """

    def __init__(self, freqdist, bins=None):
        """
        Use the expected likelihood estimate to create a probability
        distribution for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        LidstoneProbDist.__init__(self, freqdist, 0.5, bins)

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return "<ELEProbDist based on %d samples>" % self._freqdist.N()


class HeldoutProbDist(ProbDistI):
    """
    The heldout estimate for the probability distribution of the
    experiment used to generate two frequency distributions.  These
    two frequency distributions are called the "heldout frequency
    distribution" and the "base frequency distribution."  The
    "heldout estimate" uses uses the "heldout frequency
    distribution" to predict the probability of each sample, given its
    frequency in the "base frequency distribution".

    In particular, the heldout estimate approximates the probability
    for a sample that occurs *r* times in the base distribution as
    the average frequency in the heldout distribution of all samples
    that occur *r* times in the base distribution.

    This average frequency is *Tr[r]/(Nr[r].N)*, where:

    - *Tr[r]* is the total count in the heldout distribution for
      all samples that occur *r* times in the base distribution.
    - *Nr[r]* is the number of samples that occur *r* times in
      the base distribution.
    - *N* is the number of outcomes recorded by the heldout
      frequency distribution.

    In order to increase the efficiency of the ``prob`` member
    function, *Tr[r]/(Nr[r].N)* is precomputed for each value of *r*
    when the ``HeldoutProbDist`` is created.

    :type _estimate: list(float)
    :ivar _estimate: A list mapping from *r*, the number of
        times that a sample occurs in the base distribution, to the
        probability estimate for that sample.  ``_estimate[r]`` is
        calculated by finding the average frequency in the heldout
        distribution of all samples that occur *r* times in the base
        distribution.  In particular, ``_estimate[r]`` =
        *Tr[r]/(Nr[r].N)*.
    :type _max_r: int
    :ivar _max_r: The maximum number of times that any sample occurs
        in the base distribution.  ``_max_r`` is used to decide how
        large ``_estimate`` must be.
    """

    SUM_TO_ONE = False

    def __init__(self, base_fdist, heldout_fdist, bins=None):
        """
        Use the heldout estimate to create a probability distribution
        for the experiment used to generate ``base_fdist`` and
        ``heldout_fdist``.

        :type base_fdist: FreqDist
        :param base_fdist: The base frequency distribution.
        :type heldout_fdist: FreqDist
        :param heldout_fdist: The heldout frequency distribution.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """

        self._base_fdist = base_fdist
        self._heldout_fdist = heldout_fdist

        # The max number of times any sample occurs in base_fdist.
        self._max_r = base_fdist[base_fdist.max()]

        # Calculate Tr, Nr, and N.
        Tr = self._calculate_Tr()
        r_Nr = base_fdist.r_Nr(bins)
        Nr = [r_Nr[r] for r in range(self._max_r + 1)]
        N = heldout_fdist.N()

        # Use Tr, Nr, and N to compute the probability estimate for
        # each value of r.
        self._estimate = self._calculate_estimate(Tr, Nr, N)

    def _calculate_Tr(self):
        """
        Return the list *Tr*, where *Tr[r]* is the total count in
        ``heldout_fdist`` for all samples that occur *r*
        times in ``base_fdist``.

        :rtype: list(float)
        """
        Tr = [0.0] * (self._max_r + 1)
        for sample in self._heldout_fdist:
            r = self._base_fdist[sample]
            Tr[r] += self._heldout_fdist[sample]
        return Tr

    def _calculate_estimate(self, Tr, Nr, N):
        """
        Return the list *estimate*, where *estimate[r]* is the probability
        estimate for any sample that occurs *r* times in the base frequency
        distribution.  In particular, *estimate[r]* is *Tr[r]/(N[r].N)*.
        In the special case that *N[r]=0*, *estimate[r]* will never be used;
        so we define *estimate[r]=None* for those cases.

        :rtype: list(float)
        :type Tr: list(float)
        :param Tr: the list *Tr*, where *Tr[r]* is the total count in
            the heldout distribution for all samples that occur *r*
            times in base distribution.
        :type Nr: list(float)
        :param Nr: The list *Nr*, where *Nr[r]* is the number of
            samples that occur *r* times in the base distribution.
        :type N: int
        :param N: The total number of outcomes recorded by the heldout
            frequency distribution.
        """
        estimate = []
        for r in range(self._max_r + 1):
            if Nr[r] == 0:
                estimate.append(None)
            else:
                estimate.append(Tr[r] / (Nr[r] * N))
        return estimate

    def base_fdist(self):
        """
        Return the base frequency distribution that this probability
        distribution is based on.

        :rtype: FreqDist
        """
        return self._base_fdist

    def heldout_fdist(self):
        """
        Return the heldout frequency distribution that this
        probability distribution is based on.

        :rtype: FreqDist
        """
        return self._heldout_fdist

    def samples(self):
        return self._base_fdist.keys()

    def prob(self, sample):
        # Use our precomputed probability estimate.
        r = self._base_fdist[sample]
        return self._estimate[r]

    def max(self):
        # Note: the Heldout estimation is *not* necessarily monotonic;
        # so this implementation is currently broken.  However, it
        # should give the right answer *most* of the time. :)
        return self._base_fdist.max()

    def discount(self):
        raise NotImplementedError()

    def __repr__(self):
        """
        :rtype: str
        :return: A string representation of this ``ProbDist``.
        """
        s = "<HeldoutProbDist: %d base samples; %d heldout samples>"
        return s % (self._base_fdist.N(), self._heldout_fdist.N())


class CrossValidationProbDist(ProbDistI):
    """
    The cross-validation estimate for the probability distribution of
    the experiment used to generate a set of frequency distribution.
    The "cross-validation estimate" for the probability of a sample
    is found by averaging the held-out estimates for the sample in
    each pair of frequency distributions.
    """

    SUM_TO_ONE = False

    def __init__(self, freqdists, bins):
        """
        Use the cross-validation estimate to create a probability
        distribution for the experiment used to generate
        ``freqdists``.

        :type freqdists: list(FreqDist)
        :param freqdists: A list of the frequency distributions
            generated by the experiment.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        self._freqdists = freqdists

        # Create a heldout probability distribution for each pair of
        # frequency distributions in freqdists.
        self._heldout_probdists = []
        for fdist1 in freqdists:
            for fdist2 in freqdists:
                if fdist1 is not fdist2:
                    probdist = HeldoutProbDist(fdist1, fdist2, bins)
                    self._heldout_probdists.append(probdist)

    def freqdists(self):
        """
        Return the list of frequency distributions that this ``ProbDist`` is based on.

        :rtype: list(FreqDist)
        """
        return self._freqdists

    def samples(self):
        # [xx] nb: this is not too efficient
        return set(sum((list(fd) for fd in self._freqdists), []))

    def prob(self, sample):
        # Find the average probability estimate returned by each
        # heldout distribution.
        prob = 0.0
        for heldout_probdist in self._heldout_probdists:
            prob += heldout_probdist.prob(sample)
        return prob / len(self._heldout_probdists)

    def discount(self):
        raise NotImplementedError()

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return "<CrossValidationProbDist: %d-way>" % len(self._freqdists)


class WittenBellProbDist(ProbDistI):
    """
    The Witten-Bell estimate of a probability distribution. This distribution
    allocates uniform probability mass to as yet unseen events by using the
    number of events that have only been seen once. The probability mass
    reserved for unseen events is equal to *T / (N + T)*
    where *T* is the number of observed event types and *N* is the total
    number of observed events. This equates to the maximum likelihood estimate
    of a new type event occurring. The remaining probability mass is discounted
    such that all probability estimates sum to one, yielding:

        - *p = T / Z (N + T)*, if count = 0
        - *p = c / (N + T)*, otherwise
    """

    def __init__(self, freqdist, bins=None):
        """
        Creates a distribution of Witten-Bell probability estimates.  This
        distribution allocates uniform probability mass to as yet unseen
        events by using the number of events that have only been seen once. The
        probability mass reserved for unseen events is equal to *T / (N + T)*
        where *T* is the number of observed event types and *N* is the total
        number of observed events. This equates to the maximum likelihood
        estimate of a new type event occurring. The remaining probability mass
        is discounted such that all probability estimates sum to one,
        yielding:

            - *p = T / Z (N + T)*, if count = 0
            - *p = c / (N + T)*, otherwise

        The parameters *T* and *N* are taken from the ``freqdist`` parameter
        (the ``B()`` and ``N()`` values). The normalizing factor *Z* is
        calculated using these values along with the ``bins`` parameter.

        :param freqdist: The frequency counts upon which to base the
            estimation.
        :type freqdist: FreqDist
        :param bins: The number of possible event types. This must be at least
            as large as the number of bins in the ``freqdist``. If None, then
            it's assumed to be equal to that of the ``freqdist``
        :type bins: int
        """
        assert bins is None or bins >= freqdist.B(), (
            "bins parameter must not be less than %d=freqdist.B()" % freqdist.B()
        )
        if bins is None:
            bins = freqdist.B()
        self._freqdist = freqdist
        self._T = self._freqdist.B()
        self._Z = bins - self._freqdist.B()
        self._N = self._freqdist.N()
        # self._P0 is P(0), precalculated for efficiency:
        if self._N == 0:
            # if freqdist is empty, we approximate P(0) by a UniformProbDist:
            self._P0 = 1.0 / self._Z
        else:
            self._P0 = self._T / (self._Z * (self._N + self._T))

    def prob(self, sample):
        # inherit docs from ProbDistI
        c = self._freqdist[sample]
        return c / (self._N + self._T) if c != 0 else self._P0

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def freqdist(self):
        return self._freqdist

    def discount(self):
        raise NotImplementedError()

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return "<WittenBellProbDist based on %d samples>" % self._freqdist.N()


##//////////////////////////////////////////////////////
##  Good-Turing Probability Distributions
##//////////////////////////////////////////////////////

# Good-Turing frequency estimation was contributed by Alan Turing and
# his statistical assistant I.J. Good, during their collaboration in
# the WWII.  It is a statistical technique for predicting the
# probability of occurrence of objects belonging to an unknown number
# of species, given past observations of such objects and their
# species. (In drawing balls from an urn, the 'objects' would be balls
# and the 'species' would be the distinct colors of the balls (finite
# but unknown in number).
#
# Good-Turing method calculates the probability mass to assign to
# events with zero or low counts based on the number of events with
# higher counts. It does so by using the adjusted count *c\**:
#
#     - *c\* = (c + 1) N(c + 1) / N(c)*   for c >= 1
#     - *things with frequency zero in training* = N(1)  for c == 0
#
# where *c* is the original count, *N(i)* is the number of event types
# observed with count *i*. We can think the count of unseen as the count
# of frequency one (see Jurafsky & Martin 2nd Edition, p101).
#
# This method is problematic because the situation ``N(c+1) == 0``
# is quite common in the original Good-Turing estimation; smoothing or
# interpolation of *N(i)* values is essential in practice.
#
# Bill Gale and Geoffrey Sampson present a simple and effective approach,
# Simple Good-Turing.  As a smoothing curve they simply use a power curve:
#
#     Nr = a*r^b (with b < -1 to give the appropriate hyperbolic
#     relationship)
#
# They estimate a and b by simple linear regression technique on the
# logarithmic form of the equation:
#
#     log Nr = a + b*log(r)
#
# However, they suggest that such a simple curve is probably only
# appropriate for high values of r. For low values of r, they use the
# measured Nr directly.  (see M&S, p.213)
#
# Gale and Sampson propose to use r while the difference between r and
# r* is 1.96 greater than the standard deviation, and switch to r* if
# it is less or equal:
#
#     |r - r*| > 1.96 * sqrt((r + 1)^2 (Nr+1 / Nr^2) (1 + Nr+1 / Nr))
#
# The 1.96 coefficient correspond to a 0.05 significance criterion,
# some implementations can use a coefficient of 1.65 for a 0.1
# significance criterion.
#

##//////////////////////////////////////////////////////
##  Simple Good-Turing Probablity Distributions
##//////////////////////////////////////////////////////


class SimpleGoodTuringProbDist(ProbDistI):
    """
    SimpleGoodTuring ProbDist approximates from frequency to frequency of
    frequency into a linear line under log space by linear regression.
    Details of Simple Good-Turing algorithm can be found in:

    - Good Turing smoothing without tears" (Gale & Sampson 1995),
      Journal of Quantitative Linguistics, vol. 2 pp. 217-237.
    - "Speech and Language Processing (Jurafsky & Martin),
      2nd Edition, Chapter 4.5 p103 (log(Nc) =  a + b*log(c))
    - https://www.grsampson.net/RGoodTur.html

    Given a set of pair (xi, yi),  where the xi denotes the frequency and
    yi denotes the frequency of frequency, we want to minimize their
    square variation. E(x) and E(y) represent the mean of xi and yi.

    - slope: b = sigma ((xi-E(x)(yi-E(y))) / sigma ((xi-E(x))(xi-E(x)))
    - intercept: a = E(y) - b.E(x)
    """

    SUM_TO_ONE = False

    def __init__(self, freqdist, bins=None):
        """
        :param freqdist: The frequency counts upon which to base the
            estimation.
        :type freqdist: FreqDist
        :param bins: The number of possible event types. This must be
            larger than the number of bins in the ``freqdist``. If None,
            then it's assumed to be equal to ``freqdist``.B() + 1
        :type bins: int
        """
        assert (
            bins is None or bins > freqdist.B()
        ), "bins parameter must not be less than %d=freqdist.B()+1" % (freqdist.B() + 1)
        if bins is None:
            bins = freqdist.B() + 1
        self._freqdist = freqdist
        self._bins = bins
        r, nr = self._r_Nr()
        self.find_best_fit(r, nr)
        self._switch(r, nr)
        self._renormalize(r, nr)

    def _r_Nr_non_zero(self):
        r_Nr = self._freqdist.r_Nr()
        del r_Nr[0]
        return r_Nr

    def _r_Nr(self):
        """
        Split the frequency distribution in two list (r, Nr), where Nr(r) > 0
        """
        nonzero = self._r_Nr_non_zero()

        if not nonzero:
            return [], []
        return zip(*sorted(nonzero.items()))

    def find_best_fit(self, r, nr):
        """
        Use simple linear regression to tune parameters self._slope and
        self._intercept in the log-log space based on count and Nr(count)
        (Work in log space to avoid floating point underflow.)
        """
        # For higher sample frequencies the data points becomes horizontal
        # along line Nr=1. To create a more evident linear model in log-log
        # space, we average positive Nr values with the surrounding zero
        # values. (Church and Gale, 1991)

        if not r or not nr:
            # Empty r or nr?
            return

        zr = []
        for j in range(len(r)):
            i = r[j - 1] if j > 0 else 0
            k = 2 * r[j] - i if j == len(r) - 1 else r[j + 1]
            zr_ = 2.0 * nr[j] / (k - i)
            zr.append(zr_)

        log_r = [math.log(i) for i in r]
        log_zr = [math.log(i) for i in zr]

        xy_cov = x_var = 0.0
        x_mean = sum(log_r) / len(log_r)
        y_mean = sum(log_zr) / len(log_zr)
        for (x, y) in zip(log_r, log_zr):
            xy_cov += (x - x_mean) * (y - y_mean)
            x_var += (x - x_mean) ** 2
        self._slope = xy_cov / x_var if x_var != 0 else 0.0
        if self._slope >= -1:
            warnings.warn(
                "SimpleGoodTuring did not find a proper best fit "
                "line for smoothing probabilities of occurrences. "
                "The probability estimates are likely to be "
                "unreliable."
            )
        self._intercept = y_mean - self._slope * x_mean

    def _switch(self, r, nr):
        """
        Calculate the r frontier where we must switch from Nr to Sr
        when estimating E[Nr].
        """
        for i, r_ in enumerate(r):
            if len(r) == i + 1 or r[i + 1] != r_ + 1:
                # We are at the end of r, or there is a gap in r
                self._switch_at = r_
                break

            Sr = self.smoothedNr
            smooth_r_star = (r_ + 1) * Sr(r_ + 1) / Sr(r_)
            unsmooth_r_star = (r_ + 1) * nr[i + 1] / nr[i]

            std = math.sqrt(self._variance(r_, nr[i], nr[i + 1]))
            if abs(unsmooth_r_star - smooth_r_star) <= 1.96 * std:
                self._switch_at = r_
                break

    def _variance(self, r, nr, nr_1):
        r = float(r)
        nr = float(nr)
        nr_1 = float(nr_1)
        return (r + 1.0) ** 2 * (nr_1 / nr**2) * (1.0 + nr_1 / nr)

    def _renormalize(self, r, nr):
        """
        It is necessary to renormalize all the probability estimates to
        ensure a proper probability distribution results. This can be done
        by keeping the estimate of the probability mass for unseen items as
        N(1)/N and renormalizing all the estimates for previously seen items
        (as Gale and Sampson (1995) propose). (See M&S P.213, 1999)
        """
        prob_cov = 0.0
        for r_, nr_ in zip(r, nr):
            prob_cov += nr_ * self._prob_measure(r_)
        if prob_cov:
            self._renormal = (1 - self._prob_measure(0)) / prob_cov

    def smoothedNr(self, r):
        """
        Return the number of samples with count r.

        :param r: The amount of frequency.
        :type r: int
        :rtype: float
        """

        # Nr = a*r^b (with b < -1 to give the appropriate hyperbolic
        # relationship)
        # Estimate a and b by simple linear regression technique on
        # the logarithmic form of the equation: log Nr = a + b*log(r)

        return math.exp(self._intercept + self._slope * math.log(r))

    def prob(self, sample):
        """
        Return the sample's probability.

        :param sample: sample of the event
        :type sample: str
        :rtype: float
        """
        count = self._freqdist[sample]
        p = self._prob_measure(count)
        if count == 0:
            if self._bins == self._freqdist.B():
                p = 0.0
            else:
                p = p / (self._bins - self._freqdist.B())
        else:
            p = p * self._renormal
        return p

    def _prob_measure(self, count):
        if count == 0 and self._freqdist.N() == 0:
            return 1.0
        elif count == 0 and self._freqdist.N() != 0:
            return self._freqdist.Nr(1) / self._freqdist.N()

        if self._switch_at > count:
            Er_1 = self._freqdist.Nr(count + 1)
            Er = self._freqdist.Nr(count)
        else:
            Er_1 = self.smoothedNr(count + 1)
            Er = self.smoothedNr(count)

        r_star = (count + 1) * Er_1 / Er
        return r_star / self._freqdist.N()

    def check(self):
        prob_sum = 0.0
        for i in range(0, len(self._Nr)):
            prob_sum += self._Nr[i] * self._prob_measure(i) / self._renormal
        print("Probability Sum:", prob_sum)
        # assert prob_sum != 1.0, "probability sum should be one!"

    def discount(self):
        """
        This function returns the total mass of probability transfers from the
        seen samples to the unseen samples.
        """
        return self.smoothedNr(1) / self._freqdist.N()

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def freqdist(self):
        return self._freqdist

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return "<SimpleGoodTuringProbDist based on %d samples>" % self._freqdist.N()


class MutableProbDist(ProbDistI):
    """
    An mutable probdist where the probabilities may be easily modified. This
    simply copies an existing probdist, storing the probability values in a
    mutable dictionary and providing an update method.
    """

    def __init__(self, prob_dist, samples, store_logs=True):
        """
        Creates the mutable probdist based on the given prob_dist and using
        the list of samples given. These values are stored as log
        probabilities if the store_logs flag is set.

        :param prob_dist: the distribution from which to garner the
            probabilities
        :type prob_dist: ProbDist
        :param samples: the complete set of samples
        :type samples: sequence of any
        :param store_logs: whether to store the probabilities as logarithms
        :type store_logs: bool
        """
        self._samples = samples
        self._sample_dict = {samples[i]: i for i in range(len(samples))}
        self._data = array.array("d", [0.0]) * len(samples)
        for i in range(len(samples)):
            if store_logs:
                self._data[i] = prob_dist.logprob(samples[i])
            else:
                self._data[i] = prob_dist.prob(samples[i])
        self._logs = store_logs

    def max(self):
        # inherit documentation
        return max((p, v) for (v, p) in self._sample_dict.items())[1]

    def samples(self):
        # inherit documentation
        return self._samples

    def prob(self, sample):
        # inherit documentation
        i = self._sample_dict.get(sample)
        if i is None:
            return 0.0
        return 2 ** (self._data[i]) if self._logs else self._data[i]

    def logprob(self, sample):
        # inherit documentation
        i = self._sample_dict.get(sample)
        if i is None:
            return float("-inf")
        return self._data[i] if self._logs else math.log(self._data[i], 2)

    def update(self, sample, prob, log=True):
        """
        Update the probability for the given sample. This may cause the object
        to stop being the valid probability distribution - the user must
        ensure that they update the sample probabilities such that all samples
        have probabilities between 0 and 1 and that all probabilities sum to
        one.

        :param sample: the sample for which to update the probability
        :type sample: any
        :param prob: the new probability
        :type prob: float
        :param log: is the probability already logged
        :type log: bool
        """
        i = self._sample_dict.get(sample)
        assert i is not None
        if self._logs:
            self._data[i] = prob if log else math.log(prob, 2)
        else:
            self._data[i] = 2 ** (prob) if log else prob


##/////////////////////////////////////////////////////
##  Kneser-Ney Probability Distribution
##//////////////////////////////////////////////////////

# This method for calculating probabilities was introduced in 1995 by Reinhard
# Kneser and Hermann Ney. It was meant to improve the accuracy of language
# models that use backing-off to deal with sparse data. The authors propose two
# ways of doing so: a marginal distribution constraint on the back-off
# distribution and a leave-one-out distribution. For a start, the first one is
# implemented as a class below.
#
# The idea behind a back-off n-gram model is that we have a series of
# frequency distributions for our n-grams so that in case we have not seen a
# given n-gram during training (and as a result have a 0 probability for it) we
# can 'back off' (hence the name!) and try testing whether we've seen the
# n-1-gram part of the n-gram in training.
#
# The novelty of Kneser and Ney's approach was that they decided to fiddle
# around with the way this latter, backed off probability was being calculated
# whereas their peers seemed to focus on the primary probability.
#
# The implementation below uses one of the techniques described in their paper
# titled "Improved backing-off for n-gram language modeling." In the same paper
# another technique is introduced to attempt to smooth the back-off
# distribution as well as the primary one. There is also a much-cited
# modification of this method proposed by Chen and Goodman.
#
# In order for the implementation of Kneser-Ney to be more efficient, some
# changes have been made to the original algorithm. Namely, the calculation of
# the normalizing function gamma has been significantly simplified and
# combined slightly differently with beta. None of these changes affect the
# nature of the algorithm, but instead aim to cut out unnecessary calculations
# and take advantage of storing and retrieving information in dictionaries
# where possible.


class KneserNeyProbDist(ProbDistI):
    """
    Kneser-Ney estimate of a probability distribution. This is a version of
    back-off that counts how likely an n-gram is provided the n-1-gram had
    been seen in training. Extends the ProbDistI interface, requires a trigram
    FreqDist instance to train on. Optionally, a different from default discount
    value can be specified. The default discount is set to 0.75.

    """

    def __init__(self, freqdist, bins=None, discount=0.75):
        """
        :param freqdist: The trigram frequency distribution upon which to base
            the estimation
        :type freqdist: FreqDist
        :param bins: Included for compatibility with nltk.tag.hmm
        :type bins: int or float
        :param discount: The discount applied when retrieving counts of
            trigrams
        :type discount: float (preferred, but can be set to int)
        """

        if not bins:
            self._bins = freqdist.B()
        else:
            self._bins = bins
        self._D = discount

        # cache for probability calculation
        self._cache = {}

        # internal bigram and trigram frequency distributions
        self._bigrams = defaultdict(int)
        self._trigrams = freqdist

        # helper dictionaries used to calculate probabilities
        self._wordtypes_after = defaultdict(float)
        self._trigrams_contain = defaultdict(float)
        self._wordtypes_before = defaultdict(float)
        for w0, w1, w2 in freqdist:
            self._bigrams[(w0, w1)] += freqdist[(w0, w1, w2)]
            self._wordtypes_after[(w0, w1)] += 1
            self._trigrams_contain[w1] += 1
            self._wordtypes_before[(w1, w2)] += 1

    def prob(self, trigram):
        # sample must be a triple
        if len(trigram) != 3:
            raise ValueError("Expected an iterable with 3 members.")
        trigram = tuple(trigram)
        w0, w1, w2 = trigram

        if trigram in self._cache:
            return self._cache[trigram]
        else:
            # if the sample trigram was seen during training
            if trigram in self._trigrams:
                prob = (self._trigrams[trigram] - self.discount()) / self._bigrams[
                    (w0, w1)
                ]

            # else if the 'rougher' environment was seen during training
            elif (w0, w1) in self._bigrams and (w1, w2) in self._wordtypes_before:
                aftr = self._wordtypes_after[(w0, w1)]
                bfr = self._wordtypes_before[(w1, w2)]

                # the probability left over from alphas
                leftover_prob = (aftr * self.discount()) / self._bigrams[(w0, w1)]

                # the beta (including normalization)
                beta = bfr / (self._trigrams_contain[w1] - aftr)

                prob = leftover_prob * beta

            # else the sample was completely unseen during training
            else:
                prob = 0.0

            self._cache[trigram] = prob
            return prob

    def discount(self):
        """
        Return the value by which counts are discounted. By default set to 0.75.

        :rtype: float
        """
        return self._D

    def set_discount(self, discount):
        """
        Set the value by which counts are discounted to the value of discount.

        :param discount: the new value to discount counts by
        :type discount: float (preferred, but int possible)
        :rtype: None
        """
        self._D = discount

    def samples(self):
        return self._trigrams.keys()

    def max(self):
        return self._trigrams.max()

    def __repr__(self):
        """
        Return a string representation of this ProbDist

        :rtype: str
        """
        return f"<KneserNeyProbDist based on {self._trigrams.N()} trigrams"


##//////////////////////////////////////////////////////
##  Probability Distribution Operations
##//////////////////////////////////////////////////////


def log_likelihood(test_pdist, actual_pdist):
    if not isinstance(test_pdist, ProbDistI) or not isinstance(actual_pdist, ProbDistI):
        raise ValueError("expected a ProbDist.")
    # Is this right?
    return sum(
        actual_pdist.prob(s) * math.log(test_pdist.prob(s), 2) for s in actual_pdist
    )


def entropy(pdist):
    probs = (pdist.prob(s) for s in pdist.samples())
    return -sum(p * math.log(p, 2) for p in probs)


##//////////////////////////////////////////////////////
##  Conditional Distributions
##//////////////////////////////////////////////////////


class ConditionalFreqDist(defaultdict):
    """
    A collection of frequency distributions for a single experiment
    run under different conditions.  Conditional frequency
    distributions are used to record the number of times each sample
    occurred, given the condition under which the experiment was run.
    For example, a conditional frequency distribution could be used to
    record the frequency of each word (type) in a document, given its
    length.  Formally, a conditional frequency distribution can be
    defined as a function that maps from each condition to the
    FreqDist for the experiment under that condition.

    Conditional frequency distributions are typically constructed by
    repeatedly running an experiment under a variety of conditions,
    and incrementing the sample outcome counts for the appropriate
    conditions.  For example, the following code will produce a
    conditional frequency distribution that encodes how often each
    word type occurs, given the length of that word type:

        >>> from nltk.probability import ConditionalFreqDist
        >>> from nltk.tokenize import word_tokenize
        >>> sent = "the the the dog dog some other words that we do not care about"
        >>> cfdist = ConditionalFreqDist()
        >>> for word in word_tokenize(sent):
        ...     condition = len(word)
        ...     cfdist[condition][word] += 1

    An equivalent way to do this is with the initializer:

        >>> cfdist = ConditionalFreqDist((len(word), word) for word in word_tokenize(sent))

    The frequency distribution for each condition is accessed using
    the indexing operator:

        >>> cfdist[3]
        FreqDist({'the': 3, 'dog': 2, 'not': 1})
        >>> cfdist[3].freq('the')
        0.5
        >>> cfdist[3]['dog']
        2

    When the indexing operator is used to access the frequency
    distribution for a condition that has not been accessed before,
    ``ConditionalFreqDist`` creates a new empty FreqDist for that
    condition.

    """

    def __init__(self, cond_samples=None):
        """
        Construct a new empty conditional frequency distribution.  In
        particular, the count for every sample, under every condition,
        is zero.

        :param cond_samples: The samples to initialize the conditional
            frequency distribution with
        :type cond_samples: Sequence of (condition, sample) tuples
        """
        defaultdict.__init__(self, FreqDist)

        if cond_samples:
            for (cond, sample) in cond_samples:
                self[cond][sample] += 1

    def __reduce__(self):
        kv_pairs = ((cond, self[cond]) for cond in self.conditions())
        return (self.__class__, (), None, None, kv_pairs)

    def conditions(self):
        """
        Return a list of the conditions that have been accessed for
        this ``ConditionalFreqDist``.  Use the indexing operator to
        access the frequency distribution for a given condition.
        Note that the frequency distributions for some conditions
        may contain zero sample outcomes.

        :rtype: list
        """
        return list(self.keys())

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this ``ConditionalFreqDist``.

        :rtype: int
        """
        return sum(fdist.N() for fdist in self.values())

    def plot(
        self,
        *args,
        samples=None,
        title="",
        cumulative=False,
        percents=False,
        conditions=None,
        show=True,
        **kwargs,
    ):
        """
        Plot the given samples from the conditional frequency distribution.
        For a cumulative plot, specify cumulative=True. Additional ``*args`` and
        ``**kwargs`` are passed to matplotlib's plot function.
        (Requires Matplotlib to be installed.)

        :param samples: The samples to plot
        :type samples: list
        :param title: The title for the graph
        :type title: str
        :param cumulative: Whether the plot is cumulative. (default = False)
        :type cumulative: bool
        :param percents: Whether the plot uses percents instead of counts. (default = False)
        :type percents: bool
        :param conditions: The conditions to plot (default is all)
        :type conditions: list
        :param show: Whether to show the plot, or only return the ax.
        :type show: bool
        """
        try:
            import matplotlib.pyplot as plt  # import statement fix
        except ImportError as e:
            raise ValueError(
                "The plot function requires matplotlib to be installed."
                "See https://matplotlib.org/"
            ) from e

        if not conditions:
            conditions = self.conditions()
        else:
            conditions = [c for c in conditions if c in self]
        if not samples:
            samples = sorted({v for c in conditions for v in self[c]})
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 2
        ax = plt.gca()
        if conditions:
            freqs = []
            for condition in conditions:
                if cumulative:
                    # freqs should be a list of list where each sub list will be a frequency of a condition
                    freq = list(self[condition]._cumulative_frequencies(samples))
                else:
                    freq = [self[condition][sample] for sample in samples]

                if percents:
                    freq = [f / self[condition].N() * 100 for f in freq]

                freqs.append(freq)

            if cumulative:
                ylabel = "Cumulative "
                legend_loc = "lower right"
            else:
                ylabel = ""
                legend_loc = "upper right"

            if percents:
                ylabel += "Percents"
            else:
                ylabel += "Counts"

            i = 0
            for freq in freqs:
                kwargs["label"] = conditions[i]  # label for each condition
                i += 1
                ax.plot(freq, *args, **kwargs)
            ax.legend(loc=legend_loc)
            ax.grid(True, color="silver")
            ax.set_xticks(range(len(samples)))
            ax.set_xticklabels([str(s) for s in samples], rotation=90)
            if title:
                ax.set_title(title)
            ax.set_xlabel("Samples")
            ax.set_ylabel(ylabel)

        if show:
            plt.show()

        return ax

    def tabulate(self, *args, **kwargs):
        """
        Tabulate the given samples from the conditional frequency distribution.

        :param samples: The samples to plot
        :type samples: list
        :param conditions: The conditions to plot (default is all)
        :type conditions: list
        :param cumulative: A flag to specify whether the freqs are cumulative (default = False)
        :type title: bool
        """

        cumulative = _get_kwarg(kwargs, "cumulative", False)
        conditions = _get_kwarg(kwargs, "conditions", sorted(self.conditions()))
        samples = _get_kwarg(
            kwargs,
            "samples",
            sorted({v for c in conditions if c in self for v in self[c]}),
        )  # this computation could be wasted

        width = max(len("%s" % s) for s in samples)
        freqs = dict()
        for c in conditions:
            if cumulative:
                freqs[c] = list(self[c]._cumulative_frequencies(samples))
            else:
                freqs[c] = [self[c][sample] for sample in samples]
            width = max(width, max(len("%d" % f) for f in freqs[c]))

        condition_size = max(len("%s" % c) for c in conditions)
        print(" " * condition_size, end=" ")
        for s in samples:
            print("%*s" % (width, s), end=" ")
        print()
        for c in conditions:
            print("%*s" % (condition_size, c), end=" ")
            for f in freqs[c]:
                print("%*d" % (width, f), end=" ")
            print()

    # Mathematical operators

    def __add__(self, other):
        """
        Add counts from two ConditionalFreqDists.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = self.copy()
        for cond in other.conditions():
            result[cond] += other[cond]
        return result

    def __sub__(self, other):
        """
        Subtract count, but keep only results with positive counts.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = self.copy()
        for cond in other.conditions():
            result[cond] -= other[cond]
            if not result[cond]:
                del result[cond]
        return result

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = self.copy()
        for cond in other.conditions():
            result[cond] |= other[cond]
        return result

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = ConditionalFreqDist()
        for cond in self.conditions():
            newfreqdist = self[cond] & other[cond]
            if newfreqdist:
                result[cond] = newfreqdist
        return result

    # @total_ordering doesn't work here, since the class inherits from a builtin class
    def __le__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types("<=", self, other)
        return set(self.conditions()).issubset(other.conditions()) and all(
            self[c] <= other[c] for c in self.conditions()
        )

    def __lt__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types("<", self, other)
        return self <= other and self != other

    def __ge__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types(">=", self, other)
        return other <= self

    def __gt__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types(">", self, other)
        return other < self

    def deepcopy(self):
        from copy import deepcopy

        return deepcopy(self)

    copy = deepcopy

    def __repr__(self):
        """
        Return a string representation of this ``ConditionalFreqDist``.

        :rtype: str
        """
        return "<ConditionalFreqDist with %d conditions>" % len(self)


class ConditionalProbDistI(dict, metaclass=ABCMeta):
    """
    A collection of probability distributions for a single experiment
    run under different conditions.  Conditional probability
    distributions are used to estimate the likelihood of each sample,
    given the condition under which the experiment was run.  For
    example, a conditional probability distribution could be used to
    estimate the probability of each word type in a document, given
    the length of the word type.  Formally, a conditional probability
    distribution can be defined as a function that maps from each
    condition to the ``ProbDist`` for the experiment under that
    condition.
    """

    @abstractmethod
    def __init__(self):
        """
        Classes inheriting from ConditionalProbDistI should implement __init__.
        """

    def conditions(self):
        """
        Return a list of the conditions that are represented by
        this ``ConditionalProbDist``.  Use the indexing operator to
        access the probability distribution for a given condition.

        :rtype: list
        """
        return list(self.keys())

    def __repr__(self):
        """
        Return a string representation of this ``ConditionalProbDist``.

        :rtype: str
        """
        return "<%s with %d conditions>" % (type(self).__name__, len(self))


class ConditionalProbDist(ConditionalProbDistI):
    """
    A conditional probability distribution modeling the experiments
    that were used to generate a conditional frequency distribution.
    A ConditionalProbDist is constructed from a
    ``ConditionalFreqDist`` and a ``ProbDist`` factory:

    - The ``ConditionalFreqDist`` specifies the frequency
      distribution for each condition.
    - The ``ProbDist`` factory is a function that takes a
      condition's frequency distribution, and returns its
      probability distribution.  A ``ProbDist`` class's name (such as
      ``MLEProbDist`` or ``HeldoutProbDist``) can be used to specify
      that class's constructor.

    The first argument to the ``ProbDist`` factory is the frequency
    distribution that it should model; and the remaining arguments are
    specified by the ``factory_args`` parameter to the
    ``ConditionalProbDist`` constructor.  For example, the following
    code constructs a ``ConditionalProbDist``, where the probability
    distribution for each condition is an ``ELEProbDist`` with 10 bins:

        >>> from nltk.corpus import brown
        >>> from nltk.probability import ConditionalFreqDist
        >>> from nltk.probability import ConditionalProbDist, ELEProbDist
        >>> cfdist = ConditionalFreqDist(brown.tagged_words()[:5000])
        >>> cpdist = ConditionalProbDist(cfdist, ELEProbDist, 10)
        >>> cpdist['passed'].max()
        'VBD'
        >>> cpdist['passed'].prob('VBD') #doctest: +ELLIPSIS
        0.423...

    """

    def __init__(self, cfdist, probdist_factory, *factory_args, **factory_kw_args):
        """
        Construct a new conditional probability distribution, based on
        the given conditional frequency distribution and ``ProbDist``
        factory.

        :type cfdist: ConditionalFreqDist
        :param cfdist: The ``ConditionalFreqDist`` specifying the
            frequency distribution for each condition.
        :type probdist_factory: class or function
        :param probdist_factory: The function or class that maps
            a condition's frequency distribution to its probability
            distribution.  The function is called with the frequency
            distribution as its first argument,
            ``factory_args`` as its remaining arguments, and
            ``factory_kw_args`` as keyword arguments.
        :type factory_args: (any)
        :param factory_args: Extra arguments for ``probdist_factory``.
            These arguments are usually used to specify extra
            properties for the probability distributions of individual
            conditions, such as the number of bins they contain.
        :type factory_kw_args: (any)
        :param factory_kw_args: Extra keyword arguments for ``probdist_factory``.
        """
        self._probdist_factory = probdist_factory
        self._factory_args = factory_args
        self._factory_kw_args = factory_kw_args

        for condition in cfdist:
            self[condition] = probdist_factory(
                cfdist[condition], *factory_args, **factory_kw_args
            )

    def __missing__(self, key):
        self[key] = self._probdist_factory(
            FreqDist(), *self._factory_args, **self._factory_kw_args
        )
        return self[key]


class DictionaryConditionalProbDist(ConditionalProbDistI):
    """
    An alternative ConditionalProbDist that simply wraps a dictionary of
    ProbDists rather than creating these from FreqDists.
    """

    def __init__(self, probdist_dict):
        """
        :param probdist_dict: a dictionary containing the probdists indexed
            by the conditions
        :type probdist_dict: dict any -> probdist
        """
        self.update(probdist_dict)

    def __missing__(self, key):
        self[key] = DictionaryProbDist()
        return self[key]


##//////////////////////////////////////////////////////
## Adding in log-space.
##//////////////////////////////////////////////////////

# If the difference is bigger than this, then just take the bigger one:
_ADD_LOGS_MAX_DIFF = math.log(1e-30, 2)


def add_logs(logx, logy):
    """
    Given two numbers ``logx`` = *log(x)* and ``logy`` = *log(y)*, return
    *log(x+y)*.  Conceptually, this is the same as returning
    ``log(2**(logx)+2**(logy))``, but the actual implementation
    avoids overflow errors that could result from direct computation.
    """
    if logx < logy + _ADD_LOGS_MAX_DIFF:
        return logy
    if logy < logx + _ADD_LOGS_MAX_DIFF:
        return logx
    base = min(logx, logy)
    return base + math.log(2 ** (logx - base) + 2 ** (logy - base), 2)


def sum_logs(logs):
    return reduce(add_logs, logs[1:], logs[0]) if len(logs) != 0 else _NINF


##//////////////////////////////////////////////////////
##  Probabilistic Mix-in
##//////////////////////////////////////////////////////


class ProbabilisticMixIn:
    """
    A mix-in class to associate probabilities with other classes
    (trees, rules, etc.).  To use the ``ProbabilisticMixIn`` class,
    define a new class that derives from an existing class and from
    ProbabilisticMixIn.  You will need to define a new constructor for
    the new class, which explicitly calls the constructors of both its
    parent classes.  For example:

        >>> from nltk.probability import ProbabilisticMixIn
        >>> class A:
        ...     def __init__(self, x, y): self.data = (x,y)
        ...
        >>> class ProbabilisticA(A, ProbabilisticMixIn):
        ...     def __init__(self, x, y, **prob_kwarg):
        ...         A.__init__(self, x, y)
        ...         ProbabilisticMixIn.__init__(self, **prob_kwarg)

    See the documentation for the ProbabilisticMixIn
    ``constructor<__init__>`` for information about the arguments it
    expects.

    You should generally also redefine the string representation
    methods, the comparison methods, and the hashing method.
    """

    def __init__(self, **kwargs):
        """
        Initialize this object's probability.  This initializer should
        be called by subclass constructors.  ``prob`` should generally be
        the first argument for those constructors.

        :param prob: The probability associated with the object.
        :type prob: float
        :param logprob: The log of the probability associated with
            the object.
        :type logprob: float
        """
        if "prob" in kwargs:
            if "logprob" in kwargs:
                raise TypeError("Must specify either prob or logprob " "(not both)")
            else:
                ProbabilisticMixIn.set_prob(self, kwargs["prob"])
        elif "logprob" in kwargs:
            ProbabilisticMixIn.set_logprob(self, kwargs["logprob"])
        else:
            self.__prob = self.__logprob = None

    def set_prob(self, prob):
        """
        Set the probability associated with this object to ``prob``.

        :param prob: The new probability
        :type prob: float
        """
        self.__prob = prob
        self.__logprob = None

    def set_logprob(self, logprob):
        """
        Set the log probability associated with this object to
        ``logprob``.  I.e., set the probability associated with this
        object to ``2**(logprob)``.

        :param logprob: The new log probability
        :type logprob: float
        """
        self.__logprob = logprob
        self.__prob = None

    def prob(self):
        """
        Return the probability associated with this object.

        :rtype: float
        """
        if self.__prob is None:
            if self.__logprob is None:
                return None
            self.__prob = 2 ** (self.__logprob)
        return self.__prob

    def logprob(self):
        """
        Return ``log(p)``, where ``p`` is the probability associated
        with this object.

        :rtype: float
        """
        if self.__logprob is None:
            if self.__prob is None:
                return None
            self.__logprob = math.log(self.__prob, 2)
        return self.__logprob


class ImmutableProbabilisticMixIn(ProbabilisticMixIn):
    def set_prob(self, prob):
        raise ValueError("%s is immutable" % self.__class__.__name__)

    def set_logprob(self, prob):
        raise ValueError("%s is immutable" % self.__class__.__name__)


## Helper function for processing keyword arguments


def _get_kwarg(kwargs, key, default):
    if key in kwargs:
        arg = kwargs[key]
        del kwargs[key]
    else:
        arg = default
    return arg


##//////////////////////////////////////////////////////
##  Demonstration
##//////////////////////////////////////////////////////


def _create_rand_fdist(numsamples, numoutcomes):
    """
    Create a new frequency distribution, with random samples.  The
    samples are numbers from 1 to ``numsamples``, and are generated by
    summing two numbers, each of which has a uniform distribution.
    """

    fdist = FreqDist()
    for x in range(numoutcomes):
        y = random.randint(1, (1 + numsamples) // 2) + random.randint(
            0, numsamples // 2
        )
        fdist[y] += 1
    return fdist


def _create_sum_pdist(numsamples):
    """
    Return the true probability distribution for the experiment
    ``_create_rand_fdist(numsamples, x)``.
    """
    fdist = FreqDist()
    for x in range(1, (1 + numsamples) // 2 + 1):
        for y in range(0, numsamples // 2 + 1):
            fdist[x + y] += 1
    return MLEProbDist(fdist)


def demo(numsamples=6, numoutcomes=500):
    """
    A demonstration of frequency distributions and probability
    distributions.  This demonstration creates three frequency
    distributions with, and uses them to sample a random process with
    ``numsamples`` samples.  Each frequency distribution is sampled
    ``numoutcomes`` times.  These three frequency distributions are
    then used to build six probability distributions.  Finally, the
    probability estimates of these distributions are compared to the
    actual probability of each sample.

    :type numsamples: int
    :param numsamples: The number of samples to use in each demo
        frequency distributions.
    :type numoutcomes: int
    :param numoutcomes: The total number of outcomes for each
        demo frequency distribution.  These outcomes are divided into
        ``numsamples`` bins.
    :rtype: None
    """

    # Randomly sample a stochastic process three times.
    fdist1 = _create_rand_fdist(numsamples, numoutcomes)
    fdist2 = _create_rand_fdist(numsamples, numoutcomes)
    fdist3 = _create_rand_fdist(numsamples, numoutcomes)

    # Use our samples to create probability distributions.
    pdists = [
        MLEProbDist(fdist1),
        LidstoneProbDist(fdist1, 0.5, numsamples),
        HeldoutProbDist(fdist1, fdist2, numsamples),
        HeldoutProbDist(fdist2, fdist1, numsamples),
        CrossValidationProbDist([fdist1, fdist2, fdist3], numsamples),
        SimpleGoodTuringProbDist(fdist1),
        SimpleGoodTuringProbDist(fdist1, 7),
        _create_sum_pdist(numsamples),
    ]

    # Find the probability of each sample.
    vals = []
    for n in range(1, numsamples + 1):
        vals.append(tuple([n, fdist1.freq(n)] + [pdist.prob(n) for pdist in pdists]))

    # Print the results in a formatted table.
    print(
        "%d samples (1-%d); %d outcomes were sampled for each FreqDist"
        % (numsamples, numsamples, numoutcomes)
    )
    print("=" * 9 * (len(pdists) + 2))
    FORMATSTR = "      FreqDist " + "%8s " * (len(pdists) - 1) + "|  Actual"
    print(FORMATSTR % tuple(repr(pdist)[1:9] for pdist in pdists[:-1]))
    print("-" * 9 * (len(pdists) + 2))
    FORMATSTR = "%3d   %8.6f " + "%8.6f " * (len(pdists) - 1) + "| %8.6f"
    for val in vals:
        print(FORMATSTR % val)

    # Print the totals for each column (should all be 1.0)
    zvals = list(zip(*vals))
    sums = [sum(val) for val in zvals[1:]]
    print("-" * 9 * (len(pdists) + 2))
    FORMATSTR = "Total " + "%8.6f " * (len(pdists)) + "| %8.6f"
    print(FORMATSTR % tuple(sums))
    print("=" * 9 * (len(pdists) + 2))

    # Display the distributions themselves, if they're short enough.
    if len("%s" % fdist1) < 70:
        print("  fdist1: %s" % fdist1)
        print("  fdist2: %s" % fdist2)
        print("  fdist3: %s" % fdist3)
    print()

    print("Generating:")
    for pdist in pdists:
        fdist = FreqDist(pdist.generate() for i in range(5000))
        print("{:>20} {}".format(pdist.__class__.__name__[:20], ("%s" % fdist)[:55]))
    print()


def gt_demo():
    from nltk import corpus

    emma_words = corpus.gutenberg.words("austen-emma.txt")
    fd = FreqDist(emma_words)
    sgt = SimpleGoodTuringProbDist(fd)
    print("{:>18} {:>8}  {:>14}".format("word", "frequency", "SimpleGoodTuring"))
    fd_keys_sorted = (
        key for key, value in sorted(fd.items(), key=lambda item: item[1], reverse=True)
    )
    for key in fd_keys_sorted:
        print("%18s %8d  %14e" % (key, fd[key], sgt.prob(key)))


if __name__ == "__main__":
    demo(6, 10)
    demo(5, 5000)
    gt_demo()

__all__ = [
    "ConditionalFreqDist",
    "ConditionalProbDist",
    "ConditionalProbDistI",
    "CrossValidationProbDist",
    "DictionaryConditionalProbDist",
    "DictionaryProbDist",
    "ELEProbDist",
    "FreqDist",
    "SimpleGoodTuringProbDist",
    "HeldoutProbDist",
    "ImmutableProbabilisticMixIn",
    "LaplaceProbDist",
    "LidstoneProbDist",
    "MLEProbDist",
    "MutableProbDist",
    "KneserNeyProbDist",
    "ProbDistI",
    "ProbabilisticMixIn",
    "UniformProbDist",
    "WittenBellProbDist",
    "add_logs",
    "log_likelihood",
    "sum_logs",
    "entropy",
]
