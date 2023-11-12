# Natural Language Toolkit: Agreement Metrics
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Tom Lippincott <tom@cs.columbia.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Implementations of inter-annotator agreement coefficients surveyed by Artstein
and Poesio (2007), Inter-Coder Agreement for Computational Linguistics.

An agreement coefficient calculates the amount that annotators agreed on label
assignments beyond what is expected by chance.

In defining the AnnotationTask class, we use naming conventions similar to the
paper's terminology.  There are three types of objects in an annotation task:

    the coders (variables "c" and "C")
    the items to be annotated (variables "i" and "I")
    the potential categories to be assigned (variables "k" and "K")

Additionally, it is often the case that we don't want to treat two different
labels as complete disagreement, and so the AnnotationTask constructor can also
take a distance metric as a final argument.  Distance metrics are simply
functions that take two arguments, and return a value between 0.0 and 1.0
indicating the distance between them.  If not supplied, the default is binary
comparison between the arguments.

The simplest way to initialize an AnnotationTask is with a list of triples,
each containing a coder's assignment for one object in the task:

    task = AnnotationTask(data=[('c1', '1', 'v1'),('c2', '1', 'v1'),...])

Note that the data list needs to contain the same number of triples for each
individual coder, containing category values for the same set of items.

Alpha (Krippendorff 1980)
Kappa (Cohen 1960)
S (Bennet, Albert and Goldstein 1954)
Pi (Scott 1955)


TODO: Describe handling of multiple coders and missing data

Expected results from the Artstein and Poesio survey paper:

    >>> from nltk.metrics.agreement import AnnotationTask
    >>> import os.path
    >>> t = AnnotationTask(data=[x.split() for x in open(os.path.join(os.path.dirname(__file__), "artstein_poesio_example.txt"))])
    >>> t.avg_Ao()
    0.88
    >>> round(t.pi(), 5)
    0.79953
    >>> round(t.S(), 2)
    0.82

    This would have returned a wrong value (0.0) in @785fb79 as coders are in
    the wrong order. Subsequently, all values for pi(), S(), and kappa() would
    have been wrong as they are computed with avg_Ao().
    >>> t2 = AnnotationTask(data=[('b','1','stat'),('a','1','stat')])
    >>> t2.avg_Ao()
    1.0

    The following, of course, also works.
    >>> t3 = AnnotationTask(data=[('a','1','othr'),('b','1','othr')])
    >>> t3.avg_Ao()
    1.0

"""

import logging
from itertools import groupby
from operator import itemgetter

from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist

log = logging.getLogger(__name__)


class AnnotationTask:
    """Represents an annotation task, i.e. people assign labels to items.

    Notation tries to match notation in Artstein and Poesio (2007).

    In general, coders and items can be represented as any hashable object.
    Integers, for example, are fine, though strings are more readable.
    Labels must support the distance functions applied to them, so e.g.
    a string-edit-distance makes no sense if your labels are integers,
    whereas interval distance needs numeric values.  A notable case of this
    is the MASI metric, which requires Python sets.
    """

    def __init__(self, data=None, distance=binary_distance):
        """Initialize an annotation task.

        The data argument can be None (to create an empty annotation task) or a sequence of 3-tuples,
        each representing a coder's labeling of an item:
        ``(coder,item,label)``

        The distance argument is a function taking two arguments (labels) and producing a numerical distance.
        The distance from a label to itself should be zero:
        ``distance(l,l) = 0``
        """
        self.distance = distance
        self.I = set()
        self.K = set()
        self.C = set()
        self.data = []
        if data is not None:
            self.load_array(data)

    def __str__(self):
        return "\r\n".join(
            map(
                lambda x: "%s\t%s\t%s"
                % (x["coder"], x["item"].replace("_", "\t"), ",".join(x["labels"])),
                self.data,
            )
        )

    def load_array(self, array):
        """Load an sequence of annotation results, appending to any data already loaded.

        The argument is a sequence of 3-tuples, each representing a coder's labeling of an item:
            (coder,item,label)
        """
        for coder, item, labels in array:
            self.C.add(coder)
            self.K.add(labels)
            self.I.add(item)
            self.data.append({"coder": coder, "labels": labels, "item": item})

    def agr(self, cA, cB, i, data=None):
        """Agreement between two coders on a given item"""
        data = data or self.data
        # cfedermann: we don't know what combination of coder/item will come
        # first in x; to avoid StopIteration problems due to assuming an order
        # cA,cB, we allow either for k1 and then look up the missing as k2.
        k1 = next(x for x in data if x["coder"] in (cA, cB) and x["item"] == i)
        if k1["coder"] == cA:
            k2 = next(x for x in data if x["coder"] == cB and x["item"] == i)
        else:
            k2 = next(x for x in data if x["coder"] == cA and x["item"] == i)

        ret = 1.0 - float(self.distance(k1["labels"], k2["labels"]))
        log.debug("Observed agreement between %s and %s on %s: %f", cA, cB, i, ret)
        log.debug(
            'Distance between "%r" and "%r": %f', k1["labels"], k2["labels"], 1.0 - ret
        )
        return ret

    def Nk(self, k):
        return float(sum(1 for x in self.data if x["labels"] == k))

    def Nik(self, i, k):
        return float(sum(1 for x in self.data if x["item"] == i and x["labels"] == k))

    def Nck(self, c, k):
        return float(sum(1 for x in self.data if x["coder"] == c and x["labels"] == k))

    @deprecated("Use Nk, Nik or Nck instead")
    def N(self, k=None, i=None, c=None):
        """Implements the "n-notation" used in Artstein and Poesio (2007)"""
        if k is not None and i is None and c is None:
            ret = self.Nk(k)
        elif k is not None and i is not None and c is None:
            ret = self.Nik(i, k)
        elif k is not None and c is not None and i is None:
            ret = self.Nck(c, k)
        else:
            raise ValueError(
                f"You must pass either i or c, not both! (k={k!r},i={i!r},c={c!r})"
            )
        log.debug("Count on N[%s,%s,%s]: %d", k, i, c, ret)
        return ret

    def _grouped_data(self, field, data=None):
        data = data or self.data
        return groupby(sorted(data, key=itemgetter(field)), itemgetter(field))

    def Ao(self, cA, cB):
        """Observed agreement between two coders on all items."""
        data = self._grouped_data(
            "item", (x for x in self.data if x["coder"] in (cA, cB))
        )
        ret = sum(self.agr(cA, cB, item, item_data) for item, item_data in data) / len(
            self.I
        )
        log.debug("Observed agreement between %s and %s: %f", cA, cB, ret)
        return ret

    def _pairwise_average(self, function):
        """
        Calculates the average of function results for each coder pair
        """
        total = 0
        n = 0
        s = self.C.copy()
        for cA in self.C:
            s.remove(cA)
            for cB in s:
                total += function(cA, cB)
                n += 1
        ret = total / n
        return ret

    def avg_Ao(self):
        """Average observed agreement across all coders and items."""
        ret = self._pairwise_average(self.Ao)
        log.debug("Average observed agreement: %f", ret)
        return ret

    def Do_Kw_pairwise(self, cA, cB, max_distance=1.0):
        """The observed disagreement for the weighted kappa coefficient."""
        total = 0.0
        data = (x for x in self.data if x["coder"] in (cA, cB))
        for i, itemdata in self._grouped_data("item", data):
            # we should have two items; distance doesn't care which comes first
            total += self.distance(next(itemdata)["labels"], next(itemdata)["labels"])

        ret = total / (len(self.I) * max_distance)
        log.debug("Observed disagreement between %s and %s: %f", cA, cB, ret)
        return ret

    def Do_Kw(self, max_distance=1.0):
        """Averaged over all labelers"""
        ret = self._pairwise_average(
            lambda cA, cB: self.Do_Kw_pairwise(cA, cB, max_distance)
        )
        log.debug("Observed disagreement: %f", ret)
        return ret

    # Agreement Coefficients
    def S(self):
        """Bennett, Albert and Goldstein 1954"""
        Ae = 1.0 / len(self.K)
        ret = (self.avg_Ao() - Ae) / (1.0 - Ae)
        return ret

    def pi(self):
        """Scott 1955; here, multi-pi.
        Equivalent to K from Siegel and Castellan (1988).

        """
        total = 0.0
        label_freqs = FreqDist(x["labels"] for x in self.data)
        for k, f in label_freqs.items():
            total += f**2
        Ae = total / ((len(self.I) * len(self.C)) ** 2)
        return (self.avg_Ao() - Ae) / (1 - Ae)

    def Ae_kappa(self, cA, cB):
        Ae = 0.0
        nitems = float(len(self.I))
        label_freqs = ConditionalFreqDist((x["labels"], x["coder"]) for x in self.data)
        for k in label_freqs.conditions():
            Ae += (label_freqs[k][cA] / nitems) * (label_freqs[k][cB] / nitems)
        return Ae

    def kappa_pairwise(self, cA, cB):
        """ """
        Ae = self.Ae_kappa(cA, cB)
        ret = (self.Ao(cA, cB) - Ae) / (1.0 - Ae)
        log.debug("Expected agreement between %s and %s: %f", cA, cB, Ae)
        return ret

    def kappa(self):
        """Cohen 1960
        Averages naively over kappas for each coder pair.

        """
        return self._pairwise_average(self.kappa_pairwise)

    def multi_kappa(self):
        """Davies and Fleiss 1982
        Averages over observed and expected agreements for each coder pair.

        """
        Ae = self._pairwise_average(self.Ae_kappa)
        return (self.avg_Ao() - Ae) / (1.0 - Ae)

    def Disagreement(self, label_freqs):
        total_labels = sum(label_freqs.values())
        pairs = 0.0
        for j, nj in label_freqs.items():
            for l, nl in label_freqs.items():
                pairs += float(nj * nl) * self.distance(l, j)
        return 1.0 * pairs / (total_labels * (total_labels - 1))

    def alpha(self):
        """Krippendorff 1980"""
        # check for degenerate cases
        if len(self.K) == 0:
            raise ValueError("Cannot calculate alpha, no data present!")
        if len(self.K) == 1:
            log.debug("Only one annotation value, alpha returning 1.")
            return 1
        if len(self.C) == 1 and len(self.I) == 1:
            raise ValueError("Cannot calculate alpha, only one coder and item present!")

        total_disagreement = 0.0
        total_ratings = 0
        all_valid_labels_freq = FreqDist([])

        total_do = 0.0  # Total observed disagreement for all items.
        for i, itemdata in self._grouped_data("item"):
            label_freqs = FreqDist(x["labels"] for x in itemdata)
            labels_count = sum(label_freqs.values())
            if labels_count < 2:
                # Ignore the item.
                continue
            all_valid_labels_freq += label_freqs
            total_do += self.Disagreement(label_freqs) * labels_count

        do = total_do / sum(all_valid_labels_freq.values())

        de = self.Disagreement(all_valid_labels_freq)  # Expected disagreement.
        k_alpha = 1.0 - do / de

        return k_alpha

    def weighted_kappa_pairwise(self, cA, cB, max_distance=1.0):
        """Cohen 1968"""
        total = 0.0
        label_freqs = ConditionalFreqDist(
            (x["coder"], x["labels"]) for x in self.data if x["coder"] in (cA, cB)
        )
        for j in self.K:
            for l in self.K:
                total += label_freqs[cA][j] * label_freqs[cB][l] * self.distance(j, l)
        De = total / (max_distance * pow(len(self.I), 2))
        log.debug("Expected disagreement between %s and %s: %f", cA, cB, De)
        Do = self.Do_Kw_pairwise(cA, cB)
        ret = 1.0 - (Do / De)
        return ret

    def weighted_kappa(self, max_distance=1.0):
        """Cohen 1968"""
        return self._pairwise_average(
            lambda cA, cB: self.weighted_kappa_pairwise(cA, cB, max_distance)
        )


if __name__ == "__main__":

    import optparse
    import re

    from nltk.metrics import distance

    # process command-line arguments
    parser = optparse.OptionParser()
    parser.add_option(
        "-d",
        "--distance",
        dest="distance",
        default="binary_distance",
        help="distance metric to use",
    )
    parser.add_option(
        "-a",
        "--agreement",
        dest="agreement",
        default="kappa",
        help="agreement coefficient to calculate",
    )
    parser.add_option(
        "-e",
        "--exclude",
        dest="exclude",
        action="append",
        default=[],
        help="coder names to exclude (may be specified multiple times)",
    )
    parser.add_option(
        "-i",
        "--include",
        dest="include",
        action="append",
        default=[],
        help="coder names to include, same format as exclude",
    )
    parser.add_option(
        "-f",
        "--file",
        dest="file",
        help="file to read labelings from, each line with three columns: 'labeler item labels'",
    )
    parser.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        default="0",
        help="how much debugging to print on stderr (0-4)",
    )
    parser.add_option(
        "-c",
        "--columnsep",
        dest="columnsep",
        default="\t",
        help="char/string that separates the three columns in the file, defaults to tab",
    )
    parser.add_option(
        "-l",
        "--labelsep",
        dest="labelsep",
        default=",",
        help="char/string that separates labels (if labelers can assign more than one), defaults to comma",
    )
    parser.add_option(
        "-p",
        "--presence",
        dest="presence",
        default=None,
        help="convert each labeling into 1 or 0, based on presence of LABEL",
    )
    parser.add_option(
        "-T",
        "--thorough",
        dest="thorough",
        default=False,
        action="store_true",
        help="calculate agreement for every subset of the annotators",
    )
    (options, remainder) = parser.parse_args()

    if not options.file:
        parser.print_help()
        exit()

    logging.basicConfig(level=50 - 10 * int(options.verbose))

    # read in data from the specified file
    data = []
    with open(options.file) as infile:
        for l in infile:
            toks = l.split(options.columnsep)
            coder, object_, labels = (
                toks[0],
                str(toks[1:-1]),
                frozenset(toks[-1].strip().split(options.labelsep)),
            )
            if (
                (options.include == options.exclude)
                or (len(options.include) > 0 and coder in options.include)
                or (len(options.exclude) > 0 and coder not in options.exclude)
            ):
                data.append((coder, object_, labels))

    if options.presence:
        task = AnnotationTask(
            data, getattr(distance, options.distance)(options.presence)
        )
    else:
        task = AnnotationTask(data, getattr(distance, options.distance))

    if options.thorough:
        pass
    else:
        print(getattr(task, options.agreement)())

    logging.shutdown()
