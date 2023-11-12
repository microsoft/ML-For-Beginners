# Natural Language Toolkit: Metrics
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

"""
NLTK Metrics

Classes and methods for scoring processing modules.
"""

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.aline import align
from nltk.metrics.association import (
    BigramAssocMeasures,
    ContingencyMeasures,
    NgramAssocMeasures,
    QuadgramAssocMeasures,
    TrigramAssocMeasures,
)
from nltk.metrics.confusionmatrix import ConfusionMatrix
from nltk.metrics.distance import (
    binary_distance,
    custom_distance,
    edit_distance,
    edit_distance_align,
    fractional_presence,
    interval_distance,
    jaccard_distance,
    masi_distance,
    presence,
)
from nltk.metrics.paice import Paice
from nltk.metrics.scores import (
    accuracy,
    approxrand,
    f_measure,
    log_likelihood,
    precision,
    recall,
)
from nltk.metrics.segmentation import ghd, pk, windowdiff
from nltk.metrics.spearman import (
    ranks_from_scores,
    ranks_from_sequence,
    spearman_correlation,
)
