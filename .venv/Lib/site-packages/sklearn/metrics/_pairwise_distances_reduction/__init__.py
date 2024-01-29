#
# Pairwise Distances Reductions
# =============================
#
#   Authors: The scikit-learn developers.
#   License: BSD 3 clause
#
# Overview
# --------
#
#    This module provides routines to compute pairwise distances between a set
#    of row vectors of X and another set of row vectors of Y and apply a
#    reduction on top. The canonical example is the brute-force computation
#    of the top k nearest neighbors by leveraging the arg-k-min reduction.
#
#    The reduction takes a matrix of pairwise distances between rows of X and Y
#    as input and outputs an aggregate data-structure for each row of X. The
#    aggregate values are typically smaller than the number of rows in Y, hence
#    the term reduction.
#
#    For computational reasons, the reduction are performed on the fly on chunks
#    of rows of X and Y so as to keep intermediate data-structures in CPU cache
#    and avoid unnecessary round trips of large distance arrays with the RAM
#    that would otherwise severely degrade the speed by making the overall
#    processing memory-bound.
#
#    Finally, the routines follow a generic parallelization template to process
#    chunks of data with OpenMP loops (via Cython prange), either on rows of X
#    or rows of Y depending on their respective sizes.
#
#
# Dispatching to specialized implementations
# ------------------------------------------
#
#    Dispatchers are meant to be used in the Python code. Under the hood, a
#    dispatcher must only define the logic to choose at runtime to the correct
#    dtype-specialized :class:`BaseDistancesReductionDispatcher` implementation based
#    on the dtype of X and of Y.
#
#
# High-level diagram
# ------------------
#
#    Legend:
#
#      A ---⊳ B: A inherits from B
#      A ---x B: A dispatches to B
#
#
#                                      (base dispatcher)
#                               BaseDistancesReductionDispatcher
#                                              ∆
#                                              |
#                                              |
#           +------------------+---------------+---------------+------------------+
#           |                  |                               |                  |
#           |             (dispatcher)                    (dispatcher)            |
#           |               ArgKmin                      RadiusNeighbors          |
#           |                  |                               |                  |
#           |                  |                               |                  |
#           |                  |     (float{32,64} implem.)    |                  |
#           |                  | BaseDistancesReduction{32,64} |                  |
#           |                  |               ∆               |                  |
#      (dispatcher)            |               |               |             (dispatcher)
#    ArgKminClassMode          |               |               |        RadiusNeighborsClassMode
#           |                  |    +----------+----------+    |                  |
#           |                  |    |                     |    |                  |
#           |                  |    |                     |    |                  |
#           |                  x    |                     |    x                  |
#           |     +-------⊳ ArgKmin{32,64}         RadiusNeighbors{32,64} ⊲---+   |
#           x     |            |    ∆                     ∆    |              |   x
#   ArgKminClassMode{32,64}    |    |                     |    |   RadiusNeighborsClassMode{32,64}
# ===================================== Specializations ============================================
#                              |    |                     |    |
#                              |    |                     |    |
#                              x    |                     |    x
#                      EuclideanArgKmin{32,64}    EuclideanRadiusNeighbors{32,64}
#
#
#    For instance :class:`ArgKmin` dispatches to:
#      - :class:`ArgKmin64` if X and Y are two `float64` array-likes
#      - :class:`ArgKmin32` if X and Y are two `float32` array-likes
#
#    In addition, if the metric parameter is set to "euclidean" or "sqeuclidean",
#    then some direct subclass of `BaseDistancesReduction{32,64}` further dispatches
#    to one of their subclass for euclidean-specialized implementation. For instance,
#    :class:`ArgKmin64` dispatches to :class:`EuclideanArgKmin64`.
#
#    Those Euclidean-specialized implementations relies on optimal implementations of
#    a decomposition of the squared euclidean distance matrix into a sum of three terms
#    (see :class:`MiddleTermComputer{32,64}`).
#

from ._dispatcher import (
    ArgKmin,
    ArgKminClassMode,
    BaseDistancesReductionDispatcher,
    RadiusNeighbors,
    RadiusNeighborsClassMode,
    sqeuclidean_row_norms,
)

__all__ = [
    "BaseDistancesReductionDispatcher",
    "ArgKmin",
    "RadiusNeighbors",
    "ArgKminClassMode",
    "RadiusNeighborsClassMode",
    "sqeuclidean_row_norms",
]

# ruff: noqa: E501
