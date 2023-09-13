# License: BSD 3 clause
"""Helper to load LossFunction from sgd_fast.pyx to sag_fast.pyx"""

cdef class LossFunction:
    cdef double loss(self, double p, double y) noexcept nogil
    cdef double dloss(self, double p, double y) noexcept nogil


cdef class Regression(LossFunction):
    cdef double loss(self, double p, double y) noexcept nogil
    cdef double dloss(self, double p, double y) noexcept nogil


cdef class Classification(LossFunction):
    cdef double loss(self, double p, double y) noexcept nogil
    cdef double dloss(self, double p, double y) noexcept nogil


cdef class Log(Classification):
    cdef double loss(self, double p, double y) noexcept nogil
    cdef double dloss(self, double p, double y) noexcept nogil


cdef class SquaredLoss(Regression):
    cdef double loss(self, double p, double y) noexcept nogil
    cdef double dloss(self, double p, double y) noexcept nogil
