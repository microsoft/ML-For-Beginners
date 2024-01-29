# This file is part of Patsy
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Stateful transform protocol:
#   def __init__(self):
#       pass
#   def memorize_chunk(self, input_data):
#       return None
#   def memorize_finish(self):
#       return None
#   def transform(self, input_data):
#       return output_data

# BETTER WAY: always run the first row of data through the builder alone, and
# check that it gives the same output row as when running the whole block of
# data through at once. This gives us the same information, but it's robust
# against people writing their own centering functions.

# QUESTION: right now we refuse to even fit a model that contains a
# my_transform(x)-style function. Maybe we should allow it to be fit (with a
# warning), and only disallow making predictions with it? Need to revisit this
# question once it's clearer what exactly our public API will look like,
# because right now I'm not sure how to tell whether we are being called for
# fitting versus being called for prediction.

from functools import wraps
import numpy as np
from patsy.util import (atleast_2d_column_default,
                        asarray_or_pandas, pandas_friendly_reshape,
                        wide_dtype_for, safe_issubdtype,
                        no_pickling, assert_no_pickling)

# These are made available in the patsy.* namespace
__all__ = ["stateful_transform",
           "center", "standardize", "scale",
           ]

def stateful_transform(class_):
    """Create a stateful transform callable object from a class that fulfills
    the :ref:`stateful transform protocol <stateful-transform-protocol>`.
    """
    @wraps(class_)
    def stateful_transform_wrapper(*args, **kwargs):
        transform = class_()
        transform.memorize_chunk(*args, **kwargs)
        transform.memorize_finish()
        return transform.transform(*args, **kwargs)
    stateful_transform_wrapper.__patsy_stateful_transform__ = class_
    return stateful_transform_wrapper

# class NonIncrementalStatefulTransform(object):
#     def __init__(self):
#         self._data = []
#
#     def memorize_chunk(self, input_data, *args, **kwargs):
#         self._data.append(input_data)
#         self._args = _args
#         self._kwargs = kwargs
#
#     def memorize_finish(self):
#         all_data = np.vstack(self._data)
#         args = self._args
#         kwargs = self._kwargs
#         del self._data
#         del self._args
#         del self._kwargs
#         self.memorize_all(all_data, *args, **kwargs)
#
#     def memorize_all(self, input_data, *args, **kwargs):
#         raise NotImplementedError
#
#     def transform(self, input_data, *args, **kwargs):
#         raise NotImplementedError
#
# class QuantileEstimatingTransform(NonIncrementalStatefulTransform):
#     def memorize_all(self, input_data, *args, **kwargs):

class Center(object):
    """center(x)

    A stateful transform that centers input data, i.e., subtracts the mean.

    If input has multiple columns, centers each column separately.

    Equivalent to ``standardize(x, rescale=False)``
    """
    def __init__(self):
        self._sum = None
        self._count = 0

    def memorize_chunk(self, x):
        x = atleast_2d_column_default(x)
        self._count += x.shape[0]
        this_total = np.sum(x, 0, dtype=wide_dtype_for(x))
        # This is to handle potentially multi-column x's:
        if self._sum is None:
            self._sum = this_total
        else:
            self._sum += this_total

    def memorize_finish(self):
        pass

    def transform(self, x):
        x = asarray_or_pandas(x)
        # This doesn't copy data unless our input is a DataFrame that has
        # heterogeneous types. And in that case we're going to be munging the
        # types anyway, so copying isn't a big deal.
        x_arr = np.asarray(x)
        if safe_issubdtype(x_arr.dtype, np.integer):
            dt = float
        else:
            dt = x_arr.dtype
        mean_val = np.asarray(self._sum / self._count, dtype=dt)
        centered = atleast_2d_column_default(x, preserve_pandas=True) - mean_val
        return pandas_friendly_reshape(centered, x.shape)

    __getstate__ = no_pickling

center = stateful_transform(Center)

# See:
#   http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
# or page 232 of Knuth vol. 3 (3rd ed.).
class Standardize(object):
    """standardize(x, center=True, rescale=True, ddof=0)

    A stateful transform that standardizes input data, i.e. it subtracts the
    mean and divides by the sample standard deviation.

    Either centering or rescaling or both can be disabled by use of keyword
    arguments. The `ddof` argument controls the delta degrees of freedom when
    computing the standard deviation (cf. :func:`numpy.std`). The default of
    ``ddof=0`` produces the maximum likelihood estimate; use ``ddof=1`` if you
    prefer the square root of the unbiased estimate of the variance.

    If input has multiple columns, standardizes each column separately.

    .. note:: This function computes the mean and standard deviation using a
       memory-efficient online algorithm, making it suitable for use with
       large incrementally processed data-sets.
    """
    def __init__(self):
        self.current_n = 0
        self.current_mean = None
        self.current_M2 = None

    def memorize_chunk(self, x, center=True, rescale=True, ddof=0):
        x = atleast_2d_column_default(x)
        if self.current_mean is None:
            self.current_mean = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
            self.current_M2 = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
        # XX this can surely be vectorized but I am feeling lazy:
        for i in range(x.shape[0]):
            self.current_n += 1
            delta = x[i, :] - self.current_mean
            self.current_mean += delta / self.current_n
            self.current_M2 += delta * (x[i, :] - self.current_mean)

    def memorize_finish(self):
        pass

    def transform(self, x, center=True, rescale=True, ddof=0):
        # XX: this forces all inputs to double-precision real, even if the
        # input is single- or extended-precision or complex. But I got all
        # tangled up in knots trying to do that without breaking something
        # else (e.g. by requiring an extra copy).
        x = asarray_or_pandas(x, copy=True, dtype=float)
        x_2d = atleast_2d_column_default(x, preserve_pandas=True)
        if center:
            x_2d -= self.current_mean
        if rescale:
            x_2d /= np.sqrt(self.current_M2 / (self.current_n - ddof))
        return pandas_friendly_reshape(x_2d, x.shape)

    __getstate__ = no_pickling

standardize = stateful_transform(Standardize)
# R compatibility:
scale = standardize
