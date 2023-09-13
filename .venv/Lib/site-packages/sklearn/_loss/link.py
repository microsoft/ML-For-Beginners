"""
Module contains classes for invertible (and differentiable) link functions.
"""
# Author: Christian Lorentzen <lorentzen.ch@gmail.com>

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean

from ..utils.extmath import softmax


@dataclass
class Interval:
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool

    def __post_init__(self):
        """Check that low <= high"""
        if self.low > self.high:
            raise ValueError(
                f"One must have low <= high; got low={self.low}, high={self.high}."
            )

    def includes(self, x):
        """Test whether all values of x are in interval range.

        Parameters
        ----------
        x : ndarray
            Array whose elements are tested to be in interval range.

        Returns
        -------
        result : bool
        """
        if self.low_inclusive:
            low = np.greater_equal(x, self.low)
        else:
            low = np.greater(x, self.low)

        if not np.all(low):
            return False

        if self.high_inclusive:
            high = np.less_equal(x, self.high)
        else:
            high = np.less(x, self.high)

        # Note: np.all returns numpy.bool_
        return bool(np.all(high))


def _inclusive_low_high(interval, dtype=np.float64):
    """Generate values low and high to be within the interval range.

    This is used in tests only.

    Returns
    -------
    low, high : tuple
        The returned values low and high lie within the interval.
    """
    eps = 10 * np.finfo(dtype).eps
    if interval.low == -np.inf:
        low = -1e10
    elif interval.low < 0:
        low = interval.low * (1 - eps) + eps
    else:
        low = interval.low * (1 + eps) + eps

    if interval.high == np.inf:
        high = 1e10
    elif interval.high < 0:
        high = interval.high * (1 + eps) - eps
    else:
        high = interval.high * (1 - eps) - eps

    return low, high


class BaseLink(ABC):
    """Abstract base class for differentiable, invertible link functions.

    Convention:
        - link function g: raw_prediction = g(y_pred)
        - inverse link h: y_pred = h(raw_prediction)

    For (generalized) linear models, `raw_prediction = X @ coef` is the so
    called linear predictor, and `y_pred = h(raw_prediction)` is the predicted
    conditional (on X) expected value of the target `y_true`.

    The methods are not implemented as staticmethods in case a link function needs
    parameters.
    """

    is_multiclass = False  # used for testing only

    # Usually, raw_prediction may be any real number and y_pred is an open
    # interval.
    # interval_raw_prediction = Interval(-np.inf, np.inf, False, False)
    interval_y_pred = Interval(-np.inf, np.inf, False, False)

    @abstractmethod
    def link(self, y_pred, out=None):
        """Compute the link function g(y_pred).

        The link function maps (predicted) target values to raw predictions,
        i.e. `g(y_pred) = raw_prediction`.

        Parameters
        ----------
        y_pred : array
            Predicted target values.
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise link function.
        """

    @abstractmethod
    def inverse(self, raw_prediction, out=None):
        """Compute the inverse link function h(raw_prediction).

        The inverse link function maps raw predictions to predicted target
        values, i.e. `h(raw_prediction) = y_pred`.

        Parameters
        ----------
        raw_prediction : array
            Raw prediction values (in link space).
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise inverse link function.
        """


class IdentityLink(BaseLink):
    """The identity link function g(x)=x."""

    def link(self, y_pred, out=None):
        if out is not None:
            np.copyto(out, y_pred)
            return out
        else:
            return y_pred

    inverse = link


class LogLink(BaseLink):
    """The log link function g(x)=log(x)."""

    interval_y_pred = Interval(0, np.inf, False, False)

    def link(self, y_pred, out=None):
        return np.log(y_pred, out=out)

    def inverse(self, raw_prediction, out=None):
        return np.exp(raw_prediction, out=out)


class LogitLink(BaseLink):
    """The logit link function g(x)=logit(x)."""

    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred, out=None):
        return logit(y_pred, out=out)

    def inverse(self, raw_prediction, out=None):
        return expit(raw_prediction, out=out)


class HalfLogitLink(BaseLink):
    """Half the logit link function g(x)=1/2 * logit(x).

    Used for the exponential loss.
    """

    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred, out=None):
        out = logit(y_pred, out=out)
        out *= 0.5
        return out

    def inverse(self, raw_prediction, out=None):
        return expit(2 * raw_prediction, out)


class MultinomialLogit(BaseLink):
    """The symmetric multinomial logit function.

    Convention:
        - y_pred.shape = raw_prediction.shape = (n_samples, n_classes)

    Notes:
        - The inverse link h is the softmax function.
        - The sum is over the second axis, i.e. axis=1 (n_classes).

    We have to choose additional constraints in order to make

        y_pred[k] = exp(raw_pred[k]) / sum(exp(raw_pred[k]), k=0..n_classes-1)

    for n_classes classes identifiable and invertible.
    We choose the symmetric side constraint where the geometric mean response
    is set as reference category, see [2]:

    The symmetric multinomial logit link function for a single data point is
    then defined as

        raw_prediction[k] = g(y_pred[k]) = log(y_pred[k]/gmean(y_pred))
        = log(y_pred[k]) - mean(log(y_pred)).

    Note that this is equivalent to the definition in [1] and implies mean
    centered raw predictions:

        sum(raw_prediction[k], k=0..n_classes-1) = 0.

    For linear models with raw_prediction = X @ coef, this corresponds to
    sum(coef[k], k=0..n_classes-1) = 0, i.e. the sum over classes for every
    feature is zero.

    Reference
    ---------
    .. [1] Friedman, Jerome; Hastie, Trevor; Tibshirani, Robert. "Additive
        logistic regression: a statistical view of boosting" Ann. Statist.
        28 (2000), no. 2, 337--407. doi:10.1214/aos/1016218223.
        https://projecteuclid.org/euclid.aos/1016218223

    .. [2] Zahid, Faisal Maqbool and Gerhard Tutz. "Ridge estimation for
        multinomial logit models with symmetric side constraints."
        Computational Statistics 28 (2013): 1017-1034.
        http://epub.ub.uni-muenchen.de/11001/1/tr067.pdf
    """

    is_multiclass = True
    interval_y_pred = Interval(0, 1, False, False)

    def symmetrize_raw_prediction(self, raw_prediction):
        return raw_prediction - np.mean(raw_prediction, axis=1)[:, np.newaxis]

    def link(self, y_pred, out=None):
        # geometric mean as reference category
        gm = gmean(y_pred, axis=1)
        return np.log(y_pred / gm[:, np.newaxis], out=out)

    def inverse(self, raw_prediction, out=None):
        if out is None:
            return softmax(raw_prediction, copy=True)
        else:
            np.copyto(out, raw_prediction)
            softmax(out, copy=False)
            return out


_LINKS = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "half_logit": HalfLogitLink,
    "multinomial_logit": MultinomialLogit,
}
