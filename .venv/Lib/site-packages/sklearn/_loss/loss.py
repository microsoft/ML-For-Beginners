"""
This module contains loss classes suitable for fitting.

It is not part of the public API.
Specific losses are used for regression, binary classification or multiclass
classification.
"""
# Goals:
# - Provide a common private module for loss functions/classes.
# - To be used in:
#   - LogisticRegression
#   - PoissonRegressor, GammaRegressor, TweedieRegressor
#   - HistGradientBoostingRegressor, HistGradientBoostingClassifier
#   - GradientBoostingRegressor, GradientBoostingClassifier
#   - SGDRegressor, SGDClassifier
# - Replace link module of GLMs.

import numbers

import numpy as np
from scipy.special import xlogy

from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
    CyAbsoluteError,
    CyExponentialLoss,
    CyHalfBinomialLoss,
    CyHalfGammaLoss,
    CyHalfMultinomialLoss,
    CyHalfPoissonLoss,
    CyHalfSquaredError,
    CyHalfTweedieLoss,
    CyHalfTweedieLossIdentity,
    CyHuberLoss,
    CyPinballLoss,
)
from .link import (
    HalfLogitLink,
    IdentityLink,
    Interval,
    LogitLink,
    LogLink,
    MultinomialLogit,
)


# Note: The shape of raw_prediction for multiclass classifications are
# - GradientBoostingClassifier: (n_samples, n_classes)
# - HistGradientBoostingClassifier: (n_classes, n_samples)
#
# Note: Instead of inheritance like
#
#    class BaseLoss(BaseLink, CyLossFunction):
#    ...
#
#    # Note: Naturally, we would inherit in the following order
#    #     class HalfSquaredError(IdentityLink, CyHalfSquaredError, BaseLoss)
#    #   But because of https://github.com/cython/cython/issues/4350 we set BaseLoss as
#    #   the last one. This, of course, changes the MRO.
#    class HalfSquaredError(IdentityLink, CyHalfSquaredError, BaseLoss):
#
# we use composition. This way we improve maintainability by avoiding the above
# mentioned Cython edge case and have easier to understand code (which method calls
# which code).
class BaseLoss:
    """Base class for a loss function of 1-dimensional targets.

    Conventions:

        - y_true.shape = sample_weight.shape = (n_samples,)
        - y_pred.shape = raw_prediction.shape = (n_samples,)
        - If is_multiclass is true (multiclass classification), then
          y_pred.shape = raw_prediction.shape = (n_samples, n_classes)
          Note that this corresponds to the return value of decision_function.

    y_true, y_pred, sample_weight and raw_prediction must either be all float64
    or all float32.
    gradient and hessian must be either both float64 or both float32.

    Note that y_pred = link.inverse(raw_prediction).

    Specific loss classes can inherit specific link classes to satisfy
    BaseLink's abstractmethods.

    Parameters
    ----------
    sample_weight : {None, ndarray}
        If sample_weight is None, the hessian might be constant.
    n_classes : {None, int}
        The number of classes for classification, else None.

    Attributes
    ----------
    closs: CyLossFunction
    link : BaseLink
    interval_y_true : Interval
        Valid interval for y_true
    interval_y_pred : Interval
        Valid Interval for y_pred
    differentiable : bool
        Indicates whether or not loss function is differentiable in
        raw_prediction everywhere.
    need_update_leaves_values : bool
        Indicates whether decision trees in gradient boosting need to uptade
        leave values after having been fit to the (negative) gradients.
    approx_hessian : bool
        Indicates whether the hessian is approximated or exact. If,
        approximated, it should be larger or equal to the exact one.
    constant_hessian : bool
        Indicates whether the hessian is one for this loss.
    is_multiclass : bool
        Indicates whether n_classes > 2 is allowed.
    """

    # For decision trees:
    # This variable indicates whether the loss requires the leaves values to
    # be updated once the tree has been trained. The trees are trained to
    # predict a Newton-Raphson step (see grower._finalize_leaf()). But for
    # some losses (e.g. least absolute deviation) we need to adjust the tree
    # values to account for the "line search" of the gradient descent
    # procedure. See the original paper Greedy Function Approximation: A
    # Gradient Boosting Machine by Friedman
    # (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) for the theory.
    need_update_leaves_values = False
    differentiable = True
    is_multiclass = False

    def __init__(self, closs, link, n_classes=None):
        self.closs = closs
        self.link = link
        self.approx_hessian = False
        self.constant_hessian = False
        self.n_classes = n_classes
        self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        self.interval_y_pred = self.link.interval_y_pred

    def in_y_true_range(self, y):
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_true.includes(y)

    def in_y_pred_range(self, y):
        """Return True if y is in the valid range of y_pred.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_pred.includes(y)

    def loss(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        n_threads=1,
    ):
        """Compute the pointwise loss value for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        loss_out : None or C-contiguous array of shape (n_samples,)
            A location into which the result is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.
        """
        if loss_out is None:
            loss_out = np.empty_like(y_true)
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)

        return self.closs.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            n_threads=n_threads,
        )

    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        gradient_out=None,
        n_threads=1,
    ):
        """Compute loss and gradient w.r.t. raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        loss_out : None or C-contiguous array of shape (n_samples,)
            A location into which the loss is stored. If None, a new array
            might be created.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.

        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        if loss_out is None:
            if gradient_out is None:
                loss_out = np.empty_like(y_true)
                gradient_out = np.empty_like(raw_prediction)
            else:
                loss_out = np.empty_like(y_true, dtype=gradient_out.dtype)
        elif gradient_out is None:
            gradient_out = np.empty_like(raw_prediction, dtype=loss_out.dtype)

        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)

        return self.closs.loss_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )

    def gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        n_threads=1,
    ):
        """Compute gradient of loss w.r.t raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the result is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        if gradient_out is None:
            gradient_out = np.empty_like(raw_prediction)

        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)

        return self.closs.gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )

    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        hessian_out=None,
        n_threads=1,
    ):
        """Compute gradient and hessian of loss w.r.t raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        hessian_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the hessian is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : arrays of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.

        hessian : arrays of shape (n_samples,) or (n_samples, n_classes)
            Element-wise hessians.
        """
        if gradient_out is None:
            if hessian_out is None:
                gradient_out = np.empty_like(raw_prediction)
                hessian_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(hessian_out)
        elif hessian_out is None:
            hessian_out = np.empty_like(gradient_out)

        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)
        if hessian_out.ndim == 2 and hessian_out.shape[1] == 1:
            hessian_out = hessian_out.squeeze(1)

        return self.closs.gradient_hessian(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            hessian_out=hessian_out,
            n_threads=n_threads,
        )

    def __call__(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        """Compute the weighted average loss.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : float
            Mean or averaged loss function.
        """
        return np.average(
            self.loss(
                y_true=y_true,
                raw_prediction=raw_prediction,
                sample_weight=None,
                loss_out=None,
                n_threads=n_threads,
            ),
            weights=sample_weight,
        )

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This can be used as initial estimates of predictions, i.e. before the
        first iteration in fit.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or array of shape (n_samples,)
            Sample weights.

        Returns
        -------
        raw_prediction : numpy scalar or array of shape (n_classes,)
            Raw predictions of an intercept-only model.
        """
        # As default, take weighted average of the target over the samples
        # axis=0 and then transform into link-scale (raw_prediction).
        y_pred = np.average(y_true, weights=sample_weight, axis=0)
        eps = 10 * np.finfo(y_pred.dtype).eps

        if self.interval_y_pred.low == -np.inf:
            a_min = None
        elif self.interval_y_pred.low_inclusive:
            a_min = self.interval_y_pred.low
        else:
            a_min = self.interval_y_pred.low + eps

        if self.interval_y_pred.high == np.inf:
            a_max = None
        elif self.interval_y_pred.high_inclusive:
            a_max = self.interval_y_pred.high
        else:
            a_max = self.interval_y_pred.high - eps

        if a_min is None and a_max is None:
            return self.link.link(y_pred)
        else:
            return self.link.link(np.clip(y_pred, a_min, a_max))

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        """Calculate term dropped in loss.

        With this term added, the loss of perfect predictions is zero.
        """
        return np.zeros_like(y_true)

    def init_gradient_and_hessian(self, n_samples, dtype=np.float64, order="F"):
        """Initialize arrays for gradients and hessians.

        Unless hessians are constant, arrays are initialized with undefined values.

        Parameters
        ----------
        n_samples : int
            The number of samples, usually passed to `fit()`.
        dtype : {np.float64, np.float32}, default=np.float64
            The dtype of the arrays gradient and hessian.
        order : {'C', 'F'}, default='F'
            Order of the arrays gradient and hessian. The default 'F' makes the arrays
            contiguous along samples.

        Returns
        -------
        gradient : C-contiguous array of shape (n_samples,) or array of shape \
            (n_samples, n_classes)
            Empty array (allocated but not initialized) to be used as argument
            gradient_out.
        hessian : C-contiguous array of shape (n_samples,), array of shape
            (n_samples, n_classes) or shape (1,)
            Empty (allocated but not initialized) array to be used as argument
            hessian_out.
            If constant_hessian is True (e.g. `HalfSquaredError`), the array is
            initialized to ``1``.
        """
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                "Valid options for 'dtype' are np.float32 and np.float64. "
                f"Got dtype={dtype} instead."
            )

        if self.is_multiclass:
            shape = (n_samples, self.n_classes)
        else:
            shape = (n_samples,)
        gradient = np.empty(shape=shape, dtype=dtype, order=order)

        if self.constant_hessian:
            # If the hessians are constant, we consider them equal to 1.
            # - This is correct for HalfSquaredError
            # - For AbsoluteError, hessians are actually 0, but they are
            #   always ignored anyway.
            hessian = np.ones(shape=(1,), dtype=dtype)
        else:
            hessian = np.empty(shape=shape, dtype=dtype, order=order)

        return gradient, hessian


# Note: Naturally, we would inherit in the following order
#         class HalfSquaredError(IdentityLink, CyHalfSquaredError, BaseLoss)
#       But because of https://github.com/cython/cython/issues/4350 we
#       set BaseLoss as the last one. This, of course, changes the MRO.
class HalfSquaredError(BaseLoss):
    """Half squared error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half squared error is defined as::

        loss(x_i) = 0.5 * (y_true_i - raw_prediction_i)**2

    The factor of 0.5 simplifies the computation of gradients and results in a
    unit hessian (and is consistent with what is done in LightGBM). It is also
    half the Normal distribution deviance.
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfSquaredError(), link=IdentityLink())
        self.constant_hessian = sample_weight is None


class AbsoluteError(BaseLoss):
    """Absolute error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the absolute error is defined as::

        loss(x_i) = |y_true_i - raw_prediction_i|
    """

    differentiable = False
    need_update_leaves_values = True

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyAbsoluteError(), link=IdentityLink())
        self.approx_hessian = True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            return np.median(y_true, axis=0)
        else:
            return _weighted_percentile(y_true, sample_weight, 50)


class PinballLoss(BaseLoss):
    """Quantile loss aka pinball loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the pinball loss is defined as::

        loss(x_i) = rho_{quantile}(y_true_i - raw_prediction_i)

        rho_{quantile}(u) = u * (quantile - 1_{u<0})
                          = -u *(1 - quantile)  if u < 0
                             u * quantile       if u >= 0

    Note: 2 * PinballLoss(quantile=0.5) equals AbsoluteError().

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level of the quantile to be estimated. Must be in range (0, 1).
    """

    differentiable = False
    need_update_leaves_values = True

    def __init__(self, sample_weight=None, quantile=0.5):
        check_scalar(
            quantile,
            "quantile",
            target_type=numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )
        super().__init__(
            closs=CyPinballLoss(quantile=float(quantile)),
            link=IdentityLink(),
        )
        self.approx_hessian = True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            return np.percentile(y_true, 100 * self.closs.quantile, axis=0)
        else:
            return _weighted_percentile(
                y_true, sample_weight, 100 * self.closs.quantile
            )


class HuberLoss(BaseLoss):
    """Huber loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the Huber loss is defined as::

        loss(x_i) = 1/2 * abserr**2            if abserr <= delta
                    delta * (abserr - delta/2) if abserr > delta

        abserr = |y_true_i - raw_prediction_i|
        delta = quantile(abserr, self.quantile)

    Note: HuberLoss(quantile=1) equals HalfSquaredError and HuberLoss(quantile=0)
    equals delta * (AbsoluteError() - delta/2).

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level which defines the breaking point `delta` to distinguish
        between absolute error and squared error. Must be in range (0, 1).

     Reference
    ---------
    .. [1] Friedman, J.H. (2001). :doi:`Greedy function approximation: A gradient
      boosting machine <10.1214/aos/1013203451>`.
      Annals of Statistics, 29, 1189-1232.
    """

    differentiable = False
    need_update_leaves_values = True

    def __init__(self, sample_weight=None, quantile=0.9, delta=0.5):
        check_scalar(
            quantile,
            "quantile",
            target_type=numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )
        self.quantile = quantile  # This is better stored outside of Cython.
        super().__init__(
            closs=CyHuberLoss(delta=float(delta)),
            link=IdentityLink(),
        )
        self.approx_hessian = True
        self.constant_hessian = False

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        # See formula before algo 4 in Friedman (2001), but we apply it to y_true,
        # not to the residual y_true - raw_prediction. An estimator like
        # HistGradientBoostingRegressor might then call it on the residual, e.g.
        # fit_intercept_only(y_true - raw_prediction).
        if sample_weight is None:
            median = np.percentile(y_true, 50, axis=0)
        else:
            median = _weighted_percentile(y_true, sample_weight, 50)
        diff = y_true - median
        term = np.sign(diff) * np.minimum(self.closs.delta, np.abs(diff))
        return median + np.average(term, weights=sample_weight)


class HalfPoissonLoss(BaseLoss):
    """Half Poisson deviance loss with log-link, for regression.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half the Poisson deviance is defined as::

        loss(x_i) = y_true_i * log(y_true_i/exp(raw_prediction_i))
                    - y_true_i + exp(raw_prediction_i)

    Half the Poisson deviance is actually the negative log-likelihood up to
    constant terms (not involving raw_prediction) and simplifies the
    computation of the gradients.
    We also skip the constant term `y_true_i * log(y_true_i) - y_true_i`.
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfPoissonLoss(), link=LogLink())
        self.interval_y_true = Interval(0, np.inf, True, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        term = xlogy(y_true, y_true) - y_true
        if sample_weight is not None:
            term *= sample_weight
        return term


class HalfGammaLoss(BaseLoss):
    """Half Gamma deviance loss with log-link, for regression.

    Domain:
    y_true and y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half Gamma deviance loss is defined as::

        loss(x_i) = log(exp(raw_prediction_i)/y_true_i)
                    + y_true/exp(raw_prediction_i) - 1

    Half the Gamma deviance is actually proportional to the negative log-
    likelihood up to constant terms (not involving raw_prediction) and
    simplifies the computation of the gradients.
    We also skip the constant term `-log(y_true_i) - 1`.
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfGammaLoss(), link=LogLink())
        self.interval_y_true = Interval(0, np.inf, False, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        term = -np.log(y_true) - 1
        if sample_weight is not None:
            term *= sample_weight
        return term


class HalfTweedieLoss(BaseLoss):
    """Half Tweedie deviance loss with log-link, for regression.

    Domain:
    y_true in real numbers for power <= 0
    y_true in non-negative real numbers for 0 < power < 2
    y_true in positive real numbers for 2 <= power
    y_pred in positive real numbers
    power in real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half Tweedie deviance loss with p=power is defined
    as::

        loss(x_i) = max(y_true_i, 0)**(2-p) / (1-p) / (2-p)
                    - y_true_i * exp(raw_prediction_i)**(1-p) / (1-p)
                    + exp(raw_prediction_i)**(2-p) / (2-p)

    Taking the limits for p=0, 1, 2 gives HalfSquaredError with a log link,
    HalfPoissonLoss and HalfGammaLoss.

    We also skip constant terms, but those are different for p=0, 1, 2.
    Therefore, the loss is not continuous in `power`.

    Note furthermore that although no Tweedie distribution exists for
    0 < power < 1, it still gives a strictly consistent scoring function for
    the expectation.
    """

    def __init__(self, sample_weight=None, power=1.5):
        super().__init__(
            closs=CyHalfTweedieLoss(power=float(power)),
            link=LogLink(),
        )
        if self.closs.power <= 0:
            self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        elif self.closs.power < 2:
            self.interval_y_true = Interval(0, np.inf, True, False)
        else:
            self.interval_y_true = Interval(0, np.inf, False, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        if self.closs.power == 0:
            return HalfSquaredError().constant_to_optimal_zero(
                y_true=y_true, sample_weight=sample_weight
            )
        elif self.closs.power == 1:
            return HalfPoissonLoss().constant_to_optimal_zero(
                y_true=y_true, sample_weight=sample_weight
            )
        elif self.closs.power == 2:
            return HalfGammaLoss().constant_to_optimal_zero(
                y_true=y_true, sample_weight=sample_weight
            )
        else:
            p = self.closs.power
            term = np.power(np.maximum(y_true, 0), 2 - p) / (1 - p) / (2 - p)
            if sample_weight is not None:
                term *= sample_weight
            return term


class HalfTweedieLossIdentity(BaseLoss):
    """Half Tweedie deviance loss with identity link, for regression.

    Domain:
    y_true in real numbers for power <= 0
    y_true in non-negative real numbers for 0 < power < 2
    y_true in positive real numbers for 2 <= power
    y_pred in positive real numbers for power != 0
    y_pred in real numbers for power = 0
    power in real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half Tweedie deviance loss with p=power is defined
    as::

        loss(x_i) = max(y_true_i, 0)**(2-p) / (1-p) / (2-p)
                    - y_true_i * raw_prediction_i**(1-p) / (1-p)
                    + raw_prediction_i**(2-p) / (2-p)

    Note that the minimum value of this loss is 0.

    Note furthermore that although no Tweedie distribution exists for
    0 < power < 1, it still gives a strictly consistent scoring function for
    the expectation.
    """

    def __init__(self, sample_weight=None, power=1.5):
        super().__init__(
            closs=CyHalfTweedieLossIdentity(power=float(power)),
            link=IdentityLink(),
        )
        if self.closs.power <= 0:
            self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        elif self.closs.power < 2:
            self.interval_y_true = Interval(0, np.inf, True, False)
        else:
            self.interval_y_true = Interval(0, np.inf, False, False)

        if self.closs.power == 0:
            self.interval_y_pred = Interval(-np.inf, np.inf, False, False)
        else:
            self.interval_y_pred = Interval(0, np.inf, False, False)


class HalfBinomialLoss(BaseLoss):
    """Half Binomial deviance loss with logit link, for binary classification.

    This is also know as binary cross entropy, log-loss and logistic loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)

    For a given sample x_i, half Binomial deviance is defined as the negative
    log-likelihood of the Binomial/Bernoulli distribution and can be expressed
    as::

        loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i

    See The Elements of Statistical Learning, by Hastie, Tibshirani, Friedman,
    section 4.4.1 (about logistic regression).

    Note that the formulation works for classification, y = {0, 1}, as well as
    logistic regression, y = [0, 1].
    If you add `constant_to_optimal_zero` to the loss, you get half the
    Bernoulli/binomial deviance.

    More details: Inserting the predicted probability y_pred = expit(raw_prediction)
    in the loss gives the well known::

        loss(x_i) = - y_true_i * log(y_pred_i) - (1 - y_true_i) * log(1 - y_pred_i)
    """

    def __init__(self, sample_weight=None):
        super().__init__(
            closs=CyHalfBinomialLoss(),
            link=LogitLink(),
            n_classes=2,
        )
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # This is non-zero only if y_true is neither 0 nor 1.
        term = xlogy(y_true, y_true) + xlogy(1 - y_true, 1 - y_true)
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class HalfMultinomialLoss(BaseLoss):
    """Categorical cross-entropy loss, for multiclass classification.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred has n_classes elements, each element in (0, 1)

    Link:
    y_pred = softmax(raw_prediction)

    Note: We assume y_true to be already label encoded. The inverse link is
    softmax. But the full link function is the symmetric multinomial logit
    function.

    For a given sample x_i, the categorical cross-entropy loss is defined as
    the negative log-likelihood of the multinomial distribution, it
    generalizes the binary cross-entropy to more than 2 classes::

        loss_i = log(sum(exp(raw_pred_{i, k}), k=0..n_classes-1))
                - sum(y_true_{i, k} * raw_pred_{i, k}, k=0..n_classes-1)

    See [1].

    Note that for the hessian, we calculate only the diagonal part in the
    classes: If the full hessian for classes k and l and sample i is H_i_k_l,
    we calculate H_i_k_k, i.e. k=l.

    Reference
    ---------
    .. [1] :arxiv:`Simon, Noah, J. Friedman and T. Hastie.
        "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
        Multinomial Regression".
        <1311.6529>`
    """

    is_multiclass = True

    def __init__(self, sample_weight=None, n_classes=3):
        super().__init__(
            closs=CyHalfMultinomialLoss(),
            link=MultinomialLogit(),
            n_classes=n_classes,
        )
        self.interval_y_true = Interval(0, np.inf, True, False)
        self.interval_y_pred = Interval(0, 1, False, False)

    def in_y_true_range(self, y):
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_true.includes(y) and np.all(y.astype(int) == y)

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the softmax of the weighted average of the target, i.e. over
        the samples axis=0.
        """
        out = np.zeros(self.n_classes, dtype=y_true.dtype)
        eps = np.finfo(y_true.dtype).eps
        for k in range(self.n_classes):
            out[k] = np.average(y_true == k, weights=sample_weight, axis=0)
            out[k] = np.clip(out[k], eps, 1 - eps)
        return self.link.link(out[None, :]).reshape(-1)

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        return self.link.inverse(raw_prediction)

    def gradient_proba(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        proba_out=None,
        n_threads=1,
    ):
        """Compute gradient and class probabilities fow raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or array of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        proba_out : None or array of shape (n_samples, n_classes)
            A location into which the class probabilities are stored. If None,
            a new array might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples, n_classes)
            Element-wise gradients.

        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        if gradient_out is None:
            if proba_out is None:
                gradient_out = np.empty_like(raw_prediction)
                proba_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(proba_out)
        elif proba_out is None:
            proba_out = np.empty_like(gradient_out)

        return self.closs.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            proba_out=proba_out,
            n_threads=n_threads,
        )


class ExponentialLoss(BaseLoss):
    """Exponential loss with (half) logit link, for binary classification.

    This is also know as boosting loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(2 * raw_prediction)

    For a given sample x_i, the exponential loss is defined as::

        loss(x_i) = y_true_i * exp(-raw_pred_i)) + (1 - y_true_i) * exp(raw_pred_i)

    See:
    - J. Friedman, T. Hastie, R. Tibshirani.
      "Additive logistic regression: a statistical view of boosting (With discussion
      and a rejoinder by the authors)." Ann. Statist. 28 (2) 337 - 407, April 2000.
      https://doi.org/10.1214/aos/1016218223
    - A. Buja, W. Stuetzle, Y. Shen. (2005).
      "Loss Functions for Binary Class Probability Estimation and Classification:
      Structure and Applications."

    Note that the formulation works for classification, y = {0, 1}, as well as
    "exponential logistic" regression, y = [0, 1].
    Note that this is a proper scoring rule, but without it's canonical link.

    More details: Inserting the predicted probability
    y_pred = expit(2 * raw_prediction) in the loss gives::

        loss(x_i) = y_true_i * sqrt((1 - y_pred_i) / y_pred_i)
            + (1 - y_true_i) * sqrt(y_pred_i / (1 - y_pred_i))
    """

    def __init__(self, sample_weight=None):
        super().__init__(
            closs=CyExponentialLoss(),
            link=HalfLogitLink(),
            n_classes=2,
        )
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # This is non-zero only if y_true is neither 0 nor 1.
        term = -2 * np.sqrt(y_true * (1 - y_true))
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


_LOSSES = {
    "squared_error": HalfSquaredError,
    "absolute_error": AbsoluteError,
    "pinball_loss": PinballLoss,
    "huber_loss": HuberLoss,
    "poisson_loss": HalfPoissonLoss,
    "gamma_loss": HalfGammaLoss,
    "tweedie_loss": HalfTweedieLoss,
    "binomial_loss": HalfBinomialLoss,
    "multinomial_loss": HalfMultinomialLoss,
    "exponential_loss": ExponentialLoss,
}
