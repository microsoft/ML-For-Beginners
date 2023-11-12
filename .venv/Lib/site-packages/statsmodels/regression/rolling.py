"""
Rolling OLS and WLS

Implements an efficient rolling estimator that avoids repeated matrix
multiplication.

Copyright (c) 2019 Kevin Sheppard
License: 3-clause BSD
"""
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import (
    Appender,
    Substitution,
    cache_readonly,
    call_cached_func,
    get_cached_doc,
)

from collections import namedtuple

import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats

from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import (
    RegressionModel,
    RegressionResults,
)
from statsmodels.tools.validation import array_like, int_like, string_like


def strip4(line):
    if line.startswith(" "):
        return line[4:]
    return line


RollingStore = namedtuple(
    "RollingStore",
    [
        "params",
        "ssr",
        "llf",
        "nobs",
        "s2",
        "xpxi",
        "xeex",
        "centered_tss",
        "uncentered_tss",
    ],
)

common_params = "\n".join(map(strip4, model._model_params_doc.split("\n")))
window_parameters = """\
window : int
    Length of the rolling window. Must be strictly larger than the number
    of variables in the model.
"""

weight_parameters = """
weights : array_like, optional
    A 1d array of weights.  If you supply 1/W then the variables are
    pre- multiplied by 1/sqrt(W).  If no weights are supplied the
    default value is 1 and WLS results are the same as OLS.
"""

_missing_param_doc = """\
min_nobs : {int, None}
    Minimum number of observations required to estimate a model when
    data are missing.  If None, the minimum depends on the number of
    regressors in the model. Must be smaller than window.
missing : str, default "drop"
    Available options are "drop", "skip" and "raise". If "drop", any
    observations with nans are dropped and the estimates are computed using
    only the non-missing values in each window. If 'skip' blocks containing
    missing values are skipped and the corresponding results contains NaN.
    If 'raise', an error is raised. Default is 'drop'.
expanding : bool, default False
    If True, then the initial observations after min_nobs are filled using
    an expanding scheme until ``window`` observations are available, after
    which rolling is used.
"""


extra_base = _missing_param_doc
extra_parameters = window_parameters + weight_parameters + extra_base

_doc = """
Rolling %(model_type)s Least Squares

%(parameters)s
%(extra_parameters)s

See Also
--------
statsmodels.regression.linear_model.%(model)s
    %(model)s estimation and parameter testing.

Notes
-----
Tested against %(model)s for accuracy.

Results may differ from %(model)s applied to windows of data if this
model contains an implicit constant (i.e., includes dummies for all
categories) rather than an explicit constant (e.g., a column of 1s).

Examples
--------
>>> from statsmodels.regression.rolling import Rolling%(model)s
>>> from statsmodels.datasets import longley
>>> data = longley.load()
>>> exog = add_constant(data.exog, prepend=False)
>>> mod = Rolling%(model)s(data.endog, exog)
>>> rolling_res = mod.fit(reset=50)

Use params_only to skip all calculations except parameter estimation

>>> rolling_params = mod.fit(params_only=True)

Use expanding and min_nobs to fill the initial results using an
expanding scheme until window observation, and the roll.

>>> mod = Rolling%(model)s(data.endog, exog, window=60, min_nobs=12,
... expanding=True)
>>> rolling_res = mod.fit()
"""


@Substitution(
    model_type="Weighted",
    model="WLS",
    parameters=common_params,
    extra_parameters=extra_parameters,
)
@Appender(_doc)
class RollingWLS:
    def __init__(
        self,
        endog,
        exog,
        window=None,
        *,
        weights=None,
        min_nobs=None,
        missing="drop",
        expanding=False
    ):
        # Call Model.__init__ twice to use const detection in first pass
        # But to not drop in the second pass
        missing = string_like(
            missing, "missing", options=("drop", "raise", "skip")
        )
        temp_msng = "drop" if missing != "raise" else "raise"
        Model.__init__(self, endog, exog, missing=temp_msng, hasconst=None)
        k_const = self.k_constant
        const_idx = self.data.const_idx
        Model.__init__(self, endog, exog, missing="none", hasconst=False)
        self.k_constant = k_const
        self.data.const_idx = const_idx
        self._y = array_like(endog, "endog")
        nobs = self._y.shape[0]
        self._x = array_like(exog, "endog", ndim=2, shape=(nobs, None))
        window = int_like(window, "window", optional=True)
        weights = array_like(weights, "weights", optional=True, shape=(nobs,))
        self._window = window if window is not None else self._y.shape[0]
        self._weighted = weights is not None
        self._weights = np.ones(nobs) if weights is None else weights
        w12 = np.sqrt(self._weights)
        self._wy = w12 * self._y
        self._wx = w12[:, None] * self._x

        min_nobs = int_like(min_nobs, "min_nobs", optional=True)
        self._min_nobs = min_nobs if min_nobs is not None else self._x.shape[1]
        if self._min_nobs < self._x.shape[1] or self._min_nobs > self._window:
            raise ValueError(
                "min_nobs must be larger than the number of "
                "regressors in the model and less than window"
            )

        self._expanding = expanding

        self._is_nan = np.zeros_like(self._y, dtype=bool)
        self._has_nan = self._find_nans()
        self.const_idx = self.data.const_idx
        self._skip_missing = missing == "skip"

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        return Model._handle_data(
            self, endog, exog, missing, hasconst, **kwargs
        )

    def _find_nans(self):
        nans = np.isnan(self._y)
        nans |= np.any(np.isnan(self._x), axis=1)
        nans |= np.isnan(self._weights)
        self._is_nan[:] = nans
        has_nan = np.cumsum(nans)
        w = self._window
        has_nan[w - 1 :] = has_nan[w - 1 :] - has_nan[: -(w - 1)]
        if self._expanding:
            has_nan[: self._min_nobs] = False
        else:
            has_nan[: w - 1] = False

        return has_nan.astype(bool)

    def _get_data(self, idx):
        window = self._window
        if idx >= window:
            loc = slice(idx - window, idx)
        else:
            loc = slice(idx)
        y = self._y[loc]
        wy = self._wy[loc]
        wx = self._wx[loc]
        weights = self._weights[loc]
        missing = self._is_nan[loc]
        not_missing = ~missing
        if np.any(missing):
            y = y[not_missing]
            wy = wy[not_missing]
            wx = wx[not_missing]
            weights = weights[not_missing]
        return y, wy, wx, weights, not_missing

    def _fit_single(self, idx, wxpwx, wxpwy, nobs, store, params_only, method):
        if nobs < self._min_nobs:
            return
        try:
            if (method == "inv") or not params_only:
                wxpwxi = np.linalg.inv(wxpwx)
            if method == "inv":
                params = wxpwxi @ wxpwy
            else:
                _, wy, wx, _, _ = self._get_data(idx)
                if method == "lstsq":
                    params = lstsq(wx, wy)[0]
                else:  # 'pinv'
                    wxpwxiwxp = np.linalg.pinv(wx)
                    params = wxpwxiwxp @ wy

        except np.linalg.LinAlgError:
            return
        store.params[idx - 1] = params
        if params_only:
            return
        y, wy, wx, weights, _ = self._get_data(idx)

        wresid, ssr, llf = self._loglike(params, wy, wx, weights, nobs)
        wxwresid = wx * wresid[:, None]
        wxepwxe = wxwresid.T @ wxwresid
        tot_params = wx.shape[1]
        s2 = ssr / (nobs - tot_params)

        centered_tss, uncentered_tss = self._sum_of_squares(y, wy, weights)

        store.ssr[idx - 1] = ssr
        store.llf[idx - 1] = llf
        store.nobs[idx - 1] = nobs
        store.s2[idx - 1] = s2
        store.xpxi[idx - 1] = wxpwxi
        store.xeex[idx - 1] = wxepwxe
        store.centered_tss[idx - 1] = centered_tss
        store.uncentered_tss[idx - 1] = uncentered_tss

    def _loglike(self, params, wy, wx, weights, nobs):
        nobs2 = nobs / 2.0
        wresid = wy - wx @ params
        ssr = np.sum(wresid ** 2, axis=0)
        llf = -np.log(ssr) * nobs2  # concentrated likelihood
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2  # with constant
        llf += 0.5 * np.sum(np.log(weights))
        return wresid, ssr, llf

    def _sum_of_squares(self, y, wy, weights):
        mean = np.average(y, weights=weights)
        centered_tss = np.sum(weights * (y - mean) ** 2)
        uncentered_tss = np.dot(wy, wy)
        return centered_tss, uncentered_tss

    def _reset(self, idx):
        """Compute xpx and xpy using a single dot product"""
        _, wy, wx, _, not_missing = self._get_data(idx)
        nobs = not_missing.sum()
        xpx = wx.T @ wx
        xpy = wx.T @ wy
        return xpx, xpy, nobs

    def fit(
        self,
        method="inv",
        cov_type="nonrobust",
        cov_kwds=None,
        reset=None,
        use_t=False,
        params_only=False,
    ):
        """
        Estimate model parameters.

        Parameters
        ----------
        method : {'inv', 'lstsq', 'pinv'}
            Method to use when computing the the model parameters.

            * 'inv' - use moving windows inner-products and matrix inversion.
              This method is the fastest, but may be less accurate than the
              other methods.
            * 'lstsq' - Use numpy.linalg.lstsq
            * 'pinv' - Use numpy.linalg.pinv. This method matches the default
              estimator in non-moving regression estimators.
        cov_type : {'nonrobust', 'HCCM', 'HC0'}
            Covariance estimator:

            * nonrobust - The classic OLS covariance estimator
            * HCCM, HC0 - White heteroskedasticity robust covariance
        cov_kwds : dict
            Unused
        reset : int, optional
            Interval to recompute the moving window inner products used to
            estimate the model parameters. Smaller values improve accuracy,
            although in practice this setting is not required to be set.
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.
        params_only : bool, optional
            Flag indicating that only parameters should be computed. Avoids
            calculating all other statistics or performing inference.

        Returns
        -------
        RollingRegressionResults
            Estimation results where all pre-sample values are nan-filled.
        """
        method = string_like(
            method, "method", options=("inv", "lstsq", "pinv")
        )
        reset = int_like(reset, "reset", optional=True)
        reset = self._y.shape[0] if reset is None else reset
        if reset < 1:
            raise ValueError("reset must be a positive integer")

        nobs, k = self._x.shape
        store = RollingStore(
            params=np.full((nobs, k), np.nan),
            ssr=np.full(nobs, np.nan),
            llf=np.full(nobs, np.nan),
            nobs=np.zeros(nobs, dtype=int),
            s2=np.full(nobs, np.nan),
            xpxi=np.full((nobs, k, k), np.nan),
            xeex=np.full((nobs, k, k), np.nan),
            centered_tss=np.full(nobs, np.nan),
            uncentered_tss=np.full(nobs, np.nan),
        )
        w = self._window
        first = self._min_nobs if self._expanding else w
        xpx, xpy, nobs = self._reset(first)
        if not (self._has_nan[first - 1] and self._skip_missing):
            self._fit_single(first, xpx, xpy, nobs, store, params_only, method)
        wx, wy = self._wx, self._wy
        for i in range(first + 1, self._x.shape[0] + 1):
            if self._has_nan[i - 1] and self._skip_missing:
                continue
            if i % reset == 0:
                xpx, xpy, nobs = self._reset(i)
            else:
                if not self._is_nan[i - w - 1] and i > w:
                    remove_x = wx[i - w - 1 : i - w]
                    xpx -= remove_x.T @ remove_x
                    xpy -= remove_x.T @ wy[i - w - 1 : i - w]
                    nobs -= 1
                if not self._is_nan[i - 1]:
                    add_x = wx[i - 1 : i]
                    xpx += add_x.T @ add_x
                    xpy += add_x.T @ wy[i - 1 : i]
                    nobs += 1

            self._fit_single(i, xpx, xpy, nobs, store, params_only, method)

        return RollingRegressionResults(
            self, store, self.k_constant, use_t, cov_type
        )

    @classmethod
    @Appender(Model.from_formula.__doc__)
    def from_formula(
        cls, formula, data, window, weights=None, subset=None, *args, **kwargs
    ):
        if subset is not None:
            data = data.loc[subset]
        eval_env = kwargs.pop("eval_env", None)
        if eval_env is None:
            eval_env = 2
        elif eval_env == -1:
            from patsy import EvalEnvironment

            eval_env = EvalEnvironment({})
        else:
            eval_env += 1  # we're going down the stack again
        missing = kwargs.get("missing", "skip")
        from patsy import NAAction, dmatrices

        na_action = NAAction(on_NA="raise", NA_types=[])
        result = dmatrices(
            formula,
            data,
            eval_env,
            return_type="dataframe",
            NA_action=na_action,
        )

        endog, exog = result
        if (endog.ndim > 1 and endog.shape[1] > 1) or endog.ndim > 2:
            raise ValueError(
                "endog has evaluated to an array with multiple "
                "columns that has shape {0}. This occurs when "
                "the variable converted to endog is non-numeric"
                " (e.g., bool or str).".format(endog.shape)
            )

        kwargs.update({"missing": missing, "window": window})
        if weights is not None:
            kwargs["weights"] = weights
        mod = cls(endog, exog, *args, **kwargs)
        mod.formula = formula
        # since we got a dataframe, attach the original
        mod.data.frame = data
        return mod


extra_parameters = window_parameters + extra_base


@Substitution(
    model_type="Ordinary",
    model="OLS",
    parameters=common_params,
    extra_parameters=extra_parameters,
)
@Appender(_doc)
class RollingOLS(RollingWLS):
    def __init__(
        self,
        endog,
        exog,
        window=None,
        *,
        min_nobs=None,
        missing="drop",
        expanding=False
    ):
        super().__init__(
            endog,
            exog,
            window,
            weights=None,
            min_nobs=min_nobs,
            missing=missing,
            expanding=expanding,
        )


class RollingRegressionResults:
    """
    Results from rolling regressions

    Parameters
    ----------
    model : RollingWLS
        Model instance
    store : RollingStore
        Container for raw moving window results
    k_constant : bool
        Flag indicating that the model contains a constant
    use_t : bool
        Flag indicating to use the Student's t distribution when computing
        p-values.
    cov_type : str
        Name of covariance estimator
    """

    _data_in_cache = tuple()

    def __init__(
        self, model, store: RollingStore, k_constant, use_t, cov_type
    ):
        self.model = model
        self._params = store.params
        self._ssr = store.ssr
        self._llf = store.llf
        self._nobs = store.nobs
        self._s2 = store.s2
        self._xpxi = store.xpxi
        self._xepxe = store.xeex
        self._centered_tss = store.centered_tss
        self._uncentered_tss = store.uncentered_tss
        self._k_constant = k_constant
        self._nvar = self._xpxi.shape[-1]
        if use_t is None:
            use_t = cov_type == "nonrobust"
        self._use_t = use_t
        self._cov_type = cov_type
        self._use_pandas = self.model.data.row_labels is not None
        self._data_attr = []
        self._cache = {}

    def _wrap(self, val):
        """Wrap output as pandas Series or DataFrames as needed"""
        if not self._use_pandas:
            return val
        col_names = self.model.data.param_names
        row_names = self.model.data.row_labels
        if val.ndim == 1:
            return Series(val, index=row_names)
        if val.ndim == 2:
            return DataFrame(val, columns=col_names, index=row_names)
        else:  # ndim == 3
            mi = MultiIndex.from_product((row_names, col_names))
            val = np.reshape(val, (-1, val.shape[-1]))
            return DataFrame(val, columns=col_names, index=mi)

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.aic))
    def aic(self):
        return self._wrap(call_cached_func(RegressionResults.aic, self))

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.bic))
    def bic(self):
        with np.errstate(divide="ignore"):
            return self._wrap(call_cached_func(RegressionResults.bic, self))

    def info_criteria(self, crit, dk_params=0):
        return self._wrap(
            RegressionResults.info_criteria(self, crit, dk_params=dk_params)
        )

    @cache_readonly
    def params(self):
        """Estimated model parameters"""
        return self._wrap(self._params)

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.ssr))
    def ssr(self):
        return self._wrap(self._ssr)

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.llf))
    def llf(self):
        return self._wrap(self._llf)

    @cache_readonly
    @Appender(RegressionModel.df_model.__doc__)
    def df_model(self):
        return self._nvar - self._k_constant

    @cache_readonly
    def k_constant(self):
        """Flag indicating whether the model contains a constant"""
        return self._k_constant

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.centered_tss))
    def centered_tss(self):
        return self._centered_tss

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.uncentered_tss))
    def uncentered_tss(self):
        return self._uncentered_tss

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.rsquared))
    def rsquared(self):
        return self._wrap(call_cached_func(RegressionResults.rsquared, self))

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.rsquared_adj))
    def rsquared_adj(self):
        return self._wrap(
            call_cached_func(RegressionResults.rsquared_adj, self)
        )

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.nobs))
    def nobs(self):
        return self._wrap(self._nobs)

    @cache_readonly
    @Appender(RegressionModel.df_resid.__doc__)
    def df_resid(self):
        return self._wrap(self._nobs - self.df_model - self._k_constant)

    @cache_readonly
    @Appender(RegressionResults.use_t.__doc__)
    def use_t(self):
        return self._use_t

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.ess))
    def ess(self):
        return self._wrap(call_cached_func(RegressionResults.ess, self))

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.mse_model))
    def mse_model(self):
        return self._wrap(call_cached_func(RegressionResults.mse_model, self))

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.mse_resid))
    def mse_resid(self):
        return self._wrap(call_cached_func(RegressionResults.mse_resid, self))

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.mse_total))
    def mse_total(self):
        return self._wrap(call_cached_func(RegressionResults.mse_total, self))

    @cache_readonly
    def _cov_params(self):
        if self._cov_type == "nonrobust":
            return self._s2[:, None, None] * self._xpxi
        else:
            return self._xpxi @ self._xepxe @ self._xpxi

    def cov_params(self):
        """
        Estimated parameter covariance

        Returns
        -------
        array_like
            The estimated model covariances. If the original input is a numpy
            array, the returned covariance is a 3-d array with shape
            (nobs, nvar, nvar). If the original inputs are pandas types, then
            the returned covariance is a DataFrame with a MultiIndex with
            key (observation, variable), so that the covariance for
            observation with index i is cov.loc[i].
        """
        return self._wrap(self._cov_params)

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.f_pvalue))
    def f_pvalue(self):
        with np.errstate(invalid="ignore"):
            return self._wrap(
                call_cached_func(RegressionResults.f_pvalue, self)
            )

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.fvalue))
    def fvalue(self):
        if self._cov_type == "nonrobust":
            return self.mse_model / self.mse_resid
        else:
            nobs = self._params.shape[0]
            stat = np.full(nobs, np.nan)
            k = self._params.shape[1]
            r = np.eye(k)
            locs = list(range(k))
            if self.k_constant:
                locs.pop(self.model.const_idx)
            if not locs:
                return stat
            r = r[locs]
            vcv = self._cov_params
            rvcvr = r @ vcv @ r.T
            p = self._params
            for i in range(nobs):
                rp = p[i : i + 1] @ r.T
                denom = rp.shape[1]
                inv_cov = np.linalg.inv(rvcvr[i])
                stat[i] = np.squeeze(rp @ inv_cov @ rp.T) / denom
            return stat

    @cache_readonly
    @Appender(get_cached_doc(RegressionResults.bse))
    def bse(self):
        with np.errstate(invalid="ignore"):
            return self._wrap(np.sqrt(np.diagonal(self._cov_params, 0, 2)))

    @cache_readonly
    @Appender(get_cached_doc(LikelihoodModelResults.tvalues))
    def tvalues(self):
        with np.errstate(invalid="ignore"):
            return self._wrap(
                call_cached_func(LikelihoodModelResults.tvalues, self)
            )

    @cache_readonly
    @Appender(get_cached_doc(LikelihoodModelResults.pvalues))
    def pvalues(self):
        if self.use_t:
            df_resid = getattr(self, "df_resid_inference", self.df_resid)
            df_resid = np.asarray(df_resid)[:, None]
            with np.errstate(invalid="ignore"):
                return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
        else:
            with np.errstate(invalid="ignore"):
                return stats.norm.sf(np.abs(self.tvalues)) * 2

    def _conf_int(self, alpha, cols):
        bse = np.asarray(self.bse)

        if self.use_t:
            dist = stats.t
            df_resid = getattr(self, "df_resid_inference", self.df_resid)
            df_resid = np.asarray(df_resid)[:, None]
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = stats.norm
            q = dist.ppf(1 - alpha / 2)

        params = np.asarray(self.params)
        lower = params - q * bse
        upper = params + q * bse
        if cols is not None:
            cols = np.asarray(cols)
            lower = lower[:, cols]
            upper = upper[:, cols]
        return np.asarray(list(zip(lower, upper)))

    @Appender(LikelihoodModelResults.conf_int.__doc__)
    def conf_int(self, alpha=0.05, cols=None):
        ci = self._conf_int(alpha, cols)
        if not self._use_pandas:
            return ci
        ci_names = ("lower", "upper")
        row_names = self.model.data.row_labels
        col_names = self.model.data.param_names
        if cols is not None:
            col_names = [col_names[i] for i in cols]
        mi = MultiIndex.from_product((col_names, ci_names))
        ci = np.reshape(np.swapaxes(ci, 1, 2), (ci.shape[0], -1))
        return DataFrame(ci, columns=mi, index=row_names)

    @property
    def cov_type(self):
        """Name of covariance estimator"""
        return self._cov_type

    @classmethod
    @Appender(LikelihoodModelResults.load.__doc__)
    def load(cls, fname):
        return LikelihoodModelResults.load(fname)

    remove_data = LikelihoodModelResults.remove_data

    @Appender(LikelihoodModelResults.save.__doc__)
    def save(self, fname, remove_data=False):
        return LikelihoodModelResults.save(self, fname, remove_data)

    def plot_recursive_coefficient(
        self,
        variables=None,
        alpha=0.05,
        legend_loc="upper left",
        fig=None,
        figsize=None,
    ):
        r"""
        Plot the recursively estimated coefficients on a given variable

        Parameters
        ----------
        variables : {int, str, Iterable[int], Iterable[str], None}, optional
            Integer index or string name of the variables whose coefficients
            to plot. Can also be an iterable of integers or strings. Default
            plots all coefficients.
        alpha : float, optional
            The confidence intervals for the coefficient are (1 - alpha)%. Set
            to None to exclude confidence intervals.
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig

        if alpha is not None:
            ci = self._conf_int(alpha, None)

        row_labels = self.model.data.row_labels
        if row_labels is None:
            row_labels = np.arange(self._params.shape[0])
        k_variables = self._params.shape[1]
        param_names = self.model.data.param_names
        if variables is None:
            variable_idx = list(range(k_variables))
        else:
            if isinstance(variables, (int, str)):
                variables = [variables]
            variable_idx = []
            for i in range(len(variables)):
                variable = variables[i]
                if variable in param_names:
                    variable_idx.append(param_names.index(variable))
                elif isinstance(variable, int):
                    variable_idx.append(variable)
                else:
                    msg = (
                        "variable {0} is not an integer and was not found "
                        "in the list of variable "
                        "names: {1}".format(
                            variables[i], ", ".join(param_names)
                        )
                    )
                    raise ValueError(msg)

        _import_mpl()
        fig = create_mpl_fig(fig, figsize)

        loc = 0
        import pandas as pd

        if isinstance(row_labels, pd.PeriodIndex):
            row_labels = row_labels.to_timestamp()
        row_labels = np.asarray(row_labels)
        for i in variable_idx:
            ax = fig.add_subplot(len(variable_idx), 1, loc + 1)
            params = self._params[:, i]
            valid = ~np.isnan(self._params[:, i])
            row_lbl = row_labels[valid]
            ax.plot(row_lbl, params[valid])
            if alpha is not None:
                this_ci = np.reshape(ci[:, :, i], (-1, 2))
                if not np.all(np.isnan(this_ci)):
                    ax.plot(
                        row_lbl, this_ci[:, 0][valid], "k:", label="Lower CI"
                    )
                    ax.plot(
                        row_lbl, this_ci[:, 1][valid], "k:", label="Upper CI"
                    )
                    if loc == 0:
                        ax.legend(loc=legend_loc)
            ax.set_xlim(row_lbl[0], row_lbl[-1])
            ax.set_title(param_names[i])
            loc += 1

        fig.tight_layout()
        return fig
