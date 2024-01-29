from statsmodels.compat.pandas import PD_LT_2, Appender, is_numeric_dtype
from statsmodels.compat.scipy import SP_LT_19

from typing import Sequence, Union

import numpy as np
import pandas as pd

if PD_LT_2:
    from pandas.core.dtypes.common import is_categorical_dtype
else:
    # After pandas 2 is the minium, use the isinstance check
    def is_categorical_dtype(dtype):
        return isinstance(dtype, pd.CategoricalDtype)

from scipy import stats

from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    float_like,
    int_like,
)

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
QUANTILES = np.array(PERCENTILES) / 100.0


def pd_ptp(df):
    return df.max() - df.min()


def nancount(x, axis=0):
    return (1 - np.isnan(x)).sum(axis=axis)


def nanptp(arr, axis=0):
    return np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)


def nanuss(arr, axis=0):
    return np.nansum(arr ** 2, axis=axis)


def nanpercentile(arr, axis=0):
    return np.nanpercentile(arr, PERCENTILES, axis=axis)


def nankurtosis(arr, axis=0):
    return stats.kurtosis(arr, axis=axis, nan_policy="omit")


def nanskewness(arr, axis=0):
    return stats.skew(arr, axis=axis, nan_policy="omit")


MISSING = {
    "obs": nancount,
    "mean": np.nanmean,
    "std": np.nanstd,
    "max": np.nanmax,
    "min": np.nanmin,
    "ptp": nanptp,
    "var": np.nanvar,
    "skew": nanskewness,
    "uss": nanuss,
    "kurtosis": nankurtosis,
    "percentiles": nanpercentile,
}


def _kurtosis(a):
    """
    wrapper for scipy.stats.kurtosis that returns nan instead of raising Error

    missing options
    """
    try:
        res = stats.kurtosis(a)
    except ValueError:
        res = np.nan
    return res


def _skew(a):
    """
    wrapper for scipy.stats.skew that returns nan instead of raising Error

    missing options
    """
    try:
        res = stats.skew(a)
    except ValueError:
        res = np.nan
    return res


def sign_test(samp, mu0=0):
    """
    Signs test

    Parameters
    ----------
    samp : array_like
        1d array. The sample for which you want to perform the sign test.
    mu0 : float
        See Notes for the definition of the sign test. mu0 is 0 by
        default, but it is common to set it to the median.

    Returns
    -------
    M
    p-value

    Notes
    -----
    The signs test returns

    M = (N(+) - N(-))/2

    where N(+) is the number of values above `mu0`, N(-) is the number of
    values below.  Values equal to `mu0` are discarded.

    The p-value for M is calculated using the binomial distribution
    and can be interpreted the same as for a t-test. The test-statistic
    is distributed Binom(min(N(+), N(-)), n_trials, .5) where n_trials
    equals N(+) + N(-).

    See Also
    --------
    scipy.stats.wilcoxon
    """
    samp = np.asarray(samp)
    pos = np.sum(samp > mu0)
    neg = np.sum(samp < mu0)
    M = (pos - neg) / 2.0
    try:
        p = stats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue
    except AttributeError:
        # Remove after min SciPy >= 1.7
        p = stats.binom_test(min(pos, neg), pos + neg, 0.5)
    return M, p


NUMERIC_STATISTICS = (
    "nobs",
    "missing",
    "mean",
    "std_err",
    "ci",
    "std",
    "iqr",
    "iqr_normal",
    "mad",
    "mad_normal",
    "coef_var",
    "range",
    "max",
    "min",
    "skew",
    "kurtosis",
    "jarque_bera",
    "mode",
    "median",
    "percentiles",
)
CATEGORICAL_STATISTICS = ("nobs", "missing", "distinct", "top", "freq")
_additional = [
    stat for stat in CATEGORICAL_STATISTICS if stat not in NUMERIC_STATISTICS
]
DEFAULT_STATISTICS = NUMERIC_STATISTICS + tuple(_additional)


class Description:
    """
    Extended descriptive statistics for data

    Parameters
    ----------
    data : array_like
        Data to describe. Must be convertible to a pandas DataFrame.
    stats : Sequence[str], optional
        Statistics to include. If not provided the full set of statistics is
        computed. This list may evolve across versions to reflect best
        practices. Supported options are:
        "nobs", "missing", "mean", "std_err", "ci", "ci", "std", "iqr",
        "iqr_normal", "mad", "mad_normal", "coef_var", "range", "max",
        "min", "skew", "kurtosis", "jarque_bera", "mode", "freq",
        "median", "percentiles", "distinct", "top", and "freq". See Notes for
        details.
    numeric : bool, default True
        Whether to include numeric columns in the descriptive statistics.
    categorical : bool, default True
        Whether to include categorical columns in the descriptive statistics.
    alpha : float, default 0.05
        A number between 0 and 1 representing the size used to compute the
        confidence interval, which has coverage 1 - alpha.
    use_t : bool, default False
        Use the Student's t distribution to construct confidence intervals.
    percentiles : sequence[float]
        A distinct sequence of floating point values all between 0 and 100.
        The default percentiles are 1, 5, 10, 25, 50, 75, 90, 95, 99.
    ntop : int, default 5
        The number of top categorical labels to report. Default is

    Attributes
    ----------
    numeric_statistics
        The list of supported statistics for numeric data
    categorical_statistics
        The list of supported statistics for categorical data
    default_statistics
        The default list of statistics

    See Also
    --------
    pandas.DataFrame.describe
        Basic descriptive statistics
    describe
        A simplified version that returns a DataFrame

    Notes
    -----
    The selectable statistics include:

    * "nobs" - Number of observations
    * "missing" - Number of missing observations
    * "mean" - Mean
    * "std_err" - Standard Error of the mean assuming no correlation
    * "ci" - Confidence interval with coverage (1 - alpha) using the normal or
      t. This option creates two entries in any tables: lower_ci and upper_ci.
    * "std" - Standard Deviation
    * "iqr" - Interquartile range
    * "iqr_normal" - Interquartile range relative to a Normal
    * "mad" - Mean absolute deviation
    * "mad_normal" - Mean absolute deviation relative to a Normal
    * "coef_var" - Coefficient of variation
    * "range" - Range between the maximum and the minimum
    * "max" - The maximum
    * "min" - The minimum
    * "skew" - The skewness defined as the standardized 3rd central moment
    * "kurtosis" - The kurtosis defined as the standardized 4th central moment
    * "jarque_bera" - The Jarque-Bera test statistic for normality based on
      the skewness and kurtosis. This option creates two entries, jarque_bera
      and jarque_beta_pval.
    * "mode" - The mode of the data. This option creates two entries in all tables,
      mode and mode_freq which is the empirical frequency of the modal value.
    * "median" - The median of the data.
    * "percentiles" - The percentiles. Values included depend on the input value of
      ``percentiles``.
    * "distinct" - The number of distinct categories in a categorical.
    * "top" - The mode common categories. Labeled top_n for n in 1, 2, ..., ``ntop``.
    * "freq" - The frequency of the common categories. Labeled freq_n for n in 1,
      2, ..., ``ntop``.
    """

    _int_fmt = ["nobs", "missing", "distinct"]
    numeric_statistics = NUMERIC_STATISTICS
    categorical_statistics = CATEGORICAL_STATISTICS
    default_statistics = DEFAULT_STATISTICS

    def __init__(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        stats: Sequence[str] = None,
        *,
        numeric: bool = True,
        categorical: bool = True,
        alpha: float = 0.05,
        use_t: bool = False,
        percentiles: Sequence[Union[int, float]] = PERCENTILES,
        ntop: bool = 5,
    ):
        data_arr = data
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            data_arr = array_like(data, "data", maxdim=2)
        if data_arr.ndim == 1:
            data = pd.Series(data)
        numeric = bool_like(numeric, "numeric")
        categorical = bool_like(categorical, "categorical")
        include = []
        col_types = ""
        if numeric:
            include.append(np.number)
            col_types = "numeric"
        if categorical:
            include.append("category")
            col_types += "and " if col_types != "" else ""
            col_types += "categorical"
        if not numeric and not categorical:
            raise ValueError(
                "At least one of numeric and categorical must be True"
            )
        self._data = pd.DataFrame(data).select_dtypes(include)
        if self._data.shape[1] == 0:

            raise ValueError(
                f"Selecting {col_types} results in an empty DataFrame"
            )
        self._is_numeric = [is_numeric_dtype(dt) for dt in self._data.dtypes]
        self._is_cat_like = [
            is_categorical_dtype(dt) for dt in self._data.dtypes
        ]

        if stats is not None:
            undef = [stat for stat in stats if stat not in DEFAULT_STATISTICS]
            if undef:
                raise ValueError(
                    f"{', '.join(undef)} are not known statistics"
                )
        self._stats = (
            list(DEFAULT_STATISTICS) if stats is None else list(stats)
        )
        self._ntop = int_like(ntop, "ntop")
        self._compute_top = "top" in self._stats
        self._compute_freq = "freq" in self._stats
        if self._compute_top and self._ntop <= 0 < sum(self._is_cat_like):
            raise ValueError("top must be a non-negative integer")

        # Expand special stats
        replacements = {
            "mode": ["mode", "mode_freq"],
            "ci": ["upper_ci", "lower_ci"],
            "jarque_bera": ["jarque_bera", "jarque_bera_pval"],
            "top": [f"top_{i}" for i in range(1, self._ntop + 1)],
            "freq": [f"freq_{i}" for i in range(1, self._ntop + 1)],
        }

        for key in replacements:
            if key in self._stats:
                idx = self._stats.index(key)
                self._stats = (
                    self._stats[:idx]
                    + replacements[key]
                    + self._stats[idx + 1 :]
                )

        self._percentiles = array_like(
            percentiles, "percentiles", maxdim=1, dtype="d"
        )
        self._percentiles = np.sort(self._percentiles)
        if np.unique(self._percentiles).shape[0] != self._percentiles.shape[0]:
            raise ValueError("percentiles must be distinct")
        if np.any(self._percentiles >= 100) or np.any(self._percentiles <= 0):
            raise ValueError("percentiles must be strictly between 0 and 100")
        self._alpha = float_like(alpha, "alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        self._use_t = bool_like(use_t, "use_t")

    def _reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[[s for s in self._stats if s in df.index]]

    @cache_readonly
    def frame(self) -> pd.DataFrame:
        """
        Descriptive statistics for both numeric and categorical data

        Returns
        -------
        DataFrame
            The statistics
        """
        numeric = self.numeric
        categorical = self.categorical
        if categorical.shape[1] == 0:
            return numeric
        elif numeric.shape[1] == 0:
            return categorical
        df = pd.concat([numeric, categorical], axis=1)
        return self._reorder(df[self._data.columns])

    @cache_readonly
    def numeric(self) -> pd.DataFrame:
        """
        Descriptive statistics for numeric data

        Returns
        -------
        DataFrame
            The statistics of the numeric columns
        """
        df: pd.DataFrame = self._data.loc[:, self._is_numeric]
        cols = df.columns
        _, k = df.shape
        std = df.std()
        count = df.count()
        mean = df.mean()
        mad = (df - mean).abs().mean()
        std_err = std.copy()
        std_err.loc[count > 0] /= count.loc[count > 0] ** 0.5
        if self._use_t:
            q = stats.t(count - 1).ppf(1.0 - self._alpha / 2)
        else:
            q = stats.norm.ppf(1.0 - self._alpha / 2)

        def _mode(ser):
            dtype = ser.dtype if isinstance(ser.dtype, np.dtype) else ser.dtype.numpy_dtype
            ser_no_missing = ser.dropna().to_numpy(dtype=dtype)
            kwargs = {} if SP_LT_19 else {"keepdims": True}
            mode_res = stats.mode(ser_no_missing, **kwargs)
            # Changes in SciPy 1.10
            if np.isscalar(mode_res[0]):
                return float(mode_res[0]), mode_res[1]
            if mode_res[0].shape[0] > 0:
                return [float(val) for val in mode_res]
            return np.nan, np.nan

        mode_values = df.apply(_mode).T
        if mode_values.size > 0:
            if isinstance(mode_values, pd.DataFrame):
                # pandas 1.0 or later
                mode = np.asarray(mode_values[0], dtype=float)
                mode_counts = np.asarray(mode_values[1], dtype=np.int64)
            else:
                # pandas before 1.0 returns a Series of 2-elem list
                mode = []
                mode_counts = []
                for idx in mode_values.index:
                    val = mode_values.loc[idx]
                    mode.append(val[0])
                    mode_counts.append(val[1])
                mode = np.atleast_1d(mode)
                mode_counts = np.atleast_1d(mode_counts)
        else:
            mode = mode_counts = np.empty(0)
        loc = count > 0
        mode_freq = np.full(mode.shape[0], np.nan)
        mode_freq[loc] = mode_counts[loc] / count.loc[loc]
        # TODO: Workaround for pandas AbstractMethodError in extension
        #  types. Remove when quantile is supported for these
        _df = df
        try:
            from pandas.api.types import is_extension_array_dtype
            _df = df.copy()
            for col in df:
                if is_extension_array_dtype(df[col].dtype):
                    if _df[col].isnull().any():
                        _df[col] = _df[col].fillna(np.nan)
        except ImportError:
            pass

        if df.shape[1] > 0:
            iqr = _df.quantile(0.75) - _df.quantile(0.25)
        else:
            iqr = mean

        def _safe_jarque_bera(c):
            a = np.asarray(c)
            if a.shape[0] < 2:
                return (np.nan,) * 4
            return jarque_bera(a)

        jb = df.apply(
            lambda x: list(_safe_jarque_bera(x.dropna())), result_type="expand"
        ).T
        nan_mean = mean.copy()
        nan_mean.loc[nan_mean == 0] = np.nan
        coef_var = std / nan_mean

        results = {
            "nobs": pd.Series(
                np.ones(k, dtype=np.int64) * df.shape[0], index=cols
            ),
            "missing": df.shape[0] - count,
            "mean": mean,
            "std_err": std_err,
            "upper_ci": mean + q * std_err,
            "lower_ci": mean - q * std_err,
            "std": std,
            "iqr": iqr,
            "mad": mad,
            "coef_var": coef_var,
            "range": pd_ptp(df),
            "max": df.max(),
            "min": df.min(),
            "skew": jb[2],
            "kurtosis": jb[3],
            "iqr_normal": iqr / np.diff(stats.norm.ppf([0.25, 0.75])),
            "mad_normal": mad / np.sqrt(2 / np.pi),
            "jarque_bera": jb[0],
            "jarque_bera_pval": jb[1],
            "mode": pd.Series(mode, index=cols),
            "mode_freq": pd.Series(mode_freq, index=cols),
            "median": df.median(),
        }
        final = {k: v for k, v in results.items() if k in self._stats}
        results_df = pd.DataFrame(
            list(final.values()), columns=cols, index=list(final.keys())
        )
        if "percentiles" not in self._stats:
            return results_df
        # Pandas before 1.0 cannot handle empty DF
        if df.shape[1] > 0:
            # TODO: Remove when extension types support quantile
            perc = _df.quantile(self._percentiles / 100).astype(float)
        else:
            perc = pd.DataFrame(index=self._percentiles / 100, dtype=float)
        if np.all(np.floor(100 * perc.index) == (100 * perc.index)):
            perc.index = [f"{int(100 * idx)}%" for idx in perc.index]
        else:
            dupe = True
            scale = 100
            index = perc.index
            while dupe:
                scale *= 10
                idx = np.floor(scale * perc.index)
                if np.all(np.diff(idx) > 0):
                    dupe = False
            index = np.floor(scale * index) / (scale / 100)
            fmt = f"0.{len(str(scale//100))-1}f"
            output = f"{{0:{fmt}}}%"
            perc.index = [output.format(val) for val in index]

        # Add in the names of the percentiles to the output
        self._stats = self._stats + perc.index.tolist()

        return self._reorder(pd.concat([results_df, perc], axis=0))

    @cache_readonly
    def categorical(self) -> pd.DataFrame:
        """
        Descriptive statistics for categorical data

        Returns
        -------
        DataFrame
            The statistics of the categorical columns
        """

        df = self._data.loc[:, [col for col in self._is_cat_like]]
        k = df.shape[1]
        cols = df.columns
        vc = {col: df[col].value_counts(normalize=True) for col in df}
        distinct = pd.Series(
            {col: vc[col].shape[0] for col in vc}, dtype=np.int64
        )
        top = {}
        freq = {}
        for col in vc:
            single = vc[col]
            if single.shape[0] >= self._ntop:
                top[col] = single.index[: self._ntop]
                freq[col] = np.asarray(single.iloc[:5])
            else:
                val = list(single.index)
                val += [None] * (self._ntop - len(val))
                top[col] = val
                freq_val = list(single)
                freq_val += [np.nan] * (self._ntop - len(freq_val))
                freq[col] = np.asarray(freq_val)
        index = [f"top_{i}" for i in range(1, self._ntop + 1)]
        top_df = pd.DataFrame(top, dtype="object", index=index, columns=cols)
        index = [f"freq_{i}" for i in range(1, self._ntop + 1)]
        freq_df = pd.DataFrame(freq, dtype="object", index=index, columns=cols)

        results = {
            "nobs": pd.Series(
                np.ones(k, dtype=np.int64) * df.shape[0], index=cols
            ),
            "missing": df.shape[0] - df.count(),
            "distinct": distinct,
        }
        final = {k: v for k, v in results.items() if k in self._stats}
        results_df = pd.DataFrame(
            list(final.values()),
            columns=cols,
            index=list(final.keys()),
            dtype="object",
        )
        if self._compute_top:
            results_df = pd.concat([results_df, top_df], axis=0)
        if self._compute_freq:
            results_df = pd.concat([results_df, freq_df], axis=0)

        return self._reorder(results_df)

    def summary(self) -> SimpleTable:
        """
        Summary table of the descriptive statistics

        Returns
        -------
        SimpleTable
            A table instance supporting export to text, csv and LaTeX
        """
        df = self.frame.astype(object)
        if df.isnull().any().any():
            df = df.fillna("")
        cols = [str(col) for col in df.columns]
        stubs = [str(idx) for idx in df.index]
        data = []
        for _, row in df.iterrows():
            data.append([v for v in row])

        def _formatter(v):
            if isinstance(v, str):
                return v
            elif v // 1 == v:
                return str(int(v))
            return f"{v:0.4g}"

        return SimpleTable(
            data,
            header=cols,
            stubs=stubs,
            title="Descriptive Statistics",
            txt_fmt={"data_fmts": {0: "%s", 1: _formatter}},
            datatypes=[1] * len(data),
        )

    def __str__(self) -> str:
        return str(self.summary().as_text())


ds = Docstring(Description.__doc__)
ds.replace_block(
    "Returns", Parameter(None, "DataFrame", ["Descriptive statistics"])
)
ds.replace_block("Attributes", [])
ds.replace_block(
    "See Also",
    [
        (
            [("pandas.DataFrame.describe", None)],
            ["Basic descriptive statistics"],
        ),
        (
            [("Description", None)],
            ["Descriptive statistics class with additional output options"],
        ),
    ],
)


@Appender(str(ds))
def describe(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    stats: Sequence[str] = None,
    *,
    numeric: bool = True,
    categorical: bool = True,
    alpha: float = 0.05,
    use_t: bool = False,
    percentiles: Sequence[Union[int, float]] = PERCENTILES,
    ntop: bool = 5,
) -> pd.DataFrame:
    return Description(
        data,
        stats,
        numeric=numeric,
        categorical=categorical,
        alpha=alpha,
        use_t=use_t,
        percentiles=percentiles,
        ntop=ntop,
    ).frame


class Describe:
    """
    Removed.
    """

    def __init__(self, dataset):
        raise NotImplementedError("Describe has been removed")
