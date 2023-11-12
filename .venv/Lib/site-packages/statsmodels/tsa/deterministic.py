from statsmodels.compat.pandas import Appender, is_int_index, to_numpy

from abc import ABC, abstractmethod
import datetime as dt
from typing import Hashable, List, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from scipy.linalg import qr

from statsmodels.iolib.summary import d_or_f
from statsmodels.tools.validation import (
    bool_like,
    float_like,
    required_int_like,
    string_like,
)
from statsmodels.tsa.tsatools import freq_to_period

DateLike = Union[dt.datetime, pd.Timestamp, np.datetime64]
IntLike = Union[int, np.integer]


START_BEFORE_INDEX_ERR = """\
start is less than the first observation in the index. Values can only be \
created for observations after the start of the index.
"""


class DeterministicTerm(ABC):
    """Abstract Base Class for all Deterministic Terms"""

    # Set _is_dummy if the term is a dummy variable process
    _is_dummy = False

    @property
    def is_dummy(self) -> bool:
        """Flag indicating whether the values produced are dummy variables"""
        return self._is_dummy

    @abstractmethod
    def in_sample(self, index: Sequence[Hashable]) -> pd.DataFrame:
        """
        Produce deterministic trends for in-sample fitting.

        Parameters
        ----------
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.

        Returns
        -------
        DataFrame
            A DataFrame containing the deterministic terms.
        """

    @abstractmethod
    def out_of_sample(
        self,
        steps: int,
        index: Sequence[Hashable],
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.DataFrame:
        """
        Produce deterministic trends for out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of steps to forecast
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.
        forecast_index : index_like
            An Index or index-like object to use for the forecasts. If
            provided must have steps elements.

        Returns
        -------
        DataFrame
            A DataFrame containing the deterministic terms.
        """

    @abstractmethod
    def __str__(self) -> str:
        """A meaningful string representation of the term"""

    def __hash__(self) -> int:
        name: Tuple[Hashable, ...] = (type(self).__name__,)
        return hash(name + self._eq_attr)

    @property
    @abstractmethod
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        """tuple of attributes that are used for equality comparison"""

    @staticmethod
    def _index_like(index: Sequence[Hashable]) -> pd.Index:
        if isinstance(index, pd.Index):
            return index
        try:
            return pd.Index(index)
        except Exception:
            raise TypeError("index must be a pandas Index or index-like")

    @staticmethod
    def _extend_index(
        index: pd.Index,
        steps: int,
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.Index:
        """Extend the forecast index"""
        if forecast_index is not None:
            forecast_index = DeterministicTerm._index_like(forecast_index)
            assert isinstance(forecast_index, pd.Index)
            if forecast_index.shape[0] != steps:
                raise ValueError(
                    "The number of values in forecast_index "
                    f"({forecast_index.shape[0]}) must match steps ({steps})."
                )
            return forecast_index
        if isinstance(index, pd.PeriodIndex):
            return pd.period_range(
                index[-1] + 1, periods=steps, freq=index.freq
            )
        elif isinstance(index, pd.DatetimeIndex) and index.freq is not None:
            next_obs = pd.date_range(index[-1], freq=index.freq, periods=2)[1]
            return pd.date_range(next_obs, freq=index.freq, periods=steps)
        elif isinstance(index, pd.RangeIndex):
            assert isinstance(index, pd.RangeIndex)
            try:
                step = index.step
                start = index.stop
            except AttributeError:
                # TODO: Remove after pandas min ver is 1.0.0+
                step = index[-1] - index[-2] if len(index) > 1 else 1
                start = index[-1] + step
            stop = start + step * steps
            return pd.RangeIndex(start, stop, step=step)
        elif is_int_index(index) and np.all(np.diff(index) == 1):
            idx_arr = np.arange(index[-1] + 1, index[-1] + steps + 1)
            return pd.Index(idx_arr)
        # default range index
        import warnings

        warnings.warn(
            "Only PeriodIndexes, DatetimeIndexes with a frequency set, "
            "RangesIndexes, and Index with a unit increment support "
            "extending. The index is set will contain the position relative "
            "to the data length.",
            UserWarning,
            stacklevel=2,
        )
        nobs = index.shape[0]
        return pd.RangeIndex(nobs + 1, nobs + steps + 1)

    def __repr__(self) -> str:
        return self.__str__() + f" at 0x{id(self):0x}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            own_attr = self._eq_attr
            oth_attr = other._eq_attr
            if len(own_attr) != len(oth_attr):
                return False
            return all([a == b for a, b in zip(own_attr, oth_attr)])
        else:
            return False


class TimeTrendDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for all Time Trend Deterministic Terms"""

    def __init__(self, constant: bool = True, order: int = 0) -> None:
        self._constant = bool_like(constant, "constant")
        self._order = required_int_like(order, "order")

    @property
    def constant(self) -> bool:
        """Flag indicating that a constant is included"""
        return self._constant

    @property
    def order(self) -> int:
        """Order of the time trend"""
        return self._order

    @property
    def _columns(self) -> List[str]:
        columns = []
        trend_names = {1: "trend", 2: "trend_squared", 3: "trend_cubed"}
        if self._constant:
            columns.append("const")
        for power in range(1, self._order + 1):
            if power in trend_names:
                columns.append(trend_names[power])
            else:
                columns.append(f"trend**{power}")
        return columns

    def _get_terms(self, locs: np.ndarray) -> np.ndarray:
        nterms = int(self._constant) + self._order
        terms = np.tile(locs, (1, nterms))
        power = np.zeros((1, nterms), dtype=int)
        power[0, int(self._constant) :] = np.arange(1, self._order + 1)
        terms **= power
        return terms

    def __str__(self) -> str:
        terms = []
        if self._constant:
            terms.append("Constant")
        if self._order:
            terms.append(f"Powers 1 to {self._order + 1}")
        if not terms:
            terms = ["Empty"]
        terms_str = ",".join(terms)
        return f"TimeTrend({terms_str})"


class TimeTrend(TimeTrendDeterministicTerm):
    """
    Constant and time trend determinstic terms

    Parameters
    ----------
    constant : bool
        Flag indicating whether a constant should be included.
    order : int
        A non-negative int containing the powers to include (1, 2, ..., order).

    See Also
    --------
    DeterministicProcess
    Seasonality
    Fourier
    CalendarTimeTrend

    Examples
    --------
    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import TimeTrend
    >>> data = sunspots.load_pandas().data
    >>> trend_gen = TimeTrend(True, 3)
    >>> trend_gen.in_sample(data.index)
    """

    def __init__(self, constant: bool = True, order: int = 0) -> None:
        super().__init__(constant, order)

    @classmethod
    def from_string(cls, trend: str) -> "TimeTrend":
        """
        Create a TimeTrend from a string description.

        Provided for compatibility with common string names.

        Parameters
        ----------
        trend : {"n", "c", "t", "ct", "ctt"}
            The string representation of the time trend. The terms are:

            * "n": No trend terms
            * "c": A constant only
            * "t": Linear time trend only
            * "ct": A constant and a time trend
            * "ctt": A constant, a time trend and a quadratic time trend

        Returns
        -------
        TimeTrend
            The TimeTrend instance.
        """
        constant = trend.startswith("c")
        order = 0
        if "tt" in trend:
            order = 2
        elif "t" in trend:
            order = 1
        return cls(constant=constant, order=order)

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(
        self, index: Union[Sequence[Hashable], pd.Index]
    ) -> pd.DataFrame:
        index = self._index_like(index)
        nobs = index.shape[0]
        locs = np.arange(1, nobs + 1, dtype=np.double)[:, None]
        terms = self._get_terms(locs)
        return pd.DataFrame(terms, columns=self._columns, index=index)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(
        self,
        steps: int,
        index: Union[Sequence[Hashable], pd.Index],
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.DataFrame:
        index = self._index_like(index)
        nobs = index.shape[0]
        fcast_index = self._extend_index(index, steps, forecast_index)
        locs = np.arange(nobs + 1, nobs + steps + 1, dtype=np.double)[:, None]
        terms = self._get_terms(locs)
        return pd.DataFrame(terms, columns=self._columns, index=fcast_index)

    @property
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        return self._constant, self._order


class Seasonality(DeterministicTerm):
    """
    Seasonal dummy deterministic terms

    Parameters
    ----------
    period : int
        The length of a full cycle. Must be >= 2.
    initial_period : int
        The seasonal index of the first observation. 1-indexed so must
        be in {1, 2, ..., period}.

    See Also
    --------
    DeterministicProcess
    TimeTrend
    Fourier
    CalendarSeasonality

    Examples
    --------
    Solar data has an 11-year cycle

    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import Seasonality
    >>> data = sunspots.load_pandas().data
    >>> seas_gen = Seasonality(11)
    >>> seas_gen.in_sample(data.index)

    To start at a season other than 1

    >>> seas_gen = Seasonality(11, initial_period=4)
    >>> seas_gen.in_sample(data.index)
    """

    _is_dummy = True

    def __init__(self, period: int, initial_period: int = 1) -> None:
        self._period = required_int_like(period, "period")
        self._initial_period = required_int_like(
            initial_period, "initial_period"
        )
        if period < 2:
            raise ValueError("period must be >= 2")
        if not 1 <= self._initial_period <= period:
            raise ValueError("initial_period must be in {1, 2, ..., period}")

    @property
    def period(self) -> int:
        """The period of the seasonality"""
        return self._period

    @property
    def initial_period(self) -> int:
        """The seasonal index of the first observation"""
        return self._initial_period

    @classmethod
    def from_index(
        cls, index: Union[Sequence[Hashable], pd.DatetimeIndex, pd.PeriodIndex]
    ) -> "Seasonality":
        """
        Construct a seasonality directly from an index using its frequency.

        Parameters
        ----------
        index : {DatetimeIndex, PeriodIndex}
            An index with its frequency (`freq`) set.

        Returns
        -------
        Seasonality
            The initialized Seasonality instance.
        """
        index = cls._index_like(index)
        if isinstance(index, pd.PeriodIndex):
            freq = index.freq
        elif isinstance(index, pd.DatetimeIndex):
            freq = index.freq if index.freq else index.inferred_freq
        else:
            raise TypeError("index must be a DatetimeIndex or PeriodIndex")
        if freq is None:
            raise ValueError("index must have a freq or inferred_freq set")
        period = freq_to_period(freq)
        return cls(period=period)

    @property
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        return self._period, self._initial_period

    def __str__(self) -> str:
        return f"Seasonality(period={self._period})"

    @property
    def _columns(self) -> List[str]:
        period = self._period
        columns = []
        for i in range(1, period + 1):
            columns.append(f"s({i},{period})")
        return columns

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(
        self, index: Union[Sequence[Hashable], pd.Index]
    ) -> pd.DataFrame:
        index = self._index_like(index)
        nobs = index.shape[0]
        period = self._period
        term = np.zeros((nobs, period))
        offset = self._initial_period - 1
        for i in range(period):
            col = (i + offset) % period
            term[i::period, col] = 1
        return pd.DataFrame(term, columns=self._columns, index=index)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(
        self,
        steps: int,
        index: Union[Sequence[Hashable], pd.Index],
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        nobs = index.shape[0]
        period = self._period
        term = np.zeros((steps, period))
        offset = self._initial_period - 1
        for i in range(period):
            col_loc = (nobs + offset + i) % period
            term[i::period, col_loc] = 1
        return pd.DataFrame(term, columns=self._columns, index=fcast_index)


class FourierDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for all Fourier Deterministic Terms"""

    def __init__(self, order: int) -> None:
        self._order = required_int_like(order, "terms")

    @property
    def order(self) -> int:
        """The order of the Fourier terms included"""
        return self._order

    def _get_terms(self, locs: np.ndarray) -> np.ndarray:
        locs = 2 * np.pi * locs.astype(np.double)
        terms = np.empty((locs.shape[0], 2 * self._order))
        for i in range(self._order):
            for j, func in enumerate((np.sin, np.cos)):
                terms[:, 2 * i + j] = func((i + 1) * locs)
        return terms


class Fourier(FourierDeterministicTerm):
    r"""
    Fourier series deterministic terms

    Parameters
    ----------
    period : int
        The length of a full cycle. Must be >= 2.
    order : int
        The number of Fourier components to include. Must be <= 2*period.

    See Also
    --------
    DeterministicProcess
    TimeTrend
    Seasonality
    CalendarFourier

    Notes
    -----
    Both a sine and a cosine term are included for each i=1, ..., order

    .. math::

       f_{i,s,t} & = \sin\left(2 \pi i \times \frac{t}{m} \right)  \\
       f_{i,c,t} & = \cos\left(2 \pi i \times \frac{t}{m} \right)

    where m is the length of the period.

    Examples
    --------
    Solar data has an 11-year cycle

    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import Fourier
    >>> data = sunspots.load_pandas().data
    >>> fourier_gen = Fourier(11, order=2)
    >>> fourier_gen.in_sample(data.index)
    """
    _is_dummy = False

    def __init__(self, period: float, order: int):
        super().__init__(order)
        self._period = float_like(period, "period")
        if 2 * self._order > self._period:
            raise ValueError("2 * order must be <= period")

    @property
    def period(self) -> float:
        """The period of the Fourier terms"""
        return self._period

    @property
    def _columns(self) -> List[str]:
        period = self._period
        fmt_period = d_or_f(period).strip()
        columns = []
        for i in range(1, self._order + 1):
            for typ in ("sin", "cos"):
                columns.append(f"{typ}({i},{fmt_period})")
        return columns

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(
        self, index: Union[Sequence[Hashable], pd.Index]
    ) -> pd.DataFrame:
        index = self._index_like(index)
        nobs = index.shape[0]
        terms = self._get_terms(np.arange(nobs) / self._period)
        return pd.DataFrame(terms, index=index, columns=self._columns)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(
        self,
        steps: int,
        index: Union[Sequence[Hashable], pd.Index],
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        nobs = index.shape[0]
        terms = self._get_terms(np.arange(nobs, nobs + steps) / self._period)
        return pd.DataFrame(terms, index=fcast_index, columns=self._columns)

    @property
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        return self._period, self._order

    def __str__(self) -> str:
        return f"Fourier(period={self._period}, order={self._order})"


class CalendarDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for calendar deterministic terms"""

    def __init__(self, freq: str) -> None:
        try:
            index = pd.date_range("2020-01-01", freq=freq, periods=1)
            self._freq = index.freq
        except ValueError:
            raise ValueError("freq is not understood by pandas")

    @property
    def freq(self) -> str:
        """The frequency of the deterministic terms"""
        return self._freq.freqstr

    def _compute_ratio(
        self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]
    ) -> np.ndarray:
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp()
        delta = index - index.to_period(self._freq).to_timestamp()
        pi = index.to_period(self._freq)
        gap = (pi + 1).to_timestamp() - pi.to_timestamp()
        return to_numpy(delta) / to_numpy(gap)

    def _check_index_type(
        self,
        index: pd.Index,
        allowed: Union[Type, Tuple[Type, ...]] = (
            pd.DatetimeIndex,
            pd.PeriodIndex,
        ),
    ) -> Union[pd.DatetimeIndex, pd.PeriodIndex]:
        if isinstance(allowed, type):
            allowed = (allowed,)
        if not isinstance(index, allowed):
            if len(allowed) == 1:
                allowed_types = "a " + allowed[0].__name__
            else:
                allowed_types = ", ".join(a.__name__ for a in allowed[:-1])
                if len(allowed) > 2:
                    allowed_types += ","
                allowed_types += " and " + allowed[-1].__name__
            msg = (
                f"{type(self).__name__} terms can only be computed from "
                f"{allowed_types}"
            )
            raise TypeError(msg)
        assert isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex))
        return index


class CalendarFourier(CalendarDeterministicTerm, FourierDeterministicTerm):
    r"""
    Fourier series deterministic terms based on calendar time

    Parameters
    ----------
    freq : str
        A string convertible to a pandas frequency.
    order : int
        The number of Fourier components to include. Must be <= 2*period.

    See Also
    --------
    DeterministicProcess
    CalendarTimeTrend
    CalendarSeasonality
    Fourier

    Notes
    -----
    Both a sine and a cosine term are included for each i=1, ..., order

    .. math::

       f_{i,s,t} & = \sin\left(2 \pi i \tau_t \right)  \\
       f_{i,c,t} & = \cos\left(2 \pi i \tau_t \right)

    where m is the length of the period and :math:`\tau_t` is the frequency
    normalized time.  For example, when freq is "D" then an observation with
    a timestamp of 12:00:00 would have :math:`\tau_t=0.5`.

    Examples
    --------
    Here we simulate irregularly spaced hourly data and construct the calendar
    Fourier terms for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarFourier
    >>> cal_fourier_gen = CalendarFourier("D", 2)
    >>> cal_fourier_gen.in_sample(index)
    """

    def __init__(self, freq: str, order: int) -> None:
        super().__init__(freq)
        FourierDeterministicTerm.__init__(self, order)
        self._order = required_int_like(order, "terms")

    @property
    def _columns(self) -> List[str]:
        columns = []
        for i in range(1, self._order + 1):
            for typ in ("sin", "cos"):
                columns.append(f"{typ}({i},freq={self._freq.freqstr})")
        return columns

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(
        self, index: Union[Sequence[Hashable], pd.Index]
    ) -> pd.DataFrame:
        index = self._index_like(index)
        index = self._check_index_type(index)

        ratio = self._compute_ratio(index)
        terms = self._get_terms(ratio)
        return pd.DataFrame(terms, index=index, columns=self._columns)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(
        self,
        steps: int,
        index: Union[Sequence[Hashable], pd.Index],
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        self._check_index_type(fcast_index)
        assert isinstance(fcast_index, (pd.DatetimeIndex, pd.PeriodIndex))
        ratio = self._compute_ratio(fcast_index)
        terms = self._get_terms(ratio)
        return pd.DataFrame(terms, index=fcast_index, columns=self._columns)

    @property
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        return self._freq.freqstr, self._order

    def __str__(self) -> str:
        return f"Fourier(freq={self._freq.freqstr}, order={self._order})"


class CalendarSeasonality(CalendarDeterministicTerm):
    """
    Seasonal dummy deterministic terms based on calendar time

    Parameters
    ----------
    freq : str
        The frequency of the seasonal effect.
    period : str
        The pandas frequency string describing the full period.

    See Also
    --------
    DeterministicProcess
    CalendarTimeTrend
    CalendarFourier
    Seasonality

    Examples
    --------
    Here we simulate irregularly spaced data (in time) and hourly seasonal
    dummies for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarSeasonality
    >>> cal_seas_gen = CalendarSeasonality("H", "D")
    >>> cal_seas_gen.in_sample(index)
    """

    _is_dummy = True

    # out_of: freq
    _supported = {
        "W": {"H": 24 * 7, "B": 5, "D": 7},
        "D": {"H": 24},
        "Q": {"M": 3},
        "A": {"M": 12, "Q": 4},
    }

    def __init__(self, freq: str, period: str) -> None:
        freq_options: Set[str] = set()
        freq_options.update(
            *[list(val.keys()) for val in self._supported.values()]
        )
        period_options = list(self._supported.keys())

        freq = string_like(
            freq, "freq", options=tuple(freq_options), lower=False
        )
        period = string_like(
            period, "period", options=period_options, lower=False
        )
        if freq not in self._supported[period]:
            raise ValueError(
                f"The combination of freq={freq} and "
                f"period={period} is not supported."
            )
        super().__init__(freq)
        self._period = period
        self._freq_str = self._freq.freqstr.split("-")[0]

    @property
    def freq(self) -> str:
        """The frequency of the deterministic terms"""
        return self._freq.freqstr

    @property
    def period(self) -> str:
        """The full period"""
        return self._period

    def _weekly_to_loc(
        self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]
    ) -> np.ndarray:
        if self._freq.freqstr == "H":
            return index.hour + 24 * index.dayofweek
        elif self._freq.freqstr == "D":
            return index.dayofweek
        else:  # "B"
            bdays = pd.bdate_range("2000-1-1", periods=10).dayofweek.unique()
            loc = index.dayofweek
            if not loc.isin(bdays).all():
                raise ValueError(
                    "freq is B but index contains days that are not business "
                    "days."
                )
            return loc

    def _daily_to_loc(
        self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]
    ) -> np.ndarray:
        return index.hour

    def _quarterly_to_loc(
        self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]
    ) -> np.ndarray:
        return (index.month - 1) % 3

    def _annual_to_loc(
        self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]
    ) -> np.ndarray:
        if self._freq.freqstr == "M":
            return index.month - 1
        else:  # "Q"
            return index.quarter - 1

    def _get_terms(
        self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]
    ) -> np.ndarray:
        if self._period == "D":
            locs = self._daily_to_loc(index)
        elif self._period == "W":
            locs = self._weekly_to_loc(index)
        elif self._period == "Q":
            locs = self._quarterly_to_loc(index)
        else:  # "A":
            locs = self._annual_to_loc(index)
        full_cycle = self._supported[self._period][self._freq_str]
        terms = np.zeros((locs.shape[0], full_cycle))
        terms[np.arange(locs.shape[0]), locs] = 1
        return terms

    @property
    def _columns(self) -> List[str]:
        columns = []
        count = self._supported[self._period][self._freq_str]
        for i in range(count):
            columns.append(
                f"s({self._freq_str}={i + 1}, period={self._period})"
            )
        return columns

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(
        self, index: Union[Sequence[Hashable], pd.Index]
    ) -> pd.DataFrame:
        index = self._index_like(index)
        index = self._check_index_type(index)
        terms = self._get_terms(index)

        return pd.DataFrame(terms, index=index, columns=self._columns)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(
        self,
        steps: int,
        index: Union[Sequence[Hashable], pd.Index],
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        self._check_index_type(fcast_index)
        assert isinstance(fcast_index, (pd.DatetimeIndex, pd.PeriodIndex))
        terms = self._get_terms(fcast_index)
        return pd.DataFrame(terms, index=fcast_index, columns=self._columns)

    @property
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        return self._period, self._freq_str

    def __str__(self) -> str:
        return f"Seasonal(freq={self._freq_str})"


class CalendarTimeTrend(CalendarDeterministicTerm, TimeTrendDeterministicTerm):
    r"""
    Constant and time trend determinstic terms based on calendar time

    Parameters
    ----------
    freq : str
        A string convertible to a pandas frequency.
    constant : bool
        Flag indicating whether a constant should be included.
    order : int
        A non-negative int containing the powers to include (1, 2, ..., order).
    base_period : {str, pd.Timestamp}, default None
        The base period to use when computing the time stamps. This value is
        treated as 1 and so all other time indices are defined as the number
        of periods since or before this time stamp. If not provided, defaults
        to pandas base period for a PeriodIndex.

    See Also
    --------
    DeterministicProcess
    CalendarFourier
    CalendarSeasonality
    TimeTrend

    Notes
    -----
    The time stamp, :math:`\tau_t`, is the number of periods that have elapsed
    since the base_period. :math:`\tau_t` may be fractional.

    Examples
    --------
    Here we simulate irregularly spaced hourly data and construct the calendar
    time trend terms for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarTimeTrend
    >>> cal_trend_gen = CalendarTimeTrend("D", True, order=1)
    >>> cal_trend_gen.in_sample(index)

    Next, we normalize using the first time stamp

    >>> cal_trend_gen = CalendarTimeTrend("D", True, order=1,
    ...                                   base_period=index[0])
    >>> cal_trend_gen.in_sample(index)
    """

    def __init__(
        self,
        freq: str,
        constant: bool = True,
        order: int = 0,
        *,
        base_period: Optional[Union[str, DateLike]] = None,
    ) -> None:
        super().__init__(freq)
        TimeTrendDeterministicTerm.__init__(
            self, constant=constant, order=order
        )
        self._ref_i8 = 0
        if base_period is not None:
            pr = pd.period_range(base_period, periods=1, freq=self._freq)
            self._ref_i8 = pr.asi8[0]
        self._base_period = None if base_period is None else str(base_period)

    @property
    def base_period(self) -> Optional[str]:
        """The base period"""
        return self._base_period

    @classmethod
    def from_string(
        cls,
        freq: str,
        trend: str,
        base_period: Optional[Union[str, DateLike]] = None,
    ) -> "CalendarTimeTrend":
        """
        Create a TimeTrend from a string description.

        Provided for compatibility with common string names.

        Parameters
        ----------
        freq : str
            A string convertible to a pandas frequency.
        trend : {"n", "c", "t", "ct", "ctt"}
            The string representation of the time trend. The terms are:

            * "n": No trend terms
            * "c": A constant only
            * "t": Linear time trend only
            * "ct": A constant and a time trend
            * "ctt": A constant, a time trend and a quadratic time trend
        base_period : {str, pd.Timestamp}, default None
            The base period to use when computing the time stamps. This value
            is treated as 1 and so all other time indices are defined as the
            number of periods since or before this time stamp. If not
            provided, defaults to pandas base period for a PeriodIndex.

        Returns
        -------
        TimeTrend
            The TimeTrend instance.
        """
        constant = trend.startswith("c")
        order = 0
        if "tt" in trend:
            order = 2
        elif "t" in trend:
            order = 1
        return cls(freq, constant, order, base_period=base_period)

    def _terms(
        self, index: Union[pd.DatetimeIndex, pd.PeriodIndex], ratio: np.ndarray
    ) -> pd.DataFrame:
        if isinstance(index, pd.DatetimeIndex):
            index = index.to_period(self._freq)

        index_i8 = index.asi8
        index_i8 = index_i8 - self._ref_i8 + 1
        time = index_i8.astype(np.double) + ratio
        time = time[:, None]
        terms = self._get_terms(time)
        return pd.DataFrame(terms, columns=self._columns, index=index)

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(
        self, index: Union[Sequence[Hashable], pd.Index]
    ) -> pd.DataFrame:
        index = self._index_like(index)
        index = self._check_index_type(index)
        ratio = self._compute_ratio(index)
        return self._terms(index, ratio)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(
        self,
        steps: int,
        index: Union[Sequence[Hashable], pd.Index],
        forecast_index: Optional[Sequence[Hashable]] = None,
    ) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        self._check_index_type(fcast_index)
        assert isinstance(fcast_index, (pd.PeriodIndex, pd.DatetimeIndex))
        ratio = self._compute_ratio(fcast_index)
        return self._terms(fcast_index, ratio)

    @property
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        attr: Tuple[Hashable, ...] = (
            self._constant,
            self._order,
            self._freq.freqstr,
        )
        if self._base_period is not None:
            attr += (self._base_period,)
        return attr

    def __str__(self) -> str:
        value = TimeTrendDeterministicTerm.__str__(self)
        value = "Calendar" + value[:-1] + f", freq={self._freq.freqstr})"
        if self._base_period is not None:
            value = value[:-1] + f"base_period={self._base_period})"
        return value


class DeterministicProcess:
    """
    Container class for deterministic terms.

    Directly supports constants, time trends, and either seasonal dummies or
    fourier terms for a single cycle. Additional deterministic terms beyond
    the set that can be directly initialized through the constructor can be
    added.

    Parameters
    ----------
    index : {Sequence[Hashable], pd.Index}
        The index of the process. Should usually be the "in-sample" index when
        used in forecasting applications.
    period : {float, int}, default None
        The period of the seasonal or fourier components. Must be an int for
        seasonal dummies. If not provided, freq is read from index if
        available.
    constant : bool, default False
        Whether to include a constant.
    order : int, default 0
        The order of the tim trend to include. For example, 2 will include
        both linear and quadratic terms. 0 exclude time trend terms.
    seasonal : bool = False
        Whether to include seasonal dummies
    fourier : int = 0
        The order of the fourier terms to included.
    additional_terms : Sequence[DeterministicTerm]
        A sequence of additional deterministic terms to include in the process.
    drop : bool, default False
        A flag indicating to check for perfect collinearity and to drop any
        linearly dependent terms.

    See Also
    --------
    TimeTrend
    Seasonality
    Fourier
    CalendarTimeTrend
    CalendarSeasonality
    CalendarFourier

    Notes
    -----
    See the notebook `Deterministic Terms in Time Series Models
    <../examples/notebooks/generated/deterministics.html>`__ for an overview.

    Examples
    --------
    >>> from statsmodels.tsa.deterministic import DeterministicProcess
    >>> from pandas import date_range
    >>> index = date_range("2000-1-1", freq="M", periods=240)

    First a determinstic process with a constant and quadratic time trend.

    >>> dp = DeterministicProcess(index, constant=True, order=2)
    >>> dp.in_sample().head(3)
                const  trend  trend_squared
    2000-01-31    1.0    1.0            1.0
    2000-02-29    1.0    2.0            4.0
    2000-03-31    1.0    3.0            9.0

    Seasonal dummies are included by setting seasonal to True.

    >>> dp = DeterministicProcess(index, constant=True, seasonal=True)
    >>> dp.in_sample().iloc[:3,:5]
                const  s(2,12)  s(3,12)  s(4,12)  s(5,12)
    2000-01-31    1.0      0.0      0.0      0.0      0.0
    2000-02-29    1.0      1.0      0.0      0.0      0.0
    2000-03-31    1.0      0.0      1.0      0.0      0.0

    Fourier components can be used to alternatively capture seasonal patterns,

    >>> dp = DeterministicProcess(index, constant=True, fourier=2)
    >>> dp.in_sample().head(3)
                const  sin(1,12)  cos(1,12)  sin(2,12)  cos(2,12)
    2000-01-31    1.0   0.000000   1.000000   0.000000        1.0
    2000-02-29    1.0   0.500000   0.866025   0.866025        0.5
    2000-03-31    1.0   0.866025   0.500000   0.866025       -0.5

    Multiple Seasonalities can be captured using additional terms.

    >>> from statsmodels.tsa.deterministic import Fourier
    >>> index = date_range("2000-1-1", freq="D", periods=5000)
    >>> fourier = Fourier(period=365.25, order=1)
    >>> dp = DeterministicProcess(index, period=3, constant=True,
    ...                           seasonal=True, additional_terms=[fourier])
    >>> dp.in_sample().head(3)
                const  s(2,3)  s(3,3)  sin(1,365.25)  cos(1,365.25)
    2000-01-01    1.0     0.0     0.0       0.000000       1.000000
    2000-01-02    1.0     1.0     0.0       0.017202       0.999852
    2000-01-03    1.0     0.0     1.0       0.034398       0.999408
    """

    def __init__(
        self,
        index: Union[Sequence[Hashable], pd.Index],
        *,
        period: Optional[Union[float, int]] = None,
        constant: bool = False,
        order: int = 0,
        seasonal: bool = False,
        fourier: int = 0,
        additional_terms: Sequence[DeterministicTerm] = (),
        drop: bool = False,
    ):
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        self._index = index
        self._deterministic_terms: List[DeterministicTerm] = []
        self._extendable = False
        self._index_freq = None
        self._validate_index()
        period = float_like(period, "period", optional=True)
        self._constant = constant = bool_like(constant, "constant")
        self._order = required_int_like(order, "order")
        self._seasonal = seasonal = bool_like(seasonal, "seasonal")
        self._fourier = required_int_like(fourier, "fourier")
        additional_terms = tuple(additional_terms)
        self._cached_in_sample = None
        self._drop = bool_like(drop, "drop")
        self._additional_terms = additional_terms
        if constant or order:
            self._deterministic_terms.append(TimeTrend(constant, order))
        if seasonal and fourier:
            raise ValueError(
                """seasonal and fourier can be initialized through the \
constructor since these will be necessarily perfectly collinear. Instead, \
you can pass additional components using the additional_terms input."""
            )
        if (seasonal or fourier) and period is None:
            if period is None:
                self._period = period = freq_to_period(self._index_freq)
        if seasonal:
            period = required_int_like(period, "period")
            self._deterministic_terms.append(Seasonality(period))
        elif fourier:
            period = float_like(period, "period")
            assert period is not None
            self._deterministic_terms.append(Fourier(period, order=fourier))
        for term in additional_terms:
            if not isinstance(term, DeterministicTerm):
                raise TypeError(
                    "All additional terms must be instances of subsclasses "
                    "of DeterministicTerm"
                )
            if term not in self._deterministic_terms:
                self._deterministic_terms.append(term)
            else:
                raise ValueError(
                    "One or more terms in additional_terms has been added "
                    "through the parameters of the constructor. Terms must "
                    "be unique."
                )
        self._period = period
        self._retain_cols: Optional[List[Hashable]] = None

    @property
    def index(self) -> pd.Index:
        """The index of the process"""
        return self._index

    @property
    def terms(self) -> List[DeterministicTerm]:
        """The deterministic terms included in the process"""
        return self._deterministic_terms

    def _adjust_dummies(self, terms: List[pd.DataFrame]) -> List[pd.DataFrame]:
        has_const: Optional[bool] = None
        for dterm in self._deterministic_terms:
            if isinstance(dterm, (TimeTrend, CalendarTimeTrend)):
                has_const = has_const or dterm.constant
        if has_const is None:
            has_const = False
            for term in terms:
                const_col = (term == term.iloc[0]).all() & (term.iloc[0] != 0)
                has_const = has_const or const_col.any()
        drop_first = has_const
        for i, dterm in enumerate(self._deterministic_terms):
            is_dummy = dterm.is_dummy
            if is_dummy and drop_first:
                # drop first
                terms[i] = terms[i].iloc[:, 1:]
            drop_first = drop_first or is_dummy
        return terms

    def _remove_zeros_ones(self, terms: pd.DataFrame) -> pd.DataFrame:
        all_zero = np.all(terms == 0, axis=0)
        if np.any(all_zero):
            terms = terms.loc[:, ~all_zero]
        is_constant = terms.max(axis=0) == terms.min(axis=0)
        if np.sum(is_constant) > 1:
            # Retain first
            const_locs = np.where(is_constant)[0]
            is_constant[const_locs[:1]] = False
            terms = terms.loc[:, ~is_constant]
        return terms

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(self) -> pd.DataFrame:
        if self._cached_in_sample is not None:
            return self._cached_in_sample
        index = self._index
        if not self._deterministic_terms:
            return pd.DataFrame(np.empty((index.shape[0], 0)), index=index)
        raw_terms = []
        for term in self._deterministic_terms:
            raw_terms.append(term.in_sample(index))

        raw_terms = self._adjust_dummies(raw_terms)
        terms: pd.DataFrame = pd.concat(raw_terms, axis=1)
        terms = self._remove_zeros_ones(terms)
        if self._drop:
            terms_arr = to_numpy(terms)
            res = qr(terms_arr, mode="r", pivoting=True)
            r = res[0]
            p = res[-1]
            abs_diag = np.abs(np.diag(r))
            tol = abs_diag[0] * terms_arr.shape[1] * np.finfo(float).eps
            rank = int(np.sum(abs_diag > tol))
            rpx = r.T @ terms_arr
            keep = [0]
            last_rank = 1
            # Find the left-most columns that produce full rank
            for i in range(1, terms_arr.shape[1]):
                curr_rank = np.linalg.matrix_rank(rpx[: i + 1, : i + 1])
                if curr_rank > last_rank:
                    keep.append(i)
                    last_rank = curr_rank
                if curr_rank == rank:
                    break
            if len(keep) == rank:
                terms = terms.iloc[:, keep]
            else:
                terms = terms.iloc[:, np.sort(p[:rank])]
        self._retain_cols = terms.columns
        self._cached_in_sample = terms
        return terms

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(
        self,
        steps: int,
        forecast_index: Optional[Union[Sequence[Hashable], pd.Index]] = None,
    ) -> pd.DataFrame:
        steps = required_int_like(steps, "steps")
        if self._drop and self._retain_cols is None:
            self.in_sample()
        index = self._index
        if not self._deterministic_terms:
            return pd.DataFrame(np.empty((index.shape[0], 0)), index=index)
        raw_terms = []
        for term in self._deterministic_terms:
            raw_terms.append(term.out_of_sample(steps, index, forecast_index))
        terms: pd.DataFrame = pd.concat(raw_terms, axis=1)
        assert self._retain_cols is not None
        if terms.shape[1] != len(self._retain_cols):
            terms = terms[self._retain_cols]
        return terms

    def _extend_time_index(
        self,
        stop: pd.Timestamp,
    ) -> Union[pd.DatetimeIndex, pd.PeriodIndex]:
        index = self._index
        if isinstance(index, pd.PeriodIndex):
            return pd.period_range(index[0], end=stop, freq=index.freq)
        return pd.date_range(start=index[0], end=stop, freq=self._index_freq)

    def _range_from_range_index(self, start: int, stop: int) -> pd.DataFrame:
        index = self._index
        is_int64_index = is_int_index(index)
        assert isinstance(index, pd.RangeIndex) or is_int64_index
        if start < index[0]:
            raise ValueError(START_BEFORE_INDEX_ERR)
        if isinstance(index, pd.RangeIndex):
            idx_step = index.step
        else:
            idx_step = np.diff(index).max() if len(index) > 1 else 1
        if idx_step != 1 and ((start - index[0]) % idx_step) != 0:
            raise ValueError(
                f"The step of the index is not 1 (actual step={idx_step})."
                " start must be in the sequence that would have been "
                "generated by the index."
            )
        if is_int64_index:
            new_idx = pd.Index(np.arange(start, stop))
        else:
            new_idx = pd.RangeIndex(start, stop, step=idx_step)
        if new_idx[-1] <= self._index[-1]:
            # In-sample only
            in_sample = self.in_sample()
            in_sample = in_sample.loc[new_idx]
            return in_sample
        elif new_idx[0] > self._index[-1]:
            # Out of-sample only
            next_value = index[-1] + idx_step
            if new_idx[0] != next_value:
                tmp = pd.RangeIndex(next_value, stop, step=idx_step)
                oos = self.out_of_sample(tmp.shape[0], forecast_index=tmp)
                return oos.loc[new_idx]
            return self.out_of_sample(new_idx.shape[0], forecast_index=new_idx)
        # Using some from each in and out of sample
        in_sample_loc = new_idx <= self._index[-1]
        in_sample_idx = new_idx[in_sample_loc]
        out_of_sample_idx = new_idx[~in_sample_loc]
        in_sample_exog = self.in_sample().loc[in_sample_idx]
        oos_exog = self.out_of_sample(
            steps=out_of_sample_idx.shape[0], forecast_index=out_of_sample_idx
        )
        return pd.concat([in_sample_exog, oos_exog], axis=0)

    def _range_from_time_index(
        self, start: pd.Timestamp, stop: pd.Timestamp
    ) -> pd.DataFrame:
        index = self._index
        if isinstance(self._index, pd.PeriodIndex):
            if isinstance(start, pd.Timestamp):
                start = start.to_period(freq=self._index_freq)
            if isinstance(stop, pd.Timestamp):
                stop = stop.to_period(freq=self._index_freq)
        if start < index[0]:
            raise ValueError(START_BEFORE_INDEX_ERR)
        if stop <= self._index[-1]:
            return self.in_sample().loc[start:stop]
        new_idx = self._extend_time_index(stop)
        oos_idx = new_idx[new_idx > index[-1]]
        oos = self.out_of_sample(oos_idx.shape[0], oos_idx)
        if start >= oos_idx[0]:
            return oos.loc[start:stop]
        both = pd.concat([self.in_sample(), oos], axis=0)
        return both.loc[start:stop]

    def _int_to_timestamp(self, value: int, name: str) -> pd.Timestamp:
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        if value < self._index.shape[0]:
            return self._index[value]
        add_periods = value - (self._index.shape[0] - 1) + 1
        index = self._index
        if isinstance(self._index, pd.PeriodIndex):
            pr = pd.period_range(
                index[-1], freq=self._index_freq, periods=add_periods
            )
            return pr[-1].to_timestamp()
        dr = pd.date_range(
            index[-1], freq=self._index_freq, periods=add_periods
        )
        return dr[-1]

    def range(
        self,
        start: Union[IntLike, DateLike, str],
        stop: Union[IntLike, DateLike, str],
    ) -> pd.DataFrame:
        """
        Deterministic terms spanning a range of observations

        Parameters
        ----------
        start : {int, str, dt.datetime, pd.Timestamp, np.datetime64}
            The first observation.
        stop : {int, str, dt.datetime, pd.Timestamp, np.datetime64}
            The final observation. Inclusive to match most prediction
            function in statsmodels.

        Returns
        -------
        DataFrame
            A data frame of deterministic terms
        """
        if not self._extendable:
            raise TypeError(
                """The index in the deterministic process does not \
support extension. Only PeriodIndex, DatetimeIndex with a frequency, \
RangeIndex, and integral Indexes that start at 0 and have only unit \
differences can be extended when producing out-of-sample forecasts.
"""
            )
        if type(self._index) in (pd.RangeIndex,) or is_int_index(self._index):
            start = required_int_like(start, "start")
            stop = required_int_like(stop, "stop")
            # Add 1 to ensure that the end point is inclusive
            stop += 1
            return self._range_from_range_index(start, stop)
        if isinstance(start, (int, np.integer)):
            start = self._int_to_timestamp(start, "start")
        else:
            start = pd.Timestamp(start)
        if isinstance(stop, (int, np.integer)):
            stop = self._int_to_timestamp(stop, "stop")
        else:
            stop = pd.Timestamp(stop)
        return self._range_from_time_index(start, stop)

    def _validate_index(self) -> None:
        if isinstance(self._index, pd.PeriodIndex):
            self._index_freq = self._index.freq
            self._extendable = True
        elif isinstance(self._index, pd.DatetimeIndex):
            self._index_freq = self._index.freq or self._index.inferred_freq
            self._extendable = self._index_freq is not None
        elif isinstance(self._index, pd.RangeIndex):
            self._extendable = True
        elif is_int_index(self._index):
            self._extendable = self._index[0] == 0 and np.all(
                np.diff(self._index) == 1
            )

    def apply(self, index):
        """
        Create an identical determinstic process with a different index

        Parameters
        ----------
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.

        Returns
        -------
        DeterministicProcess
            The deterministic process applied to a different index
        """
        return DeterministicProcess(
            index,
            period=self._period,
            constant=self._constant,
            order=self._order,
            seasonal=self._seasonal,
            fourier=self._fourier,
            additional_terms=self._additional_terms,
            drop=self._drop,
        )
