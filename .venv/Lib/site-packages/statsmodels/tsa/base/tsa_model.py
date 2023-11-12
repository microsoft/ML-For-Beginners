from __future__ import annotations

from statsmodels.compat.pandas import (
    is_float_index,
    is_int_index,
    is_numeric_dtype,
)

import numbers
import warnings

import numpy as np
from pandas import (
    DatetimeIndex,
    Index,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
)
from pandas.tseries.frequencies import to_offset

from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning

_tsa_doc = """
    %(model)s

    Parameters
    ----------
    %(params)s
    dates : array_like, optional
        An array-like object of datetime objects. If a pandas object is given
        for endog or exog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    %(extra_params)s
    %(extra_sections)s"""

_model_doc = "Timeseries model base class"

_generic_params = base._model_params_doc
_missing_param_doc = base._missing_param_doc


def get_index_loc(key, index):
    """
    Get the location of a specific key in an index

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or a location if the underlying index is a RangeIndex
        or an Index with an integer dtype.
    index : pd.Index
        The index to search.

    Returns
    -------
    loc : int
        The location of the key
    index : pd.Index
        The index including the key; this is a copy of the original index
        unless the index had to be expanded to accommodate `key`.
    index_was_expanded : bool
        Whether or not the index was expanded to accommodate `key`.

    Notes
    -----
    If `key` is past the end of of the given index, and the index is either
    an Index with an integral dtype or a date index, this function extends
    the index up to and including key, and then returns the location in the
    new index.
    """
    base_index = index

    index = base_index
    date_index = isinstance(base_index, (PeriodIndex, DatetimeIndex))
    int_index = is_int_index(base_index)
    range_index = isinstance(base_index, RangeIndex)
    index_class = type(base_index)
    nobs = len(index)

    # Special handling for RangeIndex
    if range_index and isinstance(key, (int, np.integer)):
        # Negative indices (that lie in the Index)
        if key < 0 and -key <= nobs:
            key = nobs + key
        # Out-of-sample (note that we include key itself in the new index)
        elif key > nobs - 1:
            # See gh5835. Remove the except after pandas 0.25 required.
            try:
                base_index_start = base_index.start
                base_index_step = base_index.step
            except AttributeError:
                base_index_start = base_index._start
                base_index_step = base_index._step
            stop = base_index_start + (key + 1) * base_index_step
            index = RangeIndex(
                start=base_index_start, stop=stop, step=base_index_step
            )

    # Special handling for NumericIndex
    if (
        not range_index
        and int_index
        and not date_index
        and isinstance(key, (int, np.integer))
    ):
        # Negative indices (that lie in the Index)
        if key < 0 and -key <= nobs:
            key = nobs + key
        # Out-of-sample (note that we include key itself in the new index)
        elif key > base_index[-1]:
            index = Index(np.arange(base_index[0], int(key + 1)))

    # Special handling for date indexes
    if date_index:
        # Use index type to choose creation function
        if index_class is DatetimeIndex:
            index_fn = date_range
        else:
            index_fn = period_range
        # Integer key (i.e. already given a location)
        if isinstance(key, (int, np.integer)):
            # Negative indices (that lie in the Index)
            if key < 0 and -key < nobs:
                key = index[nobs + key]
            # Out-of-sample (note that we include key itself in the new
            # index)
            elif key > len(base_index) - 1:
                index = index_fn(
                    start=base_index[0],
                    periods=int(key + 1),
                    freq=base_index.freq,
                )
                key = index[-1]
            else:
                key = index[key]
        # Other key types (i.e. string date or some datetime-like object)
        else:
            # Convert the key to the appropriate date-like object
            if index_class is PeriodIndex:
                date_key = Period(key, freq=base_index.freq)
            else:
                date_key = Timestamp(key)

            # Out-of-sample
            if date_key > base_index[-1]:
                # First create an index that may not always include `key`
                index = index_fn(
                    start=base_index[0], end=date_key, freq=base_index.freq
                )

                # Now make sure we include `key`
                if not index[-1] == date_key:
                    index = index_fn(
                        start=base_index[0],
                        periods=len(index) + 1,
                        freq=base_index.freq,
                    )

                # To avoid possible inconsistencies with `get_loc` below,
                # set the key directly equal to the last index location
                key = index[-1]

    # Get the location
    if date_index:
        # (note that get_loc will throw a KeyError if key is invalid)
        loc = index.get_loc(key)
    elif int_index or range_index:
        # For NumericIndex and RangeIndex, key is assumed to be the location
        # and not an index value (this assumption is required to support
        # RangeIndex)
        try:
            index[key]
        # We want to raise a KeyError in this case, to keep the exception
        # consistent across index types.
        # - Attempting to index with an out-of-bound location (e.g.
        #   index[10] on an index of length 9) will raise an IndexError
        #   (as of Pandas 0.22)
        # - Attemtping to index with a type that cannot be cast to integer
        #   (e.g. a non-numeric string) will raise a ValueError if the
        #   index is RangeIndex (otherwise will raise an IndexError)
        #   (as of Pandas 0.22)
        except (IndexError, ValueError) as e:
            raise KeyError(str(e))
        loc = key
    else:
        loc = index.get_loc(key)

    # Check if we now have a modified index
    index_was_expanded = index is not base_index

    # Return the index through the end of the loc / slice
    if isinstance(loc, slice):
        end = loc.stop - 1
    else:
        end = loc

    return loc, index[: end + 1], index_was_expanded


def get_index_label_loc(key, index, row_labels):
    """
    Get the location of a specific key in an index or model row labels

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or is only being used as row labels, or a location if
        the underlying index is a RangeIndex or a NumericIndex.
    index : pd.Index
        The index to search.
    row_labels : pd.Index
        Row labels to search if key not found in index

    Returns
    -------
    loc : int
        The location of the key
    index : pd.Index
        The index including the key; this is a copy of the original index
        unless the index had to be expanded to accommodate `key`.
    index_was_expanded : bool
        Whether or not the index was expanded to accommodate `key`.

    Notes
    -----
    This function expands on `get_index_loc` by first trying the given
    base index (or the model's index if the base index was not given) and
    then falling back to try again with the model row labels as the base
    index.
    """
    try:
        loc, index, index_was_expanded = get_index_loc(key, index)
    except KeyError as e:
        try:
            if not isinstance(key, (int, np.integer)):
                loc = row_labels.get_loc(key)
            else:
                raise
            # Require scalar
            # Pandas may return a slice if there are multiple matching
            # locations that are monotonic increasing (otherwise it may
            # return an array of integer locations, see below).
            if isinstance(loc, slice):
                loc = loc.start
            if isinstance(loc, np.ndarray):
                # Pandas may return a mask (boolean array), for e.g.:
                # pd.Index(list('abcb')).get_loc('b')
                if loc.dtype == bool:
                    # Return the first True value
                    # (we know there is at least one True value if we're
                    # here because otherwise the get_loc call would have
                    # raised an exception)
                    loc = np.argmax(loc)
                # Finally, Pandas may return an integer array of
                # locations that match the given value, for e.g.
                # pd.DatetimeIndex(['2001-02', '2001-01']).get_loc('2001')
                # (this appears to be slightly undocumented behavior, since
                # only int, slice, and mask are mentioned in docs for
                # pandas.Index.get_loc as of 0.23.4)
                else:
                    loc = loc[0]
            if not isinstance(loc, numbers.Integral):
                raise

            index = row_labels[: loc + 1]
            index_was_expanded = False
        except:
            raise e
    return loc, index, index_was_expanded


def get_prediction_index(
    start,
    end,
    nobs,
    base_index,
    index=None,
    silent=False,
    index_none=False,
    index_generated=None,
    data=None,
) -> tuple[int, int, int, Index | None]:
    """
    Get the location of a specific key in an index or model row labels

    Parameters
    ----------
    start : label
        The key at which to start prediction. Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels.
    end : label
        The key at which to end prediction (note that this key will be
        *included* in prediction). Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels.
    nobs : int
    base_index : pd.Index

    index : pd.Index, optional
        Optionally an index to associate the predicted results to. If None,
        an attempt is made to create an index for the predicted results
        from the model's index or model's row labels.
    silent : bool, optional
        Argument to silence warnings.

    Returns
    -------
    start : int
        The index / observation location at which to begin prediction.
    end : int
        The index / observation location at which to end in-sample
        prediction. The maximum value for this is nobs-1.
    out_of_sample : int
        The number of observations to forecast after the end of the sample.
    prediction_index : pd.Index or None
        The index associated with the prediction results. This index covers
        the range [start, end + out_of_sample]. If the model has no given
        index and no given row labels (i.e. endog/exog is not Pandas), then
        this will be None.

    Notes
    -----
    The arguments `start` and `end` behave differently, depending on if
    they are integer or not. If either is an integer, then it is assumed
    to refer to a *location* in the index, not to an index value. On the
    other hand, if it is a date string or some other type of object, then
    it is assumed to refer to an index *value*. In all cases, the returned
    `start` and `end` values refer to index *locations* (so in the former
    case, the given location is validated and returned whereas in the
    latter case a location is found that corresponds to the given index
    value).

    This difference in behavior is necessary to support `RangeIndex`. This
    is because integers for a RangeIndex could refer either to index values
    or to index locations in an ambiguous way (while for `NumericIndex`,
    since we have required them to be full indexes, there is no ambiguity).
    """

    # Convert index keys (start, end) to index locations and get associated
    # indexes.
    try:
        start, _, start_oos = get_index_label_loc(
            start, base_index, data.row_labels
        )
    except KeyError:
        raise KeyError(
            "The `start` argument could not be matched to a"
            " location related to the index of the data."
        )
    if end is None:
        end = max(start, len(base_index) - 1)
    try:
        end, end_index, end_oos = get_index_label_loc(
            end, base_index, data.row_labels
        )
    except KeyError:
        raise KeyError(
            "The `end` argument could not be matched to a"
            " location related to the index of the data."
        )

    # Handle slices (if the given index keys cover more than one date)
    if isinstance(start, slice):
        start = start.start
    if isinstance(end, slice):
        end = end.stop - 1

    # Get the actual index for the prediction
    prediction_index = end_index[start:]

    # Validate prediction options
    if end < start:
        raise ValueError("Prediction must have `end` after `start`.")

    # Handle custom prediction index
    # First, if we were given an index, check that it's the right size and
    # use it if so
    if index is not None:
        if not len(prediction_index) == len(index):
            raise ValueError(
                "Invalid `index` provided in prediction."
                " Must have length consistent with `start`"
                " and `end` arguments."
            )
        # But if we weren't given Pandas input, this index will not be
        # used because the data will not be wrapped; in that case, issue
        # a warning
        if not isinstance(data, PandasData) and not silent:
            warnings.warn(
                "Because the model data (`endog`, `exog`) were"
                " not given as Pandas objects, the prediction"
                " output will be Numpy arrays, and the given"
                " `index` argument will only be used"
                " internally.",
                ValueWarning,
                stacklevel=2,
            )
        prediction_index = Index(index)
    # Now, if we *do not* have a supported index, but we were given some
    # kind of index...
    elif index_generated and not index_none:
        # If we are in sample, and have row labels, use them
        if data.row_labels is not None and not (start_oos or end_oos):
            prediction_index = data.row_labels[start : end + 1]
        # Otherwise, warn the user that they will get an NumericIndex
        else:
            if not silent:
                warnings.warn(
                    "No supported index is available."
                    " Prediction results will be given with"
                    " an integer index beginning at `start`.",
                    ValueWarning,
                    stacklevel=2,
                )
            warnings.warn(
                "No supported index is available. In the next"
                " version, calling this method in a model"
                " without a supported index will result in an"
                " exception.",
                FutureWarning,
                stacklevel=2,
            )
    elif index_none:
        prediction_index = None

    # For backwards compatibility, set `predict_*` values
    if prediction_index is not None:
        data.predict_start = prediction_index[0]
        data.predict_end = prediction_index[-1]
        data.predict_dates = prediction_index
    else:
        data.predict_start = None
        data.predict_end = None
        data.predict_dates = None

    # Compute out-of-sample observations
    out_of_sample = max(end - (nobs - 1), 0)
    end -= out_of_sample

    return start, end, out_of_sample, prediction_index


class TimeSeriesModel(base.LikelihoodModel):
    __doc__ = _tsa_doc % {
        "model": _model_doc,
        "params": _generic_params,
        "extra_params": _missing_param_doc,
        "extra_sections": "",
    }

    def __init__(
        self, endog, exog=None, dates=None, freq=None, missing="none", **kwargs
    ):
        super().__init__(endog, exog, missing=missing, **kwargs)

        # Date handling in indexes
        self._init_dates(dates, freq)

    def _init_dates(self, dates=None, freq=None):
        """
        Initialize dates

        Parameters
        ----------
        dates : array_like, optional
            An array like object containing dates.
        freq : str, tuple, datetime.timedelta, DateOffset or None, optional
            A frequency specification for either `dates` or the row labels from
            the endog / exog data.

        Notes
        -----
        Creates `self._index` and related attributes. `self._index` is always
        a Pandas index, and it is always NumericIndex, DatetimeIndex, or
        PeriodIndex.

        If Pandas objects, endog / exog may have any type of index. If it is
        an NumericIndex with values 0, 1, ..., nobs-1 or if it is (coerceable to)
        a DatetimeIndex or PeriodIndex *with an associated frequency*, then it
        is called a "supported" index. Otherwise it is called an "unsupported"
        index.

        Supported indexes are standardized (i.e. a list of date strings is
        converted to a DatetimeIndex) and the result is put in `self._index`.

        Unsupported indexes are ignored, and a supported NumericIndex is
        generated and put in `self._index`. Warnings are issued in this case
        to alert the user if the returned index from some operation (e.g.
        forecasting) is different from the original data's index. However,
        whenever possible (e.g. purely in-sample prediction), the original
        index is returned.

        The benefit of supported indexes is that they allow *forecasting*, i.e.
        it is possible to extend them in a reasonable way. Thus every model
        must have an underlying supported index, even if it is just a generated
        NumericIndex.
        """

        # Get our index from `dates` if available, otherwise from whatever
        # Pandas index we might have retrieved from endog, exog
        if dates is not None:
            index = dates
        else:
            index = self.data.row_labels

        # Sanity check that we do not have a `freq` without an index
        if index is None and freq is not None:
            raise ValueError("Frequency provided without associated index.")

        # If an index is available, see if it is a date-based index or if it
        # can be coerced to one. (If it cannot we'll fall back, below, to an
        # internal, 0, 1, ... nobs-1 integer index for modeling purposes)
        inferred_freq = False
        if index is not None:
            # Try to coerce to date-based index
            if not isinstance(index, (DatetimeIndex, PeriodIndex)):
                try:
                    # Only try to coerce non-numeric index types (string,
                    # list of date-times, etc.)
                    # Note that np.asarray(Float64Index([...])) yields an
                    # object dtype array in earlier versions of Pandas (and so
                    # will not have is_numeric_dtype == True), so explicitly
                    # check for it here. But note also that in very early
                    # Pandas (~0.12), Float64Index does not exist (and so the
                    # statsmodels compat makes it an empty tuple, so in that
                    # case also check if the first element is a float.
                    _index = np.asarray(index)
                    if (
                        is_numeric_dtype(_index)
                        or is_float_index(index)
                        or (isinstance(_index[0], float))
                    ):
                        raise ValueError("Numeric index given")
                    # If a non-index Pandas series was given, only keep its
                    # values (because we must have a pd.Index type, below, and
                    # pd.to_datetime will return a Series when passed
                    # non-list-like objects)
                    if isinstance(index, Series):
                        index = index.values
                    # All coercion is done via pd.to_datetime
                    # Note: date coercion via pd.to_datetime does not handle
                    # string versions of PeriodIndex objects most of the time.
                    _index = to_datetime(index)
                    # Older versions of Pandas can sometimes fail here and
                    # return a numpy array - check to make sure it's an index
                    if not isinstance(_index, Index):
                        raise ValueError("Could not coerce to date index")
                    index = _index
                except:
                    # Only want to actually raise an exception if `dates` was
                    # provided but cannot be coerced. If we got the index from
                    # the row_labels, we'll just ignore it and use the integer
                    # index below
                    if dates is not None:
                        raise ValueError(
                            "Non-date index index provided to"
                            " `dates` argument."
                        )
            # Now, if we were given, or coerced, a date-based index, make sure
            # it has an associated frequency
            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                # If no frequency, try to get an inferred frequency
                if freq is None and index.freq is None:
                    freq = index.inferred_freq
                    # If we got an inferred frequncy, alert the user
                    if freq is not None:
                        inferred_freq = True
                        if freq is not None:
                            warnings.warn(
                                "No frequency information was"
                                " provided, so inferred frequency %s"
                                " will be used." % freq,
                                ValueWarning,
                                stacklevel = 2,
                            )

                # Convert the passed freq to a pandas offset object
                if freq is not None:
                    freq = to_offset(freq)

                # Now, if no frequency information is available from the index
                # itself or from the `freq` argument, raise an exception
                if freq is None and index.freq is None:
                    # But again, only want to raise the exception if `dates`
                    # was provided.
                    if dates is not None:
                        raise ValueError(
                            "No frequency information was"
                            " provided with date index and no"
                            " frequency could be inferred."
                        )
                # However, if the index itself has no frequency information but
                # the `freq` argument is available (or was inferred), construct
                # a new index with an associated frequency
                elif freq is not None and index.freq is None:
                    resampled_index = date_range(
                        start=index[0], end=index[-1], freq=freq
                    )
                    if not inferred_freq and not resampled_index.equals(index):
                        raise ValueError(
                            "The given frequency argument could"
                            " not be matched to the given index."
                        )
                    index = resampled_index
                # Finally, if the index itself has a frequency and there was
                # also a given frequency, raise an exception if they are not
                # equal
                elif (
                    freq is not None
                    and not inferred_freq
                    and not (index.freq == freq)
                ):
                    raise ValueError(
                        "The given frequency argument is"
                        " incompatible with the given index."
                    )
            # Finally, raise an exception if we could not coerce to date-based
            # but we were given a frequency argument
            elif freq is not None:
                raise ValueError(
                    "Given index could not be coerced to dates"
                    " but `freq` argument was provided."
                )

        # Get attributes of the index
        has_index = index is not None
        date_index = isinstance(index, (DatetimeIndex, PeriodIndex))
        period_index = isinstance(index, PeriodIndex)
        int_index = is_int_index(index)
        range_index = isinstance(index, RangeIndex)
        has_freq = index.freq is not None if date_index else None
        increment = Index(range(self.endog.shape[0]))
        is_increment = index.equals(increment) if int_index else None
        if date_index:
            try:
                is_monotonic = index.is_monotonic_increasing
            except AttributeError:
                # Remove after pandas 1.5 is minimum
                is_monotonic = index.is_monotonic
        else:
            is_monotonic = None

        # Issue warnings for unsupported indexes
        if has_index and not (date_index or range_index or is_increment):
            warnings.warn(
                "An unsupported index was provided and will be"
                " ignored when e.g. forecasting.",
                ValueWarning,
                stacklevel=2,
            )
        if date_index and not has_freq:
            warnings.warn(
                "A date index has been provided, but it has no"
                " associated frequency information and so will be"
                " ignored when e.g. forecasting.",
                ValueWarning,
                stacklevel=2,
            )
        if date_index and not is_monotonic:
            warnings.warn(
                "A date index has been provided, but it is not"
                " monotonic and so will be ignored when e.g."
                " forecasting.",
                ValueWarning,
                stacklevel=2,
            )

        # Construct the internal index
        index_generated = False
        valid_index = (
            (date_index and has_freq and is_monotonic)
            or (int_index and is_increment)
            or range_index
        )

        if valid_index:
            _index = index
        else:
            _index = increment
            index_generated = True
        self._index = _index
        self._index_generated = index_generated
        self._index_none = index is None
        self._index_int64 = int_index and not range_index and not date_index
        self._index_dates = date_index and not index_generated
        self._index_freq = self._index.freq if self._index_dates else None
        self._index_inferred_freq = inferred_freq

        # For backwards compatibility, set data.dates, data.freq
        self.data.dates = self._index if self._index_dates else None
        self.data.freq = self._index.freqstr if self._index_dates else None

    def _get_index_loc(self, key, base_index=None):
        """
        Get the location of a specific key in an index

        Parameters
        ----------
        key : label
            The key for which to find the location if the underlying index is
            a DateIndex or a location if the underlying index is a RangeIndex
            or an NumericIndex.
        base_index : pd.Index, optional
            Optionally the base index to search. If None, the model's index is
            searched.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accommodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accommodate `key`.

        Notes
        -----
        If `key` is past the end of of the given index, and the index is either
        an NumericIndex or a date index, this function extends the index up to
        and including key, and then returns the location in the new index.
        """

        if base_index is None:
            base_index = self._index
        return get_index_loc(key, base_index)

    def _get_index_label_loc(self, key, base_index=None):
        """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        key : label
            The key for which to find the location if the underlying index is
            a DateIndex or is only being used as row labels, or a location if
            the underlying index is a RangeIndex or an NumericIndex.
        base_index : pd.Index, optional
            Optionally the base index to search. If None, the model's index is
            searched.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accommodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accommodate `key`.

        Notes
        -----
        This method expands on `_get_index_loc` by first trying the given
        base index (or the model's index if the base index was not given) and
        then falling back to try again with the model row labels as the base
        index.
        """
        if base_index is None:
            base_index = self._index
        return get_index_label_loc(key, base_index, self.data.row_labels)

    def _get_prediction_index(self, start, end, index=None, silent=False) -> tuple[int, int, int, Index | None]:
        """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        start : label
            The key at which to start prediction. Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        end : label
            The key at which to end prediction (note that this key will be
            *included* in prediction). Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.
        silent : bool, optional
            Argument to silence warnings.

        Returns
        -------
        start : int
            The index / observation location at which to begin prediction.
        end : int
            The index / observation location at which to end in-sample
            prediction. The maximum value for this is nobs-1.
        out_of_sample : int
            The number of observations to forecast after the end of the sample.
        prediction_index : pd.Index or None
            The index associated with the prediction results. This index covers
            the range [start, end + out_of_sample]. If the model has no given
            index and no given row labels (i.e. endog/exog is not Pandas), then
            this will be None.

        Notes
        -----
        The arguments `start` and `end` behave differently, depending on if
        they are integer or not. If either is an integer, then it is assumed
        to refer to a *location* in the index, not to an index value. On the
        other hand, if it is a date string or some other type of object, then
        it is assumed to refer to an index *value*. In all cases, the returned
        `start` and `end` values refer to index *locations* (so in the former
        case, the given location is validated and returned whereas in the
        latter case a location is found that corresponds to the given index
        value).

        This difference in behavior is necessary to support `RangeIndex`. This
        is because integers for a RangeIndex could refer either to index values
        or to index locations in an ambiguous way (while for `NumericIndex`,
        since we have required them to be full indexes, there is no ambiguity).
        """
        nobs = len(self.endog)
        return get_prediction_index(
            start,
            end,
            nobs,
            base_index=self._index,
            index=index,
            silent=silent,
            index_none=self._index_none,
            index_generated=self._index_generated,
            data=self.data,
        )

    def _get_exog_names(self):
        return self.data.xnames

    def _set_exog_names(self, vals):
        if not isinstance(vals, list):
            vals = [vals]
        self.data.xnames = vals

    # TODO: This is an antipattern, fix/remove with VAR
    # overwrite with writable property for (V)AR models
    exog_names = property(
        _get_exog_names,
        _set_exog_names,
        None,
        "The names of the exogenous variables.",
    )


class TimeSeriesModelResults(base.LikelihoodModelResults):
    def __init__(self, model, params, normalized_cov_params, scale=1.0):
        self.data = model.data
        super().__init__(model, params, normalized_cov_params, scale)


class TimeSeriesResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        base.LikelihoodResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {"predict": "dates"}
    _wrap_methods = wrap.union_dicts(
        base.LikelihoodResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(
    TimeSeriesResultsWrapper, TimeSeriesModelResults  # noqa:E305
)
