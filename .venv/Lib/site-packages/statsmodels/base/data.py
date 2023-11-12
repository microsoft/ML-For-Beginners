"""
Base tools for handling various kinds of data structures, attaching metadata to
results, and doing data cleaning
"""
from __future__ import annotations

from statsmodels.compat.python import lmap

from functools import reduce

import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex

import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError


def _asarray_2dcolumns(x):
    if np.asarray(x).ndim > 1 and np.asarray(x).squeeze().ndim == 1:
        return


def _asarray_2d_null_rows(x):
    """
    Makes sure input is an array and is 2d. Makes sure output is 2d. True
    indicates a null in the rows of 2d x.
    """
    #Have to have the asarrays because isnull does not account for array_like
    #input
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return np.any(isnull(x), axis=1)[:, None]


def _nan_rows(*arrs):
    """
    Returns a boolean array which is True where any of the rows in any
    of the _2d_ arrays in arrs are NaNs. Inputs can be any mixture of Series,
    DataFrames or array_like.
    """
    if len(arrs) == 1:
        arrs += ([[False]],)

    def _nan_row_maybe_two_inputs(x, y):
        # check for dtype bc dataframe has dtypes
        x_is_boolean_array = hasattr(x, 'dtype') and x.dtype == bool and x
        return np.logical_or(_asarray_2d_null_rows(x),
                             (x_is_boolean_array | _asarray_2d_null_rows(y)))
    return reduce(_nan_row_maybe_two_inputs, arrs).squeeze()


class ModelData:
    """
    Class responsible for handling input data and extracting metadata into the
    appropriate form
    """
    _param_names = None
    _cov_names = None

    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                 **kwargs):
        if data_util._is_recarray(endog) or data_util._is_recarray(exog):
            from statsmodels.tools.sm_exceptions import recarray_exception
            raise NotImplementedError(recarray_exception)
        if 'design_info' in kwargs:
            self.design_info = kwargs.pop('design_info')
        if 'formula' in kwargs:
            self.formula = kwargs.pop('formula')
        if missing != 'none':
            arrays, nan_idx = self.handle_missing(endog, exog, missing,
                                                  **kwargs)
            self.missing_row_idx = nan_idx
            self.__dict__.update(arrays)  # attach all the data arrays
            self.orig_endog = self.endog
            self.orig_exog = self.exog
            self.endog, self.exog = self._convert_endog_exog(self.endog,
                                                             self.exog)
        else:
            self.__dict__.update(kwargs)  # attach the extra arrays anyway
            self.orig_endog = endog
            self.orig_exog = exog
            self.endog, self.exog = self._convert_endog_exog(endog, exog)

        self.const_idx = None
        self.k_constant = 0
        self._handle_constant(hasconst)
        self._check_integrity()
        self._cache = {}

    def __getstate__(self):
        from copy import copy
        d = copy(self.__dict__)
        if "design_info" in d:
            del d["design_info"]
            d["restore_design_info"] = True
        return d

    def __setstate__(self, d):
        if "restore_design_info" in d:
            # NOTE: there may be a more performant way to do this
            from patsy import dmatrices, PatsyError
            exc = []
            try:
                data = d['frame']
            except KeyError:
                data = d['orig_endog'].join(d['orig_exog'])

            for depth in [2, 3, 1, 0, 4]:  # sequence is a guess where to likely find it
                try:
                    _, design = dmatrices(d['formula'], data, eval_env=depth,
                                          return_type='dataframe')
                    break
                except (NameError, PatsyError) as e:
                    exc.append(e)   # why do I need a reference from outside except block
                    pass
            else:
                raise exc[-1]

            self.design_info = design.design_info
            del d["restore_design_info"]
        self.__dict__.update(d)

    def _handle_constant(self, hasconst):
        if hasconst is False or self.exog is None:
            self.k_constant = 0
            self.const_idx = None
        else:
            # detect where the constant is
            check_implicit = False
            exog_max = np.max(self.exog, axis=0)
            if not np.isfinite(exog_max).all():
                raise MissingDataError('exog contains inf or nans')
            exog_min = np.min(self.exog, axis=0)
            const_idx = np.where(exog_max == exog_min)[0].squeeze()
            self.k_constant = const_idx.size

            if self.k_constant == 1:
                if self.exog[:, const_idx].mean() != 0:
                    self.const_idx = int(const_idx)
                else:
                    # we only have a zero column and no other constant
                    check_implicit = True
            elif self.k_constant > 1:
                # we have more than one constant column
                # look for ones
                values = []  # keep values if we need != 0
                for idx in const_idx:
                    value = self.exog[:, idx].mean()
                    if value == 1:
                        self.k_constant = 1
                        self.const_idx = int(idx)
                        break
                    values.append(value)
                else:
                    # we did not break, no column of ones
                    pos = (np.array(values) != 0)
                    if pos.any():
                        # take the first nonzero column
                        self.k_constant = 1
                        self.const_idx = int(const_idx[pos.argmax()])
                    else:
                        # only zero columns
                        check_implicit = True
            elif self.k_constant == 0:
                check_implicit = True
            else:
                # should not be here
                pass

            if check_implicit and not hasconst:
                # look for implicit constant
                # Compute rank of augmented matrix
                augmented_exog = np.column_stack(
                            (np.ones(self.exog.shape[0]), self.exog))
                rank_augm = np.linalg.matrix_rank(augmented_exog)
                rank_orig = np.linalg.matrix_rank(self.exog)
                self.k_constant = int(rank_orig == rank_augm)
                self.const_idx = None
            elif hasconst:
                # Ensure k_constant is 1 any time hasconst is True
                # even if one is not found
                self.k_constant = 1

    @classmethod
    def _drop_nans(cls, x, nan_mask):
        return x[nan_mask]

    @classmethod
    def _drop_nans_2d(cls, x, nan_mask):
        return x[nan_mask][:, nan_mask]

    @classmethod
    def handle_missing(cls, endog, exog, missing, **kwargs):
        """
        This returns a dictionary with keys endog, exog and the keys of
        kwargs. It preserves Nones.
        """
        none_array_names = []

        # patsy's already dropped NaNs in y/X
        missing_idx = kwargs.pop('missing_idx', None)

        if missing_idx is not None:
            # y, X already handled by patsy. add back in later.
            combined = ()
            combined_names = []
            if exog is None:
                none_array_names += ['exog']
        elif exog is not None:
            combined = (endog, exog)
            combined_names = ['endog', 'exog']
        else:
            combined = (endog,)
            combined_names = ['endog']
            none_array_names += ['exog']

        # deal with other arrays
        combined_2d = ()
        combined_2d_names = []
        if len(kwargs):
            for key, value_array in kwargs.items():
                if value_array is None or np.ndim(value_array) == 0:
                    none_array_names += [key]
                    continue
                # grab 1d arrays
                if value_array.ndim == 1:
                    combined += (np.asarray(value_array),)
                    combined_names += [key]
                elif value_array.squeeze().ndim == 1:
                    combined += (np.asarray(value_array),)
                    combined_names += [key]

                # grab 2d arrays that are _assumed_ to be symmetric
                elif value_array.ndim == 2:
                    combined_2d += (np.asarray(value_array),)
                    combined_2d_names += [key]
                else:
                    raise ValueError("Arrays with more than 2 dimensions "
                                     "are not yet handled")

        if missing_idx is not None:
            nan_mask = missing_idx
            updated_row_mask = None
            if combined:  # there were extra arrays not handled by patsy
                combined_nans = _nan_rows(*combined)
                if combined_nans.shape[0] != nan_mask.shape[0]:
                    raise ValueError("Shape mismatch between endog/exog "
                                     "and extra arrays given to model.")
                # for going back and updated endog/exog
                updated_row_mask = combined_nans[~nan_mask]
                nan_mask |= combined_nans  # for updating extra arrays only
            if combined_2d:
                combined_2d_nans = _nan_rows(combined_2d)
                if combined_2d_nans.shape[0] != nan_mask.shape[0]:
                    raise ValueError("Shape mismatch between endog/exog "
                                     "and extra 2d arrays given to model.")
                if updated_row_mask is not None:
                    updated_row_mask |= combined_2d_nans[~nan_mask]
                else:
                    updated_row_mask = combined_2d_nans[~nan_mask]
                nan_mask |= combined_2d_nans

        else:
            nan_mask = _nan_rows(*combined)
            if combined_2d:
                nan_mask = _nan_rows(*(nan_mask[:, None],) + combined_2d)

        if not np.any(nan_mask):  # no missing do not do anything
            combined = dict(zip(combined_names, combined))
            if combined_2d:
                combined.update(dict(zip(combined_2d_names, combined_2d)))
            if none_array_names:
                combined.update({k: kwargs.get(k, None)
                                 for k in none_array_names})

            if missing_idx is not None:
                combined.update({'endog': endog})
                if exog is not None:
                    combined.update({'exog': exog})

            return combined, []

        elif missing == 'raise':
            raise MissingDataError("NaNs were encountered in the data")

        elif missing == 'drop':
            nan_mask = ~nan_mask
            drop_nans = lambda x: cls._drop_nans(x, nan_mask)
            drop_nans_2d = lambda x: cls._drop_nans_2d(x, nan_mask)
            combined = dict(zip(combined_names, lmap(drop_nans, combined)))

            if missing_idx is not None:
                if updated_row_mask is not None:
                    updated_row_mask = ~updated_row_mask
                    # update endog/exog with this new information
                    endog = cls._drop_nans(endog, updated_row_mask)
                    if exog is not None:
                        exog = cls._drop_nans(exog, updated_row_mask)

                combined.update({'endog': endog})
                if exog is not None:
                    combined.update({'exog': exog})

            if combined_2d:
                combined.update(dict(zip(combined_2d_names,
                                         lmap(drop_nans_2d, combined_2d))))
            if none_array_names:
                combined.update({k: kwargs.get(k, None)
                                 for k in none_array_names})

            return combined, np.where(~nan_mask)[0].tolist()
        else:
            raise ValueError("missing option %s not understood" % missing)

    def _convert_endog_exog(self, endog, exog):

        # for consistent outputs if endog is (n,1)
        yarr = self._get_yarr(endog)
        xarr = None
        if exog is not None:
            xarr = self._get_xarr(exog)
            if xarr.ndim == 1:
                xarr = xarr[:, None]
            if xarr.ndim != 2:
                raise ValueError("exog is not 1d or 2d")

        return yarr, xarr

    @cache_writable()
    def ynames(self):
        endog = self.orig_endog
        ynames = self._get_names(endog)
        if not ynames:
            ynames = _make_endog_names(self.endog)

        if len(ynames) == 1:
            return ynames[0]
        else:
            return list(ynames)

    @cache_writable()
    def xnames(self) -> list[str] | None:
        exog = self.orig_exog
        if exog is not None:
            xnames = self._get_names(exog)
            if not xnames:
                xnames = _make_exog_names(self.exog)
            return list(xnames)
        return None

    @property
    def param_names(self):
        # for handling names of 'extra' parameters in summary, etc.
        return self._param_names or self.xnames

    @param_names.setter
    def param_names(self, values):
        self._param_names = values

    @property
    def cov_names(self):
        """
        Labels for covariance matrices

        In multidimensional models, each dimension of a covariance matrix
        differs from the number of param_names.

        If not set, returns param_names
        """
        # for handling names of covariance names in multidimensional models
        if self._cov_names is not None:
            return self._cov_names
        return self.param_names

    @cov_names.setter
    def cov_names(self, value):
        # for handling names of covariance names in multidimensional models
        self._cov_names = value

    @cache_readonly
    def row_labels(self):
        exog = self.orig_exog
        if exog is not None:
            row_labels = self._get_row_labels(exog)
        else:
            endog = self.orig_endog
            row_labels = self._get_row_labels(endog)
        return row_labels

    def _get_row_labels(self, arr):
        return None

    def _get_names(self, arr):
        if isinstance(arr, DataFrame):
            if isinstance(arr.columns, MultiIndex):
                # Flatten MultiIndexes into "simple" column names
                return ['_'.join((level for level in c if level))
                        for c in arr.columns]
            else:
                return list(arr.columns)
        elif isinstance(arr, Series):
            if arr.name:
                return [arr.name]
            else:
                return
        else:
            try:
                return arr.dtype.names
            except AttributeError:
                pass

        return None

    def _get_yarr(self, endog):
        if data_util._is_structured_ndarray(endog):
            endog = data_util.struct_to_ndarray(endog)
        endog = np.asarray(endog)
        if len(endog) == 1:  # never squeeze to a scalar
            if endog.ndim == 1:
                return endog
            elif endog.ndim > 1:
                return np.asarray([endog.squeeze()])

        return endog.squeeze()

    def _get_xarr(self, exog):
        if data_util._is_structured_ndarray(exog):
            exog = data_util.struct_to_ndarray(exog)
        return np.asarray(exog)

    def _check_integrity(self):
        if self.exog is not None:
            if len(self.exog) != len(self.endog):
                raise ValueError("endog and exog matrices are different sizes")

    def wrap_output(self, obj, how='columns', names=None):
        if how == 'columns':
            return self.attach_columns(obj)
        elif how == 'rows':
            return self.attach_rows(obj)
        elif how == 'cov':
            return self.attach_cov(obj)
        elif how == 'dates':
            return self.attach_dates(obj)
        elif how == 'columns_eq':
            return self.attach_columns_eq(obj)
        elif how == 'cov_eq':
            return self.attach_cov_eq(obj)
        elif how == 'generic_columns':
            return self.attach_generic_columns(obj, names)
        elif how == 'generic_columns_2d':
            return self.attach_generic_columns_2d(obj, names)
        elif how == 'ynames':
            return self.attach_ynames(obj)
        elif how == 'multivariate_confint':
            return self.attach_mv_confint(obj)
        else:
            return obj

    def attach_columns(self, result):
        return result

    def attach_columns_eq(self, result):
        return result

    def attach_cov(self, result):
        return result

    def attach_cov_eq(self, result):
        return result

    def attach_rows(self, result):
        return result

    def attach_dates(self, result):
        return result

    def attach_mv_confint(self, result):
        return result

    def attach_generic_columns(self, result, *args, **kwargs):
        return result

    def attach_generic_columns_2d(self, result, *args, **kwargs):
        return result

    def attach_ynames(self, result):
        return result


class PatsyData(ModelData):
    def _get_names(self, arr):
        return arr.design_info.column_names


class PandasData(ModelData):
    """
    Data handling class which knows how to reattach pandas metadata to model
    results
    """

    def _convert_endog_exog(self, endog, exog=None):
        #TODO: remove this when we handle dtype systematically
        endog = np.asarray(endog)
        exog = exog if exog is None else np.asarray(exog)
        if endog.dtype == object or exog is not None and exog.dtype == object:
            raise ValueError("Pandas data cast to numpy dtype of object. "
                             "Check input data with np.asarray(data).")
        return super(PandasData, self)._convert_endog_exog(endog, exog)

    @classmethod
    def _drop_nans(cls, x, nan_mask):
        if isinstance(x, (Series, DataFrame)):
            return x.loc[nan_mask]
        else:  # extra arguments could be plain ndarrays
            return super(PandasData, cls)._drop_nans(x, nan_mask)

    @classmethod
    def _drop_nans_2d(cls, x, nan_mask):
        if isinstance(x, (Series, DataFrame)):
            return x.loc[nan_mask].loc[:, nan_mask]
        else:  # extra arguments could be plain ndarrays
            return super(PandasData, cls)._drop_nans_2d(x, nan_mask)

    def _check_integrity(self):
        endog, exog = self.orig_endog, self.orig_exog
        # exog can be None and we could be upcasting one or the other
        if (exog is not None and
                (hasattr(endog, 'index') and hasattr(exog, 'index')) and
                not self.orig_endog.index.equals(self.orig_exog.index)):
            raise ValueError("The indices for endog and exog are not aligned")
        super(PandasData, self)._check_integrity()

    def _get_row_labels(self, arr):
        try:
            return arr.index
        except AttributeError:
            # if we've gotten here it's because endog is pandas and
            # exog is not, so just return the row labels from endog
            return self.orig_endog.index

    def attach_generic_columns(self, result, names):
        # get the attribute to use
        column_names = getattr(self, names, None)
        return Series(result, index=column_names)

    def attach_generic_columns_2d(self, result, rownames, colnames=None):
        colnames = colnames or rownames
        rownames = getattr(self, rownames, None)
        colnames = getattr(self, colnames, None)
        return DataFrame(result, index=rownames, columns=colnames)

    def attach_columns(self, result):
        # this can either be a 1d array or a scalar
        # do not squeeze because it might be a 2d row array
        # if it needs a squeeze, the bug is elsewhere
        if result.ndim <= 1:
            return Series(result, index=self.param_names)
        else:  # for e.g., confidence intervals
            return DataFrame(result, index=self.param_names)

    def attach_columns_eq(self, result):
        return DataFrame(result, index=self.xnames, columns=self.ynames)

    def attach_cov(self, result):
        return DataFrame(result, index=self.cov_names, columns=self.cov_names)

    def attach_cov_eq(self, result):
        return DataFrame(result, index=self.ynames, columns=self.ynames)

    def attach_rows(self, result):
        # assumes if len(row_labels) > len(result) it's bc it was truncated
        # at the front, for AR lags, for example
        squeezed = result.squeeze()
        k_endog = np.array(self.ynames, ndmin=1).shape[0]
        if k_endog > 1 and squeezed.shape == (k_endog,):
            squeezed = squeezed[None, :]
        # May be zero-dim, for example in the case of forecast one step in tsa
        if squeezed.ndim < 2:
            out = Series(squeezed)
        else:
            out = DataFrame(result)
            out.columns = self.ynames
        out.index = self.row_labels[-len(result):]
        return out

    def attach_dates(self, result):
        squeezed = result.squeeze()
        k_endog = np.array(self.ynames, ndmin=1).shape[0]
        if k_endog > 1 and squeezed.shape == (k_endog,):
            squeezed = np.asarray(squeezed)[None, :]
        # May be zero-dim, for example in the case of forecast one step in tsa
        if squeezed.ndim < 2:
            return Series(squeezed, index=self.predict_dates)
        else:
            return DataFrame(np.asarray(result),
                             index=self.predict_dates,
                             columns=self.ynames)

    def attach_mv_confint(self, result):
        return DataFrame(result.reshape((-1, 2)),
                         index=self.cov_names,
                         columns=['lower', 'upper'])

    def attach_ynames(self, result):
        squeezed = result.squeeze()
        # May be zero-dim, for example in the case of forecast one step in tsa
        if squeezed.ndim < 2:
            return Series(squeezed, name=self.ynames)
        else:
            return DataFrame(result, columns=self.ynames)


def _make_endog_names(endog):
    if endog.ndim == 1 or endog.shape[1] == 1:
        ynames = ['y']
    else:  # for VAR
        ynames = ['y%d' % (i+1) for i in range(endog.shape[1])]

    return ynames


def _make_exog_names(exog):
    exog_var = exog.var(0)
    if (exog_var == 0).any():
        # assumes one constant in first or last position
        # avoid exception if more than one constant
        const_idx = exog_var.argmin()
        exog_names = ['x%d' % i for i in range(1, exog.shape[1])]
        exog_names.insert(const_idx, 'const')
    else:
        exog_names = ['x%d' % i for i in range(1, exog.shape[1]+1)]

    return exog_names


def handle_missing(endog, exog=None, missing='none', **kwargs):
    klass = handle_data_class_factory(endog, exog)
    if missing == 'none':
        ret_dict = dict(endog=endog, exog=exog)
        ret_dict.update(kwargs)
        return ret_dict, None
    return klass.handle_missing(endog, exog, missing=missing, **kwargs)


def handle_data_class_factory(endog, exog):
    """
    Given inputs
    """
    if data_util._is_using_ndarray_type(endog, exog):
        klass = ModelData
    elif data_util._is_using_pandas(endog, exog):
        klass = PandasData
    elif data_util._is_using_patsy(endog, exog):
        klass = PatsyData
    # keep this check last
    elif data_util._is_using_ndarray(endog, exog):
        klass = ModelData
    else:
        raise ValueError('unrecognized data structures: %s / %s' %
                         (type(endog), type(exog)))
    return klass


def handle_data(endog, exog, missing='none', hasconst=None, **kwargs):
    # deal with lists and tuples up-front
    if isinstance(endog, (list, tuple)):
        endog = np.asarray(endog)
    if isinstance(exog, (list, tuple)):
        exog = np.asarray(exog)

    klass = handle_data_class_factory(endog, exog)
    return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
                 **kwargs)
