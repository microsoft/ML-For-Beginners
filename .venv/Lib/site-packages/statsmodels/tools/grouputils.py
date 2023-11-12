# -*- coding: utf-8 -*-
"""Tools for working with groups

This provides several functions to work with groups and a Group class that
keeps track of the different representations and has methods to work more
easily with groups.


Author: Josef Perktold,
Author: Nathaniel Smith, recipe for sparse_dummies on scipy user mailing list

Created on Tue Nov 29 15:44:53 2011 : sparse_dummies
Created on Wed Nov 30 14:28:24 2011 : combine_indices
changes: add Group class

Notes
~~~~~

This reverses the class I used before, where the class was for the data and
the group was auxiliary. Here, it is only the group, no data is kept.

sparse_dummies needs checking for corner cases, e.g.
what if a category level has zero elements? This can happen with subset
    selection even if the original groups where defined as arange.

Not all methods and options have been tried out yet after refactoring

need more efficient loop if groups are sorted -> see GroupSorted.group_iter
"""
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd

import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex


def combine_indices(groups, prefix='', sep='.', return_labels=False):
    """use np.unique to get integer group indices for product, intersection
    """
    if isinstance(groups, tuple):
        groups = np.column_stack(groups)
    else:
        groups = np.asarray(groups)

    dt = groups.dtype

    is2d = (groups.ndim == 2)  # need to store

    if is2d:
        ncols = groups.shape[1]
        if not groups.flags.c_contiguous:
            groups = np.array(groups, order='C')

        groups_ = groups.view([('', groups.dtype)] * groups.shape[1])
    else:
        groups_ = groups

    uni, uni_idx, uni_inv = np.unique(groups_, return_index=True,
                                      return_inverse=True)

    if is2d:
        uni = uni.view(dt).reshape(-1, ncols)

        # avoiding a view would be
        # for t in uni.dtype.fields.values():
        #     assert (t[0] == dt)
        #
        # uni.dtype = dt
        # uni.shape = (uni.size//ncols, ncols)

    if return_labels:
        label = [(prefix+sep.join(['%s']*len(uni[0]))) % tuple(ii)
                 for ii in uni]
        return uni_inv, uni_idx, uni, label
    else:
        return uni_inv, uni_idx, uni


# written for and used in try_covariance_grouploop.py
def group_sums(x, group, use_bincount=True):
    """simple bincount version, again

    group : ndarray, integer
        assumed to be consecutive integers

    no dtype checking because I want to raise in that case

    uses loop over columns of x

    for comparison, simple python loop
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2 and use_bincount:
        raise ValueError('not implemented yet')

    if use_bincount:

        # re-label groups or bincount takes too much memory
        if np.max(group) > 2 * x.shape[0]:
            group = pd.factorize(group)[0]

        return np.array([np.bincount(group, weights=x[:, col])
                         for col in range(x.shape[1])])
    else:
        uniques = np.unique(group)
        result = np.zeros([len(uniques)] + list(x.shape[1:]))
        for ii, cat in enumerate(uniques):
            result[ii] = x[group == cat].sum(0)
        return result


def group_sums_dummy(x, group_dummy):
    """sum by groups given group dummy variable

    group_dummy can be either ndarray or sparse matrix
    """
    if data_util._is_using_ndarray_type(group_dummy, None):
        return np.dot(x.T, group_dummy)
    else:  # check for sparse
        return x.T * group_dummy


# TODO: See if this can be entirely replaced by Grouping.dummy_sparse;
#  see GH#5687
def dummy_sparse(groups):
    """create a sparse indicator from a group array with integer labels

    Parameters
    ----------
    groups : ndarray, int, 1d (nobs,)
        an array of group indicators for each observation. Group levels are
        assumed to be defined as consecutive integers, i.e. range(n_groups)
        where n_groups is the number of group levels. A group level with no
        observations for it will still produce a column of zeros.

    Returns
    -------
    indi : ndarray, int8, 2d (nobs, n_groups)
        an indicator array with one row per observation, that has 1 in the
        column of the group level for that observation

    Examples
    --------

    >>> g = np.array([0, 0, 2, 1, 1, 2, 0])
    >>> indi = dummy_sparse(g)
    >>> indi
    <7x3 sparse matrix of type '<type 'numpy.int8'>'
        with 7 stored elements in Compressed Sparse Row format>
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)


    current behavior with missing groups
    >>> g = np.array([0, 0, 2, 0, 2, 0])
    >>> indi = dummy_sparse(g)
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)
    """
    from scipy import sparse

    indptr = np.arange(len(groups)+1)
    data = np.ones(len(groups), dtype=np.int8)
    indi = sparse.csr_matrix((data, groups, indptr))

    return indi


class Group:

    def __init__(self, group, name=''):

        # self.group = np.asarray(group)  # TODO: use checks in combine_indices
        self.name = name
        uni, uni_idx, uni_inv = combine_indices(group)

        # TODO: rename these to something easier to remember
        self.group_int, self.uni_idx, self.uni = uni, uni_idx, uni_inv

        self.n_groups = len(self.uni)

        # put this here so they can be overwritten before calling labels
        self.separator = '.'
        self.prefix = self.name
        if self.prefix:
            self.prefix = self.prefix + '='

    # cache decorator
    def counts(self):
        return np.bincount(self.group_int)

    # cache_decorator
    def labels(self):
        # is this only needed for product of groups (intersection)?
        prefix = self.prefix
        uni = self.uni
        sep = self.separator

        if uni.ndim > 1:
            label = [(prefix+sep.join(['%s']*len(uni[0]))) % tuple(ii)
                     for ii in uni]
        else:
            label = [prefix + '%s' % ii for ii in uni]
        return label

    def dummy(self, drop_idx=None, sparse=False, dtype=int):
        """
        drop_idx is only available if sparse=False

        drop_idx is supposed to index into uni
        """
        uni = self.uni
        if drop_idx is not None:
            idx = lrange(len(uni))
            del idx[drop_idx]
            uni = uni[idx]

        group = self.group

        if not sparse:
            return (group[:, None] == uni[None, :]).astype(dtype)
        else:
            return dummy_sparse(self.group_int)

    def interaction(self, other):
        if isinstance(other, self.__class__):
            other = other.group
        return self.__class__((self, other))

    def group_sums(self, x, use_bincount=True):
        return group_sums(x, self.group_int, use_bincount=use_bincount)

    def group_demean(self, x, use_bincount=True):
        nobs = float(len(x))
        means_g = group_sums(x / nobs, self.group_int,
                             use_bincount=use_bincount)
        x_demeaned = x - means_g[self.group_int]  # check reverse_index?
        return x_demeaned, means_g


class GroupSorted(Group):
    def __init__(self, group, name=''):
        super(self.__class__, self).__init__(group, name=name)

        idx = (np.nonzero(np.diff(group))[0]+1).tolist()
        self.groupidx = lzip([0] + idx, idx + [len(group)])

    def group_iter(self):
        for low, upp in self.groupidx:
            yield slice(low, upp)

    def lag_indices(self, lag):
        """return the index array for lagged values

        Warning: if k is larger then the number of observations for an
        individual, then no values for that individual are returned.

        TODO: for the unbalanced case, I should get the same truncation for
        the array with lag=0. From the return of lag_idx we would not know
        which individual is missing.

        TODO: do I want the full equivalent of lagmat in tsa?
        maxlag or lag or lags.

        not tested yet
        """
        lag_idx = np.asarray(self.groupidx)[:, 1] - lag  # asarray or already?
        mask_ok = (lag <= lag_idx)
        # still an observation that belongs to the same individual

        return lag_idx[mask_ok]


def _is_hierarchical(x):
    """
    Checks if the first item of an array-like object is also array-like
    If so, we have a MultiIndex and returns True. Else returns False.
    """
    item = x[0]
    # is there a better way to do this?
    if isinstance(item, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        return True
    else:
        return False


def _make_hierarchical_index(index, names):
    return MultiIndex.from_tuples(*[index], names=names)


def _make_generic_names(index):
    n_names = len(index.names)
    pad = str(len(str(n_names)))  # number of digits
    return [("group{0:0"+pad+"}").format(i) for i in range(n_names)]


class Grouping:
    def __init__(self, index, names=None):
        """
        index : index-like
            Can be pandas MultiIndex or Index or array-like. If array-like
            and is a MultipleIndex (more than one grouping variable),
            groups are expected to be in each row. E.g., [('red', 1),
            ('red', 2), ('green', 1), ('green', 2)]
        names : list or str, optional
            The names to use for the groups. Should be a str if only
            one grouping variable is used.

        Notes
        -----
        If index is already a pandas Index then there is no copy.
        """
        if isinstance(index, (Index, MultiIndex)):
            if names is not None:
                if hasattr(index, 'set_names'):  # newer pandas
                    index.set_names(names, inplace=True)
                else:
                    index.names = names
            self.index = index
        else:  # array_like
            if _is_hierarchical(index):
                self.index = _make_hierarchical_index(index, names)
            else:
                self.index = Index(index, name=names)
            if names is None:
                names = _make_generic_names(self.index)
                if hasattr(self.index, 'set_names'):
                    self.index.set_names(names, inplace=True)
                else:
                    self.index.names = names

        self.nobs = len(self.index)
        self.nlevels = len(self.index.names)
        self.slices = None

    @property
    def index_shape(self):
        if hasattr(self.index, 'levshape'):
            return self.index.levshape
        else:
            return self.index.shape

    @property
    def levels(self):
        if hasattr(self.index, 'levels'):
            return self.index.levels
        else:
            return pd.Categorical(self.index).levels

    @property
    def labels(self):
        # this was index_int, but that's not a very good name...
        codes = getattr(self.index, 'codes', None)
        if codes is None:
            if hasattr(self.index, 'labels'):
                codes = self.index.labels
            else:
                codes = pd.Categorical(self.index).codes[None]
        return codes

    @property
    def group_names(self):
        return self.index.names

    def reindex(self, index=None, names=None):
        """
        Resets the index in-place.
        """
        # NOTE: this is not of much use if the rest of the data does not change
        # This needs to reset cache
        if names is None:
            names = self.group_names
        self = Grouping(index, names)

    def get_slices(self, level=0):
        """
        Sets the slices attribute to be a list of indices of the sorted
        groups for the first index level. I.e., self.slices[0] is the
        index where each observation is in the first (sorted) group.
        """
        # TODO: refactor this
        groups = self.index.get_level_values(level).unique()
        groups = np.array(groups)
        groups.sort()
        if isinstance(self.index, MultiIndex):
            self.slices = [self.index.get_loc_level(x, level=level)[0]
                           for x in groups]
        else:
            self.slices = [self.index.get_loc(x) for x in groups]

    def count_categories(self, level=0):
        """
        Sets the attribute counts to equal the bincount of the (integer-valued)
        labels.
        """
        # TODO: refactor this not to set an attribute. Why would we do this?
        self.counts = np.bincount(self.labels[level])

    def check_index(self, is_sorted=True, unique=True, index=None):
        """Sanity checks"""
        if not index:
            index = self.index
        if is_sorted:
            test = pd.DataFrame(lrange(len(index)), index=index)
            test_sorted = test.sort()
            if not test.index.equals(test_sorted.index):
                raise Exception('Data is not be sorted')
        if unique:
            if len(index) != len(index.unique()):
                raise Exception('Duplicate index entries')

    def sort(self, data, index=None):
        """Applies a (potentially hierarchical) sort operation on a numpy array
        or pandas series/dataframe based on the grouping index or a
        user-supplied index.  Returns an object of the same type as the
        original data as well as the matching (sorted) Pandas index.
        """

        if index is None:
            index = self.index
        if data_util._is_using_ndarray_type(data, None):
            if data.ndim == 1:
                out = pd.Series(data, index=index, copy=True)
                out = out.sort_index()
            else:
                out = pd.DataFrame(data, index=index)
                out = out.sort_index(inplace=False)  # copies
            return np.array(out), out.index
        elif data_util._is_using_pandas(data, None):
            out = data
            out = out.reindex(index)  # copies?
            out = out.sort_index()
            return out, out.index
        else:
            msg = 'data must be a Numpy array or a Pandas Series/DataFrame'
            raise ValueError(msg)

    def transform_dataframe(self, dataframe, function, level=0, **kwargs):
        """Apply function to each column, by group
        Assumes that the dataframe already has a proper index"""
        if dataframe.shape[0] != self.nobs:
            raise Exception('dataframe does not have the same shape as index')
        out = dataframe.groupby(level=level).apply(function, **kwargs)
        if 1 in out.shape:
            return np.ravel(out)
        else:
            return np.array(out)

    def transform_array(self, array, function, level=0, **kwargs):
        """Apply function to each column, by group
        """
        if array.shape[0] != self.nobs:
            raise Exception('array does not have the same shape as index')
        dataframe = pd.DataFrame(array, index=self.index)
        return self.transform_dataframe(dataframe, function, level=level,
                                        **kwargs)

    def transform_slices(self, array, function, level=0, **kwargs):
        """Apply function to each group. Similar to transform_array but does
        not coerce array to a DataFrame and back and only works on a 1D or 2D
        numpy array. function is called function(group, group_idx, **kwargs).
        """
        array = np.asarray(array)
        if array.shape[0] != self.nobs:
            raise Exception('array does not have the same shape as index')
        # always reset because level is given. need to refactor this.
        self.get_slices(level=level)
        processed = []
        for s in self.slices:
            if array.ndim == 2:
                subset = array[s, :]
            elif array.ndim == 1:
                subset = array[s]
            processed.append(function(subset, s, **kwargs))
        processed = np.array(processed)
        return processed.reshape(-1, processed.shape[-1])

    # TODO: this is not general needs to be a PanelGrouping object
    def dummies_time(self):
        self.dummy_sparse(level=1)
        return self._dummies

    def dummies_groups(self, level=0):
        self.dummy_sparse(level=level)
        return self._dummies

    def dummy_sparse(self, level=0):
        """create a sparse indicator from a group array with integer labels

        Parameters
        ----------
        groups : ndarray, int, 1d (nobs,)
            An array of group indicators for each observation. Group levels
            are assumed to be defined as consecutive integers, i.e.
            range(n_groups) where n_groups is the number of group levels.
            A group level with no observations for it will still produce a
            column of zeros.

        Returns
        -------
        indi : ndarray, int8, 2d (nobs, n_groups)
            an indicator array with one row per observation, that has 1 in the
            column of the group level for that observation

        Examples
        --------

        >>> g = np.array([0, 0, 2, 1, 1, 2, 0])
        >>> indi = dummy_sparse(g)
        >>> indi
        <7x3 sparse matrix of type '<type 'numpy.int8'>'
            with 7 stored elements in Compressed Sparse Row format>
        >>> indi.todense()
        matrix([[1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=int8)


        current behavior with missing groups
        >>> g = np.array([0, 0, 2, 0, 2, 0])
        >>> indi = dummy_sparse(g)
        >>> indi.todense()
        matrix([[1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=int8)
        """
        indi = dummy_sparse(self.labels[level])
        self._dummies = indi
