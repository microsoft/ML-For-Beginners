cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint64_t,
)

cnp.import_array()


from pandas._libs cimport util
from pandas._libs.hashtable cimport HashTable
from pandas._libs.tslibs.nattype cimport c_NaT as NaT
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_unit_from_dtype,
    import_pandas_datetime,
)

import_pandas_datetime()


from pandas._libs.tslibs.period cimport is_period_object
from pandas._libs.tslibs.timedeltas cimport _Timedelta
from pandas._libs.tslibs.timestamps cimport _Timestamp

from pandas._libs import (
    algos,
    hashtable as _hash,
)

from pandas._libs.lib cimport eq_NA_compat
from pandas._libs.missing cimport (
    C_NA,
    checknull,
    is_matching_na,
)

# Defines shift of MultiIndex codes to avoid negative codes (missing values)
multiindex_nulls_shift = 2


cdef bint is_definitely_invalid_key(object val):
    try:
        hash(val)
    except TypeError:
        return True
    return False


cdef ndarray _get_bool_indexer(ndarray values, object val, ndarray mask = None):
    """
    Return a ndarray[bool] of locations where val matches self.values.

    If val is not NA, this is equivalent to `self.values == val`
    """
    # Caller is responsible for ensuring _check_type has already been called
    cdef:
        ndarray[uint8_t, ndim=1, cast=True] indexer
        Py_ssize_t i
        object item

    if values.descr.type_num == cnp.NPY_OBJECT:
        assert mask is None  # no mask for object dtype
        # i.e. values.dtype == object
        if not checknull(val):
            indexer = eq_NA_compat(values, val)

        else:
            # We need to check for _matching_ NA values
            indexer = np.empty(len(values), dtype=np.uint8)

            for i in range(len(values)):
                item = values[i]
                indexer[i] = is_matching_na(item, val)

    else:
        if mask is not None:
            if val is C_NA:
                indexer = mask == 1
            else:
                indexer = (values == val) & ~mask
        else:
            if util.is_nan(val):
                indexer = np.isnan(values)
            else:
                indexer = values == val

    return indexer.view(bool)


# Don't populate hash tables in monotonic indexes larger than this
_SIZE_CUTOFF = 1_000_000


cdef _unpack_bool_indexer(ndarray[uint8_t, ndim=1, cast=True] indexer, object val):
    """
    Possibly unpack a boolean mask to a single indexer.
    """
    # Returns ndarray[bool] or int
    cdef:
        ndarray[intp_t, ndim=1] found
        int count

    found = np.where(indexer)[0]
    count = len(found)

    if count > 1:
        return indexer
    if count == 1:
        return int(found[0])

    raise KeyError(val)


@cython.freelist(32)
cdef class IndexEngine:

    cdef readonly:
        ndarray values
        ndarray mask
        HashTable mapping
        bint over_size_threshold

    cdef:
        bint unique, monotonic_inc, monotonic_dec
        bint need_monotonic_check, need_unique_check
        object _np_type

    def __init__(self, ndarray values):
        self.values = values
        self.mask = None

        self.over_size_threshold = len(values) >= _SIZE_CUTOFF
        self.clear_mapping()
        self._np_type = values.dtype.type

    def __contains__(self, val: object) -> bool:
        hash(val)
        try:
            self.get_loc(val)
        except KeyError:
            return False
        return True

    cpdef get_loc(self, object val):
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t loc

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        val = self._check_type(val)

        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(val)
            values = self.values

            loc = self._searchsorted_left(val)
            if loc >= len(values):
                raise KeyError(val)
            if values[loc] != val:
                raise KeyError(val)
            return loc

        self._ensure_mapping_populated()
        if not self.unique:
            return self._get_loc_duplicates(val)
        if self.mask is not None and val is C_NA:
            return self.mapping.get_na()

        try:
            return self.mapping.get_item(val)
        except OverflowError as err:
            # GH#41775 OverflowError e.g. if we are uint64 and val is -1
            #  or if we are int64 and value is np.iinfo(np.int64).max+1
            #  (the uint64 with -1 case should actually be excluded by _check_type)
            raise KeyError(val) from err

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        """
        See ObjectEngine._searchsorted_left.__doc__.
        """
        # Caller is responsible for ensuring _check_type has already been called
        loc = self.values.searchsorted(self._np_type(val), side="left")
        return loc

    cdef _get_loc_duplicates(self, object val):
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t diff, left, right

        if self.is_monotonic_increasing:
            values = self.values
            try:
                left = values.searchsorted(val, side="left")
                right = values.searchsorted(val, side="right")
            except TypeError:
                # e.g. GH#29189 get_loc(None) with a Float64Index
                #  2021-09-29 Now only reached for object-dtype
                raise KeyError(val)

            diff = right - left
            if diff == 0:
                raise KeyError(val)
            elif diff == 1:
                return left
            else:
                return slice(left, right)

        return self._maybe_get_bool_indexer(val)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer

        indexer = _get_bool_indexer(self.values, val, self.mask)
        return _unpack_bool_indexer(indexer, val)

    def sizeof(self, deep: bool = False) -> int:
        """ return the sizeof our mapping """
        if not self.is_mapping_populated:
            return 0
        return self.mapping.sizeof(deep=deep)

    def __sizeof__(self) -> int:
        return self.sizeof()

    cpdef _update_from_sliced(self, IndexEngine other, reverse: bool):
        self.unique = other.unique
        self.need_unique_check = other.need_unique_check
        if not other.need_monotonic_check and (
                other.is_monotonic_increasing or other.is_monotonic_decreasing):
            self.need_monotonic_check = other.need_monotonic_check
            # reverse=True means the index has been reversed
            self.monotonic_inc = other.monotonic_dec if reverse else other.monotonic_inc
            self.monotonic_dec = other.monotonic_inc if reverse else other.monotonic_dec

    @property
    def is_unique(self) -> bool:
        if self.need_unique_check:
            self._do_unique_check()

        return self.unique == 1

    cdef _do_unique_check(self):
        self._ensure_mapping_populated()

    @property
    def is_monotonic_increasing(self) -> bool:
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_inc == 1

    @property
    def is_monotonic_decreasing(self) -> bool:
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_dec == 1

    cdef _do_monotonic_check(self):
        cdef:
            bint is_strict_monotonic
        if self.mask is not None and np.any(self.mask):
            self.monotonic_inc = 0
            self.monotonic_dec = 0
        else:
            try:
                values = self.values
                self.monotonic_inc, self.monotonic_dec, is_strict_monotonic = \
                    self._call_monotonic(values)
            except TypeError:
                self.monotonic_inc = 0
                self.monotonic_dec = 0
                is_strict_monotonic = 0

            self.need_monotonic_check = 0

            # we can only be sure of uniqueness if is_strict_monotonic=1
            if is_strict_monotonic:
                self.unique = 1
                self.need_unique_check = 0

    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=False)

    cdef _make_hash_table(self, Py_ssize_t n):
        raise NotImplementedError  # pragma: no cover

    cdef _check_type(self, object val):
        hash(val)
        return val

    @property
    def is_mapping_populated(self) -> bool:
        return self.mapping is not None

    cdef _ensure_mapping_populated(self):
        # this populates the mapping
        # if its not already populated
        # also satisfies the need_unique_check

        if not self.is_mapping_populated:

            values = self.values
            self.mapping = self._make_hash_table(len(values))
            self.mapping.map_locations(values, self.mask)

            if len(self.mapping) == len(values):
                self.unique = 1

        self.need_unique_check = 0

    def clear_mapping(self):
        self.mapping = None
        self.need_monotonic_check = 1
        self.need_unique_check = 1

        self.unique = 0
        self.monotonic_inc = 0
        self.monotonic_dec = 0

    def get_indexer(self, ndarray values) -> np.ndarray:
        self._ensure_mapping_populated()
        return self.mapping.lookup(values)

    def get_indexer_non_unique(self, ndarray targets):
        """
        Return an indexer suitable for taking from a non unique index
        return the labels in the same order as the target
        and a missing indexer into the targets (which correspond
        to the -1 indices in the results

        Returns
        -------
        indexer : np.ndarray[np.intp]
        missing : np.ndarray[np.intp]
        """
        cdef:
            ndarray values
            ndarray[intp_t] result, missing
            set stargets, remaining_stargets, found_nas
            dict d = {}
            object val
            Py_ssize_t count = 0, count_missing = 0
            Py_ssize_t i, j, n, n_t, n_alloc, start, end
            bint check_na_values = False

        values = self.values
        stargets = set(targets)

        na_in_stargets = any(checknull(t) for t in stargets)

        n = len(values)
        n_t = len(targets)
        if n > 10_000:
            n_alloc = 10_000
        else:
            n_alloc = n

        result = np.empty(n_alloc, dtype=np.intp)
        missing = np.empty(n_t, dtype=np.intp)

        # map each starget to its position in the index
        if (
                stargets and
                len(stargets) < 5 and
                not na_in_stargets and
                self.is_monotonic_increasing
        ):
            # if there are few enough stargets and the index is monotonically
            # increasing, then use binary search for each starget
            remaining_stargets = set()
            for starget in stargets:
                try:
                    start = values.searchsorted(starget, side="left")
                    end = values.searchsorted(starget, side="right")
                except TypeError:  # e.g. if we tried to search for string in int array
                    remaining_stargets.add(starget)
                else:
                    if start != end:
                        d[starget] = list(range(start, end))

            stargets = remaining_stargets

        if stargets:
            # otherwise, map by iterating through all items in the index

            # short-circuit na check
            if na_in_stargets:
                check_na_values = True
                # keep track of nas in values
                found_nas = set()

            for i in range(n):
                val = values[i]

                # GH#43870
                # handle lookup for nas
                # (ie. np.nan, float("NaN"), Decimal("NaN"), dt64nat, td64nat)
                if check_na_values and checknull(val):
                    match = [na for na in found_nas if is_matching_na(val, na)]

                    # matching na not found
                    if not len(match):
                        found_nas.add(val)

                        # add na to stargets to utilize `in` for stargets/d lookup
                        match_stargets = [
                            x for x in stargets if is_matching_na(val, x)
                        ]

                        if len(match_stargets):
                            # add our 'standardized' na
                            stargets.add(val)

                    # matching na found
                    else:
                        assert len(match) == 1
                        val = match[0]

                if val in stargets:
                    if val not in d:
                        d[val] = []
                    d[val].append(i)

        for i in range(n_t):
            val = targets[i]

            # ensure there are nas in values before looking for a matching na
            if check_na_values and checknull(val):
                match = [na for na in found_nas if is_matching_na(val, na)]
                if len(match):
                    assert len(match) == 1
                    val = match[0]

            # found
            if val in d:
                key = val

                for j in d[key]:

                    # realloc if needed
                    if count >= n_alloc:
                        n_alloc += 10_000
                        result = np.resize(result, n_alloc)

                    result[count] = j
                    count += 1

            # value not found
            else:

                if count >= n_alloc:
                    n_alloc += 10_000
                    result = np.resize(result, n_alloc)
                result[count] = -1
                count += 1
                missing[count_missing] = i
                count_missing += 1

        return result[0:count], missing[0:count_missing]


cdef Py_ssize_t _bin_search(ndarray values, object val) except -1:
    # GH#1757 ndarray.searchsorted is not safe to use with array of tuples
    #  (treats a tuple `val` as a sequence of keys instead of a single key),
    #  so we implement something similar.
    # This is equivalent to the stdlib's bisect.bisect_left

    cdef:
        Py_ssize_t mid = 0, lo = 0, hi = len(values) - 1
        object pval

    if hi == 0 or (hi > 0 and val > values[hi]):
        return len(values)

    while lo < hi:
        mid = (lo + hi) // 2
        pval = values[mid]
        if val < pval:
            hi = mid
        elif val > pval:
            lo = mid + 1
        else:
            while mid > 0 and val == values[mid - 1]:
                mid -= 1
            return mid

    if val <= values[mid]:
        return mid
    else:
        return mid + 1


cdef class ObjectEngine(IndexEngine):
    """
    Index Engine for use with object-dtype Index, namely the base class Index.
    """
    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.PyObjectHashTable(n)

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        # using values.searchsorted here would treat a tuple `val` as a sequence
        #  instead of a single key, so we use a different implementation
        try:
            loc = _bin_search(self.values, val)
        except TypeError as err:
            raise KeyError(val) from err
        return loc


cdef class DatetimeEngine(Int64Engine):

    cdef:
        NPY_DATETIMEUNIT _creso

    def __init__(self, ndarray values):
        super().__init__(values.view("i8"))
        self._creso = get_unit_from_dtype(values.dtype)

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # NB: caller is responsible for ensuring tzawareness compat
        #  before we get here
        if scalar is NaT:
            return NaT._value
        elif isinstance(scalar, _Timestamp):
            if scalar._creso == self._creso:
                return scalar._value
            else:
                # Note: caller is responsible for catching potential ValueError
                #  from _as_creso
                return (
                    (<_Timestamp>scalar)._as_creso(self._creso, round_ok=False)._value
                )
        raise TypeError(scalar)

    def __contains__(self, val: object) -> bool:
        # We assume before we get here:
        #  - val is hashable
        try:
            self._unbox_scalar(val)
        except ValueError:
            return False

        try:
            self.get_loc(val)
            return True
        except KeyError:
            return False

    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=True)

    cpdef get_loc(self, object val):
        # NB: the caller is responsible for ensuring that we are called
        #  with either a Timestamp or NaT (Timedelta or NaT for TimedeltaEngine)

        cdef:
            Py_ssize_t loc

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        try:
            conv = self._unbox_scalar(val)
        except (TypeError, ValueError) as err:
            raise KeyError(val) from err

        # Welcome to the spaghetti factory
        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(conv)
            values = self.values

            loc = values.searchsorted(conv, side="left")

            if loc == len(values) or values[loc] != conv:
                raise KeyError(val)
            return loc

        self._ensure_mapping_populated()
        if not self.unique:
            return self._get_loc_duplicates(conv)

        try:
            return self.mapping.get_item(conv)
        except KeyError:
            raise KeyError(val)


cdef class TimedeltaEngine(DatetimeEngine):

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        if scalar is NaT:
            return NaT._value
        elif isinstance(scalar, _Timedelta):
            if scalar._creso == self._creso:
                return scalar._value
            else:
                # Note: caller is responsible for catching potential ValueError
                #  from _as_creso
                return (
                    (<_Timedelta>scalar)._as_creso(self._creso, round_ok=False)._value
                )
        raise TypeError(scalar)


cdef class PeriodEngine(Int64Engine):

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        if scalar is NaT:
            return scalar._value
        if is_period_object(scalar):
            # NB: we assume that we have the correct freq here.
            return scalar.ordinal
        raise TypeError(scalar)

    cpdef get_loc(self, object val):
        # NB: the caller is responsible for ensuring that we are called
        #  with either a Period or NaT
        cdef:
            int64_t conv

        try:
            conv = self._unbox_scalar(val)
        except TypeError:
            raise KeyError(val)

        return Int64Engine.get_loc(self, conv)

    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=True)


cdef class BaseMultiIndexCodesEngine:
    """
    Base class for MultiIndexUIntEngine and MultiIndexPyIntEngine, which
    represent each label in a MultiIndex as an integer, by juxtaposing the bits
    encoding each level, with appropriate offsets.

    For instance: if 3 levels have respectively 3, 6 and 1 possible values,
    then their labels can be represented using respectively 2, 3 and 1 bits,
    as follows:
     _ _ _ _____ _ __ __ __
    |0|0|0| ... |0| 0|a1|a0| -> offset 0 (first level)
     — — — ————— — —— —— ——
    |0|0|0| ... |0|b2|b1|b0| -> offset 2 (bits required for first level)
     — — — ————— — —— —— ——
    |0|0|0| ... |0| 0| 0|c0| -> offset 5 (bits required for first two levels)
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾
    and the resulting unsigned integer representation will be:
     _ _ _ _____ _ __ __ __ __ __ __
    |0|0|0| ... |0|c0|b2|b1|b0|a1|a0|
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾

    Offsets are calculated at initialization, labels are transformed by method
    _codes_to_ints.

    Keys are located by first locating each component against the respective
    level, then locating (the integer representation of) codes.
    """
    def __init__(self, object levels, object labels,
                 ndarray[uint64_t, ndim=1] offsets):
        """
        Parameters
        ----------
        levels : list-like of numpy arrays
            Levels of the MultiIndex.
        labels : list-like of numpy arrays of integer dtype
            Labels of the MultiIndex.
        offsets : numpy array of uint64 dtype
            Pre-calculated offsets, one for each level of the index.
        """
        self.levels = levels
        self.offsets = offsets

        # Transform labels in a single array, and add 2 so that we are working
        # with positive integers (-1 for NaN becomes 1). This enables us to
        # differentiate between values that are missing in other and matching
        # NaNs. We will set values that are not found to 0 later:
        labels_arr = np.array(labels, dtype="int64").T + multiindex_nulls_shift
        codes = labels_arr.astype("uint64", copy=False)
        self.level_has_nans = [-1 in lab for lab in labels]

        # Map each codes combination in the index to an integer unambiguously
        # (no collisions possible), based on the "offsets", which describe the
        # number of bits to switch labels for each level:
        lab_ints = self._codes_to_ints(codes)

        # Initialize underlying index (e.g. libindex.UInt64Engine) with
        # integers representing labels: we will use its get_loc and get_indexer
        self._base.__init__(self, lab_ints)

    def _codes_to_ints(self, ndarray[uint64_t] codes) -> np.ndarray:
        raise NotImplementedError("Implemented by subclass")  # pragma: no cover

    def _extract_level_codes(self, target) -> np.ndarray:
        """
        Map the requested list of (tuple) keys to their integer representations
        for searching in the underlying integer index.

        Parameters
        ----------
        target : MultiIndex

        Returns
        ------
        int_keys : 1-dimensional array of dtype uint64 or object
            Integers representing one combination each
        """
        level_codes = list(target._recode_for_new_levels(self.levels))
        for i, codes in enumerate(level_codes):
            if self.levels[i].hasnans:
                na_index = self.levels[i].isna().nonzero()[0][0]
                codes[target.codes[i] == -1] = na_index
            codes += 1
            codes[codes > 0] += 1
            if self.level_has_nans[i]:
                codes[target.codes[i] == -1] += 1
        return self._codes_to_ints(np.array(level_codes, dtype="uint64").T)

    def get_indexer(self, target: np.ndarray) -> np.ndarray:
        """
        Returns an array giving the positions of each value of `target` in
        `self.values`, where -1 represents a value in `target` which does not
        appear in `self.values`

        Parameters
        ----------
        target : np.ndarray

        Returns
        -------
        np.ndarray[intp_t, ndim=1] of the indexer of `target` into
        `self.values`
        """
        return self._base.get_indexer(self, target)

    def get_indexer_with_fill(self, ndarray target, ndarray values,
                              str method, object limit) -> np.ndarray:
        """
        Returns an array giving the positions of each value of `target` in
        `values`, where -1 represents a value in `target` which does not
        appear in `values`

        If `method` is "backfill" then the position for a value in `target`
        which does not appear in `values` is that of the next greater value
        in `values` (if one exists), and -1 if there is no such value.

        Similarly, if the method is "pad" then the position for a value in
        `target` which does not appear in `values` is that of the next smaller
        value in `values` (if one exists), and -1 if there is no such value.

        Parameters
        ----------
        target: ndarray[object] of tuples
            need not be sorted, but all must have the same length, which must be
            the same as the length of all tuples in `values`
        values : ndarray[object] of tuples
            must be sorted and all have the same length.  Should be the set of
            the MultiIndex's values.
        method: string
            "backfill" or "pad"
        limit: int or None
            if provided, limit the number of fills to this value

        Returns
        -------
        np.ndarray[intp_t, ndim=1] of the indexer of `target` into `values`,
        filled with the `method` (and optionally `limit`) specified
        """
        assert method in ("backfill", "pad")
        cdef:
            int64_t i, j, next_code
            int64_t num_values, num_target_values
            ndarray[int64_t, ndim=1] target_order
            ndarray[object, ndim=1] target_values
            ndarray[int64_t, ndim=1] new_codes, new_target_codes
            ndarray[intp_t, ndim=1] sorted_indexer

        target_order = np.argsort(target).astype("int64")
        target_values = target[target_order]
        num_values, num_target_values = len(values), len(target_values)
        new_codes, new_target_codes = (
            np.empty((num_values,)).astype("int64"),
            np.empty((num_target_values,)).astype("int64"),
        )

        # `values` and `target_values` are both sorted, so we walk through them
        # and memoize the (ordered) set of indices in the (implicit) merged-and
        # sorted list of the two which belong to each of them
        # the effect of this is to create a factorization for the (sorted)
        # merger of the index values, where `new_codes` and `new_target_codes`
        # are the subset of the factors which appear in `values` and `target`,
        # respectively
        i, j, next_code = 0, 0, 0
        while i < num_values and j < num_target_values:
            val, target_val = values[i], target_values[j]
            if val <= target_val:
                new_codes[i] = next_code
                i += 1
            if target_val <= val:
                new_target_codes[j] = next_code
                j += 1
            next_code += 1

        # at this point, at least one should have reached the end
        # the remaining values of the other should be added to the end
        assert i == num_values or j == num_target_values
        while i < num_values:
            new_codes[i] = next_code
            i += 1
            next_code += 1
        while j < num_target_values:
            new_target_codes[j] = next_code
            j += 1
            next_code += 1

        # get the indexer, and undo the sorting of `target.values`
        algo = algos.backfill if method == "backfill" else algos.pad
        sorted_indexer = algo(new_codes, new_target_codes, limit=limit)
        return sorted_indexer[np.argsort(target_order)]

    def get_loc(self, object key):
        if is_definitely_invalid_key(key):
            raise TypeError(f"'{key}' is an invalid key")
        if not isinstance(key, tuple):
            raise KeyError(key)
        try:
            indices = [1 if checknull(v) else lev.get_loc(v) + multiindex_nulls_shift
                       for lev, v in zip(self.levels, key)]
        except KeyError:
            raise KeyError(key)

        # Transform indices into single integer:
        lab_int = self._codes_to_ints(np.array(indices, dtype="uint64"))

        return self._base.get_loc(self, lab_int)

    def get_indexer_non_unique(self, target: np.ndarray) -> np.ndarray:
        indexer = self._base.get_indexer_non_unique(self, target)

        return indexer

    def __contains__(self, val: object) -> bool:
        # We assume before we get here:
        #  - val is hashable
        # Default __contains__ looks in the underlying mapping, which in this
        # case only contains integer representations.
        try:
            self.get_loc(val)
            return True
        except (KeyError, TypeError, ValueError):
            return False


# Generated from template.
include "index_class_helper.pxi"


cdef class BoolEngine(UInt8Engine):
    cdef _check_type(self, object val):
        if not util.is_bool_object(val):
            raise KeyError(val)
        return <uint8_t>val


cdef class MaskedBoolEngine(MaskedUInt8Engine):
    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_bool_object(val):
            raise KeyError(val)
        return <uint8_t>val


@cython.internal
@cython.freelist(32)
cdef class SharedEngine:
    cdef readonly:
        object values  # ExtensionArray
        bint over_size_threshold

    cdef:
        bint unique, monotonic_inc, monotonic_dec
        bint need_monotonic_check, need_unique_check

    def __contains__(self, val: object) -> bool:
        # We assume before we get here:
        #  - val is hashable
        try:
            self.get_loc(val)
            return True
        except KeyError:
            return False

    def clear_mapping(self):
        # for compat with IndexEngine
        pass

    cpdef _update_from_sliced(self, ExtensionEngine other, reverse: bool):
        self.unique = other.unique
        self.need_unique_check = other.need_unique_check
        if not other.need_monotonic_check and (
                other.is_monotonic_increasing or other.is_monotonic_decreasing):
            self.need_monotonic_check = other.need_monotonic_check
            # reverse=True means the index has been reversed
            self.monotonic_inc = other.monotonic_dec if reverse else other.monotonic_inc
            self.monotonic_dec = other.monotonic_inc if reverse else other.monotonic_dec

    @property
    def is_unique(self) -> bool:
        if self.need_unique_check:
            arr = self.values.unique()
            self.unique = len(arr) == len(self.values)

            self.need_unique_check = False
        return self.unique

    cdef _do_monotonic_check(self):
        raise NotImplementedError

    @property
    def is_monotonic_increasing(self) -> bool:
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_inc == 1

    @property
    def is_monotonic_decreasing(self) -> bool:
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_dec == 1

    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=False)

    def sizeof(self, deep: bool = False) -> int:
        """ return the sizeof our mapping """
        return 0

    def __sizeof__(self) -> int:
        return self.sizeof()

    cdef _check_type(self, object obj):
        raise NotImplementedError

    cpdef get_loc(self, object val):
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t loc

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        self._check_type(val)

        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(val)

            values = self.values

            loc = self._searchsorted_left(val)
            if loc >= len(values):
                raise KeyError(val)
            if values[loc] != val:
                raise KeyError(val)
            return loc

        if not self.unique:
            return self._get_loc_duplicates(val)

        return self._get_loc_duplicates(val)

    cdef _get_loc_duplicates(self, object val):
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t diff

        if self.is_monotonic_increasing:
            values = self.values
            try:
                left = values.searchsorted(val, side="left")
                right = values.searchsorted(val, side="right")
            except TypeError:
                # e.g. GH#29189 get_loc(None) with a Float64Index
                raise KeyError(val)

            diff = right - left
            if diff == 0:
                raise KeyError(val)
            elif diff == 1:
                return left
            else:
                return slice(left, right)

        return self._maybe_get_bool_indexer(val)

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        """
        See ObjectEngine._searchsorted_left.__doc__.
        """
        try:
            loc = self.values.searchsorted(val, side="left")
        except TypeError as err:
            # GH#35788 e.g. val=None with float64 values
            raise KeyError(val)
        return loc

    cdef ndarray _get_bool_indexer(self, val):
        raise NotImplementedError

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer

        indexer = self._get_bool_indexer(val)
        return _unpack_bool_indexer(indexer, val)

    def get_indexer(self, values) -> np.ndarray:
        # values : type(self.values)
        # Note: we only get here with self.is_unique
        cdef:
            Py_ssize_t i, N = len(values)

        res = np.empty(N, dtype=np.intp)

        for i in range(N):
            val = values[i]
            try:
                loc = self.get_loc(val)
                # Because we are unique, loc should always be an integer
            except KeyError:
                loc = -1
            else:
                assert util.is_integer_object(loc), (loc, val)
            res[i] = loc

        return res

    def get_indexer_non_unique(self, targets):
        """
        Return an indexer suitable for taking from a non unique index
        return the labels in the same order as the target
        and a missing indexer into the targets (which correspond
        to the -1 indices in the results
        Parameters
        ----------
        targets : type(self.values)
        Returns
        -------
        indexer : np.ndarray[np.intp]
        missing : np.ndarray[np.intp]
        """
        cdef:
            Py_ssize_t i, N = len(targets)

        indexer = []
        missing = []

        # See also IntervalIndex.get_indexer_pointwise
        for i in range(N):
            val = targets[i]

            try:
                locs = self.get_loc(val)
            except KeyError:
                locs = np.array([-1], dtype=np.intp)
                missing.append(i)
            else:
                if isinstance(locs, slice):
                    # Only needed for get_indexer_non_unique
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype=np.intp)
                elif util.is_integer_object(locs):
                    locs = np.array([locs], dtype=np.intp)
                else:
                    assert locs.dtype.kind == "b"
                    locs = locs.nonzero()[0]

            indexer.append(locs)

        try:
            indexer = np.concatenate(indexer, dtype=np.intp)
        except TypeError:
            # numpy<1.20 doesn't accept dtype keyword
            indexer = np.concatenate(indexer).astype(np.intp, copy=False)
        missing = np.array(missing, dtype=np.intp)

        return indexer, missing


cdef class ExtensionEngine(SharedEngine):
    def __init__(self, values: "ExtensionArray"):
        self.values = values

        self.over_size_threshold = len(values) >= _SIZE_CUTOFF
        self.need_unique_check = True
        self.need_monotonic_check = True
        self.need_unique_check = True

    cdef _do_monotonic_check(self):
        cdef:
            bint is_unique

        values = self.values
        if values._hasna:
            self.monotonic_inc = 0
            self.monotonic_dec = 0

            nunique = len(values.unique())
            self.unique = nunique == len(values)
            self.need_unique_check = 0
            return

        try:
            ranks = values._rank()

        except TypeError:
            self.monotonic_inc = 0
            self.monotonic_dec = 0
            is_unique = 0
        else:
            self.monotonic_inc, self.monotonic_dec, is_unique = \
                self._call_monotonic(ranks)

        self.need_monotonic_check = 0

        # we can only be sure of uniqueness if is_unique=1
        if is_unique:
            self.unique = 1
            self.need_unique_check = 0

    cdef ndarray _get_bool_indexer(self, val):
        if checknull(val):
            return self.values.isna()

        try:
            return self.values == val
        except TypeError:
            # e.g. if __eq__ returns a BooleanArray instead of ndarray[bool]
            try:
                return (self.values == val).to_numpy(dtype=bool, na_value=False)
            except (TypeError, AttributeError) as err:
                # e.g. (self.values == val) returned a bool
                #  see test_get_loc_generator[string[pyarrow]]
                # e.g. self.value == val raises TypeError bc generator has no len
                #  see test_get_loc_generator[string[python]]
                raise KeyError from err

    cdef _check_type(self, object val):
        hash(val)


cdef class MaskedIndexEngine(IndexEngine):
    def __init__(self, object values):
        super().__init__(self._get_data(values))
        self.mask = self._get_mask(values)

    def _get_data(self, object values) -> np.ndarray:
        if hasattr(values, "_mask"):
            return values._data
        # We are an ArrowExtensionArray
        # Set 1 as na_value to avoid ending up with NA and an object array
        # TODO: Remove when arrow engine is implemented
        return values.to_numpy(na_value=1, dtype=values.dtype.numpy_dtype)

    def _get_mask(self, object values) -> np.ndarray:
        if hasattr(values, "_mask"):
            return values._mask
        # We are an ArrowExtensionArray
        return values.isna()

    def get_indexer(self, object values) -> np.ndarray:
        self._ensure_mapping_populated()
        return self.mapping.lookup(self._get_data(values), self._get_mask(values))

    def get_indexer_non_unique(self, object targets):
        """
        Return an indexer suitable for taking from a non unique index
        return the labels in the same order as the target
        and a missing indexer into the targets (which correspond
        to the -1 indices in the results

        Returns
        -------
        indexer : np.ndarray[np.intp]
        missing : np.ndarray[np.intp]
        """
        # TODO: Unify with parent class
        cdef:
            ndarray values, mask, target_vals, target_mask
            ndarray[intp_t] result, missing
            set stargets
            list na_pos
            dict d = {}
            object val
            Py_ssize_t count = 0, count_missing = 0
            Py_ssize_t i, j, n, n_t, n_alloc, start, end, na_idx

        target_vals = self._get_data(targets)
        target_mask = self._get_mask(targets)

        values = self.values
        assert not values.dtype == object  # go through object path instead

        mask = self.mask
        stargets = set(target_vals[~target_mask])

        n = len(values)
        n_t = len(target_vals)
        if n > 10_000:
            n_alloc = 10_000
        else:
            n_alloc = n

        result = np.empty(n_alloc, dtype=np.intp)
        missing = np.empty(n_t, dtype=np.intp)

        # map each starget to its position in the index
        if (
                stargets and
                len(stargets) < 5 and
                not np.any(target_mask) and
                self.is_monotonic_increasing
        ):
            # if there are few enough stargets and the index is monotonically
            # increasing, then use binary search for each starget
            for starget in stargets:
                start = values.searchsorted(starget, side="left")
                end = values.searchsorted(starget, side="right")
                if start != end:
                    d[starget] = list(range(start, end))

            stargets = set()

        if stargets:
            # otherwise, map by iterating through all items in the index

            na_pos = []

            for i in range(n):
                val = values[i]

                if mask[i]:
                    na_pos.append(i)

                else:
                    if val in stargets:
                        if val not in d:
                            d[val] = []
                        d[val].append(i)

        for i in range(n_t):
            val = target_vals[i]

            if target_mask[i]:
                if na_pos:
                    for na_idx in na_pos:
                        # realloc if needed
                        if count >= n_alloc:
                            n_alloc += 10_000
                            result = np.resize(result, n_alloc)

                        result[count] = na_idx
                        count += 1
                    continue

            elif val in d:
                # found
                key = val

                for j in d[key]:

                    # realloc if needed
                    if count >= n_alloc:
                        n_alloc += 10_000
                        result = np.resize(result, n_alloc)

                    result[count] = j
                    count += 1
                continue

            # value not found
            if count >= n_alloc:
                n_alloc += 10_000
                result = np.resize(result, n_alloc)
            result[count] = -1
            count += 1
            missing[count_missing] = i
            count_missing += 1

        return result[0:count], missing[0:count_missing]
