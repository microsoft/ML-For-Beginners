# This file is part of Patsy
# Copyright (C) 2011-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

__all__ = ["C", "guess_categorical", "CategoricalSniffer",
           "categorical_to_int"]

# How we handle categorical data: the big picture
# -----------------------------------------------
#
# There is no Python/NumPy standard for how to represent categorical data.
# There is no Python/NumPy standard for how to represent missing data.
#
# Together, these facts mean that when we receive some data object, we must be
# able to heuristically infer what levels it has -- and this process must be
# sensitive to the current missing data handling, because maybe 'None' is a
# level and maybe it is missing data.
#
# We don't know how missing data is represented until we get into the actual
# builder code, so anything which runs before this -- e.g., the 'C()' builtin
# -- cannot actually do *anything* meaningful with the data.
#
# Therefore, C() simply takes some data and arguments, and boxes them all up
# together into an object called (appropriately enough) _CategoricalBox. All
# the actual work of handling the various different sorts of categorical data
# (lists, string arrays, bool arrays, pandas.Categorical, etc.) happens inside
# the builder code, and we just extend this so that it also accepts
# _CategoricalBox objects as yet another categorical type.
#
# Originally this file contained a container type (called 'Categorical'), and
# the various sniffing, conversion, etc., functions were written as methods on
# that type. But we had to get rid of that type, so now this file just
# provides a set of plain old functions which are used by patsy.build to
# handle the different stages of categorical data munging.

import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
                        safe_scalar_isnan,
                        iterable,
                        have_pandas, have_pandas_categorical,
                        have_pandas_categorical_dtype,
                        safe_is_pandas_categorical,
                        pandas_Categorical_from_codes,
                        pandas_Categorical_categories,
                        pandas_Categorical_codes,
                        safe_issubdtype,
                        no_pickling, assert_no_pickling)

if have_pandas:
    import pandas

# Objects of this type will always be treated as categorical, with the
# specified levels and contrast (if given).
class _CategoricalBox(object):
    def __init__(self, data, contrast, levels):
        self.data = data
        self.contrast = contrast
        self.levels = levels

    __getstate__ = no_pickling

def C(data, contrast=None, levels=None):
    """
    Marks some `data` as being categorical, and specifies how to interpret
    it.

    This is used for three reasons:

    * To explicitly mark some data as categorical. For instance, integer data
      is by default treated as numerical. If you have data that is stored
      using an integer type, but where you want patsy to treat each different
      value as a different level of a categorical factor, you can wrap it in a
      call to `C` to accomplish this. E.g., compare::

        dmatrix("a", {"a": [1, 2, 3]})
        dmatrix("C(a)", {"a": [1, 2, 3]})

    * To explicitly set the levels or override the default level ordering for
      categorical data, e.g.::

        dmatrix("C(a, levels=["a2", "a1"])", balanced(a=2))
    * To override the default coding scheme for categorical data. The
      `contrast` argument can be any of:

      * A :class:`ContrastMatrix` object
      * A simple 2d ndarray (which is treated the same as a ContrastMatrix
        object except that you can't specify column names)
      * An object with methods called `code_with_intercept` and
        `code_without_intercept`, like the built-in contrasts
        (:class:`Treatment`, :class:`Diff`, :class:`Poly`, etc.). See
        :ref:`categorical-coding` for more details.
      * A callable that returns one of the above.
    """
    if isinstance(data, _CategoricalBox):
        if contrast is None:
            contrast = data.contrast
        if levels is None:
            levels = data.levels
        data = data.data
    return _CategoricalBox(data, contrast, levels)

def test_C():
    c1 = C("asdf")
    assert isinstance(c1, _CategoricalBox)
    assert c1.data == "asdf"
    assert c1.levels is None
    assert c1.contrast is None
    c2 = C("DATA", "CONTRAST", "LEVELS")
    assert c2.data == "DATA"
    assert c2.contrast == "CONTRAST"
    assert c2.levels == "LEVELS"
    c3 = C(c2, levels="NEW LEVELS")
    assert c3.data == "DATA"
    assert c3.contrast == "CONTRAST"
    assert c3.levels == "NEW LEVELS"
    c4 = C(c2, "NEW CONTRAST")
    assert c4.data == "DATA"
    assert c4.contrast == "NEW CONTRAST"
    assert c4.levels == "LEVELS"

    assert_no_pickling(c4)

def guess_categorical(data):
    if safe_is_pandas_categorical(data):
        return True
    if isinstance(data, _CategoricalBox):
        return True
    data = np.asarray(data)
    if safe_issubdtype(data.dtype, np.number):
        return False
    return True

def test_guess_categorical():
    if have_pandas_categorical:
        c = pandas.Categorical([1, 2, 3])
        assert guess_categorical(c)
        if have_pandas_categorical_dtype:
            assert guess_categorical(pandas.Series(c))
    assert guess_categorical(C([1, 2, 3]))
    assert guess_categorical([True, False])
    assert guess_categorical(["a", "b"])
    assert guess_categorical(["a", "b", np.nan])
    assert guess_categorical(["a", "b", None])
    assert not guess_categorical([1, 2, 3])
    assert not guess_categorical([1, 2, 3, np.nan])
    assert not guess_categorical([1.0, 2.0, 3.0])
    assert not guess_categorical([1.0, 2.0, 3.0, np.nan])

def _categorical_shape_fix(data):
    # helper function
    # data should not be a _CategoricalBox or pandas Categorical or anything
    # -- it should be an actual iterable of data, but which might have the
    # wrong shape.
    if hasattr(data, "ndim") and data.ndim > 1:
        raise PatsyError("categorical data cannot be >1-dimensional")
    # coerce scalars into 1d, which is consistent with what we do for numeric
    # factors. (See statsmodels/statsmodels#1881)
    if (not iterable(data)
        or isinstance(data, (six.text_type, six.binary_type))):
        data = [data]
    return data

class CategoricalSniffer(object):
    def __init__(self, NA_action, origin=None):
        self._NA_action = NA_action
        self._origin = origin
        self._contrast = None
        self._levels = None
        self._level_set = set()

    def levels_contrast(self):
        if self._levels is None:
            levels = list(self._level_set)
            levels.sort(key=SortAnythingKey)
            self._levels = levels
        return tuple(self._levels), self._contrast

    def sniff(self, data):
        if hasattr(data, "contrast"):
            self._contrast = data.contrast
        # returns a bool: are we confident that we found all the levels?
        if isinstance(data, _CategoricalBox):
            if data.levels is not None:
                self._levels = tuple(data.levels)
                return True
            else:
                # unbox and fall through
                data = data.data
        if safe_is_pandas_categorical(data):
            # pandas.Categorical has its own NA detection, so don't try to
            # second-guess it.
            self._levels = tuple(pandas_Categorical_categories(data))
            return True
        # fastpath to avoid doing an item-by-item iteration over boolean
        # arrays, as requested by #44
        if hasattr(data, "dtype") and safe_issubdtype(data.dtype, np.bool_):
            self._level_set = set([True, False])
            return True

        data = _categorical_shape_fix(data)

        for value in data:
            if self._NA_action.is_categorical_NA(value):
                continue
            if value is True or value is False:
                self._level_set.update([True, False])
            else:
                try:
                    self._level_set.add(value)
                except TypeError:
                    raise PatsyError("Error interpreting categorical data: "
                                     "all items must be hashable",
                                     self._origin)
        # If everything we've seen is boolean, assume that everything else
        # would be too. Otherwise we need to keep looking.
        return self._level_set == set([True, False])

    __getstate__ = no_pickling

def test_CategoricalSniffer():
    from patsy.missing import NAAction
    def t(NA_types, datas, exp_finish_fast, exp_levels, exp_contrast=None):
        sniffer = CategoricalSniffer(NAAction(NA_types=NA_types))
        for data in datas:
            done = sniffer.sniff(data)
            if done:
                assert exp_finish_fast
                break
            else:
                assert not exp_finish_fast
        assert sniffer.levels_contrast() == (exp_levels, exp_contrast)

    if have_pandas_categorical:
        # We make sure to test with both boxed and unboxed pandas objects,
        # because we used to have a bug where boxed pandas objects would be
        # treated as categorical, but their levels would be lost...
        preps = [lambda x: x,
                 C]
        if have_pandas_categorical_dtype:
            preps += [pandas.Series,
                      lambda x: C(pandas.Series(x))]
        for prep in preps:
            t([], [prep(pandas.Categorical([1, 2, None]))],
              True, (1, 2))
            # check order preservation
            t([], [prep(pandas_Categorical_from_codes([1, 0], ["a", "b"]))],
              True, ("a", "b"))
            t([], [prep(pandas_Categorical_from_codes([1, 0], ["b", "a"]))],
              True, ("b", "a"))
            # check that if someone sticks a .contrast field onto our object
            obj = prep(pandas.Categorical(["a", "b"]))
            obj.contrast = "CONTRAST"
            t([], [obj], True, ("a", "b"), "CONTRAST")

    t([], [C([1, 2]), C([3, 2])], False, (1, 2, 3))
    # check order preservation
    t([], [C([1, 2], levels=[1, 2, 3]), C([4, 2])], True, (1, 2, 3))
    t([], [C([1, 2], levels=[3, 2, 1]), C([4, 2])], True, (3, 2, 1))

    # do some actual sniffing with NAs in
    t(["None", "NaN"], [C([1, np.nan]), C([10, None])],
      False, (1, 10))
    # But 'None' can be a type if we don't make it represent NA:
    sniffer = CategoricalSniffer(NAAction(NA_types=["NaN"]))
    sniffer.sniff(C([1, np.nan, None]))
    # The level order here is different on py2 and py3 :-( Because there's no
    # consistent way to sort mixed-type values on both py2 and py3. Honestly
    # people probably shouldn't use this, but I don't know how to give a
    # sensible error.
    levels, _ = sniffer.levels_contrast()
    assert set(levels) == set([None, 1])

    # bool special cases
    t(["None", "NaN"], [C([True, np.nan, None])],
      True, (False, True))
    t([], [C([10, 20]), C([False]), C([30, 40])],
      False, (False, True, 10, 20, 30, 40))
    # exercise the fast-path
    t([], [np.asarray([True, False]), ["foo"]],
      True, (False, True))

    # check tuples too
    t(["None", "NaN"], [C([("b", 2), None, ("a", 1), np.nan, ("c", None)])],
      False, (("a", 1), ("b", 2), ("c", None)))

    # contrasts
    t([], [C([10, 20], contrast="FOO")], False, (10, 20), "FOO")

    # no box
    t([], [[10, 30], [20]], False, (10, 20, 30))
    t([], [["b", "a"], ["a"]], False, ("a", "b"))

    # 0d
    t([], ["b"], False, ("b",))

    import pytest

    # unhashable level error:
    sniffer = CategoricalSniffer(NAAction())
    pytest.raises(PatsyError, sniffer.sniff, [{}])

    # >1d is illegal
    pytest.raises(PatsyError, sniffer.sniff, np.asarray([["b"]]))

# returns either a 1d ndarray or a pandas.Series
def categorical_to_int(data, levels, NA_action, origin=None):
    assert isinstance(levels, tuple)
    # In this function, missing values are always mapped to -1

    if safe_is_pandas_categorical(data):
        data_levels_tuple = tuple(pandas_Categorical_categories(data))
        if not data_levels_tuple == levels:
            raise PatsyError("mismatching levels: expected %r, got %r"
                             % (levels, data_levels_tuple), origin)
        # pandas.Categorical also uses -1 to indicate NA, and we don't try to
        # second-guess its NA detection, so we can just pass it back.
        return pandas_Categorical_codes(data)

    if isinstance(data, _CategoricalBox):
        if data.levels is not None and tuple(data.levels) != levels:
            raise PatsyError("mismatching levels: expected %r, got %r"
                             % (levels, tuple(data.levels)), origin)
        data = data.data

    data = _categorical_shape_fix(data)

    try:
        level_to_int = dict(zip(levels, range(len(levels))))
    except TypeError:
        raise PatsyError("Error interpreting categorical data: "
                         "all items must be hashable", origin)

    # fastpath to avoid doing an item-by-item iteration over boolean arrays,
    # as requested by #44
    if hasattr(data, "dtype") and safe_issubdtype(data.dtype, np.bool_):
        if level_to_int[False] == 0 and level_to_int[True] == 1:
            return data.astype(np.int_)
    out = np.empty(len(data), dtype=int)
    for i, value in enumerate(data):
        if NA_action.is_categorical_NA(value):
            out[i] = -1
        else:
            try:
                out[i] = level_to_int[value]
            except KeyError:
                SHOW_LEVELS = 4
                level_strs = []
                if len(levels) <= SHOW_LEVELS:
                    level_strs += [repr(level) for level in levels]
                else:
                    level_strs += [repr(level)
                                   for level in levels[:SHOW_LEVELS//2]]
                    level_strs.append("...")
                    level_strs += [repr(level)
                                   for level in levels[-SHOW_LEVELS//2:]]
                level_str = "[%s]" % (", ".join(level_strs))
                raise PatsyError("Error converting data to categorical: "
                                 "observation with value %r does not match "
                                 "any of the expected levels (expected: %s)"
                                 % (value, level_str), origin)
            except TypeError:
                raise PatsyError("Error converting data to categorical: "
                                 "encountered unhashable value %r"
                                 % (value,), origin)
    if have_pandas and isinstance(data, pandas.Series):
        out = pandas.Series(out, index=data.index)
    return out

def test_categorical_to_int():
    import pytest
    from patsy.missing import NAAction
    if have_pandas:
        s = pandas.Series(["a", "b", "c"], index=[10, 20, 30])
        c_pandas = categorical_to_int(s, ("a", "b", "c"), NAAction())
        assert np.all(c_pandas == [0, 1, 2])
        assert np.all(c_pandas.index == [10, 20, 30])
        # Input must be 1-dimensional
        pytest.raises(PatsyError,
                      categorical_to_int,
                      pandas.DataFrame({10: s}), ("a", "b", "c"), NAAction())
    if have_pandas_categorical:
        constructors = [pandas_Categorical_from_codes]
        if have_pandas_categorical_dtype:
            def Series_from_codes(codes, categories):
                c = pandas_Categorical_from_codes(codes, categories)
                return pandas.Series(c)
            constructors.append(Series_from_codes)
        for con in constructors:
            cat = con([1, 0, -1], ("a", "b"))
            conv = categorical_to_int(cat, ("a", "b"), NAAction())
            assert np.all(conv == [1, 0, -1])
            # Trust pandas NA marking
            cat2 = con([1, 0, -1], ("a", "None"))
            conv2 = categorical_to_int(cat, ("a", "b"),
                                       NAAction(NA_types=["None"]))
            assert np.all(conv2 == [1, 0, -1])
            # But levels must match
            pytest.raises(PatsyError,
                          categorical_to_int,
                          con([1, 0], ("a", "b")),
                          ("a", "c"),
                          NAAction())
            pytest.raises(PatsyError,
                          categorical_to_int,
                          con([1, 0], ("a", "b")),
                          ("b", "a"),
                          NAAction())

    def t(data, levels, expected, NA_action=NAAction()):
        got = categorical_to_int(data, levels, NA_action)
        assert np.array_equal(got, expected)

    t(["a", "b", "a"], ("a", "b"), [0, 1, 0])
    t(np.asarray(["a", "b", "a"]), ("a", "b"), [0, 1, 0])
    t(np.asarray(["a", "b", "a"], dtype=object), ("a", "b"), [0, 1, 0])
    t([0, 1, 2], (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2]), (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2], dtype=float), (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2], dtype=object), (1, 2, 0), [2, 0, 1])
    t(["a", "b", "a"], ("a", "d", "z", "b"), [0, 3, 0])
    t([("a", 1), ("b", 0), ("a", 1)], (("a", 1), ("b", 0)), [0, 1, 0])

    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b", "a"], ("a", "c"), NAAction())

    t(C(["a", "b", "a"]), ("a", "b"), [0, 1, 0])
    t(C(["a", "b", "a"]), ("b", "a"), [1, 0, 1])
    t(C(["a", "b", "a"], levels=["b", "a"]), ("b", "a"), [1, 0, 1])
    # Mismatch between C() levels and expected levels
    pytest.raises(PatsyError, categorical_to_int,
                  C(["a", "b", "a"], levels=["a", "b"]),
                  ("b", "a"), NAAction())

    # ndim == 0 is okay
    t("a", ("a", "b"), [0])
    t("b", ("a", "b"), [1])
    t(True, (False, True), [1])

    # ndim == 2 is disallowed
    pytest.raises(PatsyError, categorical_to_int,
                  np.asarray([["a", "b"], ["b", "a"]]),
                  ("a", "b"), NAAction())

    # levels must be hashable
    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b"], ("a", "b", {}), NAAction())
    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b", {}], ("a", "b"), NAAction())

    t(["b", None, np.nan, "a"], ("a", "b"), [1, -1, -1, 0],
      NAAction(NA_types=["None", "NaN"]))
    t(["b", None, np.nan, "a"], ("a", "b", None), [1, -1, -1, 0],
      NAAction(NA_types=["None", "NaN"]))
    t(["b", None, np.nan, "a"], ("a", "b", None), [1, 2, -1, 0],
      NAAction(NA_types=["NaN"]))

    # Smoke test for the branch that formats the ellipsized list of levels in
    # the error message:
    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b", "q"],
                  ("a", "b", "c", "d", "e", "f", "g", "h"),
                  NAAction())
