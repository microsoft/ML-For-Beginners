# This file is part of Patsy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Exhaustive end-to-end tests of the top-level API.

import sys
import __future__
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.categorical import C
from patsy.contrasts import Helmert
from patsy.user_util import balanced, LookupFactor
from patsy.build import (design_matrix_builders,
                         build_design_matrices)
from patsy.highlevel import *
from patsy.util import (have_pandas,
                        have_pandas_categorical,
                        have_pandas_categorical_dtype,
                        pandas_Categorical_from_codes)
from patsy.origin import Origin

if have_pandas:
    import pandas

def check_result(expect_full_designs, lhs, rhs, data,
                 expected_rhs_values, expected_rhs_names,
                 expected_lhs_values, expected_lhs_names): # pragma: no cover
    assert np.allclose(rhs, expected_rhs_values)
    assert rhs.design_info.column_names == expected_rhs_names
    if lhs is not None:
        assert np.allclose(lhs, expected_lhs_values)
        assert lhs.design_info.column_names == expected_lhs_names
    else:
        assert expected_lhs_values is None
        assert expected_lhs_names is None

    if expect_full_designs:
        if lhs is None:
            new_rhs, = build_design_matrices([rhs.design_info], data)
        else:
            new_lhs, new_rhs = build_design_matrices([lhs.design_info,
                                                      rhs.design_info],
                                                     data)
            assert np.allclose(new_lhs, lhs)
            assert new_lhs.design_info.column_names == expected_lhs_names
        assert np.allclose(new_rhs, rhs)
        assert new_rhs.design_info.column_names == expected_rhs_names
    else:
        assert rhs.design_info.terms is None
        assert lhs is None or lhs.design_info.terms is None

def dmatrix_pandas(formula_like, data={}, depth=0, return_type="matrix"):
    return_type = "dataframe"
    if isinstance(depth, int):
        depth += 1
    return dmatrix(formula_like, data, depth, return_type=return_type)

def dmatrices_pandas(formula_like, data={}, depth=0, return_type="matrix"):
    return_type = "dataframe"
    if isinstance(depth, int):
        depth += 1
    return dmatrices(formula_like, data, depth, return_type=return_type)

def t(formula_like, data, depth,
      expect_full_designs,
      expected_rhs_values, expected_rhs_names,
      expected_lhs_values=None, expected_lhs_names=None): # pragma: no cover
    if isinstance(depth, int):
        depth += 1
    def data_iter_maker():
        return iter([data])
    if (isinstance(formula_like, six.string_types + (ModelDesc, DesignInfo))
        or (isinstance(formula_like, tuple)
            and isinstance(formula_like[0], DesignInfo))
        or hasattr(formula_like, "__patsy_get_model_desc__")):
        if expected_lhs_values is None:
            builder = incr_dbuilder(formula_like, data_iter_maker, depth)
            lhs = None
            (rhs,) = build_design_matrices([builder], data)
        else:
            builders = incr_dbuilders(formula_like, data_iter_maker, depth)
            lhs, rhs = build_design_matrices(builders, data)
        check_result(expect_full_designs, lhs, rhs, data,
                     expected_rhs_values, expected_rhs_names,
                     expected_lhs_values, expected_lhs_names)
    else:
        pytest.raises(PatsyError, incr_dbuilders,
                      formula_like, data_iter_maker)
        pytest.raises(PatsyError, incr_dbuilder,
                      formula_like, data_iter_maker)
    one_mat_fs = [dmatrix]
    two_mat_fs = [dmatrices]
    if have_pandas:
        one_mat_fs.append(dmatrix_pandas)
        two_mat_fs.append(dmatrices_pandas)
    if expected_lhs_values is None:
        for f in one_mat_fs:
            rhs = f(formula_like, data, depth)
            check_result(expect_full_designs, None, rhs, data,
                         expected_rhs_values, expected_rhs_names,
                         expected_lhs_values, expected_lhs_names)

        # We inline assert_raises here to avoid complications with the
        # depth argument.
        for f in two_mat_fs:
            try:
                f(formula_like, data, depth)
            except PatsyError:
                pass
            else:
                raise AssertionError
    else:
        for f in one_mat_fs:
            try:
                f(formula_like, data, depth)
            except PatsyError:
                pass
            else:
                raise AssertionError

        for f in two_mat_fs:
            (lhs, rhs) = f(formula_like, data, depth)
            check_result(expect_full_designs, lhs, rhs, data,
                         expected_rhs_values, expected_rhs_names,
                         expected_lhs_values, expected_lhs_names)

def t_invalid(formula_like, data, depth, exc=PatsyError): # pragma: no cover
    if isinstance(depth, int):
        depth += 1
    fs = [dmatrix, dmatrices]
    if have_pandas:
        fs += [dmatrix_pandas, dmatrices_pandas]
    for f in fs:
        try:
            f(formula_like, data, depth)
        except exc:
            pass
        else:
            raise AssertionError

# Exercise all the different calling conventions for the high-level API
def test_formula_likes():
    # Plain array-like, rhs only
    t([[1, 2, 3], [4, 5, 6]], {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    t((None, [[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    t(np.asarray([[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    t((None, np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    dm = DesignMatrix([[1, 2, 3], [4, 5, 6]], default_column_prefix="foo")
    t(dm, {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["foo0", "foo1", "foo2"])
    t((None, dm), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["foo0", "foo1", "foo2"])

    # Plain array-likes, lhs and rhs
    t(([1, 2], [[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    t(([[1], [2]], [[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    t((np.asarray([1, 2]), np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    t((np.asarray([[1], [2]]), np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    x_dm = DesignMatrix([[1, 2, 3], [4, 5, 6]], default_column_prefix="foo")
    y_dm = DesignMatrix([1, 2], default_column_prefix="bar")
    t((y_dm, x_dm), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["foo0", "foo1", "foo2"],
      [[1], [2]], ["bar0"])
    # number of rows must match
    t_invalid(([1, 2, 3], [[1, 2, 3], [4, 5, 6]]), {}, 0)

    # tuples must have the right size
    t_invalid(([[1, 2, 3]],), {}, 0)
    t_invalid(([[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]), {}, 0)

    # plain Series and DataFrames
    if have_pandas:
        # Names are extracted
        t(pandas.DataFrame({"x": [1, 2, 3]}), {}, 0,
          False,
          [[1], [2], [3]], ["x"])
        t(pandas.Series([1, 2, 3], name="asdf"), {}, 0,
          False,
          [[1], [2], [3]], ["asdf"])
        t((pandas.DataFrame({"y": [4, 5, 6]}),
           pandas.DataFrame({"x": [1, 2, 3]})), {}, 0,
          False,
          [[1], [2], [3]], ["x"],
          [[4], [5], [6]], ["y"])
        t((pandas.Series([4, 5, 6], name="y"),
           pandas.Series([1, 2, 3], name="x")), {}, 0,
          False,
          [[1], [2], [3]], ["x"],
          [[4], [5], [6]], ["y"])
        # Or invented
        t((pandas.DataFrame([[4, 5, 6]]),
           pandas.DataFrame([[1, 2, 3]], columns=[7, 8, 9])), {}, 0,
          False,
          [[1, 2, 3]], ["x7", "x8", "x9"],
          [[4, 5, 6]], ["y0", "y1", "y2"])
        t(pandas.Series([1, 2, 3]), {}, 0,
          False,
          [[1], [2], [3]], ["x0"])
        # indices must match
        t_invalid((pandas.DataFrame([[1]], index=[1]),
                   pandas.DataFrame([[1]], index=[2])),
                  {}, 0)

    # Foreign ModelDesc factories
    class ForeignModelSource(object):
        def __patsy_get_model_desc__(self, data):
            return ModelDesc([Term([LookupFactor("Y")])],
                             [Term([LookupFactor("X")])])
    foreign_model = ForeignModelSource()
    t(foreign_model,
      {"Y": [1, 2],
       "X": [[1, 2], [3, 4]]},
      0,
      True,
      [[1, 2], [3, 4]], ["X[0]", "X[1]"],
      [[1], [2]], ["Y"])
    class BadForeignModelSource(object):
        def __patsy_get_model_desc__(self, data):
            return data
    t_invalid(BadForeignModelSource(), {}, 0)

    # string formulas
    t("y ~ x", {"y": [1, 2], "x": [3, 4]}, 0,
      True,
      [[1, 3], [1, 4]], ["Intercept", "x"],
      [[1], [2]], ["y"])
    t("~ x", {"y": [1, 2], "x": [3, 4]}, 0,
      True,
      [[1, 3], [1, 4]], ["Intercept", "x"])
    t("x + y", {"y": [1, 2], "x": [3, 4]}, 0,
      True,
      [[1, 3, 1], [1, 4, 2]], ["Intercept", "x", "y"])

    # unicode objects on py2 (must be ascii only)
    if not six.PY3:
        # ascii is fine
        t(unicode("y ~ x"),
          {"y": [1, 2], "x": [3, 4]}, 0,
          True,
          [[1, 3], [1, 4]], ["Intercept", "x"],
          [[1], [2]], ["y"])
        # non-ascii is not (even if this would be valid on py3 with its less
        # restrict variable naming rules)
        eacute = "\xc3\xa9".decode("utf-8")
        assert isinstance(eacute, unicode)
        pytest.raises(PatsyError, dmatrix, eacute, data={eacute: [1, 2]})

    # ModelDesc
    desc = ModelDesc([], [Term([LookupFactor("x")])])
    t(desc, {"x": [1.5, 2.5, 3.5]}, 0,
      True,
      [[1.5], [2.5], [3.5]], ["x"])
    desc = ModelDesc([], [Term([]), Term([LookupFactor("x")])])
    t(desc, {"x": [1.5, 2.5, 3.5]}, 0,
      True,
      [[1, 1.5], [1, 2.5], [1, 3.5]], ["Intercept", "x"])
    desc = ModelDesc([Term([LookupFactor("y")])],
                     [Term([]), Term([LookupFactor("x")])])
    t(desc, {"x": [1.5, 2.5, 3.5], "y": [10, 20, 30]}, 0,
      True,
      [[1, 1.5], [1, 2.5], [1, 3.5]], ["Intercept", "x"],
      [[10], [20], [30]], ["y"])

    # builders
    termlists = ([],
                 [Term([LookupFactor("x")])],
                 [Term([]), Term([LookupFactor("x")])],
                 )
    builders = design_matrix_builders(termlists,
                                      lambda: iter([{"x": [1, 2, 3]}]),
                                      eval_env=0)
    # twople but with no LHS
    t((builders[0], builders[2]), {"x": [10, 20, 30]}, 0,
      True,
      [[1, 10], [1, 20], [1, 30]], ["Intercept", "x"])
    # single DesignInfo
    t(builders[2], {"x": [10, 20, 30]}, 0,
      True,
      [[1, 10], [1, 20], [1, 30]], ["Intercept", "x"])
    # twople with LHS
    t((builders[1], builders[2]), {"x": [10, 20, 30]}, 0,
      True,
      [[1, 10], [1, 20], [1, 30]], ["Intercept", "x"],
      [[10], [20], [30]], ["x"])

    # check depth arguments
    x_in_env = [1, 2, 3]
    t("~ x_in_env", {}, 0,
      True,
      [[1, 1], [1, 2], [1, 3]], ["Intercept", "x_in_env"])
    t("~ x_in_env", {"x_in_env": [10, 20, 30]}, 0,
      True,
      [[1, 10], [1, 20], [1, 30]], ["Intercept", "x_in_env"])
    # Trying to pull x_in_env out of our *caller* shouldn't work.
    t_invalid("~ x_in_env", {}, 1, exc=(NameError, PatsyError))
    # But then again it should, if called from one down on the stack:
    def check_nested_call():
        x_in_env = "asdf"
        t("~ x_in_env", {}, 1,
          True,
          [[1, 1], [1, 2], [1, 3]], ["Intercept", "x_in_env"])
    check_nested_call()
    # passing in an explicit EvalEnvironment also works:
    e = EvalEnvironment.capture(1)
    t_invalid("~ x_in_env", {}, e, exc=(NameError, PatsyError))
    e = EvalEnvironment.capture(0)
    def check_nested_call_2():
        x_in_env = "asdf"
        t("~ x_in_env", {}, e,
          True,
          [[1, 1], [1, 2], [1, 3]], ["Intercept", "x_in_env"])
    check_nested_call_2()

def test_return_pandas():
    if not have_pandas:
        return
    # basic check of pulling a Series out of the environment
    s1 = pandas.Series([1, 2, 3], name="AA", index=[10, 20, 30])
    s2 = pandas.Series([4, 5, 6], name="BB", index=[10, 20, 30])
    df1 = dmatrix("s1", return_type="dataframe")
    assert np.allclose(df1, [[1, 1], [1, 2], [1, 3]])
    assert np.array_equal(df1.columns, ["Intercept", "s1"])
    assert df1.design_info.column_names == ["Intercept", "s1"]
    assert np.array_equal(df1.index, [10, 20, 30])
    df2, df3 = dmatrices("s2 ~ s1", return_type="dataframe")
    assert np.allclose(df2, [[4], [5], [6]])
    assert np.array_equal(df2.columns, ["s2"])
    assert df2.design_info.column_names == ["s2"]
    assert np.array_equal(df2.index, [10, 20, 30])
    assert np.allclose(df3, [[1, 1], [1, 2], [1, 3]])
    assert np.array_equal(df3.columns, ["Intercept", "s1"])
    assert df3.design_info.column_names == ["Intercept", "s1"]
    assert np.array_equal(df3.index, [10, 20, 30])
    # indices are preserved if pandas is passed in directly
    df4 = dmatrix(s1, return_type="dataframe")
    assert np.allclose(df4, [[1], [2], [3]])
    assert np.array_equal(df4.columns, ["AA"])
    assert df4.design_info.column_names == ["AA"]
    assert np.array_equal(df4.index, [10, 20, 30])
    df5, df6 = dmatrices((s2, s1), return_type="dataframe")
    assert np.allclose(df5, [[4], [5], [6]])
    assert np.array_equal(df5.columns, ["BB"])
    assert df5.design_info.column_names == ["BB"]
    assert np.array_equal(df5.index, [10, 20, 30])
    assert np.allclose(df6, [[1], [2], [3]])
    assert np.array_equal(df6.columns, ["AA"])
    assert df6.design_info.column_names == ["AA"]
    assert np.array_equal(df6.index, [10, 20, 30])
    # Both combinations of with-index and without-index
    df7, df8 = dmatrices((s1, [10, 11, 12]), return_type="dataframe")
    assert np.array_equal(df7.index, s1.index)
    assert np.array_equal(df8.index, s1.index)
    df9, df10 = dmatrices(([10, 11, 12], s1), return_type="dataframe")
    assert np.array_equal(df9.index, s1.index)
    assert np.array_equal(df10.index, s1.index)
    # pandas must be available
    import patsy.highlevel
    had_pandas = patsy.highlevel.have_pandas
    try:
        patsy.highlevel.have_pandas = False
        pytest.raises(PatsyError,
                      dmatrix, "x", {"x": [1]}, 0, return_type="dataframe")
        pytest.raises(PatsyError,
                      dmatrices, "y ~ x", {"x": [1], "y": [2]}, 0,
                      return_type="dataframe")
    finally:
        patsy.highlevel.have_pandas = had_pandas

def test_term_info():
    data = balanced(a=2, b=2)
    rhs = dmatrix("a:b", data)
    assert rhs.design_info.column_names == ["Intercept", "b[T.b2]",
                                            "a[T.a2]:b[b1]", "a[T.a2]:b[b2]"]
    assert rhs.design_info.term_names == ["Intercept", "a:b"]
    assert len(rhs.design_info.terms) == 2
    assert rhs.design_info.terms[0] == INTERCEPT

def test_data_types():
    data = {"a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": np.asarray([1, 2, 3], dtype=np.float32),
            "d": [True, False, True],
            "e": ["foo", "bar", "baz"],
            "f": C([1, 2, 3]),
            "g": C(["foo", "bar", "baz"]),
            "h": np.array(["foo", 1, (1, "hi")], dtype=object),
            }
    t("~ 0 + a", data, 0, True,
      [[1], [2], [3]], ["a"])
    t("~ 0 + b", data, 0, True,
      [[1], [2], [3]], ["b"])
    t("~ 0 + c", data, 0, True,
      [[1], [2], [3]], ["c"])
    t("~ 0 + d", data, 0, True,
      [[0, 1], [1, 0], [0, 1]], ["d[False]", "d[True]"])
    t("~ 0 + e", data, 0, True,
      [[0, 0, 1], [1, 0, 0], [0, 1, 0]], ["e[bar]", "e[baz]", "e[foo]"])
    t("~ 0 + f", data, 0, True,
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["f[1]", "f[2]", "f[3]"])
    t("~ 0 + g", data, 0, True,
      [[0, 0, 1], [1, 0, 0], [0, 1, 0]], ["g[bar]", "g[baz]", "g[foo]"])
    # This depends on Python's sorting behavior:
    t("~ 0 + h", data, 0, True,
      [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
      ["h[1]", "h[foo]", "h[(1, 'hi')]"])

def test_categorical():
    data = balanced(a=2, b=2)
    # There are more exhaustive tests for all the different coding options in
    # test_build; let's just make sure that C() and stuff works.
    t("~ C(a)", data, 0,
      True,
      [[1, 0], [1, 0], [1, 1], [1, 1]], ["Intercept", "C(a)[T.a2]"])
    t("~ C(a, levels=['a2', 'a1'])", data, 0,
      True,
      [[1, 1], [1, 1], [1, 0], [1, 0]],
      ["Intercept", "C(a, levels=['a2', 'a1'])[T.a1]"])
    t("~ C(a, Treatment(reference=-1))", data, 0,
      True,
      [[1, 1], [1, 1], [1, 0], [1, 0]],
      ["Intercept", "C(a, Treatment(reference=-1))[T.a1]"])

    # Different interactions
    t("a*b", data, 0,
      True,
      [[1, 0, 0, 0],
       [1, 0, 1, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 1]],
      ["Intercept", "a[T.a2]", "b[T.b2]", "a[T.a2]:b[T.b2]"])
    t("0 + a:b", data, 0,
      True,
      [[1, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1]],
      ["a[a1]:b[b1]", "a[a2]:b[b1]", "a[a1]:b[b2]", "a[a2]:b[b2]"])
    t("1 + a + a:b", data, 0,
      True,
      [[1, 0, 0, 0],
       [1, 0, 1, 0],
       [1, 1, 0, 0],
       [1, 1, 0, 1]],
      ["Intercept", "a[T.a2]", "a[a1]:b[T.b2]", "a[a2]:b[T.b2]"])

    # Changing contrast with C()
    data["a"] = C(data["a"], Helmert)
    t("a", data, 0,
      True,
      [[1, -1], [1, -1], [1, 1], [1, 1]], ["Intercept", "a[H.a2]"])
    t("C(a, Treatment)", data, 0,
      True,
      [[1, 0], [1, 0], [1, 1], [1, 1]], ["Intercept", "C(a, Treatment)[T.a2]"])
    # That didn't affect the original object
    t("a", data, 0,
      True,
      [[1, -1], [1, -1], [1, 1], [1, 1]], ["Intercept", "a[H.a2]"])

def test_builtins():
    data = {"x": [1, 2, 3],
            "y": [4, 5, 6],
            "a b c": [10, 20, 30]}
    t("0 + I(x + y)", data, 0,
      True,
      [[1], [2], [3], [4], [5], [6]], ["I(x + y)"])
    t("Q('a b c')", data, 0,
      True,
      [[1, 10], [1, 20], [1, 30]], ["Intercept", "Q('a b c')"])
    t("center(x)", data, 0,
      True,
      [[1, -1], [1, 0], [1, 1]], ["Intercept", "center(x)"])

def test_incremental():
    # incr_dbuilder(s)
    # stateful transformations
    datas = [
        {"a": ["a2", "a2", "a2"],
         "x": [1, 2, 3]},
        {"a": ["a2", "a2", "a1"],
         "x": [4, 5, 6]},
        ]
    x = np.asarray([1, 2, 3, 4, 5, 6])
    sin_center_x = np.sin(x - np.mean(x))
    x_col = sin_center_x - np.mean(sin_center_x)
    def data_iter_maker():
        return iter(datas)
    builders = incr_dbuilders("1 ~ a + center(np.sin(center(x)))",
                              data_iter_maker)
    lhs, rhs = build_design_matrices(builders, datas[1])
    assert lhs.design_info.column_names == ["Intercept"]
    assert rhs.design_info.column_names == ["Intercept",
                                            "a[T.a2]",
                                            "center(np.sin(center(x)))"]
    assert np.allclose(lhs, [[1], [1], [1]])
    assert np.allclose(rhs, np.column_stack(([1, 1, 1],
                                             [1, 1, 0],
                                             x_col[3:])))

    builder = incr_dbuilder("~ a + center(np.sin(center(x)))",
                            data_iter_maker)
    (rhs,) = build_design_matrices([builder], datas[1])
    assert rhs.design_info.column_names == ["Intercept",
                                            "a[T.a2]",
                                            "center(np.sin(center(x)))"]
    assert np.allclose(lhs, [[1], [1], [1]])
    assert np.allclose(rhs, np.column_stack(([1, 1, 1],
                                             [1, 1, 0],
                                             x_col[3:])))

    pytest.raises(PatsyError, incr_dbuilder, "x ~ x", data_iter_maker)
    pytest.raises(PatsyError, incr_dbuilders, "x", data_iter_maker)

def test_env_transform():
    t("~ np.sin(x)", {"x": [1, 2, 3]}, 0,
      True,
      [[1, np.sin(1)], [1, np.sin(2)], [1, np.sin(3)]],
      ["Intercept", "np.sin(x)"])

# Term ordering:
#   1) all 0-order no-numeric
#   2) all 1st-order no-numeric
#   3) all 2nd-order no-numeric
#   4) ...
#   5) all 0-order with the first numeric interaction encountered
#   6) all 1st-order with the first numeric interaction encountered
#   7) ...
#   8) all 0-order with the second numeric interaction encountered
#   9) ...
def test_term_order():
    data = balanced(a=2, b=2)
    data["x1"] = np.linspace(0, 1, 4)
    data["x2"] = data["x1"] ** 2

    def t_terms(formula, order):
        m = dmatrix(formula, data)
        assert m.design_info.term_names == order

    t_terms("a + b + x1 + x2", ["Intercept", "a", "b", "x1", "x2"])
    t_terms("b + a + x2 + x1", ["Intercept", "b", "a", "x2", "x1"])
    t_terms("0 + x1 + a + x2 + b + 1", ["Intercept", "a", "b", "x1", "x2"])
    t_terms("0 + a:b + a + b + 1", ["Intercept", "a", "b", "a:b"])
    t_terms("a + a:x1 + x2 + x1 + b",
            ["Intercept", "a", "b", "x1", "a:x1", "x2"])
    t_terms("0 + a:x1:x2 + a + x2:x1:b + x2 + x1 + a:x1 + x1:x2 + x1:a:x2:a:b",
            ["a",
             "x1:x2", "a:x1:x2", "x2:x1:b", "x1:a:x2:b",
             "x2",
             "x1",
             "a:x1"])

def _check_division(expect_true_division): # pragma: no cover
    # We evaluate the formula "I(x / y)" in our *caller's* scope, so the
    # result depends on whether our caller has done 'from __future__ import
    # division'.
    data = {"x": 5, "y": 2}
    m = dmatrix("0 + I(x / y)", data, 1)
    if expect_true_division:
        assert np.allclose(m, [[2.5]])
    else:
        assert np.allclose(m, [[2]])

def test_future():
    if __future__.division.getMandatoryRelease() < sys.version_info:
        # This is Python 3, where division is already default
        return
    # no __future__.division in this module's scope
    _check_division(False)
    # create an execution context where __future__.division is in effect
    exec ("from __future__ import division\n"
          "_check_division(True)\n")

def test_multicolumn():
    data = {
        "a": ["a1", "a2"],
        "X": [[1, 2], [3, 4]],
        "Y": [[1, 3], [2, 4]],
        }
    t("X*Y", data, 0,
      True,
      [[1, 1, 2, 1, 3, 1 * 1, 2 * 1, 1 * 3, 2 * 3],
       [1, 3, 4, 2, 4, 3 * 2, 4 * 2, 3 * 4, 4 * 4]],
      ["Intercept", "X[0]", "X[1]", "Y[0]", "Y[1]",
       "X[0]:Y[0]", "X[1]:Y[0]", "X[0]:Y[1]", "X[1]:Y[1]"])
    t("a:X + Y", data, 0,
      True,
      [[1, 1, 0, 2, 0, 1, 3],
       [1, 0, 3, 0, 4, 2, 4]],
      ["Intercept",
       "a[a1]:X[0]", "a[a2]:X[0]", "a[a1]:X[1]", "a[a2]:X[1]",
       "Y[0]", "Y[1]"])

def test_dmatrix_dmatrices_no_data():
    x = [1, 2, 3]
    y = [4, 5, 6]
    assert np.allclose(dmatrix("x"), [[1, 1], [1, 2], [1, 3]])
    lhs, rhs = dmatrices("y ~ x")
    assert np.allclose(lhs, [[4], [5], [6]])
    assert np.allclose(rhs, [[1, 1], [1, 2], [1, 3]])

def test_designinfo_describe():
    lhs, rhs = dmatrices("y ~ x + a", {"y": [1, 2, 3],
                                       "x": [4, 5, 6],
                                       "a": ["a1", "a2", "a3"]})
    assert lhs.design_info.describe() == "y"
    assert rhs.design_info.describe() == "1 + a + x"

def test_evalfactor_reraise():
    # This will produce a PatsyError, but buried inside the factor evaluation,
    # so the original code has no way to give it an appropriate origin=
    # attribute. EvalFactor should notice this, and add a useful origin:
    def raise_patsy_error(x):
        raise PatsyError("WHEEEEEE")
    formula = "raise_patsy_error(X) + Y"
    try:
        dmatrix(formula, {"X": [1, 2, 3], "Y": [4, 5, 6]})
    except PatsyError as e:
        assert e.origin == Origin(formula, 0, formula.index(" "))
    else:
        assert False
    # This will produce a KeyError, which on Python 3 we can do wrap without
    # destroying the traceback, so we do so. On Python 2 we let the original
    # exception escape.
    try:
        dmatrix("1 + x[1]", {"x": {}})
    except Exception as e:
        if sys.version_info[0] >= 3:
            assert isinstance(e, PatsyError)
            assert e.origin == Origin("1 + x[1]", 4, 8)
        else:
            assert isinstance(e, KeyError)
    else:
        assert False

def test_dmatrix_NA_action():
    data = {"x": [1, 2, 3, np.nan], "y": [np.nan, 20, 30, 40]}

    return_types = ["matrix"]
    if have_pandas:
        return_types.append("dataframe")

    for return_type in return_types:
        mat = dmatrix("x + y", data=data, return_type=return_type)
        assert np.array_equal(mat, [[1, 2, 20],
                                    [1, 3, 30]])
        if return_type == "dataframe":
            assert mat.index.equals(pandas.Index([1, 2]))
        pytest.raises(PatsyError, dmatrix, "x + y", data=data,
                      return_type=return_type,
                      NA_action="raise")

        lmat, rmat = dmatrices("y ~ x", data=data, return_type=return_type)
        assert np.array_equal(lmat, [[20], [30]])
        assert np.array_equal(rmat, [[1, 2], [1, 3]])
        if return_type == "dataframe":
            assert lmat.index.equals(pandas.Index([1, 2]))
            assert rmat.index.equals(pandas.Index([1, 2]))
        pytest.raises(PatsyError,
                      dmatrices, "y ~ x", data=data, return_type=return_type,
                      NA_action="raise")

        # Initial release for the NA handling code had problems with
        # non-data-dependent matrices like "~ 1".
        lmat, rmat = dmatrices("y ~ 1", data=data, return_type=return_type)
        assert np.array_equal(lmat, [[20], [30], [40]])
        assert np.array_equal(rmat, [[1], [1], [1]])
        if return_type == "dataframe":
            assert lmat.index.equals(pandas.Index([1, 2, 3]))
            assert rmat.index.equals(pandas.Index([1, 2, 3]))
        pytest.raises(PatsyError,
                      dmatrices, "y ~ 1", data=data, return_type=return_type,
                      NA_action="raise")

def test_0d_data():
    # Use case from statsmodels/statsmodels#1881
    data_0d = {"x1": 1.1, "x2": 1.2, "a": "a1"}

    for formula, expected in [
            ("x1 + x2", [[1, 1.1, 1.2]]),
            ("C(a, levels=('a1', 'a2')) + x1", [[1, 0, 1.1]]),
            ]:
        mat = dmatrix(formula, data_0d)
        assert np.allclose(mat, expected)

        assert np.allclose(build_design_matrices([mat.design_info],
                                                 data_0d)[0],
                           expected)
        if have_pandas:
            data_series = pandas.Series(data_0d)
            assert np.allclose(dmatrix(formula, data_series), expected)

            assert np.allclose(build_design_matrices([mat.design_info],
                                                     data_series)[0],
                               expected)

def test_env_not_saved_in_builder():
    x_in_env = [1, 2, 3]
    design_matrix = dmatrix("x_in_env", {})

    x_in_env = [10, 20, 30]
    design_matrix2 = dmatrix(design_matrix.design_info, {})

    assert np.allclose(design_matrix, design_matrix2)

def test_C_and_pandas_categorical():
    if not have_pandas_categorical:
        return

    objs = [pandas_Categorical_from_codes([1, 0, 1], ["b", "a"])]
    if have_pandas_categorical_dtype:
        objs.append(pandas.Series(objs[0]))
    for obj in objs:
        d = {"obj": obj}
        assert np.allclose(dmatrix("obj", d),
                           [[1, 1],
                            [1, 0],
                            [1, 1]])

        assert np.allclose(dmatrix("C(obj)", d),
                           [[1, 1],
                            [1, 0],
                            [1, 1]])

        assert np.allclose(dmatrix("C(obj, levels=['b', 'a'])", d),
                           [[1, 1],
                            [1, 0],
                            [1, 1]])

        assert np.allclose(dmatrix("C(obj, levels=['a', 'b'])", d),
                           [[1, 0],
                            [1, 1],
                            [1, 0]])
