from statsmodels.compat.pandas import (
    assert_series_equal,
    assert_frame_equal,
    make_dataframe,
)

import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest

from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit


# FIXME: do not leave commented-out, enable or move/remove
# class TestDates:
#    @classmethod
#    def setup_class(cls):
#        nrows = 10
#        cls.dates_result = cls.dates_results = np.random.random(nrows)
#
#    def test_dates(self):
#        np.testing.assert_equal(data.wrap_output(self.dates_input, 'dates'),
#                                self.dates_result)

class TestArrays:
    @classmethod
    def setup_class(cls):
        cls.endog = np.random.random(10)
        cls.exog = np.c_[np.ones(10), np.random.random((10, 2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_result = cls.col_input = np.random.random(nvars)
        cls.row_result = cls.row_input = np.random.random(nrows)
        cls.cov_result = cls.cov_input = np.random.random((nvars, nvars))
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = 'y'
        cls.row_labels = None

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog)
        np.testing.assert_equal(self.data.exog, self.exog)

    def test_attach(self):
        data = self.data
        # this makes sure what the wrappers need work but not the wrapped
        # results themselves
        np.testing.assert_equal(data.wrap_output(self.col_input, 'columns'),
                                self.col_result)
        np.testing.assert_equal(data.wrap_output(self.row_input, 'rows'),
                                self.row_result)
        np.testing.assert_equal(data.wrap_output(self.cov_input, 'cov'),
                                self.cov_result)

    def test_names(self):
        data = self.data
        np.testing.assert_equal(data.xnames, self.xnames)
        np.testing.assert_equal(data.ynames, self.ynames)

    def test_labels(self):
        # HACK: because numpy main after NA stuff assert_equal fails on
        # pandas indices
        # FIXME: see if this can be de-hacked
        np.testing.assert_(np.all(self.data.row_labels == self.row_labels))


class TestArrays2dEndog(TestArrays):
    @classmethod
    def setup_class(cls):
        super(TestArrays2dEndog, cls).setup_class()
        cls.endog = np.random.random((10, 1))
        cls.exog = np.c_[np.ones(10), np.random.random((10, 2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog)


class TestArrays1dExog(TestArrays):
    @classmethod
    def setup_class(cls):
        super(TestArrays1dExog, cls).setup_class()
        cls.endog = np.random.random(10)
        exog = np.random.random(10)
        cls.data = sm_data.handle_data(cls.endog, exog)
        cls.exog = exog[:, None]
        cls.xnames = ['x1']
        cls.ynames = 'y'

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog.squeeze())


class TestDataFrames(TestArrays):
    @classmethod
    def setup_class(cls):
        cls.endog = pd.DataFrame(np.random.random(10), columns=['y_1'])
        exog = pd.DataFrame(np.random.random((10, 2)),
                            columns=['x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog.columns,
                                      columns=exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index

    def test_orig(self):
        assert_frame_equal(self.data.orig_endog, self.endog)
        assert_frame_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog.values)

    def test_attach(self):
        data = self.data
        # this makes sure what the wrappers need work but not the wrapped
        # results themselves
        assert_series_equal(data.wrap_output(self.col_input, 'columns'),
                            self.col_result)
        assert_series_equal(data.wrap_output(self.row_input, 'rows'),
                            self.row_result)
        assert_frame_equal(data.wrap_output(self.cov_input, 'cov'),
                           self.cov_result)


class TestDataFramesWithMultiIndex(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = pd.DataFrame(np.random.random(10), columns=['y_1'])
        mi = pd.MultiIndex.from_product([['x'], ['1', '2']])
        exog = pd.DataFrame(np.random.random((10, 2)), columns=mi)
        exog_flattened_idx = pd.Index(['const', 'x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input, index=exog_flattened_idx)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input, index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog_flattened_idx,
                                      columns=exog_flattened_idx)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index


class TestLists(TestArrays):
    @classmethod
    def setup_class(cls):
        super(TestLists, cls).setup_class()
        cls.endog = np.random.random(10).tolist()
        cls.exog = np.c_[np.ones(10), np.random.random((10, 2))].tolist()
        cls.data = sm_data.handle_data(cls.endog, cls.exog)


class TestListDataFrame(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = np.random.random(10).tolist()

        exog = pd.DataFrame(np.random.random((10, 2)),
                            columns=['x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog.columns,
                                      columns=exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y'
        cls.row_labels = cls.exog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog)
        np.testing.assert_equal(self.data.exog, self.exog.values)

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        assert_frame_equal(self.data.orig_exog, self.exog)


class TestDataFrameList(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = pd.DataFrame(np.random.random(10), columns=['y_1'])

        exog = pd.DataFrame(np.random.random((10, 2)),
                            columns=['x1', 'x2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog.values.tolist()
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog.columns,
                                      columns=exog.columns)
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.endog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog)

    def test_orig(self):
        assert_frame_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog)


class TestArrayDataFrame(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = np.random.random(10)

        exog = pd.DataFrame(np.random.random((10, 2)),
                            columns=['x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog.columns,
                                      columns=exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y'
        cls.row_labels = cls.exog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog)
        np.testing.assert_equal(self.data.exog, self.exog.values)

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        assert_frame_equal(self.data.orig_exog, self.exog)


class TestDataFrameArray(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = pd.DataFrame(np.random.random(10), columns=['y_1'])

        exog = pd.DataFrame(np.random.random((10, 2)),
                            columns=['x1', 'x2'])  # names mimic defaults
        exog.insert(0, 'const', 1)
        cls.exog = exog.values
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog.columns,
                                      columns=exog.columns)
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.endog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog)

    def test_orig(self):
        assert_frame_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog)


class TestSeriesDataFrame(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = pd.Series(np.random.random(10), name='y_1')

        exog = pd.DataFrame(np.random.random((10, 2)),
                            columns=['x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog.columns,
                                      columns=exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index

    def test_orig(self):
        assert_series_equal(self.data.orig_endog, self.endog)
        assert_frame_equal(self.data.orig_exog, self.exog)


class TestSeriesSeries(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = pd.Series(np.random.random(10), name='y_1')

        exog = pd.Series(np.random.random(10), name='x_1')
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 1
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=[exog.name])
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=[exog.name],
                                      columns=[exog.name])
        cls.xnames = ['x_1']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index

    def test_orig(self):
        assert_series_equal(self.data.orig_endog, self.endog)
        assert_series_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog.values[:, None])


def test_alignment():
    # Fix Issue GH#206
    from statsmodels.datasets.macrodata import load_pandas

    d = load_pandas().data
    # growth rates
    gs_l_realinv = 400 * np.log(d['realinv']).diff().dropna()
    gs_l_realgdp = 400 * np.log(d['realgdp']).diff().dropna()
    lint = d['realint'][:-1]  # incorrect indexing for test purposes

    endog = gs_l_realinv

    # re-index because they will not conform to lint
    realgdp = gs_l_realgdp.reindex(lint.index, method='bfill')
    data = dict(const=np.ones_like(lint), lrealgdp=realgdp, lint=lint)
    exog = pd.DataFrame(data)

    # TODO: which index do we get??
    np.testing.assert_raises(ValueError, OLS, *(endog, exog))


class TestMultipleEqsArrays(TestArrays):
    @classmethod
    def setup_class(cls):
        cls.endog = np.random.random((10, 4))
        cls.exog = np.c_[np.ones(10), np.random.random((10, 2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        neqs = 4
        cls.col_result = cls.col_input = np.random.random(nvars)
        cls.row_result = cls.row_input = np.random.random(nrows)
        cls.cov_result = cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_eq_result = cls.cov_eq_input = np.random.random((neqs, neqs))
        cls.col_eq_result = cls.col_eq_input = np.array((neqs, nvars))
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = ['y1', 'y2', 'y3', 'y4']
        cls.row_labels = None

    def test_attach(self):
        data = self.data
        # this makes sure what the wrappers need work but not the wrapped
        # results themselves
        np.testing.assert_equal(data.wrap_output(self.col_input, 'columns'),
                                self.col_result)
        np.testing.assert_equal(data.wrap_output(self.row_input, 'rows'),
                                self.row_result)
        np.testing.assert_equal(data.wrap_output(self.cov_input, 'cov'),
                                self.cov_result)
        np.testing.assert_equal(data.wrap_output(self.cov_eq_input, 'cov_eq'),
                                self.cov_eq_result)
        np.testing.assert_equal(data.wrap_output(self.col_eq_input,
                                                 'columns_eq'),
                                self.col_eq_result)


class TestMultipleEqsDataFrames(TestDataFrames):
    @classmethod
    def setup_class(cls):
        cls.endog = endog = pd.DataFrame(np.random.random((10, 4)),
                                         columns=['y_1', 'y_2', 'y_3', 'y_4'])
        exog = pd.DataFrame(np.random.random((10, 2)),
                            columns=['x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        neqs = 4
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input,
                                   index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input,
                                   index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input,
                                      index=exog.columns,
                                      columns=exog.columns)
        cls.cov_eq_input = np.random.random((neqs, neqs))
        cls.cov_eq_result = pd.DataFrame(cls.cov_eq_input,
                                         index=endog.columns,
                                         columns=endog.columns)
        cls.col_eq_input = np.random.random((nvars, neqs))
        cls.col_eq_result = pd.DataFrame(cls.col_eq_input,
                                         index=exog.columns,
                                         columns=endog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = ['y_1', 'y_2', 'y_3', 'y_4']
        cls.row_labels = cls.exog.index

    def test_attach(self):
        data = self.data
        assert_series_equal(data.wrap_output(self.col_input, 'columns'),
                            self.col_result)
        assert_series_equal(data.wrap_output(self.row_input, 'rows'),
                            self.row_result)
        assert_frame_equal(data.wrap_output(self.cov_input, 'cov'),
                           self.cov_result)
        assert_frame_equal(data.wrap_output(self.cov_eq_input, 'cov_eq'),
                           self.cov_eq_result)
        assert_frame_equal(data.wrap_output(self.col_eq_input, 'columns_eq'),
                           self.col_eq_result)


class TestMissingArray:
    @classmethod
    def setup_class(cls):
        X = np.random.random((25, 4))
        y = np.random.random(25)
        y[10] = np.nan
        X[2, 3] = np.nan
        X[14, 2] = np.nan
        cls.y, cls.X = y, X

    @pytest.mark.smoke
    def test_raise_no_missing(self):
        # GH#1700
        sm_data.handle_data(np.random.random(20), np.random.random((20, 2)),
                            'raise')

    def test_raise(self):
        with pytest.raises(Exception):
            # TODO: be more specific about exception
            sm_data.handle_data(self.y, self.X, 'raise')

    def test_drop(self):
        y = self.y
        X = self.X
        combined = np.c_[y, X]
        idx = ~np.isnan(combined).any(axis=1)
        y = y[idx]
        X = X[idx]
        data = sm_data.handle_data(self.y, self.X, 'drop')
        np.testing.assert_array_equal(data.endog, y)
        np.testing.assert_array_equal(data.exog, X)

    def test_none(self):
        data = sm_data.handle_data(self.y, self.X, 'none', hasconst=False)
        np.testing.assert_array_equal(data.endog, self.y)
        np.testing.assert_array_equal(data.exog, self.X)
        assert data.k_constant == 0

    def test_endog_only_raise(self):
        with pytest.raises(Exception):
            # TODO: be more specific about exception
            sm_data.handle_data(self.y, None, 'raise')

    def test_endog_only_drop(self):
        y = self.y
        y = y[~np.isnan(y)]
        data = sm_data.handle_data(self.y, None, 'drop')
        np.testing.assert_array_equal(data.endog, y)

    def test_mv_endog(self):
        y = self.X
        y = y[~np.isnan(y).any(axis=1)]
        data = sm_data.handle_data(self.X, None, 'drop')
        np.testing.assert_array_equal(data.endog, y)

    def test_extra_kwargs_2d(self):
        sigma = np.random.random((25, 25))
        sigma = sigma + sigma.T - np.diag(np.diag(sigma))
        data = sm_data.handle_data(self.y, self.X, 'drop', sigma=sigma)
        idx = ~np.isnan(np.c_[self.y, self.X]).any(axis=1)
        sigma = sigma[idx][:, idx]
        np.testing.assert_array_equal(data.sigma, sigma)

    def test_extra_kwargs_1d(self):
        weights = np.random.random(25)
        data = sm_data.handle_data(self.y, self.X, 'drop', weights=weights)
        idx = ~np.isnan(np.c_[self.y, self.X]).any(axis=1)
        weights = weights[idx]
        np.testing.assert_array_equal(data.weights, weights)


class TestMissingPandas:
    @classmethod
    def setup_class(cls):
        X = np.random.random((25, 4))
        y = np.random.random(25)
        y[10] = np.nan
        X[2, 3] = np.nan
        X[14, 2] = np.nan
        cls.y = pd.Series(y)
        cls.X = pd.DataFrame(X)

    @pytest.mark.smoke
    def test_raise_no_missing(self):
        # GH#1700
        sm_data.handle_data(pd.Series(np.random.random(20)),
                            pd.DataFrame(np.random.random((20, 2))),
                            'raise')

    def test_raise(self):
        with pytest.raises(Exception):
            # TODO: be more specific about exception
            sm_data.handle_data(self.y, self.X, 'raise')

    def test_drop(self):
        y = self.y
        X = self.X
        combined = np.c_[y, X]
        idx = ~np.isnan(combined).any(axis=1)
        y = y.loc[idx]
        X = X.loc[idx]
        data = sm_data.handle_data(self.y, self.X, 'drop')
        np.testing.assert_array_equal(data.endog, y.values)
        assert_series_equal(data.orig_endog, self.y.loc[idx])
        np.testing.assert_array_equal(data.exog, X.values)
        assert_frame_equal(data.orig_exog, self.X.loc[idx])

    def test_none(self):
        data = sm_data.handle_data(self.y, self.X, 'none', hasconst=False)
        np.testing.assert_array_equal(data.endog, self.y.values)
        np.testing.assert_array_equal(data.exog, self.X.values)
        assert data.k_constant == 0

    def test_endog_only_raise(self):
        with pytest.raises(Exception):
            # TODO: be more specific about exception
            sm_data.handle_data(self.y, None, 'raise')

    def test_endog_only_drop(self):
        y = self.y
        y = y.dropna()
        data = sm_data.handle_data(self.y, None, 'drop')
        np.testing.assert_array_equal(data.endog, y.values)

    def test_mv_endog(self):
        y = self.X
        y = y.loc[~np.isnan(y.values).any(axis=1)]
        data = sm_data.handle_data(self.X, None, 'drop')
        np.testing.assert_array_equal(data.endog, y.values)

    def test_labels(self):
        labels = pd.Index([0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24])
        data = sm_data.handle_data(self.y, self.X, 'drop')
        np.testing.assert_(data.row_labels.equals(labels))


class TestConstant:
    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.longley import load_pandas
        cls.data = load_pandas()

    def test_array_constant(self):
        exog = self.data.exog.copy()
        exog['const'] = 1
        data = sm_data.handle_data(self.data.endog.values, exog.values)
        np.testing.assert_equal(data.k_constant, 1)
        np.testing.assert_equal(data.const_idx, 6)

    def test_pandas_constant(self):
        exog = self.data.exog.copy()
        exog['const'] = 1
        data = sm_data.handle_data(self.data.endog, exog)
        np.testing.assert_equal(data.k_constant, 1)
        np.testing.assert_equal(data.const_idx, 6)

    def test_pandas_noconstant(self):
        exog = self.data.exog.copy()
        data = sm_data.handle_data(self.data.endog, exog)
        np.testing.assert_equal(data.k_constant, 0)
        np.testing.assert_equal(data.const_idx, None)

    def test_array_noconstant(self):
        exog = self.data.exog.copy()
        data = sm_data.handle_data(self.data.endog.values, exog.values)
        np.testing.assert_equal(data.k_constant, 0)
        np.testing.assert_equal(data.const_idx, None)


class TestHandleMissing:

    def test_pandas(self):

        df = make_dataframe()
        df.iloc[[2, 5, 10], [2, 3, 1]] = np.nan
        y, X = df[df.columns[0]], df[df.columns[1:]]
        data, _ = sm_data.handle_missing(y, X, missing='drop')

        df = df.dropna()
        y_exp, X_exp = df[df.columns[0]], df[df.columns[1:]]
        assert_frame_equal(data['exog'], X_exp)
        assert_series_equal(data['endog'], y_exp)

    def test_arrays(self):
        arr = np.random.randn(20, 4)
        arr[[2, 5, 10], [2, 3, 1]] = np.nan
        y, X = arr[:, 0], arr[:, 1:]
        data, _ = sm_data.handle_missing(y, X, missing='drop')

        bools_mask = np.ones(20, dtype=bool)
        bools_mask[[2, 5, 10]] = False
        y_exp = arr[bools_mask, 0]
        X_exp = arr[bools_mask, 1:]
        np.testing.assert_array_equal(data['endog'], y_exp)
        np.testing.assert_array_equal(data['exog'], X_exp)

    def test_pandas_array(self):
        df = make_dataframe()
        df.iloc[[2, 5, 10], [2, 3, 1]] = np.nan
        y, X = df[df.columns[0]], df[df.columns[1:]].values
        data, _ = sm_data.handle_missing(y, X, missing='drop')

        df = df.dropna()
        y_exp, X_exp = df[df.columns[0]], df[df.columns[1:]].values
        np.testing.assert_array_equal(data['exog'], X_exp)
        assert_series_equal(data['endog'], y_exp)

    def test_array_pandas(self):
        df = make_dataframe()
        df.iloc[[2, 5, 10], [2, 3, 1]] = np.nan
        y, X = df[df.columns[0]].values, df[df.columns[1:]]
        data, _ = sm_data.handle_missing(y, X, missing='drop')

        df = df.dropna()
        y_exp, X_exp = df[df.columns[0]].values, df[df.columns[1:]]
        assert_frame_equal(data['exog'], X_exp)
        np.testing.assert_array_equal(data['endog'], y_exp)

    def test_noop(self):
        df = make_dataframe()
        df.iloc[[2, 5, 10], [2, 3, 1]] = np.nan
        y, X = df[df.columns[0]], df[df.columns[1:]]
        data, _ = sm_data.handle_missing(y, X, missing='none')

        y_exp, X_exp = df[df.columns[0]], df[df.columns[1:]]
        assert_frame_equal(data['exog'], X_exp)
        assert_series_equal(data['endog'], y_exp)


class CheckHasConstant:

    def test_hasconst(self):
        for x, result in zip(self.exogs, self.results):
            mod = self.mod(self.y, x)
            assert_equal(mod.k_constant, result[0])
            assert_equal(mod.data.k_constant, result[0])
            if result[1] is None:
                assert_(mod.data.const_idx is None)
            else:
                assert_equal(mod.data.const_idx, result[1])

            # extra check after fit, some models raise on singular
            fit_kwds = getattr(self, 'fit_kwds', {})
            try:
                res = mod.fit(**fit_kwds)
            except np.linalg.LinAlgError:
                pass
            else:
                assert_equal(res.model.k_constant, result[0])
                assert_equal(res.model.data.k_constant, result[0])

    @classmethod
    def setup_class(cls):
        # create data
        np.random.seed(0)
        cls.y_c = np.random.randn(20)
        cls.y_bin = (cls.y_c > 0).astype(int)
        x1 = np.column_stack((np.ones(20), np.zeros(20)))
        result1 = (1, 0)
        x2 = np.column_stack((np.arange(20) < 10.5,
                              np.arange(20) > 10.5)).astype(float)
        result2 = (1, None)
        x3 = np.column_stack((np.arange(20), np.zeros(20)))
        result3 = (0, None)
        x4 = np.column_stack((np.arange(20), np.zeros((20, 2))))
        result4 = (0, None)
        x5 = np.column_stack((np.zeros(20), 0.5 * np.ones(20)))
        result5 = (1, 1)
        x5b = np.column_stack((np.arange(20), np.ones((20, 3))))
        result5b = (1, 1)
        x5c = np.column_stack((np.arange(20), np.ones((20, 3)) * [0.5, 1, 1]))
        result5c = (1, 2)
        # implicit and zero column
        x6 = np.column_stack((np.arange(20) < 10.5,
                              np.arange(20) > 10.5,
                              np.zeros(20))).astype(float)
        result6 = (1, None)
        x7 = np.column_stack((np.arange(20) < 10.5,
                              np.arange(20) > 10.5,
                              np.zeros((20, 2)))).astype(float)
        result7 = (1, None)

        cls.exogs = (x1, x2, x3, x4, x5, x5b, x5c, x6, x7)
        cls.results = (result1, result2, result3, result4, result5, result5b,
                       result5c, result6, result7)
        cls._initialize()


class TestHasConstantOLS(CheckHasConstant):

    @classmethod
    def _initialize(cls):

        cls.mod = OLS
        cls.y = cls.y_c


class TestHasConstantGLM(CheckHasConstant):

    @staticmethod
    def mod(y, x):
        return GLM(y, x, family=families.Binomial())

    @classmethod
    def _initialize(cls):
        cls.y = cls.y_bin


class TestHasConstantLogit(CheckHasConstant):

    @classmethod
    def _initialize(cls):
        cls.mod = Logit
        cls.y = cls.y_bin
        cls.fit_kwds = {'disp': False}


def test_dtype_object():
    # see GH#880

    X = np.random.random((40, 2))
    df = pd.DataFrame(X)
    df[2] = np.random.randint(2, size=40).astype('object')
    df['constant'] = 1

    y = pd.Series(np.random.randint(2, size=40))

    np.testing.assert_raises(ValueError, sm_data.handle_data, y, df)


def test_formula_missing_extra_arrays():
    np.random.seed(1)
    # because patsy cannot turn off missing data-handling as of 0.3.0, we need
    # separate tests to make sure that missing values are handled correctly
    # when going through formulas

    # there is a handle_formula_data step
    # then there is the regular handle_data step
    # see GH#2083

    # the untested cases are endog/exog have missing. extra has missing.
    # endog/exog are fine. extra has missing.
    # endog/exog do or do not have missing and extra has wrong dimension
    y = np.random.randn(10)
    y_missing = y.copy()
    y_missing[[2, 5]] = np.nan
    X = np.random.randn(10)
    X_missing = X.copy()
    X_missing[[1, 3]] = np.nan

    weights = np.random.uniform(size=10)
    weights_missing = weights.copy()
    weights_missing[[6]] = np.nan

    weights_wrong_size = np.random.randn(12)

    data = {'y': y,
            'X': X,
            'y_missing': y_missing,
            'X_missing': X_missing,
            'weights': weights,
            'weights_missing': weights_missing}
    data = pd.DataFrame.from_dict(data)
    data['constant'] = 1

    formula = 'y_missing ~ X_missing'

    ((endog, exog),
     missing_idx, design_info) = handle_formula_data(data, None, formula,
                                                     depth=2,
                                                     missing='drop')

    kwargs = {'missing_idx': missing_idx, 'missing': 'drop',
              'weights': data['weights_missing']}

    model_data = sm_data.handle_data(endog, exog, **kwargs)
    data_nona = data.dropna()
    assert_equal(data_nona['y'].values, model_data.endog)
    assert_equal(data_nona[['constant', 'X']].values, model_data.exog)
    assert_equal(data_nona['weights'].values, model_data.weights)

    tmp = handle_formula_data(data, None, formula, depth=2, missing='drop')
    (endog, exog), missing_idx, design_info = tmp
    weights_2d = np.random.randn(10, 10)
    weights_2d[[8, 7], [7, 8]] = np.nan   # symmetric missing values
    kwargs.update({'weights': weights_2d,
                   'missing_idx': missing_idx})

    model_data2 = sm_data.handle_data(endog, exog, **kwargs)

    good_idx = [0, 4, 6, 9]
    assert_equal(data.loc[good_idx, 'y'], model_data2.endog)
    assert_equal(data.loc[good_idx, ['constant', 'X']], model_data2.exog)
    assert_equal(weights_2d[good_idx][:, good_idx], model_data2.weights)

    tmp = handle_formula_data(data, None, formula, depth=2, missing='drop')
    (endog, exog), missing_idx, design_info = tmp

    kwargs.update({'weights': weights_wrong_size,
                   'missing_idx': missing_idx})
    assert_raises(ValueError, sm_data.handle_data, endog, exog, **kwargs)


def test_raise_nonfinite_exog():
    # we raise now in the has constant check before hitting the linear algebra
    from statsmodels.tools.sm_exceptions import MissingDataError
    x = np.arange(10)[:, None]**([0., 1.])
    # random numbers for y
    y = np.array([-0.6, -0.1, 0., -0.7, -0.5, 0.5, 0.1, -0.8, -2., 1.1])

    x[1, 1] = np.inf
    assert_raises(MissingDataError, OLS, y, x)
    x[1, 1] = np.nan
    assert_raises(MissingDataError, OLS, y, x)
