# -*- coding: utf-8 -*-
"""
Tests for Results.predict
"""
from statsmodels.compat.pandas import testing as pdt

import numpy as np
import pandas as pd

from numpy.testing import assert_allclose, assert_equal

from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM


class CheckPredictReturns:

    def test_2d(self):
        res = self.res
        data = self.data

        fitted = res.fittedvalues.iloc[1:10:2]

        pred = res.predict(data.iloc[1:10:2])
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # plain dict
        xd = dict(zip(data.columns, data.iloc[1:10:2].values.T))
        pred = res.predict(xd)
        assert_equal(pred.index, np.arange(len(pred)))
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

    def test_1d(self):
        # one observation
        res = self.res
        data = self.data

        pred = res.predict(data.iloc[:1])
        pdt.assert_index_equal(pred.index, data.iloc[:1].index)
        assert_allclose(pred.values, res.fittedvalues[0], rtol=1e-13)

        fittedm = res.fittedvalues.mean()
        xmean = data.mean()
        pred = res.predict(xmean.to_frame().T)
        assert_equal(pred.index, np.arange(1))
        assert_allclose(pred, fittedm, rtol=1e-13)

        # Series
        pred = res.predict(data.mean())
        assert_equal(pred.index, np.arange(1))
        assert_allclose(pred.values, fittedm, rtol=1e-13)

        # dict with scalar value (is plain dict)
        # Note: this warns about dropped nan, even though there are None -FIXED
        pred = res.predict(data.mean().to_dict())
        assert_equal(pred.index, np.arange(1))
        assert_allclose(pred.values, fittedm, rtol=1e-13)

    def test_nopatsy(self):
        res = self.res
        data = self.data
        fitted = res.fittedvalues.iloc[1:10:2]

        # plain numpy array
        pred = res.predict(res.model.exog[1:10:2], transform=False)
        assert_allclose(pred, fitted.values, rtol=1e-13)

        # pandas DataFrame
        x = pd.DataFrame(res.model.exog[1:10:2],
                         index = data.index[1:10:2],
                         columns=res.model.exog_names)
        pred = res.predict(x)
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # one observation - 1-D
        pred = res.predict(res.model.exog[1], transform=False)
        assert_allclose(pred, fitted.values[0], rtol=1e-13)

        # one observation - pd.Series
        pred = res.predict(x.iloc[0])
        pdt.assert_index_equal(pred.index, fitted.index[:1])
        assert_allclose(pred.values[0], fitted.values[0], rtol=1e-13)


class TestPredictOLS(CheckPredictReturns):

    @classmethod
    def setup_class(cls):
        nobs = 30
        np.random.seed(987128)
        x = np.random.randn(nobs, 3)
        y = x.sum(1) + np.random.randn(nobs)
        index = ['obs%02d' % i for i in range(nobs)]
        # add one extra column to check that it does not matter
        cls.data = pd.DataFrame(np.round(np.column_stack((y, x)), 4),
                                columns='y var1 var2 var3'.split(),
                                index=index)

        cls.res = OLS.from_formula('y ~ var1 + var2', data=cls.data).fit()


class TestPredictGLM(CheckPredictReturns):

    @classmethod
    def setup_class(cls):
        nobs = 30
        np.random.seed(987128)
        x = np.random.randn(nobs, 3)
        y = x.sum(1) + np.random.randn(nobs)
        index = ['obs%02d' % i for i in range(nobs)]
        # add one extra column to check that it does not matter
        cls.data = pd.DataFrame(np.round(np.column_stack((y, x)), 4),
                                columns='y var1 var2 var3'.split(),
                                index=index)

        cls.res = GLM.from_formula('y ~ var1 + var2', data=cls.data).fit()

    def test_predict_offset(self):
        res = self.res
        data = self.data

        fitted = res.fittedvalues.iloc[1:10:2]
        offset = np.arange(len(fitted))
        fitted = fitted + offset

        pred = res.predict(data.iloc[1:10:2], offset=offset)
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # plain dict
        xd = dict(zip(data.columns, data.iloc[1:10:2].values.T))
        pred = res.predict(xd, offset=offset)
        assert_equal(pred.index, np.arange(len(pred)))
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # offset as pandas.Series
        data2 = data.iloc[1:10:2].copy()
        data2['offset'] = offset
        pred = res.predict(data2, offset=data2['offset'])
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # check nan in exog is ok, preserves index matching offset length
        data2 = data.iloc[1:10:2].copy()
        data2['offset'] = offset
        data2.iloc[0, 1] = np.nan
        pred = res.predict(data2, offset=data2['offset'])
        pdt.assert_index_equal(pred.index, fitted.index)
        fitted_nan = fitted.copy()
        fitted_nan[0] = np.nan
        assert_allclose(pred.values, fitted_nan.values, rtol=1e-13)
