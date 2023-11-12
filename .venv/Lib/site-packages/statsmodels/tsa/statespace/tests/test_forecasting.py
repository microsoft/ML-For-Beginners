r"""
Tests for forecasting-related features not tested elsewhere
"""

import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_allclose

from statsmodels.tsa.statespace import sarimax


@pytest.mark.parametrize('data', ['list', 'numpy', 'range', 'date', 'period'])
def test_append_multistep(data):
    # Test that `MLEResults.append` works when called repeatedly
    endog = [1., 0.5, 1.5, 0.9, 0.2, 0.34]
    if data == 'numpy':
        endog = np.array(endog)
    elif data == 'range':
        endog = pd.Series(endog)
    elif data == 'date':
        index = pd.date_range(start='2000-01-01', periods=6, freq='MS')
        endog = pd.Series(endog, index=index)
    elif data == 'period':
        index = pd.period_range(start='2000-01', periods=6, freq='M')
        endog = pd.Series(endog, index=index)

    # Base model fitting
    mod = sarimax.SARIMAX(endog[:2], order=(1, 0, 0))
    res = mod.smooth([0.5, 1.0])
    assert_allclose(res.model.endog[:, 0], [1., 0.5])
    assert_allclose(res.forecast(1), 0.25)

    # First append
    res1 = res.append(endog[2:3])
    assert_allclose(res1.model.endog[:, 0], [1., 0.5, 1.5])
    assert_allclose(res1.forecast(1), 0.75)

    # Second append
    res2 = res1.append(endog[3:5])
    assert_allclose(res2.model.endog[:, 0], [1., 0.5, 1.5, 0.9, 0.2])
    assert_allclose(res2.forecast(1), 0.1)

    # Third append
    res3 = res2.append(endog[5:6])
    print(res3.model.endog)
    assert_allclose(res3.model.endog[:, 0], [1., 0.5, 1.5, 0.9, 0.2, 0.34])
    assert_allclose(res3.forecast(1), 0.17)
