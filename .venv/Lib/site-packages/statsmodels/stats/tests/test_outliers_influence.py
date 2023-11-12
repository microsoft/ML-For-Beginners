import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.datasets import statecrime
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import (
    reset_ramsey,
    variance_inflation_factor,
)
from statsmodels.tools import add_constant


def test_reset_stata():
    data = statecrime.load_pandas().data
    mod = OLS(data.violent, add_constant(data[["murder", "hs_grad"]]))
    res = mod.fit()
    stat = reset_ramsey(res, degree=4)
    assert_almost_equal(stat.fvalue, 1.52, decimal=2)
    assert_almost_equal(stat.pvalue, 0.2221, decimal=4)

    exog_idx = list(data.columns).index("urban")
    data_arr = np.asarray(data)
    vif = variance_inflation_factor(data_arr, exog_idx)
    assert_almost_equal(vif, 16.4394, decimal=4)

    exog_idx = list(data.columns).index("urban")
    vif_df = variance_inflation_factor(data, exog_idx)
    assert_almost_equal(vif_df, 16.4394, decimal=4)
