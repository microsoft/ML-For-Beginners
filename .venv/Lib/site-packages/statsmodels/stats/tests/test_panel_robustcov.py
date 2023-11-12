# -*- coding: utf-8 -*-
"""Test for panel robust covariance estimators after pooled ols
this follows the example from xtscc paper/help

Created on Tue May 22 20:27:57 2012

Author: Josef Perktold
"""

from statsmodels.compat.python import lmap
import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.stats.sandwich_covariance as sw


def test_panel_robust_cov():
    import statsmodels.datasets.grunfeld as gr
    from .results.results_panelrobust import results as res_stata

    dtapa = gr.data.load_pandas()
    #Stata example/data seems to miss last firm
    dtapa_endog = dtapa.endog[:200]
    dtapa_exog = dtapa.exog[:200]
    res = OLS(dtapa_endog, add_constant(dtapa_exog[['value', 'capital']],
              prepend=False)).fit()

    #time indicator in range(max Ti)
    time = np.require(dtapa_exog[['year']], requirements="W")
    time -= time.min()
    time = np.squeeze(time).astype(int)

    #sw.cov_nw_panel requires bounds instead of index
    tidx = [(i*20, 20*(i+1)) for i in range(10)]

    #firm index in range(n_firms)
    firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                                    return_inverse=True)

    #panel newey west standard errors
    cov = sw.cov_nw_panel(res, 0, tidx, use_correction='hac')
    #dropping numpy 1.4 soon
    #np.testing.assert_allclose(cov, res_stata.cov_pnw0_stata, rtol=1e-6)
    assert_almost_equal(cov, res_stata.cov_pnw0_stata, decimal=4)

    cov = sw.cov_nw_panel(res, 1, tidx, use_correction='hac')
    #np.testing.assert_allclose(cov, res_stata.cov_pnw1_stata, rtol=1e-6)
    assert_almost_equal(cov, res_stata.cov_pnw1_stata, decimal=4)

    cov = sw.cov_nw_panel(res, 4, tidx)  #check default
    #np.testing.assert_allclose(cov, res_stata.cov_pnw4_stata, rtol=1e-6)
    assert_almost_equal(cov, res_stata.cov_pnw4_stata, decimal=4)

    #cluster robust standard errors
    cov_clu = sw.cov_cluster(res, firm_id)
    assert_almost_equal(cov_clu, res_stata.cov_clu_stata, decimal=4)

    #cluster robust standard errors, non-int groups
    cov_clu = sw.cov_cluster(res, lmap(str, firm_id))
    assert_almost_equal(cov_clu, res_stata.cov_clu_stata, decimal=4)

    #Driscoll and Kraay panel robust standard errors
    rcov = sw.cov_nw_groupsum(res, 0, time, use_correction=0)
    assert_almost_equal(rcov, res_stata.cov_dk0_stata, decimal=4)

    rcov = sw.cov_nw_groupsum(res, 1, time, use_correction=0)
    assert_almost_equal(rcov, res_stata.cov_dk1_stata, decimal=4)

    rcov = sw.cov_nw_groupsum(res, 4, time) #check default
    assert_almost_equal(rcov, res_stata.cov_dk4_stata, decimal=4)
