# -*- coding: utf-8 -*-
"""Tests for sandwich robust covariance estimation

see also in regression for cov_hac compared to Gretl and
sandbox.panel test_random_panel for comparing cov_cluster, cov_hac_panel and
cov_white

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.stats.sandwich_covariance as sw


def test_cov_cluster_2groups():
    # comparing cluster robust standard errors to Peterson
    # requires Petersen's test_data
    # http://www.kellogg.northwestern.edu/faculty/petersen
    #      .../htm/papers/se/test_data.txt
    import os
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    fpath = os.path.join(cur_dir, "test_data.txt")
    pet = np.genfromtxt(fpath)
    endog = pet[:, -1]
    group = pet[:, 0].astype(int)
    time = pet[:, 1].astype(int)
    exog = add_constant(pet[:, 2])
    res = OLS(endog, exog).fit()

    cov01, covg, covt = sw.cov_cluster_2groups(res, group, group2=time)

    # Reference number from Petersen
    # http://www.kellogg.northwestern.edu/faculty/petersen/htm
    #       .../papers/se/test_data.htm

    bse_petw = [0.0284, 0.0284]
    bse_pet0 = [0.0670, 0.0506]
    bse_pet1 = [0.0234, 0.0334]  # year
    bse_pet01 = [0.0651, 0.0536]  # firm and year
    bse_0 = sw.se_cov(covg)
    bse_1 = sw.se_cov(covt)
    bse_01 = sw.se_cov(cov01)
    # print res.HC0_se, bse_petw - res.HC0_se
    # print bse_0, bse_0 - bse_pet0
    # print bse_1, bse_1 - bse_pet1
    # print bse_01, bse_01 - bse_pet01
    assert_almost_equal(bse_petw, res.HC0_se, decimal=4)
    assert_almost_equal(bse_0, bse_pet0, decimal=4)
    assert_almost_equal(bse_1, bse_pet1, decimal=4)
    assert_almost_equal(bse_01, bse_pet01, decimal=4)


def test_hac_simple():
    from statsmodels.datasets import macrodata
    d2 = macrodata.load_pandas().data
    g_gdp = 400 * np.diff(np.log(d2['realgdp'].values))
    g_inv = 400 * np.diff(np.log(d2['realinv'].values))
    exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1].values])
    res_olsg = OLS(g_inv, exogg).fit()

    # > NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE,
    #             verbose=TRUE, adjust=TRUE)
    # Lag truncation parameter chosen: 4
    #             (Intercept)                   ggdp                  lint
    cov1_r = [
        [+1.40643899878678802, -0.3180328707083329709, -0.060621111216488610],
        [-0.31803287070833292, 0.1097308348999818661, +0.000395311760301478],
        [-0.06062111121648865, 0.0003953117603014895, +0.087511528912470993]
    ]

    # > NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE,
    #             verbose=TRUE, adjust=FALSE)
    # Lag truncation parameter chosen: 4
    #         (Intercept)                  ggdp                  lint
    cov2_r = [
        [+1.3855512908840137, -0.313309610252268500, -0.059720797683570477],
        [-0.3133096102522685, +0.108101169035130618, +0.000389440793564339],
        [-0.0597207976835705, +0.000389440793564336, +0.086211852740503622]
    ]

    cov1 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=True)
    se1 = sw.se_cov(cov1)
    cov2 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    se2 = sw.se_cov(cov2)
    # Relax precision requirements for this test due to failure in NumPy 1.23
    assert_allclose(cov1, cov1_r)
    assert_allclose(cov2, cov2_r)
    assert_allclose(np.sqrt(np.diag(cov1_r)), se1)
    assert_allclose(np.sqrt(np.diag(cov2_r)), se2)

    # compare default for nlags
    cov3 = sw.cov_hac_simple(res_olsg, use_correction=False)
    cov4 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    assert_allclose(cov3, cov4)
