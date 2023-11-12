# -*- coding: utf-8 -*-
import warnings

from statsmodels.compat.pandas import PD_LT_1_4

import os

import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
                           assert_raises, assert_array_equal,
                           assert_array_less, assert_allclose)
import pytest

try:
    import matplotlib.pyplot as plt
    missing_matplotlib = False
    plt.switch_backend('Agg')

except ImportError:
    missing_matplotlib = True

# Example data
# https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
#     viewer.htm#statug_introreg_sect012.htm
X = pd.DataFrame([['Minas Graes', 2.068, 2.070, 1.580, 1, 0],
                  ['Minas Graes', 2.068, 2.074, 1.602, 2, 1],
                  ['Minas Graes', 2.090, 2.090, 1.613, 3, 0],
                  ['Minas Graes', 2.097, 2.093, 1.613, 4, 1],
                  ['Minas Graes', 2.117, 2.125, 1.663, 5, 0],
                  ['Minas Graes', 2.140, 2.146, 1.681, 6, 1],
                  ['Matto Grosso', 2.045, 2.054, 1.580, 7, 0],
                  ['Matto Grosso', 2.076, 2.088, 1.602, 8, 1],
                  ['Matto Grosso', 2.090, 2.093, 1.643, 9, 0],
                  ['Matto Grosso', 2.111, 2.114, 1.643, 10, 1],
                  ['Santa Cruz', 2.093, 2.098, 1.653, 11, 0],
                  ['Santa Cruz', 2.100, 2.106, 1.623, 12, 1],
                  ['Santa Cruz', 2.104, 2.101, 1.653, 13, 0]],
                 columns=['Loc', 'Basal', 'Occ', 'Max', 'id', 'alt'])


def test_auto_col_name():
    # Test auto generated variable names when endog_names is None
    mod = Factor(None, 2, corr=np.eye(11), endog_names=None,
                 smc=False)
    assert_array_equal(mod.endog_names,
                       ['var00', 'var01', 'var02', 'var03', 'var04', 'var05',
                        'var06', 'var07', 'var08', 'var09', 'var10'])


def test_direct_corr_matrix():
    # Test specifying the correlation matrix directly
    mod = Factor(None, 2, corr=np.corrcoef(X.iloc[:, 1:-1], rowvar=0),
                 smc=False)
    results = mod.fit(tol=1e-10)
    a = np.array([[0.965392158864, 0.225880658666255],
                  [0.967587154301, 0.212758741910989],
                  [0.929891035996, -0.000603217967568],
                  [0.486822656362, -0.869649573289374]])
    assert_array_almost_equal(results.loadings, a, decimal=8)
    # Test set and get endog_names
    mod.endog_names = X.iloc[:, 1:-1].columns
    assert_array_equal(mod.endog_names, ['Basal', 'Occ', 'Max', 'id'])

    # Test set endog_names with the wrong number of elements
    assert_raises(ValueError, setattr, mod, 'endog_names',
                  X.iloc[:, :1].columns)


def test_unknown_fa_method_error():
    # Test raise error if an unkonwn FA method is specified in fa.method
    mod = Factor(X.iloc[:, 1:-1], 2, method='ab')
    assert_raises(ValueError, mod.fit)


def test_example_compare_to_R_output():
    # Testing basic functions and compare to R output

    # R code for producing the results:
    # library(psych)
    # library(GPArotation)
    # Basal = c(2.068,	2.068,	2.09,	2.097,	2.117,	2.14,	2.045,	2.076,	2.09,	2.111,	2.093,	2.1,	2.104)
    # Occ = c(2.07,	2.074,	2.09,	2.093,	2.125,	2.146,	2.054,	2.088,	2.093,	2.114,	2.098,	2.106,	2.101)
    # Max = c(1.58,	1.602,	1.613,	1.613,	1.663,	1.681,	1.58,	1.602,	1.643,	1.643,	1.653,	1.623,	1.653)
    # id = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    # Y <- cbind(Basal, Occ, Max, id)
    # a <- fa(Y, nfactors=2, fm="pa", rotate="none", SMC=FALSE, min.err=1e-10)
    # b <- cbind(a$loadings[,1], -a$loadings[,2])
    # b
    # a <- fa(Y, nfactors=2, fm="pa", rotate="Promax", SMC=TRUE, min.err=1e-10)
    # b <- cbind(a$loadings[,1], a$loadings[,2])
    # b
    # a <- fa(Y, nfactors=2, fm="pa", rotate="Varimax", SMC=TRUE, min.err=1e-10)
    # b <- cbind(a$loadings[,1], a$loadings[,2])
    # b
    # a <- fa(Y, nfactors=2, fm="pa", rotate="quartimax", SMC=TRUE, min.err=1e-10)
    # b <- cbind(a$loadings[,1], -a$loadings[,2])
    # b
    # a <- fa(Y, nfactors=2, fm="pa", rotate="oblimin", SMC=TRUE, min.err=1e-10)
    # b <- cbind(a$loadings[,1], a$loadings[,2])
    # b

    # No rotation without squared multiple correlations prior
    # produce same results as in R `fa`
    mod = Factor(X.iloc[:, 1:-1], 2, smc=False)
    results = mod.fit(tol=1e-10)
    a = np.array([[0.965392158864, 0.225880658666255],
                  [0.967587154301, 0.212758741910989],
                  [0.929891035996, -0.000603217967568],
                  [0.486822656362, -0.869649573289374]])
    assert_array_almost_equal(results.loadings, a, decimal=8)

    # No rotation WITH squared multiple correlations prior
    # produce same results as in R `fa`
    mod = Factor(X.iloc[:, 1:-1], 2, smc=True)
    results = mod.fit()
    a = np.array([[0.97541115, 0.20280987],
                  [0.97113975, 0.17207499],
                  [0.9618705, -0.2004196],
                  [0.37570708, -0.45821379]])
    assert_array_almost_equal(results.loadings, a, decimal=8)

    # Same as R GRArotation
    results.rotate('varimax')
    a = np.array([[0.98828898, -0.12587155],
                  [0.97424206, -0.15354033],
                  [0.84418097, -0.502714],
                  [0.20601929, -0.55558235]])
    assert_array_almost_equal(results.loadings, a, decimal=8)

    results.rotate('quartimax')  # Same as R fa
    a = np.array([[0.98935598, 0.98242714, 0.94078972, 0.33442284],
                  [0.117190049, 0.086943252, -0.283332952, -0.489159543]])
    assert_array_almost_equal(results.loadings, a.T, decimal=8)

    results.rotate('equamax')  # Not the same as R fa

    results.rotate('promax')  # Not the same as R fa

    results.rotate('biquartimin')  # Not the same as R fa

    results.rotate('oblimin')  # Same as R fa
    a = np.array([[1.02834170170, 1.00178840104, 0.71824931384,
                   -0.00013510048],
                  [0.06563421, 0.03096076, -0.39658839, -0.59261944]])
    assert_array_almost_equal(results.loadings, a.T, decimal=8)

    # Testing result summary string
    results.rotate('varimax')
    desired = (
"""   Factor analysis results
=============================
      Eigenvalues
-----------------------------
 Basal   Occ    Max      id
-----------------------------
 2.9609 0.3209 0.0000 -0.0000
-----------------------------

-----------------------------
      Communality
-----------------------------
  Basal   Occ    Max     id
-----------------------------
  0.9926 0.9727 0.9654 0.3511
-----------------------------

-----------------------------
   Pre-rotated loadings
-----------------------------------
            factor 0       factor 1
-----------------------------------
Basal         0.9754         0.2028
Occ           0.9711         0.1721
Max           0.9619        -0.2004
id            0.3757        -0.4582
-----------------------------

-----------------------------
   varimax rotated loadings
-----------------------------------
            factor 0       factor 1
-----------------------------------
Basal         0.9883        -0.1259
Occ           0.9742        -0.1535
Max           0.8442        -0.5027
id            0.2060        -0.5556
=============================
""")
    actual = results.summary().as_text()
    actual = "\n".join(line.rstrip() for line in actual.splitlines()) + "\n"
    assert_equal(actual, desired)


@pytest.mark.skipif(missing_matplotlib, reason='matplotlib not available')
def test_plots(close_figures):
    mod = Factor(X.iloc[:, 1:], 3)
    results = mod.fit()
    results.rotate('oblimin')
    fig = results.plot_scree()

    fig_loadings = results.plot_loadings()
    assert_equal(3, len(fig_loadings))


@pytest.mark.smoke
def test_getframe_smoke():
    #  mostly smoke tests for now
    mod = Factor(X.iloc[:, 1:-1], 2, smc=True)
    res = mod.fit()

    df = res.get_loadings_frame(style='raw')
    assert_(isinstance(df, pd.DataFrame))

    lds = res.get_loadings_frame(style='strings', decimals=3, threshold=0.3)


    # The Styler option require jinja2, skip if not available
    try:
        from jinja2 import Template  # noqa:F401
    except ImportError:
        return
        # TODO: separate this and do pytest.skip?

    # Old implementation that warns
    if PD_LT_1_4:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            lds.to_latex()
    else:
        # Smoke test using new style to_latex
        lds.style.to_latex()
    try:
        from pandas.io import formats as pd_formats
    except ImportError:
        from pandas import formats as pd_formats

    ldf = res.get_loadings_frame(style='display')
    assert_(isinstance(ldf, pd_formats.style.Styler))
    assert_(isinstance(ldf.data, pd.DataFrame))

    res.get_loadings_frame(style='display', decimals=3, threshold=0.2)

    res.get_loadings_frame(style='display', decimals=3, color_max='GAINSBORO')

    res.get_loadings_frame(style='display', decimals=3, threshold=0.45, highlight_max=False, sort_=False)


def test_factor_missing():
    xm = X.iloc[:, 1:-1].copy()
    nobs, k_endog = xm.shape
    xm.iloc[2,2] = np.nan
    mod = Factor(xm, 2)
    assert_equal(mod.nobs, nobs - 1)
    assert_equal(mod.k_endog, k_endog)
    assert_equal(mod.endog.shape, (nobs - 1, k_endog))


def _zscore(x):
    # helper function
    return (x - x.mean(0)) / x.std(0)


@pytest.mark.smoke
def test_factor_scoring():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    csv_path = os.path.join(dir_path, 'results', 'factor_data.csv')
    y = pd.read_csv(csv_path)
    csv_path = os.path.join(dir_path, 'results', 'factors_stata.csv')
    f_s = pd.read_csv(csv_path)
    #  mostly smoke tests for now
    mod = Factor(y, 2)
    res = mod.fit(maxiter=1)
    res.rotate('varimax')
    f_reg = res.factor_scoring(method='reg')
    assert_allclose(f_reg * [1, -1], f_s[["f1", 'f2']].values,
                    atol=1e-4, rtol=1e-3)
    f_bart = res.factor_scoring()
    assert_allclose(f_bart * [1, -1], f_s[["f1b", 'f2b']].values,
                    atol=1e-4, rtol=1e-3)

    # check we have high correlation to ols and gls
    f_ols = res.factor_scoring(method='ols')
    f_gls = res.factor_scoring(method='gls')
    f_reg_z = _zscore(f_reg)
    f_ols_z = _zscore(f_ols)
    f_gls_z = _zscore(f_gls)
    assert_array_less(0.98, (f_ols_z * f_reg_z).mean(0))
    assert_array_less(0.999, (f_gls_z * f_reg_z).mean(0))

    # with oblique rotation
    res.rotate('oblimin')
    # Note: Stata has second factor with flipped sign compared to statsmodels
    assert_allclose(res._corr_factors()[0, 1],  (-1) * 0.25651037, rtol=1e-3)
    f_reg = res.factor_scoring(method='reg')
    assert_allclose(f_reg * [1, -1], f_s[["f1o", 'f2o']].values,
                    atol=1e-4, rtol=1e-3)
    f_bart = res.factor_scoring()
    assert_allclose(f_bart * [1, -1], f_s[["f1ob", 'f2ob']].values,
                    atol=1e-4, rtol=1e-3)

    # check we have high correlation to ols and gls
    f_ols = res.factor_scoring(method='ols')
    f_gls = res.factor_scoring(method='gls')
    f_reg_z = _zscore(f_reg)
    f_ols_z = _zscore(f_ols)
    f_gls_z = _zscore(f_gls)
    assert_array_less(0.97, (f_ols_z * f_reg_z).mean(0))
    assert_array_less(0.999, (f_gls_z * f_reg_z).mean(0))

    # check provided endog
    f_ols2 = res.factor_scoring(method='ols', endog=res.model.endog)
    assert_allclose(f_ols2, f_ols, rtol=1e-13)
