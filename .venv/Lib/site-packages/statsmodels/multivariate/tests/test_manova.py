# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose

from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from statsmodels.tools import add_constant

# Example data
# https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
#     viewer.htm#statug_introreg_sect012.htm
X = pd.DataFrame([['Minas Graes', 2.068, 2.070, 1.580],
                  ['Minas Graes', 2.068, 2.074, 1.602],
                  ['Minas Graes', 2.090, 2.090, 1.613],
                  ['Minas Graes', 2.097, 2.093, 1.613],
                  ['Minas Graes', 2.117, 2.125, 1.663],
                  ['Minas Graes', 2.140, 2.146, 1.681],
                  ['Matto Grosso', 2.045, 2.054, 1.580],
                  ['Matto Grosso', 2.076, 2.088, 1.602],
                  ['Matto Grosso', 2.090, 2.093, 1.643],
                  ['Matto Grosso', 2.111, 2.114, 1.643],
                  ['Santa Cruz', 2.093, 2.098, 1.653],
                  ['Santa Cruz', 2.100, 2.106, 1.623],
                  ['Santa Cruz', 2.104, 2.101, 1.653]],
                 columns=['Loc', 'Basal', 'Occ', 'Max'])


def test_manova_sas_example():
    # Results should be the same as figure 4.5 of
    # https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
    # viewer.htm#statug_introreg_sect012.htm
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    r = mod.mv_test()
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Value'],
                        0.60143661, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Value'],
                        0.44702843, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Value'],
                        0.58210348, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Value'],
                        0.35530890, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'F Value'],
                        0.77, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'F Value'],
                        0.86, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'F Value'],
                        0.75, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'F Value'],
                        1.07, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Num DF'],
                        3, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Den DF'],
                        16, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Den DF'],
                        18, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Den DF'],
                        9.0909, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Den DF'],
                        9, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Pr > F'],
                        0.6032, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Pr > F'],
                        0.5397, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Pr > F'],
                        0.6272, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Pr > F'],
                        0.4109, decimal=4)


def test_manova_no_formula():
    # Same as previous test only skipping formula interface
    exog = add_constant(pd.get_dummies(X[['Loc']], drop_first=True,
                                       dtype=float))
    endog = X[['Basal', 'Occ', 'Max']]
    mod = MANOVA(endog, exog)
    intercept = np.zeros((1, 3))
    intercept[0, 0] = 1
    loc = np.zeros((2, 3))
    loc[0, 1] = loc[1, 2] = 1
    hypotheses = [('Intercept', intercept), ('Loc', loc)]
    r = mod.mv_test(hypotheses)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Value'],
                        0.60143661, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Value'],
                        0.44702843, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace",
                                             'Value'],
                        0.58210348, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Value'],
                        0.35530890, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'F Value'],
                        0.77, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'F Value'],
                        0.86, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace",
                                             'F Value'],
                        0.75, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'F Value'],
                        1.07, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace",
                                             'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Num DF'],
                        3, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Den DF'],
                        16, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Den DF'],
                        18, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace",
                                             'Den DF'],
                        9.0909, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Den DF'],
                        9, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Pr > F'],
                        0.6032, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Pr > F'],
                        0.5397, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace",
                                             'Pr > F'],
                        0.6272, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Pr > F'],
                        0.4109, decimal=4)


@pytest.mark.smoke
def test_manova_no_formula_no_hypothesis():
    # Same as previous test only skipping formula interface
    exog = add_constant(pd.get_dummies(X[['Loc']], drop_first=True,
                                       dtype=float))
    endog = X[['Basal', 'Occ', 'Max']]
    mod = MANOVA(endog, exog)
    r = mod.mv_test()
    assert isinstance(r, MultivariateTestResults)


def test_manova_test_input_validation():
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    hypothesis = [('test', np.array([[1, 1, 1]]), None)]
    mod.mv_test(hypothesis)
    hypothesis = [('test', np.array([[1, 1]]), None)]
    assert_raises(ValueError, mod.mv_test, hypothesis)
    """
    assert_raises_regex(ValueError,
                        ('Contrast matrix L should have the same number of '
                         'columns as exog! 2 != 3'),
                        mod.mv_test, hypothesis)
    """
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1], [1]]))]
    mod.mv_test(hypothesis)
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1]]))]
    assert_raises(ValueError, mod.mv_test, hypothesis)
    """
    assert_raises_regex(ValueError,
                        ('Transform matrix M should have the same number of '
                         'rows as the number of columns of endog! 2 != 3'),
                        mod.mv_test, hypothesis)
    """

def test_endog_1D_array():
    assert_raises(ValueError, MANOVA.from_formula, 'Basal ~ Loc', X)


def test_manova_demeaned():
    # see last example in #8713
    # If a term has no effect, all eigenvalues below threshold, then computaion
    # raised numpy exception with empty arrays.
    # currently we have an option to skip the intercept test, but don't handle
    # empty arrays directly
    ng = 5
    loc = ["Basal", "Occ", "Max"] * ng
    y1 = (np.random.randn(ng, 3) + [0, 0.5, 1]).ravel()
    y2 = (np.random.randn(ng, 3) + [0.25, 0.75, 1]).ravel()
    y3 = (np.random.randn(ng, 3) + [0.3, 0.6, 1]).ravel()
    dta = pd.DataFrame(dict(Loc=loc, Basal=y1, Occ=y2, Max=y3))
    mod = MANOVA.from_formula('Basal + Occ + Max ~ C(Loc, Helmert)', data=dta)
    res1 = mod.mv_test()

    # subtract sample means to have insignificant intercept
    means = dta[["Basal", "Occ", "Max"]].mean()
    dta[["Basal", "Occ", "Max"]] = dta[["Basal", "Occ", "Max"]] - means
    mod = MANOVA.from_formula('Basal + Occ + Max ~ C(Loc, Helmert)', data=dta)
    res2 = mod.mv_test(skip_intercept_test=True)

    stat1 = res1.results["C(Loc, Helmert)"]["stat"].to_numpy(float)
    stat2 = res2.results["C(Loc, Helmert)"]["stat"].to_numpy(float)
    assert_allclose(stat1, stat2, rtol=1e-10)
