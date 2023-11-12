'''examples to check summary, not converted to tests yet


'''

import numpy as np  # noqa: F401
import pytest
from numpy.testing import assert_equal

from statsmodels.datasets import macrodata
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS


def test_escaped_variable_name():
    # Rename 'cpi' column to 'CPI_'
    data = macrodata.load().data
    data.rename(columns={'cpi': 'CPI_'}, inplace=True)

    mod = OLS.from_formula('CPI_ ~ 1 + np.log(realgdp)', data=data)
    res = mod.fit()
    assert 'CPI\\_' in res.summary().as_latex()
    assert 'CPI_' in res.summary().as_text()


def test_wrong_len_xname(reset_randomstate):
    y = np.random.randn(100)
    x = np.random.randn(100, 2)
    res = OLS(y, x).fit()
    with pytest.raises(ValueError):
        res.summary(xname=['x1'])
    with pytest.raises(ValueError):
        res.summary(xname=['x1', 'x2', 'x3'])


class TestSummaryLatex:
    def test__repr_latex_(self):
        desired = r'''
\begin{center}
\begin{tabular}{lcccccc}
\toprule
               & \textbf{coef} & \textbf{std err} & \textbf{t} & \textbf{P$> |$t$|$} & \textbf{[0.025} & \textbf{0.975]}  \\
\midrule
\textbf{const} &       7.2248  &        0.866     &     8.346  &         0.000        &        5.406    &        9.044     \\
\textbf{x1}    &      -0.6609  &        0.177     &    -3.736  &         0.002        &       -1.033    &       -0.289     \\
\bottomrule
\end{tabular}
\end{center}
'''
        x = [1, 5, 7, 3, 5, 5, 8, 3, 3, 4, 6, 4, 2, 7, 4, 2, 1, 9, 2, 6]
        x = add_constant(x)
        y = [6, 4, 2, 7, 4, 2, 1, 9, 2, 6, 1, 5, 7, 3, 5, 5, 8, 3, 3, 4]
        reg = OLS(y, x).fit()

        actual = reg.summary().tables[1]._repr_latex_()
        actual = '\n%s\n' % actual
        assert_equal(actual, desired)


if __name__ == '__main__':

    from statsmodels.regression.tests.test_regression import TestOLS

    #def mytest():
    aregression = TestOLS()
    TestOLS.setup_class()
    results = aregression.res1
    r_summary = str(results.summary_old())
    print(r_summary)
    olsres = results

    print('\n\n')

    r_summary = str(results.summary())
    print(r_summary)
    print('\n\n')


    from statsmodels.discrete.tests.test_discrete import TestProbitNewton

    aregression = TestProbitNewton()
    TestProbitNewton.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)
    print('\n\n')

    probres = results

    from statsmodels.robust.tests.test_rlm import TestHampel

    aregression = TestHampel()
    #TestHampel.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)
    rlmres = results

    print('\n\n')

    from statsmodels.genmod.tests.test_glm import TestGlmBinomial

    aregression = TestGlmBinomial()
    #TestGlmBinomial.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)

    #print(results.summary2(return_fmt='latex'))
    #print(results.summary2(return_fmt='csv'))

    smry = olsres.summary()
    print(smry.as_csv())

#    import matplotlib.pyplot as plt
#    plt.plot(rlmres.model.endog,'o')
#    plt.plot(rlmres.fittedvalues,'-')
#
#    plt.show()
