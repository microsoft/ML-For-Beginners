import numpy as np
import pandas

from statsmodels.tools import data


def test_missing_data_pandas():
    """
    Fixes GH: #144
    """
    X = np.random.random((10, 5))
    X[1, 2] = np.nan
    df = pandas.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(rnames.tolist(), [0, 2, 3, 4, 5, 6, 7, 8, 9])


def test_dataframe():
    X = np.random.random((10, 5))
    df = pandas.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(vals, df.values)
    np.testing.assert_equal(rnames.tolist(), df.index.tolist())
    np.testing.assert_equal(cnames, df.columns.tolist())


def test_patsy_577():
    X = np.random.random((10, 2))
    df = pandas.DataFrame(X, columns=["var1", "var2"])
    from patsy import dmatrix

    endog = dmatrix("var1 - 1", df)
    np.testing.assert_(data._is_using_patsy(endog, None))
    exog = dmatrix("var2 - 1", df)
    np.testing.assert_(data._is_using_patsy(endog, exog))
