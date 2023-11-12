import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

pdf_output = False


if pdf_output:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("test_mice.pdf")
else:
    pdf = None


def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)


def teardown_module():
    if pdf_output:
        pdf.close()


def gendat():
    """
    Create a data set with missing values.
    """

    np.random.seed(34243)

    n = 200
    p = 5

    exog = np.random.normal(size=(n, p))
    exog[:, 0] = exog[:, 1] - exog[:, 2] + 2*exog[:, 4]
    exog[:, 0] += np.random.normal(size=n)
    exog[:, 2] = 1*(exog[:, 2] > 0)

    endog = exog.sum(1) + np.random.normal(size=n)

    df = pd.DataFrame(exog)
    df.columns = ["x%d" % k for k in range(1, p+1)]

    df["y"] = endog

    # loc is inclusive of right end, so needed to lower index by 1
    df.loc[0:59, "x1"] = np.nan
    df.loc[0:39, "x2"] = np.nan
    df.loc[10:29:2, "x3"] = np.nan
    df.loc[20:49:3, "x4"] = np.nan
    df.loc[40:44, "x5"] = np.nan
    df.loc[30:99:2, "y"] = np.nan

    return df


class TestMICEData:

    def test_default(self):
        # Test with all defaults.

        df = gendat()
        orig = df.copy()
        mx = pd.notnull(df)
        imp_data = mice.MICEData(df)
        nrow, ncol = df.shape

        assert_allclose(imp_data.ix_miss['x1'], np.arange(60))
        assert_allclose(imp_data.ix_obs['x1'], np.arange(60, 200))
        assert_allclose(imp_data.ix_miss['x2'], np.arange(40))
        assert_allclose(imp_data.ix_miss['x3'], np.arange(10, 30, 2))
        assert_allclose(imp_data.ix_obs['x3'],
                        np.concatenate((np.arange(10),
                                        np.arange(11, 30, 2),
                                        np.arange(30, 200))))
        assert_equal([set(imp_data.data[col]) for col in imp_data.data],
                     [set(df[col].dropna()) for col in df])

        for k in range(3):
            imp_data.update_all()
            assert_equal(imp_data.data.shape[0], nrow)
            assert_equal(imp_data.data.shape[1], ncol)
            assert_allclose(orig[mx], imp_data.data[mx])
            assert_equal([set(imp_data.data[col]) for col in imp_data.data],
                         [set(df[col].dropna()) for col in df])

        fml = 'x1 ~ x2 + x3 + x4 + x5 + y'
        assert_equal(imp_data.conditional_formula['x1'], fml)

        assert_equal(imp_data._cycle_order, ['x5', 'x3', 'x4', 'y', 'x2', 'x1'])

        # Should make a copy
        assert not (df is imp_data.data)

        (endog_obs, exog_obs, exog_miss,
         predict_obs_kwds, predict_miss_kwds) = imp_data.get_split_data('x3')
        assert_equal(len(endog_obs), 190)
        assert_equal(exog_obs.shape, [190, 6])
        assert_equal(exog_miss.shape, [10, 6])

    def test_settingwithcopywarning(self):
        "Test that MICEData does not throw a SettingWithCopyWarning when imputing (https://github.com/statsmodels/statsmodels/issues/5430)"

        df = gendat()
        # There need to be some ints in here for the error to be thrown
        df['intcol'] = np.arange(len(df))
        df['intcol'] = df.intcol.astype('int32')

        miceData = mice.MICEData(df)

        with pd.option_context('mode.chained_assignment', 'warn'):
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                miceData.update_all()
                # Only include pandas warnings. There are many from patsy
                # and sometimes warnings from other packages here
                ws = [w for w in ws if "\\pandas\\" in w.filename]
                assert len(ws) == 0

    def test_next_sample(self):

        df = gendat()
        imp_data = mice.MICEData(df)

        all_x = []
        for j in range(2):
            x = imp_data.next_sample()
            assert isinstance(x, pd.DataFrame)
            assert_equal(df.shape, x.shape)
            all_x.append(x)

        # The returned dataframes are all the same object
        assert all_x[0] is all_x[1]


    def test_pertmeth(self):
        # Test with specified perturbation method.

        df = gendat()
        orig = df.copy()
        mx = pd.notnull(df)
        nrow, ncol = df.shape

        for pert_meth in "gaussian", "boot":

            imp_data = mice.MICEData(df, perturbation_method=pert_meth)

            for k in range(2):
                imp_data.update_all()
                assert_equal(imp_data.data.shape[0], nrow)
                assert_equal(imp_data.data.shape[1], ncol)
                assert_allclose(orig[mx], imp_data.data[mx])

        assert_equal(imp_data._cycle_order, ['x5', 'x3', 'x4', 'y', 'x2', 'x1'])


    def test_phreg(self):

        np.random.seed(8742)
        n = 300
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        event_time = np.random.exponential(size=n) * np.exp(x1)
        obs_time = np.random.exponential(size=n)
        time = np.where(event_time < obs_time, event_time, obs_time)
        status = np.where(time == event_time, 1, 0)
        df = pd.DataFrame({"time": time, "status": status, "x1": x1, "x2": x2})
        df.loc[10:40, 'time'] = np.nan
        df.loc[10:40, 'status'] = np.nan
        df.loc[30:50, 'x1'] = np.nan
        df.loc[40:60, 'x2'] = np.nan

        from statsmodels.duration.hazard_regression import PHReg

        # Save the dataset size at each iteration.
        hist = []

        def cb(imp):
            hist.append(imp.data.shape)

        for pm in "gaussian", "boot":
            idata = mice.MICEData(df, perturbation_method=pm, history_callback=cb)
            idata.set_imputer("time", "0 + x1 + x2", model_class=PHReg,
                              init_kwds={"status": mice.PatsyFormula("status")},
                              predict_kwds={"pred_type": "hr"},
                              perturbation_method=pm)

            x = idata.next_sample()
            assert isinstance(x, pd.DataFrame)

        assert all([val == (299, 4) for val in hist])

    def test_set_imputer(self):
        # Test with specified perturbation method.

        from statsmodels.regression.linear_model import RegressionResultsWrapper
        from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

        df = gendat()
        orig = df.copy()
        mx = pd.notnull(df)
        nrow, ncol = df.shape

        imp_data = mice.MICEData(df)
        imp_data.set_imputer('x1', 'x3 + x4 + x3*x4')
        imp_data.set_imputer('x2', 'x4 + I(x5**2)')
        imp_data.set_imputer('x3', model_class=sm.GLM,
                             init_kwds={"family": sm.families.Binomial()})

        imp_data.update_all()
        assert_equal(imp_data.data.shape[0], nrow)
        assert_equal(imp_data.data.shape[1], ncol)
        assert_allclose(orig[mx], imp_data.data[mx])
        for j in range(1, 6):
            if j == 3:
                assert_equal(isinstance(imp_data.models['x3'], sm.GLM), True)
                assert_equal(isinstance(imp_data.models['x3'].family, sm.families.Binomial), True)
                assert_equal(isinstance(imp_data.results['x3'], GLMResultsWrapper), True)
            else:
                assert_equal(isinstance(imp_data.models['x%d' % j], sm.OLS), True)
                assert_equal(isinstance(imp_data.results['x%d' % j], RegressionResultsWrapper), True)

        fml = 'x1 ~ x3 + x4 + x3*x4'
        assert_equal(imp_data.conditional_formula['x1'], fml)

        fml = 'x4 ~ x1 + x2 + x3 + x5 + y'
        assert_equal(imp_data.conditional_formula['x4'], fml)

        assert_equal(imp_data._cycle_order, ['x5', 'x3', 'x4', 'y', 'x2', 'x1'])


    @pytest.mark.matplotlib
    def test_plot_missing_pattern(self, close_figures):

        df = gendat()
        imp_data = mice.MICEData(df)

        for row_order in "pattern", "raw":
            for hide_complete_rows in False, True:
                for color_row_patterns in False, True:
                    plt.clf()
                    fig = imp_data.plot_missing_pattern(row_order=row_order,
                                      hide_complete_rows=hide_complete_rows,
                                      color_row_patterns=color_row_patterns)
                    close_or_save(pdf, fig)
                    close_figures()


    @pytest.mark.matplotlib
    def test_plot_bivariate(self, close_figures):

        df = gendat()
        imp_data = mice.MICEData(df)
        imp_data.update_all()

        plt.clf()
        for plot_points in False, True:
            fig = imp_data.plot_bivariate('x2', 'x4', plot_points=plot_points)
            fig.get_axes()[0].set_title('plot_bivariate')
            close_or_save(pdf, fig)
            close_figures()


    @pytest.mark.matplotlib
    def test_fit_obs(self, close_figures):

        df = gendat()
        imp_data = mice.MICEData(df)
        imp_data.update_all()

        plt.clf()
        for plot_points in False, True:
            fig = imp_data.plot_fit_obs('x4', plot_points=plot_points)
            fig.get_axes()[0].set_title('plot_fit_scatterplot')
            close_or_save(pdf, fig)
            close_figures()


    @pytest.mark.matplotlib
    def test_plot_imputed_hist(self, close_figures):

        df = gendat()
        imp_data = mice.MICEData(df)
        imp_data.update_all()

        plt.clf()
        for plot_points in False, True:
            fig = imp_data.plot_imputed_hist('x4')
            fig.get_axes()[0].set_title('plot_imputed_hist')
            close_or_save(pdf, fig)
            close_figures()


class TestMICE:

    def test_MICE(self):

        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE("y ~ x1 + x2 + x1:x2", sm.OLS, imp_data)
        result = mi.fit(1, 3)

        assert issubclass(result.__class__, mice.MICEResults)

        # Smoke test for results
        smr = result.summary()


    def test_MICE1(self):

        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE("y ~ x1 + x2 + x1:x2", sm.OLS, imp_data)

        from statsmodels.regression.linear_model import RegressionResultsWrapper

        for j in range(3):
            x = mi.next_sample()
            assert issubclass(x.__class__, RegressionResultsWrapper)


    def test_MICE1_regularized(self):

        df = gendat()
        imp = mice.MICEData(df, perturbation_method='boot')
        imp.set_imputer('x1', 'x2 + y', fit_kwds={'alpha': 1, 'L1_wt': 0})
        imp.update_all()


    def test_MICE2(self):

        from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE("x3 ~ x1 + x2", sm.GLM, imp_data,
                       init_kwds={"family": sm.families.Binomial()})

        for j in range(3):
            x = mi.next_sample()
            assert isinstance(x, GLMResultsWrapper)
            assert isinstance(x.family, sm.families.Binomial)

    @pytest.mark.slow
    def t_est_combine(self):

        np.random.seed(3897)
        x1 = np.random.normal(size=300)
        x2 = np.random.normal(size=300)
        y = x1 + x2 + np.random.normal(size=300)
        x1[0:100] = np.nan
        x2[250:] = np.nan
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        idata = mice.MICEData(df)
        mi = mice.MICE("y ~ x1 + x2", sm.OLS, idata, n_skip=20)
        result = mi.fit(10, 20)

        fmi = np.asarray([0.1778143, 0.11057262, 0.29626521])
        assert_allclose(result.frac_miss_info, fmi, atol=1e-5)

        params = np.asarray([-0.03486102, 0.96236808, 0.9970371])
        assert_allclose(result.params, params, atol=1e-5)

        tvalues = np.asarray([-0.54674776, 15.28091069, 13.61359403])
        assert_allclose(result.tvalues, tvalues, atol=1e-5)


def test_micedata_miss1():
    # test for #4375
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(50, 4))
    data.columns = ['var1', 'var2', 'var3', 'var4']
    # one column with a single missing value
    data.iloc[1, 1] = np.nan
    data.iloc[[1, 3], 2] = np.nan

    data_imp = mice.MICEData(data)
    data_imp.update_all()

    assert_equal(data_imp.data.isnull().values.sum(), 0)

    ix_miss = {'var1': np.array([], dtype=np.int64),
                 'var2': np.array([1], dtype=np.int64),
                 'var3': np.array([1, 3], dtype=np.int64),
                 'var4': np.array([], dtype=np.int64)}

    for k in ix_miss:
        assert_equal(data_imp.ix_miss[k], ix_miss[k])
