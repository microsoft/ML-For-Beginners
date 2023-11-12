"""
Author: Samuel Scherrer
"""
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN

from itertools import product
import json
import pathlib

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace

# This contains tests for the exponential smoothing implementation in
# tsa/exponential_smoothing/ets.py.
#
# Tests are mostly done by comparing results with the R implementation in the
# package forecast for the datasets `oildata` (non-seasonal) and `austourists`
# (seasonal).
#
# Therefore, a parametrized pytest fixture ``setup_model`` is provided, which
# returns a constructed model, model parameters from R in the format expected
# by ETSModel, and a dictionary of reference results. Use like this:
#
#     def test_<testname>(setup_model):
#         model, params, results_R = setup_model
#         # perform some tests
#         ...

###############################################################################
# UTILS
###############################################################################

# Below I define parameter lists for all possible model and data combinations
# (for data, see below). These are used for parametrizing the pytest fixture
# ``setup_model``, which should be used for all tests comparing to R output.


def remove_invalid_models_from_list(modellist):
    # remove invalid models (no trend but damped)
    for i, model in enumerate(modellist):
        if model[1] is None and model[3]:
            del modellist[i]


ERRORS = ("add", "mul")
TRENDS = ("add", "mul", None)
SEASONALS = ("add", "mul", None)
DAMPED = (True, False)

MODELS_DATA_SEASONAL = list(
    product(ERRORS, TRENDS, ("add", "mul"), DAMPED, ("austourists",),)
)
MODELS_DATA_NONSEASONAL = list(
    product(ERRORS, TRENDS, (None,), DAMPED, ("oildata",),)
)
remove_invalid_models_from_list(MODELS_DATA_SEASONAL)
remove_invalid_models_from_list(MODELS_DATA_NONSEASONAL)


def short_model_name(error, trend, seasonal, damped=False):
    short_name = {"add": "A", "mul": "M", None: "N", True: "d", False: ""}
    return (
        short_name[error]
        + short_name[trend]
        + short_name[damped]
        + short_name[seasonal]
    )


ALL_MODELS_AND_DATA = MODELS_DATA_NONSEASONAL + MODELS_DATA_SEASONAL
ALL_MODEL_IDS = [
    short_model_name(*mod[:3], mod[3]) for mod in ALL_MODELS_AND_DATA
]


@pytest.fixture(params=ALL_MODELS_AND_DATA, ids=ALL_MODEL_IDS)
def setup_model(
    request,
    austourists,
    oildata,
    ets_austourists_fit_results_R,
    ets_oildata_fit_results_R,
):
    params = request.param
    error, trend, seasonal, damped = params[0:4]
    data = params[4]
    if data == "austourists":
        data = austourists
        seasonal_periods = 4
        results = ets_austourists_fit_results_R[damped]
    else:
        data = oildata
        seasonal_periods = None
        results = ets_oildata_fit_results_R[damped]

    name = short_model_name(error, trend, seasonal)
    if name not in results:
        pytest.skip(f"model {name} not implemented or not converging in R")

    results_R = results[name]
    params = get_params_from_R(results_R)

    model = ETSModel(
        data,
        seasonal_periods=seasonal_periods,
        error=error,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped,
    )

    return model, params, results_R


@pytest.fixture
def austourists_model(austourists):
    return ETSModel(
        austourists,
        seasonal_periods=4,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
    )


@pytest.fixture
def austourists_model_fit(austourists_model):
    return austourists_model.fit(disp=False)


@pytest.fixture
def oildata_model(oildata):
    return ETSModel(oildata, error="add", trend="add", damped_trend=True,)


#############################################################################
# DATA
#############################################################################


@pytest.fixture
def austourists():
    # austourists dataset from fpp2 package
    # https://cran.r-project.org/web/packages/fpp2/index.html
    data = [
        30.05251300,
        19.14849600,
        25.31769200,
        27.59143700,
        32.07645600,
        23.48796100,
        28.47594000,
        35.12375300,
        36.83848500,
        25.00701700,
        30.72223000,
        28.69375900,
        36.64098600,
        23.82460900,
        29.31168300,
        31.77030900,
        35.17787700,
        19.77524400,
        29.60175000,
        34.53884200,
        41.27359900,
        26.65586200,
        28.27985900,
        35.19115300,
        42.20566386,
        24.64917133,
        32.66733514,
        37.25735401,
        45.24246027,
        29.35048127,
        36.34420728,
        41.78208136,
        49.27659843,
        31.27540139,
        37.85062549,
        38.83704413,
        51.23690034,
        31.83855162,
        41.32342126,
        42.79900337,
        55.70835836,
        33.40714492,
        42.31663797,
        45.15712257,
        59.57607996,
        34.83733016,
        44.84168072,
        46.97124960,
        60.01903094,
        38.37117851,
        46.97586413,
        50.73379646,
        61.64687319,
        39.29956937,
        52.67120908,
        54.33231689,
        66.83435838,
        40.87118847,
        51.82853579,
        57.49190993,
        65.25146985,
        43.06120822,
        54.76075713,
        59.83447494,
        73.25702747,
        47.69662373,
        61.09776802,
        66.05576122,
    ]
    index = pd.date_range("1999-01-01", "2015-12-31", freq="Q")
    return pd.Series(data, index)


@pytest.fixture
def oildata():
    # oildata dataset from fpp2 package
    # https://cran.r-project.org/web/packages/fpp2/index.html
    data = [
        111.0091346,
        130.8284341,
        141.2870879,
        154.2277747,
        162.7408654,
        192.1664835,
        240.7997253,
        304.2173901,
        384.0045673,
        429.6621566,
        359.3169299,
        437.2518544,
        468.4007898,
        424.4353365,
        487.9794299,
        509.8284478,
        506.3472527,
        340.1842374,
        240.2589210,
        219.0327876,
        172.0746632,
        252.5900922,
        221.0710774,
        276.5187735,
        271.1479517,
        342.6186005,
        428.3558357,
        442.3945534,
        432.7851482,
        437.2497186,
        437.2091599,
        445.3640981,
        453.1950104,
        454.4096410,
        422.3789058,
        456.0371217,
        440.3866047,
        425.1943725,
        486.2051735,
        500.4290861,
        521.2759092,
        508.9476170,
        488.8888577,
        509.8705750,
        456.7229123,
        473.8166029,
        525.9508706,
        549.8338076,
        542.3404698,
    ]
    return pd.Series(data, index=pd.date_range("1965", "2013", freq="AS"))


#############################################################################
# REFERENCE RESULTS
#############################################################################


def obtain_R_results(path):
    with path.open("r", encoding="utf-8") as f:
        R_results = json.load(f)

    # remove invalid models
    results = {}
    for damped in R_results:
        new_key = damped == "TRUE"
        results[new_key] = {}
        for model in R_results[damped]:
            if len(R_results[damped][model]):
                results[new_key][model] = R_results[damped][model]

    # get correct types
    for damped in results:
        for model in results[damped]:
            for key in ["alpha", "beta", "gamma", "phi", "sigma2"]:
                results[damped][model][key] = float(
                    results[damped][model][key][0]
                )
            for key in [
                "states",
                "initstate",
                "residuals",
                "fitted",
                "forecast",
                "simulation",
            ]:
                results[damped][model][key] = np.asarray(
                    results[damped][model][key]
                )
    return results


@pytest.fixture
def ets_austourists_fit_results_R():
    """
    Dictionary of ets fit results obtained with script ``results/fit_ets.R``.
    """
    path = (
        pathlib.Path(__file__).parent
        / "results"
        / "fit_ets_results_seasonal.json"
    )
    return obtain_R_results(path)


@pytest.fixture
def ets_oildata_fit_results_R():
    """
    Dictionary of ets fit results obtained with script ``results/fit_ets.R``.
    """
    path = (
        pathlib.Path(__file__).parent
        / "results"
        / "fit_ets_results_nonseasonal.json"
    )
    return obtain_R_results(path)


def fit_austourists_with_R_params(model, results_R, set_state=False):
    """
    Fit the model with params as found by R's forecast package
    """
    params = get_params_from_R(results_R)
    with model.fix_params(dict(zip(model.param_names, params))):
        fit = model.fit(disp=False)

    if set_state:
        states_R = get_states_from_R(results_R, model._k_states)
        fit.states = states_R
    return fit


def get_params_from_R(results_R):
    # get params from R
    params = [results_R[name] for name in ["alpha", "beta", "gamma", "phi"]]
    # in R, initial states are order l[-1], b[-1], s[-1], s[-2], ..., s[-m]
    params += list(results_R["initstate"])
    params = list(filter(np.isfinite, params))
    return params


def get_states_from_R(results_R, k_states):
    if k_states > 1:
        xhat_R = results_R["states"][1:, 0:k_states]
    else:
        xhat_R = results_R["states"][1:]
        xhat_R = np.reshape(xhat_R, (len(xhat_R), 1))
    return xhat_R


#############################################################################
# BASIC TEST CASES
#############################################################################


def test_fit_model_austouritsts(setup_model):
    model, params, results_R = setup_model
    model.fit(disp=False)


#############################################################################
# TEST OF MODEL EQUATIONS VS R
#############################################################################


def test_smooth_vs_R(setup_model):
    model, params, results_R = setup_model

    yhat, xhat = model.smooth(params, return_raw=True)

    yhat_R = results_R["fitted"]
    xhat_R = get_states_from_R(results_R, model._k_states)

    assert_allclose(xhat, xhat_R, rtol=1e-5, atol=1e-5)
    assert_allclose(yhat, yhat_R, rtol=1e-5, atol=1e-5)


def test_residuals_vs_R(setup_model):
    model, params, results_R = setup_model

    yhat = model.smooth(params, return_raw=True)[0]

    residuals = model._residuals(yhat)
    assert_allclose(residuals, results_R["residuals"], rtol=1e-5, atol=1e-5)


def test_loglike_vs_R(setup_model):
    model, params, results_R = setup_model

    loglike = model.loglike(params)
    # the calculation of log likelihood in R is only up to a constant:
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R["loglik"][0] + const

    assert_allclose(loglike, loglike_R, rtol=1e-5, atol=1e-5)


def test_forecast_vs_R(setup_model):
    model, params, results_R = setup_model

    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    fcast = fit.forecast(4)
    expected = np.asarray(results_R["forecast"])

    assert_allclose(expected, fcast.values, rtol=1e-3, atol=1e-4)


def test_simulate_vs_R(setup_model):
    model, params, results_R = setup_model

    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    innov = np.asarray([[1.76405235, 0.40015721, 0.97873798, 2.2408932]]).T
    sim = fit.simulate(4, anchor="end", repetitions=1, random_errors=innov)
    expected = np.asarray(results_R["simulation"])

    assert_allclose(expected, sim.values, rtol=1e-5, atol=1e-5)


def test_fit_vs_R(setup_model, reset_randomstate):
    model, params, results_R = setup_model

    if PLATFORM_WIN and model.short_name == "AAdA":
        start = params
    else:
        start = None
    fit = model.fit(disp=True, pgtol=1e-8, start_params=start)

    # check log likelihood: we want to have a fit that is better, i.e. a fit
    # that has a **higher** log-likelihood
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R["loglik"][0] + const
    loglike = fit.llf
    try:
        assert loglike >= loglike_R - 1e-4
    except AssertionError:
        fit = model.fit(disp=True, pgtol=1e-8, start_params=params)
        loglike = fit.llf
        try:
            assert loglike >= loglike_R - 1e-4
        except AssertionError:
            if PLATFORM_LINUX32:
                # Linux32 often fails to produce the correct solution.
                # Fixing this is low priority given the rareness of
                # its application
                pytest.xfail("Known to fail on 32-bit Linux")
            else:
                raise


def test_predict_vs_R(setup_model):
    model, params, results_R = setup_model
    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    n = fit.nobs
    prediction = fit.predict(end=n + 3, dynamic=n)

    yhat_R = results_R["fitted"]
    assert_allclose(prediction[:n], yhat_R, rtol=1e-5, atol=1e-5)

    forecast_R = results_R["forecast"]
    assert_allclose(prediction[n:], forecast_R, rtol=1e-3, atol=1e-4)


#############################################################################
# OTHER TESTS
#############################################################################


def test_initialization_known(austourists):
    initial_level, initial_trend = [36.46466837, 34.72584983]
    model = ETSModel(
        austourists,
        error="add",
        trend="add",
        damped_trend=True,
        initialization_method="known",
        initial_level=initial_level,
        initial_trend=initial_trend,
    )
    internal_params = model._internal_params(model._start_params)
    assert initial_level == internal_params[4]
    assert initial_trend == internal_params[5]
    assert internal_params[6] == 0


def test_initialization_heuristic(oildata):
    model_estimated = ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    )
    model_heuristic = ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
        initialization_method="heuristic",
    )
    fit_estimated = model_estimated.fit(disp=False)
    fit_heuristic = model_heuristic.fit(disp=False)
    yhat_estimated = fit_estimated.fittedvalues.values
    yhat_heuristic = fit_heuristic.fittedvalues.values

    # this test is mostly just to see if it works, so we only test whether the
    # result is not totally off
    assert_allclose(yhat_estimated[10:], yhat_heuristic[10:], rtol=0.5)


def test_bounded_fit(oildata):
    beta = [0.99, 0.99]
    model1 = ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
        bounds={"smoothing_trend": beta},
    )
    fit1 = model1.fit(disp=False)
    assert fit1.smoothing_trend == 0.99

    # same using with fix_params semantic
    model2 = ETSModel(oildata, error="add", trend="add", damped_trend=True,)
    with model2.fix_params({"smoothing_trend": 0.99}):
        fit2 = model2.fit(disp=False)
    assert fit2.smoothing_trend == 0.99
    assert_allclose(fit1.params, fit2.params)
    fit2.summary()  # check if summary runs without failing

    # using fit_constrained
    fit3 = model2.fit_constrained({"smoothing_trend": 0.99})
    assert fit3.smoothing_trend == 0.99
    assert_allclose(fit1.params, fit3.params)
    fit3.summary()


def test_seasonal_periods(austourists):
    # test auto-deduction of period
    model = ETSModel(austourists, error="add", trend="add", seasonal="add")
    assert model.seasonal_periods == 4

    # test if seasonal period raises error
    try:
        model = ETSModel(austourists, seasonal="add", seasonal_periods=0)
    except ValueError:
        pass


def test_simulate_keywords(austourists_model_fit):
    """
    check whether all keywords are accepted and work without throwing errors.
    """
    fit = austourists_model_fit

    # test anchor
    assert_almost_equal(
        fit.simulate(4, anchor=-1, random_state=0).values,
        fit.simulate(4, anchor="2015-12-31", random_state=0).values,
    )
    assert_almost_equal(
        fit.simulate(4, anchor="end", random_state=0).values,
        fit.simulate(4, anchor="2015-12-31", random_state=0).values,
    )

    # test different random error options
    fit.simulate(4, repetitions=10)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm())
    fit.simulate(4, repetitions=10, random_errors=np.random.randn(4, 10))
    fit.simulate(4, repetitions=10, random_errors="bootstrap")

    # test seeding
    res = fit.simulate(4, repetitions=10, random_state=10).values
    res2 = fit.simulate(
        4, repetitions=10, random_state=np.random.RandomState(10)
    ).values
    assert np.all(res == res2)


def test_predict_ranges(austourists_model_fit):
    # in total 68 observations
    fit = austourists_model_fit

    # first prediction is 0, last is 10 -> 11 predictions
    pred = fit.predict(start=0, end=10)
    assert len(pred) == 11

    pred = fit.predict(start=10, end=20)
    assert len(pred) == 11

    pred = fit.predict(start=10, dynamic=10, end=30)
    assert len(pred) == 21

    # try boolean dynamic
    pred = fit.predict(start=0, dynamic=True, end=70)
    assert len(pred) == 71
    pred = fit.predict(start=0, dynamic=True, end=70)
    assert len(pred) == 71

    # try only out oof sample prediction
    pred = fit.predict(start=80, end=84)
    assert len(pred) == 5


def test_summary(austourists_model):
    # just try to run summary to see if it works
    fit = austourists_model.fit(disp=False)
    fit.summary()

    # now without estimated initial states
    austourists_model.set_initialization_method("heuristic")
    fit = austourists_model.fit(disp=False)
    fit.summary()

    # and with fixed params
    fit = austourists_model.fit_constrained({"smoothing_trend": 0.9})
    fit.summary()


def test_score(austourists_model_fit):
    score_cs = austourists_model_fit.model.score(austourists_model_fit.params)
    score_fd = austourists_model_fit.model.score(
        austourists_model_fit.params,
        approx_complex_step=False,
        approx_centered=True,
    )
    assert_almost_equal(score_cs, score_fd, 4)


def test_hessian(austourists_model_fit):
    # The hessian approximations are not very consistent, but the test makes
    # sure they run
    austourists_model_fit.model.hessian(austourists_model_fit.params)
    austourists_model_fit.model.hessian(
        austourists_model_fit.params,
        approx_complex_step=False,
        approx_centered=True,
    )


def test_prediction_results(austourists_model_fit):
    # simple test case starting at 0
    pred = austourists_model_fit.get_prediction(start=0, dynamic=30, end=40,)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 41
    assert np.all(~np.isnan(summary["mean"]))

    # simple test case starting at not 0
    pred = austourists_model_fit.get_prediction(start=10, dynamic=30, end=40)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 31
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample prediction
    pred = austourists_model_fit.get_prediction(start=0, dynamic=30, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 81
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting in-sample
    pred = austourists_model_fit.get_prediction(start=67, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 14
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting at end of sample
    pred = austourists_model_fit.get_prediction(start=68, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 13
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting just out of sample
    pred = austourists_model_fit.get_prediction(start=69, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 12
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting long out of sample
    pred = austourists_model_fit.get_prediction(start=79, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 2
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, `start`== `end`
    pred = austourists_model_fit.get_prediction(start=80, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 1
    assert np.all(~np.isnan(summary["mean"]))


@pytest.fixture
def statespace_comparison(austourists):
    ets_model = ETSModel(
        austourists,
        seasonal_periods=4,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
    )
    ets_results = ets_model.fit(disp=False)

    statespace_model = statespace.ExponentialSmoothing(
        austourists,
        trend=True,
        damped_trend=True,
        seasonal=4,
        initialization_method="known",
        initial_level=ets_results.initial_level,
        initial_trend=ets_results.initial_trend,
        # See GH 7893
        initial_seasonal=ets_results.initial_seasonal[::-1],
    )
    with statespace_model.fix_params(
        {
            "smoothing_level": ets_results.smoothing_level,
            "smoothing_trend": ets_results.smoothing_trend,
            "smoothing_seasonal": ets_results.smoothing_seasonal,
            "damping_trend": ets_results.damping_trend,
        }
    ):
        statespace_results = statespace_model.fit()
    ets_results.test_serial_correlation("ljungbox")
    statespace_results.test_serial_correlation("ljungbox")
    return ets_results, statespace_results


def test_results_vs_statespace(statespace_comparison):
    ets_results, statespace_results = statespace_comparison

    assert_almost_equal(ets_results.llf, statespace_results.llf)
    assert_almost_equal(ets_results.scale, statespace_results.scale)
    assert_almost_equal(
        ets_results.fittedvalues.values, statespace_results.fittedvalues.values
    )

    # compare diagnostics
    assert_almost_equal(
        ets_results.test_serial_correlation(method="ljungbox"),
        statespace_results.test_serial_correlation(method="ljungbox"),
    )
    assert_almost_equal(
        ets_results.test_normality(method="jarquebera"),
        statespace_results.test_normality(method="jarquebera"),
    )

    # heteroskedasticity is somewhat different, because of burn in period?
    ets_het = ets_results.test_heteroskedasticity(method="breakvar")[0]
    statespace_het = statespace_results.test_heteroskedasticity(
        method="breakvar"
    )[0]
    # het[0] is test statistic, het[1] p-value
    if not PLATFORM_LINUX32:
        # Skip on Linux-32 bit due to random failures. These values are not
        # close at all in any way so it isn't clear what this is testing
        assert_allclose(ets_het[0], statespace_het[0], rtol=0.2)
        assert_allclose(ets_het[1], statespace_het[1], rtol=0.7)


def test_prediction_results_vs_statespace(statespace_comparison):
    ets_results, statespace_results = statespace_comparison

    # comparison of two predictions
    ets_pred = ets_results.get_prediction(start=10, dynamic=10, end=40)
    statespace_pred = statespace_results.get_prediction(
        start=10, dynamic=10, end=40
    )

    statespace_summary = statespace_pred.summary_frame()
    ets_summary = ets_pred.summary_frame()

    # import matplotlib.pyplot as plt
    # plt.switch_backend('TkAgg')
    # plt.plot(ets_summary["mean"] - statespace_summary["mean"])
    # plt.grid()
    # plt.show()

    assert_almost_equal(
        ets_summary["mean"].values[:-10],
        statespace_summary["mean"].values[:-10],
    )

    assert_almost_equal(
        ets_summary["mean"].values[-10:],
        statespace_summary["mean"].values[-10:],
        4,
    )

    # comparison of dynamic prediction at end of sample -> this works
    ets_pred = ets_results.get_prediction(start=60, end=80,)
    statespace_pred = statespace_results.get_prediction(start=60, end=80)
    statespace_summary = statespace_pred.summary_frame()
    ets_summary = ets_pred.summary_frame()

    assert_almost_equal(
        ets_summary["mean"].values, statespace_summary["mean"].values, 4
    )


@pytest.mark.skip
def test_prediction_results_slow_AAN(oildata):
    # slow test with high number of simulation repetitions for comparison
    # Note: runs succesfull with specified tolerance
    fit = ETSModel(oildata, error="add", trend="add").fit(disp=False)

    pred_exact = fit.get_prediction(start=40, end=55)
    summary_exact = pred_exact.summary_frame()

    pred_sim = fit.get_prediction(
        start=40,
        end=55,
        simulate_repetitions=int(1e6),
        random_state=11,
        method="simulated",
    )
    summary_sim = pred_sim.summary_frame()
    # check if mean converges to expected mean
    assert_allclose(
        summary_sim["mean"].values,
        summary_sim["mean_numerical"].values,
        rtol=1e-3,
        atol=1e-3,
    )

    import matplotlib.pyplot as plt

    plt.switch_backend("TkAgg")
    for i in range(1000):
        plt.plot(
            pred_sim._results.simulation_results.iloc[:, i],
            color="grey",
            alpha=0.1,
        )
    plt.plot(oildata[40:], "-", label="data")
    plt.plot(summary_exact["mean"], "--", label="mean")
    plt.plot(summary_sim["pi_lower"], ":", label="sim lower")
    plt.plot(summary_exact["pi_lower"], ".-", label="exact lower")
    plt.plot(summary_sim["pi_upper"], ":", label="sim upper")
    plt.plot(summary_exact["pi_upper"], ".-", label="exact upper")
    # plt.legend()
    plt.show()

    # check if prediction intervals are equal
    assert_allclose(
        summary_sim["pi_lower"].values,
        summary_exact["pi_lower"].values,
        rtol=1e-4,
        atol=1e-4,
    )

    assert_allclose(
        summary_sim["pi_upper"].values,
        summary_exact["pi_upper"].values,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.skip
def test_prediction_results_slow_AAdA(austourists):
    # slow test with high number of simulation repetitions for comparison
    # Note: succesfull with specified tolerance
    fit = ETSModel(
        austourists,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
        seasonal_periods=4,
    ).fit(disp=False)
    pred_exact = fit.get_prediction(start=60, end=75)
    summary_exact = pred_exact.summary_frame()

    pred_sim = fit.get_prediction(
        start=60,
        end=75,
        simulate_repetitions=int(1e6),
        random_state=11,
        method="simulated",
    )
    summary_sim = pred_sim.summary_frame()
    # check if mean converges to expected mean
    assert_allclose(
        summary_sim["mean"].values,
        summary_sim["mean_numerical"].values,
        rtol=1e-3,
        atol=1e-3,
    )

    import matplotlib.pyplot as plt

    plt.switch_backend("TkAgg")
    for i in range(1000):
        plt.plot(
            pred_sim._results.simulation_results.iloc[:, i],
            color="grey",
            alpha=0.1,
        )
    plt.plot(fit.endog[60:], "-", label="data")
    plt.plot(summary_exact["mean"], "--", label="mean")
    plt.plot(summary_sim["pi_lower"], ":", label="sim lower")
    plt.plot(summary_exact["pi_lower"], ".-", label="exact lower")
    plt.plot(summary_sim["pi_upper"], ":", label="sim upper")
    plt.plot(summary_exact["pi_upper"], ".-", label="exact upper")
    plt.show()

    # check if prediction intervals are equal
    assert_allclose(
        summary_sim["pi_lower"].values,
        summary_exact["pi_lower"].values,
        rtol=2e-2,
        atol=1e-4,
    )

    assert_allclose(
        summary_sim["pi_upper"].values,
        summary_exact["pi_upper"].values,
        rtol=2e-2,
        atol=1e-4,
    )


def test_convergence_simple():
    # issue 6883
    gen = np.random.RandomState(0)
    e = gen.standard_normal(12000)
    y = e.copy()
    for i in range(1, e.shape[0]):
        y[i] = y[i - 1] - 0.2 * e[i - 1] + e[i]
    y = y[200:]
    mod = holtwinters.ExponentialSmoothing(
        y, initialization_method="estimated"
    )
    res = mod.fit()
    ets_res = ETSModel(y).fit()

    # the smoothing level should be very similar, the initial state might be
    # different as it doesn't influence the final result too much
    assert_allclose(
        res.params["smoothing_level"],
        ets_res.smoothing_level,
        rtol=1e-4,
        atol=1e-4,
    )

    # the first few values are influenced by differences in initial state, so
    # we don't test them here
    assert_allclose(
        res.fittedvalues[10:], ets_res.fittedvalues[10:], rtol=1e-4, atol=1e-4
    )


def test_exact_prediction_intervals(austourists_model_fit):

    fit = austourists_model_fit._results

    class DummyModel:
        def __init__(self, short_name):
            self.short_name = short_name

    # compare AAdN with AAN
    fit.damping_trend = 1 - 1e-3
    fit.model = DummyModel("AAdN")
    steps = 5
    s_AAdN = fit._relative_forecast_variance(steps)
    fit.model = DummyModel("AAN")
    s_AAN = fit._relative_forecast_variance(steps)
    assert_almost_equal(s_AAdN, s_AAN, 2)

    # compare AAdA with AAA
    fit.damping_trend = 1 - 1e-3
    fit.model = DummyModel("AAdA")
    steps = 5
    s_AAdA = fit._relative_forecast_variance(steps)
    fit.model = DummyModel("AAA")
    s_AAA = fit._relative_forecast_variance(steps)
    assert_almost_equal(s_AAdA, s_AAA, 2)


def test_one_step_ahead(setup_model):
    model, params, results_R = setup_model
    model2 = ETSModel(
        pd.Series(model.endog),
        seasonal_periods=model.seasonal_periods,
        error=model.error,
        trend=model.trend,
        seasonal=model.seasonal,
        damped_trend=model.damped_trend,
    )
    res = model2.smooth(params)

    fcast1 = res.forecast(steps=1)
    fcast2 = res.forecast(steps=2)
    assert_allclose(fcast1.iloc[0], fcast2.iloc[0])

    pred1 = res.get_prediction(start=model2.nobs, end=model2.nobs,
                               simulate_repetitions=2)
    pred2 = res.get_prediction(start=model2.nobs, end=model2.nobs + 1,
                               simulate_repetitions=2)
    df1 = pred1.summary_frame(alpha=0.05)
    df2 = pred1.summary_frame(alpha=0.05)
    assert_allclose(df1.iloc[0, 0], df2.iloc[0, 0])


@pytest.mark.parametrize("trend", [None, "add"])
@pytest.mark.parametrize("seasonal", [None, "add"])
@pytest.mark.parametrize("nobs", [9, 10])
def test_estimated_initialization_short_data(oildata, trend, seasonal, nobs):
    # GH 7319
    res = ETSModel(
        oildata[:nobs],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=4,
        initialization_method='estimated'
    ).fit()
    assert ~np.any(np.isnan(res.params))


@pytest.mark.parametrize("method", ["estimated", "heuristic"])
def test_seasonal_order(reset_randomstate, method):
    seasonal = np.arange(12.0)
    time_series = np.array(list(seasonal) * 100)
    res = ETSModel(
        time_series,
        seasonal="add",
        seasonal_periods=12,
        initialization_method=method,
    ).fit()
    assert_allclose(
        res.initial_seasonal + res.initial_level,
        seasonal,
        atol=1e-4,
        rtol=1e-4,
    )
    assert res.mae < 1e-6


def test_aicc_0_dof():
    # GH8172
    endog = [109.0, 101.0, 104.0, 90.0, 105.0]

    model = ETSModel(
        endog=endog,
        initialization_method='known',
        initial_level=100.0,
        initial_trend=0.0,
        error='add',
        trend='add',
        damped_trend=True
    )
    aicc = model.fit().aicc
    assert not np.isfinite(aicc)
    assert aicc > 0
