import datetime as dt

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_equal,
    assert_raises,
)
import pandas as pd
import pytest

from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
    ArmaProcess,
    arma_acf,
    arma_acovf,
    arma_generate_sample,
    arma_impulse_response,
    index2lpol,
    lpol2index,
    lpol_fiar,
    lpol_fima,
)
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import (
    armarep,  # benchmarkdata
)

arlist = [
    [1.0],
    [1, -0.9],  # ma representation will need many terms to get high precision
    [1, 0.9],
    [1, -0.9, 0.3],
]

malist = [[1.0], [1, 0.9], [1, -0.9], [1, 0.9, -0.3]]

DECIMAL_4 = 4


def test_arma_acovf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acovf([1, -phi], [1], N)
    # rep 2: manually
    rep2 = [1.0 * sigma * phi ** i / (1 - phi ** 2) for i in range(N)]
    assert_allclose(rep1, rep2)


def test_arma_acovf_persistent():
    # Test arma_acovf in case where there is a near-unit root.
    # .999 is high enough to trigger the "while ir[-1] > 5*1e-5:" clause,
    # but not high enough to trigger the "nobs_ir > 50000" clause.
    ar = np.array([1, -0.9995])
    ma = np.array([1])
    process = ArmaProcess(ar, ma)
    res = process.acovf(10)

    # Theoretical variance sig2 given by:
    # sig2 = .9995**2 * sig2 + 1
    sig2 = 1 / (1 - 0.9995 ** 2)

    corrs = 0.9995 ** np.arange(10)
    expected = sig2 * corrs
    assert_equal(res.ndim, 1)
    assert_allclose(res, expected)


def test_arma_acf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acf([1, -phi], [1], N)
    # rep 2: manually
    acovf = np.array(
        [1.0 * sigma * phi ** i / (1 - phi ** 2) for i in range(N)]
    )
    rep2 = acovf / (1.0 / (1 - phi ** 2))
    assert_allclose(rep1, rep2)


def test_arma_acf_compare_R_ARMAacf():
    # Test specific cases against output from R's ARMAacf
    bd_example_3_3_2 = arma_acf([1, -1, 0.25], [1, 1])
    assert_allclose(bd_example_3_3_2, results_arma_acf.bd_example_3_3_2)
    example_1 = arma_acf([1, -1, 0.25], [1, 1, 0.2])
    assert_allclose(example_1, results_arma_acf.custom_example_1)
    example_2 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3])
    assert_allclose(example_2, results_arma_acf.custom_example_2)
    example_3 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3, -0.35])
    assert_allclose(example_3, results_arma_acf.custom_example_3)
    example_4 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3, -0.35, -0.1])
    assert_allclose(example_4, results_arma_acf.custom_example_4)
    example_5 = arma_acf([1, -1, 0.25, -0.1], [1, 1, 0.2])
    assert_allclose(example_5, results_arma_acf.custom_example_5)
    example_6 = arma_acf([1, -1, 0.25, -0.1, 0.05], [1, 1, 0.2])
    assert_allclose(example_6, results_arma_acf.custom_example_6)
    example_7 = arma_acf([1, -1, 0.25, -0.1, 0.05, -0.02], [1, 1, 0.2])
    assert_allclose(example_7, results_arma_acf.custom_example_7)


def test_arma_acov_compare_theoretical_arma_acov():
    # Test against the older version of this function, which used a different
    # approach that nicely shows the theoretical relationship
    # See GH:5324 when this was removed for full version of the function
    # including documentation and inline comments

    def arma_acovf_historical(ar, ma, nobs=10):
        if np.abs(np.sum(ar) - 1) > 0.9:
            nobs_ir = max(1000, 2 * nobs)
        else:
            nobs_ir = max(100, 2 * nobs)
        ir = arma_impulse_response(ar, ma, leads=nobs_ir)
        while ir[-1] > 5 * 1e-5:
            nobs_ir *= 10
            ir = arma_impulse_response(ar, ma, leads=nobs_ir)
        if nobs_ir > 50000 and nobs < 1001:
            end = len(ir)
            acovf = np.array(
                [
                    np.dot(ir[: end - nobs - t], ir[t : end - nobs])
                    for t in range(nobs)
                ]
            )
        else:
            acovf = np.correlate(ir, ir, "full")[len(ir) - 1 :]
        return acovf[:nobs]

    assert_allclose(
        arma_acovf([1, -0.5], [1, 0.2]),
        arma_acovf_historical([1, -0.5], [1, 0.2]),
    )
    assert_allclose(
        arma_acovf([1, -0.99], [1, 0.2]),
        arma_acovf_historical([1, -0.99], [1, 0.2]),
    )


def _manual_arma_generate_sample(ar, ma, eta):
    T = len(eta)
    ar = ar[::-1]
    ma = ma[::-1]
    p, q = len(ar), len(ma)
    rep2 = [0] * max(p, q)  # initialize with zeroes
    for t in range(T):
        yt = eta[t]
        if p:
            yt += np.dot(rep2[-p:], ar)
        if q:
            # left pad shocks with zeros
            yt += np.dot([0] * (q - t) + list(eta[max(0, t - q) : t]), ma)
        rep2.append(yt)
    return np.array(rep2[max(p, q) :])


@pytest.mark.parametrize("ar", arlist)
@pytest.mark.parametrize("ma", malist)
@pytest.mark.parametrize("dist", [np.random.standard_normal])
def test_arma_generate_sample(dist, ar, ma):
    # Test that this generates a true ARMA process
    # (amounts to just a test that scipy.signal.lfilter does what we want)
    T = 100
    np.random.seed(1234)
    eta = dist(T)

    # rep1: from module function
    np.random.seed(1234)
    rep1 = arma_generate_sample(ar, ma, T, distrvs=dist)
    # rep2: "manually" create the ARMA process
    ar_params = -1 * np.array(ar[1:])
    ma_params = np.array(ma[1:])
    rep2 = _manual_arma_generate_sample(ar_params, ma_params, eta)
    assert_array_almost_equal(rep1, rep2, 13)


def test_fi():
    # test identity of ma and ar representation of fi lag polynomial
    n = 100
    mafromar = arma_impulse_response(lpol_fiar(0.4, n=n), [1], n)
    assert_array_almost_equal(mafromar, lpol_fima(0.4, n=n), 13)


def test_arma_impulse_response():
    arrep = arma_impulse_response(armarep.ma, armarep.ar, leads=21)[1:]
    marep = arma_impulse_response(armarep.ar, armarep.ma, leads=21)[1:]
    assert_array_almost_equal(armarep.marep.ravel(), marep, 14)
    # difference in sign convention to matlab for AR term
    assert_array_almost_equal(-armarep.arrep.ravel(), arrep, 14)


@pytest.mark.parametrize("ar", arlist)
@pytest.mark.parametrize("ma", malist)
def test_spectrum(ar, ma):
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)

    arma = ArmaFft(ar, ma, 20)
    spdr, wr = arma.spdroots(w)
    spdp, wp = arma.spdpoly(w, 200)
    spdd, wd = arma.spddirect(nfreq * 2)
    assert_equal(w, wr)
    assert_equal(w, wp)
    assert_almost_equal(w, wd[:nfreq], decimal=14)
    assert_almost_equal(
        spdr,
        spdd[:nfreq],
        decimal=7,
        err_msg="spdr spdd not equal for %s, %s" % (ar, ma),
    )
    assert_almost_equal(
        spdr,
        spdp,
        decimal=7,
        err_msg="spdr spdp not equal for %s, %s" % (ar, ma),
    )


@pytest.mark.parametrize("ar", arlist)
@pytest.mark.parametrize("ma", malist)
def test_armafft(ar, ma):
    # test other methods
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)

    arma = ArmaFft(ar, ma, 20)
    ac1 = arma.invpowerspd(1024)[:10]
    ac2 = arma.acovf(10)[:10]
    assert_allclose(
        ac1, ac2, atol=1e-15, err_msg="acovf not equal for %s, %s" % (ar, ma)
    )


def test_lpol2index_index2lpol():
    process = ArmaProcess([1, 0, 0, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [0.8])
    assert_equal(locs, [2])

    process = ArmaProcess([1, 0.1, 0.1, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [-0.1, -0.1, 0.8])
    assert_equal(locs, [0, 1, 2])
    ar = index2lpol(coefs, locs)
    assert_equal(process.arcoefs, ar)


class TestArmaProcess:
    def test_empty_coeff(self):
        process = ArmaProcess()
        assert_equal(process.arcoefs, np.array([]))
        assert_equal(process.macoefs, np.array([]))

        process = ArmaProcess([1, -0.8])
        assert_equal(process.arcoefs, np.array([0.8]))
        assert_equal(process.macoefs, np.array([]))

        process = ArmaProcess(ma=[1, -0.8])
        assert_equal(process.arcoefs, np.array([]))
        assert_equal(process.macoefs, np.array([-0.8]))

    def test_from_roots(self):
        ar = [1.8, -0.9]
        ma = [0.3]

        ar.insert(0, -1)
        ma.insert(0, 1)
        ar_p = -1 * np.array(ar)
        ma_p = ma
        process_direct = ArmaProcess(ar_p, ma_p)

        process = ArmaProcess.from_roots(np.array(process_direct.maroots), np.array(process_direct.arroots))

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

        process_direct = ArmaProcess(ar=ar_p)
        process = ArmaProcess.from_roots(arroots=np.array(process_direct.arroots))

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

        process_direct = ArmaProcess(ma=ma_p)
        process = ArmaProcess.from_roots(maroots=np.array(process_direct.maroots))

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

        process_direct = ArmaProcess()
        process = ArmaProcess.from_roots()

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

    def test_from_coeff(self):
        ar = [1.8, -0.9]
        ma = [0.3]
        process = ArmaProcess.from_coeffs(np.array(ar), np.array(ma))

        ar.insert(0, -1)
        ma.insert(0, 1)
        ar_p = -1 * np.array(ar)
        ma_p = ma
        process_direct = ArmaProcess(ar_p, ma_p)

        assert_equal(process.arcoefs, process_direct.arcoefs)
        assert_equal(process.macoefs, process_direct.macoefs)
        assert_equal(process.nobs, process_direct.nobs)
        assert_equal(process.maroots, process_direct.maroots)
        assert_equal(process.arroots, process_direct.arroots)
        assert_equal(process.isinvertible, process_direct.isinvertible)
        assert_equal(process.isstationary, process_direct.isstationary)

    def test_process_multiplication(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        process2 = ArmaProcess.from_coeffs([0.7])
        process3 = process1 * process2
        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([]))

        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        process2 = ArmaProcess.from_coeffs([0.7])
        process3 = process1 * process2

        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([0.2]))

        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        process2 = process1 * (np.array([1.0, -0.7]), np.array([1.0]))
        assert_equal(process2.arcoefs, np.array([1.6, -0.7 * 0.9]))

        assert_raises(TypeError, process1.__mul__, [3])

    def test_str_repr(self):
        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        out = process1.__str__()
        print(out)
        assert_(out.find("AR: [1.0, -0.9]") != -1)
        assert_(out.find("MA: [1.0, 0.2]") != -1)

        out = process1.__repr__()
        assert_(out.find("nobs=100") != -1)
        assert_(out.find("at " + str(hex(id(process1)))) != -1)

    def test_acf(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        acf = process1.acf(10)
        expected = np.array(0.9) ** np.arange(10.0)
        assert_array_almost_equal(acf, expected)

        acf = process1.acf()
        assert_(acf.shape[0] == process1.nobs)

    def test_pacf(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        pacf = process1.pacf(10)
        expected = np.array([1, 0.9] + [0] * 8)
        assert_array_almost_equal(pacf, expected)

        pacf = process1.pacf()
        assert_(pacf.shape[0] == process1.nobs)

    def test_isstationary(self):
        process1 = ArmaProcess.from_coeffs([1.1])
        assert_equal(process1.isstationary, False)

        process1 = ArmaProcess.from_coeffs([1.8, -0.9])
        assert_equal(process1.isstationary, True)

        process1 = ArmaProcess.from_coeffs([1.5, -0.5])
        print(np.abs(process1.arroots))
        assert_equal(process1.isstationary, False)

    def test_arma2ar(self):
        process1 = ArmaProcess.from_coeffs([], [0.8])
        vals = process1.arma2ar(100)
        assert_almost_equal(vals, (-0.8) ** np.arange(100.0))

    def test_invertroots(self):
        process1 = ArmaProcess.from_coeffs([], [2.5])
        process2 = process1.invertroots(True)
        assert_almost_equal(process2.ma, np.array([1.0, 0.4]))

        process1 = ArmaProcess.from_coeffs([], [0.4])
        process2 = process1.invertroots(True)
        assert_almost_equal(process2.ma, np.array([1.0, 0.4]))

        process1 = ArmaProcess.from_coeffs([], [2.5])
        roots, invertable = process1.invertroots(False)
        assert_equal(invertable, False)
        assert_almost_equal(roots, np.array([1, 0.4]))

    def test_generate_sample(self):
        process = ArmaProcess.from_coeffs([0.9])
        np.random.seed(12345)
        sample = process.generate_sample()
        np.random.seed(12345)
        expected = np.random.randn(100)
        for i in range(1, 100):
            expected[i] = 0.9 * expected[i - 1] + expected[i]
        assert_almost_equal(sample, expected)

        process = ArmaProcess.from_coeffs([1.6, -0.9])
        np.random.seed(12345)
        sample = process.generate_sample()
        np.random.seed(12345)
        expected = np.random.randn(100)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 100):
            expected[i] = (
                1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
            )
        assert_almost_equal(sample, expected)

        process = ArmaProcess.from_coeffs([1.6, -0.9])
        np.random.seed(12345)
        sample = process.generate_sample(burnin=100)
        np.random.seed(12345)
        expected = np.random.randn(200)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 200):
            expected[i] = (
                1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
            )
        assert_almost_equal(sample, expected[100:])

        np.random.seed(12345)
        sample = process.generate_sample(nsample=(100, 5))
        assert_equal(sample.shape, (100, 5))

    def test_impulse_response(self):
        process = ArmaProcess.from_coeffs([0.9])
        ir = process.impulse_response(10)
        assert_almost_equal(ir, 0.9 ** np.arange(10))

    def test_periodogram(self):
        process = ArmaProcess()
        pg = process.periodogram()
        assert_almost_equal(pg[0], np.linspace(0, np.pi, 100, False))
        assert_almost_equal(pg[1], np.sqrt(2 / np.pi) / 2 * np.ones(100))


@pytest.mark.parametrize("d", [0, 1])
@pytest.mark.parametrize("seasonal", [True])
def test_from_estimation(d, seasonal):
    ar = [0.8] if not seasonal else [0.8, 0, 0, 0.2, -0.16]
    ma = [0.4] if not seasonal else [0.4, 0, 0, 0.2, -0.08]
    ap = ArmaProcess.from_coeffs(ar, ma, 500)
    idx = pd.date_range(dt.datetime(1900, 1, 1), periods=500, freq="Q")
    data = ap.generate_sample(500)
    if d == 1:
        data = np.cumsum(data)
    data = pd.Series(data, index=idx)
    seasonal_order = (1, 0, 1, 4) if seasonal else None
    mod = ARIMA(data, order=(1, d, 1), seasonal_order=seasonal_order)
    res = mod.fit()
    ap_from = ArmaProcess.from_estimation(res)
    shape = (5,) if seasonal else (1,)
    assert ap_from.arcoefs.shape == shape
    assert ap_from.macoefs.shape == shape
