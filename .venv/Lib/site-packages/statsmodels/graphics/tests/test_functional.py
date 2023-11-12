
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest

from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
    banddepth,
    fboxplot,
    hdrboxplot,
    rainbowplot,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


data = elnino.load()
data.raw_data = np.asarray(data.raw_data)
labels = data.raw_data[:, 0].astype(int)
data = data.raw_data[:, 1:]


@pytest.mark.matplotlib
def test_hdr_basic(close_figures):
    try:
        _, hdr = hdrboxplot(data, labels=labels, seed=12345)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    assert len(hdr.extra_quantiles) == 0

    median_t = [24.247, 25.625, 25.964, 24.999, 23.648, 22.302,
                21.231, 20.366, 20.168, 20.434, 21.111, 22.299]

    assert_almost_equal(hdr.median, median_t, decimal=2)

    quant = np.vstack([hdr.outliers, hdr.hdr_90, hdr.hdr_50])
    quant_t = np.vstack([[24.36, 25.42, 25.40, 24.96, 24.21, 23.35,
                          22.50, 21.89, 22.04, 22.88, 24.57, 25.89],
                         [27.25, 28.23, 28.85, 28.82, 28.37, 27.43,
                          25.73, 23.88, 22.26, 22.22, 22.21, 23.19],
                         [23.70, 26.08, 27.17, 26.74, 26.77, 26.15,
                          25.59, 24.95, 24.69, 24.64, 25.85, 27.08],
                         [28.12, 28.82, 29.24, 28.45, 27.36, 25.19,
                          23.61, 22.27, 21.31, 21.37, 21.60, 22.81],
                         [25.48, 26.99, 27.51, 27.04, 26.23, 24.94,
                          23.69, 22.72, 22.26, 22.64, 23.33, 24.44],
                         [23.11, 24.50, 24.66, 23.44, 21.74, 20.58,
                          19.68, 18.84, 18.76, 18.99, 19.66, 20.86],
                         [24.84, 26.23, 26.67, 25.93, 24.87, 23.57,
                          22.46, 21.45, 21.26, 21.57, 22.14, 23.41],
                         [23.62, 25.10, 25.34, 24.22, 22.74, 21.52,
                          20.40, 19.56, 19.63, 19.67, 20.37, 21.76]])

    assert_almost_equal(quant, quant_t, decimal=0)

    labels_pos = np.all(np.in1d(data, hdr.outliers).reshape(data.shape),
                        axis=1)
    outliers = labels[labels_pos]
    assert_equal([1982, 1983, 1997, 1998], outliers)
    assert_equal(labels[hdr.outliers_idx], outliers)


@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_basic_brute(close_figures, reset_randomstate):
    try:
        _, hdr = hdrboxplot(data, ncomp=2, labels=labels, use_brute=True)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    assert len(hdr.extra_quantiles) == 0

    median_t = [24.247, 25.625, 25.964, 24.999, 23.648, 22.302,
                21.231, 20.366, 20.168, 20.434, 21.111, 22.299]

    assert_almost_equal(hdr.median, median_t, decimal=2)


@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_plot(close_figures):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        hdrboxplot(data, labels=labels.tolist(), ax=ax, threshold=1,
                   seed=12345)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    ax.set_xlabel("Month of the year")
    ax.set_ylabel("Sea surface temperature (C)")
    ax.set_xticks(np.arange(13, step=3) - 1)
    ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    ax.set_xlim([-0.2, 11.2])


@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_alpha(close_figures):
    try:
        _, hdr = hdrboxplot(data, alpha=[0.7], seed=12345)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    extra_quant_t = np.vstack([[25.1, 26.5, 27.0, 26.4, 25.4, 24.1,
                                23.0, 22.0, 21.7, 22.1, 22.7, 23.8],
                               [23.4, 24.8, 25.0, 23.9, 22.4, 21.1,
                                20.0, 19.3, 19.2, 19.4, 20.1, 21.3]])
    assert_almost_equal(hdr.extra_quantiles, extra_quant_t, decimal=0)


@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_multiple_alpha(close_figures):
    try:
        _, hdr = hdrboxplot(data, alpha=[0.4, 0.92], seed=12345)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    extra_quant_t = [[25.712, 27.052, 27.711, 27.200,
                      26.162, 24.833, 23.639, 22.378,
                      22.250, 22.640, 23.472, 24.649],
                     [22.973, 24.526, 24.608, 23.343,
                      21.908, 20.655, 19.750, 19.046,
                      18.812, 18.989, 19.520, 20.685],
                     [24.667, 26.033, 26.416, 25.584,
                      24.308, 22.849, 21.684, 20.948,
                      20.483, 21.019, 21.751, 22.890],
                     [23.873, 25.371, 25.667, 24.644,
                      23.177, 21.923, 20.791, 20.015,
                      19.697, 19.951, 20.622, 21.858]]
    assert_almost_equal(hdr.extra_quantiles, np.vstack(extra_quant_t),
                        decimal=0)


@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_threshold(close_figures):
    try:
        _, hdr = hdrboxplot(data, alpha=[0.8], threshold=0.93,
                            seed=12345)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    labels_pos = np.all(np.in1d(data, hdr.outliers).reshape(data.shape),
                        axis=1)
    outliers = labels[labels_pos]
    assert_equal([1968, 1982, 1983, 1997, 1998], outliers)


@pytest.mark.matplotlib
def test_hdr_bw(close_figures):
    try:
        _, hdr = hdrboxplot(data, bw='cv_ml', seed=12345)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    median_t = [24.25, 25.64, 25.99, 25.04, 23.71, 22.38,
                21.31, 20.44, 20.24, 20.51, 21.19, 22.38]
    assert_almost_equal(hdr.median, median_t, decimal=2)


@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_ncomp(close_figures):
    try:
        _, hdr = hdrboxplot(data, ncomp=3, seed=12345)
    except WindowsError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')

    median_t = [24.33, 25.71, 26.04, 25.08, 23.74, 22.40,
                21.32, 20.45, 20.25, 20.53, 21.20, 22.39]
    assert_almost_equal(hdr.median, median_t, decimal=2)


def test_banddepth_BD2():
    xx = np.arange(500) / 150.
    y1 = 1 + 0.5 * np.sin(xx)
    y2 = 0.3 + np.sin(xx + np.pi/6)
    y3 = -0.5 + np.sin(xx + np.pi/6)
    y4 = -1 + 0.3 * np.cos(xx + np.pi/6)

    data = np.asarray([y1, y2, y3, y4])
    depth = banddepth(data, method='BD2')
    expected_depth = [0.5, 5./6, 5./6, 0.5]
    assert_almost_equal(depth, expected_depth)

    # Plot to visualize why we expect this output
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for ii, yy in enumerate([y1, y2, y3, y4]):
    #    ax.plot(xx, yy, label="y%s" % ii)

    # ax.legend()
    # plt.close(fig)


def test_banddepth_MBD():
    xx = np.arange(5001) / 5000.
    y1 = np.zeros(xx.shape)
    y2 = 2 * xx - 1
    y3 = np.ones(xx.shape) * 0.5
    y4 = np.ones(xx.shape) * -0.25

    data = np.asarray([y1, y2, y3, y4])
    depth = banddepth(data, method='MBD')
    expected_depth = [5./6, (2*(0.75-3./8)+3)/6, 3.5/6, (2*3./8+3)/6]
    assert_almost_equal(depth, expected_depth, decimal=4)


@pytest.mark.matplotlib
def test_fboxplot_rainbowplot(close_figures):
    # Test fboxplot and rainbowplot together, is much faster.
    def harmfunc(t):
        """Test function, combination of a few harmonic terms."""
        # Constant, 0 with p=0.9, 1 with p=1 - for creating outliers
        ci = int(np.random.random() > 0.9)
        a1i = np.random.random() * 0.05
        a2i = np.random.random() * 0.05
        b1i = (0.15 - 0.1) * np.random.random() + 0.1
        b2i = (0.15 - 0.1) * np.random.random() + 0.1

        func = (1 - ci) * (a1i * np.sin(t) + a2i * np.cos(t)) + \
            ci * (b1i * np.sin(t) + b2i * np.cos(t))

        return func

    np.random.seed(1234567)
    # Some basic test data, Model 6 from Sun and Genton.
    t = np.linspace(0, 2 * np.pi, 250)
    data = [harmfunc(t) for _ in range(20)]

    # fboxplot test
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, depth, ix_depth, ix_outliers = fboxplot(data, wfactor=2, ax=ax)

    ix_expected = np.array([13, 4, 15, 19, 8, 6, 3, 16, 9, 7, 1, 5, 2,
                            12, 17, 11, 14, 10, 0, 18])
    assert_equal(ix_depth, ix_expected)
    ix_expected2 = np.array([2, 11, 17, 18])
    assert_equal(ix_outliers, ix_expected2)

    # rainbowplot test (re-uses depth variable)
    xdata = np.arange(data[0].size)
    fig = rainbowplot(data, xdata=xdata, depth=depth, cmap=plt.cm.rainbow)
