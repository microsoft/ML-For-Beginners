import itertools
import os

import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_,
                           assert_almost_equal, assert_array_almost_equal)
from pytest import raises as assert_raises
import pytest
from scipy._lib._testutils import check_free_memory

from scipy.interpolate import RectBivariateSpline

from scipy.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
     sproot, splprep, splint, spalde, splder, splantider, insert, dblint)
from scipy.interpolate.dfitpack import regrid_smth
from scipy.interpolate._fitpack2 import dfitpack_int


def data_file(basename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', basename)


def norm2(x):
    return np.sqrt(np.dot(x.T, x))


def f1(x, d=0):
    """Derivatives of sin->cos->-sin->-cos."""
    if d % 4 == 0:
        return np.sin(x)
    if d % 4 == 1:
        return np.cos(x)
    if d % 4 == 2:
        return -np.sin(x)
    if d % 4 == 3:
        return -np.cos(x)


def makepairs(x, y):
    """Helper function to create an array of pairs of x and y."""
    xy = np.array(list(itertools.product(np.asarray(x), np.asarray(y))))
    return xy.T


class TestSmokeTests:
    """
    Smoke tests (with a few asserts) for fitpack routines -- mostly
    check that they are runnable
    """
    def check_1(self, per=0, s=0, a=0, b=2*np.pi, at_nodes=False,
                xb=None, xe=None):
        if xb is None:
            xb = a
        if xe is None:
            xe = b

        N = 20
        # nodes and middle points of the nodes
        x = np.linspace(a, b, N + 1)
        x1 = a + (b - a) * np.arange(1, N, dtype=float) / float(N - 1)
        v = f1(x)

        def err_est(k, d):
            # Assume f has all derivatives < 1
            h = 1.0 / N
            tol = 5 * h**(.75*(k-d))
            if s > 0:
                tol += 1e5*s
            return tol

        for k in range(1, 6):
            tck = splrep(x, v, s=s, per=per, k=k, xe=xe)
            tt = tck[0][k:-k] if at_nodes else x1

            for d in range(k+1):
                tol = err_est(k, d)
                err = norm2(f1(tt, d) - splev(tt, tck, d)) / norm2(f1(tt, d))
                assert err < tol

    def check_2(self, per=0, N=20, ia=0, ib=2*np.pi):
        a, b, dx = 0, 2*np.pi, 0.2*np.pi
        x = np.linspace(a, b, N+1)    # nodes
        v = np.sin(x)

        def err_est(k, d):
            # Assume f has all derivatives < 1
            h = 1.0 / N
            tol = 5 * h**(.75*(k-d))
            return tol

        nk = []
        for k in range(1, 6):
            tck = splrep(x, v, s=0, per=per, k=k, xe=b)
            nk.append([splint(ia, ib, tck), spalde(dx, tck)])

        k = 1
        for r in nk:
            d = 0
            for dr in r[1]:
                tol = err_est(k, d)
                assert_allclose(dr, f1(dx, d), atol=0, rtol=tol)
                d = d+1
            k = k+1

    def test_smoke_splrep_splev(self):
        self.check_1(s=1e-6)
        self.check_1(b=1.5*np.pi)
        self.check_1(b=1.5*np.pi, xe=2*np.pi, per=1, s=1e-1)

    @pytest.mark.parametrize('per', [0, 1])
    @pytest.mark.parametrize('at_nodes', [True, False])
    def test_smoke_splrep_splev_2(self, per, at_nodes):
        self.check_1(per=per, at_nodes=at_nodes)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('per', [0, 1])
    def test_smoke_splint_spalde(self, N, per):
        self.check_2(per=per, N=N)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('per', [0, 1])
    def test_smoke_splint_spalde_iaib(self, N, per):
        self.check_2(ia=0.2*np.pi, ib=np.pi, N=N, per=per)

    def test_smoke_sproot(self):
        # sproot is only implemented for k=3
        a, b = 0.1, 15
        x = np.linspace(a, b, 20)
        v = np.sin(x)

        for k in [1, 2, 4, 5]:
            tck = splrep(x, v, s=0, per=0, k=k, xe=b)
            with assert_raises(ValueError):
                sproot(tck)

        k = 3
        tck = splrep(x, v, s=0, k=3)
        roots = sproot(tck)
        assert_allclose(splev(roots, tck), 0, atol=1e-10, rtol=1e-10)
        assert_allclose(roots, np.pi * np.array([1, 2, 3, 4]), rtol=1e-3)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
    def test_smoke_splprep_splrep_splev(self, N, k):
        a, b, dx = 0, 2.*np.pi, 0.2*np.pi
        x = np.linspace(a, b, N+1)    # nodes
        v = np.sin(x)

        tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
        uv = splev(dx, tckp)
        err1 = abs(uv[1] - np.sin(uv[0]))
        assert err1 < 1e-2

        tck = splrep(x, v, s=0, per=0, k=k)
        err2 = abs(splev(uv[0], tck) - np.sin(uv[0]))
        assert err2 < 1e-2

        # Derivatives of parametric cubic spline at u (first function)
        if k == 3:
            tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
            for d in range(1, k+1):
                uv = splev(dx, tckp, d)

    def test_smoke_bisplrep_bisplev(self):
        xb, xe = 0, 2.*np.pi
        yb, ye = 0, 2.*np.pi
        kx, ky = 3, 3
        Nx, Ny = 20, 20

        def f2(x, y):
            return np.sin(x+y)

        x = np.linspace(xb, xe, Nx + 1)
        y = np.linspace(yb, ye, Ny + 1)
        xy = makepairs(x, y)
        tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)

        tt = [tck[0][kx:-kx], tck[1][ky:-ky]]
        t2 = makepairs(tt[0], tt[1])
        v1 = bisplev(tt[0], tt[1], tck)
        v2 = f2(t2[0], t2[1])
        v2.shape = len(tt[0]), len(tt[1])

        assert norm2(np.ravel(v1 - v2)) < 1e-2


class TestSplev:
    def test_1d_shape(self):
        x = [1,2,3,4,5]
        y = [4,5,6,7,8]
        tck = splrep(x, y)
        z = splev([1], tck)
        assert_equal(z.shape, (1,))
        z = splev(1, tck)
        assert_equal(z.shape, ())

    def test_2d_shape(self):
        x = [1, 2, 3, 4, 5]
        y = [4, 5, 6, 7, 8]
        tck = splrep(x, y)
        t = np.array([[1.0, 1.5, 2.0, 2.5],
                      [3.0, 3.5, 4.0, 4.5]])
        z = splev(t, tck)
        z0 = splev(t[0], tck)
        z1 = splev(t[1], tck)
        assert_equal(z, np.row_stack((z0, z1)))

    def test_extrapolation_modes(self):
        # test extrapolation modes
        #    * if ext=0, return the extrapolated value.
        #    * if ext=1, return 0
        #    * if ext=2, raise a ValueError
        #    * if ext=3, return the boundary value.
        x = [1,2,3]
        y = [0,2,4]
        tck = splrep(x, y, k=1)

        rstl = [[-2, 6], [0, 0], None, [0, 4]]
        for ext in (0, 1, 3):
            assert_array_almost_equal(splev([0, 4], tck, ext=ext), rstl[ext])

        assert_raises(ValueError, splev, [0, 4], tck, ext=2)


class TestSplder:
    def setup_method(self):
        # non-uniform grid, just to make it sure
        x = np.linspace(0, 1, 100)**3
        y = np.sin(20 * x)
        self.spl = splrep(x, y)

        # double check that knots are non-uniform
        assert_(np.diff(self.spl[0]).ptp() > 0)

    def test_inverse(self):
        # Check that antiderivative + derivative is identity.
        for n in range(5):
            spl2 = splantider(self.spl, n)
            spl3 = splder(spl2, n)
            assert_allclose(self.spl[0], spl3[0])
            assert_allclose(self.spl[1], spl3[1])
            assert_equal(self.spl[2], spl3[2])

    def test_splder_vs_splev(self):
        # Check derivative vs. FITPACK

        for n in range(3+1):
            # Also extrapolation!
            xx = np.linspace(-1, 2, 2000)
            if n == 3:
                # ... except that FITPACK extrapolates strangely for
                # order 0, so let's not check that.
                xx = xx[(xx >= 0) & (xx <= 1)]

            dy = splev(xx, self.spl, n)
            spl2 = splder(self.spl, n)
            dy2 = splev(xx, spl2)
            if n == 1:
                assert_allclose(dy, dy2, rtol=2e-6)
            else:
                assert_allclose(dy, dy2)

    def test_splantider_vs_splint(self):
        # Check antiderivative vs. FITPACK
        spl2 = splantider(self.spl)

        # no extrapolation, splint assumes function is zero outside
        # range
        xx = np.linspace(0, 1, 20)

        for x1 in xx:
            for x2 in xx:
                y1 = splint(x1, x2, self.spl)
                y2 = splev(x2, spl2) - splev(x1, spl2)
                assert_allclose(y1, y2)

    def test_order0_diff(self):
        assert_raises(ValueError, splder, self.spl, 4)

    def test_kink(self):
        # Should refuse to differentiate splines with kinks

        spl2 = insert(0.5, self.spl, m=2)
        splder(spl2, 2)  # Should work
        assert_raises(ValueError, splder, spl2, 3)

        spl2 = insert(0.5, self.spl, m=3)
        splder(spl2, 1)  # Should work
        assert_raises(ValueError, splder, spl2, 2)

        spl2 = insert(0.5, self.spl, m=4)
        assert_raises(ValueError, splder, spl2, 1)

    def test_multidim(self):
        # c can have trailing dims
        for n in range(3):
            t, c, k = self.spl
            c2 = np.c_[c, c, c]
            c2 = np.dstack((c2, c2))

            spl2 = splantider((t, c2, k), n)
            spl3 = splder(spl2, n)

            assert_allclose(t, spl3[0])
            assert_allclose(c2, spl3[1])
            assert_equal(k, spl3[2])


class TestSplint:
    def test_len_c(self):
        n, k = 7, 3
        x = np.arange(n)
        y = x**3
        t, c, k = splrep(x, y, s=0)

        # note that len(c) == len(t) == 11 (== len(x) + 2*(k-1))
        assert len(t) == len(c) == n + 2*(k-1)

        # integrate directly: $\int_0^6 x^3 dx = 6^4 / 4$
        res = splint(0, 6, (t, c, k))
        assert_allclose(res, 6**4 / 4, atol=1e-15)

        # check that the coefficients past len(t) - k - 1 are ignored
        c0 = c.copy()
        c0[len(t)-k-1:] = np.nan
        res0 = splint(0, 6, (t, c0, k))
        assert_allclose(res0, 6**4 / 4, atol=1e-15)

        # however, all other coefficients *are* used
        c0[6] = np.nan
        assert np.isnan(splint(0, 6, (t, c0, k)))

        # check that the coefficient array can have length `len(t) - k - 1`
        c1 = c[:len(t) - k - 1]
        res1 = splint(0, 6, (t, c1, k))
        assert_allclose(res1, 6**4 / 4, atol=1e-15)

        # however shorter c arrays raise. The error from f2py is a
        # `dftipack.error`, which is an Exception but not ValueError etc.
        with assert_raises(Exception, match=r">=n-k-1"):
            splint(0, 1, (np.ones(10), np.ones(5), 3))


class TestBisplrep:
    def test_overflow(self):
        from numpy.lib.stride_tricks import as_strided
        if dfitpack_int.itemsize == 8:
            size = 1500000**2
        else:
            size = 400**2
        # Don't allocate a real array, as it's very big, but rely
        # on that it's not referenced
        x = as_strided(np.zeros(()), shape=(size,))
        assert_raises(OverflowError, bisplrep, x, x, x, w=x,
                      xb=0, xe=1, yb=0, ye=1, s=0)

    def test_regression_1310(self):
        # Regression test for gh-1310
        data = np.load(data_file('bug-1310.npz'))['data']

        # Shouldn't crash -- the input data triggers work array sizes
        # that caused previously some data to not be aligned on
        # sizeof(double) boundaries in memory, which made the Fortran
        # code to crash when compiled with -O3
        bisplrep(data[:,0], data[:,1], data[:,2], kx=3, ky=3, s=0,
                 full_output=True)

    @pytest.mark.skipif(dfitpack_int != np.int64, reason="needs ilp64 fitpack")
    def test_ilp64_bisplrep(self):
        check_free_memory(28000)  # VM size, doesn't actually use the pages
        x = np.linspace(0, 1, 400)
        y = np.linspace(0, 1, 400)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        tck = bisplrep(x, y, z, kx=3, ky=3, s=0)
        assert_allclose(bisplev(0.5, 0.5, tck), 0.0)


def test_dblint():
    # Basic test to see it runs and gives the correct result on a trivial
    # problem. Note that `dblint` is not exposed in the interpolate namespace.
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    xx, yy = np.meshgrid(x, y)
    rect = RectBivariateSpline(x, y, 4 * xx * yy)
    tck = list(rect.tck)
    tck.extend(rect.degrees)

    assert_almost_equal(dblint(0, 1, 0, 1, tck), 1)
    assert_almost_equal(dblint(0, 0.5, 0, 1, tck), 0.25)
    assert_almost_equal(dblint(0.5, 1, 0, 1, tck), 0.75)
    assert_almost_equal(dblint(-100, 100, -100, 100, tck), 1)


def test_splev_der_k():
    # regression test for gh-2188: splev(x, tck, der=k) gives garbage or crashes
    # for x outside of knot range

    # test case from gh-2188
    tck = (np.array([0., 0., 2.5, 2.5]),
           np.array([-1.56679978, 2.43995873, 0., 0.]),
           1)
    t, c, k = tck
    x = np.array([-3, 0, 2.5, 3])

    # an explicit form of the linear spline
    assert_allclose(splev(x, tck), c[0] + (c[1] - c[0]) * x/t[2])
    assert_allclose(splev(x, tck, 1), (c[1]-c[0]) / t[2])

    # now check a random spline vs splder
    np.random.seed(1234)
    x = np.sort(np.random.random(30))
    y = np.random.random(30)
    t, c, k = splrep(x, y)

    x = [t[0] - 1., t[-1] + 1.]
    tck2 = splder((t, c, k), k)
    assert_allclose(splev(x, (t, c, k), k), splev(x, tck2))


def test_splprep_segfault():
    # regression test for gh-3847: splprep segfaults if knots are specified
    # for task=-1
    t = np.arange(0, 1.1, 0.1)
    x = np.sin(2*np.pi*t)
    y = np.cos(2*np.pi*t)
    tck, u = splprep([x, y], s=0)
    np.arange(0, 1.01, 0.01)

    uknots = tck[0]  # using the knots from the previous fitting
    tck, u = splprep([x, y], task=-1, t=uknots)  # here is the crash


def test_bisplev_integer_overflow():
    np.random.seed(1)

    x = np.linspace(0, 1, 11)
    y = x
    z = np.random.randn(11, 11).ravel()
    kx = 1
    ky = 1

    nx, tx, ny, ty, c, fp, ier = regrid_smth(
        x, y, z, None, None, None, None, kx=kx, ky=ky, s=0.0)
    tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)], kx, ky)

    xp = np.zeros([2621440])
    yp = np.zeros([2621440])

    assert_raises((RuntimeError, MemoryError), bisplev, xp, yp, tck)


@pytest.mark.xslow
def test_gh_1766():
    # this should fail gracefully instead of segfaulting (int overflow)
    size = 22
    kx, ky = 3, 3
    def f2(x, y):
        return np.sin(x+y)

    x = np.linspace(0, 10, size)
    y = np.linspace(50, 700, size)
    xy = makepairs(x, y)
    tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)
    # the size value here can either segfault
    # or produce a MemoryError on main
    tx_ty_size = 500000
    tck[0] = np.arange(tx_ty_size)
    tck[1] = np.arange(tx_ty_size) * 4
    tt_0 = np.arange(50)
    tt_1 = np.arange(50) * 3
    with pytest.raises(MemoryError):
        bisplev(tt_0, tt_1, tck, 1, 1)


def test_spalde_scalar_input():
    # Ticket #629
    x = np.linspace(0, 10)
    y = x**3
    tck = splrep(x, y, k=3, t=[5])
    res = spalde(np.float64(1), tck)
    des = np.array([1., 3., 6., 6.])
    assert_almost_equal(res, des)


def test_spalde_nc():
    # regression test for https://github.com/scipy/scipy/issues/19002
    # here len(t) = 29 and len(c) = 25 (== len(t) - k - 1) 
    x = np.asarray([-10., -9., -8., -7., -6., -5., -4., -3., -2.5, -2., -1.5,
                    -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6.],
                    dtype="float")
    t = [-10.0, -10.0, -10.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
         -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
         5.0, 6.0, 6.0, 6.0, 6.0]
    c = np.asarray([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    k = 3

    res = spalde(x, (t, c, k))
    res_splev = np.asarray([splev(x, (t, c, k), nu) for nu in range(4)])
    assert_allclose(res, res_splev.T, atol=1e-15)
