
import pytest
import numpy as np
from numpy.testing import assert_allclose

from statsmodels.robust import norms
from statsmodels.tools.numdiff import (
    _approx_fprime_scalar,
    # _approx_fprime_cs_scalar,  # not yet
    )
from .results import results_norms as res_r

cases = [
    (norms.Hampel, (1.5, 3.5, 8.), res_r.res_hampel),
    (norms.TukeyBiweight, (4,), res_r.res_biweight),
    (norms.HuberT, (1.345,), res_r.res_huber),
    ]

norms_other = [
    (norms.LeastSquares, ()),
    (norms.TrimmedMean, (1.9,)),  # avoid arg at integer used in example
    (norms.AndrewWave, ()),
    (norms.RamsayE, ()),
    # norms.MQuantileNorm,  # requires keywords in init
    ]

dtypes = ["int", np.float64, np.complex128]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("case", cases)
def test_norm(case, dtype):
    ncls, args, res = case
    if ncls in [norms.HuberT] and dtype == np.complex128:
        # skip for now
        return

    norm = ncls(*args)
    x = np.array([-9, -6, -2, -1, 0, 1, 2, 6, 9], dtype=dtype)

    weights = norm.weights(x)
    rho = norm.rho(x)
    psi = norm.psi(x)
    psi_deriv = norm.psi_deriv(x)
    assert_allclose(weights, res.weights, rtol=1e-12, atol=1e-20)
    assert_allclose(rho, res.rho, rtol=1e-12, atol=1e-20)
    assert_allclose(psi, res.psi, rtol=1e-12, atol=1e-20)
    assert_allclose(psi_deriv, res.psi_deriv, rtol=1e-12, atol=1e-20)

    dtype2 = np.promote_types(dtype, "float")
    assert weights.dtype == dtype2
    assert rho.dtype == dtype2
    assert psi.dtype == dtype2
    assert psi_deriv.dtype == dtype2

    psid = _approx_fprime_scalar(x, norm.rho)
    assert_allclose(psid, res.psi, rtol=1e-6, atol=1e-8)
    psidd = _approx_fprime_scalar(x, norm.psi)
    assert_allclose(psidd, res.psi_deriv, rtol=1e-6, atol=1e-8)

    # complex step derivatives are not yet supported if method uses np.abs
    # psid = _approx_fprime_cs_scalar(x, norm.rho)
    # assert_allclose(psid, res.psi, rtol=1e-12, atol=1e-20)
    # psidd = _approx_fprime_cs_scalar(x, norm.psi)
    # assert_allclose(psidd, res.psi_deriv, rtol=1e-12, atol=1e-20)

    # check scalar value
    methods = ["weights", "rho", "psi", "psi_deriv"]
    for meth in methods:
        resm = [getattr(norm, meth)(xi) for xi in x]
        assert_allclose(resm, getattr(res, meth))


@pytest.mark.parametrize("case", norms_other)
def test_norms_consistent(case):
    # test that norm methods are consistent with each other
    ncls, args = case
    norm = ncls(*args)
    x = np.array([-9, -6, -2, -1, 0, 1, 2, 6, 9], dtype=float)

    weights = norm.weights(x)
    rho = norm.rho(x)  # not used
    psi = norm.psi(x)
    psi_deriv = norm.psi_deriv(x)

    # check location and u-shape of rho
    assert_allclose(rho[4], 0, atol=1e-12)
    assert np.all(np.diff(rho[4:]) >= 0)
    assert np.all(np.diff(rho[:4]) <= 0)

    # avoid zero division nan:
    assert_allclose(weights, (psi + 1e-50) / (x + 1e-50), rtol=1e-6, atol=1e-8)
    psid = _approx_fprime_scalar(x, norm.rho)
    assert_allclose(psi, psid, rtol=1e-6, atol=1e-6)
    psidd = _approx_fprime_scalar(x, norm.psi)
    assert_allclose(psi_deriv, psidd, rtol=1e-6, atol=1e-8)
