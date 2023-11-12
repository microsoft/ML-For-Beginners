# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:21:26 2018

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess


class CheckPenalty:

    def test_symmetry(self):
        pen = self.pen
        x = self.params

        p = np.array([pen.func(np.atleast_1d(xi)) for xi in x])
        assert_allclose(p, p[::-1], rtol=1e-10)
        # func(0) should be 0
        assert_allclose(pen.func(0 * np.atleast_1d(x[0])), 0, rtol=1e-10)

    def test_derivatives(self):
        pen = self.pen
        x = self.params

        ps = np.array([pen.deriv(np.atleast_1d(xi)) for xi in x])
        psn = np.array([approx_fprime(np.atleast_1d(xi), pen.func) for xi in x])
        assert_allclose(ps, psn, rtol=1e-7, atol=1e-8)

        ph = np.array([pen.deriv2(np.atleast_1d(xi)) for xi in x])
        phn = np.array([approx_hess(np.atleast_1d(xi), pen.func) for xi in x])
        if ph.ndim == 2:
            # SmoothedSCAD returns only diagonal if hessian if independent
            # TODO should ww allow this also in L@?
            ph = np.array([np.diag(phi) for phi in ph])
        assert_allclose(ph, phn, rtol=1e-7, atol=1e-8)


class TestL2Constraints0(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.L2ConstraintsPenalty()

    def test_equivalence(self):
        # compare plain penalty with included weights or restriction
        pen = self.pen
        x = self.params
        k = x.shape[1]

        pen2 = smpen.L2ConstraintsPenalty(weights=np.ones(k))
        pen3 = smpen.L2ConstraintsPenalty(restriction=np.eye(k))
        f = pen.func(x.T)
        d = pen.deriv(x.T)
        d2 =  np.array([pen.deriv2(np.atleast_1d(xi)) for xi in x])
        for pen_ in [pen2, pen3]:
            assert_allclose(pen_.func(x.T), f, rtol=1e-7, atol=1e-8)
            assert_allclose(pen_.deriv(x.T), d, rtol=1e-7, atol=1e-8)
            d2_ =  np.array([pen.deriv2(np.atleast_1d(xi)) for xi in x])
            assert_allclose(d2_, d2, rtol=1e-10, atol=1e-8)


class TestL2Constraints1(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.L2ConstraintsPenalty(restriction=[[1,0], [1, 1]])

    def test_values(self):
        pen = self.pen
        x = self.params
        r = pen.restriction
        f = (r.dot(x.T)**2).sum(0)
        assert_allclose(pen.func(x.T), f, rtol=1e-7, atol=1e-8)


class TestSmoothedSCAD(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.SCADSmoothed(tau=0.05, c0=0.05)


class TestPseudoHuber(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.PseudoHuber(0.1)

    def test_backward_compatibility(self):
        wts = [0.5]
        pen = smpen.PseudoHuber(0.1, weights=wts)
        assert_equal(pen.weights, wts)

    def test_deprecated_priority(self):
        weights = [1.0]
        pen = smpen.PseudoHuber(0.1, weights=weights)

        assert_equal(pen.weights, weights)

    def test_weights_assignment(self):
        weights = [1.0, 2.0]
        pen = smpen.PseudoHuber(0.1, weights=weights)
        assert_equal(pen.weights, weights)


class TestL2(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.L2()

    def test_backward_compatibility(self):
        wts = [0.5]
        pen = smpen.L2(weights=wts)
        assert_equal(pen.weights, wts)

    def test_deprecated_priority(self):
        weights = [1.0]
        pen = smpen.L2(weights=weights)
        assert_equal(pen.weights, weights)

    def test_weights_assignment(self):
        weights = [1.0, 2.0]
        pen = smpen.L2(weights=weights)
        assert_equal(pen.weights, weights)


class TestNonePenalty(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.NonePenalty()
