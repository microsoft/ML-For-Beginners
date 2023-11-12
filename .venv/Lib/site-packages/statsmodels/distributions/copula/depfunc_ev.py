# -*- coding: utf-8 -*-
""" Pickand's dependence functions as generators for EV-copulas


Created on Wed Jan 27 14:33:40 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess


class PickandDependence:

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, t, *args):
        raise NotImplementedError

    def deriv(self, t, *args):
        """First derivative of the dependence function

        implemented through numerical differentiation
        """
        t = np.atleast_1d(t)
        return _approx_fprime_cs_scalar(t, self.evaluate)

    def deriv2(self, t, *args):
        """Second derivative of the dependence function

        implemented through numerical differentiation
        """
        if np.size(t) == 1:
            d2 = approx_hess([t], self.evaluate, args=args)[0]
        else:
            d2 = np.array([approx_hess([ti], self.evaluate, args=args)[0, 0]
                           for ti in t])
        return d2


class AsymLogistic(PickandDependence):
    '''asymmetric logistic model of Tawn 1988

    special case: a1=a2=1 : Gumbel

    restrictions:
     - theta in (0,1]
     - a1, a2 in [0,1]
    '''
    k_args = 3

    def _check_args(self, a1, a2, theta):
        condth = (theta > 0) and (theta <= 1)
        conda1 = (a1 >= 0) and (a1 <= 1)
        conda2 = (a2 >= 0) and (a2 <= 1)
        return condth and conda1 and conda2

    def evaluate(self, t, a1, a2, theta):

        # if not np.all(_check_args(a1, a2, theta)):
        #    raise ValueError('invalid args')

        transf = (1 - a2) * (1-t)
        transf += (1 - a1) * t
        transf += ((a1 * t)**(1./theta) + (a2 * (1-t))**(1./theta))**theta

        return transf

    def deriv(self, t, a1, a2, theta):
        b = theta

        d1 = ((a1 * (a1 * t)**(1/b - 1) - a2 * (a2 * (1 - t))**(1/b - 1)) *
              ((a1 * t)**(1/b) + (a2 * (1 - t))**(1/b))**(b - 1) - a1 + a2)
        return d1

    def deriv2(self, t, a1, a2, theta):
        b = theta
        d2 = ((1 - b) * (a1 * t)**(1/b) * (a2 * (1 - t))**(1/b) *
              ((a1 * t)**(1/b) + (a2 * (1 - t))**(1/b))**(b - 2)
              )/(b * (1 - t)**2 * t**2)
        return d2


transform_tawn = AsymLogistic()


class AsymNegLogistic(PickandDependence):
    '''asymmetric negative logistic model of Joe 1990

    special case:  a1=a2=1 : symmetric negative logistic of Galambos 1978

    restrictions:
     - theta in (0,inf)
     - a1, a2 in (0,1]
    '''
    k_args = 3

    def _check_args(self, a1, a2, theta):
        condth = (theta > 0)
        conda1 = (a1 > 0) and (a1 <= 1)
        conda2 = (a2 > 0) and (a2 <= 1)
        return condth and conda1 and conda2

    def evaluate(self, t, a1, a2, theta):
        # if not np.all(self._check_args(a1, a2, theta)):
        #     raise ValueError('invalid args')

        a1, a2 = a2, a1
        transf = 1 - ((a1 * (1-t))**(-1./theta) +
                      (a2 * t)**(-1./theta))**(-theta)
        return transf

    def deriv(self, t, a1, a2, theta):
        a1, a2 = a2, a1
        m1 = -1 / theta
        m2 = m1 - 1

        # (a1^(-1/θ) (1 - t)^(-1/θ - 1) - a2^(-1/θ) t^(-1/θ - 1))*
        # (a1^(-1/θ) (1 - t)^(-1/θ) + (a2 t)^(-1/θ))^(-θ - 1)

        d1 = (a1**m1 * (1 - t)**m2 - a2**m1 * t**m2) * (
                (a1 * (1 - t))**m1 + (a2 * t)**m1)**(-theta - 1)
        return d1

    def deriv2(self, t, a1, a2, theta):
        b = theta
        a1, a2 = a2, a1
        a1tp = (a1 * (1 - t))**(1/b)
        a2tp = (a2 * t)**(1/b)
        a1tn = (a1 * (1 - t))**(-1/b)
        a2tn = (a2 * t)**(-1/b)

        t1 = (b + 1) * a2tp * a1tp * (a1tn + a2tn)**(-b)
        t2 = b * (1 - t)**2 * t**2 * (a1tp + a2tp)**2
        d2 = t1 / t2
        return d2


transform_joe = AsymNegLogistic()


class AsymMixed(PickandDependence):
    '''asymmetric mixed model of Tawn 1988

    special case:  k=0, theta in [0,1] : symmetric mixed model of
        Tiago de Oliveira 1980

    restrictions:
     - theta > 0
     - theta + 3*k > 0
     - theta + k <= 1
     - theta + 2*k <= 1
    '''
    k_args = 2

    def _check_args(self, theta, k):
        condth = (theta >= 0)
        cond1 = (theta + 3*k > 0) and (theta + k <= 1) and (theta + 2*k <= 1)
        return condth & cond1

    def evaluate(self, t, theta, k):
        transf = 1 - (theta + k) * t + theta * t*t + k * t**3
        return transf

    def deriv(self, t, theta, k):
        d_dt = - (theta + k) + 2 * theta * t + 3 * k * t**2
        return d_dt

    def deriv2(self, t, theta, k):
        d2_dt2 = 2 * theta + 6 * k * t
        return d2_dt2


# backwards compatibility for now
transform_tawn2 = AsymMixed()


class AsymBiLogistic(PickandDependence):
    '''bilogistic model of Coles and Tawn 1994, Joe, Smith and Weissman 1992

    restrictions:
     - (beta, delta) in (0,1)^2 or
     - (beta, delta) in (-inf,0)^2

    not vectorized because of numerical integration
    '''
    k_args = 2

    def _check_args(self, beta, delta):
        cond1 = (beta > 0) and (beta <= 1) and (delta > 0) and (delta <= 1)
        cond2 = (beta < 0) and (delta < 0)
        return cond1 | cond2

    def evaluate(self, t, beta, delta):
        # if not np.all(_check_args(beta, delta)):
        #    raise ValueError('invalid args')

        def _integrant(w):
            term1 = (1 - beta) * np.power(w, -beta) * (1-t)
            term2 = (1 - delta) * np.power(1-w, -delta) * t
            return np.maximum(term1, term2)

        from scipy.integrate import quad
        transf = quad(_integrant, 0, 1)[0]
        return transf


transform_bilogistic = AsymBiLogistic()


class HR(PickandDependence):
    '''model of Huesler Reiss 1989

    special case:  a1=a2=1 : symmetric negative logistic of Galambos 1978

    restrictions:
     - lambda in (0,inf)
    '''
    k_args = 1

    def _check_args(self, lamda):
        cond = (lamda > 0)
        return cond

    def evaluate(self, t, lamda):
        # if not np.all(self._check_args(lamda)):
        #    raise ValueError('invalid args')

        term = np.log((1. - t) / t) * 0.5 / lamda

        from scipy.stats import norm
        # use special if I want to avoid stats import
        transf = ((1 - t) * norm._cdf(lamda + term) +
                  t * norm._cdf(lamda - term))
        return transf

    def _derivs(self, t, lamda, order=(1, 2)):
        if not isinstance(order, (int, np.integer)):
            if (1 in order) and (2 in order):
                order = -1
            else:
                raise ValueError("order should be 1, 2, or (1,2)")

        dn = 1 / np.sqrt(2 * np.pi)
        a = lamda
        g = np.log((1. - t) / t) * 0.5 / a
        gd1 = 1 / (2 * a * (t - 1) * t)
        gd2 = (0.5 - t) / (a * ((1 - t) * t)**2)
        # f = stats.norm.cdf(t)
        # fd1 = np.exp(-t**2 / 2) / sqrt(2 * np.pi)  # stats.norm.pdf(t)
        # fd2 = fd1 * t
        tp = a + g
        fp = stats.norm.cdf(tp)
        fd1p = np.exp(-tp**2 / 2) * dn  # stats.norm.pdf(t)
        fd2p = -fd1p * tp
        tn = a - g
        fn = stats.norm.cdf(tn)
        fd1n = np.exp(-tn**2 / 2) * dn  # stats.norm.pdf(t)
        fd2n = -fd1n * tn

        if order in (1, -1):
            # d1 = g'(t) (-t f'(a - g(t)) - (t - 1) f'(a + g(t))) + f(a - g(t))
            #      - f(a + g(t))
            d1 = gd1 * (-t * fd1n - (t - 1) * fd1p) + fn - fp
        if order in (2, -1):
            # d2 = g'(t)^2 (t f''(a - g(t)) - (t - 1) f''(a + g(t))) +
            #     (-(t - 1) g''(t) - 2 g'(t)) f'(a + g(t)) -
            #     (t g''(t) + 2 g'(t)) f'(a - g(t))
            d2 = (gd1**2 * (t * fd2n - (t - 1) * fd2p) +
                  (-(t - 1) * gd2 - 2 * gd1) * fd1p -
                  (t * gd2 + 2 * gd1) * fd1n
                  )

        if order == 1:
            return d1
        elif order == 2:
            return d2
        elif order == -1:
            return (d1, d2)

    def deriv(self, t, lamda):
        return self._derivs(t, lamda, 1)

    def deriv2(self, t, lamda):
        return self._derivs(t, lamda, 2)


transform_hr = HR()


# def transform_tev(t, rho, df):
class TEV(PickandDependence):
    '''t-EV model of Demarta and McNeil 2005

    restrictions:
     - rho in (-1,1)
     - x > 0
    '''
    k_args = 2

    def _check_args(self, rho, df):
        x = df  # alias, Genest and Segers use chi, copual package uses df
        cond1 = (x > 0)
        cond2 = (rho > 0) and (rho < 1)
        return cond1 and cond2

    def evaluate(self, t, rho, df):
        x = df  # alias, Genest and Segers use chi, copual package uses df
        # if not np.all(self, _check_args(rho, x)):
        #    raise ValueError('invalid args')

        from scipy.stats import t as stats_t
        # use special if I want to avoid stats import

        term1 = (np.power(t/(1.-t), 1./x) - rho)  # for t
        term2 = (np.power((1.-t)/t, 1./x) - rho)  # for 1-t
        term0 = np.sqrt(1. + x) / np.sqrt(1 - rho*rho)
        z1 = term0 * term1
        z2 = term0 * term2
        transf = t * stats_t._cdf(z1, x+1) + (1 - t) * stats_t._cdf(z2, x+1)
        return transf


transform_tev = TEV()
