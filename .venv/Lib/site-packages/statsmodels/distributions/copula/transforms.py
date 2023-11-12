# -*- coding: utf-8 -*-
""" Transformation Classes as generators for Archimedean copulas


Created on Wed Jan 27 14:33:40 2021

Author: Josef Perktold
License: BSD-3

"""
import warnings

import numpy as np
from scipy.special import expm1, gamma


class Transforms:

    def __init__(self):
        pass

    def deriv2_inverse(self, phi, args):
        t = self.inverse(phi, args)
        phi_d1 = self.deriv(t, args)
        phi_d2 = self.deriv2(t, args)
        return np.abs(phi_d2 / phi_d1**3)

    def derivk_inverse(self, k, phi, theta):
        raise NotImplementedError("not yet implemented")


class TransfFrank(Transforms):

    def evaluate(self, t, theta):
        t = np.asarray(t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            val = -(np.log(-expm1(-theta*t)) - np.log(-expm1(-theta)))
        return val
        # return - np.log(expm1(-theta*t) / expm1(-theta))

    def inverse(self, phi, theta):
        phi = np.asarray(phi)
        return -np.log1p(np.exp(-phi) * expm1(-theta)) / theta

    def deriv(self, t, theta):
        t = np.asarray(t)
        tmp = np.exp(-t*theta)
        return -theta * tmp/(tmp - 1)

    def deriv2(self, t, theta):
        t = np.asarray(t)
        tmp = np.exp(theta * t)
        d2 = - theta**2 * tmp / (tmp - 1)**2
        return d2

    def deriv2_inverse(self, phi, theta):

        et = np.exp(theta)
        ept = np.exp(phi + theta)
        d2 = (et - 1) * ept / (theta * (ept - et + 1)**2)
        return d2

    def deriv3_inverse(self, phi, theta):
        et = np.exp(theta)
        ept = np.exp(phi + theta)
        d3 = -(((et - 1) * ept * (ept + et - 1)) /
               (theta * (ept - et + 1)**3))
        return d3

    def deriv4_inverse(self, phi, theta):
        et = np.exp(theta)
        ept = np.exp(phi + theta)
        p = phi
        b = theta
        d4 = ((et - 1) * ept *
              (-4 * ept + np.exp(2 * (p + b)) + 4 * np.exp(p + 2 * b) -
               2 * et + np.exp(2 * b) + 1)
              ) / (b * (ept - et + 1)**4)

        return d4

    def is_completly_monotonic(self, theta):
        # range of theta for which it is copula for d>2 (more than 2 rvs)
        return theta > 0 & theta < 1


class TransfClayton(Transforms):

    def _checkargs(self, theta):
        return theta > 0

    def evaluate(self, t, theta):
        return np.power(t, -theta) - 1.

    def inverse(self, phi, theta):
        return np.power(1 + phi, -1/theta)

    def deriv(self, t, theta):
        return -theta * np.power(t, -theta-1)

    def deriv2(self, t, theta):
        return theta * (theta + 1) * np.power(t, -theta-2)

    def deriv_inverse(self, phi, theta):
        return -(1 + phi)**(-(theta + 1) / theta) / theta

    def deriv2_inverse(self, phi, theta):
        return ((theta + 1) * (1 + phi)**(-1 / theta - 2)) / theta**2

    def deriv3_inverse(self, phi, theta):
        th = theta  # shorthand
        d3 = -((1 + th) * (1 + 2 * th) / th**3 * (1 + phi)**(-1 / th - 3))
        return d3

    def deriv4_inverse(self, phi, theta):
        th = theta  # shorthand
        d4 = ((1 + th) * (1 + 2 * th) * (1 + 3 * th) / th**4
              ) * (1 + phi)**(-1 / th - 4)
        return d4

    def derivk_inverse(self, k, phi, theta):
        thi = 1 / theta  # shorthand
        d4 = (-1)**k * gamma(k + thi) / gamma(thi) * (1 + phi)**(-(k + thi))
        return d4

    def is_completly_monotonic(self, theta):
        return theta > 0


class TransfGumbel(Transforms):
    '''
    requires theta >=1
    '''

    def _checkargs(self, theta):
        return theta >= 1

    def evaluate(self, t, theta):
        return np.power(-np.log(t), theta)

    def inverse(self, phi, theta):
        return np.exp(-np.power(phi, 1. / theta))

    def deriv(self, t, theta):
        return - theta * (-np.log(t))**(theta - 1) / t

    def deriv2(self, t, theta):
        tmp1 = np.log(t)
        d2 = (theta*(-1)**(1 + theta) * tmp1**(theta-1) * (1 - theta) +
              theta*(-1)**(1 + theta)*tmp1**theta)/(t**2*tmp1)
        # d2 = (theta * tmp1**(-1 + theta) * (1 - theta) + theta * tmp1**theta
        #       ) / (t**2 * tmp1)

        return d2

    def deriv2_inverse(self, phi, theta):
        th = theta  # shorthand
        d2 = ((phi**(2 / th) + (th - 1) * phi**(1 / th))) / (phi**2 * th**2)
        d2 *= np.exp(-phi**(1 / th))
        return d2

    def deriv3_inverse(self, phi, theta):
        p = phi  # shorthand
        b = theta
        d3 = (-p**(3 / b) + (3 - 3 * b) * p**(2 / b) +
              ((3 - 2 * b) * b - 1) * p**(1 / b)
              ) / (p * b)**3
        d3 *= np.exp(-p**(1 / b))
        return d3

    def deriv4_inverse(self, phi, theta):
        p = phi  # shorthand
        b = theta
        d4 = (((6 * b**3 - 11 * b**2 + 6. * b - 1) * p**(1 / b) +
               (11 * b**2 - 18 * b + 7) * p**(2 / b) +
               (6 * (b - 1)) * p**(3 / b) +
               p**(4 / b))
              )/(p * b)**4

        d4 *= np.exp(-p**(1 / b))
        return d4

    def is_completly_monotonic(self, theta):
        return theta > 1


class TransfIndep(Transforms):

    def evaluate(self, t, *args):
        t = np.asarray(t)
        return -np.log(t)

    def inverse(self, phi, *args):
        phi = np.asarray(phi)
        return np.exp(-phi)

    def deriv(self, t, *args):
        t = np.asarray(t)
        return - 1./t

    def deriv2(self, t, *args):
        t = np.asarray(t)
        return 1. / t**2

    def deriv2_inverse(self, phi, *args):
        return np.exp(-phi)

    def deriv3_inverse(self, phi, *args):
        return -np.exp(-phi)

    def deriv4_inverse(self, phi, *args):
        return np.exp(-phi)


class _TransfPower(Transforms):
    """generic multivariate Archimedean copula with additional power transforms

    Nelson p.144, equ. 4.5.2

    experimental, not yet tested and used
    """

    def __init__(self, transform):
        self.transform = transform

    def evaluate(self, t, alpha, beta, *tr_args):
        t = np.asarray(t)

        phi = np.power(self.transform.evaluate(np.power(t, alpha), *tr_args),
                       beta)
        return phi

    def inverse(self, phi, alpha, beta, *tr_args):
        phi = np.asarray(phi)
        transf = self.transform
        phi_inv = np.power(transf.evaluate(np.power(phi, 1. / beta), *tr_args),
                           1. / alpha)
        return phi_inv
