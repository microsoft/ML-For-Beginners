# -*- coding: utf-8 -*-
"""Nonlinear Transformation classes


Created on Sat Apr 16 16:06:11 2011

Author: Josef Perktold
License : BSD
"""
import numpy as np


class TransformFunction:

    def __call__(self, x):
        self.func(x)



## Hump and U-shaped functions


class SquareFunc(TransformFunction):
    '''class to hold quadratic function with inverse function and derivative

    using instance methods instead of class methods, if we want extension
    to parametrized function
    '''

    def func(self, x):
        return np.power(x, 2.)

    def inverseplus(self, x):
        return np.sqrt(x)

    def inverseminus(self, x):
        return 0.0 - np.sqrt(x)

    def derivplus(self, x):
        return 0.5/np.sqrt(x)

    def derivminus(self, x):
        return 0.0 - 0.5/np.sqrt(x)




class NegSquareFunc(TransformFunction):
    '''negative quadratic function

    '''
    def func(self, x):
        return -np.power(x,2)

    def inverseplus(self, x):
        return np.sqrt(-x)

    def inverseminus(self, x):
        return 0.0 - np.sqrt(-x)

    def derivplus(self, x):
        return 0.0 - 0.5/np.sqrt(-x)

    def derivminus(self, x):
        return 0.5/np.sqrt(-x)


class AbsFunc(TransformFunction):
    '''class for absolute value transformation
    '''

    def func(self, x):
        return np.abs(x)

    def inverseplus(self, x):
        return x

    def inverseminus(self, x):
        return 0.0 - x

    def derivplus(self, x):
        return 1.0

    def derivminus(self, x):
        return 0.0 - 1.0


## monotonic functions
# more monotone functions in families.links, some for restricted domains


class LogFunc(TransformFunction):

    def func(self, x):
        return np.log(x)

    def inverse(self, y):
        return np.exp(y)

    def deriv(self, x):
        return 1./x

class ExpFunc(TransformFunction):


    def func(self, x):
        return np.exp(x)

    def inverse(self, y):
        return np.log(y)

    def deriv(self, x):
        return np.exp(x)


class BoxCoxNonzeroFunc(TransformFunction):

    def __init__(self, lamda):
        self.lamda = lamda

    def func(self, x):
        return (np.power(x, self.lamda) - 1)/self.lamda

    def inverse(self, y):
        return (self.lamda * y + 1)/self.lamda

    def deriv(self, x):
        return np.power(x, self.lamda - 1)


class AffineFunc(TransformFunction):

    def __init__(self, constant, slope):
        self.constant = constant
        self.slope = slope

    def func(self, x):
        return self.constant + self.slope * x

    def inverse(self, y):
        return (y - self.constant) / self.slope

    def deriv(self, x):
        return self.slope


class ChainFunc(TransformFunction):

    def __init__(self, finn, fout):
        self.finn = finn
        self.fout = fout

    def func(self, x):
        return self.fout.func(self.finn.func(x))

    def inverse(self, y):
        return self.f1.inverse(self.fout.inverse(y))

    def deriv(self, x):
        z = self.finn.func(x)
        return self.fout.deriv(z) * self.finn.deriv(x)


#def inverse(x):
#    return np.divide(1.0,x)
#
#mux, stdx = 0.05, 0.1
#mux, stdx = 9.0, 1.0
#def inversew(x):
#    return 1.0/(1+mux+x*stdx)
#def inversew_inv(x):
#    return (1.0/x - 1.0 - mux)/stdx #.np.divide(1.0,x)-10
#
#def identit(x):
#    return x


if __name__ == '__main__':
    absf = AbsFunc()
    absf.func(5) == 5
    absf.func(-5) == 5
    absf.inverseplus(5) == 5
    absf.inverseminus(5) == -5

    chainf = ChainFunc(AffineFunc(1,2), BoxCoxNonzeroFunc(2))
    print(chainf.func(3.))
    chainf2 = ChainFunc(BoxCoxNonzeroFunc(2), AffineFunc(1,2))
    print(chainf.func(3.))
