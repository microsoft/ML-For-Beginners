## copied from nonlinear_transform_gen.py

''' A class for the distribution of a non-linear monotonic transformation of a continuous random variable

simplest usage:
example: create log-gamma distribution, i.e. y = log(x),
            where x is gamma distributed (also available in scipy.stats)
    loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp)

example: what is the distribution of the discount factor y=1/(1+x)
            where interest rate x is normally distributed with N(mux,stdx**2)')?
            (just to come up with a story that implies a nice transformation)
    invnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, a=-np.inf)

This class does not work well for distributions with difficult shapes,
    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.

Note: I'm working from my version of scipy.stats.distribution.
      But this script runs under scipy 0.6.0 (checked with numpy: 1.2.0rc2 and python 2.4)

This is not yet thoroughly tested, polished or optimized

TODO:
  * numargs handling is not yet working properly, numargs needs to be specified (default = 0 or 1)
  * feeding args and kwargs to underlying distribution is untested and incomplete
  * distinguish args and kwargs for the transformed and the underlying distribution
    - currently all args and no kwargs are transmitted to underlying distribution
    - loc and scale only work for transformed, but not for underlying distribution
    - possible to separate args for transformation and underlying distribution parameters

  * add _rvs as method, will be faster in many cases


Created on Tuesday, October 28, 2008, 12:40:37 PM
Author: josef-pktd
License: BSD

'''
from scipy import stats
from scipy.stats import distributions
import numpy as np


def get_u_argskwargs(**kwargs):
    # Todo: What's this? wrong spacing, used in Transf_gen TransfTwo_gen
    u_kwargs = dict((k.replace('u_', '', 1), v) for k, v in kwargs.items()
                    if k.startswith('u_'))
    u_args = u_kwargs.pop('u_args', None)
    return u_args, u_kwargs


class Transf_gen(distributions.rv_continuous):
    '''a class for non-linear monotonic transformation of a continuous random variable

    '''

    def __init__(self, kls, func, funcinv, *args, **kwargs):
        # print(args
        # print(kwargs

        self.func = func
        self.funcinv = funcinv
        # explicit for self.__dict__.update(kwargs)
        # need to set numargs because inspection does not work
        self.numargs = kwargs.pop('numargs', 0)
        # print(self.numargs
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)
        b = kwargs.pop('b', np.inf)
        self.decr = kwargs.pop('decr', False)
        # defines whether it is a decreasing (True)
        #       or increasing (False) monotonic transformation

        self.u_args, self.u_kwargs = get_u_argskwargs(**kwargs)
        self.kls = kls  # (self.u_args, self.u_kwargs)
        # possible to freeze the underlying distribution

        super(Transf_gen, self).__init__(a=a, b=b, name=name,
                                         shapes=kls.shapes,
                                         longname=longname,
                                         # extradoc = extradoc
                                         )

    def _cdf(self, x, *args, **kwargs):
        # print(args
        if not self.decr:
            return self.kls._cdf(self.funcinv(x), *args, **kwargs)
            # note scipy _cdf only take *args not *kwargs
        else:
            return 1.0 - self.kls._cdf(self.funcinv(x), *args, **kwargs)

    def _ppf(self, q, *args, **kwargs):
        if not self.decr:
            return self.func(self.kls._ppf(q, *args, **kwargs))
        else:
            return self.func(self.kls._ppf(1 - q, *args, **kwargs))


def inverse(x):
    return np.divide(1.0, x)


mux, stdx = 0.05, 0.1
mux, stdx = 9.0, 1.0


def inversew(x):
    return 1.0 / (1 + mux + x * stdx)


def inversew_inv(x):
    return (1.0 / x - 1.0 - mux) / stdx  # .np.divide(1.0,x)-10


def identit(x):
    return x


invdnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True,  # a=-np.inf,
                         numargs=0, name='discf', longname='normal-based discount factor',
                         # extradoc = '\ndistribution of discount factor y=1/(1+x)) with x N(0.05,0.1**2)'
                         )

lognormalg = Transf_gen(stats.norm, np.exp, np.log,
                        numargs=2, a=0, name='lnnorm',
                        longname='Exp transformed normal',
                        # extradoc = '\ndistribution of y = exp(x), with x standard normal'
                        # 'precision for moment andstats is not very high, 2-3 decimals'
                        )

loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp, numargs=1)

## copied form nonlinear_transform_short.py

'''univariate distribution of a non-linear monotonic transformation of a
random variable

'''


class ExpTransf_gen(distributions.rv_continuous):
    '''Distribution based on log/exp transformation

    the constructor can be called with a distribution class
    and generates the distribution of the transformed random variable

    '''

    def __init__(self, kls, *args, **kwargs):
        # print(args
        # print(kwargs
        # explicit for self.__dict__.update(kwargs)
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0
        super(ExpTransf_gen, self).__init__(a=a, name=name)
        self.kls = kls

    def _cdf(self, x, *args):
        # print(args
        return self.kls._cdf(np.log(x), *args)

    def _ppf(self, q, *args):
        return np.exp(self.kls._ppf(q, *args))


class LogTransf_gen(distributions.rv_continuous):
    '''Distribution based on log/exp transformation

    the constructor can be called with a distribution class
    and generates the distribution of the transformed random variable

    '''

    def __init__(self, kls, *args, **kwargs):
        # explicit for self.__dict__.update(kwargs)
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0

        super(LogTransf_gen, self).__init__(a=a, name=name)
        self.kls = kls

    def _cdf(self, x, *args):
        # print(args
        return self.kls._cdf(np.exp(x), *args)

    def _ppf(self, q, *args):
        return np.log(self.kls._ppf(q, *args))


def examples_transf():
    ##lognormal = ExpTransf(a=0.0, xa=-10.0, name = 'Log transformed normal')
    ##print(lognormal.cdf(1)
    ##print(stats.lognorm.cdf(1,1)
    ##print(lognormal.stats()
    ##print(stats.lognorm.stats(1)
    ##print(lognormal.rvs(size=10)

    print('Results for lognormal')
    lognormalg = ExpTransf_gen(stats.norm, a=0, name='Log transformed normal general')
    print(lognormalg.cdf(1))
    print(stats.lognorm.cdf(1, 1))
    print(lognormalg.stats())
    print(stats.lognorm.stats(1))
    print(lognormalg.rvs(size=5))

    ##print('Results for loggamma'
    ##loggammag = ExpTransf_gen(stats.gamma)
    ##print(loggammag._cdf(1,10)
    ##print(stats.loggamma.cdf(1,10)

    print('Results for expgamma')
    loggammaexpg = LogTransf_gen(stats.gamma)
    print(loggammaexpg._cdf(1, 10))
    print(stats.loggamma.cdf(1, 10))
    print(loggammaexpg._cdf(2, 15))
    print(stats.loggamma.cdf(2, 15))

    # this requires change in scipy.stats.distribution
    # print(loggammaexpg.cdf(1,10)

    print('Results for loglaplace')
    loglaplaceg = LogTransf_gen(stats.laplace)
    print(loglaplaceg._cdf(2, 10))
    print(stats.loglaplace.cdf(2, 10))
    loglaplaceexpg = ExpTransf_gen(stats.laplace)
    print(loglaplaceexpg._cdf(2, 10))


## copied from transformtwo.py

'''
Created on Apr 28, 2009

@author: Josef Perktold
'''

''' A class for the distribution of a non-linear u-shaped or hump shaped transformation of a
continuous random variable

This is a companion to the distributions of non-linear monotonic transformation to the case
when the inverse mapping is a 2-valued correspondence, for example for absolute value or square

simplest usage:
example: create squared distribution, i.e. y = x**2,
            where x is normal or t distributed


This class does not work well for distributions with difficult shapes,
    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.


This verifies for normal - chi2, normal - halfnorm, foldnorm, and t - F

TODO:
  * numargs handling is not yet working properly,
    numargs needs to be specified (default = 0 or 1)
  * feeding args and kwargs to underlying distribution works in t distribution example
  * distinguish args and kwargs for the transformed and the underlying distribution
    - currently all args and no kwargs are transmitted to underlying distribution
    - loc and scale only work for transformed, but not for underlying distribution
    - possible to separate args for transformation and underlying distribution parameters

  * add _rvs as method, will be faster in many cases

'''


class TransfTwo_gen(distributions.rv_continuous):
    '''Distribution based on a non-monotonic (u- or hump-shaped transformation)

    the constructor can be called with a distribution class, and functions
    that define the non-linear transformation.
    and generates the distribution of the transformed random variable

    Note: the transformation, it's inverse and derivatives need to be fully
    specified: func, funcinvplus, funcinvminus, derivplus,  derivminus.
    Currently no numerical derivatives or inverse are calculated

    This can be used to generate distribution instances similar to the
    distributions in scipy.stats.

    '''

    # a class for non-linear non-monotonic transformation of a continuous random variable
    def __init__(self, kls, func, funcinvplus, funcinvminus, derivplus,
                 derivminus, *args, **kwargs):
        # print(args
        # print(kwargs

        self.func = func
        self.funcinvplus = funcinvplus
        self.funcinvminus = funcinvminus
        self.derivplus = derivplus
        self.derivminus = derivminus
        # explicit for self.__dict__.update(kwargs)
        # need to set numargs because inspection does not work
        self.numargs = kwargs.pop('numargs', 0)
        # print(self.numargs
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)  # attached to self in super
        b = kwargs.pop('b', np.inf)  # self.a, self.b would be overwritten
        self.shape = kwargs.pop('shape', False)
        # defines whether it is a `u` shaped or `hump' shaped
        #       transformation

        self.u_args, self.u_kwargs = get_u_argskwargs(**kwargs)
        self.kls = kls  # (self.u_args, self.u_kwargs)
        # possible to freeze the underlying distribution

        super(TransfTwo_gen, self).__init__(a=a, b=b,
                                            name=name,
                                            shapes=kls.shapes,
                                            longname=longname,
                                            # extradoc = extradoc
                                            )

    def _rvs(self, *args):
        self.kls._size = self._size  # size attached to self, not function argument
        return self.func(self.kls._rvs(*args))

    def _pdf(self, x, *args, **kwargs):
        # print(args
        if self.shape == 'u':
            signpdf = 1
        elif self.shape == 'hump':
            signpdf = -1
        else:
            raise ValueError('shape can only be `u` or `hump`')

        return signpdf * (self.derivplus(x) * self.kls._pdf(self.funcinvplus(x), *args, **kwargs) -
                          self.derivminus(x) * self.kls._pdf(self.funcinvminus(x), *args,
                                                             **kwargs))
        # note scipy _cdf only take *args not *kwargs

    def _cdf(self, x, *args, **kwargs):
        # print(args
        if self.shape == 'u':
            return self.kls._cdf(self.funcinvplus(x), *args, **kwargs) - \
                self.kls._cdf(self.funcinvminus(x), *args, **kwargs)
            # note scipy _cdf only take *args not *kwargs
        else:
            return 1.0 - self._sf(x, *args, **kwargs)

    def _sf(self, x, *args, **kwargs):
        # print(args
        if self.shape == 'hump':
            return self.kls._cdf(self.funcinvplus(x), *args, **kwargs) - \
                self.kls._cdf(self.funcinvminus(x), *args, **kwargs)
            # note scipy _cdf only take *args not *kwargs
        else:
            return 1.0 - self._cdf(x, *args, **kwargs)

    def _munp(self, n, *args, **kwargs):
        return self._mom0_sc(n, *args)


# ppf might not be possible in general case?
# should be possible in symmetric case
#    def _ppf(self, q, *args, **kwargs):
#        if self.shape == 'u':
#            return self.func(self.kls._ppf(q,*args, **kwargs))
#        elif self.shape == 'hump':
#            return self.func(self.kls._ppf(1-q,*args, **kwargs))

# TODO: rename these functions to have unique names

class SquareFunc:
    '''class to hold quadratic function with inverse function and derivative

    using instance methods instead of class methods, if we want extension
    to parametrized function
    '''

    def inverseplus(self, x):
        return np.sqrt(x)

    def inverseminus(self, x):
        return 0.0 - np.sqrt(x)

    def derivplus(self, x):
        return 0.5 / np.sqrt(x)

    def derivminus(self, x):
        return 0.0 - 0.5 / np.sqrt(x)

    def squarefunc(self, x):
        return np.power(x, 2)


sqfunc = SquareFunc()

squarenormalg = TransfTwo_gen(stats.norm, sqfunc.squarefunc, sqfunc.inverseplus,
                              sqfunc.inverseminus, sqfunc.derivplus, sqfunc.derivminus,
                              shape='u', a=0.0, b=np.inf,
                              numargs=0, name='squarenorm', longname='squared normal distribution',
                              # extradoc = '\ndistribution of the square of a normal random variable' +\
                              #            ' y=x**2 with x N(0.0,1)'
                              )
# u_loc=l, u_scale=s)
squaretg = TransfTwo_gen(stats.t, sqfunc.squarefunc, sqfunc.inverseplus,
                         sqfunc.inverseminus, sqfunc.derivplus, sqfunc.derivminus,
                         shape='u', a=0.0, b=np.inf,
                         numargs=1, name='squarenorm', longname='squared t distribution',
                         # extradoc = '\ndistribution of the square of a t random variable' +\
                         #           ' y=x**2 with x t(dof,0.0,1)'
                         )


def inverseplus(x):
    return np.sqrt(-x)


def inverseminus(x):
    return 0.0 - np.sqrt(-x)


def derivplus(x):
    return 0.0 - 0.5 / np.sqrt(-x)


def derivminus(x):
    return 0.5 / np.sqrt(-x)


def negsquarefunc(x):
    return -np.power(x, 2)


negsquarenormalg = TransfTwo_gen(stats.norm, negsquarefunc, inverseplus, inverseminus,
                                 derivplus, derivminus, shape='hump', a=-np.inf, b=0.0,
                                 numargs=0, name='negsquarenorm',
                                 longname='negative squared normal distribution',
                                 # extradoc = '\ndistribution of the negative square of a normal random variable' +\
                                 #            ' y=-x**2 with x N(0.0,1)'
                                 )


# u_loc=l, u_scale=s)

def inverseplus(x):
    return x


def inverseminus(x):
    return 0.0 - x


def derivplus(x):
    return 1.0


def derivminus(x):
    return 0.0 - 1.0


def absfunc(x):
    return np.abs(x)


absnormalg = TransfTwo_gen(stats.norm, np.abs, inverseplus, inverseminus,
                           derivplus, derivminus, shape='u', a=0.0, b=np.inf,
                           numargs=0, name='absnorm', longname='absolute of normal distribution',
                           # extradoc = '\ndistribution of the absolute value of a normal random variable' +\
                           #           ' y=abs(x) with x N(0,1)'
                           )
