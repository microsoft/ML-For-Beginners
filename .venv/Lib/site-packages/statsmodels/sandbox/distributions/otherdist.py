'''Parametric Mixture Distributions

Created on Sat Jun 04 2011

Author: Josef Perktold


Notes:

Compound Poisson has mass point at zero
https://en.wikipedia.org/wiki/Compound_Poisson_distribution
and would need special treatment

need a distribution that has discrete mass points and contiuous range, e.g.
compound Poisson, Tweedie (for some parameter range),
pdf of Tobit model (?) - truncation with clipping

Question: Metaclasses and class factories for generating new distributions from
existing distributions by transformation, mixing, compounding

'''


import numpy as np
from scipy import stats

class ParametricMixtureD:
    '''mixtures with a discrete distribution

    The mixing distribution is a discrete distribution like scipy.stats.poisson.
    All distribution in the mixture of the same type and parametrized
    by the outcome of the mixing distribution and have to be a continuous
    distribution (or have a pdf method).
    As an example, a mixture of normal distributed random variables with
    Poisson as the mixing distribution.


    assumes vectorized shape, loc and scale as in scipy.stats.distributions

    assume mixing_dist is frozen

    initialization looks fragile for all possible cases of lower and upper
    bounds of the distributions.

    '''
    def __init__(self, mixing_dist, base_dist, bd_args_func, bd_kwds_func,
                 cutoff=1e-3):
        '''create a mixture distribution

        Parameters
        ----------
        mixing_dist : discrete frozen distribution
            mixing distribution
        base_dist : continuous distribution
            parametrized distributions in the mixture
        bd_args_func : callable
            function that builds the tuple of args for the base_dist.
            The function obtains as argument the values in the support of
            the mixing distribution and should return an empty tuple or
            a tuple of arrays.
        bd_kwds_func : callable
            function that builds the dictionary of kwds for the base_dist.
            The function obtains as argument the values in the support of
            the mixing distribution and should return an empty dictionary or
            a dictionary with arrays as values.
        cutoff : float
            If the mixing distribution has infinite support, then the
            distribution is truncated with approximately (subject to integer
            conversion) the cutoff probability in the missing tail. Random
            draws that are outside the truncated range are clipped, that is
            assigned to the highest or lowest value in the truncated support.

        '''
        self.mixing_dist = mixing_dist
        self.base_dist = base_dist
        #self.bd_args = bd_args
        if not np.isneginf(mixing_dist.dist.a):
            lower = mixing_dist.dist.a
        else:
            lower = mixing_dist.ppf(1e-4)
        if not np.isposinf(mixing_dist.dist.b):
            upper = mixing_dist.dist.b
        else:
            upper = mixing_dist.isf(1e-4)
        self.ma = lower
        self.mb = upper
        mixing_support = np.arange(lower, upper+1)
        self.mixing_probs = mixing_dist.pmf(mixing_support)

        self.bd_args = bd_args_func(mixing_support)
        self.bd_kwds = bd_kwds_func(mixing_support)

    def rvs(self, size=1):
        mrvs = self.mixing_dist.rvs(size)
        #TODO: check strange cases ? this assumes continous integers
        mrvs_idx = (np.clip(mrvs, self.ma, self.mb) - self.ma).astype(int)

        bd_args = tuple(md[mrvs_idx] for md in self.bd_args)
        bd_kwds = dict((k, self.bd_kwds[k][mrvs_idx]) for k in self.bd_kwds)
        kwds = {'size':size}
        kwds.update(bd_kwds)
        rvs = self.base_dist.rvs(*self.bd_args, **kwds)
        return rvs, mrvs_idx





    def pdf(self, x):
        x = np.asarray(x)
        if np.size(x) > 1:
            x = x[...,None] #[None, ...]
        bd_probs = self.base_dist.pdf(x, *self.bd_args, **self.bd_kwds)
        prob = (bd_probs * self.mixing_probs).sum(-1)
        return prob, bd_probs

    def cdf(self, x):
        x = np.asarray(x)
        if np.size(x) > 1:
            x = x[...,None] #[None, ...]
        bd_probs = self.base_dist.cdf(x, *self.bd_args, **self.bd_kwds)
        prob = (bd_probs * self.mixing_probs).sum(-1)
        return prob, bd_probs


#try:

class ClippedContinuous:
    '''clipped continuous distribution with a masspoint at clip_lower


    Notes
    -----
    first version, to try out possible designs
    insufficient checks for valid arguments and not clear
    whether it works for distributions that have compact support

    clip_lower is fixed and independent of the distribution parameters.
    The clip_lower point in the pdf has to be interpreted as a mass point,
    i.e. different treatment in integration and expect function, which means
    none of the generic methods for this can be used.

    maybe this will be better designed as a mixture between a degenerate or
    discrete and a continuous distribution

    Warning: uses equality to check for clip_lower values in function
    arguments, since these are floating points, the comparison might fail
    if clip_lower values are not exactly equal.
    We could add a check whether the values are in a small neighborhood, but
    it would be expensive (need to search and check all values).

    '''

    def __init__(self, base_dist, clip_lower):
        self.base_dist = base_dist
        self.clip_lower = clip_lower

    def _get_clip_lower(self, kwds):
        '''helper method to get clip_lower from kwds or attribute

        '''
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            clip_lower = kwds.pop('clip_lower')
        return clip_lower, kwds

    def rvs(self, *args, **kwds):
        clip_lower, kwds = self._get_clip_lower(kwds)
        rvs_ = self.base_dist.rvs(*args, **kwds)
        #same as numpy.clip ?
        rvs_[rvs_ < clip_lower] = clip_lower
        return rvs_



    def pdf(self, x, *args, **kwds):
        x = np.atleast_1d(x)
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            #allow clip_lower to be a possible parameter
            clip_lower = kwds.pop('clip_lower')
        pdf_raw = np.atleast_1d(self.base_dist.pdf(x, *args, **kwds))
        clip_mask = (x == self.clip_lower)
        if np.any(clip_mask):
            clip_prob = self.base_dist.cdf(clip_lower, *args, **kwds)
            pdf_raw[clip_mask] = clip_prob

        #the following will be handled by sub-classing rv_continuous
        pdf_raw[x < clip_lower] = 0

        return pdf_raw

    def cdf(self, x, *args, **kwds):
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            #allow clip_lower to be a possible parameter
            clip_lower = kwds.pop('clip_lower')
        cdf_raw = self.base_dist.cdf(x, *args, **kwds)

        #not needed if equality test is used
##        clip_mask = (x == self.clip_lower)
##        if np.any(clip_mask):
##            clip_prob = self.base_dist.cdf(clip_lower, *args, **kwds)
##            pdf_raw[clip_mask] = clip_prob

        #the following will be handled by sub-classing rv_continuous
        #if self.a is defined
        cdf_raw[x < clip_lower] = 0

        return cdf_raw

    def sf(self, x, *args, **kwds):
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            #allow clip_lower to be a possible parameter
            clip_lower = kwds.pop('clip_lower')

        sf_raw = self.base_dist.sf(x, *args, **kwds)
        sf_raw[x <= clip_lower] = 1

        return sf_raw


    def ppf(self, x, *args, **kwds):
        raise NotImplementedError

    def plot(self, x, *args, **kwds):

        clip_lower, kwds = self._get_clip_lower(kwds)
        mass = self.pdf(clip_lower, *args, **kwds)
        xr = np.concatenate(([clip_lower+1e-6], x[x>clip_lower]))
        import matplotlib.pyplot as plt
        #x = np.linspace(-4, 4, 21)
        #plt.figure()
        plt.xlim(clip_lower-0.1, x.max())
        #remove duplicate calculation
        xpdf = self.pdf(x, *args, **kwds)
        plt.ylim(0, max(mass, xpdf.max())*1.1)
        plt.plot(xr, self.pdf(xr, *args, **kwds))
        #plt.vline(clip_lower, self.pdf(clip_lower, *args, **kwds))
        plt.stem([clip_lower], [mass],
                 linefmt='b-', markerfmt='bo', basefmt='r-')
        return




if __name__ == '__main__':

    doplots = 1

    #*********** Poisson-Normal Mixture
    mdist = stats.poisson(2.)
    bdist = stats.norm
    bd_args_fn = lambda x: ()
    #bd_kwds_fn = lambda x: {'loc': np.atleast_2d(10./(1+x))}
    bd_kwds_fn = lambda x: {'loc': x, 'scale': 0.1*np.ones_like(x)} #10./(1+x)}


    pd = ParametricMixtureD(mdist, bdist, bd_args_fn, bd_kwds_fn)
    print(pd.pdf(1))
    p, bp = pd.pdf(np.linspace(0,20,21))
    pc, bpc = pd.cdf(np.linspace(0,20,21))
    print(pd.rvs())
    rvs, m = pd.rvs(size=1000)


    if doplots:
        import matplotlib.pyplot as plt
        plt.hist(rvs, bins = 100)
        plt.title('poisson mixture of normal distributions')

    #********** clipped normal distribution (Tobit)

    bdist = stats.norm
    clip_lower_ = 0. #-0.5
    cnorm = ClippedContinuous(bdist, clip_lower_)
    x = np.linspace(1e-8, 4, 11)
    print(cnorm.pdf(x))
    print(cnorm.cdf(x))

    if doplots:
        #plt.figure()
        #cnorm.plot(x)
        plt.figure()
        cnorm.plot(x = np.linspace(-1, 4, 51), loc=0.5, scale=np.sqrt(2))
        plt.title('clipped normal distribution')

        fig = plt.figure()
        for i, loc in enumerate([0., 0.5, 1.,2.]):
            fig.add_subplot(2,2,i+1)
            cnorm.plot(x = np.linspace(-1, 4, 51), loc=loc, scale=np.sqrt(2))
            plt.title('clipped normal, loc = %3.2f' % loc)


        loc = 1.5
        rvs = cnorm.rvs(loc=loc, size=2000)
        plt.figure()
        plt.hist(rvs, bins=50)
        plt.title('clipped normal rvs, loc = %3.2f' % loc)


    #plt.show()
