# -*- coding: utf-8 -*-
#pylint: disable-msg=W0142
"""Statistical power, solving for nobs, ... - trial version

Created on Sat Jan 12 21:48:06 2013

Author: Josef Perktold

Example
roundtrip - root with respect to all variables

       calculated, desired
nobs   33.367204205 33.367204205
effect 0.5 0.5
alpha  0.05 0.05
power   0.8 0.8


TODO:
refactoring
 - rename beta -> power,    beta (type 2 error is beta = 1-power)  DONE
 - I think the current implementation can handle any kinds of extra keywords
   (except for maybe raising meaningful exceptions
 - streamline code, I think internally classes can be merged
   how to extend to k-sample tests?
   user interface for different tests that map to the same (internal) test class
 - sequence of arguments might be inconsistent,
   arg and/or kwds so python checks what's required and what can be None.
 - templating for docstrings ?


"""
import warnings

import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding


def nct_cdf(x, df, nc):
    return special.nctdtr(df, nc, x)


def nct_sf(x, df, nc):
    return 1 - special.nctdtr(df, nc, x)


def ncf_cdf(x, dfn, dfd, nc):
    return special.ncfdtr(dfn, dfd, nc, x)


def ncf_sf(x, dfn, dfd, nc):
    return 1 - special.ncfdtr(dfn, dfd, nc, x)


def ncf_ppf(q, dfn, dfd, nc):
    return special.ncfdtri(dfn, dfd, nc, q)


def ttest_power(effect_size, nobs, alpha, df=None, alternative='two-sided'):
    '''Calculate power of a ttest
    '''
    d = effect_size
    if df is None:
        df = nobs - 1

    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.  #no inplace changes, does not work
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " +
                         "or 'smaller'")

    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit_upp = stats.t.isf(alpha_, df)
        #print crit_upp, df, d*np.sqrt(nobs)
        # use private methods, generic methods return nan with negative d
        if np.any(np.isnan(crit_upp)):
            # avoid endless loop, https://github.com/scipy/scipy/issues/2667
            pow_ = np.nan
        else:
            # pow_ = stats.nct._sf(crit_upp, df, d*np.sqrt(nobs))
            # use scipy.special
            pow_ = nct_sf(crit_upp, df, d*np.sqrt(nobs))
    if alternative in ['two-sided', '2s', 'smaller']:
        crit_low = stats.t.ppf(alpha_, df)
        #print crit_low, df, d*np.sqrt(nobs)
        if np.any(np.isnan(crit_low)):
            pow_ = np.nan
        else:
            # pow_ += stats.nct._cdf(crit_low, df, d*np.sqrt(nobs))
            pow_ += nct_cdf(crit_low, df, d*np.sqrt(nobs))
    return pow_


def normal_power(effect_size, nobs, alpha, alternative='two-sided', sigma=1.):
    """Calculate power of a normal distributed test statistic

    This is an generalization of `normal_power` when variance under Null and
    Alternative differ.

    Parameters
    ----------
    effect size : float
        difference in the estimated means or statistics under the alternative
        normalized by the standard deviation (without division by sqrt(nobs).
    nobs : float or int
        number of observations
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        extra argument to choose whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    """

    d = effect_size

    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.  #no inplace changes, does not work
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " +
                         "or 'smaller'")

    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit = stats.norm.isf(alpha_)
        pow_ = stats.norm.sf(crit - d*np.sqrt(nobs)/sigma)
    if alternative in ['two-sided', '2s', 'smaller']:
        crit = stats.norm.ppf(alpha_)
        pow_ += stats.norm.cdf(crit - d*np.sqrt(nobs)/sigma)
    return pow_


def normal_power_het(diff, nobs, alpha, std_null=1., std_alternative=None,
                 alternative='two-sided'):
    """Calculate power of a normal distributed test statistic

    This is an generalization of `normal_power` when variance under Null and
    Alternative differ.

    Parameters
    ----------
    diff : float
        difference in the estimated means or statistics under the alternative.
    nobs : float or int
        number of observations
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    std_null : float
        standard deviation under the Null hypothesis without division by
        sqrt(nobs)
    std_alternative : float
        standard deviation under the Alternative hypothesis without division
        by sqrt(nobs)
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        extra argument to choose whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.

    Returns
    -------
    power : float
    """

    d = diff
    if std_alternative is None:
        std_alternative = std_null

    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.  #no inplace changes, does not work
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " +
                         "or 'smaller'")

    std_ratio = std_null / std_alternative
    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit = stats.norm.isf(alpha_)
        pow_ = stats.norm.sf(crit * std_ratio -
                             d*np.sqrt(nobs) / std_alternative)
    if alternative in ['two-sided', '2s', 'smaller']:
        crit = stats.norm.ppf(alpha_)
        pow_ += stats.norm.cdf(crit * std_ratio -
                               d*np.sqrt(nobs) / std_alternative)
    return pow_


def normal_sample_size_one_tail(diff, power, alpha, std_null=1.,
                                std_alternative=None):
    """explicit sample size computation if only one tail is relevant

    The sample size is based on the power in one tail assuming that the
    alternative is in the tail where the test has power that increases
    with sample size.
    Use alpha/2 to compute the one tail approximation to the two-sided
    test, i.e. consider only one tail of two-sided test.

    Parameters
    ----------
    diff : float
        difference in the estimated means or statistics under the alternative.
    power : float in interval (0,1)
        power of the test, e.g. 0.8, is one minus the probability of a type II
        error. Power is the probability that the test correctly rejects the
        Null Hypothesis if the Alternative Hypothesis is true.
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
        Note: alpha is used for one tail. Use alpha/2 for two-sided
        alternative.
    std_null : float
        standard deviation under the Null hypothesis without division by
        sqrt(nobs)
    std_alternative : float
        standard deviation under the Alternative hypothesis without division
        by sqrt(nobs). Defaults to None. If None, ``std_alternative`` is set
        to the value of ``std_null``.

    Returns
    -------
    nobs : float
        Sample size to achieve (at least) the desired power.
        If the minimum power is satisfied for all positive sample sizes, then
        ``nobs`` will be zero. This will be the case when power <= alpha if
        std_alternative is equal to std_null.

    """

    if std_alternative is None:
        std_alternative = std_null

    crit_power = stats.norm.isf(power)
    crit = stats.norm.isf(alpha)
    n1 = (np.maximum(crit * std_null - crit_power * std_alternative, 0)
          / diff)**2
    return n1


def ftest_anova_power(effect_size, nobs, alpha, k_groups=2, df=None):
    '''power for ftest for one way anova with k equal sized groups

    nobs total sample size, sum over all groups

    should be general nobs observations, k_groups restrictions ???
    '''
    df_num = k_groups - 1
    df_denom = nobs - k_groups
    crit = stats.f.isf(alpha, df_num, df_denom)
    pow_ = ncf_sf(crit, df_num, df_denom, effect_size**2 * nobs)
    return pow_


def ftest_power(effect_size, df2, df1, alpha, ncc=1):
    '''Calculate the power of a F-test.

    Parameters
    ----------
    effect_size : float
        The effect size is here Cohen's ``f``, the square root of ``f2``.
    df2 : int or float
        Denominator degrees of freedom.
        This corresponds to the df_resid in Wald tests.
    df1 : int or float
        Numerator degrees of freedom.
        This corresponds to the number of constraints in Wald tests.
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    ncc : int
        degrees of freedom correction for non-centrality parameter.
        see Notes

    Returns
    -------
    power : float
        Power of the test, e.g. 0.8, is one minus the probability of a
        type II error. Power is the probability that the test correctly
        rejects the Null Hypothesis if the Alternative Hypothesis is true.

    Notes
    -----
    changed in 0.14: use df2, df1 instead of df_num, df_denom as arg names.
    The latter had reversed meaning.

    The sample size is given implicitly by ``df2`` with fixed number of
    constraints given by numerator degrees of freedom ``df1``:

        nobs = df2 + df1 + ncc

    Set ncc=0 to match t-test, or f-test in LikelihoodModelResults.
    ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

    ftest_power with ncc=0 should also be correct for f_test in regression
    models, with df_num (df1) as number of constraints and d_denom (df2) as
    df_resid.
    '''
    df_num, df_denom = df1, df2
    nc = effect_size**2 * (df_denom + df_num + ncc)
    crit = stats.f.isf(alpha, df_num, df_denom)
    # pow_ = stats.ncf.sf(crit, df_num, df_denom, nc)
    # use scipy.special for ncf
    pow_ = ncf_sf(crit, df_num, df_denom, nc)
    return pow_ #, crit, nc


def ftest_power_f2(effect_size, df_num, df_denom, alpha, ncc=1):
    '''Calculate the power of a F-test.

    Based on Cohen's `f^2` effect size.

    This assumes

        df_num : numerator degrees of freedom, (number of constraints)
        df_denom : denominator degrees of freedom (df_resid in regression)
        nobs = df_denom + df_num + ncc
        nc = effect_size * nobs  (noncentrality index)

    Power is computed one-sided in the upper tail.

    Parameters
    ----------
    effect_size : float
        Cohen's f2 effect size or noncentrality divided by nobs.
    df_num : int or float
        Numerator degrees of freedom.
        This corresponds to the number of constraints in Wald tests.
    df_denom : int or float
        Denominator degrees of freedom.
        This corresponds to the df_resid in Wald tests.
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    ncc : int
        degrees of freedom correction for non-centrality parameter.
        see Notes

    Returns
    -------
    power : float
        Power of the test, e.g. 0.8, is one minus the probability of a
        type II error. Power is the probability that the test correctly
        rejects the Null Hypothesis if the Alternative Hypothesis is true.

    Notes

    The sample size is given implicitly by ``df_denom`` with fixed number of
    constraints given by numerator degrees of freedom ``df_num``:

        nobs = df_denom + df_num + ncc

    Set ncc=0 to match t-test, or f-test in LikelihoodModelResults.
    ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

    ftest_power with ncc=0 should also be correct for f_test in regression
    models, with df_num (df1) as number of constraints and d_denom (df2) as
    df_resid.
    '''

    nc = effect_size * (df_denom + df_num + ncc)
    crit = stats.f.isf(alpha, df_num, df_denom)
    # pow_ = stats.ncf.sf(crit, df_num, df_denom, nc)
    # use scipy.special for ncf
    pow_ = ncf_sf(crit, df_num, df_denom, nc)
    return pow_


#class based implementation
#--------------------------

class Power:
    '''Statistical Power calculations, Base Class

    so far this could all be class methods
    '''

    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        # used only for instance level start values
        self.start_ttp = dict(effect_size=0.01, nobs=10., alpha=0.15,
                              power=0.6, nobs1=10., ratio=1,
                              df_num=10, df_denom=3   # for FTestPower
                              )
        # TODO: nobs1 and ratio are for ttest_ind,
        #      need start_ttp for each test/class separately,
        # possible rootfinding problem for effect_size, starting small seems to
        # work
        from collections import defaultdict
        self.start_bqexp = defaultdict(dict)
        for key in ['nobs', 'nobs1', 'df_num', 'df_denom']:
            self.start_bqexp[key] = dict(low=2., start_upp=50.)
        for key in ['df_denom']:
            self.start_bqexp[key] = dict(low=1., start_upp=50.)
        for key in ['ratio']:
            self.start_bqexp[key] = dict(low=1e-8, start_upp=2)
        for key in ['alpha']:
            self.start_bqexp[key] = dict(low=1e-12, upp=1 - 1e-12)

    def power(self, *args, **kwds):
        raise NotImplementedError

    def _power_identity(self, *args, **kwds):
        power_ = kwds.pop('power')
        return self.power(*args, **kwds) - power_

    def solve_power(self, **kwds):
        '''solve for any one of the parameters of a t-test

        for t-test the keywords are:
            effect_size, nobs, alpha, power

        exactly one needs to be ``None``, all others need numeric values

        *attaches*

        cache_fit_res : list
            Cache of the result of the root finding procedure for the latest
            call to ``solve_power``, mainly for debugging purposes.
            The first element is the success indicator, one if successful.
            The remaining elements contain the return information of the up to
            three solvers that have been tried.


        '''
        #TODO: maybe use explicit kwds,
        #    nicer but requires inspect? and not generic across tests
        #    I'm duplicating this in the subclass to get informative docstring
        key = [k for k,v in kwds.items() if v is None]
        #print kwds, key
        if len(key) != 1:
            raise ValueError('need exactly one keyword that is None')
        key = key[0]

        if key == 'power':
            del kwds['power']
            return self.power(**kwds)

        if kwds['effect_size'] == 0:
            import warnings
            from statsmodels.tools.sm_exceptions import HypothesisTestWarning
            warnings.warn('Warning: Effect size of 0 detected', HypothesisTestWarning)
            if key == 'power':
                return kwds['alpha']
            if key == 'alpha':
                return kwds['power']
            else:
                raise ValueError('Cannot detect an effect-size of 0. Try changing your effect-size.')


        self._counter = 0

        def func(x):
            kwds[key] = x
            fval = self._power_identity(**kwds)
            self._counter += 1
            #print self._counter,
            if self._counter > 500:
                raise RuntimeError('possible endless loop (500 NaNs)')
            if np.isnan(fval):
                return np.inf
            else:
                return fval

        #TODO: I'm using the following so I get a warning when start_ttp is not defined
        try:
            start_value = self.start_ttp[key]
        except KeyError:
            start_value = 0.9
            import warnings
            from statsmodels.tools.sm_exceptions import ValueWarning
            warnings.warn('Warning: using default start_value for {0}'.format(key), ValueWarning)

        fit_kwds = self.start_bqexp[key]
        fit_res = []
        #print vars()
        try:
            val, res = brentq_expanding(func, full_output=True, **fit_kwds)
            failed = False
            fit_res.append(res)
        except ValueError:
            failed = True
            fit_res.append(None)

        success = None
        if (not failed) and res.converged:
            success = 1
        else:
            # try backup
            # TODO: check more cases to make this robust
            if not np.isnan(start_value):
                val, infodict, ier, msg = optimize.fsolve(func, start_value,
                                                          full_output=True) #scalar
                #val = optimize.newton(func, start_value) #scalar
                fval = infodict['fvec']
                fit_res.append(infodict)
            else:
                ier = -1
                fval = 1
                fit_res.append([None])

            if ier == 1 and np.abs(fval) < 1e-4 :
                success = 1
            else:
                #print infodict
                if key in ['alpha', 'power', 'effect_size']:
                    val, r = optimize.brentq(func, 1e-8, 1-1e-8,
                                             full_output=True) #scalar
                    success = 1 if r.converged else 0
                    fit_res.append(r)
                else:
                    success = 0

        if not success == 1:
            import warnings
            from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
                convergence_doc)
            warnings.warn(convergence_doc, ConvergenceWarning)

        #attach fit_res, for reading only, should be needed only for debugging
        fit_res.insert(0, success)
        self.cache_fit_res = fit_res
        return val

    def plot_power(self, dep_var='nobs', nobs=None, effect_size=None,
                   alpha=0.05, ax=None, title=None, plt_kwds=None, **kwds):
        """
        Plot power with number of observations or effect size on x-axis

        Parameters
        ----------
        dep_var : {'nobs', 'effect_size', 'alpha'}
            This specifies which variable is used for the horizontal axis.
            If dep_var='nobs' (default), then one curve is created for each
            value of ``effect_size``. If dep_var='effect_size' or alpha, then
            one curve is created for each value of ``nobs``.
        nobs : {scalar, array_like}
            specifies the values of the number of observations in the plot
        effect_size : {scalar, array_like}
            specifies the values of the effect_size in the plot
        alpha : {float, array_like}
            The significance level (type I error) used in the power
            calculation. Can only be more than a scalar, if ``dep_var='alpha'``
        ax : None or axis instance
            If ax is None, than a matplotlib figure is created. If ax is a
            matplotlib axis instance, then it is reused, and the plot elements
            are created with it.
        title : str
            title for the axis. Use an empty string, ``''``, to avoid a title.
        plt_kwds : {None, dict}
            not used yet
        kwds : dict
            These remaining keyword arguments are used as arguments to the
            power function. Many power function support ``alternative`` as a
            keyword argument, two-sample test support ``ratio``.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.

        Notes
        -----
        This works only for classes where the ``power`` method has
        ``effect_size``, ``nobs`` and ``alpha`` as the first three arguments.
        If the second argument is ``nobs1``, then the number of observations
        in the plot are those for the first sample.
        TODO: fix this for FTestPower and GofChisquarePower

        TODO: maybe add line variable, if we want more than nobs and effectsize
        """
        #if pwr_kwds is None:
        #    pwr_kwds = {}
        from statsmodels.graphics import utils
        from statsmodels.graphics.plottools import rainbow
        fig, ax = utils.create_mpl_ax(ax)
        import matplotlib.pyplot as plt
        colormap = plt.cm.Dark2 #pylint: disable-msg=E1101
        plt_alpha = 1 #0.75
        lw = 2
        if dep_var == 'nobs':
            colors = rainbow(len(effect_size))
            colors = [colormap(i) for i in np.linspace(0, 0.9, len(effect_size))]
            for ii, es in enumerate(effect_size):
                power = self.power(es, nobs, alpha, **kwds)
                ax.plot(nobs, power, lw=lw, alpha=plt_alpha,
                        color=colors[ii], label='es=%4.2F' % es)
                xlabel = 'Number of Observations'
        elif dep_var in ['effect size', 'effect_size', 'es']:
            colors = rainbow(len(nobs))
            colors = [colormap(i) for i in np.linspace(0, 0.9, len(nobs))]
            for ii, n in enumerate(nobs):
                power = self.power(effect_size, n, alpha, **kwds)
                ax.plot(effect_size, power, lw=lw, alpha=plt_alpha,
                        color=colors[ii], label='N=%4.2F' % n)
                xlabel = 'Effect Size'
        elif dep_var in ['alpha']:
            # experimental nobs as defining separate lines
            colors = rainbow(len(nobs))

            for ii, n in enumerate(nobs):
                power = self.power(effect_size, n, alpha, **kwds)
                ax.plot(alpha, power, lw=lw, alpha=plt_alpha,
                        color=colors[ii], label='N=%4.2F' % n)
                xlabel = 'alpha'
        else:
            raise ValueError('depvar not implemented')

        if title is None:
            title = 'Power of Test'
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(loc='lower right')
        return fig


class TTestPower(Power):
    '''Statistical Power calculations for one sample or paired sample t-test

    '''

    def power(self, effect_size, nobs, alpha, df=None, alternative='two-sided'):
        '''Calculate the power of a t-test for one sample or paired samples.

        Parameters
        ----------
        effect_size : float
            standardized effect size, mean divided by the standard deviation.
            effect size has to be positive.
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        df : int or float
            degrees of freedom. By default this is None, and the df from the
            one sample or paired ttest is used, ``df = nobs1 - 1``
        alternative : str, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.
            .

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       '''
        # for debugging
        #print 'calling ttest power with', (effect_size, nobs, alpha, df, alternative)
        return ttest_power(effect_size, nobs, alpha, df=df,
                           alternative=alternative)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs=None, alpha=None, power=None,
                    alternative='two-sided'):
        '''solve for any one parameter of the power of a one sample t-test

        for the one sample t-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        This test can also be used for a paired t-test, where effect size is
        defined in terms of the mean difference, and nobs is the number of
        pairs.

        Parameters
        ----------
        effect_size : float
            Standardized effect size.The effect size is here Cohen's f, square
            root of "f2".
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        alternative : str, 'two-sided' (default) or 'one-sided'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test.
            'one-sided' assumes we are in the relevant tail.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.

        *attaches*

        cache_fit_res : list
            Cache of the result of the root finding procedure for the latest
            call to ``solve_power``, mainly for debugging purposes.
            The first element is the success indicator, one if successful.
            The remaining elements contain the return information of the up to
            three solvers that have been tried.

        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        # for debugging
        #print 'calling ttest solve with', (effect_size, nobs, alpha, power, alternative)
        return super(TTestPower, self).solve_power(effect_size=effect_size,
                                                      nobs=nobs,
                                                      alpha=alpha,
                                                      power=power,
                                                      alternative=alternative)

class TTestIndPower(Power):
    '''Statistical Power calculations for t-test for two independent sample

    currently only uses pooled variance

    '''


    def power(self, effect_size, nobs1, alpha, ratio=1, df=None,
              alternative='two-sided'):
        '''Calculate the power of a t-test for two independent sample

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation. `effect_size` has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ratio given the other
            arguments, it has to be explicitly set to None.
        df : int or float
            degrees of freedom. By default this is None, and the df from the
            ttest with pooled variance is used, ``df = (nobs1 - 1 + nobs2 - 1)``
        alternative : str, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        '''

        nobs2 = nobs1*ratio
        #pooled variance
        if df is None:
            df = (nobs1 - 1 + nobs2 - 1)

        nobs = 1./ (1. / nobs1 + 1. / nobs2)
        #print 'calling ttest power with', (effect_size, nobs, alpha, df, alternative)
        return ttest_power(effect_size, nobs, alpha, df=df, alternative=alternative)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs1=None, alpha=None, power=None,
                    ratio=1., alternative='two-sided'):
        '''solve for any one parameter of the power of a two sample t-test

        for t-test the keywords are:
            effect_size, nobs1, alpha, power, ratio

        exactly one needs to be ``None``, all others need numeric values

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation. `effect_size` has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ratio given the other
            arguments it has to be explicitly set to None.
        alternative : str, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        return super(TTestIndPower, self).solve_power(effect_size=effect_size,
                                                      nobs1=nobs1,
                                                      alpha=alpha,
                                                      power=power,
                                                      ratio=ratio,
                                                      alternative=alternative)

class NormalIndPower(Power):
    '''Statistical Power calculations for z-test for two independent samples.

    currently only uses pooled variance

    '''

    def __init__(self, ddof=0, **kwds):
        self.ddof = ddof
        super(NormalIndPower, self).__init__(**kwds)

    def power(self, effect_size, nobs1, alpha, ratio=1,
              alternative='two-sided'):
        '''Calculate the power of a z-test for two independent sample

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation. effect size has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
            ``ratio`` can be set to zero in order to get the power for a
            one sample test.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
        alternative : str, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        '''

        ddof = self.ddof  # for correlation, ddof=3

        # get effective nobs, factor for std of test statistic
        if ratio > 0:
            nobs2 = nobs1*ratio
            #equivalent to nobs = n1*n2/(n1+n2)=n1*ratio/(1+ratio)
            nobs = 1./ (1. / (nobs1 - ddof) + 1. / (nobs2 - ddof))
        else:
            nobs = nobs1 - ddof
        return normal_power(effect_size, nobs, alpha, alternative=alternative)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs1=None, alpha=None, power=None,
                    ratio=1., alternative='two-sided'):
        '''solve for any one parameter of the power of a two sample z-test

        for z-test the keywords are:
            effect_size, nobs1, alpha, power, ratio

        exactly one needs to be ``None``, all others need numeric values

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation.
            If ratio=0, then this is the standardized mean in the one sample
            test.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
            ``ratio`` can be set to zero in order to get the power for a
            one sample test.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ration given the other
            arguments it has to be explicitly set to None.
        alternative : str, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        return super(NormalIndPower, self).solve_power(effect_size=effect_size,
                                                      nobs1=nobs1,
                                                      alpha=alpha,
                                                      power=power,
                                                      ratio=ratio,
                                                      alternative=alternative)


class FTestPower(Power):
    """Statistical Power calculations for generic F-test of a constraint

    This class is not recommended, use `FTestPowerF2` with corrected interface.

    This is based on Cohen's f as effect size measure.

    Warning: Methods in this class have the names df_num and df_denom reversed.

    See Also
    --------
    FTestPowerF2 :
        Class with Cohen's f-squared as effect size, corrected keyword names.

    Examples
    --------
    Sample size and power for multiple regression base on R-squared

    Compute effect size from R-squared

    >>> r2 = 0.1
    >>> f2 = r2 / (1 - r2)
    >>> f = np.sqrt(f2)
    >>> r2, f2, f
    (0.1, 0.11111111111111112, 0.33333333333333337)

    Find sample size by solving for denominator df, wrongly named ``df_num``

    >>> df1 = 1  # number of constraints in hypothesis test
    >>> df2 = FTestPower().solve_power(effect_size=f, alpha=0.1, power=0.9,
                                       df_denom=df1)
    >>> ncc = 1  # default
    >>> nobs = df2 + df1 + ncc
    >>> df2, nobs
    (76.46459758305376, 78.46459758305376)

    verify power at df2

    >>> FTestPower().power(effect_size=f, alpha=0.1, df_denom=df1, df_num=df2)
    0.8999999972109698

    """

    def power(self, effect_size, df_num, df_denom, alpha, ncc=1):
        '''Calculate the power of a F-test.

        The effect size is Cohen's ``f``, square root of ``f2``.

        The sample size is given by ``nobs = df_denom + df_num + ncc``

        Warning: The meaning of df_num and df_denom is reversed.

        Parameters
        ----------
        effect_size : float
            Standardized effect size. The effect size is here Cohen's ``f``,
            square root of ``f2``.
        df_num : int or float
            Warning incorrect name
            denominator degrees of freedom,
            This corresponds to the number of constraints in Wald tests.
        df_denom : int or float
            Warning incorrect name
            numerator degrees of freedom.
            This corresponds to the df_resid in Wald tests.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ncc : int
            degrees of freedom correction for non-centrality parameter.
            see Notes

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Notes
        -----

        sample size is given implicitly by df_num

        set ncc=0 to match t-test, or f-test in LikelihoodModelResults.
        ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

        ftest_power with ncc=0 should also be correct for f_test in regression
        models, with df_num and d_denom as defined there. (not verified yet)
        '''

        pow_ = ftest_power(effect_size, df_num, df_denom, alpha, ncc=ncc)
        #print effect_size, df_num, df_denom, alpha, pow_
        return pow_

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, df_num=None, df_denom=None,
                    alpha=None, power=None, ncc=1, **kwargs):
        '''solve for any one parameter of the power of a F-test

        for the one sample F-test the keywords are:
            effect_size, df_num, df_denom, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        The effect size is Cohen's ``f``, square root of ``f2``.

        The sample size is given by ``nobs = df_denom + df_num + ncc``.

        Warning: The meaning of df_num and df_denom is reversed.

        Parameters
        ----------
        effect_size : float
            Standardized effect size. The effect size is here Cohen's ``f``,
            square root of ``f2``.
        df_num : int or float
            Warning incorrect name
            denominator degrees of freedom,
            This corresponds to the number of constraints in Wald tests.
            Sample size is given by ``nobs = df_denom + df_num + ncc``
        df_denom : int or float
            Warning incorrect name
            numerator degrees of freedom.
            This corresponds to the df_resid in Wald tests.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ncc : int
            degrees of freedom correction for non-centrality parameter.
            see Notes
        kwargs : empty
            ``kwargs`` are not used and included for backwards compatibility.
            If ``nobs`` is used as keyword, then a warning is issued. All
            other keywords in ``kwargs`` raise a ValueError.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The method uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        if kwargs:
            if "nobs" in kwargs:
                warnings.warn("nobs is not used")
            else:
                raise ValueError(f"incorrect keyword(s) {kwargs}")
        return super(FTestPower, self).solve_power(effect_size=effect_size,
                                                      df_num=df_num,
                                                      df_denom=df_denom,
                                                      alpha=alpha,
                                                      power=power,
                                                      ncc=ncc)



class FTestPowerF2(Power):
    """Statistical Power calculations for generic F-test of a constraint

    This is based on Cohen's f^2 as effect size measure.

    Examples
    --------
    Sample size and power for multiple regression base on R-squared

    Compute effect size from R-squared

    >>> r2 = 0.1
    >>> f2 = r2 / (1 - r2)
    >>> f = np.sqrt(f2)
    >>> r2, f2, f
    (0.1, 0.11111111111111112, 0.33333333333333337)

    Find sample size by solving for denominator degrees of freedom.

    >>> df1 = 1  # number of constraints in hypothesis test
    >>> df2 = FTestPowerF2().solve_power(effect_size=f2, alpha=0.1, power=0.9,
                                         df_num=df1)
    >>> ncc = 1  # default
    >>> nobs = df2 + df1 + ncc
    >>> df2, nobs
    (76.46459758305376, 78.46459758305376)

    verify power at df2

    >>> FTestPowerF2().power(effect_size=f, alpha=0.1, df_num=df1, df_denom=df2)
    0.8999999972109698

    """

    def power(self, effect_size, df_num, df_denom, alpha, ncc=1):
        '''Calculate the power of a F-test.

        The effect size is Cohen's ``f^2``.

        The sample size is given by ``nobs = df_denom + df_num + ncc``

        Parameters
        ----------
        effect_size : float
            The effect size is here Cohen's ``f2``. This is equal to
            the noncentrality of an F-test divided by nobs.
        df_num : int or float
            Numerator degrees of freedom,
            This corresponds to the number of constraints in Wald tests.
        df_denom : int or float
            Denominator degrees of freedom.
            This corresponds to the df_resid in Wald tests.
        alpha : float in interval (0,1)
            Significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ncc : int
            Degrees of freedom correction for non-centrality parameter.
            see Notes

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Notes
        -----
        The sample size is given implicitly by df_denom

        set ncc=0 to match t-test, or f-test in LikelihoodModelResults.
        ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

        ftest_power with ncc=0 should also be correct for f_test in regression
        models, with df_num and d_denom as defined there. (not verified yet)
        '''

        pow_ = ftest_power_f2(effect_size, df_num, df_denom, alpha, ncc=ncc)
        return pow_

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, df_num=None, df_denom=None,
                    alpha=None, power=None, ncc=1):
        '''Solve for any one parameter of the power of a F-test

        for the one sample F-test the keywords are:
            effect_size, df_num, df_denom, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        The effect size is Cohen's ``f2``.

        The sample size is given by ``nobs = df_denom + df_num + ncc``, and
        can be found by solving for df_denom.

        Parameters
        ----------
        effect_size : float
            The effect size is here Cohen's ``f2``. This is equal to
            the noncentrality of an F-test divided by nobs.
        df_num : int or float
            Numerator degrees of freedom,
            This corresponds to the number of constraints in Wald tests.
        df_denom : int or float
            Denominator degrees of freedom.
            This corresponds to the df_resid in Wald tests.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ncc : int
            degrees of freedom correction for non-centrality parameter.
            see Notes

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''

        return super(FTestPowerF2, self).solve_power(effect_size=effect_size,
                                                      df_num=df_num,
                                                      df_denom=df_denom,
                                                      alpha=alpha,
                                                      power=power,
                                                      ncc=ncc)


class FTestAnovaPower(Power):
    '''Statistical Power calculations F-test for one factor balanced ANOVA

    This is based on Cohen's f as effect size measure.

    See Also
    --------
    statsmodels.stats.oneway.effectsize_oneway

    '''

    def power(self, effect_size, nobs, alpha, k_groups=2):
        '''Calculate the power of a F-test for one factor ANOVA.

        Parameters
        ----------
        effect_size : float
            standardized effect size. The effect size is here Cohen's f, square
            root of "f2".
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        k_groups : int or float
            number of groups in the ANOVA or k-sample comparison. Default is 2.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       '''
        return ftest_anova_power(effect_size, nobs, alpha, k_groups=k_groups)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs=None, alpha=None, power=None,
                    k_groups=2):
        '''solve for any one parameter of the power of a F-test

        for the one sample F-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.


        Parameters
        ----------
        effect_size : float
            standardized effect size, mean divided by the standard deviation.
            effect size has to be positive.
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        # update start values for root finding
        if k_groups is not None:
            self.start_ttp['nobs'] = k_groups * 10
            self.start_bqexp['nobs'] = dict(low=k_groups * 2,
                                            start_upp=k_groups * 10)
        # first attempt at special casing
        if effect_size is None:
            return self._solve_effect_size(effect_size=effect_size,
                                           nobs=nobs,
                                           alpha=alpha,
                                           k_groups=k_groups,
                                           power=power)

        return super(FTestAnovaPower, self).solve_power(effect_size=effect_size,
                                                      nobs=nobs,
                                                      alpha=alpha,
                                                      k_groups=k_groups,
                                                      power=power)

    def _solve_effect_size(self, effect_size=None, nobs=None, alpha=None,
                           power=None, k_groups=2):
        '''experimental, test failure in solve_power for effect_size
        '''
        def func(x):
            effect_size = x
            return self._power_identity(effect_size=effect_size,
                                          nobs=nobs,
                                          alpha=alpha,
                                          k_groups=k_groups,
                                          power=power)

        val, r = optimize.brentq(func, 1e-8, 1-1e-8, full_output=True)
        if not r.converged:
            print(r)
        return val


class GofChisquarePower(Power):
    '''Statistical Power calculations for one sample chisquare test

    '''

    def power(self, effect_size, nobs, alpha, n_bins, ddof=0):#alternative='two-sided'):
        '''Calculate the power of a chisquare test for one sample

        Only two-sided alternative is implemented

        Parameters
        ----------
        effect_size : float
            standardized effect size, according to Cohen's definition.
            see :func:`statsmodels.stats.gof.chisquare_effectsize`
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        n_bins : int
            number of bins or cells in the distribution.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       '''
        from statsmodels.stats.gof import chisquare_power
        return chisquare_power(effect_size, nobs, n_bins, alpha, ddof=0)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs=None, alpha=None,
                    power=None, n_bins=2):
        '''solve for any one parameter of the power of a one sample chisquare-test

        for the one sample chisquare-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        n_bins needs to be defined, a default=2 is used.


        Parameters
        ----------
        effect_size : float
            standardized effect size, according to Cohen's definition.
            see :func:`statsmodels.stats.gof.chisquare_effectsize`
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        n_bins : int
            number of bins or cells in the distribution

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        return super(GofChisquarePower, self).solve_power(effect_size=effect_size,
                                                      nobs=nobs,
                                                      n_bins=n_bins,
                                                      alpha=alpha,
                                                      power=power)

class _GofChisquareIndPower(Power):
    '''Statistical Power calculations for chisquare goodness-of-fit test

    TODO: this is not working yet
          for 2sample case need two nobs in function
          no one-sided chisquare test, is there one? use normal distribution?
          -> drop one-sided options?
    '''


    def power(self, effect_size, nobs1, alpha, ratio=1,
              alternative='two-sided'):
        '''Calculate the power of a chisquare for two independent sample

        Parameters
        ----------
        effect_size : float
            standardize effect size, difference between the two means divided
            by the standard deviation. effect size has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ration given the other
            arguments it has to be explicitely set to None.
        alternative : str, 'two-sided' (default) or 'one-sided'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test.
            'one-sided' assumes we are in the relevant tail.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        '''

        from statsmodels.stats.gof import chisquare_power
        nobs2 = nobs1*ratio
        #equivalent to nobs = n1*n2/(n1+n2)=n1*ratio/(1+ratio)
        nobs = 1./ (1. / nobs1 + 1. / nobs2)
        return chisquare_power(effect_size, nobs, alpha)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs1=None, alpha=None, power=None,
                    ratio=1., alternative='two-sided'):
        '''solve for any one parameter of the power of a two sample z-test

        for z-test the keywords are:
            effect_size, nobs1, alpha, power, ratio

        exactly one needs to be ``None``, all others need numeric values

        Parameters
        ----------
        effect_size : float
            standardize effect size, difference between the two means divided
            by the standard deviation.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ration given the other
            arguments it has to be explicitely set to None.
        alternative : str, 'two-sided' (default) or 'one-sided'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test.
            'one-sided' assumes we are in the relevant tail.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        return super(_GofChisquareIndPower, self).solve_power(effect_size=effect_size,
                                                      nobs1=nobs1,
                                                      alpha=alpha,
                                                      power=power,
                                                      ratio=ratio,
                                                      alternative=alternative)

#shortcut functions
tt_solve_power = TTestPower().solve_power
tt_ind_solve_power = TTestIndPower().solve_power
zt_ind_solve_power = NormalIndPower().solve_power
