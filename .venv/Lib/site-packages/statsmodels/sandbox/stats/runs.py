'''runstest

formulas for mean and var of runs taken from SAS manual NPAR tests, also idea
for runstest_1samp and runstest_2samp

Description in NIST handbook and dataplot does not explain their expected
values, or variance

Note:
There are (at least) two definitions of runs used in literature. The classical
definition which is also used here, is that runs are sequences of identical
observations separated by observations with different realizations.
The second definition allows for overlapping runs, or runs where counting a
run is also started after a run of a fixed length of the same kind.


TODO
* add one-sided tests where possible or where it makes sense

'''

import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like

class Runs:
    '''class for runs in a binary sequence


    Parameters
    ----------
    x : array_like, 1d
        data array,


    Notes
    -----
    This was written as a more general class for runs. This has some redundant
    calculations when only the runs_test is used.

    TODO: make it lazy

    The runs test could be generalized to more than 1d if there is a use case
    for it.

    This should be extended once I figure out what the distribution of runs
    of any length k is.

    The exact distribution for the runs test is also available but not yet
    verified.

    '''

    def __init__(self, x):
        self.x = np.asarray(x)

        self.runstart = runstart = np.nonzero(np.diff(np.r_[[-np.inf], x, [np.inf]]))[0]
        self.runs = runs = np.diff(runstart)
        self.runs_sign = runs_sign = x[runstart[:-1]]
        self.runs_pos = runs[runs_sign==1]
        self.runs_neg = runs[runs_sign==0]
        self.runs_freqs = np.bincount(runs)
        self.n_runs = len(self.runs)
        self.n_pos = (x==1).sum()

    def runs_test(self, correction=True):
        '''basic version of runs test

        Parameters
        ----------
        correction : bool
            Following the SAS manual, for samplesize below 50, the test
            statistic is corrected by 0.5. This can be turned off with
            correction=False, and was included to match R, tseries, which
            does not use any correction.

        pvalue based on normal distribution, with integer correction

        '''
        self.npo = npo = (self.runs_pos).sum()
        self.nne = nne = (self.runs_neg).sum()

        #n_r = self.n_runs
        n = npo + nne
        npn = npo * nne
        rmean = 2. * npn / n + 1
        rvar = 2. * npn * (2.*npn - n) / n**2. / (n-1.)
        rstd = np.sqrt(rvar)
        rdemean = self.n_runs - rmean
        if n >= 50 or not correction:
            z = rdemean
        else:
            if rdemean > 0.5:
                z = rdemean - 0.5
            elif rdemean < 0.5:
                z = rdemean + 0.5
            else:
                z = 0.

        z /= rstd
        pval = 2 * stats.norm.sf(np.abs(z))
        return z, pval

def runstest_1samp(x, cutoff='mean', correction=True):
    '''use runs test on binary discretized data above/below cutoff

    Parameters
    ----------
    x : array_like
        data, numeric
    cutoff : {'mean', 'median'} or number
        This specifies the cutoff to split the data into large and small
        values.
    correction : bool
        Following the SAS manual, for samplesize below 50, the test
        statistic is corrected by 0.5. This can be turned off with
        correction=False, and was included to match R, tseries, which
        does not use any correction.

    Returns
    -------
    z_stat : float
        test statistic, asymptotically normally distributed
    p-value : float
        p-value, reject the null hypothesis if it is below an type 1 error
        level, alpha .

    '''

    x = array_like(x, "x")
    if cutoff == 'mean':
        cutoff = np.mean(x)
    elif cutoff == 'median':
        cutoff = np.median(x)
    else:
        cutoff = float(cutoff)
    xindicator = (x >= cutoff).astype(int)
    return Runs(xindicator).runs_test(correction=correction)

def runstest_2samp(x, y=None, groups=None, correction=True):
    '''Wald-Wolfowitz runstest for two samples

    This tests whether two samples come from the same distribution.

    Parameters
    ----------
    x : array_like
        data, numeric, contains either one group, if y is also given, or
        both groups, if additionally a group indicator is provided
    y : array_like (optional)
        data, numeric
    groups : array_like
        group labels or indicator the data for both groups is given in a
        single 1-dimensional array, x. If group labels are not [0,1], then
    correction : bool
        Following the SAS manual, for samplesize below 50, the test
        statistic is corrected by 0.5. This can be turned off with
        correction=False, and was included to match R, tseries, which
        does not use any correction.

    Returns
    -------
    z_stat : float
        test statistic, asymptotically normally distributed
    p-value : float
        p-value, reject the null hypothesis if it is below an type 1 error
        level, alpha .


    Notes
    -----
    Wald-Wolfowitz runs test.

    If there are ties, then then the test statistic and p-value that is
    reported, is based on the higher p-value between sorting all tied
    observations of the same group


    This test is intended for continuous distributions
    SAS has treatment for ties, but not clear, and sounds more complicated
    (minimum and maximum possible runs prevent use of argsort)
    (maybe it's not so difficult, idea: add small positive noise to first
    one, run test, then to the other, run test, take max(?) p-value - DONE
    This gives not the minimum and maximum of the number of runs, but should
    be close. Not true, this is close to minimum but far away from maximum.
    maximum number of runs would use alternating groups in the ties.)
    Maybe adding random noise would be the better approach.

    SAS has exact distribution for sample size <=30, does not look standard
    but should be easy to add.

    currently two-sided test only

    This has not been verified against a reference implementation. In a short
    Monte Carlo simulation where both samples are normally distribute, the test
    seems to be correctly sized for larger number of observations (30 or
    larger), but conservative (i.e. reject less often than nominal) with a
    sample size of 10 in each group.

    See Also
    --------
    runs_test_1samp
    Runs
    RunsProb

    '''
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
        groups = np.concatenate((np.zeros(len(x)), np.ones(len(y))))
        # note reassigning x
        x = np.concatenate((x, y))
        gruni = np.arange(2)
    elif groups is not None:
        gruni = np.unique(groups)
        if gruni.size != 2:  # pylint: disable=E1103
            raise ValueError('not exactly two groups specified')
        #require groups to be numeric ???
    else:
        raise ValueError('either y or groups is necessary')

    xargsort = np.argsort(x)
    #check for ties
    x_sorted = x[xargsort]
    x_diff = np.diff(x_sorted)  # used for detecting and handling ties
    if x_diff.min() == 0:
        print('ties detected')   #replace with warning
        x_mindiff = x_diff[x_diff > 0].min()
        eps = x_mindiff/2.
        xx = x.copy()  #do not change original, just in case

        xx[groups==gruni[0]] += eps
        xargsort = np.argsort(xx)
        xindicator = groups[xargsort]
        z0, p0 = Runs(xindicator).runs_test(correction=correction)

        xx[groups==gruni[0]] -= eps   #restore xx = x
        xx[groups==gruni[1]] += eps
        xargsort = np.argsort(xx)
        xindicator = groups[xargsort]
        z1, p1 = Runs(xindicator).runs_test(correction=correction)

        idx = np.argmax([p0,p1])
        return [z0, z1][idx], [p0, p1][idx]

    else:
        xindicator = groups[xargsort]
        return Runs(xindicator).runs_test(correction=correction)


class TotalRunsProb:
    '''class for the probability distribution of total runs

    This is the exact probability distribution for the (Wald-Wolfowitz)
    runs test. The random variable is the total number of runs if the
    sample has (n0, n1) observations of groups 0 and 1.


    Notes
    -----
    Written as a class so I can store temporary calculations, but I do not
    think it matters much.

    Formulas taken from SAS manual for one-sided significance level.

    Could be converted to a full univariate distribution, subclassing
    scipy.stats.distributions.

    *Status*
    Not verified yet except for mean.



    '''

    def __init__(self, n0, n1):
        self.n0 = n0
        self.n1 = n1
        self.n = n = n0 + n1
        self.comball = comb(n, n1)

    def runs_prob_even(self, r):
        n0, n1 = self.n0, self.n1
        tmp0 = comb(n0-1, r//2-1)
        tmp1 = comb(n1-1, r//2-1)
        return tmp0 * tmp1 * 2. / self.comball

    def runs_prob_odd(self, r):
        n0, n1 = self.n0, self.n1
        k = (r+1)//2
        tmp0 = comb(n0-1, k-1)
        tmp1 = comb(n1-1, k-2)
        tmp3 = comb(n0-1, k-2)
        tmp4 = comb(n1-1, k-1)
        return (tmp0 * tmp1 + tmp3 * tmp4)  / self.comball

    def pdf(self, r):
        r = np.asarray(r)
        r_isodd = np.mod(r, 2) > 0
        r_odd = r[r_isodd]
        r_even = r[~r_isodd]
        runs_pdf = np.zeros(r.shape)
        runs_pdf[r_isodd] = self.runs_prob_odd(r_odd)
        runs_pdf[~r_isodd] = self.runs_prob_even(r_even)
        return runs_pdf


    def cdf(self, r):
        r_ = np.arange(2,r+1)
        cdfval = self.runs_prob_even(r_[::2]).sum()
        cdfval += self.runs_prob_odd(r_[1::2]).sum()
        return cdfval


class RunsProb:
    '''distribution of success runs of length k or more (classical definition)

    The underlying process is assumed to be a sequence of Bernoulli trials
    of a given length n.

    not sure yet, how to interpret or use the distribution for runs
    of length k or more.

    Musseli also has longest success run, and waiting time distribution
    negative binomial of order k and geometric of order k

    need to compare with Godpole

    need a MonteCarlo function to do some quick tests before doing more


    '''



    def pdf(self, x, k, n, p):
        '''distribution of success runs of length k or more

        Parameters
        ----------
        x : float
            count of runs of length n
        k : int
            length of runs
        n : int
            total number of observations or trials
        p : float
            probability of success in each Bernoulli trial

        Returns
        -------
        pdf : float
            probability that x runs of length of k are observed

        Notes
        -----
        not yet vectorized

        References
        ----------
        Muselli 1996, theorem 3
        '''

        q = 1-p
        m = np.arange(x, (n+1)//(k+1)+1)[:,None]
        terms = (-1)**(m-x) * comb(m, x) * p**(m*k) * q**(m-1) \
                * (comb(n - m*k, m - 1) + q * comb(n - m*k, m))
        return terms.sum(0)

    def pdf_nb(self, x, k, n, p):
        pass
        #y = np.arange(m-1, n-mk+1

'''
>>> [np.sum([RunsProb().pdf(xi, k, 16, 10/16.) for xi in range(0,16)]) for k in range(16)]
[0.99999332193894064, 0.99999999999999367, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
>>> [(np.arange(0,16) * [RunsProb().pdf(xi, k, 16, 10/16.) for xi in range(0,16)]).sum() for k in range(16)]
[6.9998931510341809, 4.1406249999999929, 2.4414062500000075, 1.4343261718749996, 0.83923339843749856, 0.48875808715820324, 0.28312206268310569, 0.1629814505577086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
>>> np.array([(np.arange(0,16) * [RunsProb().pdf(xi, k, 16, 10/16.) for xi in range(0,16)]).sum() for k in range(16)])/11
array([ 0.63635392,  0.37642045,  0.22194602,  0.13039329,  0.07629395,
        0.04443255,  0.02573837,  0.0148165 ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])
>>> np.diff([(np.arange(0,16) * [RunsProb().pdf(xi, k, 16, 10/16.) for xi in range(0,16)]).sum() for k in range(16)][::-1])
array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.16298145,  0.12014061,  0.20563602,
        0.35047531,  0.59509277,  1.00708008,  1.69921875,  2.85926815])
'''



def median_test_ksample(x, groups):
    '''chisquare test for equality of median/location

    This tests whether all groups have the same fraction of observations
    above the median.

    Parameters
    ----------
    x : array_like
        data values stacked for all groups
    groups : array_like
        group labels or indicator

    Returns
    -------
    stat : float
       test statistic
    pvalue : float
       pvalue from the chisquare distribution
    others ????
       currently some test output, table and expected

    '''
    x = np.asarray(x)
    gruni = np.unique(groups)
    xli = [x[groups==group] for group in gruni]
    xmedian = np.median(x)
    counts_larger = np.array([(xg > xmedian).sum() for xg in xli])
    counts = np.array([len(xg) for xg in xli])
    counts_smaller = counts - counts_larger
    nobs = counts.sum()
    n_larger = (x > xmedian).sum()
    n_smaller = nobs - n_larger
    table = np.vstack((counts_smaller, counts_larger))

    #the following should be replaced by chisquare_contingency table
    expected = np.vstack((counts * 1. / nobs * n_smaller,
                          counts * 1. / nobs * n_larger))

    if (expected < 5).any():
        print('Warning: There are cells with less than 5 expected' \
        'observations. The chisquare distribution might not be a good' \
        'approximation for the true distribution.')

    #check ddof
    return stats.chisquare(table.ravel(), expected.ravel(), ddof=1), table, expected




def cochrans_q(x):
    '''Cochran's Q test for identical effect of k treatments

    Cochran's Q is a k-sample extension of the McNemar test. If there are only
    two treatments, then Cochran's Q test and McNemar test are equivalent.

    Test that the probability of success is the same for each treatment.
    The alternative is that at least two treatments have a different
    probability of success.

    Parameters
    ----------
    x : array_like, 2d (N,k)
        data with N cases and k variables

    Returns
    -------
    q_stat : float
       test statistic
    pvalue : float
       pvalue from the chisquare distribution

    Notes
    -----
    In Wikipedia terminology, rows are blocks and columns are treatments.
    The number of rows N, should be large for the chisquare distribution to be
    a good approximation.
    The Null hypothesis of the test is that all treatments have the
    same effect.

    References
    ----------
    https://en.wikipedia.org/wiki/Cochran_test
    SAS Manual for NPAR TESTS

    '''

    warnings.warn("Deprecated, use stats.cochrans_q instead", FutureWarning)

    x = np.asarray(x)
    gruni = np.unique(x)
    N, k = x.shape
    count_row_success = (x==gruni[-1]).sum(1, float)
    count_col_success = (x==gruni[-1]).sum(0, float)
    count_row_ss = count_row_success.sum()
    count_col_ss = count_col_success.sum()
    assert count_row_ss == count_col_ss  #just a calculation check


    #this is SAS manual
    q_stat = (k-1) * (k *  np.sum(count_col_success**2) - count_col_ss**2) \
             / (k * count_row_ss - np.sum(count_row_success**2))

    #Note: the denominator looks just like k times the variance of the
    #columns

    #Wikipedia uses a different, but equivalent expression
##    q_stat = (k-1) * (k *  np.sum(count_row_success**2) - count_row_ss**2) \
##             / (k * count_col_ss - np.sum(count_col_success**2))

    return q_stat, stats.chi2.sf(q_stat, k-1)

def mcnemar(x, y=None, exact=True, correction=True):
    '''McNemar test

    Parameters
    ----------
    x, y : array_like
        two paired data samples. If y is None, then x can be a 2 by 2
        contingency table. x and y can have more than one dimension, then
        the results are calculated under the assumption that axis zero
        contains the observation for the samples.
    exact : bool
        If exact is true, then the binomial distribution will be used.
        If exact is false, then the chisquare distribution will be used, which
        is the approximation to the distribution of the test statistic for
        large sample sizes.
    correction : bool
        If true, then a continuity correction is used for the chisquare
        distribution (if exact is false.)

    Returns
    -------
    stat : float or int, array
        The test statistic is the chisquare statistic if exact is false. If the
        exact binomial distribution is used, then this contains the min(n1, n2),
        where n1, n2 are cases that are zero in one sample but one in the other
        sample.

    pvalue : float or array
        p-value of the null hypothesis of equal effects.

    Notes
    -----
    This is a special case of Cochran's Q test. The results when the chisquare
    distribution is used are identical, except for continuity correction.

    '''

    warnings.warn("Deprecated, use stats.TableSymmetry instead", FutureWarning)

    x = np.asarray(x)
    if y is None and x.shape[0] == x.shape[1]:
        if x.shape[0] != 2:
            raise ValueError('table needs to be 2 by 2')
        n1, n2 = x[1, 0], x[0, 1]
    else:
        # I'm not checking here whether x and y are binary,
        # is not this also paired sign test
        n1 = np.sum(x < y, 0)
        n2 = np.sum(x > y, 0)

    if exact:
        stat = np.minimum(n1, n2)
        # binom is symmetric with p=0.5
        pval = stats.binom.cdf(stat, n1 + n2, 0.5) * 2
        pval = np.minimum(pval, 1)  # limit to 1 if n1==n2
    else:
        corr = int(correction) # convert bool to 0 or 1
        stat = (np.abs(n1 - n2) - corr)**2 / (1. * (n1 + n2))
        df = 1
        pval = stats.chi2.sf(stat, df)
    return stat, pval


def symmetry_bowker(table):
    '''Test for symmetry of a (k, k) square contingency table

    This is an extension of the McNemar test to test the Null hypothesis
    that the contingency table is symmetric around the main diagonal, that is

    n_{i, j} = n_{j, i}  for all i, j

    Parameters
    ----------
    table : array_like, 2d, (k, k)
        a square contingency table that contains the count for k categories
        in rows and columns.

    Returns
    -------
    statistic : float
        chisquare test statistic
    p-value : float
        p-value of the test statistic based on chisquare distribution
    df : int
        degrees of freedom of the chisquare distribution

    Notes
    -----
    Implementation is based on the SAS documentation, R includes it in
    `mcnemar.test` if the table is not 2 by 2.

    The pvalue is based on the chisquare distribution which requires that the
    sample size is not very small to be a good approximation of the true
    distribution. For 2x2 contingency tables exact distribution can be
    obtained with `mcnemar`

    See Also
    --------
    mcnemar


    '''

    warnings.warn("Deprecated, use stats.TableSymmetry instead", FutureWarning)

    table = np.asarray(table)
    k, k2 = table.shape
    if k != k2:
        raise ValueError('table needs to be square')

    #low_idx = np.tril_indices(k, -1)  # this does not have Fortran order
    upp_idx = np.triu_indices(k, 1)

    tril = table.T[upp_idx]   # lower triangle in column order
    triu = table[upp_idx]     # upper triangle in row order

    stat = ((tril - triu)**2 / (tril + triu + 1e-20)).sum()
    df = k * (k-1) / 2.
    pval = stats.chi2.sf(stat, df)

    return stat, pval, df


if __name__ == '__main__':

    x1 = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])

    print(Runs(x1).runs_test())
    print(runstest_1samp(x1, cutoff='mean'))
    print(runstest_2samp(np.arange(16,0,-1), groups=x1))
    print(TotalRunsProb(7,9).cdf(11))
    print(median_test_ksample(np.random.randn(100), np.random.randint(0,2,100)))
    print(cochrans_q(np.random.randint(0,2,(100,8))))
