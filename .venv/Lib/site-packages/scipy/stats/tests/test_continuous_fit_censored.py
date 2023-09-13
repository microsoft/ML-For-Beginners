# Tests for fitting specific distributions to censored data.

import numpy as np
from numpy.testing import assert_allclose

from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
                         gumbel_l, gumbel_r, invgauss, invweibull, laplace,
                         logistic, lognorm, nct, ncx2, norm, weibull_max,
                         weibull_min)


# In some tests, we'll use this optimizer for improved accuracy.
def optimizer(func, x0, args=(), disp=0):
    return fmin(func, x0, args=args, disp=disp, xtol=1e-12, ftol=1e-12)


def test_beta():
    """
    Test fitting beta shape parameters to interval-censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(0.10, 0.50, 0.75, 0.80),
    +                    right=c(0.20, 0.55, 0.90, 0.95))
    > result = fitdistcens(data, 'beta', control=list(reltol=1e-14))

    > result
    Fitting of the distribution ' beta ' on censored data by maximum likelihood
    Parameters:
           estimate
    shape1 1.419941
    shape2 1.027066
    > result$sd
       shape1    shape2
    0.9914177 0.6866565
    """
    data = CensoredData(interval=[[0.10, 0.20],
                                  [0.50, 0.55],
                                  [0.75, 0.90],
                                  [0.80, 0.95]])

    # For this test, fit only the shape parameters; loc and scale are fixed.
    a, b, loc, scale = beta.fit(data, floc=0, fscale=1, optimizer=optimizer)

    assert_allclose(a, 1.419941, rtol=5e-6)
    assert_allclose(b, 1.027066, rtol=5e-6)
    assert loc == 0
    assert scale == 1


def test_cauchy_right_censored():
    """
    Test fitting the Cauchy distribution to right-censored data.

    Calculation in R, with two values not censored [1, 10] and
    one right-censored value [30].

    > library(fitdistrplus)
    > data <- data.frame(left=c(1, 10, 30), right=c(1, 10, NA))
    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' cauchy ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    location 7.100001
    scale    7.455866
    """
    data = CensoredData(uncensored=[1, 10], right=[30])
    loc, scale = cauchy.fit(data, optimizer=optimizer)
    assert_allclose(loc, 7.10001, rtol=5e-6)
    assert_allclose(scale, 7.455866, rtol=5e-6)


def test_cauchy_mixed():
    """
    Test fitting the Cauchy distribution to data with mixed censoring.

    Calculation in R, with:
    * two values not censored [1, 10],
    * one left-censored [1],
    * one right-censored [30], and
    * one interval-censored [[4, 8]].

    > library(fitdistrplus)
    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))
    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' cauchy ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    location 4.605150
    scale    5.900852
    """
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30],
                        interval=[[4, 8]])
    loc, scale = cauchy.fit(data, optimizer=optimizer)
    assert_allclose(loc, 4.605150, rtol=5e-6)
    assert_allclose(scale, 5.900852, rtol=5e-6)


def test_chi2_mixed():
    """
    Test fitting just the shape parameter (df) of chi2 to mixed data.

    Calculation in R, with:
    * two values not censored [1, 10],
    * one left-censored [1],
    * one right-censored [30], and
    * one interval-censored [[4, 8]].

    > library(fitdistrplus)
    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))
    > result = fitdistcens(data, 'chisq', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' chisq ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    df 5.060329
    """
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30],
                        interval=[[4, 8]])
    df, loc, scale = chi2.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(df, 5.060329, rtol=5e-6)
    assert loc == 0
    assert scale == 1


def test_expon_right_censored():
    """
    For the exponential distribution with loc=0, the exact solution for
    fitting n uncensored points x[0]...x[n-1] and m right-censored points
    x[n]..x[n+m-1] is

        scale = sum(x)/n

    That is, divide the sum of all the values (not censored and
    right-censored) by the number of uncensored values.  (See, for example,
    https://en.wikipedia.org/wiki/Censoring_(statistics)#Likelihood.)

    The second derivative of the log-likelihood function is

        n/scale**2 - 2*sum(x)/scale**3

    from which the estimate of the standard error can be computed.

    -----

    Calculation in R, for reference only. The R results are not
    used in the test.

    > library(fitdistrplus)
    > dexps <- function(x, scale) {
    +     return(dexp(x, 1/scale))
    + }
    > pexps <- function(q, scale) {
    +     return(pexp(q, 1/scale))
    + }
    > left <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,
    +                                     16, 16, 20, 20, 21, 22)
    > right <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,
    +                                     NA, NA, NA, NA, NA, NA)
    > result = fitdistcens(data, 'exps', start=list(scale=mean(data$left)),
    +                      control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' exps ' on censored data by maximum likelihood
    Parameters:
          estimate
    scale    19.85
    > result$sd
       scale
    6.277119
    """
    # This data has 10 uncensored values and 6 right-censored values.
    obs = [1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15, 16, 16, 20, 20, 21, 22]
    cens = [False]*10 + [True]*6
    data = CensoredData.right_censored(obs, cens)

    loc, scale = expon.fit(data, floc=0, optimizer=optimizer)

    assert loc == 0
    # Use the analytical solution to compute the expected value.  This
    # is the sum of the observed values divided by the number of uncensored
    # values.
    n = len(data) - data.num_censored()
    total = data._uncensored.sum() + data._right.sum()
    expected = total / n
    assert_allclose(scale, expected, 1e-8)


def test_gamma_right_censored():
    """
    Fit gamma shape and scale to data with one right-censored value.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, 25.0),
    +                    right=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, NA))
    > result = fitdistcens(data, 'gamma', start=list(shape=1, scale=10),
    +                      control=list(reltol=1e-13))
    > result
    Fitting of the distribution ' gamma ' on censored data by maximum
      likelihood
    Parameters:
          estimate
    shape 1.447623
    scale 8.360197
    > result$sd
        shape     scale
    0.7053086 5.1016531
    """
    # The last value is right-censored.
    x = CensoredData.right_censored([2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0,
                                     25.0],
                                    [0]*7 + [1])

    a, loc, scale = gamma.fit(x, floc=0, optimizer=optimizer)

    assert_allclose(a, 1.447623, rtol=5e-6)
    assert loc == 0
    assert_allclose(scale, 8.360197, rtol=5e-6)


def test_gumbel():
    """
    Fit gumbel_l and gumbel_r to censored data.

    This R calculation should match gumbel_r.

    > library(evd)
    > libary(fitdistrplus)
    > data = data.frame(left=c(0, 2, 3, 9, 10, 10),
    +                   right=c(1, 2, 3, 9, NA, NA))
    > result = fitdistcens(data, 'gumbel',
    +                      control=list(reltol=1e-14),
    +                      start=list(loc=4, scale=5))
    > result
    Fitting of the distribution ' gumbel ' on censored data by maximum
    likelihood
    Parameters:
          estimate
    loc   4.487853
    scale 4.843640
    """
    # First value is interval-censored. Last two are right-censored.
    uncensored = np.array([2, 3, 9])
    right = np.array([10, 10])
    interval = np.array([[0, 1]])
    data = CensoredData(uncensored, right=right, interval=interval)
    loc, scale = gumbel_r.fit(data, optimizer=optimizer)
    assert_allclose(loc, 4.487853, rtol=5e-6)
    assert_allclose(scale, 4.843640, rtol=5e-6)

    # Negate the data and reverse the intervals, and test with gumbel_l.
    data2 = CensoredData(-uncensored, left=-right,
                         interval=-interval[:, ::-1])
    # Fitting gumbel_l to data2 should give the same result as above, but
    # with loc negated.
    loc2, scale2 = gumbel_l.fit(data2, optimizer=optimizer)
    assert_allclose(loc2, -4.487853, rtol=5e-6)
    assert_allclose(scale2, 4.843640, rtol=5e-6)


def test_invgauss():
    """
    Fit just the shape parameter of invgauss to data with one value
    left-censored and one value right-censored.

    Calculation in R; using a fixed dispersion parameter amounts to fixing
    the scale to be 1.

    > library(statmod)
    > library(fitdistrplus)
    > left <- c(NA, 0.4813096, 0.5571880, 0.5132463, 0.3801414, 0.5904386,
    +           0.4822340, 0.3478597, 3, 0.7191797, 1.5810902, 0.4442299)
    > right <- c(0.15, 0.4813096, 0.5571880, 0.5132463, 0.3801414, 0.5904386,
    +            0.4822340, 0.3478597, NA, 0.7191797, 1.5810902, 0.4442299)
    > data <- data.frame(left=left, right=right)
    > result = fitdistcens(data, 'invgauss', control=list(reltol=1e-12),
    +                      fix.arg=list(dispersion=1), start=list(mean=3))
    > result
    Fitting of the distribution ' invgauss ' on censored data by maximum
      likelihood
    Parameters:
         estimate
    mean 0.853469
    Fixed parameters:
               value
    dispersion     1
    > result$sd
        mean
    0.247636

    Here's the R calculation with the dispersion as a free parameter to
    be fit.

    > result = fitdistcens(data, 'invgauss', control=list(reltol=1e-12),
    +                      start=list(mean=3, dispersion=1))
    > result
    Fitting of the distribution ' invgauss ' on censored data by maximum
    likelihood
    Parameters:
                estimate
    mean       0.8699819
    dispersion 1.2261362

    The parametrization of the inverse Gaussian distribution in the
    `statmod` package is not the same as in SciPy (see
        https://arxiv.org/abs/1603.06687
    for details).  The translation from R to SciPy is

        scale = 1/dispersion
        mu    = mean * dispersion

    > 1/result$estimate['dispersion']  # 1/dispersion
    dispersion
     0.8155701
    > result$estimate['mean'] * result$estimate['dispersion']
        mean
    1.066716

    Those last two values are the SciPy scale and shape parameters.
    """
    # One point is left-censored, and one is right-censored.
    x = [0.4813096, 0.5571880, 0.5132463, 0.3801414,
         0.5904386, 0.4822340, 0.3478597, 0.7191797,
         1.5810902, 0.4442299]
    data = CensoredData(uncensored=x, left=[0.15], right=[3])

    # Fit only the shape parameter.
    mu, loc, scale = invgauss.fit(data, floc=0, fscale=1, optimizer=optimizer)

    assert_allclose(mu, 0.853469, rtol=5e-5)
    assert loc == 0
    assert scale == 1

    # Fit the shape and scale.
    mu, loc, scale = invgauss.fit(data, floc=0, optimizer=optimizer)

    assert_allclose(mu, 1.066716, rtol=5e-5)
    assert loc == 0
    assert_allclose(scale, 0.8155701, rtol=5e-5)


def test_invweibull():
    """
    Fit invweibull to censored data.

    Here is the calculation in R.  The 'frechet' distribution from the evd
    package matches SciPy's invweibull distribution.  The `loc` parameter
    is fixed at 0.

    > library(evd)
    > libary(fitdistrplus)
    > data = data.frame(left=c(0, 2, 3, 9, 10, 10),
    +                   right=c(1, 2, 3, 9, NA, NA))
    > result = fitdistcens(data, 'frechet',
    +                      control=list(reltol=1e-14),
    +                      start=list(loc=4, scale=5))
    > result
    Fitting of the distribution ' frechet ' on censored data by maximum
    likelihood
    Parameters:
           estimate
    scale 2.7902200
    shape 0.6379845
    Fixed parameters:
        value
    loc     0
    """
    # In the R data, the first value is interval-censored, and the last
    # two are right-censored.  The rest are not censored.
    data = CensoredData(uncensored=[2, 3, 9], right=[10, 10],
                        interval=[[0, 1]])
    c, loc, scale = invweibull.fit(data, floc=0, optimizer=optimizer)
    assert_allclose(c, 0.6379845, rtol=5e-6)
    assert loc == 0
    assert_allclose(scale, 2.7902200, rtol=5e-6)


def test_laplace():
    """
    Fir the Laplace distribution to left- and right-censored data.

    Calculation in R:

    > library(fitdistrplus)
    > dlaplace <- function(x, location=0, scale=1) {
    +     return(0.5*exp(-abs((x - location)/scale))/scale)
    + }
    > plaplace <- function(q, location=0, scale=1) {
    +     z <- (q - location)/scale
    +     s <- sign(z)
    +     f <- -s*0.5*exp(-abs(z)) + (s+1)/2
    +     return(f)
    + }
    > left <- c(NA, -41.564, 50.0, 15.7384, 50.0, 10.0452, -2.0684,
    +           -19.5399, 50.0,   9.0005, 27.1227, 4.3113, -3.7372,
    +           25.3111, 14.7987,  34.0887,  50.0, 42.8496, 18.5862,
    +           32.8921, 9.0448, -27.4591, NA, 19.5083, -9.7199)
    > right <- c(-50.0, -41.564,  NA, 15.7384, NA, 10.0452, -2.0684,
    +            -19.5399, NA, 9.0005, 27.1227, 4.3113, -3.7372,
    +            25.3111, 14.7987, 34.0887, NA,  42.8496, 18.5862,
    +            32.8921, 9.0448, -27.4591, -50.0, 19.5083, -9.7199)
    > data <- data.frame(left=left, right=right)
    > result <- fitdistcens(data, 'laplace', start=list(location=10, scale=10),
    +                       control=list(reltol=1e-13))
    > result
    Fitting of the distribution ' laplace ' on censored data by maximum
      likelihood
    Parameters:
             estimate
    location 14.79870
    scale    30.93601
    > result$sd
         location     scale
    0.1758864 7.0972125
    """
    # The value -50 is left-censored, and the value 50 is right-censored.
    obs = np.array([-50.0, -41.564, 50.0, 15.7384, 50.0, 10.0452, -2.0684,
                    -19.5399, 50.0, 9.0005, 27.1227, 4.3113, -3.7372,
                    25.3111, 14.7987, 34.0887, 50.0, 42.8496, 18.5862,
                    32.8921, 9.0448, -27.4591, -50.0, 19.5083, -9.7199])
    x = obs[(obs != -50.0) & (obs != 50)]
    left = obs[obs == -50.0]
    right = obs[obs == 50.0]
    data = CensoredData(uncensored=x, left=left, right=right)
    loc, scale = laplace.fit(data, loc=10, scale=10, optimizer=optimizer)
    assert_allclose(loc, 14.79870, rtol=5e-6)
    assert_allclose(scale, 30.93601, rtol=5e-6)


def test_logistic():
    """
    Fit the logistic distribution to left-censored data.

    Calculation in R:
    > library(fitdistrplus)
    > left = c(13.5401, 37.4235, 11.906 , 13.998 ,  NA    ,  0.4023,  NA    ,
    +          10.9044, 21.0629,  9.6985,  NA    , 12.9016, 39.164 , 34.6396,
    +          NA    , 20.3665, 16.5889, 18.0952, 45.3818, 35.3306,  8.4949,
    +          3.4041,  NA    ,  7.2828, 37.1265,  6.5969, 17.6868, 17.4977,
    +          16.3391, 36.0541)
    > right = c(13.5401, 37.4235, 11.906 , 13.998 ,  0.    ,  0.4023,  0.    ,
    +           10.9044, 21.0629,  9.6985,  0.    , 12.9016, 39.164 , 34.6396,
    +           0.    , 20.3665, 16.5889, 18.0952, 45.3818, 35.3306,  8.4949,
    +           3.4041,  0.    ,  7.2828, 37.1265,  6.5969, 17.6868, 17.4977,
    +           16.3391, 36.0541)
    > data = data.frame(left=left, right=right)
    > result = fitdistcens(data, 'logis', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' logis ' on censored data by maximum
      likelihood
    Parameters:
              estimate
    location 14.633459
    scale     9.232736
    > result$sd
    location    scale
    2.931505 1.546879
    """
    # Values that are zero are left-censored; the true values are less than 0.
    x = np.array([13.5401, 37.4235, 11.906, 13.998, 0.0, 0.4023, 0.0, 10.9044,
                  21.0629, 9.6985, 0.0, 12.9016, 39.164, 34.6396, 0.0, 20.3665,
                  16.5889, 18.0952, 45.3818, 35.3306, 8.4949, 3.4041, 0.0,
                  7.2828, 37.1265, 6.5969, 17.6868, 17.4977, 16.3391,
                  36.0541])
    data = CensoredData.left_censored(x, censored=(x == 0))
    loc, scale = logistic.fit(data, optimizer=optimizer)
    assert_allclose(loc, 14.633459, rtol=5e-7)
    assert_allclose(scale, 9.232736, rtol=5e-6)


def test_lognorm():
    """
    Ref: https://math.montana.edu/jobo/st528/documents/relc.pdf

    The data is the locomotive control time to failure example that starts
    on page 8.  That's the 8th page in the PDF; the page number shown in
    the text is 270).
    The document includes SAS output for the data.
    """
    # These are the uncensored measurements.  There are also 59 right-censored
    # measurements where the lower bound is 135.
    miles_to_fail = [22.5, 37.5, 46.0, 48.5, 51.5, 53.0, 54.5, 57.5, 66.5,
                     68.0, 69.5, 76.5, 77.0, 78.5, 80.0, 81.5, 82.0, 83.0,
                     84.0, 91.5, 93.5, 102.5, 107.0, 108.5, 112.5, 113.5,
                     116.0, 117.0, 118.5, 119.0, 120.0, 122.5, 123.0, 127.5,
                     131.0, 132.5, 134.0]

    data = CensoredData.right_censored(miles_to_fail + [135]*59,
                                       [0]*len(miles_to_fail) + [1]*59)
    sigma, loc, scale = lognorm.fit(data, floc=0)

    assert loc == 0
    # Convert the lognorm parameters to the mu and sigma of the underlying
    # normal distribution.
    mu = np.log(scale)
    # The expected results are from the 17th page of the PDF document
    # (labeled page 279), in the SAS output on the right side of the page.
    assert_allclose(mu, 5.1169, rtol=5e-4)
    assert_allclose(sigma, 0.7055, rtol=5e-3)


def test_nct():
    """
    Test fitting the noncentral t distribution to censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(1, 2, 3, 5, 8, 10, 25, 25),
    +                    right=c(1, 2, 3, 5, 8, 10, NA, NA))
    > result = fitdistcens(data, 't', control=list(reltol=1e-14),
    +                      start=list(df=1, ncp=2))
    > result
    Fitting of the distribution ' t ' on censored data by maximum likelihood
    Parameters:
         estimate
    df  0.5432336
    ncp 2.8893565

    """
    data = CensoredData.right_censored([1, 2, 3, 5, 8, 10, 25, 25],
                                       [0, 0, 0, 0, 0, 0, 1, 1])
    # Fit just the shape parameter df and nc; loc and scale are fixed.
    with np.errstate(over='ignore'):  # remove context when gh-14901 is closed
        df, nc, loc, scale = nct.fit(data, floc=0, fscale=1,
                                     optimizer=optimizer)
    assert_allclose(df, 0.5432336, rtol=5e-6)
    assert_allclose(nc, 2.8893565, rtol=5e-6)
    assert loc == 0
    assert scale == 1


def test_ncx2():
    """
    Test fitting the shape parameters (df, ncp) of ncx2 to mixed data.

    Calculation in R, with
    * 5 not censored values [2.7, 0.2, 6.5, 0.4, 0.1],
    * 1 interval-censored value [[0.6, 1.0]], and
    * 2 right-censored values [8, 8].

    > library(fitdistrplus)
    > data <- data.frame(left=c(2.7, 0.2, 6.5, 0.4, 0.1, 0.6, 8, 8),
    +                    right=c(2.7, 0.2, 6.5, 0.4, 0.1, 1.0, NA, NA))
    > result = fitdistcens(data, 'chisq', control=list(reltol=1e-14),
    +                      start=list(df=1, ncp=2))
    > result
    Fitting of the distribution ' chisq ' on censored data by maximum
    likelihood
    Parameters:
        estimate
    df  1.052871
    ncp 2.362934
    """
    data = CensoredData(uncensored=[2.7, 0.2, 6.5, 0.4, 0.1], right=[8, 8],
                        interval=[[0.6, 1.0]])
    with np.errstate(over='ignore'):  # remove context when gh-14901 is closed
        df, ncp, loc, scale = ncx2.fit(data, floc=0, fscale=1,
                                       optimizer=optimizer)
    assert_allclose(df, 1.052871, rtol=5e-6)
    assert_allclose(ncp, 2.362934, rtol=5e-6)
    assert loc == 0
    assert scale == 1


def test_norm():
    """
    Test fitting the normal distribution to interval-censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(0.10, 0.50, 0.75, 0.80),
    +                    right=c(0.20, 0.55, 0.90, 0.95))
    > result = fitdistcens(data, 'norm', control=list(reltol=1e-14))

    > result
    Fitting of the distribution ' norm ' on censored data by maximum likelihood
    Parameters:
          estimate
    mean 0.5919990
    sd   0.2868042
    > result$sd
         mean        sd
    0.1444432 0.1029451
    """
    data = CensoredData(interval=[[0.10, 0.20],
                                  [0.50, 0.55],
                                  [0.75, 0.90],
                                  [0.80, 0.95]])

    loc, scale = norm.fit(data, optimizer=optimizer)

    assert_allclose(loc, 0.5919990, rtol=5e-6)
    assert_allclose(scale, 0.2868042, rtol=5e-6)


def test_weibull_censored1():
    # Ref: http://www.ams.sunysb.edu/~zhu/ams588/Lecture_3_likelihood.pdf

    # Survival times; '*' indicates right-censored.
    s = "3,5,6*,8,10*,11*,15,20*,22,23,27*,29,32,35,40,26,28,33*,21,24*"

    times, cens = zip(*[(float(t[0]), len(t) == 2)
                        for t in [w.split('*') for w in s.split(',')]])
    data = CensoredData.right_censored(times, cens)

    c, loc, scale = weibull_min.fit(data, floc=0)

    # Expected values are from the reference.
    assert_allclose(c, 2.149, rtol=1e-3)
    assert loc == 0
    assert_allclose(scale, 28.99, rtol=1e-3)

    # Flip the sign of the data, and make the censored values
    # left-censored. We should get the same parameters when we fit
    # weibull_max to the flipped data.
    data2 = CensoredData.left_censored(-np.array(times), cens)

    c2, loc2, scale2 = weibull_max.fit(data2, floc=0)

    assert_allclose(c2, 2.149, rtol=1e-3)
    assert loc2 == 0
    assert_allclose(scale2, 28.99, rtol=1e-3)


def test_weibull_min_sas1():
    # Data and SAS results from
    #   https://support.sas.com/documentation/cdl/en/qcug/63922/HTML/default/
    #         viewer.htm#qcug_reliability_sect004.htm

    text = """
           450 0    460 1   1150 0   1150 0   1560 1
          1600 0   1660 1   1850 1   1850 1   1850 1
          1850 1   1850 1   2030 1   2030 1   2030 1
          2070 0   2070 0   2080 0   2200 1   3000 1
          3000 1   3000 1   3000 1   3100 0   3200 1
          3450 0   3750 1   3750 1   4150 1   4150 1
          4150 1   4150 1   4300 1   4300 1   4300 1
          4300 1   4600 0   4850 1   4850 1   4850 1
          4850 1   5000 1   5000 1   5000 1   6100 1
          6100 0   6100 1   6100 1   6300 1   6450 1
          6450 1   6700 1   7450 1   7800 1   7800 1
          8100 1   8100 1   8200 1   8500 1   8500 1
          8500 1   8750 1   8750 0   8750 1   9400 1
          9900 1  10100 1  10100 1  10100 1  11500 1
    """

    life, cens = np.array([int(w) for w in text.split()]).reshape(-1, 2).T
    life = life/1000.0

    data = CensoredData.right_censored(life, cens)

    c, loc, scale = weibull_min.fit(data, floc=0, optimizer=optimizer)
    assert_allclose(c, 1.0584, rtol=1e-4)
    assert_allclose(scale, 26.2968, rtol=1e-5)
    assert loc == 0


def test_weibull_min_sas2():
    # http://support.sas.com/documentation/cdl/en/ormpug/67517/HTML/default/
    #      viewer.htm#ormpug_nlpsolver_examples06.htm

    # The last two values are right-censored.
    days = np.array([143, 164, 188, 188, 190, 192, 206, 209, 213, 216, 220,
                     227, 230, 234, 246, 265, 304, 216, 244])

    data = CensoredData.right_censored(days, [0]*(len(days) - 2) + [1]*2)

    c, loc, scale = weibull_min.fit(data, 1, loc=100, scale=100,
                                    optimizer=optimizer)

    assert_allclose(c, 2.7112, rtol=5e-4)
    assert_allclose(loc, 122.03, rtol=5e-4)
    assert_allclose(scale, 108.37, rtol=5e-4)
