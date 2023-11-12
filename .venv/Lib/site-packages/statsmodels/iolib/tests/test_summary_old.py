import warnings

import pytest


@pytest.mark.xfail(strict=True)
def test_regression_summary():
    #little luck getting this test to pass (It should?), can be used for
    #visual testing of the regression.summary table
    #fixed, might fail at minute changes
    from statsmodels.regression.tests.test_regression import TestOLS
    #from test_regression import TestOLS
    import time
    from string import Template
    t = time.localtime()
    desired = Template(
'''     Summary of Regression Results
=======================================
| Dependent Variable:                y|
| Model:                           OLS|
| Method:                Least Squares|
| Date:               $XXcurrentXdateXX|
| Time:                       $XXtimeXXX|
| # obs:                          16.0|
| Df residuals:                    9.0|
| Df model:                        6.0|
==============================================================================
|                   coefficient     std. error    t-statistic          prob. |
------------------------------------------------------------------------------
| x1                      15.06          84.91         0.1774         0.8631 |
| x2                   -0.03582        0.03349        -1.0695         0.3127 |
| x3                     -2.020         0.4884        -4.1364         0.0025 |
| x4                     -1.033         0.2143        -4.8220         0.0009 |
| x5                   -0.05110         0.2261        -0.2261         0.8262 |
| x6                      1829.          455.5         4.0159         0.0030 |
| const              -3.482e+06      8.904e+05        -3.9108         0.0036 |
==============================================================================
|                          Models stats                      Residual stats  |
------------------------------------------------------------------------------
| R-squared:                     0.9955   Durbin-Watson:              2.559  |
| Adjusted R-squared:            0.9925   Omnibus:                   0.7486  |
| F-statistic:                    330.3   Prob(Omnibus):             0.6878  |
| Prob (F-statistic):         4.984e-10   JB:                        0.6841  |
| Log likelihood:                -109.6   Prob(JB):                  0.7103  |
| AIC criterion:                  233.2   Skew:                      0.4200  |
| BIC criterion:                  238.6   Kurtosis:                   2.434  |
------------------------------------------------------------------------------'''
).substitute(XXcurrentXdateXX = str(time.strftime("%a, %d %b %Y",t)),
             XXtimeXXX = str(time.strftime("%H:%M:%S",t)))
    desired = str(desired)
    aregression = TestOLS()
    TestOLS.setup_class()
    results = aregression.res1
    # be quiet!
    original_filters = warnings.filters[:] # copy original
    warnings.simplefilter("ignore")
    try:
        r_summary = str(results.summary_old())
    finally:
        warnings.filters = original_filters # restore filters

##    print('###')
##    print(r_summary)
##    print('###')
##    print(desired)
##    print('###')
    actual = r_summary
    import numpy as np
    actual = '\n'.join((line.rstrip() for line in actual.split('\n')))
#    print len(actual), len(desired)
#    print repr(actual)
#    print repr(desired)
#    counter = 0
#    for c1,c2 in zip(actual, desired):
#        if not c1==c2 and counter<20:
#            print c1,c2
#            counter += 1
    np.testing.assert_(actual == desired)
