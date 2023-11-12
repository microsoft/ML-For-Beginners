
""". regress totemp gnpdefl gnp unemp armed pop year

      Source |       SS       df       MS              Number of obs =      16
-------------+------------------------------           F(  6,     9) =  330.29
       Model |   184172402     6  30695400.3           Prob > F      =  0.0000
    Residual |  836424.129     9  92936.0144           R-squared     =  0.9955
-------------+------------------------------           Adj R-squared =  0.9925
       Total |   185008826    15  12333921.7           Root MSE      =  304.85

------------------------------------------------------------------------------
      totemp |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
     gnpdefl |   15.06167   84.91486     0.18   0.863    -177.0291    207.1524
         gnp |  -.0358191    .033491    -1.07   0.313     -.111581    .0399428
       unemp |  -2.020229   .4883995    -4.14   0.003    -3.125065   -.9153928
       armed |  -1.033227   .2142741    -4.82   0.001    -1.517948   -.5485049
         pop |  -.0511045   .2260731    -0.23   0.826    -.5625173    .4603083
        year |   1829.151   455.4785     4.02   0.003     798.7873    2859.515
       _cons |   -3482258   890420.3    -3.91   0.004     -5496529    -1467987
------------------------------------------------------------------------------
"""


#From Stata using Longley dataset as in the test and example for GLM
"""
. glm totemp gnpdefl gnp unemp armed pop year

Iteration 0:   log likelihood = -109.61744

Generalized linear models                          No. of obs      =        16
Optimization     : ML                              Residual df     =         9
                                                   Scale parameter =  92936.01
Deviance         =  836424.1293                    (1/df) Deviance =  92936.01
Pearson          =  836424.1293                    (1/df) Pearson  =  92936.01

Variance function: V(u) = 1                        [Gaussian]
Link function    : g(u) = u                        [Identity]

                                                   AIC             =  14.57718
Log likelihood   = -109.6174355                    BIC             =  836399.2

------------------------------------------------------------------------------
             |                 OIM
      totemp |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
     gnpdefl |   15.06167   84.91486     0.18   0.859    -151.3684    181.4917
         gnp |  -.0358191    .033491    -1.07   0.285    -.1014603     .029822
       unemp |  -2.020229   .4883995    -4.14   0.000    -2.977475   -1.062984
       armed |  -1.033227   .2142741    -4.82   0.000    -1.453196   -.6132571
         pop |  -.0511045   .2260731    -0.23   0.821    -.4941996    .3919906
        year |   1829.151   455.4785     4.02   0.000     936.4298    2721.873
       _cons |   -3482258   890420.3    -3.91   0.000     -5227450    -1737066
------------------------------------------------------------------------------
"""

#RLM Example

"""
. rreg stackloss airflow watertemp acidconc

   Huber iteration 1:  maximum difference in weights = .48402478
   Huber iteration 2:  maximum difference in weights = .07083248
   Huber iteration 3:  maximum difference in weights = .03630349
Biweight iteration 4:  maximum difference in weights = .2114744
Biweight iteration 5:  maximum difference in weights = .04709559
Biweight iteration 6:  maximum difference in weights = .01648123
Biweight iteration 7:  maximum difference in weights = .01050023
Biweight iteration 8:  maximum difference in weights = .0027233

Robust regression                                      Number of obs =      21
                                                       F(  3,    17) =   74.15
                                                       Prob > F      =  0.0000

------------------------------------------------------------------------------
   stackloss |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
     airflow |   .8526511   .1223835     6.97   0.000     .5944446    1.110858
   watertemp |   .8733594   .3339811     2.61   0.018     .1687209    1.577998
    acidconc |  -.1224349   .1418364    -0.86   0.400    -.4216836    .1768139
       _cons |   -41.6703   10.79559    -3.86   0.001      -64.447   -18.89361
------------------------------------------------------------------------------

"""
