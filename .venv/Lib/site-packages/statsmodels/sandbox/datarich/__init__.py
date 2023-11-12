'''

Econometrics for a Datarich Environment
=======================================

Introduction
------------
In many cases we are performing statistical analysis when many observed variables are
available, when we are in a data rich environment. Machine learning has a wide variety
of tools for dimension reduction and penalization when there are many varibles compared
to the number of observation. Chemometrics has a long tradition of using Partial Least
Squares, NIPALS and similar in these cases. In econometrics the same problem shows up
when there are either many possible regressors, many (weak) instruments or when there are
a large number of moment conditions in GMM.

This section is intended to collect some models and tools in this area that are relevant
for the statical analysis and econometrics.

Covariance Matrices
===================
Several methods are available to reduce the small sample noise in estimated covariance
matrices with many variable.
Some applications:
weighting matrix with many moments,
covariance matrix for portfolio choice

Dimension Reduction
===================
Principal Component and Partial Least Squares try to extract the important low dimensional
factors from the data with many variables.

Regression with many regressors
===============================
Factor models, selection of regressors and shrinkage and penalization are used to improve
the statistical properties, when the presence of too many regressors leads to over-fitting
and too noisy small sample estimators and statistics.

Regression with many moments or many instruments
================================================
The same tools apply and can be used in these two cases.
e.g. Tychonov regularization of weighting matrix in GMM, similar to Ridge regression, the
weighting matrix can be shrunk towards the identity matrix.
Simplest case will be part of GMM. I do not know how much will be standalone
functions.


Intended Content
================

PLS
---
what should be available in class?

Factormodel and supporting helper functions
-------------------------------------------

PCA based
~~~~~~~~~
First version based PCA on Stock/Watson and Bai/Ng, and recent papers on the
selection of the number of factors. Not sure about Forni et al. in approach.
Basic support of this needs additional results for PCA, error covariance matrix
of data on reduced factors, required for criteria in Bai/Ng.
Selection criteria based on eigenvalue cutoffs.

Paper on PCA and structural breaks. Could add additional results during
find_nfact to test for parameter stability. I have not read the paper yet.

Idea: for forecasting, use up to h-step ahead endogenous variables to directly
get the forecasts.

Asymptotic results and distribution: not too much idea yet.
Standard OLS results are conditional on factors, paper by Haerdle (abstract
seems to suggest that this is ok, Park 2009).

Simulation: add function to simulate DGP of Bai/Ng and recent extension.
Sensitivity of selection criteria to heteroscedasticity and autocorrelation.

Bai, J. & Ng, S., 2002. Determining the Number of Factors in
    Approximate Factor Models. Econometrica, 70(1), pp.191-221.

Kapetanios, G., 2010. A Testing Procedure for Determining the Number
    of Factors in Approximate Factor Models With Large Datasets. Journal
    of Business and Economic Statistics, 28(3), pp.397-409.

Onatski, A., 2010. Determining the Number of Factors from Empirical
    Distribution of Eigenvalues. Review of Economics and Statistics,
    92(4), pp.1004-1016.

Alessi, L., Barigozzi, M. & Capasso, M., 2010. Improved penalization
    for determining the number of factors in approximate factor models.
    Statistics & Probability Letters, 80(23-24), pp.1806-1813.

Breitung, J. & Eickmeier, S., Testing for structural breaks in dynamic
    factor models. Journal of Econometrics, In Press, Accepted Manuscript.
    Available at:
    http://www.sciencedirect.com/science/article/B6VC0-51G3W92-1/2/f45ce2332443374fd770e42e5a68ddb4
    [Accessed November 15, 2010].

Croux, C., Renault, E. & Werker, B., 2004. Dynamic factor models.
    Journal of Econometrics, 119(2), pp.223-230.

Forni, M. et al., 2009. Opening the Black Box: Structural Factor
    Models with Large Cross Sections. Econometric Theory, 25(05),
    pp.1319-1347.

Forni, M. et al., 2000. The Generalized Dynamic-Factor Model:
    Identification and Estimation. Review of Economics and Statistics,
    82(4), pp.540-554.

Forni, M. & Lippi, M., The general dynamic factor model: One-sided
    representation results. Journal of Econometrics, In Press, Accepted
    Manuscript. Available at:
    http://www.sciencedirect.com/science/article/B6VC0-51FNPJN-1/2/4fcdd0cfb66e3050ff5d19bf2752ed19
    [Accessed November 15, 2010].

Kapetanios, G., 2010. A Testing Procedure for Determining the Number
    of Factors in Approximate Factor Models With Large Datasets. Journal
    of Business and Economic Statistics, 28(3), pp.397-409.

Onatski, A., 2010. Determining the Number of Factors from Empirical
    Distribution of Eigenvalues. Review of Economics and Statistics,
    92(4), pp.1004-1016.

Park, B.U. et al., 2009. Time Series Modelling With Semiparametric
    Factor Dynamics. Journal of the American Statistical Association,
    104(485), pp.284-298.



other factor algorithm
~~~~~~~~~~~~~~~~~~~~~~
PLS should fit in reasonably well.

Bai/Ng have a recent paper, where they compare LASSO, PCA, and similar, individual
and in combination.
Check how much we can use scikits.learn for this.


miscellaneous
~~~~~~~~~~~~~
Time series modeling of factors for prediction, ARMA, VARMA.
SUR and correlation structure
What about sandwich estimation, robust covariance matrices?
Similarity to Factor-Garch and Go-Garch
Updating: incremental PCA, ...?


TODO next
=========
MVOLS : OLS with multivariate endogenous and identical exogenous variables.
    rewrite and expand current varma_process.VAR
PCA : write a class after all, and/or adjust the current donated class
    and keep adding required statistics, e.g.
    residual variance, projection of X on k-factors, ... updating ?
FactorModelUnivariate : started, does basic principal component regression,
    based on standard information criteria, not Bai/Ng adjusted
FactorModelMultivariate : follow pattern for univariate version and use
    MVOLS






'''
