"""
.. _statsrefmanual:

==========================================
Statistical functions (:mod:`scipy.stats`)
==========================================

.. currentmodule:: scipy.stats

This module contains a large number of probability distributions,
summary and frequency statistics, correlation functions and statistical
tests, masked statistics, kernel density estimation, quasi-Monte Carlo
functionality, and more.

Statistics is a very large area, and there are topics that are out of scope
for SciPy and are covered by other packages. Some of the most important ones
are:

- `statsmodels <https://www.statsmodels.org/stable/index.html>`__:
  regression, linear models, time series analysis, extensions to topics
  also covered by ``scipy.stats``.
- `Pandas <https://pandas.pydata.org/>`__: tabular data, time series
  functionality, interfaces to other statistical languages.
- `PyMC <https://docs.pymc.io/>`__: Bayesian statistical
  modeling, probabilistic machine learning.
- `scikit-learn <https://scikit-learn.org/>`__: classification, regression,
  model selection.
- `Seaborn <https://seaborn.pydata.org/>`__: statistical data visualization.
- `rpy2 <https://rpy2.github.io/>`__: Python to R bridge.


Probability distributions
=========================

Each univariate distribution is an instance of a subclass of `rv_continuous`
(`rv_discrete` for discrete distributions):

.. autosummary::
   :toctree: generated/

   rv_continuous
   rv_discrete
   rv_histogram

Continuous distributions
------------------------

.. autosummary::
   :toctree: generated/

   alpha             -- Alpha
   anglit            -- Anglit
   arcsine           -- Arcsine
   argus             -- Argus
   beta              -- Beta
   betaprime         -- Beta Prime
   bradford          -- Bradford
   burr              -- Burr (Type III)
   burr12            -- Burr (Type XII)
   cauchy            -- Cauchy
   chi               -- Chi
   chi2              -- Chi-squared
   cosine            -- Cosine
   crystalball       -- Crystalball
   dgamma            -- Double Gamma
   dweibull          -- Double Weibull
   erlang            -- Erlang
   expon             -- Exponential
   exponnorm         -- Exponentially Modified Normal
   exponweib         -- Exponentiated Weibull
   exponpow          -- Exponential Power
   f                 -- F (Snecdor F)
   fatiguelife       -- Fatigue Life (Birnbaum-Saunders)
   fisk              -- Fisk
   foldcauchy        -- Folded Cauchy
   foldnorm          -- Folded Normal
   genlogistic       -- Generalized Logistic
   gennorm           -- Generalized normal
   genpareto         -- Generalized Pareto
   genexpon          -- Generalized Exponential
   genextreme        -- Generalized Extreme Value
   gausshyper        -- Gauss Hypergeometric
   gamma             -- Gamma
   gengamma          -- Generalized gamma
   genhalflogistic   -- Generalized Half Logistic
   genhyperbolic     -- Generalized Hyperbolic
   geninvgauss       -- Generalized Inverse Gaussian
   gibrat            -- Gibrat
   gompertz          -- Gompertz (Truncated Gumbel)
   gumbel_r          -- Right Sided Gumbel, Log-Weibull, Fisher-Tippett, Extreme Value Type I
   gumbel_l          -- Left Sided Gumbel, etc.
   halfcauchy        -- Half Cauchy
   halflogistic      -- Half Logistic
   halfnorm          -- Half Normal
   halfgennorm       -- Generalized Half Normal
   hypsecant         -- Hyperbolic Secant
   invgamma          -- Inverse Gamma
   invgauss          -- Inverse Gaussian
   invweibull        -- Inverse Weibull
   johnsonsb         -- Johnson SB
   johnsonsu         -- Johnson SU
   kappa4            -- Kappa 4 parameter
   kappa3            -- Kappa 3 parameter
   ksone             -- Distribution of Kolmogorov-Smirnov one-sided test statistic
   kstwo             -- Distribution of Kolmogorov-Smirnov two-sided test statistic
   kstwobign         -- Limiting Distribution of scaled Kolmogorov-Smirnov two-sided test statistic.
   laplace           -- Laplace
   laplace_asymmetric    -- Asymmetric Laplace
   levy              -- Levy
   levy_l
   levy_stable
   logistic          -- Logistic
   loggamma          -- Log-Gamma
   loglaplace        -- Log-Laplace (Log Double Exponential)
   lognorm           -- Log-Normal
   loguniform        -- Log-Uniform
   lomax             -- Lomax (Pareto of the second kind)
   maxwell           -- Maxwell
   mielke            -- Mielke's Beta-Kappa
   moyal             -- Moyal
   nakagami          -- Nakagami
   ncx2              -- Non-central chi-squared
   ncf               -- Non-central F
   nct               -- Non-central Student's T
   norm              -- Normal (Gaussian)
   norminvgauss      -- Normal Inverse Gaussian
   pareto            -- Pareto
   pearson3          -- Pearson type III
   powerlaw          -- Power-function
   powerlognorm      -- Power log normal
   powernorm         -- Power normal
   rdist             -- R-distribution
   rayleigh          -- Rayleigh
   rel_breitwigner   -- Relativistic Breit-Wigner
   rice              -- Rice
   recipinvgauss     -- Reciprocal Inverse Gaussian
   semicircular      -- Semicircular
   skewcauchy        -- Skew Cauchy
   skewnorm          -- Skew normal
   studentized_range    -- Studentized Range
   t                 -- Student's T
   trapezoid         -- Trapezoidal
   triang            -- Triangular
   truncexpon        -- Truncated Exponential
   truncnorm         -- Truncated Normal
   truncpareto       -- Truncated Pareto
   truncweibull_min  -- Truncated minimum Weibull distribution
   tukeylambda       -- Tukey-Lambda
   uniform           -- Uniform
   vonmises          -- Von-Mises (Circular)
   vonmises_line     -- Von-Mises (Line)
   wald              -- Wald
   weibull_min       -- Minimum Weibull (see Frechet)
   weibull_max       -- Maximum Weibull (see Frechet)
   wrapcauchy        -- Wrapped Cauchy

The ``fit`` method of the univariate continuous distributions uses
maximum likelihood estimation to fit the distribution to a data set.
The ``fit`` method can accept regular data or *censored data*.
Censored data is represented with instances of the `CensoredData`
class.

.. autosummary::
   :toctree: generated/

   CensoredData


Multivariate distributions
--------------------------

.. autosummary::
   :toctree: generated/

   multivariate_normal    -- Multivariate normal distribution
   matrix_normal          -- Matrix normal distribution
   dirichlet              -- Dirichlet
   dirichlet_multinomial  -- Dirichlet multinomial distribution
   wishart                -- Wishart
   invwishart             -- Inverse Wishart
   multinomial            -- Multinomial distribution
   special_ortho_group    -- SO(N) group
   ortho_group            -- O(N) group
   unitary_group          -- U(N) group
   random_correlation     -- random correlation matrices
   multivariate_t         -- Multivariate t-distribution
   multivariate_hypergeom -- Multivariate hypergeometric distribution
   random_table           -- Distribution of random tables with given marginals
   uniform_direction      -- Uniform distribution on S(N-1)
   vonmises_fisher        -- Von Mises-Fisher distribution

`scipy.stats.multivariate_normal` methods accept instances
of the following class to represent the covariance.

.. autosummary::
   :toctree: generated/

   Covariance             -- Representation of a covariance matrix


Discrete distributions
----------------------

.. autosummary::
   :toctree: generated/

   bernoulli                -- Bernoulli
   betabinom                -- Beta-Binomial
   binom                    -- Binomial
   boltzmann                -- Boltzmann (Truncated Discrete Exponential)
   dlaplace                 -- Discrete Laplacian
   geom                     -- Geometric
   hypergeom                -- Hypergeometric
   logser                   -- Logarithmic (Log-Series, Series)
   nbinom                   -- Negative Binomial
   nchypergeom_fisher       -- Fisher's Noncentral Hypergeometric
   nchypergeom_wallenius    -- Wallenius's Noncentral Hypergeometric
   nhypergeom               -- Negative Hypergeometric
   planck                   -- Planck (Discrete Exponential)
   poisson                  -- Poisson
   randint                  -- Discrete Uniform
   skellam                  -- Skellam
   yulesimon                -- Yule-Simon
   zipf                     -- Zipf (Zeta)
   zipfian                  -- Zipfian


An overview of statistical functions is given below.  Many of these functions
have a similar version in `scipy.stats.mstats` which work for masked arrays.

Summary statistics
==================

.. autosummary::
   :toctree: generated/

   describe          -- Descriptive statistics
   gmean             -- Geometric mean
   hmean             -- Harmonic mean
   pmean             -- Power mean
   kurtosis          -- Fisher or Pearson kurtosis
   mode              -- Modal value
   moment            -- Central moment
   expectile         -- Expectile
   skew              -- Skewness
   kstat             --
   kstatvar          --
   tmean             -- Truncated arithmetic mean
   tvar              -- Truncated variance
   tmin              --
   tmax              --
   tstd              --
   tsem              --
   variation         -- Coefficient of variation
   find_repeats
   rankdata
   tiecorrect
   trim_mean
   gstd              -- Geometric Standard Deviation
   iqr
   sem
   bayes_mvs
   mvsdist
   entropy
   differential_entropy
   median_abs_deviation

Frequency statistics
====================

.. autosummary::
   :toctree: generated/

   cumfreq
   percentileofscore
   scoreatpercentile
   relfreq

.. autosummary::
   :toctree: generated/

   binned_statistic     -- Compute a binned statistic for a set of data.
   binned_statistic_2d  -- Compute a 2-D binned statistic for a set of data.
   binned_statistic_dd  -- Compute a d-D binned statistic for a set of data.

Hypothesis Tests and related functions
======================================
SciPy has many functions for performing hypothesis tests that return a
test statistic and a p-value, and several of them return confidence intervals
and/or other related information.

The headings below are based on common uses of the functions within, but due to
the wide variety of statistical procedures, any attempt at coarse-grained
categorization will be imperfect. Also, note that tests within the same heading
are not interchangeable in general (e.g. many have different distributional
assumptions).

One Sample Tests / Paired Sample Tests
--------------------------------------
One sample tests are typically used to assess whether a single sample was
drawn from a specified distribution or a distribution with specified properties
(e.g. zero mean).

.. autosummary::
   :toctree: generated/

   ttest_1samp
   binomtest
   skewtest
   kurtosistest
   normaltest
   jarque_bera
   shapiro
   anderson
   cramervonmises
   ks_1samp
   goodness_of_fit
   chisquare
   power_divergence

Paired sample tests are often used to assess whether two samples were drawn
from the same distribution; they differ from the independent sample tests below
in that each observation in one sample is treated as paired with a
closely-related observation in the other sample (e.g. when environmental
factors are controlled between observations within a pair but not among pairs).
They can also be interpreted or used as one-sample tests (e.g. tests on the
mean or median of *differences* between paired observations).

.. autosummary::
   :toctree: generated/

   ttest_rel
   wilcoxon

Association/Correlation Tests
-----------------------------

These tests are often used to assess whether there is a relationship (e.g.
linear) between paired observations in multiple samples or among the
coordinates of multivariate observations.

.. autosummary::
   :toctree: generated/

   linregress
   pearsonr
   spearmanr
   pointbiserialr
   kendalltau
   weightedtau
   somersd
   siegelslopes
   theilslopes
   page_trend_test
   multiscale_graphcorr

These association tests and are to work with samples in the form of contingency
tables. Supporting functions are available in `scipy.stats.contingency`.

.. autosummary::
   :toctree: generated/

   chi2_contingency
   fisher_exact
   barnard_exact
   boschloo_exact

Independent Sample Tests
------------------------
Independent sample tests are typically used to assess whether multiple samples
were independently drawn from the same distribution or different distributions
with a shared property (e.g. equal means).

Some tests are specifically for comparing two samples.

.. autosummary::
   :toctree: generated/

   ttest_ind_from_stats
   poisson_means_test
   ttest_ind
   mannwhitneyu
   ranksums
   brunnermunzel
   mood
   ansari
   cramervonmises_2samp
   epps_singleton_2samp
   ks_2samp
   kstest

Others are generalized to multiple samples.

.. autosummary::
   :toctree: generated/

   f_oneway
   tukey_hsd
   dunnett
   kruskal
   alexandergovern
   fligner
   levene
   bartlett
   median_test
   friedmanchisquare
   anderson_ksamp

Resampling and Monte Carlo Methods
----------------------------------
The following functions can reproduce the p-value and confidence interval
results of most of the functions above, and often produce accurate results in a
wider variety of conditions. They can also be used to perform hypothesis tests
and generate confidence intervals for custom statistics. This flexibility comes
at the cost of greater computational requirements and stochastic results.

.. autosummary::
   :toctree: generated/

   monte_carlo_test
   permutation_test
   bootstrap

Instances of the following object can be passed into some hypothesis test
functions to perform a resampling or Monte Carlo version of the hypothesis
test.

.. autosummary::
   :toctree: generated/

   MonteCarloMethod
   PermutationMethod
   BootstrapMethod

Multiple Hypothesis Testing and Meta-Analysis
---------------------------------------------
These functions are for assessing the results of individual tests as a whole.
Functions for performing specific multiple hypothesis tests (e.g. post hoc
tests) are listed above.

.. autosummary::
   :toctree: generated/

   combine_pvalues
   false_discovery_control

Deprecated and Legacy Functions
-------------------------------

.. autosummary::
   :toctree: generated/

   binom_test

The following functions are related to the tests above but do not belong in the
above categories.

Quasi-Monte Carlo
=================

.. toctree::
   :maxdepth: 4

   stats.qmc

Contingency Tables
==================

.. toctree::
   :maxdepth: 4

   stats.contingency

Masked statistics functions
===========================

.. toctree::

   stats.mstats


Other statistical functionality
===============================

Transformations
---------------

.. autosummary::
   :toctree: generated/

   boxcox
   boxcox_normmax
   boxcox_llf
   yeojohnson
   yeojohnson_normmax
   yeojohnson_llf
   obrientransform
   sigmaclip
   trimboth
   trim1
   zmap
   zscore
   gzscore

Statistical distances
---------------------

.. autosummary::
   :toctree: generated/

   wasserstein_distance
   energy_distance

Sampling
--------

.. toctree::
   :maxdepth: 4

   stats.sampling

Random variate generation / CDF Inversion
-----------------------------------------

.. autosummary::
   :toctree: generated/

   rvs_ratio_uniforms

Fitting / Survival Analysis
---------------------------

.. autosummary::
   :toctree: generated/

   fit
   ecdf
   logrank

Directional statistical functions
---------------------------------

.. autosummary::
   :toctree: generated/

   directional_stats
   circmean
   circvar
   circstd

Sensitivity Analysis
--------------------

.. autosummary::
   :toctree: generated/

   sobol_indices

Plot-tests
----------

.. autosummary::
   :toctree: generated/

   ppcc_max
   ppcc_plot
   probplot
   boxcox_normplot
   yeojohnson_normplot

Univariate and multivariate kernel density estimation
-----------------------------------------------------

.. autosummary::
   :toctree: generated/

   gaussian_kde

Warnings / Errors used in :mod:`scipy.stats`
--------------------------------------------

.. autosummary::
   :toctree: generated/

   DegenerateDataWarning
   ConstantInputWarning
   NearConstantInputWarning
   FitError

Result classes used in :mod:`scipy.stats`
-----------------------------------------

.. warning::

    These classes are private, but they are included here because instances
    of them are returned by other statistical functions. User import and
    instantiation is not supported.

.. toctree::
   :maxdepth: 2

   stats._result_classes

"""

from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
                               DegenerateDataWarning, FitError)
from ._stats_py import *
from ._variation import variation
from .distributions import *
from ._morestats import *
from ._multicomp import *
from ._binomtest import binomtest
from ._binned_statistic import *
from ._kde import gaussian_kde
from . import mstats
from . import qmc
from ._multivariate import *
from . import contingency
from .contingency import chi2_contingency
from ._censored_data import CensoredData  # noqa
from ._resampling import (bootstrap, monte_carlo_test, permutation_test,
                          MonteCarloMethod, PermutationMethod, BootstrapMethod)
from ._entropy import *
from ._hypotests import *
from ._rvs_sampling import rvs_ratio_uniforms
from ._page_trend_test import page_trend_test
from ._mannwhitneyu import mannwhitneyu
from ._fit import fit, goodness_of_fit
from ._covariance import Covariance
from ._sensitivity_analysis import *
from ._survival import *

# Deprecated namespaces, to be removed in v2.0.0
from . import (
    biasedurn, kde, morestats, mstats_basic, mstats_extras, mvn, statlib, stats
)


__all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
