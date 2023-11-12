# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:17:58 2020

Author: Josef Perktold
License: BSD-3

"""

import warnings

import numpy as np

from statsmodels.tools.decorators import cache_readonly

from statsmodels.stats.diagnostic_gen import (
    test_chisquare_binning
    )
from statsmodels.discrete._diagnostics_count import (
    test_poisson_dispersion,
    # _test_poisson_dispersion_generic,
    test_poisson_zeroinflation_jh,
    test_poisson_zeroinflation_broek,
    test_poisson_zeros,
    test_chisquare_prob,
    plot_probs
    )


class CountDiagnostic:
    """Diagnostic and specification tests and plots for Count model

    status: experimental

    Parameters
    ----------
    results : Results instance of a count model.
    y_max : int
        Largest count to include when computing predicted probabilities for
        counts. Default is the largest observed count.

    """

    def __init__(self, results, y_max=None):
        self.results = results
        self.y_max = y_max

    @cache_readonly
    def probs_predicted(self):
        if self.y_max is not None:
            kwds = {"y_values": np.arange(self.y_max + 1)}
        else:
            kwds = {}
        return self.results.predict(which="prob", **kwds)

    def test_chisquare_prob(self, bin_edges=None, method=None):
        """Moment test for binned probabilites using OPG.

        Paramters
        ---------
        binedges : array_like or None
            This defines which counts are included in the test on frequencies
            and how counts are combined in bins.
            The default if bin_edges is None will change in future.
            See Notes and Example sections below.
        method : str
            Currently only `method = "opg"` is available.
            If method is None, the OPG will be used, but the default might
            change in future versions.
            See Notes section below.

        Returns
        -------
        test result

        Notes
        -----
        Warning: The current default can have many empty or nearly empty bins.
        The default number of bins is given by max(endog).
        Currently it is recommended to limit the number of bins explicitly,
        see Examples below.
        Binning will change in future and automatic binning will be added.

        Currently only the outer product of gradient, OPG, method is
        implemented. In many case, the OPG version of a specification test
        overrejects in small samples.
        Specialized tests that use observed or expected information matrix
        often have better small sample properties.
        The default method will change if better methods are added.

        Examples
        --------
        The following call is a test for the probability of zeros
        `test_chisquare_prob(bin_edges=np.arange(3))`

        `test_chisquare_prob(bin_edges=np.arange(10))` tests the hypothesis
        that the frequencies for counts up to 7 correspond to the estimated
        Poisson distributions.
        In this case, edges are 0, ..., 9 which defines 9 bins for
        counts 0 to 8. The last bin is dropped, so the joint test hypothesis is
        that the observed aggregated frequencies for counts 0 to 7 correspond
        to the model prediction for those frequencies. Predicted probabilites
        Prob(y_i = k | x) are aggregated over observations ``i``.

        """
        kwds = {}
        if bin_edges is not None:
            # TODO: verify upper bound, we drop last bin (may be open, inf)
            kwds["y_values"] = np.arange(bin_edges[-2] + 1)
        probs = self.results.predict(which="prob", **kwds)
        res = test_chisquare_prob(self.results, probs, bin_edges=bin_edges,
                                  method=method)
        return res

    def plot_probs(self, label='predicted', upp_xlim=None,
                   fig=None):
        """Plot observed versus predicted frequencies for entire sample.
        """
        probs_predicted = self.probs_predicted.sum(0)
        k_probs = len(probs_predicted)
        freq = np.bincount(self.results.model.endog.astype(int),
                           minlength=k_probs)[:k_probs]
        fig = plot_probs(freq, probs_predicted,
                         label=label, upp_xlim=upp_xlim,
                         fig=fig)
        return fig


class PoissonDiagnostic(CountDiagnostic):
    """Diagnostic and specification tests and plots for Poisson model

    status: experimental

    Parameters
    ----------
    results : PoissonResults instance

    """

    def _init__(self, results):
        self.results = results

    def test_dispersion(self):
        """Test for excess (over or under) dispersion in Poisson.

        Returns
        -------
        dispersion results
        """
        res = test_poisson_dispersion(self.results)
        return res

    def test_poisson_zeroinflation(self, method="prob", exog_infl=None):
        """Test for excess zeros, zero inflation or deflation.

        Parameters
        ----------
        method : str
            Three methods ara available for the test:

             - "prob" : moment test for the probability of zeros
             - "broek" : score test against zero inflation with or without
                explanatory variables for inflation

        exog_infl : array_like or None
            Optional explanatory variables under the alternative of zero
            inflation, or deflation. Only used if method is "broek".

        Returns
        -------
        results

        Notes
        -----
        If method = "prob", then the moment test of He et al 1_ is used based
        on the explicit formula in Tang and Tang 2_.

        If method = "broek" and exog_infl is None, then the test by Van den
        Broek 3_ is used. This is a score test against and alternative of
        constant zero inflation or deflation.

        If method = "broek" and exog_infl is provided, then the extension of
        the broek test to varying zero inflation or deflation by Jansakul and
        Hinde is used.

        Warning: The Broek and the Jansakul and Hinde tests are not numerically
        stable when the probability of zeros in Poisson is small, i.e. if the
        conditional means of the estimated Poisson distribution are large.
        In these cases, p-values will not be accurate.
        """
        if method == "prob":
            if exog_infl is not None:
                warnings.warn('exog_infl is only used if method = "broek"')
            res = test_poisson_zeros(self.results)
        elif method == "broek":
            if exog_infl is None:
                res = test_poisson_zeroinflation_broek(self.results)
            else:
                exog_infl = np.asarray(exog_infl)
                if exog_infl.ndim == 1:
                    exog_infl = exog_infl[:, None]
                res = test_poisson_zeroinflation_jh(self.results,
                                                    exog_infl=exog_infl)

        return res

    def _chisquare_binned(self, sort_var=None, bins=10, k_max=None, df=None,
                          sort_method="quicksort", frac_upp=0.1,
                          alpha_nc=0.05):
        """Hosmer-Lemeshow style test for count data.

        Note, this does not take into account that parameters are estimated.
        The distribution of the test statistic is only an approximation.

        This corresponds to the Hosmer-Lemeshow type test for an ordinal
        response variable. The outcome space y = k is partitioned into bins
        and treated as ordinal variable.
        The observations are split into approximately equal sized groups
        of observations sorted according the ``sort_var``.

        """

        if sort_var is None:
            sort_var = self.results.predict(which="lin")

        endog = self.results.model.endog
        # not sure yet how this is supposed to work
        # max_count = endog.max * 2
        # no option for max count in predict
        # counts = (endog == np.arange(max_count)).astype(int)
        expected = self.results.predict(which="prob")
        counts = (endog[:, None] == np.arange(expected.shape[1])).astype(int)

        # truncate upper tail
        if k_max is None:
            nobs = len(endog)
            icumcounts_sum = nobs - counts.sum(0).cumsum(0)
            k_max = np.argmax(icumcounts_sum < nobs * frac_upp) - 1
        expected = expected[:, :k_max]
        counts = counts[:, :k_max]
        # we should correct for or include truncated upper bin
        # inplace modification, we cannot reuse expected and counts anymore
        expected[:, -1] += 1 - expected.sum(1)
        counts[:, -1] += 1 - counts.sum(1)

        # TODO: what's the correct df, same as for multinomial/ordered ?
        res = test_chisquare_binning(counts, expected, sort_var=sort_var,
                                     bins=bins, df=df, ordered=True,
                                     sort_method=sort_method,
                                     alpha_nc=alpha_nc)
        return res
