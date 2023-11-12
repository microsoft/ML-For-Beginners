import numpy as np

from statsmodels.iolib.table import SimpleTable


class HypothesisTestResults:
    """
    Results class for hypothesis tests.

    Parameters
    ----------
    test_statistic : float
    crit_value : float
    pvalue : float, 0 <= `pvalue` <= 1
    df : int
        Degrees of freedom.
    signif : float, 0 < `signif` < 1
        Significance level.
    method : str
        The kind of test (e.g. ``"f"`` for F-test, ``"wald"`` for Wald-test).
    title : str
        A title describing the test. It will be part of the summary.
    h0 : str
        A string describing the null hypothesis. It will be used in the
        summary.
    """
    def __init__(self, test_statistic, crit_value, pvalue, df,
                 signif, method, title, h0):
        self.test_statistic = test_statistic
        self.crit_value = crit_value
        self.pvalue = pvalue
        self.df = df
        self.signif = signif
        self.method = method.capitalize()
        if test_statistic < crit_value:
            self.conclusion = "fail to reject"
        else:
            self.conclusion = "reject"
        self.title = title
        self.h0 = h0
        self.conclusion_str = "Conclusion: %s H_0" % self.conclusion
        self.signif_str = " at {:.0%} significance level".format(self.signif)

    def summary(self):
        """Return summary"""
        title = self.title + ". " + self.h0 + ". " \
                                  + self.conclusion_str + self.signif_str + "."
        data_fmt = {"data_fmts": ["%#0.4g", "%#0.4g", "%#0.3F", "%s"]}
        html_data_fmt = dict(data_fmt)
        html_data_fmt["data_fmts"] = ["<td>" + i + "</td>"
                                      for i in html_data_fmt["data_fmts"]]
        return SimpleTable(data=[[self.test_statistic, self.crit_value,
                                  self.pvalue, str(self.df)]],
                           headers=['Test statistic', 'Critical value',
                                    'p-value', 'df'],
                           title=title,
                           txt_fmt=data_fmt,
                           html_fmt=html_data_fmt,
                           ltx_fmt=data_fmt)

    def __str__(self):
        return "<" + self.__module__ + "." + self.__class__.__name__ \
                   + " object. " + self.h0 + ": " + self.conclusion \
                   + self.signif_str \
                   + ". Test statistic: {:.3f}".format(self.test_statistic) \
                   + ", critical value: {:.3f}>".format(self.crit_value) \
                   + ", p-value: {:.3f}>".format(self.pvalue)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return np.allclose(self.test_statistic, other.test_statistic) \
            and np.allclose(self.crit_value, other.crit_value) \
            and np.allclose(self.pvalue, other.pvalue) \
            and np.allclose(self.signif, other.signif)


class CausalityTestResults(HypothesisTestResults):
    """
    Results class for Granger-causality and instantaneous causality.

    Parameters
    ----------
    causing : list of str
        This list contains the potentially causing variables.
    caused : list of str
        This list contains the potentially caused variables.
    test_statistic : float
    crit_value : float
    pvalue : float
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    test : str {``"granger"``, ``"inst"``}, default: ``"granger"``
        If ``"granger"``, Granger-causality has been tested. If ``"inst"``,
        instantaneous causality has been tested.
    method : str {``"f"``, ``"wald"``}
        The kind of test. ``"f"`` indicates an F-test, ``"wald"`` indicates a
        Wald-test.
    """
    def __init__(self, causing, caused, test_statistic, crit_value, pvalue, df,
                 signif, test="granger", method=None):
        self.causing = causing
        self.caused = caused
        self.test = test
        if method is None or method.lower() not in ["f", "wald"]:
            raise ValueError('The method ("f" for F-test, "wald" for '
                             "Wald-test) must not be None.")
        method = method.capitalize()
        # attributes used in summary and string representation:
        title = "Granger" if self.test == "granger" else "Instantaneous"
        title += " causality %s-test" % method
        h0 = "H_0: "
        if len(self.causing) == 1:
            h0 += "{} does not ".format(self.causing[0])
        else:
            h0 += "{} do not ".format(self.causing)
        h0 += "Granger-" if self.test == "granger" else "instantaneously "
        h0 += "cause "
        if len(self.caused) == 1:
            h0 += self.caused[0]
        else:
            h0 += "[" + ", ".join(caused) + "]"

        super().__init__(test_statistic, crit_value,
                         pvalue, df, signif, method,
                         title, h0)

    def __eq__(self, other):
        basic_test = super().__eq__(other)
        if not basic_test:
            return False
        test = self.test == other.test
        variables = (self.causing == other.causing and
                     self.caused == other.caused)
        # instantaneous causality is a symmetric relation ==> causing and
        # caused may be swapped
        if not variables and self.test == "inst":
            variables = (self.causing == other.caused and
                         self.caused == other.causing)
        return test and variables


class NormalityTestResults(HypothesisTestResults):
    """
    Results class for the Jarque-Bera-test for nonnormality.

    Parameters
    ----------
    test_statistic : float
        The test's test statistic.
    crit_value : float
        The test's critical value.
    pvalue : float
        The test's p-value.
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    """
    def __init__(self, test_statistic, crit_value, pvalue, df, signif):
        method = "Jarque-Bera"
        title = "normality (skew and kurtosis) test"
        h0 = 'H_0: data generated by normally-distributed process'
        super().__init__(test_statistic, crit_value,
                         pvalue, df, signif,
                         method, title, h0)


class WhitenessTestResults(HypothesisTestResults):
    """
    Results class for the Portmanteau-test for residual autocorrelation.

    Parameters
    ----------
    test_statistic : float
        The test's test statistic.
    crit_value : float
        The test's critical value.
    pvalue : float
        The test's p-value.
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    nlags : int
        Number of lags tested.
    """
    def __init__(self, test_statistic, crit_value, pvalue, df, signif, nlags,
                 adjusted):
        self.lags = nlags
        self.adjusted = adjusted
        method = "Portmanteau"
        title = "{}-test for residual autocorrelation".format(method)
        if adjusted:
            title = "Adjusted " + title
        h0 = "H_0: residual autocorrelation up to lag {} is zero".format(nlags)
        super().__init__(
            test_statistic,
            crit_value,
            pvalue,
            df,
            signif,
            method,
            title,
            h0
        )
