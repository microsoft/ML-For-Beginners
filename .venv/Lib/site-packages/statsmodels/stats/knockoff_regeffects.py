import numpy as np


class RegressionEffects:
    """
    Base class for regression effects used in RegressionFDR.

    Any implementation of the class must provide a method called
    'stats' that takes a RegressionFDR object and returns effect sizes
    for the model coefficients.  Greater values for these statistics
    imply greater evidence that the effect is real.

    Knockoff effect sizes are based on fitting the regression model to
    an extended design matrix [X X'], where X' is a design matrix with
    the same shape as the actual design matrix X.  The construction of
    X' guarantees that there are no true associations between the
    columns of X' and the dependent variable of the regression.  If X
    has p columns, then the effect size of covariate j is based on the
    strength of the estimated association for coefficient j compared
    to the strength of the estimated association for coefficient p+j.
    """

    def stats(self, parent):
        raise NotImplementedError


class CorrelationEffects(RegressionEffects):
    """
    Marginal correlation effect sizes for FDR control.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.

    Notes
    -----
    This class implements the marginal correlation approach to
    constructing test statistics for a knockoff analysis, as
    described under (1) in section 2.2 of the Barber and Candes
    paper.
    """

    def stats(self, parent):
        s1 = np.dot(parent.exog1.T, parent.endog)
        s2 = np.dot(parent.exog2.T, parent.endog)
        return np.abs(s1) - np.abs(s2)


class ForwardEffects(RegressionEffects):
    """
    Forward selection effect sizes for FDR control.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.
    pursuit : bool
        If True, 'basis pursuit' is used, which amounts to performing
        a full regression at each selection step to adjust the working
        residual vector.  If False (the default), the residual is
        adjusted by regressing out each selected variable marginally.
        Setting pursuit=True will be considerably slower, but may give
        better results when exog is not orthogonal.

    Notes
    -----
    This class implements the forward selection approach to
    constructing test statistics for a knockoff analysis, as
    described under (5) in section 2.2 of the Barber and Candes
    paper.
    """

    def __init__(self, pursuit):
        self.pursuit = pursuit

    def stats(self, parent):
        nvar = parent.exog.shape[1]
        rv = parent.endog.copy()
        vl = [(i, parent.exog[:, i]) for i in range(nvar)]
        z = np.empty(nvar)
        past = []
        for i in range(nvar):
            dp = np.r_[[np.abs(np.dot(rv, x[1])) for x in vl]]
            j = np.argmax(dp)
            z[vl[j][0]] = nvar - i - 1
            x = vl[j][1]
            del vl[j]
            if self.pursuit:
                for v in past:
                    x -= np.dot(x, v)*v
                past.append(x)
            rv -= np.dot(rv, x) * x
        z1 = z[0:nvar//2]
        z2 = z[nvar//2:]
        st = np.where(z1 > z2, z1, z2) * np.sign(z1 - z2)
        return st


class OLSEffects(RegressionEffects):
    """
    OLS regression for knockoff analysis.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.

    Notes
    -----
    This class implements the ordinary least squares regression
    approach to constructing test statistics for a knockoff analysis,
    as described under (2) in section 2.2 of the Barber and Candes
    paper.
    """

    def stats(self, parent):
        from statsmodels.regression.linear_model import OLS

        model = OLS(parent.endog, parent.exog)
        result = model.fit()
        q = len(result.params) // 2
        stats = np.abs(result.params[0:q]) - np.abs(result.params[q:])
        return stats


class RegModelEffects(RegressionEffects):
    """
    Use any regression model for Regression FDR analysis.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.
    model_cls : class
        Any model with appropriate fit or fit_regularized
        functions
    regularized : bool
        If True, use fit_regularized to fit the model
    model_kws : dict
        Keywords passed to model initializer
    fit_kws : dict
        Dictionary of keyword arguments for fit or fit_regularized
    """

    def __init__(self, model_cls, regularized=False, model_kws=None,
                 fit_kws=None):
        self.model_cls = model_cls
        self.regularized = regularized
        self.model_kws = model_kws if model_kws is not None else {}
        self.fit_kws = fit_kws if fit_kws is not None else {}

    def stats(self, parent):
        model = self.model_cls(parent.endog, parent.exog, **self.model_kws)
        if self.regularized:
            params = model.fit_regularized(**self.fit_kws).params
        else:
            params = model.fit(**self.fit_kws).params
        q = len(params) // 2
        stats = np.abs(params[0:q]) - np.abs(params[q:])
        return stats
