from statsmodels.compat.python import lzip

import numpy as np
from scipy import stats

from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant

from . import utils

__all__ = ["qqplot", "qqplot_2samples", "qqline", "ProbPlot"]


class ProbPlot:
    """
    Q-Q and P-P Probability Plots

    Can take arguments specifying the parameters for dist or fit them
    automatically. (See fit under kwargs.)

    Parameters
    ----------
    data : array_like
        A 1d data array
    dist : callable
        Compare x against dist. A scipy.stats or statsmodels distribution. The
        default is scipy.stats.distributions.norm (a standard normal). Can be
        a SciPy frozen distribution.
    fit : bool
        If fit is false, loc, scale, and distargs are passed to the
        distribution. If fit is True then the parameters for dist are fit
        automatically using dist.fit. The quantiles are formed from the
        standardized data, after subtracting the fitted loc and dividing by
        the fitted scale. fit cannot be used if dist is a SciPy frozen
        distribution.
    distargs : tuple
        A tuple of arguments passed to dist to specify it fully so dist.ppf
        may be called. distargs must not contain loc or scale. These values
        must be passed using the loc or scale inputs. distargs cannot be used
        if dist is a SciPy frozen distribution.
    a : float
        Offset for the plotting position of an expected order statistic, for
        example. The plotting positions are given by
        (i - a)/(nobs - 2*a + 1) for i in range(0,nobs+1)
    loc : float
        Location parameter for dist. Cannot be used if dist is a SciPy frozen
        distribution.
    scale : float
        Scale parameter for dist. Cannot be used if dist is a SciPy frozen
        distribution.

    See Also
    --------
    scipy.stats.probplot

    Notes
    -----
    1) Depends on matplotlib.
    2) If `fit` is True then the parameters are fit using the
        distribution's `fit()` method.
    3) The call signatures for the `qqplot`, `ppplot`, and `probplot`
        methods are similar, so examples 1 through 4 apply to all
        three methods.
    4) The three plotting methods are summarized below:
        ppplot : Probability-Probability plot
            Compares the sample and theoretical probabilities (percentiles).
        qqplot : Quantile-Quantile plot
            Compares the sample and theoretical quantiles
        probplot : Probability plot
            Same as a Q-Q plot, however probabilities are shown in the scale of
            the theoretical distribution (x-axis) and the y-axis contains
            unscaled quantiles of the sample data.

    Examples
    --------
    The first example shows a Q-Q plot for regression residuals

    >>> # example 1
    >>> import statsmodels.api as sm
    >>> from matplotlib import pyplot as plt
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> model = sm.OLS(data.endog, data.exog)
    >>> mod_fit = model.fit()
    >>> res = mod_fit.resid # residuals
    >>> pplot = sm.ProbPlot(res)
    >>> fig = pplot.qqplot()
    >>> h = plt.title("Ex. 1 - qqplot - residuals of OLS fit")
    >>> plt.show()

    qqplot of the residuals against quantiles of t-distribution with 4
    degrees of freedom:

    >>> # example 2
    >>> import scipy.stats as stats
    >>> pplot = sm.ProbPlot(res, stats.t, distargs=(4,))
    >>> fig = pplot.qqplot()
    >>> h = plt.title("Ex. 2 - qqplot - residuals against quantiles of t-dist")
    >>> plt.show()

    qqplot against same as above, but with mean 3 and std 10:

    >>> # example 3
    >>> pplot = sm.ProbPlot(res, stats.t, distargs=(4,), loc=3, scale=10)
    >>> fig = pplot.qqplot()
    >>> h = plt.title("Ex. 3 - qqplot - resids vs quantiles of t-dist")
    >>> plt.show()

    Automatically determine parameters for t distribution including the
    loc and scale:

    >>> # example 4
    >>> pplot = sm.ProbPlot(res, stats.t, fit=True)
    >>> fig = pplot.qqplot(line="45")
    >>> h = plt.title("Ex. 4 - qqplot - resids vs. quantiles of fitted t-dist")
    >>> plt.show()

    A second `ProbPlot` object can be used to compare two separate sample
    sets by using the `other` kwarg in the `qqplot` and `ppplot` methods.

    >>> # example 5
    >>> import numpy as np
    >>> x = np.random.normal(loc=8.25, scale=2.75, size=37)
    >>> y = np.random.normal(loc=8.75, scale=3.25, size=37)
    >>> pp_x = sm.ProbPlot(x, fit=True)
    >>> pp_y = sm.ProbPlot(y, fit=True)
    >>> fig = pp_x.qqplot(line="45", other=pp_y)
    >>> h = plt.title("Ex. 5 - qqplot - compare two sample sets")
    >>> plt.show()

    In qqplot, sample size of `other` can be equal or larger than the first.
    In case of larger, size of `other` samples will be reduced to match the
    size of the first by interpolation

    >>> # example 6
    >>> x = np.random.normal(loc=8.25, scale=2.75, size=37)
    >>> y = np.random.normal(loc=8.75, scale=3.25, size=57)
    >>> pp_x = sm.ProbPlot(x, fit=True)
    >>> pp_y = sm.ProbPlot(y, fit=True)
    >>> fig = pp_x.qqplot(line="45", other=pp_y)
    >>> title = "Ex. 6 - qqplot - compare different sample sizes"
    >>> h = plt.title(title)
    >>> plt.show()

    In ppplot, sample size of `other` and the first can be different. `other`
    will be used to estimate an empirical cumulative distribution function
    (ECDF). ECDF(x) will be plotted against p(x)=0.5/n, 1.5/n, ..., (n-0.5)/n
    where x are sorted samples from the first.

    >>> # example 7
    >>> x = np.random.normal(loc=8.25, scale=2.75, size=37)
    >>> y = np.random.normal(loc=8.75, scale=3.25, size=57)
    >>> pp_x = sm.ProbPlot(x, fit=True)
    >>> pp_y = sm.ProbPlot(y, fit=True)
    >>> pp_y.ppplot(line="45", other=pp_x)
    >>> plt.title("Ex. 7A- ppplot - compare two sample sets, other=pp_x")
    >>> pp_x.ppplot(line="45", other=pp_y)
    >>> plt.title("Ex. 7B- ppplot - compare two sample sets, other=pp_y")
    >>> plt.show()

    The following plot displays some options, follow the link to see the
    code.

    .. plot:: plots/graphics_gofplots_qqplot.py
    """

    def __init__(
        self,
        data,
        dist=stats.norm,
        fit=False,
        distargs=(),
        a=0,
        loc=0,
        scale=1,
    ):

        self.data = data
        self.a = a
        self.nobs = data.shape[0]
        self.distargs = distargs
        self.fit = fit

        self._is_frozen = isinstance(dist, stats.distributions.rv_frozen)
        if self._is_frozen and (
            fit or loc != 0 or scale != 1 or distargs != ()
        ):
            raise ValueError(
                "Frozen distributions cannot be combined with fit, loc, scale"
                " or distargs."
            )
        # propertes
        self._cache = {}
        if self._is_frozen:
            self.dist = dist
            dist_gen = dist.dist
            shapes = dist_gen.shapes
            if shapes is not None:
                shape_args = tuple(map(str.strip, shapes.split(",")))
            else:
                shape_args = ()
            numargs = len(shape_args)
            args = dist.args
            if len(args) >= numargs + 1:
                self.loc = args[numargs]
            else:
                self.loc = dist.kwds.get("loc", loc)
            if len(args) >= numargs + 2:
                self.scale = args[numargs + 1]
            else:
                self.scale = dist.kwds.get("scale", scale)
            fit_params = []
            for i, arg in enumerate(shape_args):
                if arg in dist.kwds:
                    value = dist.kwds[arg]
                else:
                    value = dist.args[i]
                fit_params.append(value)
            self.fit_params = np.r_[fit_params, self.loc, self.scale]
        elif fit:
            self.fit_params = dist.fit(data)
            self.loc = self.fit_params[-2]
            self.scale = self.fit_params[-1]
            if len(self.fit_params) > 2:
                self.dist = dist(*self.fit_params[:-2], **dict(loc=0, scale=1))
            else:
                self.dist = dist(loc=0, scale=1)
        elif distargs or loc != 0 or scale != 1:
            try:
                self.dist = dist(*distargs, **dict(loc=loc, scale=scale))
            except Exception:
                distargs = ", ".join([str(da) for da in distargs])
                cmd = "dist({distargs}, loc={loc}, scale={scale})"
                cmd = cmd.format(distargs=distargs, loc=loc, scale=scale)
                raise TypeError(
                    "Initializing the distribution failed.  This "
                    "can occur if distargs contains loc or scale. "
                    "The distribution initialization command "
                    "is:\n{cmd}".format(cmd=cmd)
                )
            self.loc = loc
            self.scale = scale
            self.fit_params = np.r_[distargs, loc, scale]
        else:
            self.dist = dist
            self.loc = loc
            self.scale = scale
            self.fit_params = np.r_[loc, scale]

    @cache_readonly
    def theoretical_percentiles(self):
        """Theoretical percentiles"""
        return plotting_pos(self.nobs, self.a)

    @cache_readonly
    def theoretical_quantiles(self):
        """Theoretical quantiles"""
        try:
            return self.dist.ppf(self.theoretical_percentiles)
        except TypeError:
            msg = "%s requires more parameters to compute ppf".format(
                self.dist.name,
            )
            raise TypeError(msg)
        except Exception as exc:
            msg = "failed to compute the ppf of {0}".format(self.dist.name)
            raise type(exc)(msg)

    @cache_readonly
    def sorted_data(self):
        """sorted data"""
        sorted_data = np.array(self.data, copy=True)
        sorted_data.sort()
        return sorted_data

    @cache_readonly
    def sample_quantiles(self):
        """sample quantiles"""
        if self.fit and self.loc != 0 and self.scale != 1:
            return (self.sorted_data - self.loc) / self.scale
        else:
            return self.sorted_data

    @cache_readonly
    def sample_percentiles(self):
        """Sample percentiles"""
        _check_for(self.dist, "cdf")
        if self._is_frozen:
            return self.dist.cdf(self.sorted_data)
        quantiles = (self.sorted_data - self.fit_params[-2]) / self.fit_params[
            -1
        ]
        return self.dist.cdf(quantiles)

    def ppplot(
        self,
        xlabel=None,
        ylabel=None,
        line=None,
        other=None,
        ax=None,
        **plotkwargs,
    ):
        """
        Plot of the percentiles of x versus the percentiles of a distribution.

        Parameters
        ----------
        xlabel : str or None, optional
            User-provided labels for the x-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        ylabel : str or None, optional
            User-provided labels for the y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : {None, "45", "s", "r", q"}, optional
            Options for the reference line to which the data is compared:

            - "45": 45-degree line
            - "s": standardized line, the expected order statistics are
              scaled by the standard deviation of the given sample and have
              the mean added to them
            - "r": A regression line is fit
            - "q": A line is fit through the quartiles.
            - None: by default no reference line is added to the plot.

        other : ProbPlot, array_like, or None, optional
            If provided, ECDF(x) will be plotted against p(x) where x are
            sorted samples from `self`. ECDF is an empirical cumulative
            distribution function estimated from `other` and
            p(x) = 0.5/n, 1.5/n, ..., (n-0.5)/n where n is the number of
            samples in `self`. If an array-object is provided, it will be
            turned into a `ProbPlot` instance default parameters. If not
            provided (default), `self.dist(x)` is be plotted against p(x).

        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        **plotkwargs
            Additional arguments to be passed to the `plot` command.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        if other is not None:
            check_other = isinstance(other, ProbPlot)
            if not check_other:
                other = ProbPlot(other)

            p_x = self.theoretical_percentiles
            ecdf_x = ECDF(other.sample_quantiles)(self.sample_quantiles)

            fig, ax = _do_plot(
                p_x, ecdf_x, self.dist, ax=ax, line=line, **plotkwargs
            )

            if xlabel is None:
                xlabel = "Probabilities of 2nd Sample"
            if ylabel is None:
                ylabel = "Probabilities of 1st Sample"

        else:
            fig, ax = _do_plot(
                self.theoretical_percentiles,
                self.sample_percentiles,
                self.dist,
                ax=ax,
                line=line,
                **plotkwargs,
            )
            if xlabel is None:
                xlabel = "Theoretical Probabilities"
            if ylabel is None:
                ylabel = "Sample Probabilities"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        return fig

    def qqplot(
        self,
        xlabel=None,
        ylabel=None,
        line=None,
        other=None,
        ax=None,
        swap: bool = False,
        **plotkwargs,
    ):
        """
        Plot of the quantiles of x versus the quantiles/ppf of a distribution.

        Can also be used to plot against the quantiles of another `ProbPlot`
        instance.

        Parameters
        ----------
        xlabel : {None, str}
            User-provided labels for the x-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        ylabel : {None, str}
            User-provided labels for the y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : {None, "45", "s", "r", q"}, optional
            Options for the reference line to which the data is compared:

            - "45" - 45-degree line
            - "s" - standardized line, the expected order statistics are scaled
              by the standard deviation of the given sample and have the mean
              added to them
            - "r" - A regression line is fit
            - "q" - A line is fit through the quartiles.
            - None - by default no reference line is added to the plot.

        other : {ProbPlot, array_like, None}, optional
            If provided, the sample quantiles of this `ProbPlot` instance are
            plotted against the sample quantiles of the `other` `ProbPlot`
            instance. Sample size of `other` must be equal or larger than
            this `ProbPlot` instance. If the sample size is larger, sample
            quantiles of `other` will be interpolated to match the sample size
            of this `ProbPlot` instance. If an array-like object is provided,
            it will be turned into a `ProbPlot` instance using default
            parameters. If not provided (default), the theoretical quantiles
            are used.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        swap : bool, optional
            Flag indicating to swap the x and y labels.
        **plotkwargs
            Additional arguments to be passed to the `plot` command.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        if other is not None:
            check_other = isinstance(other, ProbPlot)
            if not check_other:
                other = ProbPlot(other)

            s_self = self.sample_quantiles
            s_other = other.sample_quantiles

            if len(s_self) > len(s_other):
                raise ValueError(
                    "Sample size of `other` must be equal or "
                    + "larger than this `ProbPlot` instance"
                )
            elif len(s_self) < len(s_other):
                # Use quantiles of the smaller set and interpolate quantiles of
                # the larger data set
                p = plotting_pos(self.nobs, self.a)
                s_other = stats.mstats.mquantiles(s_other, p)
            fig, ax = _do_plot(
                s_other, s_self, self.dist, ax=ax, line=line, **plotkwargs
            )

            if xlabel is None:
                xlabel = "Quantiles of 2nd Sample"
            if ylabel is None:
                ylabel = "Quantiles of 1st Sample"
            if swap:
                xlabel, ylabel = ylabel, xlabel

        else:
            fig, ax = _do_plot(
                self.theoretical_quantiles,
                self.sample_quantiles,
                self.dist,
                ax=ax,
                line=line,
                **plotkwargs,
            )
            if xlabel is None:
                xlabel = "Theoretical Quantiles"
            if ylabel is None:
                ylabel = "Sample Quantiles"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig

    def probplot(
        self,
        xlabel=None,
        ylabel=None,
        line=None,
        exceed=False,
        ax=None,
        **plotkwargs,
    ):
        """
        Plot of unscaled quantiles of x against the prob of a distribution.

        The x-axis is scaled linearly with the quantiles, but the probabilities
        are used to label the axis.

        Parameters
        ----------
        xlabel : {None, str}, optional
            User-provided labels for the x-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        ylabel : {None, str}, optional
            User-provided labels for the y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : {None, "45", "s", "r", q"}, optional
            Options for the reference line to which the data is compared:

            - "45" - 45-degree line
            - "s" - standardized line, the expected order statistics are scaled
              by the standard deviation of the given sample and have the mean
              added to them
            - "r" - A regression line is fit
            - "q" - A line is fit through the quartiles.
            - None - by default no reference line is added to the plot.

        exceed : bool, optional
            If False (default) the raw sample quantiles are plotted against
            the theoretical quantiles, show the probability that a sample will
            not exceed a given value. If True, the theoretical quantiles are
            flipped such that the figure displays the probability that a
            sample will exceed a given value.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        **plotkwargs
            Additional arguments to be passed to the `plot` command.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        if exceed:
            fig, ax = _do_plot(
                self.theoretical_quantiles[::-1],
                self.sorted_data,
                self.dist,
                ax=ax,
                line=line,
                **plotkwargs,
            )
            if xlabel is None:
                xlabel = "Probability of Exceedance (%)"

        else:
            fig, ax = _do_plot(
                self.theoretical_quantiles,
                self.sorted_data,
                self.dist,
                ax=ax,
                line=line,
                **plotkwargs,
            )
            if xlabel is None:
                xlabel = "Non-exceedance Probability (%)"

        if ylabel is None:
            ylabel = "Sample Quantiles"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _fmt_probplot_axis(ax, self.dist, self.nobs)

        return fig


def qqplot(
    data,
    dist=stats.norm,
    distargs=(),
    a=0,
    loc=0,
    scale=1,
    fit=False,
    line=None,
    ax=None,
    **plotkwargs,
):
    """
    Q-Q plot of the quantiles of x versus the quantiles/ppf of a distribution.

    Can take arguments specifying the parameters for dist or fit them
    automatically. (See fit under Parameters.)

    Parameters
    ----------
    data : array_like
        A 1d data array.
    dist : callable
        Comparison distribution. The default is
        scipy.stats.distributions.norm (a standard normal).
    distargs : tuple
        A tuple of arguments passed to dist to specify it fully
        so dist.ppf may be called.
    a : float
        Offset for the plotting position of an expected order statistic, for
        example. The plotting positions are given by (i - a)/(nobs - 2*a + 1)
        for i in range(0,nobs+1)
    loc : float
        Location parameter for dist
    scale : float
        Scale parameter for dist
    fit : bool
        If fit is false, loc, scale, and distargs are passed to the
        distribution. If fit is True then the parameters for dist
        are fit automatically using dist.fit. The quantiles are formed
        from the standardized data, after subtracting the fitted loc
        and dividing by the fitted scale.
    line : {None, "45", "s", "r", "q"}
        Options for the reference line to which the data is compared:

        - "45" - 45-degree line
        - "s" - standardized line, the expected order statistics are scaled
          by the standard deviation of the given sample and have the mean
          added to them
        - "r" - A regression line is fit
        - "q" - A line is fit through the quartiles.
        - None - by default no reference line is added to the plot.

    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    **plotkwargs
        Additional matplotlib arguments to be passed to the `plot` command.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    scipy.stats.probplot

    Notes
    -----
    Depends on matplotlib. If `fit` is True then the parameters are fit using
    the distribution's fit() method.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from matplotlib import pyplot as plt
    >>> data = sm.datasets.longley.load()
    >>> exog = sm.add_constant(data.exog)
    >>> mod_fit = sm.OLS(data.endog, exog).fit()
    >>> res = mod_fit.resid # residuals
    >>> fig = sm.qqplot(res)
    >>> plt.show()

    qqplot of the residuals against quantiles of t-distribution with 4 degrees
    of freedom:

    >>> import scipy.stats as stats
    >>> fig = sm.qqplot(res, stats.t, distargs=(4,))
    >>> plt.show()

    qqplot against same as above, but with mean 3 and std 10:

    >>> fig = sm.qqplot(res, stats.t, distargs=(4,), loc=3, scale=10)
    >>> plt.show()

    Automatically determine parameters for t distribution including the
    loc and scale:

    >>> fig = sm.qqplot(res, stats.t, fit=True, line="45")
    >>> plt.show()

    The following plot displays some options, follow the link to see the code.

    .. plot:: plots/graphics_gofplots_qqplot.py
    """
    probplot = ProbPlot(
        data, dist=dist, distargs=distargs, fit=fit, a=a, loc=loc, scale=scale
    )
    fig = probplot.qqplot(ax=ax, line=line, **plotkwargs)
    return fig


def qqplot_2samples(
    data1, data2, xlabel=None, ylabel=None, line=None, ax=None
):
    """
    Q-Q Plot of two samples' quantiles.

    Can take either two `ProbPlot` instances or two array-like objects. In the
    case of the latter, both inputs will be converted to `ProbPlot` instances
    using only the default values - so use `ProbPlot` instances if
    finer-grained control of the quantile computations is required.

    Parameters
    ----------
    data1 : {array_like, ProbPlot}
        Data to plot along x axis. If the sample sizes are unequal, the longer
        series is always plotted along the x-axis.
    data2 : {array_like, ProbPlot}
        Data to plot along y axis. Does not need to have the same number of
        observations as data 1. If the sample sizes are unequal, the longer
        series is always plotted along the x-axis.
    xlabel : {None, str}
        User-provided labels for the x-axis. If None (default),
        other values are used.
    ylabel : {None, str}
        User-provided labels for the y-axis. If None (default),
        other values are used.
    line : {None, "45", "s", "r", q"}
        Options for the reference line to which the data is compared:

        - "45" - 45-degree line
        - "s" - standardized line, the expected order statistics are scaled
          by the standard deviation of the given sample and have the mean
          added to them
        - "r" - A regression line is fit
        - "q" - A line is fit through the quartiles.
        - None - by default no reference line is added to the plot.

    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    scipy.stats.probplot

    Notes
    -----
    1) Depends on matplotlib.
    2) If `data1` and `data2` are not `ProbPlot` instances, instances will be
       created using the default parameters. Therefore, it is recommended to use
       `ProbPlot` instance if fine-grained control is needed in the computation
       of the quantiles.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from statsmodels.graphics.gofplots import qqplot_2samples
    >>> x = np.random.normal(loc=8.5, scale=2.5, size=37)
    >>> y = np.random.normal(loc=8.0, scale=3.0, size=37)
    >>> pp_x = sm.ProbPlot(x)
    >>> pp_y = sm.ProbPlot(y)
    >>> qqplot_2samples(pp_x, pp_y)
    >>> plt.show()

    .. plot:: plots/graphics_gofplots_qqplot_2samples.py

    >>> fig = qqplot_2samples(pp_x, pp_y, xlabel=None, ylabel=None,
    ...                       line=None, ax=None)
    """
    if not isinstance(data1, ProbPlot):
        data1 = ProbPlot(data1)

    if not isinstance(data2, ProbPlot):
        data2 = ProbPlot(data2)
    if data2.data.shape[0] > data1.data.shape[0]:
        fig = data1.qqplot(
            xlabel=ylabel, ylabel=xlabel, line=line, other=data2, ax=ax
        )
    else:
        fig = data2.qqplot(
            xlabel=ylabel,
            ylabel=xlabel,
            line=line,
            other=data1,
            ax=ax,
            swap=True,
        )

    return fig


def qqline(ax, line, x=None, y=None, dist=None, fmt="r-", **lineoptions):
    """
    Plot a reference line for a qqplot.

    Parameters
    ----------
    ax : matplotlib axes instance
        The axes on which to plot the line
    line : str {"45","r","s","q"}
        Options for the reference line to which the data is compared.:

        - "45" - 45-degree line
        - "s"  - standardized line, the expected order statistics are scaled by
                 the standard deviation of the given sample and have the mean
                 added to them
        - "r"  - A regression line is fit
        - "q"  - A line is fit through the quartiles.
        - None - By default no reference line is added to the plot.

    x : ndarray
        X data for plot. Not needed if line is "45".
    y : ndarray
        Y data for plot. Not needed if line is "45".
    dist : scipy.stats.distribution
        A scipy.stats distribution, needed if line is "q".
    fmt : str, optional
        Line format string passed to `plot`.
    **lineoptions
        Additional arguments to be passed to the `plot` command.

    Notes
    -----
    There is no return value. The line is plotted on the given `ax`.

    Examples
    --------
    Import the food expenditure dataset.  Plot annual food expenditure on x-axis
    and household income on y-axis.  Use qqline to add regression line into the
    plot.

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from statsmodels.graphics.gofplots import qqline

    >>> foodexp = sm.datasets.engel.load()
    >>> x = foodexp.exog
    >>> y = foodexp.endog
    >>> ax = plt.subplot(111)
    >>> plt.scatter(x, y)
    >>> ax.set_xlabel(foodexp.exog_name[0])
    >>> ax.set_ylabel(foodexp.endog_name)
    >>> qqline(ax, "r", x, y)
    >>> plt.show()

    .. plot:: plots/graphics_gofplots_qqplot_qqline.py
    """
    lineoptions = lineoptions.copy()
    for ls in ("-", "--", "-.", ":"):
        if ls in fmt:
            lineoptions.setdefault("linestyle", ls)
            fmt = fmt.replace(ls, "")
            break
    for marker in (
        ".",
        ",",
        "o",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "8",
        "s",
        "p",
        "P",
        "*",
        "h",
        "H",
        "+",
        "x",
        "X",
        "D",
        "d",
        "|",
        "_",
    ):
        if marker in fmt:
            lineoptions.setdefault("marker", marker)
            fmt = fmt.replace(marker, "")
            break
    if fmt:
        lineoptions.setdefault("color", fmt)

    if line == "45":
        end_pts = lzip(ax.get_xlim(), ax.get_ylim())
        end_pts[0] = min(end_pts[0])
        end_pts[1] = max(end_pts[1])
        ax.plot(end_pts, end_pts, **lineoptions)
        ax.set_xlim(end_pts)
        ax.set_ylim(end_pts)
        return  # does this have any side effects?
    if x is None or y is None:
        raise ValueError("If line is not 45, x and y cannot be None.")
    x = np.array(x)
    y = np.array(y)
    if line == "r":
        # could use ax.lines[0].get_xdata(), get_ydata(),
        # but don't know axes are "clean"
        y = OLS(y, add_constant(x)).fit().fittedvalues
        ax.plot(x, y, **lineoptions)
    elif line == "s":
        m, b = np.std(y), np.mean(y)
        ref_line = x * m + b
        ax.plot(x, ref_line, **lineoptions)
    elif line == "q":
        _check_for(dist, "ppf")
        q25 = stats.scoreatpercentile(y, 25)
        q75 = stats.scoreatpercentile(y, 75)
        theoretical_quartiles = dist.ppf([0.25, 0.75])
        m = (q75 - q25) / np.diff(theoretical_quartiles)
        b = q25 - m * theoretical_quartiles[0]
        ax.plot(x, m * x + b, **lineoptions)


# about 10x faster than plotting_position in sandbox and mstats
def plotting_pos(nobs, a=0.0, b=None):
    """
    Generates sequence of plotting positions

    Parameters
    ----------
    nobs : int
        Number of probability points to plot
    a : float, default 0.0
        alpha parameter for the plotting position of an expected order
        statistic
    b : float, default None
        beta parameter for the plotting position of an expected order
        statistic. If None, then b is set to a.

    Returns
    -------
    ndarray
        The plotting positions

    Notes
    -----
    The plotting positions are given by (i - a)/(nobs + 1 - a - b) for i in
    range(1, nobs+1)

    See Also
    --------
    scipy.stats.mstats.plotting_positions
        Additional information on alpha and beta
    """
    b = a if b is None else b
    return (np.arange(1.0, nobs + 1) - a) / (nobs + 1 - a - b)


def _fmt_probplot_axis(ax, dist, nobs):
    """
    Formats a theoretical quantile axis to display the corresponding
    probabilities on the quantiles' scale.

    Parameters
    ----------
    ax : AxesSubplot, optional
        The axis to be formatted
    nobs : scalar
        Number of observations in the sample
    dist : scipy.stats.distribution
        A scipy.stats distribution sufficiently specified to implement its
        ppf() method.

    Returns
    -------
    There is no return value. This operates on `ax` in place
    """
    _check_for(dist, "ppf")
    axis_probs = np.linspace(10, 90, 9, dtype=float)
    small = np.array([1.0, 2, 5])
    axis_probs = np.r_[small, axis_probs, 100 - small[::-1]]
    if nobs >= 50:
        axis_probs = np.r_[small / 10, axis_probs, 100 - small[::-1] / 10]
    if nobs >= 500:
        axis_probs = np.r_[small / 100, axis_probs, 100 - small[::-1] / 100]
    axis_probs /= 100.0
    axis_qntls = dist.ppf(axis_probs)
    ax.set_xticks(axis_qntls)
    ax.set_xticklabels(
        [str(lbl) for lbl in (axis_probs * 100)],
        rotation=45,
        rotation_mode="anchor",
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax.set_xlim([axis_qntls.min(), axis_qntls.max()])


def _do_plot(
    x, y, dist=None, line=None, ax=None, fmt="b", step=False, **kwargs
):
    """
    Boiler plate plotting function for the `ppplot`, `qqplot`, and
    `probplot` methods of the `ProbPlot` class

    Parameters
    ----------
    x : array_like
        X-axis data to be plotted
    y : array_like
        Y-axis data to be plotted
    dist : scipy.stats.distribution
        A scipy.stats distribution, needed if `line` is "q".
    line : {"45", "s", "r", "q", None}, default None
        Options for the reference line to which the data is compared.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    fmt : str, optional
        matplotlib-compatible formatting string for the data markers
    kwargs : keywords
        These are passed to matplotlib.plot

    Returns
    -------
    fig : Figure
        The figure containing `ax`.
    ax : AxesSubplot
        The original axes if provided.  Otherwise a new instance.
    """
    plot_style = {
        "marker": "o",
        "markerfacecolor": "C0",
        "markeredgecolor": "C0",
        "linestyle": "none",
    }

    plot_style.update(**kwargs)
    where = plot_style.pop("where", "pre")

    fig, ax = utils.create_mpl_ax(ax)
    ax.set_xmargin(0.02)

    if step:
        ax.step(x, y, fmt, where=where, **plot_style)
    else:
        ax.plot(x, y, fmt, **plot_style)
    if line:
        if line not in ["r", "q", "45", "s"]:
            msg = "%s option for line not understood" % line
            raise ValueError(msg)

        qqline(ax, line, x=x, y=y, dist=dist)

    return fig, ax


def _check_for(dist, attr="ppf"):
    if not hasattr(dist, attr):
        raise AttributeError(f"distribution must have a {attr} method")
