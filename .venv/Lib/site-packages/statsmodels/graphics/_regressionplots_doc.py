_plot_added_variable_doc = """\
    Create an added variable plot for a fitted regression model.

    Parameters
    ----------
    %(extra_params_doc)sfocus_exog : int or string
        The column index of exog, or a variable name, indicating the
        variable whose role in the regression is to be assessed.
    resid_type : str
        The type of residuals to use for the dependent variable.  If
        None, uses `resid_deviance` for GLM/GEE and `resid` otherwise.
    use_glm_weights : bool
        Only used if the model is a GLM or GEE.  If True, the
        residuals for the focus predictor are computed using WLS, with
        the weights obtained from the IRLS calculations for fitting
        the GLM. If False, unweighted regression is used.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to fit when refitting the
        model.
    ax: Axes
        Matplotlib Axes instance

    Returns
    -------
    Figure
        A matplotlib figure instance.
"""

_plot_partial_residuals_doc = """\
    Create a partial residual, or 'component plus residual' plot for a
    fitted regression model.

    Parameters
    ----------
    %(extra_params_doc)sfocus_exog : int or string
        The column index of exog, or variable name, indicating the
        variable whose role in the regression is to be assessed.
    ax: Axes
        Matplotlib Axes instance

    Returns
    -------
    Figure
        A matplotlib figure instance.
"""

_plot_ceres_residuals_doc = """\
    Conditional Expectation Partial Residuals (CERES) plot.

    Produce a CERES plot for a fitted regression model.

    Parameters
    ----------
    %(extra_params_doc)s
    focus_exog : {int, str}
        The column index of results.model.exog, or the variable name,
        indicating the variable whose role in the regression is to be
        assessed.
    frac : float
        Lowess tuning parameter for the adjusted model used in the
        CERES analysis.  Not used if `cond_means` is provided.
    cond_means : array_like, optional
        If provided, the columns of this array span the space of the
        conditional means E[exog | focus exog], where exog ranges over
        some or all of the columns of exog (other than the focus exog).
    ax : matplotlib.Axes instance, optional
        The axes on which to draw the plot. If not provided, a new
        axes instance is created.

    Returns
    -------
    Figure
        The figure on which the partial residual plot is drawn.

    Notes
    -----
    `cond_means` is intended to capture the behavior of E[x1 |
    x2], where x2 is the focus exog and x1 are all the other exog
    variables.  If all the conditional mean relationships are
    linear, it is sufficient to set cond_means equal to the focus
    exog.  Alternatively, cond_means may consist of one or more
    columns containing functional transformations of the focus
    exog (e.g. x2^2) that are thought to capture E[x1 | x2].

    If nothing is known or suspected about the form of E[x1 | x2],
    set `cond_means` to None, and it will be estimated by
    smoothing each non-focus exog against the focus exog.  The
    values of `frac` control these lowess smooths.

    If cond_means contains only the focus exog, the results are
    equivalent to a partial residual plot.

    If the focus variable is believed to be independent of the
    other exog variables, `cond_means` can be set to an (empty)
    nx0 array.

    References
    ----------
    .. [1] RD Cook and R Croos-Dabrera (1998).  Partial residual plots
       in generalized linear models.  Journal of the American
       Statistical Association, 93:442.

    .. [2] RD Cook (1993). Partial residual plots.  Technometrics 35:4.

    Examples
    --------
    Using a model built from the the state crime dataset, make a CERES plot with
    the rate of Poverty as the focus variable.

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.formula.api as smf
    >>> from statsmodels.graphics.regressionplots import plot_ceres_residuals

    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> plot_ceres_residuals(results, 'poverty')
    >>> plt.show()

    .. plot:: plots/graphics_regression_ceres_residuals.py
"""


_plot_influence_doc = """\
    Plot of influence in regression. Plots studentized resids vs. leverage.

    Parameters
    ----------
    {extra_params_doc}
    external : bool
        Whether to use externally or internally studentized residuals. It is
        recommended to leave external as True.
    alpha : float
        The alpha value to identify large studentized residuals. Large means
        abs(resid_studentized) > t.ppf(1-alpha/2, dof=results.df_resid)
    criterion : str {{'DFFITS', 'Cooks'}}
        Which criterion to base the size of the points on. Options are
        DFFITS or Cook's D.
    size : float
        The range of `criterion` is mapped to 10**2 - size**2 in points.
    plot_alpha : float
        The `alpha` of the plotted points.
    ax : AxesSubplot
        An instance of a matplotlib Axes.
    **kwargs
        Additional parameters passed through to `plot`.

    Returns
    -------
    Figure
        The matplotlib figure that contains the Axes.

    Notes
    -----
    Row labels for the observations in which the leverage, measured by the
    diagonal of the hat matrix, is high or the residuals are large, as the
    combination of large residuals and a high influence value indicates an
    influence point. The value of large residuals can be controlled using the
    `alpha` parameter. Large leverage points are identified as
    hat_i > 2 * (df_model + 1)/nobs.

    Examples
    --------
    Using a model built from the the state crime dataset, plot the influence in
    regression.  Observations with high leverage, or large residuals will be
    labeled in the plot to show potential influence points.

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.formula.api as smf

    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> sm.graphics.influence_plot(results)
    >>> plt.show()

    .. plot:: plots/graphics_regression_influence.py
    """


_plot_leverage_resid2_doc = """\
    Plot leverage statistics vs. normalized residuals squared

    Parameters
    ----------
    results : results instance
        A regression results instance
    alpha : float
        Specifies the cut-off for large-standardized residuals. Residuals
        are assumed to be distributed N(0, 1) with alpha=alpha.
    ax : Axes
        Matplotlib Axes instance
    **kwargs
        Additional parameters passed the plot command.

    Returns
    -------
    Figure
        A matplotlib figure instance.

    Examples
    --------
    Using a model built from the the state crime dataset, plot the leverage
    statistics vs. normalized residuals squared.  Observations with
    Large-standardized Residuals will be labeled in the plot.

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.formula.api as smf

    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> sm.graphics.plot_leverage_resid2(results)
    >>> plt.show()

    .. plot:: plots/graphics_regression_leverage_resid2.py
    """
