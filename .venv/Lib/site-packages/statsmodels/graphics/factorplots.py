# -*- coding: utf-8 -*-
"""
Authors:    Josef Perktold, Skipper Seabold, Denis A. Engemann
"""
from statsmodels.compat.python import lrange

import numpy as np

from statsmodels.graphics.plottools import rainbow
import statsmodels.graphics.utils as utils


def interaction_plot(x, trace, response, func=np.mean, ax=None, plottype='b',
                     xlabel=None, ylabel=None, colors=None, markers=None,
                     linestyles=None, legendloc='best', legendtitle=None,
                     **kwargs):
    """
    Interaction plot for factor level statistics.

    Note. If categorial factors are supplied levels will be internally
    recoded to integers. This ensures matplotlib compatibility. Uses
    a DataFrame to calculate an `aggregate` statistic for each level of the
    factor or group given by `trace`.

    Parameters
    ----------
    x : array_like
        The `x` factor levels constitute the x-axis. If a `pandas.Series` is
        given its name will be used in `xlabel` if `xlabel` is None.
    trace : array_like
        The `trace` factor levels will be drawn as lines in the plot.
        If `trace` is a `pandas.Series` its name will be used as the
        `legendtitle` if `legendtitle` is None.
    response : array_like
        The reponse or dependent variable. If a `pandas.Series` is given
        its name will be used in `ylabel` if `ylabel` is None.
    func : function
        Anything accepted by `pandas.DataFrame.aggregate`. This is applied to
        the response variable grouped by the trace levels.
    ax : axes, optional
        Matplotlib axes instance
    plottype : str {'line', 'scatter', 'both'}, optional
        The type of plot to return. Can be 'l', 's', or 'b'
    xlabel : str, optional
        Label to use for `x`. Default is 'X'. If `x` is a `pandas.Series` it
        will use the series names.
    ylabel : str, optional
        Label to use for `response`. Default is 'func of response'. If
        `response` is a `pandas.Series` it will use the series names.
    colors : list, optional
        If given, must have length == number of levels in trace.
    markers : list, optional
        If given, must have length == number of levels in trace
    linestyles : list, optional
        If given, must have length == number of levels in trace.
    legendloc : {None, str, int}
        Location passed to the legend command.
    legendtitle : {None, str}
        Title of the legend.
    **kwargs
        These will be passed to the plot command used either plot or scatter.
        If you want to control the overall plotting options, use kwargs.

    Returns
    -------
    Figure
        The figure given by `ax.figure` or a new instance.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> weight = np.random.randint(1,4,size=60)
    >>> duration = np.random.randint(1,3,size=60)
    >>> days = np.log(np.random.randint(1,30, size=60))
    >>> fig = interaction_plot(weight, duration, days,
    ...             colors=['red','blue'], markers=['D','^'], ms=10)
    >>> import matplotlib.pyplot as plt
    >>> plt.show()

    .. plot::

       import numpy as np
       from statsmodels.graphics.factorplots import interaction_plot
       np.random.seed(12345)
       weight = np.random.randint(1,4,size=60)
       duration = np.random.randint(1,3,size=60)
       days = np.log(np.random.randint(1,30, size=60))
       fig = interaction_plot(weight, duration, days,
                   colors=['red','blue'], markers=['D','^'], ms=10)
       import matplotlib.pyplot as plt
       #plt.show()
    """

    from pandas import DataFrame
    fig, ax = utils.create_mpl_ax(ax)

    response_name = ylabel or getattr(response, 'name', 'response')
    ylabel = '%s of %s' % (func.__name__, response_name)
    xlabel = xlabel or getattr(x, 'name', 'X')
    legendtitle = legendtitle or getattr(trace, 'name', 'Trace')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    x_values = x_levels = None
    if isinstance(x[0], str):
        x_levels = [l for l in np.unique(x)]
        x_values = lrange(len(x_levels))
        x = _recode(x, dict(zip(x_levels, x_values)))

    data = DataFrame(dict(x=x, trace=trace, response=response))
    plot_data = data.groupby(['trace', 'x']).aggregate(func).reset_index()

    # return data
    # check plot args
    n_trace = len(plot_data['trace'].unique())

    linestyles = ['-'] * n_trace if linestyles is None else linestyles
    markers = ['.'] * n_trace if markers is None else markers
    colors = rainbow(n_trace) if colors is None else colors

    if len(linestyles) != n_trace:
        raise ValueError("Must be a linestyle for each trace level")
    if len(markers) != n_trace:
        raise ValueError("Must be a marker for each trace level")
    if len(colors) != n_trace:
        raise ValueError("Must be a color for each trace level")

    if plottype == 'both' or plottype == 'b':
        for i, (values, group) in enumerate(plot_data.groupby('trace')):
            # trace label
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i],
                    marker=markers[i], label=label,
                    linestyle=linestyles[i], **kwargs)
    elif plottype == 'line' or plottype == 'l':
        for i, (values, group) in enumerate(plot_data.groupby('trace')):
            # trace label
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i],
                    label=label, linestyle=linestyles[i], **kwargs)
    elif plottype == 'scatter' or plottype == 's':
        for i, (values, group) in enumerate(plot_data.groupby('trace')):
            # trace label
            label = str(group['trace'].values[0])
            ax.scatter(group['x'], group['response'], color=colors[i],
                    label=label, marker=markers[i], **kwargs)

    else:
        raise ValueError("Plot type %s not understood" % plottype)
    ax.legend(loc=legendloc, title=legendtitle)
    ax.margins(.1)

    if all([x_levels, x_values]):
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_levels)
    return fig


def _recode(x, levels):
    """ Recode categorial data to int factor.

    Parameters
    ----------
    x : array_like
        array like object supporting with numpy array methods of categorially
        coded data.
    levels : dict
        mapping of labels to integer-codings

    Returns
    -------
    out : instance numpy.ndarray
    """
    from pandas import Series
    name = None
    index = None

    if isinstance(x, Series):
        name = x.name
        index = x.index
        x = x.values

    if x.dtype.type not in [np.str_, np.object_]:
        raise ValueError('This is not a categorial factor.'
                         ' Array of str type required.')

    elif not isinstance(levels, dict):
        raise ValueError('This is not a valid value for levels.'
                         ' Dict required.')

    elif not (np.unique(x) == np.unique(list(levels.keys()))).all():
        raise ValueError('The levels do not match the array values.')

    else:
        out = np.empty(x.shape[0], dtype=int)
        for level, coding in levels.items():
            out[x == level] = coding

        if name:
            out = Series(out, name=name, index=index)

        return out
