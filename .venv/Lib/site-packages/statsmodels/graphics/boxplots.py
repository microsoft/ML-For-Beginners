"""Variations on boxplots."""

# Author: Ralf Gommers
# Based on code by Flavio Coelho and Teemu Ikonen.

import numpy as np
from scipy.stats import gaussian_kde

from . import utils

__all__ = ['violinplot', 'beanplot']


def violinplot(data, ax=None, labels=None, positions=None, side='both',
               show_boxplot=True, plot_opts=None):
    """
    Make a violin plot of each dataset in the `data` sequence.

    A violin plot is a boxplot combined with a kernel density estimate of the
    probability density function per point.

    Parameters
    ----------
    data : sequence[array_like]
        Data arrays, one array per value in `positions`.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    labels : list[str], optional
        Tick labels for the horizontal axis.  If not given, integers
        ``1..len(data)`` are used.
    positions : array_like, optional
        Position array, used as the horizontal axis of the plot.  If not given,
        spacing of the violins will be equidistant.
    side : {'both', 'left', 'right'}, optional
        How to plot the violin.  Default is 'both'.  The 'left', 'right'
        options can be used to create asymmetric violin plots.
    show_boxplot : bool, optional
        Whether or not to show normal box plots on top of the violins.
        Default is True.
    plot_opts : dict, optional
        A dictionary with plotting options.  Any of the following can be
        provided, if not present in `plot_opts` the defaults will be used::

          - 'violin_fc', MPL color.  Fill color for violins.  Default is 'y'.
          - 'violin_ec', MPL color.  Edge color for violins.  Default is 'k'.
          - 'violin_lw', scalar.  Edge linewidth for violins.  Default is 1.
          - 'violin_alpha', float.  Transparancy of violins.  Default is 0.5.
          - 'cutoff', bool.  If True, limit violin range to data range.
                Default is False.
          - 'cutoff_val', scalar.  Where to cut off violins if `cutoff` is
                True.  Default is 1.5 standard deviations.
          - 'cutoff_type', {'std', 'abs'}.  Whether cutoff value is absolute,
                or in standard deviations.  Default is 'std'.
          - 'violin_width' : float.  Relative width of violins.  Max available
                space is 1, default is 0.8.
          - 'label_fontsize', MPL fontsize.  Adjusts fontsize only if given.
          - 'label_rotation', scalar.  Adjusts label rotation only if given.
                Specify in degrees.
          - 'bw_factor', Adjusts the scipy gaussian_kde kernel. default: None.
                Options for scalar or callable.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    beanplot : Bean plot, builds on `violinplot`.
    matplotlib.pyplot.boxplot : Standard boxplot.

    Notes
    -----
    The appearance of violins can be customized with `plot_opts`.  If
    customization of boxplot elements is required, set `show_boxplot` to False
    and plot it on top of the violins by calling the Matplotlib `boxplot`
    function directly.  For example::

        violinplot(data, ax=ax, show_boxplot=False)
        ax.boxplot(data, sym='cv', whis=2.5)

    It can happen that the axis labels or tick labels fall outside the plot
    area, especially with rotated labels on the horizontal axis.  With
    Matplotlib 1.1 or higher, this can easily be fixed by calling
    ``ax.tight_layout()``.  With older Matplotlib one has to use ``plt.rc`` or
    ``plt.rcParams`` to fix this, for example::

        plt.rc('figure.subplot', bottom=0.25)
        violinplot(data, ax=ax)

    References
    ----------
    J.L. Hintze and R.D. Nelson, "Violin Plots: A Box Plot-Density Trace
    Synergism", The American Statistician, Vol. 52, pp.181-84, 1998.

    Examples
    --------
    We use the American National Election Survey 1996 dataset, which has Party
    Identification of respondents as independent variable and (among other
    data) age as dependent variable.

    >>> data = sm.datasets.anes96.load_pandas()
    >>> party_ID = np.arange(7)
    >>> labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",
    ...           "Independent-Indpendent", "Independent-Republican",
    ...           "Weak Republican", "Strong Republican"]

    Group age by party ID, and create a violin plot with it:

    >>> plt.rcParams['figure.subplot.bottom'] = 0.23  # keep labels visible
    >>> age = [data.exog['age'][data.endog == id] for id in party_ID]
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> sm.graphics.violinplot(age, ax=ax, labels=labels,
    ...                        plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
    ...                                   'label_fontsize':'small',
    ...                                   'label_rotation':30})
    >>> ax.set_xlabel("Party identification of respondent.")
    >>> ax.set_ylabel("Age")
    >>> plt.show()

    .. plot:: plots/graphics_boxplot_violinplot.py
    """
    plot_opts = {} if plot_opts is None else plot_opts
    if max([np.size(arr) for arr in data]) == 0:
        msg = "No Data to make Violin: Try again!"
        raise ValueError(msg)

    fig, ax = utils.create_mpl_ax(ax)

    data = list(map(np.asarray, data))
    if positions is None:
        positions = np.arange(len(data)) + 1

    # Determine available horizontal space for each individual violin.
    pos_span = np.max(positions) - np.min(positions)
    width = np.min([0.15 * np.max([pos_span, 1.]),
                    plot_opts.get('violin_width', 0.8) / 2.])

    # Plot violins.
    for pos_data, pos in zip(data, positions):
        _single_violin(ax, pos, pos_data, width, side, plot_opts)

    if show_boxplot:
        ax.boxplot(data, notch=1, positions=positions, vert=1)

    # Set ticks and tick labels of horizontal axis.
    _set_ticks_labels(ax, data, labels, positions, plot_opts)

    return fig


def _single_violin(ax, pos, pos_data, width, side, plot_opts):
    """"""
    bw_factor = plot_opts.get('bw_factor', None)

    def _violin_range(pos_data, plot_opts):
        """Return array with correct range, with which violins can be plotted."""
        cutoff = plot_opts.get('cutoff', False)
        cutoff_type = plot_opts.get('cutoff_type', 'std')
        cutoff_val = plot_opts.get('cutoff_val', 1.5)

        s = 0.0
        if not cutoff:
            if cutoff_type == 'std':
                s = cutoff_val * np.std(pos_data)
            else:
                s = cutoff_val

        x_lower = kde.dataset.min() - s
        x_upper = kde.dataset.max() + s
        return np.linspace(x_lower, x_upper, 100)

    pos_data = np.asarray(pos_data)
    # Kernel density estimate for data at this position.
    kde = gaussian_kde(pos_data, bw_method=bw_factor)

    # Create violin for pos, scaled to the available space.
    xvals = _violin_range(pos_data, plot_opts)
    violin = kde.evaluate(xvals)
    violin = width * violin / violin.max()

    if side == 'both':
        envelope_l, envelope_r = (-violin + pos, violin + pos)
    elif side == 'right':
        envelope_l, envelope_r = (pos, violin + pos)
    elif side == 'left':
        envelope_l, envelope_r = (-violin + pos, pos)
    else:
        msg = "`side` parameter should be one of {'left', 'right', 'both'}."
        raise ValueError(msg)

    # Draw the violin.
    ax.fill_betweenx(xvals, envelope_l, envelope_r,
                     facecolor=plot_opts.get('violin_fc', '#66c2a5'),
                     edgecolor=plot_opts.get('violin_ec', 'k'),
                     lw=plot_opts.get('violin_lw', 1),
                     alpha=plot_opts.get('violin_alpha', 0.5))

    return xvals, violin


def _set_ticks_labels(ax, data, labels, positions, plot_opts):
    """Set ticks and labels on horizontal axis."""

    # Set xticks and limits.
    ax.set_xlim([np.min(positions) - 0.5, np.max(positions) + 0.5])
    ax.set_xticks(positions)

    label_fontsize = plot_opts.get('label_fontsize')
    label_rotation = plot_opts.get('label_rotation')
    if label_fontsize or label_rotation:
        from matplotlib.artist import setp

    if labels is not None:
        if not len(labels) == len(data):
            msg = "Length of `labels` should equal length of `data`."
            raise ValueError(msg)

        xticknames = ax.set_xticklabels(labels)
        if label_fontsize:
            setp(xticknames, fontsize=label_fontsize)

        if label_rotation:
            setp(xticknames, rotation=label_rotation)

    return


def beanplot(data, ax=None, labels=None, positions=None, side='both',
             jitter=False, plot_opts={}):
    """
    Bean plot of each dataset in a sequence.

    A bean plot is a combination of a `violinplot` (kernel density estimate of
    the probability density function per point) with a line-scatter plot of all
    individual data points.

    Parameters
    ----------
    data : sequence[array_like]
        Data arrays, one array per value in `positions`.
    ax : AxesSubplot
        If given, this subplot is used to plot in instead of a new figure being
        created.
    labels : list[str], optional
        Tick labels for the horizontal axis.  If not given, integers
        ``1..len(data)`` are used.
    positions : array_like, optional
        Position array, used as the horizontal axis of the plot.  If not given,
        spacing of the violins will be equidistant.
    side : {'both', 'left', 'right'}, optional
        How to plot the violin.  Default is 'both'.  The 'left', 'right'
        options can be used to create asymmetric violin plots.
    jitter : bool, optional
        If True, jitter markers within violin instead of plotting regular lines
        around the center.  This can be useful if the data is very dense.
    plot_opts : dict, optional
        A dictionary with plotting options.  All the options for `violinplot`
        can be specified, they will simply be passed to `violinplot`.  Options
        specific to `beanplot` are:

          - 'violin_width' : float.  Relative width of violins.  Max available
                space is 1, default is 0.8.
          - 'bean_color', MPL color.  Color of bean plot lines.  Default is 'k'.
                Also used for jitter marker edge color if `jitter` is True.
          - 'bean_size', scalar.  Line length as a fraction of maximum length.
                Default is 0.5.
          - 'bean_lw', scalar.  Linewidth, default is 0.5.
          - 'bean_show_mean', bool.  If True (default), show mean as a line.
          - 'bean_show_median', bool.  If True (default), show median as a
                marker.
          - 'bean_mean_color', MPL color.  Color of mean line.  Default is 'b'.
          - 'bean_mean_lw', scalar.  Linewidth of mean line, default is 2.
          - 'bean_mean_size', scalar.  Line length as a fraction of maximum length.
                Default is 0.5.
          - 'bean_median_color', MPL color.  Color of median marker.  Default
                is 'r'.
          - 'bean_median_marker', MPL marker.  Marker type, default is '+'.
          - 'jitter_marker', MPL marker.  Marker type for ``jitter=True``.
                Default is 'o'.
          - 'jitter_marker_size', int.  Marker size.  Default is 4.
          - 'jitter_fc', MPL color.  Jitter marker face color.  Default is None.
          - 'bean_legend_text', str.  If given, add a legend with given text.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    violinplot : Violin plot, also used internally in `beanplot`.
    matplotlib.pyplot.boxplot : Standard boxplot.

    References
    ----------
    P. Kampstra, "Beanplot: A Boxplot Alternative for Visual Comparison of
    Distributions", J. Stat. Soft., Vol. 28, pp. 1-9, 2008.

    Examples
    --------
    We use the American National Election Survey 1996 dataset, which has Party
    Identification of respondents as independent variable and (among other
    data) age as dependent variable.

    >>> data = sm.datasets.anes96.load_pandas()
    >>> party_ID = np.arange(7)
    >>> labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",
    ...           "Independent-Indpendent", "Independent-Republican",
    ...           "Weak Republican", "Strong Republican"]

    Group age by party ID, and create a violin plot with it:

    >>> plt.rcParams['figure.subplot.bottom'] = 0.23  # keep labels visible
    >>> age = [data.exog['age'][data.endog == id] for id in party_ID]
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> sm.graphics.beanplot(age, ax=ax, labels=labels,
    ...                      plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
    ...                                 'label_fontsize':'small',
    ...                                 'label_rotation':30})
    >>> ax.set_xlabel("Party identification of respondent.")
    >>> ax.set_ylabel("Age")
    >>> plt.show()

    .. plot:: plots/graphics_boxplot_beanplot.py
    """
    fig, ax = utils.create_mpl_ax(ax)

    data = list(map(np.asarray, data))
    if positions is None:
        positions = np.arange(len(data)) + 1

    # Determine available horizontal space for each individual violin.
    pos_span = np.max(positions) - np.min(positions)
    violin_width = np.min([0.15 * np.max([pos_span, 1.]),
                    plot_opts.get('violin_width', 0.8) / 2.])
    bean_width = np.min([0.15 * np.max([pos_span, 1.]),
                    plot_opts.get('bean_size', 0.5) / 2.])
    bean_mean_width = np.min([0.15 * np.max([pos_span, 1.]),
                    plot_opts.get('bean_mean_size', 0.5) / 2.])

    legend_txt = plot_opts.get('bean_legend_text', None)
    for pos_data, pos in zip(data, positions):
        # Draw violins.
        xvals, violin = _single_violin(ax, pos, pos_data, violin_width, side, plot_opts)

        if jitter:
            # Draw data points at random coordinates within violin envelope.
            jitter_coord = pos + _jitter_envelope(pos_data, xvals, violin, side)
            ax.plot(jitter_coord, pos_data, ls='',
                    marker=plot_opts.get('jitter_marker', 'o'),
                    ms=plot_opts.get('jitter_marker_size', 4),
                    mec=plot_opts.get('bean_color', 'k'),
                    mew=1, mfc=plot_opts.get('jitter_fc', 'none'),
                    label=legend_txt)
        else:
            # Draw bean lines.
            ax.hlines(pos_data, pos - bean_width, pos + bean_width,
                      lw=plot_opts.get('bean_lw', 0.5),
                      color=plot_opts.get('bean_color', 'k'),
                      label=legend_txt)

        # Show legend if required.
        if legend_txt is not None:
            _show_legend(ax)
            legend_txt = None  # ensure we get one entry per call to beanplot

        # Draw mean line.
        if plot_opts.get('bean_show_mean', True):
            ax.hlines(np.mean(pos_data), pos - bean_mean_width, pos + bean_mean_width,
                      lw=plot_opts.get('bean_mean_lw', 2.),
                      color=plot_opts.get('bean_mean_color', 'b'))

        # Draw median marker.
        if plot_opts.get('bean_show_median', True):
            ax.plot(pos, np.median(pos_data),
                    marker=plot_opts.get('bean_median_marker', '+'),
                    color=plot_opts.get('bean_median_color', 'r'))

    # Set ticks and tick labels of horizontal axis.
    _set_ticks_labels(ax, data, labels, positions, plot_opts)

    return fig


def _jitter_envelope(pos_data, xvals, violin, side):
    """Determine envelope for jitter markers."""
    if side == 'both':
        low, high = (-1., 1.)
    elif side == 'right':
        low, high = (0, 1.)
    elif side == 'left':
        low, high = (-1., 0)
    else:
        raise ValueError("`side` input incorrect: %s" % side)

    jitter_envelope = np.interp(pos_data, xvals, violin)
    jitter_coord = jitter_envelope * np.random.uniform(low=low, high=high,
                                                       size=pos_data.size)

    return jitter_coord


def _show_legend(ax):
    """Utility function to show legend."""
    leg = ax.legend(loc=1, shadow=True, fancybox=True, labelspacing=0.2,
                    borderpad=0.15)
    ltext  = leg.get_texts()
    llines = leg.get_lines()
    frame  = leg.get_frame()

    from matplotlib.artist import setp
    setp(ltext, fontsize='small')
    setp(llines, linewidth=1)
