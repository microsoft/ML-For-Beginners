'''correlation plots

Author: Josef Perktold
License: BSD-3

example for usage with different options in
statsmodels/sandbox/examples/thirdparty/ex_ratereturn.py

'''
import numpy as np

from . import utils


def plot_corr(dcorr, xnames=None, ynames=None, title=None, normcolor=False,
              ax=None, cmap='RdYlBu_r'):
    """Plot correlation of many variables in a tight color grid.

    Parameters
    ----------
    dcorr : ndarray
        Correlation matrix, square 2-D array.
    xnames : list[str], optional
        Labels for the horizontal axis.  If not given (None), then the
        matplotlib defaults (integers) are used.  If it is an empty list, [],
        then no ticks and labels are added.
    ynames : list[str], optional
        Labels for the vertical axis.  Works the same way as `xnames`.
        If not given, the same names as for `xnames` are re-used.
    title : str, optional
        The figure title. If None, the default ('Correlation Matrix') is used.
        If ``title=''``, then no title is added.
    normcolor : bool or tuple of scalars, optional
        If False (default), then the color coding range corresponds to the
        range of `dcorr`.  If True, then the color range is normalized to
        (-1, 1).  If this is a tuple of two numbers, then they define the range
        for the color bar.
    ax : AxesSubplot, optional
        If `ax` is None, then a figure is created. If an axis instance is
        given, then only the main plot but not the colorbar is created.
    cmap : str or Matplotlib Colormap instance, optional
        The colormap for the plot.  Can be any valid Matplotlib Colormap
        instance or name.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.graphics.api as smg

    >>> hie_data = sm.datasets.randhie.load_pandas()
    >>> corr_matrix = np.corrcoef(hie_data.data.T)
    >>> smg.plot_corr(corr_matrix, xnames=hie_data.names)
    >>> plt.show()

    .. plot:: plots/graphics_correlation_plot_corr.py
    """
    if ax is None:
        create_colorbar = True
    else:
        create_colorbar = False

    fig, ax = utils.create_mpl_ax(ax)

    nvars = dcorr.shape[0]

    if ynames is None:
        ynames = xnames
    if title is None:
        title = 'Correlation Matrix'
    if isinstance(normcolor, tuple):
        vmin, vmax = normcolor
    elif normcolor:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = None, None

    axim = ax.imshow(dcorr, cmap=cmap, interpolation='nearest',
                     extent=(0,nvars,0,nvars), vmin=vmin, vmax=vmax)

    # create list of label positions
    labelPos = np.arange(0, nvars) + 0.5

    if isinstance(ynames, list) and len(ynames) == 0:
        ax.set_yticks([])
    elif ynames is not None:
        ax.set_yticks(labelPos)
        ax.set_yticks(labelPos[:-1]+0.5, minor=True)
        ax.set_yticklabels(ynames[::-1], fontsize='small',
                           horizontalalignment='right')

    if isinstance(xnames, list) and len(xnames) == 0:
        ax.set_xticks([])
    elif xnames is not None:
        ax.set_xticks(labelPos)
        ax.set_xticks(labelPos[:-1]+0.5, minor=True)
        ax.set_xticklabels(xnames, fontsize='small', rotation=45,
                           horizontalalignment='right')


    if not title == '':
        ax.set_title(title)

    if create_colorbar:
        fig.colorbar(axim, use_gridspec=True)
    fig.tight_layout()

    ax.tick_params(which='minor', length=0)
    ax.tick_params(direction='out', top=False, right=False)
    try:
        ax.grid(True, which='minor', linestyle='-', color='w', lw=1)
    except AttributeError:
        # Seems to fail for axes created with AxesGrid.  MPL bug?
        pass

    return fig


def plot_corr_grid(dcorrs, titles=None, ncols=None, normcolor=False, xnames=None,
                   ynames=None, fig=None, cmap='RdYlBu_r'):
    """
    Create a grid of correlation plots.

    The individual correlation plots are assumed to all have the same
    variables, axis labels can be specified only once.

    Parameters
    ----------
    dcorrs : list or iterable of ndarrays
        List of correlation matrices.
    titles : list[str], optional
        List of titles for the subplots.  By default no title are shown.
    ncols : int, optional
        Number of columns in the subplot grid.  If not given, the number of
        columns is determined automatically.
    normcolor : bool or tuple, optional
        If False (default), then the color coding range corresponds to the
        range of `dcorr`.  If True, then the color range is normalized to
        (-1, 1).  If this is a tuple of two numbers, then they define the range
        for the color bar.
    xnames : list[str], optional
        Labels for the horizontal axis.  If not given (None), then the
        matplotlib defaults (integers) are used.  If it is an empty list, [],
        then no ticks and labels are added.
    ynames : list[str], optional
        Labels for the vertical axis.  Works the same way as `xnames`.
        If not given, the same names as for `xnames` are re-used.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.
    cmap : str or Matplotlib Colormap instance, optional
        The colormap for the plot.  Can be any valid Matplotlib Colormap
        instance or name.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    In this example we just reuse the same correlation matrix several times.
    Of course in reality one would show a different correlation (measuring a
    another type of correlation, for example Pearson (linear) and Spearman,
    Kendall (nonlinear) correlations) for the same variables.

    >>> hie_data = sm.datasets.randhie.load_pandas()
    >>> corr_matrix = np.corrcoef(hie_data.data.T)
    >>> sm.graphics.plot_corr_grid([corr_matrix] * 8, xnames=hie_data.names)
    >>> plt.show()

    .. plot:: plots/graphics_correlation_plot_corr_grid.py
    """
    if ynames is None:
        ynames = xnames

    if not titles:
        titles = ['']*len(dcorrs)

    n_plots = len(dcorrs)
    if ncols is not None:
        nrows = int(np.ceil(n_plots / float(ncols)))
    else:
        # Determine number of rows and columns, square if possible, otherwise
        # prefer a wide (more columns) over a high layout.
        if n_plots < 4:
            nrows, ncols = 1, n_plots
        else:
            nrows = int(np.sqrt(n_plots))
            ncols = int(np.ceil(n_plots / float(nrows)))

    # Create a figure with the correct size
    aspect = min(ncols / float(nrows), 1.8)
    vsize = np.sqrt(nrows) * 5
    fig = utils.create_mpl_fig(fig, figsize=(vsize * aspect + 1, vsize))

    for i, c in enumerate(dcorrs):
        ax = fig.add_subplot(nrows, ncols, i+1)
        # Ensure to only plot labels on bottom row and left column
        _xnames = xnames if nrows * ncols - (i+1) < ncols else []
        _ynames = ynames if (i+1) % ncols == 1 else []
        plot_corr(c, xnames=_xnames, ynames=_ynames, title=titles[i],
                  normcolor=normcolor, ax=ax, cmap=cmap)

    # Adjust figure margins and add a colorbar
    fig.subplots_adjust(bottom=0.1, left=0.09, right=0.9, top=0.9)
    cax = fig.add_axes([0.92, 0.1, 0.025, 0.8])
    fig.colorbar(fig.axes[0].images[0], cax=cax)

    return fig
