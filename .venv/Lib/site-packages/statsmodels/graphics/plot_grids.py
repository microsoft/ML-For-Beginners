'''create scatterplot with confidence ellipsis

Author: Josef Perktold
License: BSD-3

TODO: update script to use sharex, sharey, and visible=False
    see https://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label
    for sharex I need to have the ax of the last_row when editing the earlier
    rows. Or you axes_grid1, imagegrid
    http://matplotlib.sourceforge.net/mpl_toolkits/axes_grid/users/overview.html
'''


import numpy as np
from scipy import stats

from . import utils

__all__ = ['scatter_ellipse']


def _make_ellipse(mean, cov, ax, level=0.95, color=None):
    """Support function for scatter_ellipse."""
    from matplotlib.patches import Ellipse

    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180 * angle / np.pi # convert to degrees
    v = 2 * np.sqrt(v * stats.chi2.ppf(level, 2)) #get size corresponding to level
    ell = Ellipse(mean[:2], v[0], v[1], 180 + angle, facecolor='none',
                  edgecolor=color,
                  #ls='dashed',  #for debugging
                  lw=1.5)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)


def scatter_ellipse(data, level=0.9, varnames=None, ell_kwds=None,
                    plot_kwds=None, add_titles=False, keep_ticks=False,
                    fig=None):
    """Create a grid of scatter plots with confidence ellipses.

    ell_kwds, plot_kdes not used yet

    looks ok with 5 or 6 variables, too crowded with 8, too empty with 1

    Parameters
    ----------
    data : array_like
        Input data.
    level : scalar, optional
        Default is 0.9.
    varnames : list[str], optional
        Variable names.  Used for y-axis labels, and if `add_titles` is True
        also for titles.  If not given, integers 1..data.shape[1] are used.
    ell_kwds : dict, optional
        UNUSED
    plot_kwds : dict, optional
        UNUSED
    add_titles : bool, optional
        Whether or not to add titles to each subplot.  Default is False.
        Titles are constructed from `varnames`.
    keep_ticks : bool, optional
        If False (default), remove all axis ticks.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise `fig` itself.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> from statsmodels.graphics.plot_grids import scatter_ellipse
    >>> data = sm.datasets.statecrime.load_pandas().data
    >>> fig = plt.figure(figsize=(8,8))
    >>> scatter_ellipse(data, varnames=data.columns, fig=fig)
    >>> plt.show()

    ..plot :: plots/graphics_correlation_plot_corr_grid.py
    """
    fig = utils.create_mpl_fig(fig)
    import matplotlib.ticker as mticker

    data = np.asanyarray(data)  #needs mean and cov
    nvars = data.shape[1]
    if varnames is None:
        #assuming single digit, nvars<=10  else use 'var%2d'
        varnames = ['var%d' % i for i in range(nvars)]

    plot_kwds_ = dict(ls='none', marker='.', color='k', alpha=0.5)
    if plot_kwds:
        plot_kwds_.update(plot_kwds)

    ell_kwds_= dict(color='k')
    if ell_kwds:
        ell_kwds_.update(ell_kwds)

    dmean = data.mean(0)
    dcov = np.cov(data, rowvar=0)

    for i in range(1, nvars):
        #print '---'
        ax_last=None
        for j in range(i):
            #print i,j, i*(nvars-1)+j+1
            ax = fig.add_subplot(nvars-1, nvars-1, (i-1)*(nvars-1)+j+1)
##                                 #sharey=ax_last) #sharey does not allow empty ticks?
##            if j == 0:
##                print 'new ax_last', j
##                ax_last = ax
##                ax.set_ylabel(varnames[i])
            #TODO: make sure we have same xlim and ylim

            formatter = mticker.FormatStrFormatter('% 3.1f')
            ax.yaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(formatter)

            idx = np.array([j,i])
            ax.plot(*data[:,idx].T, **plot_kwds_)

            if np.isscalar(level):
                level = [level]
            for alpha in level:
                _make_ellipse(dmean[idx], dcov[idx[:,None], idx], ax, level=alpha,
                         **ell_kwds_)

            if add_titles:
                ax.set_title('%s-%s' % (varnames[i], varnames[j]))
            if not ax.is_first_col():
                if not keep_ticks:
                    ax.set_yticks([])
                else:
                    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
            else:
                ax.set_ylabel(varnames[i])
            if ax.is_last_row():
                ax.set_xlabel(varnames[j])
            else:
                if not keep_ticks:
                    ax.set_xticks([])
                else:
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))

            dcorr = np.corrcoef(data, rowvar=0)
            dc = dcorr[idx[:,None], idx]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
##            xt = xlim[0] + 0.1 * (xlim[1] - xlim[0])
##            yt = ylim[0] + 0.1 * (ylim[1] - ylim[0])
##            if dc[1,0] < 0 :
##                yt = ylim[0] + 0.1 * (ylim[1] - ylim[0])
##            else:
##                yt = ylim[1] - 0.2 * (ylim[1] - ylim[0])
            yrangeq = ylim[0] + 0.4 * (ylim[1] - ylim[0])
            if dc[1,0] < -0.25 or (dc[1,0] < 0.25 and dmean[idx][1] > yrangeq):
                yt = ylim[0] + 0.1 * (ylim[1] - ylim[0])
            else:
                yt = ylim[1] - 0.2 * (ylim[1] - ylim[0])
            xt = xlim[0] + 0.1 * (xlim[1] - xlim[0])
            ax.text(xt, yt, '$\\rho=%0.2f$'% dc[1,0])

    for ax in fig.axes:
        if ax.is_last_row(): # or ax.is_first_col():
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        if ax.is_first_col():
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3))

    return fig
