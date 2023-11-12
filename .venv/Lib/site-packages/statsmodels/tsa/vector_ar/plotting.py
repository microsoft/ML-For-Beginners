from statsmodels.compat.python import lrange

import numpy as np

import statsmodels.tsa.vector_ar.util as util


class MPLConfigurator:

    def __init__(self):
        self._inverse_actions = []

    def revert(self):
        for action in self._inverse_actions:
            action()

    def set_fontsize(self, size):
        import matplotlib as mpl
        old_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = size

        def revert():
            mpl.rcParams['font.size'] = old_size

        self._inverse_actions.append(revert)


#-------------------------------------------------------------------------------
# Plotting functions

def plot_mts(Y, names=None, index=None):
    """
    Plot multiple time series
    """
    import matplotlib.pyplot as plt

    k = Y.shape[1]
    rows, cols = k, 1

    fig = plt.figure(figsize=(10, 10))

    for j in range(k):
        ts = Y[:, j]

        ax = fig.add_subplot(rows, cols, j+1)
        if index is not None:
            ax.plot(index, ts)
        else:
            ax.plot(ts)

        if names is not None:
            ax.set_title(names[j])

    return fig


def plot_var_forc(prior, forc, err_upper, err_lower,
                  index=None, names=None, plot_stderr=True,
                  legend_options=None):
    import matplotlib.pyplot as plt

    n, k = prior.shape
    rows, cols = k, 1

    fig = plt.figure(figsize=(10, 10))

    prange = np.arange(n)
    rng_f = np.arange(n - 1, n + len(forc))
    rng_err = np.arange(n, n + len(forc))

    for j in range(k):
        ax = plt.subplot(rows, cols, j+1)

        p1 = ax.plot(prange, prior[:, j], 'k', label='Observed')
        p2 = ax.plot(rng_f, np.r_[prior[-1:, j], forc[:, j]], 'k--',
                     label='Forecast')

        if plot_stderr:
            p3 = ax.plot(rng_err, err_upper[:, j], 'k-.',
                         label='Forc 2 STD err')
            ax.plot(rng_err, err_lower[:, j], 'k-.')

        if names is not None:
            ax.set_title(names[j])

        if legend_options is None:
            legend_options = {"loc": "upper right"}
        ax.legend(**legend_options)
    return fig


def plot_with_error(y, error, x=None, axes=None, value_fmt='k',
                    error_fmt='k--', alpha=0.05, stderr_type = 'asym'):
    """
    Make plot with optional error bars

    Parameters
    ----------
    y :
    error : array or None
    """
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()

    x = x if x is not None else lrange(len(y))
    plot_action = lambda y, fmt: axes.plot(x, y, fmt)
    plot_action(y, value_fmt)

    #changed this
    if error is not None:
        if stderr_type == 'asym':
            q = util.norm_signif_level(alpha)
            plot_action(y - q * error, error_fmt)
            plot_action(y + q * error, error_fmt)
        if stderr_type in ('mc','sz1','sz2','sz3'):
            plot_action(error[0], error_fmt)
            plot_action(error[1], error_fmt)


def plot_full_acorr(acorr, fontsize=8, linewidth=8, xlabel=None,
                    err_bound=None):
    """

    Parameters
    ----------
    """
    import matplotlib.pyplot as plt

    config = MPLConfigurator()
    config.set_fontsize(fontsize)

    k = acorr.shape[1]
    fig, axes = plt.subplots(k, k, figsize=(10, 10), squeeze=False)

    for i in range(k):
        for j in range(k):
            ax = axes[i][j]
            acorr_plot(acorr[:, i, j], linewidth=linewidth,
                       xlabel=xlabel, ax=ax)

            if err_bound is not None:
                ax.axhline(err_bound, color='k', linestyle='--')
                ax.axhline(-err_bound, color='k', linestyle='--')

    adjust_subplots()
    config.revert()

    return fig


def acorr_plot(acorr, linewidth=8, xlabel=None, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if xlabel is None:
        xlabel = np.arange(len(acorr))

    ax.vlines(xlabel, [0], acorr, lw=linewidth)

    ax.axhline(0, color='k')
    ax.set_ylim([-1, 1])

    # hack?
    ax.set_xlim([-1, xlabel[-1] + 1])


def plot_acorr_with_error():
    raise NotImplementedError


def adjust_subplots(**kwds):
    import matplotlib.pyplot as plt

    passed_kwds = dict(bottom=0.05, top=0.925,
                       left=0.05, right=0.95,
                       hspace=0.2)
    passed_kwds.update(kwds)
    plt.subplots_adjust(**passed_kwds)


#-------------------------------------------------------------------------------
# Multiple impulse response (cum_effects, etc.) cplots

def irf_grid_plot(values, stderr, impcol, rescol, names, title,
                  signif=0.05, hlines=None, subplot_params=None,
                  plot_params=None, figsize=(10,10), stderr_type='asym'):
    """
    Reusable function to make flexible grid plots of impulse responses and
    comulative effects

    values : (T + 1) x k x k
    stderr : T x k x k
    hlines : k x k
    """
    import matplotlib.pyplot as plt

    if subplot_params is None:
        subplot_params = {}
    if plot_params is None:
        plot_params = {}

    nrows, ncols, to_plot = _get_irf_plot_config(names, impcol, rescol)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             squeeze=False, figsize=figsize)

    # fill out space
    adjust_subplots()

    fig.suptitle(title, fontsize=14)

    subtitle_temp = r'%s$\rightarrow$%s'

    k = len(names)

    rng = lrange(len(values))
    for (j, i, ai, aj) in to_plot:
        ax = axes[ai][aj]

        # HACK?
        if stderr is not None:
            if stderr_type == 'asym':
                sig = np.sqrt(stderr[:, j * k + i, j * k + i])
                plot_with_error(values[:, i, j], sig, x=rng, axes=ax,
                                alpha=signif, value_fmt='b', stderr_type=stderr_type)
            if stderr_type in ('mc','sz1','sz2','sz3'):
                errs = stderr[0][:, i, j], stderr[1][:, i, j]
                plot_with_error(values[:, i, j], errs, x=rng, axes=ax,
                                alpha=signif, value_fmt='b', stderr_type=stderr_type)
        else:
            plot_with_error(values[:, i, j], None, x=rng, axes=ax,
                            value_fmt='b')

        ax.axhline(0, color='k')

        if hlines is not None:
            ax.axhline(hlines[i,j], color='k')

        sz = subplot_params.get('fontsize', 12)
        ax.set_title(subtitle_temp % (names[j], names[i]), fontsize=sz)

    return fig


def _get_irf_plot_config(names, impcol, rescol):
    nrows = ncols = k = len(names)
    if impcol is not None and rescol is not None:
        # plot one impulse-response pair
        nrows = ncols = 1
        j = util.get_index(names, impcol)
        i = util.get_index(names, rescol)
        to_plot = [(j, i, 0, 0)]
    elif impcol is not None:
        # plot impacts of impulse in one variable
        ncols = 1
        j = util.get_index(names, impcol)
        to_plot = [(j, i, i, 0) for i in range(k)]
    elif rescol is not None:
        # plot only things having impact on particular variable
        ncols = 1
        i = util.get_index(names, rescol)
        to_plot = [(j, i, j, 0) for j in range(k)]
    else:
        # plot everything
        to_plot = [(j, i, i, j) for i in range(k) for j in range(k)]

    return nrows, ncols, to_plot

#-------------------------------------------------------------------------------
# Forecast error variance decomposition
