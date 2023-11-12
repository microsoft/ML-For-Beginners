import matplotlib.pyplot as plt
import numpy as np


def plot_scree(eigenvals, total_var, ncomp=None, x_label='factor'):
    """
    Plot of the ordered eigenvalues and variance explained for the loadings

    Parameters
    ----------
    eigenvals : array_like
        The eigenvalues
    total_var : float
        the total variance (for plotting percent variance explained)
    ncomp : int, optional
        Number of factors to include in the plot.  If None, will
        included the same as the number of maximum possible loadings
    x_label : str
        label of x-axis

    Returns
    -------
    Figure
        Handle to the figure.
    """
    fig = plt.figure()
    ncomp = len(eigenvals) if ncomp is None else ncomp
    vals = eigenvals
    vals = vals[:ncomp]
    #    vals = np.cumsum(vals)

    ax = fig.add_subplot(121)
    ax.plot(np.arange(ncomp), vals[: ncomp], 'b-o')
    ax.autoscale(tight=True)
    xlim = np.array(ax.get_xlim())
    sp = xlim[1] - xlim[0]
    xlim += 0.02 * np.array([-sp, sp])
    ax.set_xticks(np.arange(ncomp))
    ax.set_xlim(xlim)

    ylim = np.array(ax.get_ylim())
    scale = 0.02
    sp = ylim[1] - ylim[0]
    ylim += scale * np.array([-sp, sp])
    ax.set_ylim(ylim)
    ax.set_title('Scree Plot')
    ax.set_ylabel('Eigenvalue')
    ax.set_xlabel(x_label)

    per_variance = vals / total_var
    cumper_variance = np.cumsum(per_variance)
    ax = fig.add_subplot(122)

    ax.plot(np.arange(ncomp), per_variance[: ncomp], 'b-o')
    ax.plot(np.arange(ncomp), cumper_variance[: ncomp], 'g--o')
    ax.autoscale(tight=True)
    xlim = np.array(ax.get_xlim())
    sp = xlim[1] - xlim[0]
    xlim += 0.02 * np.array([-sp, sp])
    ax.set_xticks(np.arange(ncomp))
    ax.set_xlim(xlim)

    ylim = np.array(ax.get_ylim())
    scale = 0.02
    sp = ylim[1] - ylim[0]
    ylim += scale * np.array([-sp, sp])
    ax.set_ylim(ylim)
    ax.set_title('Variance Explained')
    ax.set_ylabel('Proportion')
    ax.set_xlabel(x_label)
    ax.legend(['Proportion', 'Cumulative'], loc=5)
    fig.tight_layout()
    return fig


def plot_loadings(loadings, col_names=None, row_names=None,
                  loading_pairs=None, percent_variance=None,
                  title='Factor patterns'):
    """
    Plot factor loadings in 2-d plots

    Parameters
    ----------
    loadings : array like
        Each column is a component (or factor)
    col_names : a list of strings
        column names of `loadings`
    row_names : a list of strings
        row names of `loadings`
    loading_pairs : None or a list of tuples
        Specify plots. Each tuple (i, j) represent one figure, i and j is
        the loading number for x-axis and y-axis, respectively. If `None`,
        all combinations of the loadings will be plotted.
    percent_variance : array_like
        The percent variance explained by each factor.

    Returns
    -------
    figs : a list of figure handles
    """
    k_var, n_factor = loadings.shape
    if loading_pairs is None:
        loading_pairs = []
        for i in range(n_factor):
            for j in range(i + 1,n_factor):
                loading_pairs.append([i, j])
    if col_names is None:
        col_names = ["factor %d" % i for i in range(n_factor)]
    if row_names is None:
        row_names = ["var %d" % i for i in range(k_var)]
    figs = []
    for item in loading_pairs:
        i = item[0]
        j = item[1]
        fig = plt.figure(figsize=(7, 7))
        figs.append(fig)
        ax = fig.add_subplot(111)
        for k in range(loadings.shape[0]):
            plt.text(loadings[k, i], loadings[k, j],
                     row_names[k], fontsize=12)
        ax.plot(loadings[:, i], loadings[:, j], 'bo')
        ax.set_title(title)
        if percent_variance is not None:
            x_str = '%s (%.1f%%)' % (col_names[i], percent_variance[i])
            y_str = '%s (%.1f%%)' % (col_names[j], percent_variance[j])
            ax.set_xlabel(x_str)
            ax.set_ylabel(y_str)
        else:
            ax.set_xlabel(col_names[i])
            ax.set_ylabel(col_names[j])
        v = 1.05
        xlim = np.array([-v, v])
        ylim = np.array([-v, v])
        ax.plot(xlim, [0, 0], 'k--')
        ax.plot([0, 0], ylim, 'k--')
        ax.set_aspect('equal', 'datalim')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.tight_layout()
    return figs
