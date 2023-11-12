import numpy as np
import pytest

from statsmodels.datasets import randhie
from statsmodels.graphics.correlation import plot_corr, plot_corr_grid


@pytest.mark.matplotlib
def test_plot_corr(close_figures):
    hie_data = randhie.load_pandas()
    corr_matrix = np.corrcoef(hie_data.data.values.T)

    plot_corr(corr_matrix, xnames=hie_data.names)

    plot_corr(corr_matrix, xnames=[], ynames=hie_data.names)

    plot_corr(corr_matrix, normcolor=True, title='', cmap='jet')


@pytest.mark.matplotlib
def test_plot_corr_grid(close_figures):
    hie_data = randhie.load_pandas()
    corr_matrix = np.corrcoef(hie_data.data.values.T)

    plot_corr_grid([corr_matrix] * 2, xnames=hie_data.names)

    plot_corr_grid([corr_matrix] * 5, xnames=[], ynames=hie_data.names)

    plot_corr_grid([corr_matrix] * 3, normcolor=True, titles='', cmap='jet')
