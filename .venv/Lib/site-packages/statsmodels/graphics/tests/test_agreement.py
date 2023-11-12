import numpy as np
import pandas as pd
import pytest

from statsmodels.graphics.agreement import mean_diff_plot

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


@pytest.mark.matplotlib
def test_mean_diff_plot(close_figures):

    # Seed the random number generator.
    # This ensures that the results below are reproducible.
    np.random.seed(11111)
    m1 = np.random.random(20)
    m2 = np.random.random(20)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # basic test.
    mean_diff_plot(m1, m2, ax=ax)

    # Test with pandas Series.
    p1 = pd.Series(m1)
    p2 = pd.Series(m2)
    mean_diff_plot(p1, p2)

    # Test plotting on assigned axis.
    fig, ax = plt.subplots(2)
    mean_diff_plot(m1, m2, ax=ax[0])

    # Test the setting of confidence intervals.
    mean_diff_plot(m1, m2, sd_limit=0)

    # Test asethetic controls.
    mean_diff_plot(m1, m2, scatter_kwds={'color': 'green', 's': 10})

    mean_diff_plot(m1, m2, mean_line_kwds={'color': 'green', 'lw': 5})

    mean_diff_plot(m1, m2, limit_lines_kwds={'color': 'green',
                                             'lw': 5,
                                             'ls': 'dotted'})
