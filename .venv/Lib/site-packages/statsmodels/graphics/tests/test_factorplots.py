import numpy as np
from numpy.testing import assert_equal, assert_raises
from pandas import Series
import pytest

from statsmodels.graphics.factorplots import _recode, interaction_plot

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


class TestInteractionPlot:

    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        cls.weight = np.random.randint(1,4,size=60)
        cls.duration = np.random.randint(1,3,size=60)
        cls.days = np.log(np.random.randint(1,30, size=60))

    @pytest.mark.matplotlib
    def test_plot_both(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days,
                 colors=['red','blue'], markers=['D','^'], ms=10)

    @pytest.mark.matplotlib
    def test_plot_rainbow(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days,
                 markers=['D','^'], ms=10)

    @pytest.mark.matplotlib
    @pytest.mark.parametrize('astype', ['str', 'int'])
    def test_plot_pandas(self, astype, close_figures):
        weight = Series(self.weight, name='Weight').astype(astype)
        duration = Series(self.duration, name='Duration')
        days = Series(self.days, name='Days')
        fig = interaction_plot(weight, duration, days,
                               markers=['D', '^'], ms=10)
        ax = fig.axes[0]
        trace = ax.get_legend().get_title().get_text()
        assert_equal(trace, 'Duration')
        assert_equal(ax.get_ylabel(), 'mean of Days')
        assert_equal(ax.get_xlabel(), 'Weight')

    @pytest.mark.matplotlib
    def test_formatting(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days, colors=['r','g'], linestyles=['--','-.'])
        assert_equal(isinstance(fig, plt.Figure), True)

    @pytest.mark.matplotlib
    def test_formatting_errors(self, close_figures):
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, markers=['D'])
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, colors=['b','r','g'])
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, linestyles=['--','-.',':'])

    @pytest.mark.matplotlib
    def test_plottype(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days, plottype='line')
        assert_equal(isinstance(fig, plt.Figure), True)
        fig = interaction_plot(self.weight, self.duration, self.days, plottype='scatter')
        assert_equal(isinstance(fig, plt.Figure), True)
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, plottype='unknown')

    def test_recode_series(self):
        series = Series(['a', 'b'] * 10, index=np.arange(0, 40, 2),
                        name='index_test')
        series_ = _recode(series, {'a': 0, 'b': 1})
        assert_equal(series_.index.values, series.index.values,
                     err_msg='_recode changed the index')
