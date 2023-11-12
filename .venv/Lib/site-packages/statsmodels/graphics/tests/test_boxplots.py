import numpy as np
import pytest

from statsmodels.datasets import anes96
from statsmodels.graphics.boxplots import beanplot, violinplot

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


@pytest.fixture(scope="module")
def age_and_labels():
    # Test violinplot and beanplot with the same dataset.
    data = anes96.load_pandas()
    party_ID = np.arange(7)
    labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",
              "Independent-Independent", "Independent-Republican",
              "Weak Republican", "Strong Republican"]
    age = [data.exog['age'][data.endog == id] for id in party_ID]
    age = np.array(age, dtype="object")
    return age, labels


# TODO: Remove once SciPy 1.5.0 is the minimum
IGNORE_VISIBLE_DEPR = """\
ignore:Creating an ndarray from ragged nested sequences:\
numpy.VisibleDeprecationWarning:"""


@pytest.mark.filterwarnings(IGNORE_VISIBLE_DEPR)
@pytest.mark.matplotlib
def test_violinplot(age_and_labels, close_figures):
    age, labels = age_and_labels

    fig, ax = plt.subplots(1, 1)
    violinplot(age, ax=ax, labels=labels,
               plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs',
                          'label_fontsize': 'small',
                          'label_rotation': 30})


@pytest.mark.filterwarnings(IGNORE_VISIBLE_DEPR)
@pytest.mark.matplotlib
def test_violinplot_bw_factor(age_and_labels, close_figures):
    age, labels = age_and_labels

    fig, ax = plt.subplots(1, 1)
    violinplot(age, ax=ax, labels=labels,
               plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs',
                          'label_fontsize': 'small',
                          'label_rotation': 30,
                          'bw_factor': .2})


@pytest.mark.matplotlib
def test_beanplot(age_and_labels, close_figures):
    age, labels = age_and_labels

    fig, ax = plt.subplots(1, 1)
    beanplot(age, ax=ax, labels=labels,
             plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs',
                        'label_fontsize': 'small',
                        'label_rotation': 30})


@pytest.mark.matplotlib
def test_beanplot_jitter(age_and_labels, close_figures):
    age, labels = age_and_labels

    fig, ax = plt.subplots(1, 1)
    beanplot(age, ax=ax, labels=labels, jitter=True,
             plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs',
                        'label_fontsize': 'small',
                        'label_rotation': 30})


@pytest.mark.matplotlib
def test_beanplot_side_right(age_and_labels, close_figures):
    age, labels = age_and_labels

    fig, ax = plt.subplots(1, 1)
    beanplot(age, ax=ax, labels=labels, jitter=True, side='right',
             plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs',
                        'label_fontsize': 'small',
                        'label_rotation': 30})


@pytest.mark.matplotlib
def test_beanplot_side_left(age_and_labels, close_figures):
    age, labels = age_and_labels

    fig, ax = plt.subplots(1, 1)
    beanplot(age, ax=ax, labels=labels, jitter=True, side='left',
             plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs',
                        'label_fontsize': 'small',
                        'label_rotation': 30})


@pytest.mark.matplotlib
def test_beanplot_legend_text(age_and_labels, close_figures):
    age, labels = age_and_labels

    fig, ax = plt.subplots(1, 1)
    beanplot(age, ax=ax, labels=labels,
             plot_opts={'bean_legend_text': 'text'})
