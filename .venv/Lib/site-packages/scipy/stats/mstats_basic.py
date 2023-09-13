# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

import warnings
from . import _mstats_basic


__all__ = [  # noqa: F822
    'argstoarray',
    'count_tied_groups',
    'describe',
    'f_oneway', 'find_repeats','friedmanchisquare',
    'kendalltau','kendalltau_seasonal','kruskal','kruskalwallis',
    'ks_twosamp', 'ks_2samp', 'kurtosis', 'kurtosistest',
    'ks_1samp', 'kstest',
    'linregress',
    'mannwhitneyu', 'meppf','mode','moment','mquantiles','msign',
    'normaltest',
    'obrientransform',
    'pearsonr','plotting_positions','pointbiserialr',
    'rankdata',
    'scoreatpercentile','sem',
    'sen_seasonal_slopes','skew','skewtest','spearmanr',
    'siegelslopes', 'theilslopes',
    'tmax','tmean','tmin','trim','trimboth',
    'trimtail','trima','trimr','trimmed_mean','trimmed_std',
    'trimmed_stde','trimmed_var','tsem','ttest_1samp','ttest_onesamp',
    'ttest_ind','ttest_rel','tvar',
    'variation',
    'winsorize',
    'brunnermunzel', 'ma', 'masked', 'nomask', 'namedtuple',
    'distributions', 'stats_linregress', 'stats_LinregressResult',
    'stats_theilslopes', 'stats_siegelslopes', 'ModeResult',
    'SpearmanrResult', 'KendalltauResult', 'PointbiserialrResult',
    'Ttest_1sampResult', 'Ttest_indResult', 'Ttest_relResult',
    'MannwhitneyuResult', 'KruskalResult', 'trimdoc', 'trim1',
    'DescribeResult', 'stde_median', 'SkewtestResult', 'KurtosistestResult',
    'NormaltestResult', 'F_onewayResult', 'FriedmanchisquareResult',
    'BrunnerMunzelResult'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.stats.mstats_basic is deprecated and has no attribute "
            f"{name}. Try looking in scipy.stats instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.stats` namespace, "
                  "the `scipy.stats.mstats_basic` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_mstats_basic, name)
