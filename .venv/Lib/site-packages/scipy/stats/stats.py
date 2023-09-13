# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

import warnings
from . import _stats_py


__all__ = [  # noqa: F822
    'find_repeats', 'gmean', 'hmean', 'pmean', 'mode', 'tmean', 'tvar',
    'tmin', 'tmax', 'tstd', 'tsem', 'moment',
    'skew', 'kurtosis', 'describe', 'skewtest', 'kurtosistest',
    'normaltest', 'jarque_bera', 'itemfreq',
    'scoreatpercentile', 'percentileofscore',
    'cumfreq', 'relfreq', 'obrientransform',
    'sem', 'zmap', 'zscore', 'gzscore', 'iqr', 'gstd',
    'median_absolute_deviation', 'median_abs_deviation',
    'sigmaclip', 'trimboth', 'trim1', 'trim_mean',
    'f_oneway', 'F_onewayConstantInputWarning',
    'F_onewayBadInputSizesWarning',
    'PearsonRConstantInputWarning', 'PearsonRNearConstantInputWarning',
    'pearsonr', 'fisher_exact',
    'SpearmanRConstantInputWarning', 'spearmanr', 'pointbiserialr',
    'kendalltau', 'weightedtau', 'multiscale_graphcorr',
    'linregress', 'siegelslopes', 'theilslopes', 'ttest_1samp',
    'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel',
    'kstest', 'ks_1samp', 'ks_2samp',
    'chisquare', 'power_divergence',
    'tiecorrect', 'ranksums', 'kruskal', 'friedmanchisquare',
    'rankdata',
    'combine_pvalues', 'wasserstein_distance', 'energy_distance',
    'brunnermunzel', 'alexandergovern', 'gcd', 'namedtuple', 'array',
    'ma', 'cdist', 'check_random_state', 'MapWrapper',
    'rng_integers', 'float_factorial', 'linalg', 'distributions',
    'mstats_basic', 'ModeResult', 'DescribeResult',
    'SkewtestResult', 'KurtosistestResult', 'NormaltestResult',
    'Jarque_beraResult', 'HistogramResult', 'CumfreqResult',
    'RelfreqResult', 'SigmaclipResult', 'F_onewayResult',
    'AlexanderGovernResult', 'AlexanderGovernConstantInputWarning',
    'SpearmanrResult', 'PointbiserialrResult', 'KendalltauResult',
    'WeightedTauResult', 'MGCResult', 'Ttest_1sampResult', 'Ttest_indResult',
    'Ttest_relResult', 'Power_divergenceResult', 'KstestResult',
    'Ks_2sampResult', 'RanksumsResult', 'KruskalResult',
    'FriedmanchisquareResult', 'BrunnerMunzelResult', 'RepeatedResults'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.stats.stats is deprecated and has no attribute "
            f"{name}. Try looking in scipy.stats instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.stats` namespace, "
                  "the `scipy.stats.stats` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_stats_py, name)
