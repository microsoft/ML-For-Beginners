# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'find_repeats', 'gmean', 'hmean', 'pmean', 'mode', 'tmean', 'tvar',
    'tmin', 'tmax', 'tstd', 'tsem', 'moment',
    'skew', 'kurtosis', 'describe', 'skewtest', 'kurtosistest',
    'normaltest', 'jarque_bera',
    'scoreatpercentile', 'percentileofscore',
    'cumfreq', 'relfreq', 'obrientransform',
    'sem', 'zmap', 'zscore', 'gzscore', 'iqr', 'gstd',
    'median_abs_deviation',
    'sigmaclip', 'trimboth', 'trim1', 'trim_mean',
    'f_oneway',
    'pearsonr', 'fisher_exact',
    'spearmanr', 'pointbiserialr',
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
    'HistogramResult', 'CumfreqResult',
    'RelfreqResult', 'SigmaclipResult', 'F_onewayResult',
    'AlexanderGovernResult',
    'PointbiserialrResult',
    'MGCResult', 'Ttest_1sampResult', 'Ttest_indResult',
    'Ttest_relResult', 'Power_divergenceResult', 'KstestResult',
    'Ks_2sampResult', 'RanksumsResult', 'KruskalResult',
    'FriedmanchisquareResult', 'BrunnerMunzelResult', 'RepeatedResults'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="stats",
                                   private_modules=["_stats_py"], all=__all__,
                                   attribute=name)
