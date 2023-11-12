from scipy import stats
from statsmodels.stats import gof

poissrvs = stats.poisson.rvs(0.6, size = 200)

freq, expfreq, histsupp = gof.gof_binning_discrete(poissrvs, stats.poisson, (0.6,), nsupp=20)
(chi2val, pval) = stats.chisquare(freq, expfreq)
print(chi2val, pval)

print(gof.gof_chisquare_discrete(stats.poisson, (0.6,), poissrvs, 0.05,
                                     'Poisson'))
