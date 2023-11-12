import warnings

from statsmodels.stats.diagnostic import (
    OLS,
    ResultsStore,
    acorr_breusch_godfrey,
    acorr_ljungbox,
    acorr_lm,
    breaks_cusumolsresid,
    breaks_hansen,
    compare_cox,
    compare_j,
    het_arch,
    het_breuschpagan,
    het_goldfeldquandt,
    het_white,
    linear_harvey_collier,
    linear_lm,
    linear_rainbow,
    recursive_olsresiduals,
    spec_white,
)
from statsmodels.tsa.stattools import adfuller

__all__ = ["OLS", "ResultsStore", "acorr_breusch_godfrey", "acorr_ljungbox",
           "acorr_lm", "adfuller", "breaks_cusumolsresid", "breaks_hansen",
           "compare_cox", "compare_j", "het_arch", "het_breuschpagan",
           "het_goldfeldquandt", "het_white", "linear_harvey_collier",
           "linear_lm", "linear_rainbow", "recursive_olsresiduals",
           "spec_white"]


warnings.warn("The statsmodels.sandbox.stats.diagnostic module is deprecated. "
              "Use statsmodels.stats.diagnostic.", FutureWarning,
              stacklevel=2)
