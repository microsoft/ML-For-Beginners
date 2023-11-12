"""
Test Results for ordinal models from R MASS lib
"""

import numpy as np
import os
import pandas as pd
from statsmodels.tools.testing import Holder

# R (v3.4.4) code inspired from
# https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
# library(readr) # to open the file
# library(MASS) # to perform ordinal regression
#
# ## load the data, 400 rows with 3 exogs(2 binaries, 1 float)
# ##and target 3-ordinal variable
# ologit_ucla <- read_csv("ologit_ucla.csv")
# ologit_ucla$apply <- as.factor(ologit_ucla$apply)
# ologit_ucla$apply <- factor(ologit_ucla$apply,
#               levels=c("unlikely", "somewhat likely", "very likely"))
#
# ## fit ordered logit model
# r_logit <- polr(apply ~ pared + public + gpa,
#          data = ologit_ucla,
#          method = 'logit', # or 'probit'
#          Hess=TRUE)
#
# ## fit ordered probit model
# r_probit <- polr(apply ~ pared + public + gpa,
#      data = ologit_ucla,
#      method = 'probit',
#      Hess=TRUE)
#
# ## fit ordered cloglog model
# r_cloglog <- polr(apply ~ pared + public + gpa,
#          data = ologit_ucla,
#          method = 'cloglog',
#          Hess=TRUE)
#
# ## with r = r_logit or r_probit or r_cloglog
# ## we add p-values
# (ctable <- coef(summary(r)))
# p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
# (ctable <- cbind(ctable, "p value" = p))
# ## show 7 first predictions
# head(predict(r, subset(ologit_ucla,
#                        select=c("pared", "public","gpa")), type='prob'),7)

data_store = Holder()
cur_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(cur_dir, "ologit_ucla.csv"))

# df_unordered['apply'] is pd.Categorical with ordered = False
df_unordered = df.copy()
df_unordered['apply'] = pd.Categorical(df['apply'], ordered=False)
# but categories are set in order
df_unordered['apply'] = df_unordered['apply'].cat.set_categories(
    ['unlikely', 'somewhat likely', 'very likely'])

# df['apply'] is pd.Categorical with ordered = True
df['apply'] = pd.Categorical(df['apply'], ordered=True)
df['apply'] = df['apply'].cat.set_categories(
    ['unlikely', 'somewhat likely', 'very likely'])

data_store.df_unordered = df_unordered
data_store.df = df
data_store.nobs = 400
data_store.n_ordinal_cat = 3

res_ord_logit = Holder()
res_ord_logit.coefficients_val = \
    np.array([1.04769011, -0.05878572, 0.61594057])
res_ord_logit.coefficients_stdE = np.array([0.2658, 0.2979, 0.2606])
res_ord_logit.coefficients_tval = np.array([3.9418, -0.1974, 2.3632])
res_ord_logit.coefficients_pval = \
    np.array([8.087070e-05, 8.435464e-01, 1.811594e-02])
res_ord_logit.thresholds = np.array([2.203915, 4.299363])
res_ord_logit.prob_pred = np.array([[0.5488310, 0.3593310, 0.09183798],
                                    [0.3055632, 0.4759496, 0.21848725],
                                    [0.2293835, 0.4781951, 0.29242138],
                                    [0.6161224, 0.3126888, 0.07118879],
                                    [0.6560149, 0.2833901, 0.06059505],
                                    [0.6609240, 0.2797117, 0.05936431],
                                    [0.6518332, 0.2865114, 0.06165547]])
res_ord_logit.resid_prob = np.array(
    [+0.90816202,  0.08707593, -0.77061649,  0.54493358,  0.59541984,
     -0.33907603,  0.59017771,  0.55970937, -0.41190566,  0.01851403,
     -0.73753054,  0.53932241,  0.87609730,  0.56880356, -0.73936739,
     -0.42539653, -0.47329831, -0.29581150,  0.90792753,  0.42811409])
res_ord_logit.resid_prob_stats = [
    3.5137670974297e-06, -0.7706164931682951, 0.9434781714439548,
    0.2550630116905416]

res_ord_probit = Holder()
res_ord_probit.coefficients_val = np.array([0.59811, 0.01016, 0.35815])
res_ord_probit.coefficients_stdE = np.array([0.1579, 0.1728, 0.1568])
res_ord_probit.coefficients_tval = np.array([3.78881, 0.05878, 2.28479])
res_ord_probit.coefficients_pval = \
    np.array([1.513681e-04, 9.531256e-01, 2.232519e-02])
res_ord_probit.thresholds = np.array([1.2968, 2.5028])
res_ord_probit.prob_pred = np.array([[0.5514181, 0.3576848, 0.09089707],
                                     [0.3260107, 0.4488799, 0.22510933],
                                     [0.2349733, 0.4506351, 0.31439162],
                                     [0.6142501, 0.3184778, 0.06727214],
                                     [0.6519891, 0.2928449, 0.05516602],
                                     [0.6402204, 0.3009945, 0.05878509],
                                     [0.6480094, 0.2956162, 0.05637442]])

res_ord_cloglog = Holder()
res_ord_cloglog.coefficients_val = np.array([0.5166455, 0.1081131, 0.3343895])
res_ord_cloglog.coefficients_stdE = np.array([0.1613525, 0.1680675, 0.1542065])
res_ord_cloglog.coefficients_tval = np.array([3.2019668, 0.6432721, 2.1684534])
res_ord_cloglog.coefficients_pval = \
    np.array([1.364927e-03, 5.200475e-01, 3.012421e-02])
res_ord_cloglog.thresholds = np.array([0.8705304, 1.9744660])
res_ord_cloglog.prob_pred = np.array([[0.5519526, 0.3592524, 0.08879500],
                                      [0.3855287, 0.3842645, 0.23020682],
                                      [0.2899487, 0.3540202, 0.35603111],
                                      [0.6067184, 0.3333548, 0.05992678],
                                      [0.6411418, 0.3133969, 0.04546127],
                                      [0.5940557, 0.3400072, 0.06593710],
                                      [0.6374521, 0.3156622, 0.04688570]])
