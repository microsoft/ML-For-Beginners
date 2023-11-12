import numpy as np

"""
Generate data sets for testing OLS.fit_regularized

After running this script, rerun lasso_r_results.R in R to rebuild the
results file "glmnet_r_results.py".

Currently only tests OLS.  Our implementation covers GLS, but it's not
clear if glmnet does.
"""

n = 300
p = 5

np.random.seed(83423)

exog = np.random.normal(size=(n, p))
params = (-1.)**np.arange(p)
params[::3] = 0
expval = np.dot(exog, params)
endog = expval + np.random.normal(size=n)
data = np.concatenate((endog[:, None], exog), axis=1)
data = np.around(100*data)

fname = "lasso_data.csv"
np.savetxt(fname, data, fmt="%.0f", delimiter=",")
