import numpy as np

"""
Generate data sets for testing elastic net fits of GLMs.
"""

n = 200
p = 5

# Logistic
exog = np.random.normal(size=(n, p))
lin_pred = exog.sum(1) * 0.2
exp_val = 1 / (1 + np.exp(-lin_pred))
endog = 1 * (np.random.uniform(size=n) < exp_val)
mat = np.concatenate((endog[:, None], exog), axis=1)
np.savetxt("enet_binomial.csv", mat, fmt="%.2f", delimiter=",")

# Poisson
exog = np.random.normal(size=(n, p))
lin_pred = exog.sum(1) * 0.2
exp_val = np.exp(lin_pred)
endog = np.random.poisson(exp_val)
mat = np.concatenate((endog[:, None], exog), axis=1)
np.savetxt("enet_poisson.csv", mat, fmt="%.2f", delimiter=",")
