"""
Generate test data sets for lme.

After running this script, run lme_results.R with R
to update the output.
"""

import numpy as np
import os

np.random.seed(348491)

# Number of groups
ngroup = 100

# Sample size range per group
n_min = 1
n_max = 5

dsix = 0

# Number of random effects
for pr in [1, 2]:

    re_sd = np.linspace(-0.5, 1.5, pr)

    # Number of fixed effects
    for pf in [1, 2, 3]:

        # Error standard deviation
        for sig in [0.5, 2]:

            params = np.linspace(-1, 1, pf)

            endog = []
            exog_fe = []
            exog_re = []
            groups = []
            for i in range(ngroup):

                n = np.random.randint(n_min, n_max, 1)
                x_fe = np.random.normal(size=(n, pf))
                x_re = np.zeros((n, pr))
                u = np.linspace(-1, 1, n)
                for j in range(pr):
                    x_re[:, j] = u**j

                re = np.random.normal(size=pr) * re_sd

                expval = np.dot(x_fe, params) + np.dot(x_re, re)

                endog.append(expval + sig*np.random.normal(size=n))
                exog_fe.append(x_fe)
                exog_re.append(x_re)
                groups.append(i*np.ones(n))

            endog = np.concatenate(endog)
            exog_fe = np.concatenate(exog_fe, axis=0)
            exog_re = np.concatenate(exog_re, axis=0)
            groups = np.concatenate(groups, axis=0)

            data = np.concatenate((groups[:, None], endog[:, None],
                                   exog_fe, exog_re), axis=1)

            header = (["groups,endog"] +
                      ["exog_fe_%d" % k for k in range(pf)] +
                      ["exog_re_%d" % k for k in range(pr)])
            header = ",".join(header)

            cur_dir = os.path.dirname(os.path.abspath(__file__))

            fname = os.path.join(cur_dir, "lme%02d.csv" % dsix)
            np.savetxt(fname, data, fmt="%.3f", header=header,
                       delimiter=",", comments="")
            dsix += 1
