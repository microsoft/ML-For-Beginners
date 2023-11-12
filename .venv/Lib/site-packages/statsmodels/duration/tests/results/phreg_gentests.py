import numpy as np

"""
Generate data sets for testing Cox proportional hazards regression
models.

After updating the test data sets, use R to run the survival.R script
to update the R results.
"""

# The current data may not reflect this seed
np.random.seed(5234)

# Loop over pairs containing (sample size, number of variables).
for (n, p) in (20, 1), (50, 1), (50, 2), (100, 5), (1000, 10):

    exog = np.random.normal(size=(5*n, p))
    coef = np.linspace(-0.5, 0.5, p)
    lpred = np.dot(exog, coef)
    expected_survival_time = np.exp(-lpred)

    # Survival times are exponential
    survival_time = -np.log(np.random.uniform(size=5*n))
    survival_time *= expected_survival_time

    # Set this to get a reasonable amount of censoring
    expected_censoring_time = np.mean(expected_survival_time)

    # Theses are the observation times.
    censoring_time = -np.log(np.random.uniform(size=5*n))
    censoring_time *= expected_censoring_time

    # Entry times
    entry_time = -np.log(np.random.uniform(size=5*n))
    entry_time *= 0.5*expected_censoring_time

    # 1=failure (death), 0=no failure (no death)
    status = 1*(survival_time <= censoring_time)

    # The censoring time of the failure time, whichever comes first
    time = np.where(status == 1, survival_time, censoring_time)

    # Round time so that we have ties
    time = np.around(time, decimals=1)

    # Only take cases where the entry time is before the failure or
    # censoring time.  Take exactly n such cases.
    ii = np.flatnonzero(entry_time < time)
    ii = ii[np.random.permutation(len(ii))[0:n]]
    status = status[ii]
    time = time[ii]
    exog = exog[ii, :]
    entry_time = entry_time[ii]

    data = np.concatenate((time[:, None], status[:, None],
                           entry_time[:, None], exog),
                          axis=1)

    fname = "results/survival_data_%d_%d.csv" % (n, p)
    np.savetxt(fname, data, fmt="%.5f")
