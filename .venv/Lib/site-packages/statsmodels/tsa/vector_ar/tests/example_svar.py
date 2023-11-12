import numpy as np
import pandas as pd

import statsmodels.datasets.macrodata
from statsmodels.tsa.vector_ar.svar_model import SVAR

mdatagen = statsmodels.datasets.macrodata.load().data
mdata = mdatagen[['realgdp','realcons','realinv']]
names = mdata.dtype.names
start = pd.datetime(1959, 3, 31)
end = pd.datetime(2009, 9, 30)
#qtr = pd.DatetimeIndex(start=start, end=end, freq=pd.datetools.BQuarterEnd())
qtr = pd.date_range(start=start, end=end, freq='BQ-MAR')
data = pd.DataFrame(mdata, index=qtr)
data = (np.log(data)).diff().dropna()

#define structural inputs
A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
A_guess = np.asarray([0.5, 0.25, -0.38])
B_guess = np.asarray([0.5, 0.1, 0.05])
mymodel = SVAR(data, svar_type='AB', A=A, B=B, freq='Q')
res = mymodel.fit(maxlags=3, maxiter=10000, maxfun=10000, solver='bfgs')
res.irf(periods=30).plot(impulse='realgdp', plot_stderr=True,
                         stderr_type='mc', repl=100)
