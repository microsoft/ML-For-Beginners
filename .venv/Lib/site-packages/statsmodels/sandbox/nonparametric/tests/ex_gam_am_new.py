# -*- coding: utf-8 -*-
"""Example for gam.AdditiveModel and PolynomialSmoother

This example was written as a test case.
The data generating process is chosen so the parameters are well identified
and estimated.

Created on Fri Nov 04 13:45:43 2011

Author: Josef Perktold

"""
from statsmodels.compat.python import lrange

import numpy as np

from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.regression.linear_model import OLS

np.random.seed(8765993)
#seed is chosen for nice result, not randomly
#other seeds are pretty off in the prediction

#DGP: simple polynomial
order = 3
sigma_noise = 0.5
nobs = 1000  #1000 #with 1000, OLS and Additivemodel agree in params at 2 decimals
lb, ub = -3.5, 4#2.5
x1 = np.linspace(lb, ub, nobs)
x2 = np.sin(2*x1)
x = np.column_stack((x1/x1.max()*2, x2))
exog = (x[:,:,None]**np.arange(order+1)[None, None, :]).reshape(nobs, -1)
idx = lrange((order+1)*2)
del idx[order+1]
exog_reduced = exog[:,idx]  #remove duplicate constant
y_true = exog.sum(1) / 2.
z = y_true #alias check
d = x
y = y_true + sigma_noise * np.random.randn(nobs)

example = 1

if example == 1:
    m = AdditiveModel(d)
    m.fit(y)

    y_pred = m.results.predict(d)


for ss in m.smoothers:
    print(ss.params)

res_ols = OLS(y, exog_reduced).fit()
print(res_ols.params)

#from numpy.testing import assert_almost_equal
#assert_almost_equal(y_pred, res_ols.fittedvalues, 3)

if example > 0:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(exog)

    y_pred = m.results.mu# + m.results.alpha #m.results.predict(d)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(y, '.', alpha=0.25)
    plt.plot(y_true, 'k-', label='true')

    plt.plot(res_ols.fittedvalues, 'g-', label='OLS', lw=2, alpha=-.7)
    plt.plot(y_pred, 'r-', label='AM')
    plt.legend(loc='upper left')
    plt.title('gam.AdditiveModel')

    counter = 2
    for ii, xx in zip(['z', 'x1', 'x2'], [z, x[:,0], x[:,1]]):
        sortidx = np.argsort(xx)
        #plt.figure()
        plt.subplot(2, 2, counter)
        plt.plot(xx[sortidx], y[sortidx], '.', alpha=0.25)
        plt.plot(xx[sortidx], y_true[sortidx], 'k.', label='true', lw=2)
        plt.plot(xx[sortidx], y_pred[sortidx], 'r.', label='AM')
        plt.legend(loc='upper left')
        plt.title('gam.AdditiveModel ' + ii)
        counter += 1

    plt.show()
