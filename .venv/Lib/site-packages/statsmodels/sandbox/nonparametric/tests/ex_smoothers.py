# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 10:51:39 2011

@author: josef
"""
import numpy as np

from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS


#DGP: simple polynomial
order = 3
sigma_noise = 0.5
nobs = 100
lb, ub = -1, 2
x = np.linspace(lb, ub, nobs)
x = np.sin(x)
exog = x[:,None]**np.arange(order+1)
y_true = exog.sum(1)
y = y_true + sigma_noise * np.random.randn(nobs)



#xind = np.argsort(x)
pmod = smoothers.PolySmoother(2, x)
pmod.fit(y)  #no return
y_pred = pmod.predict(x)
error = y - y_pred
mse = (error*error).mean()
print(mse)
res_ols = OLS(y, exog[:,:3]).fit()
print(np.squeeze(pmod.coef) - res_ols.params)


weights = np.ones(nobs)
weights[:nobs//3] = 0.1
weights[-nobs//5:] = 2

pmodw = smoothers.PolySmoother(2, x)
pmodw.fit(y, weights=weights)  #no return
y_predw = pmodw.predict(x)
error = y - y_predw
mse = (error*error).mean()
print(mse)
res_wls = WLS(y, exog[:,:3], weights=weights).fit()
print(np.squeeze(pmodw.coef) - res_wls.params)



doplot = 1
if doplot:
    import matplotlib.pyplot as plt
    plt.plot(y, '.')
    plt.plot(y_true, 'b-', label='true')
    plt.plot(y_pred, '-', label='poly')
    plt.plot(y_predw, '-', label='poly -w')
    plt.legend(loc='upper left')

    plt.close()
    #plt.show()
