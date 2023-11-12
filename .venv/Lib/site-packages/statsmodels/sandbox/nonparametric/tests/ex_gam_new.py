# -*- coding: utf-8 -*-
"""Example for GAM with Poisson Model and PolynomialSmoother

This example was written as a test case.
The data generating process is chosen so the parameters are well identified
and estimated.

Created on Fri Nov 04 13:45:43 2011

Author: Josef Perktold
"""
from statsmodels.compat.python import lrange
import time

import numpy as np

from scipy import stats

from statsmodels.sandbox.gam import Model as GAM
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM

np.seterr(all='raise')
np.random.seed(8765993)
#seed is chosen for nice result, not randomly
#other seeds are pretty off in the prediction or end in overflow

#DGP: simple polynomial
order = 3
sigma_noise = 0.1
nobs = 1000
#lb, ub = -0.75, 3#1.5#0.75 #2.5
lb, ub = -3.5, 3
x1 = np.linspace(lb, ub, nobs)
x2 = np.sin(2*x1)
x = np.column_stack((x1/x1.max()*1, 1.*x2))
exog = (x[:,:,None]**np.arange(order+1)[None, None, :]).reshape(nobs, -1)
idx = lrange((order+1)*2)
del idx[order+1]
exog_reduced = exog[:,idx]  #remove duplicate constant
y_true = exog.sum(1) #/ 4.
z = y_true #alias check
d = x
y = y_true + sigma_noise * np.random.randn(nobs)

example = 3

if example == 2:
    print("binomial")
    f = family.Binomial()
    mu_true = f.link.inverse(z)
    #b = np.asarray([scipy.stats.bernoulli.rvs(p) for p in f.link.inverse(y)])
    b = np.asarray([stats.bernoulli.rvs(p) for p in f.link.inverse(z)])
    b.shape = y.shape
    m = GAM(b, d, family=f)
    toc = time.time()
    m.fit(b)
    tic = time.time()
    print(tic-toc)
    #for plotting
    yp = f.link.inverse(y)
    p = b


if example == 3:
    print("Poisson")
    f = family.Poisson()
    #y = y/y.max() * 3
    yp = f.link.inverse(z)
    p = np.asarray([stats.poisson.rvs(val) for val in f.link.inverse(z)],
                   float)
    p.shape = y.shape
    m = GAM(p, d, family=f)
    toc = time.time()
    m.fit(p)
    tic = time.time()
    print(tic-toc)

for ss in m.smoothers:
    print(ss.params)

if example > 1:
    import matplotlib.pyplot as plt
    plt.figure()
    for i in np.array(m.history[2:15:3]):
        plt.plot(i.T)

    plt.figure()
    plt.plot(exog)
    #plt.plot(p, '.', lw=2)
    plt.plot(y_true, lw=2)

    y_pred = m.results.mu # + m.results.alpha #m.results.predict(d)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(p, '.')
    plt.plot(yp, 'b-', label='true')
    plt.plot(y_pred, 'r-', label='GAM')
    plt.legend(loc='upper left')
    plt.title('gam.GAM Poisson')

    counter = 2
    for ii, xx in zip(['z', 'x1', 'x2'], [z, x[:,0], x[:,1]]):
        sortidx = np.argsort(xx)
        #plt.figure()
        plt.subplot(2, 2, counter)
        plt.plot(xx[sortidx], p[sortidx], 'k.', alpha=0.5)
        plt.plot(xx[sortidx], yp[sortidx], 'b.', label='true')
        plt.plot(xx[sortidx], y_pred[sortidx], 'r.', label='GAM')
        plt.legend(loc='upper left')
        plt.title('gam.GAM Poisson ' + ii)
        counter += 1

    res = GLM(p, exog_reduced, family=f).fit()

    #plot component, compared to true component
    x1 = x[:,0]
    x2 = x[:,1]
    f1 = exog[:,:order+1].sum(1) - 1 #take out constant
    f2 = exog[:,order+1:].sum(1) - 1
    plt.figure()
    #Note: need to correct for constant which is indeterminatedly distributed
    #plt.plot(x1, m.smoothers[0](x1)-m.smoothers[0].params[0]+1, 'r')
    #better would be subtract f(0) m.smoothers[0](np.array([0]))
    plt.plot(x1, f1, linewidth=2)
    plt.plot(x1, m.smoothers[0](x1)-m.smoothers[0].params[0], 'r')

    plt.figure()
    plt.plot(x2, f2, linewidth=2)
    plt.plot(x2, m.smoothers[1](x2)-m.smoothers[1].params[0], 'r')


    plt.show()
