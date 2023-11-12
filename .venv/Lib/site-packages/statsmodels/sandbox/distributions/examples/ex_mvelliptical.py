# -*- coding: utf-8 -*-
"""examples for multivariate normal and t distributions


Created on Fri Jun 03 16:00:26 2011

@author: josef


for comparison I used R mvtnorm version 0.9-96

"""
import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.distributions.mixture_rvs as mix
import statsmodels.sandbox.distributions.mv_normal as mvd


cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                   [ 0.5 ,  1.5 ,  0.6 ],
                   [ 0.75,  0.6 ,  2.  ]])

mu = np.array([-1, 0.0, 2.0])

#************** multivariate normal distribution ***************

mvn3 = mvd.MVNormal(mu, cov3)

#compare with random sample
x = mvn3.rvs(size=1000000)

xli = [[2., 1., 1.5],
       [0., 2., 1.5],
       [1.5, 1., 2.5],
       [0., 1., 1.5]]

xliarr = np.asarray(xli).T[None,:, :]

#from R session
#pmvnorm(lower=-Inf,upper=(x[0,.]-mu)/sqrt(diag(cov3)),mean=rep(0,3),corr3)
r_cdf = [0.3222292, 0.3414643, 0.5450594, 0.3116296]
r_cdf_errors = [1.715116e-05, 1.590284e-05, 5.356471e-05, 3.567548e-05]
n_cdf = [mvn3.cdf(a) for a in xli]
assert_array_almost_equal(r_cdf, n_cdf, decimal=4)

print(n_cdf)
print('')
print((x<np.array(xli[0])).all(-1).mean(0))
print((x[...,None]<xliarr).all(1).mean(0))
print(mvn3.expect_mc(lambda x: (x<xli[0]).all(-1), size=100000))
print(mvn3.expect_mc(lambda x: (x[...,None]<xliarr).all(1), size=100000))

#other methods
mvn3n = mvn3.normalized()

assert_array_almost_equal(mvn3n.cov, mvn3n.corr, decimal=15)
assert_array_almost_equal(mvn3n.mean, np.zeros(3), decimal=15)

xn = mvn3.normalize(x)
xn_cov = np.cov(xn, rowvar=0)
assert_array_almost_equal(mvn3n.cov, xn_cov, decimal=2)
assert_array_almost_equal(np.zeros(3), xn.mean(0), decimal=2)

mvn3n2 = mvn3.normalized2()
assert_array_almost_equal(mvn3n.cov, mvn3n2.cov, decimal=2)
#mistake: "normalized2" standardizes - FIXED
#assert_array_almost_equal(np.eye(3), mvn3n2.cov, decimal=2)

xs = mvn3.standardize(x)
xs_cov = np.cov(xn, rowvar=0)
#another mixup xs is normalized
#assert_array_almost_equal(np.eye(3), xs_cov, decimal=2)
assert_array_almost_equal(mvn3.corr, xs_cov, decimal=2)
assert_array_almost_equal(np.zeros(3), xs.mean(0), decimal=2)

mv2m = mvn3.marginal(np.array([0,1]))
print(mv2m.mean)
print(mv2m.cov)

mv2c = mvn3.conditional(np.array([0,1]), [0])
print(mv2c.mean)
print(mv2c.cov)

mv2c = mvn3.conditional(np.array([0]), [0, 0])
print(mv2c.mean)
print(mv2c.cov)

mod = sm.OLS(x[:,0], sm.add_constant(x[:,1:], prepend=True))
res = mod.fit()
print(res.model.predict(np.array([1,0,0])))
mv2c = mvn3.conditional(np.array([0]), [0, 0])
print(mv2c.mean)
mv2c = mvn3.conditional(np.array([0]), [1, 1])
print(res.model.predict(np.array([1,1,1])))
print(mv2c.mean)

#the following wrong input does not raise an exception but produces wrong numbers
#mv2c = mvn3.conditional(np.array([0]), [[1, 1],[2,2]])

#************** multivariate t distribution ***************

mvt3 = mvd.MVT(mu, cov3, 4)
xt = mvt3.rvs(size=100000)
assert_array_almost_equal(mvt3.cov, np.cov(xt, rowvar=0), decimal=1)
mvt3s = mvt3.standardized()
mvt3n = mvt3.normalized()

#the following should be equal or correct up to numerical precision of float
assert_array_almost_equal(mvt3.corr, mvt3n.sigma, decimal=15)
assert_array_almost_equal(mvt3n.corr, mvt3n.sigma, decimal=15)
assert_array_almost_equal(np.eye(3), mvt3s.sigma, decimal=15)

xts = mvt3.standardize(xt)
xts_cov = np.cov(xts, rowvar=0)
xtn = mvt3.normalize(xt)
xtn_cov = np.cov(xtn, rowvar=0)
xtn_corr = np.corrcoef(xtn, rowvar=0)

assert_array_almost_equal(mvt3n.mean, xtn.mean(0), decimal=2)
#the following might fail sometimes (random test), add seed in tests
assert_array_almost_equal(mvt3n.corr, xtn_corr, decimal=1)
#watch out cov is not the same as sigma for t distribution, what's right here?
#normalize by sigma or by cov ? now normalized by sigma
assert_array_almost_equal(mvt3n.cov, xtn_cov, decimal=1)
assert_array_almost_equal(mvt3s.cov, xts_cov, decimal=1)

a = [0.0, 1.0, 1.5]
mvt3_cdf0 = mvt3.cdf(a)
print(mvt3_cdf0)
print((xt<np.array(a)).all(-1).mean(0))
print('R', 0.3026741) # "error": 0.0004832187
print('R', 0.3026855) # error 3.444375e-06   with smaller abseps
print('diff', mvt3_cdf0 - 0.3026855)
a = [0.0, 0.5, 1.0]
mvt3_cdf1 = mvt3.cdf(a)
print(mvt3_cdf1)
print((xt<np.array(a)).all(-1).mean(0))
print('R', 0.1946621) # "error": 0.0002524817)
print('R', 0.1946217) # "error:"2.748699e-06    with smaller abseps)
print('diff', mvt3_cdf1 - 0.1946217)

assert_array_almost_equal(mvt3_cdf0, 0.3026855, decimal=5)
assert_array_almost_equal(mvt3_cdf1, 0.1946217, decimal=5)

mu2 = np.array([4, 2.0, 2.0])
mvn32 = mvd.MVNormal(mu2, cov3/2., 4)
md = mix.mv_mixture_rvs([0.4, 0.6], 5, [mvt3, mvt3n], 3)
rvs = mix.mv_mixture_rvs([0.4, 0.6], 2000, [mvn3, mvn32], 3)
#rvs2 = rvs[:,:2]
fig = plt.figure()
fig.add_subplot(2, 2, 1)
plt.plot(rvs[:,0], rvs[:,1], '.', alpha=0.25)
plt.title('1 versus 0')
fig.add_subplot(2, 2, 2)
plt.plot(rvs[:,0], rvs[:,2], '.', alpha=0.25)
plt.title('2 versus 0')
fig.add_subplot(2, 2, 3)
plt.plot(rvs[:,1], rvs[:,2], '.', alpha=0.25)
plt.title('2 versus 1')
#plt.show()
