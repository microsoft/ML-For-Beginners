# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 07:56:22 2010

Author: josef-pktd
"""
import sympy as sy


def pdf(x, mu, sigma):
    """Return the probability density function as an expression in x"""
    #x = sy.sympify(x)
    return 1/(sigma*sy.sqrt(2*sy.pi)) * sy.exp(-(x-mu)**2 / (2*sigma**2))

def cdf(x, mu, sigma):
    """Return the cumulative density function as an expression in x"""
    #x = sy.sympify(x)
    return (1+sy.erf((x-mu)/(sigma*sy.sqrt(2))))/2


mu = sy.Symbol('mu')
sigma = sy.Symbol('sigma')
sigma2 = sy.Symbol('sigma2')
x = sy.Symbol('x')
y = sy.Symbol('y')
df = sy.Symbol('df')
s = sy.Symbol('s')

dldxnorm = sy.log(pdf(x, mu,sigma)).diff(x)
print(sy.simplify(dldxnorm))
print(sy.diff(sy.log(sy.gamma((s+1)/2)),s))

print(sy.diff((df+1)/2. * sy.log(1+df/(df-2)), df))

#standard t distribution, not verified
tllf1 = sy.log(sy.gamma((df+1)/2.)) - sy.log(sy.gamma(df/2.)) - 0.5*sy.log((df)*sy.pi)
tllf2 = (df+1.)/2. * sy.log(1. + (y-mu)**2/(df)/sigma2) + 0.5 * sy.log(sigma2)
tllf2std = (df+1.)/2. * sy.log(1. + y**2/df) + 0.5
tllf = tllf1 - tllf2
print(tllf1.diff(df))
print(tllf2.diff(y))
dlddf = (tllf1-tllf2).diff(df)
print(dlddf)
print(sy.cse(dlddf))
print('\n derivative of loglike of t distribution wrt df')
for k,v in sy.cse(dlddf)[0]:
    print(k, '=', v)

print(sy.cse(dlddf)[1][0])

print('\nstandard t distribution, dll_df, dll_dy')
tllfstd = tllf1 - tllf2std
print(tllfstd.diff(df))
print(tllfstd.diff(y))

print('\n')

print(dlddf.subs(dict(y=1,mu=1,sigma2=1.5,df=10.0001)))
print(dlddf.subs(dict(y=1,mu=1,sigma2=1.5,df=10.0001)).evalf())
# Note: derivatives of nested function does not work in sympy
#       at least not higher order derivatives (second or larger)
#       looks like print(failure
