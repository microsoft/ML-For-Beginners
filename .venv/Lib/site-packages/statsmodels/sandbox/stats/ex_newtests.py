import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.stattools import adfuller

macrod = macro.load().data

print(macro.NOTE)

print(macrod.dtype.names)

datatrendli = [
               ('realgdp', 1),
               ('realcons', 1),
               ('realinv', 1),
               ('realgovt', 1),
               ('realdpi', 1),
               ('cpi', 1),
               ('m1', 1),
               ('tbilrate', 0),
               ('unemp',0),
               ('pop', 1),
               ('infl',0),
               ('realint', 0)
               ]

print('%-10s %5s %-8s' % ('variable', 'trend', '  adf'))
for name, torder in datatrendli:
    c_order = {0: "n", 1: "c"}
    adf_, pval = adfuller(macrod[name], regression=c_order[torder])[:2]
    print('%-10s %5d %8.4f %8.4f' % (name, torder, adf_, pval))
