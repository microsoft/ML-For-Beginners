# -*- coding: utf-8 -*-
"""Test values for multinomial_proportion_confint.

Author: Sébastien Lerique
"""


import collections
import numpy as np

from statsmodels.tools.testing import Holder

res_multinomial = collections.defaultdict(Holder)

# The following examples come from the Sison & Glaz paper, and the values were
# computed using the R MultinomialCI package.

# Floating-point arithmetic errors get blown up in the Edgeworth expansion
# (starting in g1 and g2, but mostly when computing f, because of the
# polynomials), which explains why we only obtain a precision of 4 decimals
# when comparing to values computed in R.

# We test with any method name that starts with 'sison', as that is the
# criterion.
key1 = ('sison', 'Sison-Glaz example 1')
res_multinomial[key1].proportions = [56, 72, 73, 59, 62, 87, 58]
res_multinomial[key1].cis = np.array([
    [.07922912, .1643361], [.11349036, .1985973],
    [.11563169, .2007386], [.08565310, .1707601],
    [.09207709, .1771840], [.14561028, .2307172],
    [.08351178, .1686187]])
res_multinomial[key1].precision = 4

key2 = ('sisonandglaz', 'Sison-Glaz example 2')
res_multinomial[key2].proportions = [5] * 50
res_multinomial[key2].cis = [0, .05304026] * np.ones((50, 2))
res_multinomial[key2].precision = 4

key3 = ('sison-whatever', 'Sison-Glaz example 3')
res_multinomial[key3].proportions = (
    [1] * 10 + [12] * 10 + [5] * 10 + [3] * 10 + [4] * 10)
res_multinomial[key3].cis = np.concatenate([
    [0, .04120118] * np.ones((10, 2)),
    [.012, .08520118] * np.ones((10, 2)),
    [0, .05720118] * np.ones((10, 2)),
    [0, .04920118] * np.ones((10, 2)),
    [0, .05320118] * np.ones((10, 2))
])
res_multinomial[key3].precision = 4

# The examples from the Sison & Glaz paper only include 3 decimals.
gkey1 = ('goodman', 'Sison-Glaz example 1')
res_multinomial[gkey1].proportions = [56, 72, 73, 59, 62, 87, 58]
res_multinomial[gkey1].cis = np.array([
    [.085, .166],
    [.115, .204],
    [.116, .207],
    [.091, .173],
    [.096, .181],
    [.143, .239],
    [.089, .171]])
res_multinomial[gkey1].precision = 3

gkey2 = ('goodman', 'Sison-Glaz example 2')
res_multinomial[gkey2].proportions = [5] * 50
res_multinomial[gkey2].cis = [.005, .075] * np.ones((50, 2))
res_multinomial[gkey2].precision = 3

gkey3 = ('goodman', 'Sison-Glaz example 3')
res_multinomial[gkey3].proportions = (
    [1] * 10 + [12] * 10 + [5] * 10 + [3] * 10 + [4] * 10)
res_multinomial[gkey3].cis = np.concatenate([
    [0, .049] * np.ones((10, 2)),
    [.019, .114] * np.ones((10, 2)),
    [.005, .075] * np.ones((10, 2)),
    [.002, .062] * np.ones((10, 2)),
    [.004, .069] * np.ones((10, 2))
])
res_multinomial[gkey3].precision = 3
