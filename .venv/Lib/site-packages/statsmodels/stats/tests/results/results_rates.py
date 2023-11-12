# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:21:09 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from statsmodels.tools.testing import Holder

NA = np.nan

# > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), tsmethod="minlike",
#                      midp=FALSE)
# > cat_items(pe, prefix="res.")
res_pexact_cond = res = Holder()
res.statistic = 60
res.parameter = 43.7956463130352
res.p_value = 0.000675182658686321
res.conf_int = np.array([
    1.34983090611567, 3.27764509862914
    ])
res.estimate = 2.10999757175465
res.null_value = 1
res.alternative = 'two.sided'
res.method = ('Exact two-sided Poisson test (sum of minimum likelihood'
              ' method)')
res.data_name = 'c(60, 30) time base: c(51477.5, 54308.7)'


# > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), tsmethod="minlike",
#                      midp=TRUE)
# > cat_items(pe, prefix="res.")
res_pexact_cond_midp = res = Holder()
res.statistic = 60
res.parameter = 43.7956463130352
res.p_value = 0.000557262406619052
res.conf_int = np.array([
    NA, NA
    ])
res.estimate = 2.10999757175465
res.null_value = 1
res.alternative = 'two.sided'
res.method = ('Exact two-sided Poisson test (sum of minimum'
              ' likelihood method), mid-p version')
res.data_name = 'c(60, 30) time base: c(51477.5, 54308.7)'
