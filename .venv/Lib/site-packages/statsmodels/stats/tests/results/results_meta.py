# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:31:21 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from statsmodels.tools.testing import Holder

"""
example from Kacker 2004, computed with R metafor

> y = c(61.0, 61.4 , 62.21, 62.3 , 62.34, 62.6 , 62.7 , 62.84, 65.9)
> v = c(0.2025, 1.2100, 0.0900, 0.2025, 0.3844, 0.5625, 0.0676, 0.0225, 1.8225)
> res = rma(y, v, data=dat, method="PM", control=list(tol=1e-9))
> convert_items(res, prefix="exk1_metafor.")

"""

exk1_metafor = Holder()
exk1_metafor.b = 62.4076199113286
exk1_metafor.beta = 62.4076199113286
exk1_metafor.se = 0.338030602684471
exk1_metafor.zval = 184.621213037276
exk1_metafor.pval = 0
exk1_metafor.ci_lb = 61.7450921043947
exk1_metafor.ci_ub = 63.0701477182625
exk1_metafor.vb = 0.114264688351227
exk1_metafor.tau2 = 0.705395309224248
exk1_metafor.se_tau2 = 0.51419109758052
exk1_metafor.tau2_f = 0.705395309224248
exk1_metafor.k = 9
exk1_metafor.k_f = 9
exk1_metafor.k_eff = 9
exk1_metafor.k_all = 9
exk1_metafor.p = 1
exk1_metafor.p_eff = 1
exk1_metafor.parms = 2
exk1_metafor.m = 1
exk1_metafor.QE = 24.801897741835
exk1_metafor.QEp = 0.00167935146372742
exk1_metafor.QM = 34084.9923033553
exk1_metafor.QMp = 0
exk1_metafor.I2 = 83.7218626490482
exk1_metafor.H2 = 6.14320900751909
exk1_metafor.yi = np.array([
    61, 61.4, 62.21, 62.3, 62.34, 62.6, 62.7, 62.84, 65.9
    ])
exk1_metafor.vi = np.array([
    0.2025, 1.21, 0.09, 0.2025, 0.3844, 0.5625, 0.0676, 0.0225, 1.8225
    ])
exk1_metafor.X = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1
    ]).reshape(9, 1, order='F')

exk1_metafor.yi_f = np.array([
    61, 61.4, 62.21, 62.3, 62.34, 62.6, 62.7, 62.84, 65.9
    ])
exk1_metafor.vi_f = np.array([
    0.2025, 1.21, 0.09, 0.2025, 0.3844, 0.5625, 0.0676, 0.0225, 1.8225
    ])
exk1_metafor.X_f = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1
    ]).reshape(9, 1, order='F')

exk1_metafor.M = np.array([
    0.907895309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.91539530922425, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0.795395309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.907895309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.08979530922425, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1.26789530922425, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.772995309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.727895309224248, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 2.52789530922425
    ]).reshape(9, 9, order='F')

exk1_metafor.ids = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9
    ])
exk1_metafor.slab = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9
    ])
exk1_metafor.measure = 'GEN'
exk1_metafor.method = 'PM'
exk1_metafor.test = 'z'
exk1_metafor.s2w = 1
exk1_metafor.btt = 1
exk1_metafor.digits = np.array([
    4, 4, 4, 4, 4, 4, 4, 4, 4
    ])
exk1_metafor.level = 0.05
exk1_metafor.add = 0.5
exk1_metafor.to = 'only0'
exk1_metafor.fit_stats = np.array([
    -12.722152033808, 21.73438033144, 29.4443040676159, 29.8387532222884,
    31.4443040676159, -11.7892200590463, 23.5784401180925, 27.5784401180925,
    27.7373232014522, 29.9784401180925
    ]).reshape(5, 2, order='F')

exk1_metafor.model = 'rma.uni'


# > res = rma(y, v, data=dat, method="DL", control=list(tol=1e-9))
# > convert_items(res, prefix="exk1_dl.")

exk1_dl = Holder()
exk1_dl.b = 62.3901386044504
exk1_dl.beta = 62.3901386044504
exk1_dl.se = 0.245749668040304
exk1_dl.zval = 253.876797075543
exk1_dl.pval = 0
exk1_dl.ci_lb = 61.9084781058787
exk1_dl.ci_ub = 62.8717991030221
exk1_dl.vb = 0.0603928993419195
exk1_dl.tau2 = 0.288049246973751
exk1_dl.se_tau2 = 0.269366223207558
exk1_dl.tau2_f = 0.288049246973751
exk1_dl.k = 9
exk1_dl.k_f = 9
exk1_dl.k_eff = 9
exk1_dl.k_all = 9
exk1_dl.p = 1
exk1_dl.p_eff = 1
exk1_dl.parms = 2
exk1_dl.m = 1
exk1_dl.QE = 24.801897741835
exk1_dl.QEp = 0.00167935146372742
exk1_dl.QM = 64453.4280933367
exk1_dl.QMp = 0
exk1_dl.I2 = 67.744403741711
exk1_dl.H2 = 3.10023721772938


# > res = rma(y, v, data=dat, method="DL", test="knha", control=list(tol=1e-9))
# > convert_items(res, prefix="exk1_dl_hksj.")

exk1_dl_hksj = Holder()
exk1_dl_hksj.b = 62.3901386044504
exk1_dl_hksj.beta = 62.3901386044504
exk1_dl_hksj.se = 0.29477605699879
exk1_dl_hksj.zval = 211.652666908108
exk1_dl_hksj.pval = 2.77938607433693e-16
exk1_dl_hksj.ci_lb = 61.710383798052
exk1_dl_hksj.ci_ub = 63.0698934108488
exk1_dl_hksj.vb = 0.0868929237797541
exk1_dl_hksj.tau2 = 0.288049246973751
exk1_dl_hksj.se_tau2 = 0.269366223207558
exk1_dl_hksj.tau2_f = 0.288049246973751
exk1_dl_hksj.k = 9
exk1_dl_hksj.k_f = 9
exk1_dl_hksj.k_eff = 9
exk1_dl_hksj.k_all = 9
exk1_dl_hksj.p = 1
exk1_dl_hksj.p_eff = 1
exk1_dl_hksj.parms = 2
exk1_dl_hksj.m = 1
exk1_dl_hksj.QE = 24.801897741835
exk1_dl_hksj.QEp = 0.00167935146372742
exk1_dl_hksj.QM = 44796.8514093144
exk1_dl_hksj.QMp = 2.77938607433693e-16
exk1_dl_hksj.I2 = 67.744403741711
exk1_dl_hksj.H2 = 3.10023721772938


# > res = rma(y, v, data=dat, method="FE", control=list(tol=1e-9))
# > convert_items(res, prefix="exk1_fe.")

exk1_fe = Holder()
exk1_fe.b = 62.5833970939982
exk1_fe.beta = 62.5833970939982
exk1_fe.se = 0.107845705498231
exk1_fe.zval = 580.304953311515
exk1_fe.pval = 0
exk1_fe.ci_lb = 62.3720233953344
exk1_fe.ci_ub = 62.7947707926621
exk1_fe.vb = 0.0116306961944112
exk1_fe.tau2 = 0
exk1_fe.tau2_f = 0
exk1_fe.k = 9
exk1_fe.k_f = 9
exk1_fe.k_eff = 9
exk1_fe.k_all = 9
exk1_fe.p = 1
exk1_fe.p_eff = 1
exk1_fe.parms = 1
exk1_fe.m = 1
exk1_fe.QE = 24.801897741835
exk1_fe.QEp = 0.00167935146372742
exk1_fe.QM = 336753.838837879
exk1_fe.QMp = 0
exk1_fe.I2 = 67.744403741711
exk1_fe.H2 = 3.10023721772938


# > res = rma(y, v, data=dat, method="FE", test="knha", control=list(tol=1e-9))
# Warning message:
# In rma(y, v, data = dat, method = "FE", test = "knha",
#        control = list(tol = 1e-09)) :
#  Knapp & Hartung method is not meant to be used in the context of FE models.
# > convert_items(res, prefix="exk1_fe_hksj.")

exk1_fe_hksj = Holder()
exk1_fe_hksj.b = 62.5833970939982
exk1_fe_hksj.beta = 62.5833970939982
exk1_fe_hksj.se = 0.189889223522271
exk1_fe_hksj.zval = 329.57845597098
exk1_fe_hksj.pval = 8.04326466920145e-18
exk1_fe_hksj.ci_lb = 62.1455117593252
exk1_fe_hksj.ci_ub = 63.0212824286713
exk1_fe_hksj.vb = 0.0360579172098909
exk1_fe_hksj.tau2 = 0
exk1_fe_hksj.tau2_f = 0
exk1_fe_hksj.k = 9
exk1_fe_hksj.k_f = 9
exk1_fe_hksj.k_eff = 9
exk1_fe_hksj.k_all = 9
exk1_fe_hksj.p = 1
exk1_fe_hksj.p_eff = 1
exk1_fe_hksj.parms = 1
exk1_fe_hksj.m = 1
exk1_fe_hksj.QE = 24.801897741835
exk1_fe_hksj.QEp = 0.00167935146372742
exk1_fe_hksj.QM = 108621.958640215
exk1_fe_hksj.QMp = 8.04326466920145e-18
exk1_fe_hksj.I2 = 67.744403741711
exk1_fe_hksj.H2 = 3.10023721772938


# effect size for proportions, metafor `escalc` function

# > library(metafor)
# > dat <- dat.fine1993
# > dat_or <- escalc(measure="OR", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_or[c("y2i", "v2i")]
# > cat_items(r)
y_or = np.array([
    0.13613217432458, 0.768370601797533, 0.374938517449009, 1.65822807660353,
    0.784954729813068, 0.361663949151077, 0.575364144903562,
    0.250542525502324, 0.650587566141149, 0.0918075492531228,
    0.273865253802803, 0.485755524477543, 0.182321556793955,
    0.980829253011726, 1.31218638896617, -0.259511195485084, 0.138402322859119
    ])
v_or = np.array([
    0.399242424242424, 0.244867149758454, 0.152761481951271, 0.463095238095238,
    0.189078465394255, 0.0689052107900588, 0.240651709401709,
    0.142027027027027, 0.280657748049052, 0.210140736456526,
    0.0373104717196078, 0.0427774287950624, 0.194901960784314,
    0.509259259259259, 1.39835164835165, 0.365873015873016, 0.108630952380952
    ])

# > dat_rr <- escalc(measure="RR", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_rr[c("y2i", "v2i")]
# > cat_items(r)
y_rr = np.array([
    0.0595920972022457, 0.434452644981417, 0.279313822781264,
    0.934309237376833, 0.389960921572199, 0.219327702635984,
    0.328504066972036, 0.106179852041229, 0.28594445255324,
    0.0540672212702757, 0.164912297594691, 0.300079561474504,
    0.0813456394539525, 0.693147180559945, 0.177206456127184,
    -0.131336002061087, 0.0622131845015728
    ])
v_rr = np.array([
    0.0761562998405104, 0.080905695611578, 0.0856909430438842,
    0.175974025974026, 0.0551968864468864, 0.0267002515563729,
    0.074017094017094, 0.0257850995555914, 0.0590338164251208,
    0.073266499582289, 0.0137191240428942, 0.0179386112192693,
    0.0400361415752742, 0.3, 0.0213675213675214, 0.0922402159244264,
    0.021962676962677
    ])

# > dat_rd <- escalc(measure="RD", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_rd[c("y2i", "v2i")]
# > cat_items(r)
y_rd = np.array([
    0.0334928229665072, 0.186554621848739, 0.071078431372549,
    0.386363636363636, 0.19375, 0.0860946401581211, 0.14, 0.0611028315946349,
    0.158888888888889, 0.0222222222222222, 0.0655096935584741,
    0.114173373020248, 0.045021186440678, 0.2, 0.150793650793651,
    -0.0647773279352226, 0.0342342342342342
    ])
v_rd = np.array([
    0.0240995805934916, 0.0137648162576944, 0.00539777447807907,
    0.0198934072126221, 0.0109664132254464, 0.00376813659489987,
    0.0142233846153846, 0.00842011053321928, 0.0163926076817558,
    0.0122782676856751, 0.00211164860232433, 0.00219739135615223,
    0.0119206723560942, 0.016, 0.014339804116826, 0.0226799351233969,
    0.00663520262409963
    ])


# > dat_as <- escalc(measure="AS", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_as[c("y2i", "v2i")]
# > cat_items(r)
y_as = np.array([
    0.0337617513001424, 0.189280827304914, 0.0815955178338458,
    0.399912703180945, 0.194987153482868, 0.0882233598093272,
    0.141897054604164, 0.0618635353537276, 0.160745373792417,
    0.0225840453649413, 0.0669694915300637, 0.117733830714136,
    0.0452997410410423, 0.221071594001477, 0.220332915310739,
    -0.0648275244591966, 0.0344168494848509
    ])
v_as = np.array([
    0.0245215311004785, 0.0144957983193277, 0.00714869281045752,
    0.0238636363636364, 0.0113839285714286, 0.00402569468666434,
    0.0146153846153846, 0.00864381520119225, 0.0169444444444444,
    0.0126984126984127, 0.0022181832395247, 0.00242071803917245,
    0.0120497881355932, 0.0222222222222222, 0.0317460317460317,
    0.0227732793522267, 0.00671171171171171
    ])

eff_prop1 = Holder(y_rd=y_rd, v_rd=v_rd, y_rr=y_rr, v_rr=v_rr,
                   y_or=y_or, v_or=v_or, y_as=y_as, v_as=v_as)


# package meta metabin OR
NA = np.nan  # for R output
results_or_dl_hk = Holder()
# > res_mb_hk = metabin(e2i, nei, c2i, nci, data=dat2, sm="OR",
#     Q.Cochrane=FALSE, method="Inverse", method.tau="DL", hakn=TRUE,
#     backtransf=FALSE)
# > cat_items(res_mb_hk, prefix="results_or_dl_hk.")
results_or_dl_hk.event_e = np.array([
    18, 22, 21, 14, 42, 80, 13, 37, 23, 19, 106, 170, 34, 18, 13, 12, 42
    ])
results_or_dl_hk.n_e = np.array([
    19, 34, 72, 22, 70, 183, 26, 61, 36, 45, 246, 386, 59, 45, 14, 26, 74
    ])
results_or_dl_hk.event_c = np.array([
    12, 12, 15, 5, 13, 33, 18, 30, 12, 14, 76, 46, 17, 3, 14, 10, 40
    ])
results_or_dl_hk.n_c = np.array([
    22, 35, 68, 20, 32, 94, 50, 55, 25, 35, 208, 141, 32, 15, 18, 19, 75
    ])
results_or_dl_hk.method = 'Inverse'
results_or_dl_hk.incr = 0.5
results_or_dl_hk.Q_CMH = 24.9044036917599
results_or_dl_hk.df_Q_CMH = 1
results_or_dl_hk.pval_Q_CMH = 6.02446516918864e-07
results_or_dl_hk.incr_e = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
results_or_dl_hk.incr_c = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
results_or_dl_hk.studlab = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    ])
results_or_dl_hk.TE = np.array([
    2.70805020110221, 1.25672336971146, 0.374938517449009, 1.65822807660353,
    0.784954729813068, 0.361663949151077, 0.575364144903562,
    0.250542525502324, 0.650587566141149, 0.0918075492531229,
    0.273865253802803, 0.485755524477543, 0.182321556793955,
    0.980829253011726, 1.31218638896617, -0.259511195485085, 0.138402322859119
    ])
results_or_dl_hk.seTE = np.array([
    1.1130538571376, 0.505568465186247, 0.390847133738078, 0.680511012471685,
    0.434831536798166, 0.26249802054503, 0.490562645746402, 0.376864733063505,
    0.529771411128472, 0.458411099840008, 0.193159187510219,
    0.206827050443269, 0.441477021807833, 0.713624032148063, 1.18251919576455,
    0.604874380241894, 0.329592099997789
    ])
results_or_dl_hk.lower = np.array([
    0.526504728259124, 0.265827386227229, -0.391107788138334,
    0.324451001076142, -0.0672994216535403, -0.152822717130236,
    -0.386120972920066, -0.488098778345447, -0.387745319709617,
    -0.806661696546688, -0.104719797000245, 0.0803819545800872,
    -0.682957505951402, -0.41784814850073, -1.00550864575963,
    -1.44504319593018, -0.507586322725471
    ])
results_or_dl_hk.upper = np.array([
    4.8895956739453, 2.2476193531957, 1.14098482303635, 2.99200515213092,
    1.63720888127968, 0.87615061543239, 1.53684926272719, 0.989183829350096,
    1.68892045199192, 0.990276795052934, 0.65245030460585, 0.891129094374998,
    1.04760061953931, 2.37950665452418, 3.62988142369196, 0.926020804960014,
    0.784390968443709
    ])
results_or_dl_hk.zval = np.array([
    2.43299116546472, 2.48576297030017, 0.959297088514073, 2.43673951811695,
    1.80519273186346, 1.37777781485798, 1.17286578970589, 0.664807564946931,
    1.22805336882057, 0.200273399324678, 1.41782152499639, 2.34860731919001,
    0.412980852428863, 1.37443416817026, 1.10965335164625, -0.429033207492279,
    0.419920024964335
    ])
results_or_dl_hk.pval = np.array([
    0.0149746662574385, 0.012927403688425, 0.337409101687073,
    0.0148203508441569, 0.0710445278963136, 0.168271897623853,
    0.240849630303111, 0.50617358344946, 0.21942693438385, 0.841266765353313,
    0.15624287815226, 0.0188437679780689, 0.67962064241999, 0.169306934699306,
    0.267148431658854, 0.667899058451397, 0.674543878468201
    ])
results_or_dl_hk.w_fixed = np.array([
    0.807174887892376, 3.91237113402062, 6.54615278162192, 2.15938303341902,
    5.28880958450166, 14.5126905285409, 4.15538290788013, 7.04091341579448,
    3.5630585898709, 4.75871559633028, 26.8021269609001, 23.3768140855494,
    5.1307847082495, 1.96363636363636, 0.715127701375246, 2.73318872017354,
    9.2054794520548
    ])
results_or_dl_hk.w_random = np.array([
    0.806082838403413, 3.8868480814364, 6.47501150011912, 2.15158503393154,
    5.24227532998639, 14.1675952027965, 4.12660238520421, 6.95867952431796,
    3.54187734484692, 4.72100880622866, 25.6483453428049, 22.4942390635843,
    5.0869781966579, 1.95718594121596, 0.714270384745876, 2.72070780232418,
    9.06541462543771
    ])
results_or_dl_hk.TE_fixed = 0.428036725396544
results_or_dl_hk.seTE_fixed = 0.0902874968199668
results_or_dl_hk.lower_fixed = 0.251076483375135
results_or_dl_hk.upper_fixed = 0.604996967417954
results_or_dl_hk.zval_fixed = 4.74081949851871
results_or_dl_hk.pval_fixed = 2.12855503378502e-06
results_or_dl_hk.TE_random = 0.429520368698268
results_or_dl_hk.seTE_random = 0.0915952752397692
results_or_dl_hk.lower_random = 0.235347059333852
results_or_dl_hk.upper_random = 0.623693678062684
results_or_dl_hk.zval_random = 4.68932887175579
results_or_dl_hk.pval_random = 0.000246175101510513
results_or_dl_hk.null_effect = 0
results_or_dl_hk.seTE_predict = 0.100339885576592
results_or_dl_hk.lower_predict = 0.215650965184522
results_or_dl_hk.upper_predict = 0.643389772212014
results_or_dl_hk.level_predict = 0.95
results_or_dl_hk.k = 17
results_or_dl_hk.Q = 16.181374262823
results_or_dl_hk.df_Q = 16
results_or_dl_hk.pval_Q = 0.440375456698129
results_or_dl_hk.tau2 = 0.0016783981912744
results_or_dl_hk.se_tau2 = 0.0529437644950009
results_or_dl_hk.lower_tau2 = 0
results_or_dl_hk.upper_tau2 = 0.45893520964914
results_or_dl_hk.tau = 0.0409682583383086
results_or_dl_hk.lower_tau = 0
results_or_dl_hk.upper_tau = 0.677447569668045
results_or_dl_hk.method_tau_ci = 'J'
results_or_dl_hk.sign_lower_tau = ''
results_or_dl_hk.sign_upper_tau = ''
results_or_dl_hk.H = 1.00565197331206
results_or_dl_hk.lower_H = 1
results_or_dl_hk.upper_H = 1.43793739981697
results_or_dl_hk.I2 = 0.0112088293538659
results_or_dl_hk.lower_I2 = 0
results_or_dl_hk.upper_I2 = 0.516362418389019
results_or_dl_hk.Rb = 0.0117679262339789
results_or_dl_hk.lower_Rb = 0
results_or_dl_hk.upper_Rb = 0.725656377301252
results_or_dl_hk.approx_TE = np.array([
 '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''
    ])
results_or_dl_hk.approx_seTE = np.array([
 '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''
    ])
results_or_dl_hk.sm = 'OR'
results_or_dl_hk.level = 0.95
results_or_dl_hk.level_comb = 0.95
results_or_dl_hk.df_hakn = 16
results_or_dl_hk.method_tau = 'DL'
results_or_dl_hk.method_bias = 'score'
results_or_dl_hk.title = ''
results_or_dl_hk.complab = ''
results_or_dl_hk.outclab = ''
results_or_dl_hk.label_e = 'Experimental'
results_or_dl_hk.label_c = 'Control'
results_or_dl_hk.label_left = ''
results_or_dl_hk.label_right = ''
results_or_dl_hk.data = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 34, 72, 22,
    70, 183, 26, 61, 36, 45, 246, 386, 59, 45, 14, 26, 74, 22, 35, 68, 20, 32,
    94, 50, 55, 25, 35, 208, 141, 32, 15, 18, 19, 75, 16, 22, 44, 19, 62, 130,
    24, 51, 30, 43, 169, 279, 56, 42, 14, 21, NA, 20, 22, 40, 12, 27, 65, 30,
    44, 17, 35, 139, 97, 30, 10, 18, 15, NA, 18, 22, 21, 14, 42, 80, 13, 37,
    23, 19, 106, 170, 34, 18, 13, 12, 42, 12, 12, 15, 5, 13, 33, 18, 30, 12,
    14, 76, 46, 17, 3, 14, 10, 40, 4, 15, 10, 5, 26, 47, 5, 19, 13, 8, 67, 97,
    21, 9, 12, 6, NA, 8, 8, 3, 4, 6, 14, 10, 19, 4, 4, 42, 21, 9, 1, 13, 4,
    NA, 4, 15, 3, 2, 15, 30, 3, 11, 10, 6, 51, 73, 20, 9, 9, 5, 23, 3, 6, 0,
    3, 5, 11, 9, 15, 4, 0, 35, 8, 7, 1, 12, 1, 30, 18, 22, 21, 14, 42, 80, 13,
    37, 23, 19, 106, 170, 34, 18, 13, 12, 42, 19, 34, 72, 22, 70, 183, 26, 61,
    36, 45, 246, 386, 59, 45, 14, 26, 74, 12, 12, 15, 5, 13, 33, 18, 30, 12,
    14, 76, 46, 17, 3, 14, 10, 40, 22, 35, 68, 20, 32, 94, 50, 55, 25, 35,
    208, 141, 32, 15, 18, 19, 75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ]).reshape(17, 17, order='F')

results_or_dl_hk.byseparator = ' = '
results_or_dl_hk.pscale = 1
results_or_dl_hk.irscale = 1
results_or_dl_hk.irunit = 'person-years'
results_or_dl_hk.version = '4.11-0'
