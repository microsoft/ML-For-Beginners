import numpy as np

from statsmodels.tools.tools import Bunch


epanechnikov_hsheather_q75 = Bunch()
epanechnikov_hsheather_q75.table = np.array([
    [.6440143,  .0122001,   52.79,  0.000,    .6199777,   .6680508],
    [62.39648,   13.5509,    4.60,  0.000,    35.69854,   89.09443]
])
epanechnikov_hsheather_q75.psrsquared = 0.6966
epanechnikov_hsheather_q75.rank = 2
epanechnikov_hsheather_q75.sparsity = 223.784434936344
epanechnikov_hsheather_q75.bwidth = .1090401129546568
# epanechnikov_hsheather_q75.kbwidth = 59.62067927472172  # Stata 12 results
epanechnikov_hsheather_q75.kbwidth = 59.30  # TODO: why do we need lower tol?
epanechnikov_hsheather_q75.df_m = 1
epanechnikov_hsheather_q75.df_r = 233
epanechnikov_hsheather_q75.f_r = .0044685860313942
epanechnikov_hsheather_q75.N = 235
epanechnikov_hsheather_q75.q_v = 745.2352905273438
epanechnikov_hsheather_q75.q = .75
epanechnikov_hsheather_q75.sum_rdev = 43036.06956481934
epanechnikov_hsheather_q75.sum_adev = 13058.50008841318
epanechnikov_hsheather_q75.convcode = 0


biweight_bofinger = Bunch()
biweight_bofinger.table = np.array([
    [.5601805,  .0136491,  41.04,  0.000,  .533289,  .5870719],
    [81.48233,  15.1604,  5.37,  0.000,  51.61335,  111.3513]
 ])
biweight_bofinger.psrsquared = 0.6206
biweight_bofinger.rank = 2
biweight_bofinger.sparsity = 216.8218989750115
biweight_bofinger.bwidth = .2173486679767846
biweight_bofinger.kbwidth = 91.50878448104551
biweight_bofinger.df_m = 1
biweight_bofinger.df_r = 233
biweight_bofinger.f_r = .0046120802590851
biweight_bofinger.N = 235
biweight_bofinger.q_v = 582.541259765625
biweight_bofinger.q = .5
biweight_bofinger.sum_rdev = 46278.05667114258
biweight_bofinger.sum_adev = 17559.93220318131
biweight_bofinger.convcode = 0


biweight_hsheather = Bunch()
biweight_hsheather.table = np.array([
    [.5601805,  .0128449,  43.61,  0.000,  .5348735,  .5854875],
    [81.48233,  14.26713,  5.71,  0.000,  53.37326,  109.5914]
 ])
biweight_hsheather.psrsquared = 0.6206
biweight_hsheather.rank = 2
biweight_hsheather.sparsity = 204.0465407204423
biweight_hsheather.bwidth = .1574393314202373
biweight_hsheather.kbwidth = 64.53302151153288
biweight_hsheather.df_m = 1
biweight_hsheather.df_r = 233
biweight_hsheather.f_r = .0049008427022052
biweight_hsheather.N = 235
biweight_hsheather.q_v = 582.541259765625
biweight_hsheather.q = .5
biweight_hsheather.sum_rdev = 46278.05667114258
biweight_hsheather.sum_adev = 17559.93220318131
biweight_hsheather.convcode = 0

biweight_chamberlain = Bunch()
biweight_chamberlain.table = np.array([
    [.5601805,  .0114969,  48.72,  0.000,  .5375294,  .5828315],
    [81.48233,  12.76983,  6.38,  0.000,  56.32325,  106.6414]
])
biweight_chamberlain.psrsquared = 0.6206
biweight_chamberlain.rank = 2
biweight_chamberlain.sparsity = 182.6322495257494
biweight_chamberlain.bwidth = .063926976464458
biweight_chamberlain.kbwidth = 25.61257055690209
biweight_chamberlain.df_m = 1
biweight_chamberlain.df_r = 233
biweight_chamberlain.f_r = .005475484218131
biweight_chamberlain.N = 235
biweight_chamberlain.q_v = 582.541259765625
biweight_chamberlain.q = .5
biweight_chamberlain.sum_rdev = 46278.05667114258
biweight_chamberlain.sum_adev = 17559.93220318131
biweight_chamberlain.convcode = 0

epanechnikov_bofinger = Bunch()
epanechnikov_bofinger.table = np.array([
    [.5601805,  .0209663,  26.72,  0.000,  .5188727,  .6014882],
    [81.48233,  23.28774,  3.50,  0.001,  35.60088,  127.3638]
    ])
epanechnikov_bofinger.psrsquared = 0.6206
epanechnikov_bofinger.rank = 2
epanechnikov_bofinger.sparsity = 333.0579553401614
epanechnikov_bofinger.bwidth = .2173486679767846
epanechnikov_bofinger.kbwidth = 91.50878448104551
epanechnikov_bofinger.df_m = 1
epanechnikov_bofinger.df_r = 233
epanechnikov_bofinger.f_r = .0030024804511235
epanechnikov_bofinger.N = 235
epanechnikov_bofinger.q_v = 582.541259765625
epanechnikov_bofinger.q = .5
epanechnikov_bofinger.sum_rdev = 46278.05667114258
epanechnikov_bofinger.sum_adev = 17559.93220318131
epanechnikov_bofinger.convcode = 0

epanechnikov_hsheather = Bunch()
epanechnikov_hsheather.table = np.array([
    [.5601805,  .0170484,  32.86,  0.000,  .5265918,  .5937692],
    [81.48233,  18.93605,  4.30,  0.000,  44.17457,  118.7901]
    ])
epanechnikov_hsheather.psrsquared = 0.6206
epanechnikov_hsheather.rank = 2
epanechnikov_hsheather.sparsity = 270.8207209067576
epanechnikov_hsheather.bwidth = .1574393314202373
epanechnikov_hsheather.kbwidth = 64.53302151153288
epanechnikov_hsheather.df_m = 1
epanechnikov_hsheather.df_r = 233
epanechnikov_hsheather.f_r = .0036924796472434
epanechnikov_hsheather.N = 235
epanechnikov_hsheather.q_v = 582.541259765625
epanechnikov_hsheather.q = .5
epanechnikov_hsheather.sum_rdev = 46278.05667114258
epanechnikov_hsheather.sum_adev = 17559.93220318131
epanechnikov_hsheather.convcode = 0

epanechnikov_chamberlain = Bunch()
epanechnikov_chamberlain.table = np.array([
    [.5601805,  .0130407,  42.96,  0.000,  .5344876,  .5858733],
    [81.48233,  14.48467,  5.63,  0.000,  52.94468,  110.02]
    ])
epanechnikov_chamberlain.psrsquared = 0.6206
epanechnikov_chamberlain.rank = 2
epanechnikov_chamberlain.sparsity = 207.1576340635951
epanechnikov_chamberlain.bwidth = .063926976464458
epanechnikov_chamberlain.kbwidth = 25.61257055690209
epanechnikov_chamberlain.df_m = 1
epanechnikov_chamberlain.df_r = 233
epanechnikov_chamberlain.f_r = .0048272418466269
epanechnikov_chamberlain.N = 235
epanechnikov_chamberlain.q_v = 582.541259765625
epanechnikov_chamberlain.q = .5
epanechnikov_chamberlain.sum_rdev = 46278.05667114258
epanechnikov_chamberlain.sum_adev = 17559.93220318131
epanechnikov_chamberlain.convcode = 0

epan2_bofinger = Bunch()
epan2_bofinger.table = np.array([
    [.5601805,  .0143484,  39.04,  0.000,  .5319113,  .5884496],
    [81.48233,  15.93709,  5.11,  0.000,  50.08313,  112.8815]
    ])
epan2_bofinger.psrsquared = 0.6206
epan2_bofinger.rank = 2
epan2_bofinger.sparsity = 227.9299402797656
epan2_bofinger.bwidth = .2173486679767846
epan2_bofinger.kbwidth = 91.50878448104551
epan2_bofinger.df_m = 1
epan2_bofinger.df_r = 233
epan2_bofinger.f_r = .0043873130435281
epan2_bofinger.N = 235
epan2_bofinger.q_v = 582.541259765625
epan2_bofinger.q = .5
epan2_bofinger.sum_rdev = 46278.05667114258
epan2_bofinger.sum_adev = 17559.93220318131
epan2_bofinger.convcode = 0

epan2_hsheather = Bunch()
epan2_hsheather.table = np.array([
    [.5601805,  .0131763,   42.51,  0.000,   .5342206,   .5861403],
    [81.48233,  14.63518,    5.57,  0.000,   52.64815,   110.3165]
    ])
epan2_hsheather.psrsquared = 0.6206
epan2_hsheather.rank = 2
epan2_hsheather.sparsity = 209.3102085912557
epan2_hsheather.bwidth = .1574393314202373
epan2_hsheather.kbwidth = 64.53302151153288
epan2_hsheather.df_m = 1
epan2_hsheather.df_r = 233
epan2_hsheather.f_r = .0047775978378236
epan2_hsheather.N = 235
epan2_hsheather.q_v = 582.541259765625
epan2_hsheather.q = .5
epan2_hsheather.sum_rdev = 46278.05667114258
epan2_hsheather.sum_adev = 17559.93220318131
epan2_hsheather.convcode = 0

epan2_chamberlain = Bunch()
epan2_chamberlain.table = np.array([
    [.5601805,  .0117925,  47.50,  0.000,  .5369469,  .583414],
    [81.48233,  13.0982,  6.22,  0.000,  55.67629,  107.2884]
    ])
epan2_chamberlain.psrsquared = 0.6206
epan2_chamberlain.rank = 2
epan2_chamberlain.sparsity = 187.3286437436797
epan2_chamberlain.bwidth = .063926976464458
epan2_chamberlain.kbwidth = 25.61257055690209
epan2_chamberlain.df_m = 1
epan2_chamberlain.df_r = 233
epan2_chamberlain.f_r = .0053382119253919
epan2_chamberlain.N = 235
epan2_chamberlain.q_v = 582.541259765625
epan2_chamberlain.q = .5
epan2_chamberlain.sum_rdev = 46278.05667114258
epan2_chamberlain.sum_adev = 17559.93220318131
epan2_chamberlain.convcode = 0


rectangle_bofinger = Bunch()
rectangle_bofinger.table = np.array([
    [.5601805,  .0158331,  35.38,  0.000,  .5289861,  .5913748],
    [81.48233,  17.5862,  4.63,  0.000,  46.83404,  116.1306]
    ])
rectangle_bofinger.psrsquared = 0.6206
rectangle_bofinger.rank = 2
rectangle_bofinger.sparsity = 251.515372550242
rectangle_bofinger.bwidth = .2173486679767846
rectangle_bofinger.kbwidth = 91.50878448104551
rectangle_bofinger.df_m = 1
rectangle_bofinger.df_r = 233
rectangle_bofinger.f_r = .0039759001203803
rectangle_bofinger.N = 235
rectangle_bofinger.q_v = 582.541259765625
rectangle_bofinger.q = .5
rectangle_bofinger.sum_rdev = 46278.05667114258
rectangle_bofinger.sum_adev = 17559.93220318131
rectangle_bofinger.convcode = 0

rectangle_hsheather = Bunch()
rectangle_hsheather.table = np.array([
    [.5601805,  .0137362,  40.78,  0.000,  .5331174,  .5872435],
    [81.48233,  15.25712,  5.34,  0.000,  51.42279,  111.5419]
    ])
rectangle_hsheather.psrsquared = 0.6206
rectangle_hsheather.rank = 2
rectangle_hsheather.sparsity = 218.2051806505069
rectangle_hsheather.bwidth = .1574393314202373
rectangle_hsheather.kbwidth = 64.53302151153288
rectangle_hsheather.df_m = 1
rectangle_hsheather.df_r = 233
rectangle_hsheather.f_r = .004582842611797
rectangle_hsheather.N = 235
rectangle_hsheather.q_v = 582.541259765625
rectangle_hsheather.q = .5
rectangle_hsheather.sum_rdev = 46278.05667114258
rectangle_hsheather.sum_adev = 17559.93220318131
rectangle_hsheather.convcode = 0

rectangle_chamberlain = Bunch()
rectangle_chamberlain.table = np.array([
    [.5601805,  .0118406,  47.31,  0.000,  .5368522,  .5835087],
    [81.48233,  13.1516,  6.20,  0.000,  55.57108,  107.3936]
    ])
rectangle_chamberlain.psrsquared = 0.6206
rectangle_chamberlain.rank = 2
rectangle_chamberlain.sparsity = 188.0923150272497
rectangle_chamberlain.bwidth = .063926976464458
rectangle_chamberlain.kbwidth = 25.61257055690209
rectangle_chamberlain.df_m = 1
rectangle_chamberlain.df_r = 233
rectangle_chamberlain.f_r = .0053165383171297
rectangle_chamberlain.N = 235
rectangle_chamberlain.q_v = 582.541259765625
rectangle_chamberlain.q = .5
rectangle_chamberlain.sum_rdev = 46278.05667114258
rectangle_chamberlain.sum_adev = 17559.93220318131
rectangle_chamberlain.convcode = 0

triangle_bofinger = Bunch()
triangle_bofinger.table = np.array([
    [.5601805,  .0138712,  40.38,  0.000,  .5328515,  .5875094],
    [81.48233,  15.40706,  5.29,  0.000,  51.12738,  111.8373]
    ])
triangle_bofinger.psrsquared = 0.6206
triangle_bofinger.rank = 2
triangle_bofinger.sparsity = 220.3495620604223
triangle_bofinger.bwidth = .2173486679767846
triangle_bofinger.kbwidth = 91.50878448104551
triangle_bofinger.df_m = 1
triangle_bofinger.df_r = 233
triangle_bofinger.f_r = .0045382436463649
triangle_bofinger.N = 235
triangle_bofinger.q_v = 582.541259765625
triangle_bofinger.q = .5
triangle_bofinger.sum_rdev = 46278.05667114258
triangle_bofinger.sum_adev = 17559.93220318131
triangle_bofinger.convcode = 0

triangle_hsheather = Bunch()
triangle_hsheather.table = np.array([
    [.5601805,  .0128874,  43.47,  0.000,  .5347898,  .5855711],
    [81.48233,  14.31431,  5.69,  0.000,  53.2803,  109.6844]
    ])
triangle_hsheather.psrsquared = 0.6206
triangle_hsheather.rank = 2
triangle_hsheather.sparsity = 204.7212998199564
triangle_hsheather.bwidth = .1574393314202373
triangle_hsheather.kbwidth = 64.53302151153288
triangle_hsheather.df_m = 1
triangle_hsheather.df_r = 233
triangle_hsheather.f_r = .004884689579831
triangle_hsheather.N = 235
triangle_hsheather.q_v = 582.541259765625
triangle_hsheather.q = .5
triangle_hsheather.sum_rdev = 46278.05667114258
triangle_hsheather.sum_adev = 17559.93220318131
triangle_hsheather.convcode = 0

triangle_chamberlain = Bunch()
triangle_chamberlain.table = np.array([
    [.5601805,  .0115725,  48.41,  0.000,  .5373803,  .5829806],
    [81.48233,  12.85389,  6.34,  0.000,  56.15764,  106.807]
    ])
triangle_chamberlain.psrsquared = 0.6206
triangle_chamberlain.rank = 2
triangle_chamberlain.sparsity = 183.8344452913298
triangle_chamberlain.bwidth = .063926976464458
triangle_chamberlain.kbwidth = 25.61257055690209
triangle_chamberlain.df_m = 1
triangle_chamberlain.df_r = 233
triangle_chamberlain.f_r = .0054396769790083
triangle_chamberlain.N = 235
triangle_chamberlain.q_v = 582.541259765625
triangle_chamberlain.q = .5
triangle_chamberlain.sum_rdev = 46278.05667114258
triangle_chamberlain.sum_adev = 17559.93220318131
triangle_chamberlain.convcode = 0

gaussian_bofinger = Bunch()
gaussian_bofinger.table = np.array([
    [.5601805,  .0197311,  28.39,  0.000,  .5213062,  .5990547],
    [81.48233,  21.91582,  3.72,  0.000,  38.30383,  124.6608]
    ])
gaussian_bofinger.psrsquared = 0.6206
gaussian_bofinger.rank = 2
gaussian_bofinger.sparsity = 313.4370075776719
gaussian_bofinger.bwidth = .2173486679767846
gaussian_bofinger.kbwidth = 91.50878448104551
gaussian_bofinger.df_m = 1
gaussian_bofinger.df_r = 233
gaussian_bofinger.f_r = .0031904337261521
gaussian_bofinger.N = 235
gaussian_bofinger.q_v = 582.541259765625
gaussian_bofinger.q = .5
gaussian_bofinger.sum_rdev = 46278.05667114258
gaussian_bofinger.sum_adev = 17559.93220318131
gaussian_bofinger.convcode = 0

gaussian_hsheather = Bunch()
gaussian_hsheather.table = np.array([
    [.5601805,  .016532,  33.88,  0.000,  .5276092,  .5927518],
    [81.48233,  18.36248,  4.44,  0.000,  45.30462,  117.66]
    ])
gaussian_hsheather.psrsquared = 0.6206
gaussian_hsheather.rank = 2
gaussian_hsheather.sparsity = 262.6175743002715
gaussian_hsheather.bwidth = .1574393314202373
gaussian_hsheather.kbwidth = 64.53302151153288
gaussian_hsheather.df_m = 1
gaussian_hsheather.df_r = 233
gaussian_hsheather.f_r = .0038078182797341
gaussian_hsheather.N = 235
gaussian_hsheather.q_v = 582.541259765625
gaussian_hsheather.q = .5
gaussian_hsheather.sum_rdev = 46278.05667114258
gaussian_hsheather.sum_adev = 17559.93220318131
gaussian_hsheather.convcode = 0

gaussian_chamberlain = Bunch()
gaussian_chamberlain.table = np.array([
    [.5601805,  .0128123,  43.72,  0.000,  .5349378,  .5854232],
    [81.48233,  14.23088,  5.73,  0.000,  53.44468,  109.52]
    ])
gaussian_chamberlain.psrsquared = 0.6206
gaussian_chamberlain.rank = 2
gaussian_chamberlain.sparsity = 203.5280962791137
gaussian_chamberlain.bwidth = .063926976464458
gaussian_chamberlain.kbwidth = 25.61257055690209
gaussian_chamberlain.df_m = 1
gaussian_chamberlain.df_r = 233
gaussian_chamberlain.f_r = .004913326554328
gaussian_chamberlain.N = 235
gaussian_chamberlain.q_v = 582.541259765625
gaussian_chamberlain.q = .5
gaussian_chamberlain.sum_rdev = 46278.05667114258
gaussian_chamberlain.sum_adev = 17559.93220318131
gaussian_chamberlain.convcode = 0

cosine_bofinger = Bunch()
cosine_bofinger.table = np.array([
    [.5601805,  .0121011,  46.29,  0.000,  .536339,  .5840219],
    [81.48233,  13.44092,  6.06,  0.000,  55.00106,  107.9636]
    ])
cosine_bofinger.psrsquared = 0.6206
cosine_bofinger.rank = 2
cosine_bofinger.sparsity = 192.2302014415605
cosine_bofinger.bwidth = .2173486679767846
cosine_bofinger.kbwidth = 91.50878448104551
cosine_bofinger.df_m = 1
cosine_bofinger.df_r = 233
cosine_bofinger.f_r = .0052020961976883
cosine_bofinger.N = 235
cosine_bofinger.q_v = 582.541259765625
cosine_bofinger.q = .5
cosine_bofinger.sum_rdev = 46278.05667114258
cosine_bofinger.sum_adev = 17559.93220318131
cosine_bofinger.convcode = 0

cosine_hsheather = Bunch()
cosine_hsheather.table = np.array([
    [.5601805,  .0116679,  48.01,  0.000,  .5371924,  .5831685],
    [81.48233,  12.9598,  6.29,  0.000,  55.94897,  107.0157]
])
cosine_hsheather.psrsquared = 0.6206
cosine_hsheather.rank = 2
cosine_hsheather.sparsity = 185.349198428224
cosine_hsheather.bwidth = .1574393314202373
cosine_hsheather.kbwidth = 64.53302151153288
cosine_hsheather.df_m = 1
cosine_hsheather.df_r = 233
cosine_hsheather.f_r = .0053952216059205
cosine_hsheather.N = 235
cosine_hsheather.q_v = 582.541259765625
cosine_hsheather.q = .5
cosine_hsheather.sum_rdev = 46278.05667114258
cosine_hsheather.sum_adev = 17559.93220318131
cosine_hsheather.convcode = 0

cosine_chamberlain = Bunch()
cosine_chamberlain.table = np.array([
    [.5601805,  .0106479,  52.61,  0.000,  .539202,  .5811589],
    [81.48233,  11.82688,  6.89,  0.000,  58.18104,  104.7836]
])
cosine_chamberlain.psrsquared = 0.6206
cosine_chamberlain.rank = 2
cosine_chamberlain.sparsity = 169.1463943762948
cosine_chamberlain.bwidth = .063926976464458
cosine_chamberlain.kbwidth = 25.61257055690209
cosine_chamberlain.df_m = 1
cosine_chamberlain.df_r = 233
cosine_chamberlain.f_r = .0059120385254878
cosine_chamberlain.N = 235
cosine_chamberlain.q_v = 582.541259765625
cosine_chamberlain.q = .5
cosine_chamberlain.sum_rdev = 46278.05667114258
cosine_chamberlain.sum_adev = 17559.93220318131
cosine_chamberlain.convcode = 0

parzen_bofinger = Bunch()
parzen_bofinger.table = np.array([
    [.5601805,  .012909,  43.39,  0.000,  .5347471,  .5856138],
    [81.48233,  14.33838,  5.68,  0.000,  53.23289,  109.7318]
])
parzen_bofinger.psrsquared = 0.6206
parzen_bofinger.rank = 2
parzen_bofinger.sparsity = 205.0654663067616
parzen_bofinger.bwidth = .2173486679767846
parzen_bofinger.kbwidth = 91.50878448104551
parzen_bofinger.df_m = 1
parzen_bofinger.df_r = 233
parzen_bofinger.f_r = .0048764914834762
parzen_bofinger.N = 235
parzen_bofinger.q_v = 582.541259765625
parzen_bofinger.q = .5
parzen_bofinger.sum_rdev = 46278.05667114258
parzen_bofinger.sum_adev = 17559.93220318131
parzen_bofinger.convcode = 0

parzen_hsheather = Bunch()
parzen_hsheather.table = np.array([
    [.5601805,  .0122688,  45.66,  0.000,  .5360085,  .5843524],
    [81.48233,  13.62723,  5.98,  0.000,  54.63401,  108.3307]
])
parzen_hsheather.psrsquared = 0.6206
parzen_hsheather.rank = 2
parzen_hsheather.sparsity = 194.8946558099188
parzen_hsheather.bwidth = .1574393314202373
parzen_hsheather.kbwidth = 64.53302151153288
parzen_hsheather.df_m = 1
parzen_hsheather.df_r = 233
parzen_hsheather.f_r = .0051309770185556
parzen_hsheather.N = 235
parzen_hsheather.q_v = 582.541259765625
parzen_hsheather.q = .5
parzen_hsheather.sum_rdev = 46278.05667114258
parzen_hsheather.sum_adev = 17559.93220318131
parzen_hsheather.convcode = 0

parzen_chamberlain = Bunch()
parzen_chamberlain.table = np.array([
    [.5601805,  .0110507,  50.69,  0.000,  .5384084,  .5819526],
    [81.48233,  12.2743,  6.64,  0.000,  57.29954,  105.6651]
])
parzen_chamberlain.psrsquared = 0.6206
parzen_chamberlain.rank = 2
parzen_chamberlain.sparsity = 175.5452813763412
parzen_chamberlain.bwidth = .063926976464458
parzen_chamberlain.kbwidth = 25.61257055690209
parzen_chamberlain.df_m = 1
parzen_chamberlain.df_r = 233
parzen_chamberlain.f_r = .0056965359146063
parzen_chamberlain.N = 235
parzen_chamberlain.q_v = 582.541259765625
parzen_chamberlain.q = .5
parzen_chamberlain.sum_rdev = 46278.05667114258
parzen_chamberlain.sum_adev = 17559.93220318131
parzen_chamberlain.convcode = 0

Rquantreg = Bunch()
Rquantreg.fittedvalues = np.array([
    278.946531823426, 327.662259651587, 472.195784028597, 366.902127539958,
    411.817682123087, 490.131199885949, 443.36524597881, 503.536477958636,
    636.406081281679, 709.736288922034, 312.165058899648, 357.917286612496,
    427.907157504212, 333.474578745265, 396.777813086185, 447.125068738706,
    325.117049130677, 349.771067249961, 481.598886608367, 306.106158691415,
    388.420502955027, 511.05437589194, 313.836609745169, 372.960145596262,
    485.358358918327, 284.379882747628, 346.21761302202, 470.314386890694,
    292.735362869831, 345.174109237497, 431.875199165716, 312.003504742171,
    396.809344806674, 474.141604734191, 463.93526593027, 430.280150030025,
    453.602705891221, 579.151509166254, 320.586493222875, 379.637682965454,
    261.63071606774, 452.262394881918, 560.558633285135, 361.453261675451,
    433.779038355879, 334.560374744198, 465.46340752116, 615.361560833631,
    934.235038902725, 699.263962186247, 403.470311834164, 431.875199165716,
    610.619729234566, 592.662336978523, 365.021855935831, 494.3226340628,
    571.610146001548, 820.79435094965, 1244.13870203673, 480.7121636877,
    1031.46758784013, 362.238578597709, 467.705425607653, 577.073721157792,
    591.087071047729, 323.397708458532, 569.0193777981, 547.264281981988,
    304.861831677351, 727.261856258699, 382.899300447742, 380.080019200391,
    387.548077543462, 455.939284414524, 461.007422390708, 469.504909066202,
    571.628605113602, 482.528691661418, 447.239662606701, 443.405235218729,
    617.975233440748, 888.754433927184, 390.908239427091, 479.934781986261,
    872.520684097057, 468.389025193617, 467.583874563025, 600.891141536243,
    328.94286096288, 524.54251098332, 697.904660068404, 443.782036008026,
    501.879849409525, 647.703493205875, 458.730704476452, 401.721440320659,
    507.49016247142, 722.834547664808, 380.206920088912, 481.131145133749,
    299.904199685339, 347.488751302111, 488.171723133137, 368.620444647759,
    1135.31179977255, 831.440717300088, 578.509405496803, 437.83023460596,
    618.056493326843, 550.238043601025, 289.910800377653, 583.813030650723,
    502.185524461957, 519.079736225537, 280.441976767315, 334.638935071642,
    489.10661171308, 651.842445716676, 1050.52148262534, 346.050058523346,
    729.252186278533, 558.451123252156, 529.859949563712, 668.51276154189,
    1113.66644452647, 747.093352541566, 858.247029841287, 917.230577685989,
    390.29683477012, 339.650965997876, 350.536705877117, 295.394850812194,
    502.711215489629, 465.29644549865, 398.753783734842, 328.660789278315,
    747.883662637697, 2102.02108846944, 499.727307212846, 278.882398125705,
    335.933513242034, 387.198765462081, 672.019128653175, 439.917678218736,
    461.755563134383, 669.580269655421, 614.886877182468, 657.691667171562,
    913.373865558921, 596.066409235456, 562.142375003233, 844.328711460744,
    517.593907165227, 463.853902738299, 1087.39298812569, 583.363256443676,
    601.337926899021, 720.657238588974, 386.421327566452, 493.072243052398,
    493.072243052398, 493.072243052398, 487.416418391789, 625.768736956952,
    717.442211022949, 649.545102325439, 315.764483006998, 387.279123698926,
    725.770230770932, 534.438688677752, 265.753297513836, 265.753297513836,
    275.266770483504, 444.714192650306, 357.226955834115, 466.694950404995,
    521.878226495306, 514.492832530541, 401.82351113515, 308.781536524607,
    410.920025494164, 506.942199600954, 426.771590593594, 444.329655290455,
    567.951437149727, 314.044484679614, 341.637888287124, 390.011222091572,
    371.617953454683, 491.650387255751, 309.86707973756, 339.215312155678,
    401.313514123063, 274.198577993648, 366.737853359169, 602.621623875386,
    397.5527766612, 431.743315417498, 502.103203876367, 758.493699184236,
    354.616793574895, 495.299694296035, 445.994704272822, 461.179388784645,
    492.398877810457, 300.177654706392, 351.604018662754, 397.001476569931,
    443.325261537312, 495.754449130328, 597.467889923833, 495.384438403965,
    563.913157758648, 890.789267794131, 326.949960612346, 296.399786939819,
    336.191062827908, 406.145865402583, 678.81933687482, 997.557647439337,
    365.6649642219, 415.242904378452, 543.581822640472, 310.924419326878,
    519.951702825405, 751.022676264054, 422.151172626615, 604.684498888169,
    836.51478813311, 277.051441764913, 287.126391607546, 327.57744312456,
    343.712072517474, 408.684566875422, 535.079861284096
])

Rquantreg.residuals = np.array([
    -23.1071072288498, -16.7035925924416, 13.4842301424879,
    36.0952280042049, 83.7430928106889, 143.666615246444, 187.391321726407,
    196.904426307396, 194.552540223671, 105.623928368427, 25.8363284299391,
    54.4440518161423, 92.0934604769094, 118.926894162067, 115.9422447507,
    211.714461564889, 67.4824476183351, 93.7875665496341, 158.517491105043,
    27.7332282843317, 78.537815573821, 32.3425283712905, 3.88323401131549,
    51.360750756311, 33.6032971350797, 53.6215045819588, 73.423561609355,
    6.00566204286901, 93.6248004402887, 78.1042405680565, 71.4819720245138,
    42.635363803128, 100.508824022316, 114.377859224531, 190.661878666847,
    120.44727756794, 74.7742708230821, 61.3298388891191, 80.7338623237625,
    56.3613389354645, 14.9298935780977, 136.086423177107, 103.639168918795,
    83.406904735325, 29.1204765079443, 43.2188638018142, 87.68702809519,
    195.534593927816, 133.719017238015, 350.614829501073, 119.230897841979,
    140.205463451968, 296.7772153462, 218.915256717207, 62.7756642652107,
    155.675830060721, 288.990008558354, 322.6267348787, 788.540488171593,
    109.906163322267, 538.923550476946, 121.241446063999, 132.774973479468,
    119.128384587689, 183.709108010048, 67.2007219409504, 43.5425230980599,
    161.49786961119, -7.94264590491014, 344.200840210711, 113.698279589573,
    123.317422174319, -29.9069666643728, -25.6016916976065,
    163.691619008174, 113.036341875648, 8.59293713333579, 61.3520511228119,
    141.397517015987, 184.594660239251, 94.1259408216085, 79.6405061341515,
    91.6733481942127, 113.234603595283, 161.045070172193, 225.290448255665,
    226.095598886257, 160.387924286533, 32.4551953759993, 103.909707216036,
    73.543904350086, 313.336630439861, 319.717144678425, 374.616677214425,
    220.710022883649, 137.027675366142, 172.507934775539, 254.168723294869,
    180.994568087336, 247.26860148984, 72.4144228992704, 14.0322012670994,
    29.7478676180694, 91.1972063421171, -271.391948301552,
    -1.13686837721616e-13, -43.7483617057195, -45.7799925843755,
    316.918702117259, 263.070052700293, -26.2008042070246,
    185.270818370404, 128.400761958987, 126.90766445377, 39.11640958216,
    13.8128950328002, 125.400191450337, 10.1671163220818, 453.849292287238,
    60.1679571929374, -37.0832869573327, 29.6859273527971,
    -18.5990635840888, 32.0472277998125, 187.478653019356,
    131.972664872163, 54.6380484088667, 592.550594160211, 93.7637141660844,
    60.0193510971532, 93.5633996863289, -46.5847395899585,
    25.0901440883857, 35.3349035566588, 38.0569510057356, 46.1382548463761,
    -21.4915275364559, -274.821124029843, 23.7637774954742,
    56.1174237129451, 137.267370647069, 194.004180243594, 257.734838779675,
    151.279738459292, 175.792712085095, 5.37066578891586, 161.87201799729,
    301.825301306357, 337.590468355399, 141.753670144823, 248.534867350933,
    138.672153434971, 191.302922030722, 169.266111420092, 337.411668163301,
    247.595365061674, 324.241547334534, 441.345158140916,
    -2.96336484193375, 128.045086149367, 128.045086149367,
    128.045086149367, 61.183812841238, 119.466557499153, 120.358284525061,
    145.795139311381, 102.833084929086, 121.51832618868, 157.507780852421,
    208.08888035649, -23.4330955930956, -23.4330955930956,
    -9.26578861762408, 170.044609958764, 28.0914420512634,
    48.9250186205624, 186.600476930648, 219.742798279756, 31.1774726408891,
    18.6372355131512, 18.1199081442213, 112.698628091063,
    -25.9726125403696, 176.470984835994, 252.045003846191,
    46.8335329758523, 54.1229163256514, 51.9888300300891, 32.4204776768067,
    179.14892185478, -12.2969284011601, 14.2728509760057,
    -17.3759292437432, 10.6022252751131, 64.362111397076, 198.730134737533,
    50.898482335236, 146.167755331756, 68.4178084237137, 106.826836694182,
    89.9409707510827, 185.120132493866, 130.283190180665, 170.618786347098,
    116.24297250849, 0.822265604206791, 26.3943953963694, 0,
    145.19420242141, 186.007125906439, 209.892380176326, 201.416658681816,
    247.283084092749, 414.930873548859, 115.050091509315, 57.2015105045924,
    131.809734683313, 120.611481397197, 211.419693391706, 321.245634797383,
    -34.6644264712713, 1.15862100546684, 53.2587319600932,
    97.5747986822971, 255.069199707627, 387.139369759176, 63.3685935821613,
    168.076643464964, 157.44822913826, 28.387531965303, 19.3926869633912,
    -28.3781151570223, 124.288724993746, 113.917339005042, 215.240302135105
])
