"""
Results for VARMAX tests

Results from Stata using script `test_varmax_stata.do`.
See also Stata time series documentation, in particular `dfactor`.

Data from:

http://www.jmulti.de/download/datasets/e1.dat

Author: Chad Fulton
License: Simplified-BSD
"""

lutkepohl_dfm = {
    'params': [
        .0063728, .00660177, .00636009,   # Factor loadings
        .00203899, .00009016, .00005348,    # Idiosyncratic variances
        .33101874, .63927819,             # Factor transitions
    ],
    'bse_oim': [
        .002006,  .0012514, .0012128,   # Factor loadings
        .0003359, .0000184, .0000141,   # Idiosyncratic variances
        .1196637, .1218577,             # Factor transitions
    ],
    'loglike': 594.0902026190786,
    'aic': -1172.18,
    'bic': -1153.641,
}

lutkepohl_dfm2 = {
    'params': [
        .03411188, .03478764,  # Factor loadings: y1
        .03553366, .0344871,  # Factor loadings: y2
        .03536757, .03433391,  # Factor loadings: y3
        .00224401, .00014678, .00010922,   # Idiosyncratic variances
        .08845946, .08862982,      # Factor transitions: Phi, row 1
        .08754759, .08758589    # Phi, row 2
    ],
    'bse_oim': None,
    'loglike': 496.379832917306,
    'aic': -974.7597,
    'bic': -953.9023,
}

lutkepohl_dfm_exog1 = {
    'params': [
        -.01254697, -.00734604, -.00671296,  # Factor loadings
        .01803325, .02066737, .01983089,     # Beta.constant
        .00198667, .00008426, .00005684,     # Idiosyncratic variances
        .31140829,                           # Factor transition
    ],
    'var_oim': [
        .00004224, 2.730e-06, 3.625e-06,
        .00003087, 2.626e-06, 2.013e-06,
        1.170e-07, 5.133e-10, 3.929e-10,
        .07412117
    ],
    'loglike': 596.9781590009525,
    'aic': -1173.956,
    'bic': -1150.781,
}

lutkepohl_dfm_exog2 = {
    'params': [
        .01249096, .00731147, .00680776,  # Factor loadings
        .02187812, -.00009851,  # Betas, y1
        .02302646, -.00006045,  # Betas, y2
        .02009233, -6.683e-06,  # Betas, y3
        .0019856, .00008378, .00005581,     # Idiosyncratic variances
        .2995768,                           # Factor transition
    ],
    'var_oim': [
        .00004278,  2.659e-06, 3.766e-06,
        .00013003, 6.536e-08,
        .00001079, 5.424e-09,
        8.393e-06, 4.217e-09,
        1.168e-07, 5.140e-10, 4.181e-10,
        .07578382,
    ],
    'loglike': 597.4550537198315,
    'aic': -1168.91,
    'bic': -1138.783,
}

lutkepohl_dfm_gen = {
    'params': [
        .00312295, .00332555, .00318837,    # Factor loadings
        # .00195462,                        # Covariance, lower triangle
        #  3.642e-06, .00010047,
        # .00007018,  .00002565, .00006118
        # Note: the following are the Cholesky of the covariance
        # matrix defined just above
        .04421108,                          # Cholesky, lower triangle
        .00008238,  .01002313,
        .00158738,  .00254603,  .00722343,
        .987374,                            # Factor transition
        -.25613562, .00392166, .44859028,   # Error transition parameters
        .01635544, -.249141, .08170863,
        -.02280001, .02059063, -.41808254
    ],
    'var_oim': [
        1.418e-06, 1.030e-06, 9.314e-07,   # Factor loadings
        None,                               # Cholesky, lower triangle
        None, None,
        None, None, None,
        .00021421,                          # Factor transition
        .01307587, .29167522, .43204063,    # Error transition parameters
        .00076899, .01742173, .0220161,
        .00055435, .01456365, .01707167
    ],
    'loglike': 607.7715711926285,
    'aic': -1177.543,
    'bic': -1133.511,
}

lutkepohl_dfm_ar2 = {
    'params': [
        .00419132, .0044007, .00422976,    # Factor loadings
        .00188101, .0000786, .0000418,     # Idiosyncratic variance
        .97855802,                         # Factor transition
        -.28856258, -.14910552,            # Error transition parameters
        -.41544832, -.26706536,
        -.72661178, -.27278821,
    ],
    'var_oim': [
        1.176e-06, 7.304e-07, 6.726e-07,   # Factor loadings
        9.517e-08, 2.300e-10, 1.389e-10,   # Idiosyncratic variance
        .00041159,                         # Factor transition
        .0131511, .01296008,               # Error transition parameters
        .01748435, .01616862,
        .03262051, .02546648,
    ],
    'loglike': 607.4203109232711,
    'aic': -1188.841,
    'bic': -1158.713,
}

lutkepohl_dfm_scalar = {
    'params': [
        .04424851, .00114077, .00275081,  # Factor loadings
        .01812298, .02071169, .01987196,  # Beta.constant
        .00012067,                        # Idiosyncratic variance
        -.19915198,                       # Factor transition
    ],
    'var_oim': [
        .00001479, 1.664e-06, 1.671e-06,
        .00001985, 1.621e-06, 1.679e-06,
        1.941e-10,
        .01409482
    ],
    'loglike': 588.7677809701966,
    'aic': -1161.536,
    'bic': -1142.996,
}

lutkepohl_sfm = {
    'params': [
        .02177607, .02089956, .02239669,  # Factor loadings
        .00201477, .00013623, 7.452e-16   # Idiosyncratic variance
    ],
    'var_oim': [
        .00003003, 4.729e-06, 3.344e-06,
        1.083e-07, 4.950e-10, 0
    ],
    'loglike': 532.2215594949788,
    'aic': -1054.443,
    'bic': -1042.856,
}

lutkepohl_sur = {
    'params': [
        .02169026, -.00009184,            # Betas, y1
        .0229165, -.00005654,             # Betas, y2
        .01998994, -3.049e-06,            # Betas, y3
        # .00215703,                      # Covariance, lower triangle
        # .0000484,  .00014252,
        # .00012772, .00005642, .00010673,
        # Note: the following are the Cholesky of the covariance
        # matrix defined just above
        .04644384,                          # Cholesky, lower triangle
        .00104212,  .0118926,
        .00274999,  .00450315,  .00888196,
    ],
    'var_oim': [
        .0001221, 6.137e-08,
        8.067e-06, 4.055e-09,
        6.042e-06, 3.036e-09,
        None,
        None, None,
        None, None, None
    ],
    'loglike': 597.6181259116113,
    'aic': -1171.236,
    'bic': -1143.426,
}


lutkepohl_sur_auto = {
    'params': [
        .02243063, -.00011112,  # Betas, y1
        .02286952, -.0000554,   # Betas, y2
        .0020338, .00013843,    # Idiosyncratic variance
        -.21127833, .50884609,  # Error transition parameters
        .04292935, .00855789,
    ],
    'var_oim': [
        .00008357, 4.209e-08,
        8.402e-06, 4.222e-09,
        1.103e-07, 5.110e-10,
        .01259537, .19382105,
        .00085936, .01321035,
    ],
    'loglike': 352.7250284160132,
    'aic': -685.4501,
    'bic': -662.2752
}
