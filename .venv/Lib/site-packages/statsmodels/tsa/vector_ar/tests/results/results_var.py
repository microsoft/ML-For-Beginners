"""
Test Results for the VAR model.  Obtained from Stata using
datasets/macrodata/var.do
"""

import numpy as np


class MacrodataResults:
    def __init__(self):
        params = [
            -0.2794863875, 0.0082427826, 0.6750534746, 0.2904420695,
            0.0332267098,  -0.0073250059, 0.0015269951,  -0.1004938623,
            -0.1231841792, 0.2686635768, 0.2325045441, 0.0257430635,
            0.0235035714, 0.0054596064, -1.97116e+00, 0.3809752365,
            4.4143364022, 0.8001168377, 0.2255078864,  -0.1241109271,
            -0.0239026118]
        params = np.asarray(params).reshape(3, -1)
        params = np.hstack((params[:, -1][:, None],
                            params[:, :-1:2],
                            params[:, 1::2]))
        self.params = params
        self.neqs = 3
        self.nobs = 200
        self.df_eq = 7
        self.nobs_1 = 200
        self.df_model_1 = 6
        self.rmse_1 = .0075573716985351
        self.rsquared_1 = .2739094844780006
        self.llf_1 = 696.8213727557811
        self.nobs_2 = 200
        self.rmse_2 = .0065444260782597
        self.rsquared_2 = .1423626064753714
        self.llf_2 = 725.6033255319256
        self.nobs_3 = 200
        self.rmse_3 = .0395942039671031
        self.rsquared_3 = .2955406949737428
        self.llf_3 = 365.5895183036045
        # These are from Stata.  They use the LL based definition
        # We return Lutkepohl statistics.  See Stata TS manual page 436
        #        self.bic = -19.06939794312953
        #        self.aic = -19.41572126661708
        #        self.hqic = -19.27556951526737
        # These are from R.  See var.R in macrodata folder
        self.bic = -2.758301611618373e+01
        self.aic = -2.792933943967127e+01
        self.hqic = -2.778918768832157e+01
        self.fpe = 7.421287668357018e-13
        self.detsig = 6.01498432283e-13
        self.llf = 1962.572126661708

        self.chi2_1 = 75.44775165699033
        # do not know how they calculate this;  it's not -2 * (ll1 - ll0)

        self.chi2_2 = 33.19878716815366
        self.chi2_3 = 83.90568280242312
        bse = [
            .1666662376, .1704584393, .1289691456, .1433308696, .0257313781,
            .0253307796, .0010992645, .1443272761, .1476111934, .1116828804,
            .1241196435, .0222824956, .021935591, .0009519255, .8731894193,
            .8930573331, .6756886998, .7509319263, .1348105496, .1327117543,
            .0057592114]
        bse = np.asarray(bse).reshape(3, -1)
        bse = np.hstack((bse[:, -1][:, None],
                         bse[:, :-1:2],
                         bse[:, 1::2]))
        self.bse = bse
