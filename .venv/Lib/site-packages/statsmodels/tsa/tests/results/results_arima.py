import os

import numpy as np
from numpy import genfromtxt

cur_dir = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cur_dir, "results_arima_forecasts.csv")
with open(path, "rb") as fd:
    forecast_results = genfromtxt(fd, names=True, delimiter=",", dtype=float)

# NOTE:
# stata gives no indication of no convergence for 112 CSS but gives a
# different answer than x12arima, gretl simply fails to converge
# redid stata with starting parameters from x12arima

# it looks like stata uses a different formula for the CSS likelihood
# they appear to be using a larger sample than R, gretl, or us.
# CSS results are therefore taken from R and gretl


class ARIMA111:
    def __init__(self, method="mle"):
        self.k_ar = 1
        self.k_diff = 1
        self.k_ma = 1
        if method == "mle":
            # from stata
            from .arima111_results import results

            # unpack stata results
            self.__dict__.update(results)
            self.resid = self.resid[1:]
            self.params = self.params[:-1]
            self.sigma2 = self.sigma**2
            self.aic = self.icstats[4]
            self.bic = self.icstats[5]
            self.fittedvalues = self.xb[1:]  # no idea why this initial value
            self.linear = self.y[1:]
            # stata bse are OPG
            # self.bse = np.diag(self.cov_params) ** .5

            # from gretl
            self.arroots = [1.0640 + 0j]
            self.maroots = [1.2971 + 0j]
            self.hqic = 496.8653
            self.aic_gretl = 491.5112
            self.bic_gretl = 504.7442
            self.tvalues = [4.280, 20.57, -8.590]
            self.pvalues = [1.87e-5, 5.53e-94, 8.73e-18]
            self.cov_params = [[0.0423583,   -0.00167449,    0.00262911],
                               [-0.00167449, 0.00208858,    -0.0035068],
                               [0.00262911, -0.0035068, 0.00805622]]
            self.bse = np.sqrt(np.diag(self.cov_params))
            # these bse are approx [.205811, .0457010, .0897565]

            # from stata
            # forecast = genfromtxt(open(cur_dir+"/arima111_forecasts.csv"),
            #                delimiter=",", skip_header=1, usecols=[1,2,3,4,5])
            # self.forecast = forecast[203:,1]
            # self.fcerr = forecast[203:,2]
            # self.fc_conf_int = forecast[203:,3:]

            # from gretl
            self.forecast = forecast_results['fc111c'][-25:]
            self.forecasterr = forecast_results['fc111cse'][-25:]
            self.forecast_dyn = forecast_results['fc111cdyn']
            self.forecasterr_dyn = forecast_results['fc111cdynse']
        else:
            # coefs, bse, tvalues, and pvalues taken from R because gretl
            # uses mean not constant
            self.bse = [0.21583833, 0.03844939, 0.08566390]
            self.params = [1.0087257, 0.9455393, -0.8021834]
            self.sigma2 = 0.6355913
            self.tvalues = [4.673524, 24.591788, -9.364311]
            self.pvalues = [5.464467e-06, 0, 0]
            self.cov_params = np.array([
                [0.046586183,  0.002331183, -0.004647432],
                [0.002331183,  0.001478356, -0.002726201],
                [-0.004647432, -0.002726201,  0.007338304]])

            # from gretl
            self.llf = -239.6601
            self.aic = 487.3202
            self.bic = 500.5334
            self.hqic = 492.6669
            self.arroots = [1.0578 + 0j]
            self.maroots = [1.2473 + 0j]
            # cov_params = np.array([[0.00369569, -0.00271777, 0.00269806],
            #                        [0, 0.00209573, -0.00224559],
            #                        [0, 0, 0.00342769]])
            # self.cov_params = cov_params + cov_params.T - \
            #                np.diag(np.diag(cov_params))
            # self.bse = np.sqrt(np.diag(self.cov_params))

            self.resid = [-0.015830, -0.236884, -0.093946, -0.281152,
                          -0.089983, -0.226336, -0.351666, -0.198703,
                          -0.258418, -0.259026, -0.149513, -0.325703,
                          -0.165703, -0.279229, -0.295711, -0.120018,
                          -0.289870, -0.154243, -0.348403, -0.273902,
                          -0.240894, -0.182791, -0.252930, -0.152441,
                          -0.296412, -0.128941, 0.024068, -0.243972,
                          -0.011436, -0.392437, -0.217022, -0.118190,
                          -0.133489, -0.045755, -0.169953, 0.025010,
                          -0.107754, -0.119661, 0.070794, -0.065586,
                          -0.080390, 0.007741, -0.016138, -0.235283,
                          -0.121907, -0.125546, -0.428463, -0.087713,
                          -0.298131, -0.277757, -0.261422, -0.248326,
                          -0.137826, -0.043771, 0.437100, -0.150051,
                          0.751890, 0.424180, 0.450514, 0.277089,
                          0.732583, 0.225086, -0.403648, -0.040509,
                          -0.132975, -0.112572, -0.696214, 0.003079,
                          -0.003491, -0.108758, 0.401383, -0.162302,
                          -0.141547, 0.175094, 0.245346, 0.607134, 0.519045,
                          0.248419, 0.920521, 1.097613, 0.755983, 1.271156,
                          1.216969, -0.121014, 0.340712, 0.732750, 0.068915,
                          0.603912, 0.060157, -0.803110, -1.044392, 1.040311,
                          -0.984497, -1.611668, -0.258198, -0.112970,
                          -0.091071, 0.226487, 0.097475, -0.311423, -0.061105,
                          -0.449488, 0.317277, -0.329734, -0.181248, 0.443263,
                          -2.223262, 0.096836, -0.033782, 0.456032, 0.476052,
                          0.197564, 0.263362, 0.021578, 0.216803, 0.284249,
                          0.343786, 0.196981, 0.773819, 0.169070, -0.343097,
                          0.918962, 0.096363, 0.298610, 1.571685, -0.236620,
                          -1.073822, -0.194208, -0.250742, -0.101530,
                          -0.076437, -0.056319, 0.059811, -0.041620,
                          -0.128404, -0.403446, 0.059654, -0.347208,
                          -0.095257, 0.217668, -0.015057, 0.087431, 0.275062,
                          -0.263580, -0.122746, 0.195629, 0.367272,
                          -0.184188, 0.146368, 0.127777, -0.587128,
                          -0.498538, 0.172490, -0.456741, -0.694000,
                          0.199392, -0.140634, -0.029636, 0.364818,
                          -0.097080, 0.510745, 0.230842, 0.595504, 0.709721,
                          0.012218, 0.520223, -0.445174, -0.168341,
                          -0.935465, -0.894203, 0.733417, -0.279707,
                          0.258861, 0.417969, -0.443542, -0.477955, 0.288992,
                          0.442126, 0.075826, 0.665759, 0.571509, -0.204055,
                          0.835901, -0.375693, 3.292828, -1.469299,
                          -0.122206, 0.617909, -2.250468, 0.570871, 1.166013,
                          0.079873, 0.463372, 1.981434, -0.142869, 3.023376,
                          -3.713161, -6.120150, -0.007487, 1.267027, 1.176930]

            self.linear = [
                29.3658, 29.6069, 29.6339, 29.8312, 29.8400,
                30.0663, 30.1617, 30.1187, 30.2384, 30.2990,
                30.3595, 30.5457, 30.5457, 30.7192, 30.7757,
                30.8100, 31.0399, 31.0942, 31.2984, 31.2939,
                31.3609, 31.4628, 31.6329, 31.7324, 31.9464,
                32.0089, 32.2559, 32.6940, 32.8614, 33.2924,
                33.3170, 33.5182, 33.8335, 34.1458, 34.5700,
                34.8750, 35.4078, 35.8197, 36.2292, 36.8656,
                37.3804, 37.8923, 38.5161, 39.1353, 39.5219,
                40.0255, 40.5285, 40.6877, 41.1981, 41.4778,
                41.7614, 42.0483, 42.3378, 42.7438, 43.2629,
                44.3501, 44.8481, 46.3758, 47.6495, 49.0229,
                50.2674, 52.0749, 53.4036, 54.0405, 55.0330,
                55.9126, 56.7962, 56.9969, 57.9035, 58.8088,
                59.5986, 60.9623, 61.7415, 62.5249, 63.6547,
                64.8929, 66.5810, 68.2516, 69.6795, 71.9024,
                74.4440, 76.7288, 79.6830, 82.7210, 84.3593,
                86.4672, 89.0311, 90.8961, 93.3398, 95.2031,
                96.0444, 96.4597, 99.0845, 99.5117, 99.0582,
                99.9130, 100.8911, 101.8735, 103.2025, 104.4114,
                105.1611, 106.1495, 106.6827, 108.0297, 108.6812,
                109.4567, 110.9233, 109.4032, 110.2338, 110.9440,
                112.2239, 113.6024, 114.7366, 115.9784, 116.9832,
                118.2158, 119.5562, 121.0030, 122.3262, 124.3309,
                125.7431, 126.5810, 128.8036, 130.2014, 131.8283,
                134.9366, 136.1738, 136.3942, 137.4507, 138.4015,
                139.4764, 140.5563, 141.6402, 142.8416, 143.9284,
                144.9034, 145.5403, 146.6472, 147.2953, 148.1823,
                149.4151, 150.4126, 151.5249, 152.8636, 153.6227,
                154.5044, 155.7327, 157.1842, 158.0536, 159.2722,
                160.4871, 160.8985, 161.3275, 162.4567, 162.8940,
                163.0006, 164.0406, 164.7296, 165.5352, 166.7971,
                167.5893, 169.0692, 170.3045, 171.9903, 173.8878,
                175.0798, 176.8452, 177.5683, 178.5355, 178.5942,
                178.5666, 180.2797, 180.9411, 182.1820, 183.6435,
                184.1780, 184.6110, 185.8579, 187.3242, 188.4342,
                190.2285, 192.0041, 192.9641, 195.0757, 195.9072,
                200.8693, 200.8222, 202.0821, 204.1505, 203.0031,
                204.7540, 207.2581, 208.6696, 210.5136, 214.1399,
                215.5866, 220.6022, 218.2942, 212.6785, 213.2020,
                215.2081]
            # forecasting is not any different for css
            # except you lose the first p+1 observations for in-sample
            # these results are from x-12 arima
            self.forecast = forecast_results['fc111c_css'][-25:]
            self.forecasterr = forecast_results['fc111cse_css'][-25:]
            self.forecast_dyn = forecast_results['fc111cdyn_css']
            self.forecasterr_dyn = forecast_results['fc111cdynse_css']


class ARIMA211:
    def __init__(self, method="mle"):
        if method == 'mle':
            # from stata
            from .arima111_results import results

            self.__dict__.update(results)
            self.resid = self.resid[1:]
            self.params = self.params[:-1]
            self.sigma2 = self.sigma**2
            self.aic = self.icstats[4]
            self.bic = self.icstats[5]
            self.fittedvalues = self.xb[1:]  # no idea why this initial value
            self.linear = self.y[1:]
            self.k_diff = 1

            # stata bse are OPG
            # self.bse = np.diag(self.cov_params) ** .5

            # from gretl
            self.arroots = [1.027 + 0j, 5.7255 + 0j]
            self.maroots = [1.1442+0j]
            self.hqic = 496.5314
            self.aic_gretl = 489.8388
            self.bic_gretl = 506.3801
            self.tvalues = [3.468, 11.14, -1.941, 12.55]
            self.pvalues = [.0005, 8.14e-29, .0522, 3.91e-36]
            cov_params = np.array([
                [0.0616906,  -0.00250187, 0.0010129,    0.00260485],
                [0, 0.0105302,   -0.00867819,   -0.00525614],
                [0, 0,         0.00759185,    0.00361962],
                [0, 0, 0,                      0.00484898]])
            self.cov_params = (
                cov_params + cov_params.T - np.diag(np.diag(cov_params)))
            self.bse = np.sqrt(np.diag(self.cov_params))
            # these bse are approx [0.248376, 0.102617, 0.0871312, 0.0696346]

            self.forecast = forecast_results['fc211c'][-25:]
            self.forecasterr = forecast_results['fc211cse'][-25:]
            self.forecast_dyn = forecast_results['fc211cdyn'][-25:]
            self.forecasterr_dyn = forecast_results['fc211cdynse'][-25:]
        else:
            from .arima211_css_results import results

            self.__dict__.update(results)
            self.resid = self.resid[1:]
            self.params = self.params[:-1]
            self.sigma2 = self.sigma**2
            self.aic = self.icstats[4]
            self.bic = self.icstats[5]
            self.fittedvalues = self.xb[1:]  # no idea why this initial value
            self.linear = self.y[1:]
            self.k_diff = 1

            # from gretl
            self.arroots = [1.0229 + 0j, 4.4501 + 0j]
            self.maroots = [1.0604 + 0j]
            self.hqic = 489.3225
            self.aic_gretl = 482.6486
            self.bic_gretl = 499.1402
            self.tvalues = [.7206, 22.54, -19.04]
            self.pvalues = [.4712, 1.52e-112, 2.19e-10, 8.00e-81]
            cov_params = np.array([
                [8.20496e-04, -0.0011992, 4.57078e-04, 0.00109907],
                [0, 0.00284432, -0.0016752, -0.00220223],
                [0, 0, 0.00119783, 0.00108868],
                [0, 0, 0, 0.00245324]])
            self.cov_params = (
                cov_params + cov_params.T - np.diag(np.diag(cov_params)))
            self.bse = np.sqrt(np.diag(self.cov_params))
            # forecasting is not any different for css
            # except you lose the first p+1 observations for in-sample
            self.forecast = forecast_results['fc111c_css'][-25:]
            self.forecasterr = forecast_results['fc111cse_css'][-25:]
            self.forecast_dyn = forecast_results['fc111cdyn_css']
            self.forecasterr_dyn = forecast_results['fc111cdynse_css']


class ARIMA112:
    def __init__(self, method="mle"):
        self.df_model = 3
        self.k = 5
        self.k_ar = 1
        self.k_ma = 2
        self.k_exog = 1
        self.k_diff = 1
        if method == "mle":
            from .arima112_results import results

            # from gretl
            self.arroots = [1.0324 + 0j]
            self.maroots = [1.1447 + 0j, -4.8613+0j]
            self.hqic = 495.5852
            self.aic_gretl = 488.8925
            self.bic_gretl = 505.4338
            self.tvalues = [3.454, 31.10, -7.994, -2.127]
            self.pvalues = [0.0006, 2.1e-212, 1.31e-15, .0334]
            cov_params = np.array([
                [0.0620096, -0.00172172, 0.00181301, 0.00103271],
                [0, 9.69682e-04, -9.70767e-04, -8.99814e-04],
                [0, 0, 0.00698068, -0.00443871],
                [0, 0, 0, 0.00713662]])
            self.cov_params = (
                cov_params + cov_params.T - np.diag(np.diag(cov_params)))
            self.bse = np.sqrt(np.diag(self.cov_params))

            # from gretl
            self.forecast = forecast_results['fc112c'][-25:]
            self.forecasterr = forecast_results['fc112cse'][-25:]
            self.forecast_dyn = forecast_results['fc112cdyn']
            self.forecasterr_dyn = forecast_results['fc112cdynse']

            # unpack stata results
            self.__dict__ = results
            self.resid = self.resid[1:]
            self.params = self.params[:-1]
            self.sigma2 = self.sigma**2
            self.aic = self.icstats[4]
            self.bic = self.icstats[5]
            self.fittedvalues = self.xb[1:]  # no idea why this initial value
            self.linear = self.y[1:]
            # stata bse are OPG
            # self.bse = np.diag(self.cov_params) ** .5
        else:
            # NOTE: this looks like a "hard" problem
            #  unable to replicate stata's results even with their starting
            #   values
            # unable to replicate x12 results in stata using their starting
            #   values. x-12 has better likelihood and we can replicate so
            #   use their results

            # taken from R using X12-arima values as init params
            self.bse = [0.07727588, 0.09356658, 0.10503567, 0.07727970]
            self.params = [0.9053219, -0.692412, 1.0736728, 0.1720008]
            self.sigma2 = 0.6820727
            self.tvalues = [11.715452, -7.400215, 10.221983,  2.225692]
            self.pvalues = [0, 3.791634e-12, 0, 2.716275e-02]
            self.cov_params = np.array([
                [0.0059715623, 0.001327824, -0.001592129, -0.0008061933],
                [0.0013278238, 0.008754705, -0.008024634, -0.0045933413],
                [-0.0015921293, -0.008024634,  0.011032492,  0.0072509641],
                [-0.0008061933, -0.004593341,  0.007250964,  0.0059721516]])

            # from x12arima via gretl
            # gretl did not converge for this model...
            self.llf = -246.7534
            self.nobs = 202
            # self.params = [.905322, -.692425, 1.07366, 0.172024]
            # self.sigma2 = 0.682072819129
            # self.bse = [0.0756430, 0.118440, 0.140691, 0.105266]

            self.resid = [
                -1.214477, -0.069772, -1.064510, -0.249555,
                -0.874206, -0.322177, -1.003579, -0.310040, -0.890506,
                -0.421211, -0.715219, -0.564119, -0.636560, -0.580912,
                -0.717440, -0.424277, -0.747835, -0.424739, -0.805958,
                -0.516877, -0.690127, -0.473072, -0.694766, -0.435627,
                -0.736474, -0.388060, -0.429596, -0.557224, -0.342308,
                -0.741842, -0.442199, -0.491319, -0.420884, -0.388057,
                -0.466176, -0.257193, -0.429646, -0.349683, -0.205870,
                -0.335547, -0.290300, -0.216572, -0.234272, -0.427951,
                -0.255446, -0.338097, -0.579033, -0.213860, -0.556756,
                -0.389907, -0.510060, -0.409759, -0.396778, -0.258727,
                0.160063, -0.467109, 0.688004, -0.021120, 0.503044,
                0.031500, 0.878365, -0.003548, -0.079327, 0.038289,
                0.032773, -0.050780, -0.560124, 0.185655, -0.111981,
                -0.020714, 0.363254, -0.218484, -0.006161, 0.165950,
                0.252365, 0.599220, 0.488921, 0.347677, 1.079814,
                1.102745, 0.959907, 1.570836, 1.454934, 0.343521,
                1.125826, 1.154059, 0.666141, 1.269685, 0.551831,
                -0.027476, -0.305192, 1.715665, -0.990662, -0.548239,
                -0.011636, 0.197796, -0.050128, 0.480031, 0.061198,
                -0.049562, 0.064436, -0.300420, 0.494730, -0.411527,
                0.109242, 0.375255, -2.184482, 0.717733, -0.673064,
                0.751681, -0.092543, 0.438016, -0.024881, 0.250085,
                0.096010, 0.452618, 0.265491, 0.374299, 0.820424,
                0.238176, -0.059646, 1.214061, 0.028679, 0.797567,
                1.614444, -0.094717, -0.408067, 0.299198, -0.021561,
                0.231915, 0.084190, 0.199192, 0.201132, 0.148509,
                0.035431, -0.203352, 0.264744, -0.319785, 0.150305,
                0.184628, 0.074637, 0.148340, 0.357372, -0.241250,
                0.119294, 0.204413, 0.458730, -0.190477, 0.416587,
                0.084216, -0.363361, -0.310339, 0.309728, -0.549677,
                -0.449092, 0.183025, -0.259015, -0.000883, 0.267255,
                -0.188068, 0.577697, 0.049310, 0.746401, 0.565829,
                0.178270, 0.709983, -0.348012, 0.273262, -0.873288,
                -0.403100, 0.720072, -0.428076, 0.488246, 0.248152,
                -0.313214, -0.323137, 0.414843, 0.308909, 0.134180,
                0.732275, 0.535639, -0.056128, 1.128355, -0.449151,
                3.879123, -2.303860, 1.712549, -0.074407, -1.162052,
                0.848316, 1.262031, 0.009320, 1.017563, 1.978597,
                -0.001637, 3.782223, -4.119563, -3.666488, 0.345244,
                0.869998, 0.635321]

            self.linear = [
                30.5645, 29.4398, 30.6045, 29.7996, 30.6242,
                30.1622, 30.8136, 30.2300, 30.8705, 30.4612, 30.9252,
                30.7841, 31.0166, 31.0209, 31.1974, 31.1143, 31.4978,
                31.3647, 31.7560, 31.5369, 31.8101, 31.7531, 32.0748,
                32.0156, 32.3865, 32.2681, 32.7096, 33.0072, 33.1923,
                33.6418, 33.5422, 33.8913, 34.1209, 34.4881, 34.8662,
                35.1572, 35.7296, 36.0497, 36.5059, 37.1355, 37.5903,
                38.1166, 38.7343, 39.3280, 39.6554, 40.2381, 40.6790,
                40.8139, 41.4568, 41.5899, 42.0101, 42.2098, 42.5968,
                42.9587, 43.5399, 44.6671, 44.9120, 46.8211, 47.5970,
                49.2685, 50.1216, 52.3035, 53.0793, 53.9617, 54.8672,
                55.8508, 56.6601, 56.8143, 58.0120, 58.7207, 59.6367,
                61.0185, 61.6062, 62.5340, 63.6476, 64.9008, 66.6111,
                68.1523, 69.5202, 71.8973, 74.2401, 76.4292, 79.4451,
                82.2565, 83.5742, 86.0459, 88.4339, 90.2303, 92.8482,
                94.4275, 95.3052, 95.7843, 99.0907, 98.4482, 98.8116,
                99.6022, 100.8501, 101.6200, 103.2388, 104.1496,
                105.0356, 106.0004, 106.5053, 108.1115, 108.3908,
                109.5247, 110.8845, 108.7823, 110.8731, 110.6483,
                112.7925, 113.3620, 115.0249, 115.7499, 117.1040,
                118.0474, 119.6345, 120.8257, 122.2796, 124.2618,
                125.4596, 126.2859, 128.8713, 129.7024, 131.7856,
                134.7947, 135.5081, 135.9008, 137.2216, 138.0681,
                139.3158, 140.3008, 141.4989, 142.6515, 143.7646,
                144.7034, 145.3353, 146.6198, 147.0497, 148.2154,
                149.3254, 150.3517, 151.4426, 152.8413, 153.3807,
                154.4956, 155.6413, 157.1905, 157.7834, 159.3158,
                160.2634, 160.7103, 161.1903, 162.5497, 162.6491,
                163.0170, 164.1590, 164.7009, 165.6327, 166.8881,
                167.5223, 169.2507, 170.1536, 172.1342, 173.7217,
                174.8900, 176.7480, 177.1267, 178.4733, 178.1031,
                178.5799, 180.4281, 180.7118, 182.3518, 183.5132,
                184.0231, 184.4852, 185.9911, 187.2658, 188.3677,
                190.2644, 191.8561, 192.6716, 195.1492, 195.3209,
                201.7039, 198.9875, 202.7744, 203.0621, 202.7257,
                204.6580, 207.3287, 208.1154, 210.5164, 213.9986,
                214.8278, 221.0086, 215.8405, 212.3258, 213.5990,
                215.7497]

            self.yr = []

            self.arroots = [-1.4442 + 0j]
            self.maroots = [-1.1394 + 0j, -5.1019+0j]
            self.hqic = 510.1902
            self.aic = 503.5069
            self.bic = 520.0234
            # TODO: Document source for these non-used results
            #  (and why they are not used)
            # self.tvalues = [11.97, -5.846, 7.631, 1.634]
            # self.pvalues = [5.21e-33, 5.03e-9, 2.32e-14, .1022]
            # cov_params = np.array([
            #         [0.0620096, -0.00172172, 0.00181301, 0.00103271],
            #         [0, 9.69682e-04, -9.70767e-04, -8.99814e-04],
            #         [0, 0, 0.00698068, -0.00443871],
            #         [0, 0, 0, 0.00713662]])
            # self.cov_params = cov_params + cov_params.T - \
            #                np.diag(np.diag(cov_params))
            # self.bse = np.sqrt(np.diag(self.cov_params))
            self.forecast = forecast_results['fc112c_css'][-25:]
            self.forecasterr = forecast_results['fc112cse_css'][-25:]
            self.forecast_dyn = forecast_results['fc112cdyn_css']
            self.forecasterr_dyn = forecast_results['fc112cdynse_css']
