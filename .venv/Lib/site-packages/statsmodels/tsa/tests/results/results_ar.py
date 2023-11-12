import os

import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))


class ARLagResults:
    """
    Results are from R vars::VARselect for sunspot data.

    Comands run were

    var_select <- VARselect(SUNACTIVITY, lag.max=16, type=c("const"))
    """
    def __init__(self, type="const"):
        # order of results is AIC, HQ, SC, FPE
        if type == "const":
            ic = [
                6.311751824815273, 6.321813007357017,  6.336872456958734,
                551.009492543133547, 5.647615009344886, 5.662706783157502,
                5.685295957560077, 283.614444209634655,  5.634199640773091,
                5.654322005856580, 5.684440905060013, 279.835333966272003,
                5.639415797766900,  5.664568754121261,  5.702217378125553,
                281.299267441683185,  5.646102475432464,  5.676286023057697,
                5.721464371862848, 283.187210932784524,  5.628416873122441,
                5.663631012018546,  5.716339085624555, 278.223839284844701,
                5.584204185137150,  5.624448915304128,  5.684686713710994,
                266.191975554941564,  5.541163244029505,  5.586438565467356,
                5.654206088675081, 254.979353737235556, 5.483155367013447,
                5.533461279722170,  5.608758527730753, 240.611088468544949,
                5.489939895595428,  5.545276399575022,  5.628103372384465,
                242.251199397394288,  5.496713895370946,  5.557080990621412,
                5.647437688231713, 243.900349905069504,  5.503539311586831,
                5.568936998108170,  5.666823420519329, 245.573823561989144,
                5.510365149977393,  5.580793427769605,  5.686209574981622,
                247.259396991133599,  5.513740912139918,  5.589199781203001,
                5.702145653215877, 248.099655693709479,  5.515627471325321,
                5.596116931659277,  5.716592528473011, 248.572915484827206,
                5.515935627515806,  5.601455679120634,  5.729461000735226,
                248.654927915301300]
            self.ic = np.asarray(ic).reshape(4, -1, order='F')


class ARResultsOLS:
    """
    Results of fitting an AR(9) model to the sunspot data.

    Results were taken from Stata using the var command.
   """
    def __init__(self, constant=True):
        self.avobs = 300.
        if constant:
            self.params = [
                6.7430535917332, 1.1649421971129,  -.40535742259304,
                -.16653934246587,  .14980629416032,  -.09462417064796,
                .00491001240749,  .0504665930841, -.08635349190816,
                .25349103194757]
            # These are returned by stata VAR, using the (V)AR scale/sigma
            # we return the true OLS bse by default
            # the stata residuals can be achieved
            # by np.sqrt(np.diag(res1.cov_params()))
            self.bse_stata = [
                2.413485601, .0560359041, .0874490762,
                .0900894414, .0899348339, .0900100797,
                .0898385666, .0896997939, .0869773089,
                .0559505756]
            # The below are grom gretl's ARIMA command with conditional
            # maximum likelihood
            self.bse_gretl = [
                2.45474, 0.0569939, 0.0889440, 0.0916295,
                0.0914723, 0.0915488, 0.0913744, 0.0912332,
                0.0884642, 0.0569071]
            self.rmse = 15.1279294937327
            self.fpe = 236.4827257929261
            self.llf = -1235.559128419549
            # NOTE: we use a different definition of these ic than Stata
            # but our order selection results agree with R VARselect
            # close to Stata for Lutkepohl but we penalize the ic for
            # the trend terms
            # self.bic = 8.427186938618863
            # self.aic = 8.30372752279699
            # self.hqic = 8.353136159250697

            # NOTE: predictions were taken from gretl, but agree with Stata
            #   test predict
            # TODO: remove one of the files
            filename = os.path.join(cur_dir, "AROLSConstantPredict.csv")
            predictresults = np.loadtxt(filename)
            fv = predictresults[:300, 0]
            pv = predictresults[300:, 1]
            del predictresults

            # cases - in sample predict
            # n = -1, start = 0 (fitted values)
            self.FVOLSnneg1start0 = fv
            # n=-1, start=9
            self.FVOLSnneg1start9 = fv
            # n=-1, start=100
            self.FVOLSnneg1start100 = fv[100-9:]
            # n = 200, start = 0
            self.FVOLSn200start0 = fv[:192]
            # n = 200, start = 200
            self.FVOLSn200start200 = np.hstack((fv[200-9:], pv[:101-9]))
            # n = 200, start = -109 use above
            self.FVOLSn200startneg109 = self.FVOLSn200start200
            # n = 100, start = 325, post-sample forecasting
            self.FVOLSn100start325 = np.hstack((fv[-1], pv))
            # n = 301, start = 9
            self.FVOLSn301start9 = np.hstack((fv, pv[:2]))
            # n = 301, start = 0
            self.FVOLSdefault = fv
            # n = 4, start = 312
            self.FVOLSn4start312 = np.hstack((fv[-1], pv[:8]))
            # n = 15, start = 312
            self.FVOLSn15start312 = np.hstack((fv[-1], pv[:19]))

        elif not constant:
            self.params = [
                1.19582389902985, -0.40591818219637,
                -0.15813796884843, 0.16620079925202,
                -0.08570200254617, 0.01876298948686,
                0.06130211910707, -0.08461507700047,
                0.27995084653313]
            self.bse_stata = [
                .055645055, .088579237, .0912031179, .0909032462,
                .0911161784, .0908611473, .0907743174, .0880993504,
                .0558560278]
            self.bse_gretl = [
                0.0564990, 0.0899386, 0.0926027, 0.0922983,
                0.0925145, 0.0922555, 0.0921674, 0.0894513,
                0.0567132]
            self.rmse = 15.29712618677774
            self.sigma = 226.9820074869752
            self.llf = -1239.41217278661
            # See note above; TODO: be more specific about "above"
            # self.bic = 8.433861292817106
            # self.hqic = 8.367215591385756
            # self.aic = 8.322747818577421
            self.fpe = 241.0221316614273

            filename = os.path.join(cur_dir, "AROLSNoConstantPredict.csv")
            predictresults = np.loadtxt(filename)
            fv = predictresults[:300, 0]
            pv = predictresults[300:, 1]
            del predictresults

            # cases - in sample predict
            # n = -1, start = 0 (fitted values)
            self.FVOLSnneg1start0 = fv
            # n=-1, start=9
            self.FVOLSnneg1start9 = fv
            # n=-1, start=100
            self.FVOLSnneg1start100 = fv[100-9:]
            # n = 200, start = 0
            self.FVOLSn200start0 = fv[:192]
            # n = 200, start = 200
            self.FVOLSn200start200 = np.hstack((fv[200-9:], pv[:101-9]))
            # n = 200, start = -109 use above
            self.FVOLSn200startneg109 = self.FVOLSn200start200
            # n = 100, start = 325, post-sample forecasting
            self.FVOLSn100start325 = np.hstack((fv[-1], pv))
            # n = 301, start = 9
            self.FVOLSn301start9 = np.hstack((fv, pv[:2]))
            # n = 301, start = 0
            self.FVOLSdefault = fv
            # n = 4, start = 312
            self.FVOLSn4start312 = np.hstack((fv[-1], pv[:8]))
            # n = 15, start = 312
            self.FVOLSn15start312 = np.hstack((fv[-1], pv[:19]))


class ARResultsMLE:
    """
    Results of fitting an AR(9) model to the sunspot data using exact MLE.

    Results were taken from gretl.
    """
    def __init__(self, constant=True):
        self.avobs = 300
        if constant:

            # NOTE: Stata's estimated parameters differ from gretl
            filename = os.path.join(cur_dir, "ARMLEConstantPredict.csv")
            filename2 = os.path.join(cur_dir,
                                     'results_ar_forecast_mle_dynamic.csv')
            predictresults = np.loadtxt(filename, delimiter=",")
            pv = predictresults[:, 1]
            dynamicpv = np.genfromtxt(filename2, delimiter=",", skip_header=1)

            # cases - in sample predict
            # start = 0 (fitted values)
            self.FVMLEdefault = pv[:309]
            # start=9
            self.FVMLEstart9end308 = pv[9:309]
            # start=100, end=309
            self.FVMLEstart100end308 = pv[100:309]
            # start = 0, end
            self.FVMLEstart0end200 = pv[:201]
            # n = 200, start = 200
            self.FVMLEstart200end334 = pv[200:]
            # start = 309, end=334 post-sample forecasting
            self.FVMLEstart308end334 = pv[308:]
            # end = 310, start = 9
            self.FVMLEstart9end309 = pv[9:310]
            # end = 301, start = 0
            self.FVMLEstart0end301 = pv[:302]
            # end = 312, start = 4
            self.FVMLEstart4end312 = pv[4:313]
            # end = 7, start = 2
            self.FVMLEstart2end7 = pv[2:8]

            self.fcdyn = dynamicpv[:, 0]
            self.fcdyn2 = dynamicpv[:, 1]
            self.fcdyn3 = dynamicpv[:, 2]
            self.fcdyn4 = dynamicpv[:, 3]

        else:
            pass
