"""
Sandbox Panel Estimators

References
-----------

Baltagi, Badi H. `Econometric Analysis of Panel Data.` 4th ed. Wiley, 2008.
"""
from functools import reduce

import numpy as np

from statsmodels.regression.linear_model import GLS

__all__ = ["PanelModel"]

from pandas import Panel


def group(X):
    """
    Returns unique numeric values for groups without sorting.

    Examples
    --------
    >>> X = np.array(['a','a','b','c','b','c'])
    >>> group(X)
    >>> g
    array([ 0.,  0.,  1.,  2.,  1.,  2.])
    """
    uniq_dict = {}
    group = np.zeros(len(X))
    for i in range(len(X)):
        if not X[i] in uniq_dict:
            uniq_dict.update({X[i] : len(uniq_dict)})
        group[i] = uniq_dict[X[i]]
    return group

def repanel_cov(groups, sigmas):
    '''calculate error covariance matrix for random effects model

    Parameters
    ----------
    groups : ndarray, (nobs, nre) or (nobs,)
        array of group/category observations
    sigma : ndarray, (nre+1,)
        array of standard deviations of random effects,
        last element is the standard deviation of the
        idiosyncratic error

    Returns
    -------
    omega : ndarray, (nobs, nobs)
        covariance matrix of error
    omegainv : ndarray, (nobs, nobs)
        inverse covariance matrix of error
    omegainvsqrt : ndarray, (nobs, nobs)
        squareroot inverse covariance matrix of error
        such that omega = omegainvsqrt * omegainvsqrt.T

    Notes
    -----
    This does not use sparse matrices and constructs nobs by nobs
    matrices. Also, omegainvsqrt is not sparse, i.e. elements are non-zero
    '''

    if groups.ndim == 1:
        groups = groups[:,None]
    nobs, nre = groups.shape
    omega = sigmas[-1]*np.eye(nobs)
    for igr in range(nre):
        group = groups[:,igr:igr+1]
        groupuniq = np.unique(group)
        dummygr = sigmas[igr] * (group == groupuniq).astype(float)
        omega +=  np.dot(dummygr, dummygr.T)
    ev, evec = np.linalg.eigh(omega)  #eig does not work
    omegainv = np.dot(evec, (1/ev * evec).T)
    omegainvhalf = evec/np.sqrt(ev)
    return omega, omegainv, omegainvhalf



class PanelData(Panel):
    pass

class PanelModel:
    """
    An abstract statistical model class for panel (longitudinal) datasets.

    Parameters
    ----------
    endog : array_like or str
        If a pandas object is used then endog should be the name of the
        endogenous variable as a string.
#    exog
#    panel_arr
#    time_arr
    panel_data : pandas.Panel object

    Notes
    -----
    If a pandas object is supplied it is assumed that the major_axis is time
    and that the minor_axis has the panel variable.
    """
    def __init__(self, endog=None, exog=None, panel=None, time=None,
            xtnames=None, equation=None, panel_data=None):
        if panel_data is None:
#            if endog == None and exog == None and panel == None and \
#                    time == None:
#                raise ValueError("If pandel_data is False then endog, exog, \
#panel_arr, and time_arr cannot be None.")
            self.initialize(endog, exog, panel, time, xtnames, equation)
#        elif aspandas != False:
#            if not isinstance(endog, str):
#                raise ValueError("If a pandas object is supplied then endog \
#must be a string containing the name of the endogenous variable")
#            if not isinstance(aspandas, Panel):
#                raise ValueError("Only pandas.Panel objects are supported")
#            self.initialize_pandas(endog, aspandas, panel_name)


    def initialize(self, endog, exog, panel, time, xtnames, equation):
        """
        Initialize plain array model.

        See PanelModel
        """
#TODO: for now, we are going assume a constant, and then make the first
#panel the base, add a flag for this....

        # get names
        names = equation.split(" ")
        self.endog_name = names[0]
        exog_names = names[1:]  # this makes the order matter in the array
        self.panel_name = xtnames[0]
        self.time_name = xtnames[1]


        novar = exog.var(0) == 0
        if True in novar:
            cons_index = np.where(novar == 1)[0][0] # constant col. num
            exog_names.insert(cons_index, 'cons')

        self._cons_index = novar # used again in fit_fixed
        self.exog_names = exog_names
        self.endog = np.squeeze(np.asarray(endog))
        exog = np.asarray(exog)
        self.exog = exog
        self.panel = np.asarray(panel)
        self.time = np.asarray(time)

        self.paneluniq = np.unique(panel)
        self.timeuniq = np.unique(time)
#TODO: this  structure can possibly be extracted somewhat to deal with
#names in general

#TODO: add some dimension checks, etc.

#    def initialize_pandas(self, endog, aspandas):
#        """
#        Initialize pandas objects.
#
#        See PanelModel.
#        """
#        self.aspandas = aspandas
#        endog = aspandas[endog].values
#        self.endog = np.squeeze(endog)
#        exog_name = aspandas.columns.tolist()
#        exog_name.remove(endog)
#        self.exog = aspandas.filterItems(exog_name).values
#TODO: can the above be simplified to slice notation?
#        if panel_name != None:
#            self.panel_name = panel_name
#        self.exog_name = exog_name
#        self.endog_name = endog
#        self.time_arr = aspandas.major_axis
        #TODO: is time always handled correctly in fromRecords?
#        self.panel_arr = aspandas.minor_axis
#TODO: all of this might need to be refactored to explicitly rely (internally)
# on the pandas LongPanel structure for speed and convenience.
# not sure this part is finished...

#TODO: does not conform to new initialize
    def initialize_pandas(self, panel_data, endog_name, exog_name):
        self.panel_data = panel_data
        endog = panel_data[endog_name].values # does this create a copy?
        self.endog = np.squeeze(endog)
        if exog_name is None:
            exog_name = panel_data.columns.tolist()
            exog_name.remove(endog_name)
        self.exog = panel_data.filterItems(exog_name).values # copy?
        self._exog_name = exog_name
        self._endog_name = endog_name
        self._timeseries = panel_data.major_axis # might not need these
        self._panelseries = panel_data.minor_axis

#TODO: this could be pulled out and just have a by kwd that takes
# the panel or time array
#TODO: this also needs to be expanded for 'twoway'
    def _group_mean(self, X, index='oneway', counts=False, dummies=False):
        """
        Get group means of X by time or by panel.

        index default is panel
        """
        if index == 'oneway':
            Y = self.panel
            uniq = self.paneluniq
        elif index == 'time':
            Y = self.time
            uniq = self.timeuniq
        else:
            raise ValueError("index %s not understood" % index)
        print(Y, uniq, uniq[:,None], len(Y), len(uniq), len(uniq[:,None]),
              index)
        #TODO: use sparse matrices
        dummy = (Y == uniq[:,None]).astype(float)
        if X.ndim > 1:
            mean = np.dot(dummy,X)/dummy.sum(1)[:,None]
        else:
            mean = np.dot(dummy,X)/dummy.sum(1)
        if counts is False and dummies is False:
            return mean
        elif counts is True and dummies is False:
            return mean, dummy.sum(1)
        elif counts is True and dummies is True:
            return mean, dummy.sum(1), dummy
        elif counts is False and dummies is True:
            return mean, dummy

#TODO: Use kwd arguments or have fit_method methods?
    def fit(self, model=None, method=None, effects='oneway'):
        """
        method : LSDV, demeaned, MLE, GLS, BE, FE, optional
        model :
                between
                fixed
                random
                pooled
                [gmm]
        effects :
                oneway
                time
                twoway
        femethod : demeaned (only one implemented)
                   WLS
        remethod :
                swar -
                amemiya
                nerlove
                walhus


        Notes
        -----
        This is unfinished.  None of the method arguments work yet.
        Only oneway effects should work.
        """
        if method: # get rid of this with default
            method = method.lower()
        model = model.lower()
        if method and method not in ["lsdv", "demeaned", "mle",
                                     "gls", "be", "fe"]:
            # get rid of if method with default
            raise ValueError("%s not a valid method" % method)
#        if method == "lsdv":
#            self.fit_lsdv(model)
        if model == 'pooled':
            return GLS(self.endog, self.exog).fit()
        if model == 'between':
            return self._fit_btwn(method, effects)
        if model == 'fixed':
            return self._fit_fixed(method, effects)

#    def fit_lsdv(self, effects):
#        """
#        Fit using least squares dummy variables.
#
#        Notes
#        -----
#        Should only be used for small `nobs`.
#        """
#        pdummies = None
#        tdummies = None

    def _fit_btwn(self, method, effects):
        # group mean regression or WLS
        if effects != "twoway":
            endog = self._group_mean(self.endog, index=effects)
            exog = self._group_mean(self.exog, index=effects)
        else:
            raise ValueError("%s effects is not valid for the between "
                             "estimator" % effects)
        befit = GLS(endog, exog).fit()
        return befit

    def _fit_fixed(self, method, effects):
        endog = self.endog
        exog = self.exog
        demeantwice = False
        if effects in ["oneway","twoways"]:
            if effects == "twoways":
                demeantwice = True
                effects = "oneway"
            endog_mean, counts = self._group_mean(endog, index=effects,
                counts=True)
            exog_mean = self._group_mean(exog, index=effects)
            counts = counts.astype(int)
            endog = endog - np.repeat(endog_mean, counts)
            exog = exog - np.repeat(exog_mean, counts, axis=0)
        if demeantwice or effects == "time":
            endog_mean, dummies = self._group_mean(endog, index="time",
                dummies=True)
            exog_mean = self._group_mean(exog, index="time")
            # This allows unbalanced panels
            endog = endog - np.dot(endog_mean, dummies)
            exog = exog - np.dot(dummies.T, exog_mean)
        fefit = GLS(endog, exog[:,-self._cons_index]).fit()
#TODO: might fail with one regressor
        return fefit




class SURPanel(PanelModel):
    pass

class SEMPanel(PanelModel):
    pass

class DynamicPanel(PanelModel):
    pass

if __name__ == "__main__":
    import numpy.lib.recfunctions as nprf
    import pandas
    from pandas import Panel

    import statsmodels.api as sm

    data = sm.datasets.grunfeld.load()
    # Baltagi does not include American Steel
    endog = data.endog[:-20]
    fullexog = data.exog[:-20]
#    fullexog.sort(order=['firm','year'])
    panel_arr = nprf.append_fields(fullexog, 'investment', endog, float,
            usemask=False)

    panel_df = pandas.DataFrame(panel_arr)
    panel_panda = panel_df.set_index(['year', 'firm']).to_panel()


    # the most cumbersome way of doing it as far as preprocessing by hand
    exog = fullexog[['value','capital']].view(float).reshape(-1,2)
    exog = sm.add_constant(exog, prepend=False)
    panel = group(fullexog['firm'])
    year = fullexog['year']
    panel_mod = PanelModel(endog, exog, panel, year, xtnames=['firm','year'],
            equation='invest value capital')
# note that equation does not actually do anything but name the variables
    panel_ols = panel_mod.fit(model='pooled')

    panel_be = panel_mod.fit(model='between', effects='oneway')
    panel_fe = panel_mod.fit(model='fixed', effects='oneway')

    panel_bet = panel_mod.fit(model='between', effects='time')
    panel_fet = panel_mod.fit(model='fixed', effects='time')

    panel_fe2 = panel_mod.fit(model='fixed', effects='twoways')


#see also Baltagi (3rd edt) 3.3 THE RANDOM EFFECTS MODEL p.35
#for explicit formulas for spectral decomposition
#but this works also for unbalanced panel
#
#I also just saw: 9.4.2 The Random Effects Model p.176 which is
#partially almost the same as I did
#
#this needs to use sparse matrices for larger datasets
#
#"""
#
#import numpy as np
#

    groups = np.array([0,0,0,1,1,2,2,2])
    nobs = groups.shape[0]
    groupuniq = np.unique(groups)
    periods = np.array([0,1,2,1,2,0,1,2])
    perioduniq = np.unique(periods)

    dummygr = (groups[:,None] == groupuniq).astype(float)
    dummype = (periods[:,None] == perioduniq).astype(float)

    sigma = 1.
    sigmagr = np.sqrt(2.)
    sigmape = np.sqrt(3.)

    #dummyall = np.c_[sigma*np.ones((nobs,1)), sigmagr*dummygr,
    #                                           sigmape*dummype]
    #exclude constant ?
    dummyall = np.c_[sigmagr*dummygr, sigmape*dummype]
    # omega is the error variance-covariance matrix for the stacked
    # observations
    omega = np.dot(dummyall, dummyall.T) + sigma* np.eye(nobs)
    print(omega)
    print(np.linalg.cholesky(omega))
    ev, evec = np.linalg.eigh(omega)  #eig does not work
    omegainv = np.dot(evec, (1/ev * evec).T)
    omegainv2 = np.linalg.inv(omega)
    omegacomp = np.dot(evec, (ev * evec).T)
    print(np.max(np.abs(omegacomp - omega)))
    #check
    #print(np.dot(omegainv,omega)
    print(np.max(np.abs(np.dot(omegainv,omega) - np.eye(nobs))))
    omegainvhalf = evec/np.sqrt(ev)  #not sure whether ev should not be column
    print(np.max(np.abs(np.dot(omegainvhalf,omegainvhalf.T) - omegainv)))

    # now we can use omegainvhalf in GLS (instead of the cholesky)








    sigmas2 = np.array([sigmagr, sigmape, sigma])
    groups2 = np.column_stack((groups, periods))
    omega_, omegainv_, omegainvhalf_ = repanel_cov(groups2, sigmas2)
    print(np.max(np.abs(omega_ - omega)))
    print(np.max(np.abs(omegainv_ - omegainv)))
    print(np.max(np.abs(omegainvhalf_ - omegainvhalf)))

    # notation Baltagi (3rd) section 9.4.1 (Fixed Effects Model)
    Pgr = reduce(np.dot,[dummygr,
            np.linalg.inv(np.dot(dummygr.T, dummygr)),dummygr.T])
    Qgr = np.eye(nobs) - Pgr
    # within group effect: np.dot(Qgr, groups)
    # but this is not memory efficient, compared to groupstats
    print(np.max(np.abs(np.dot(Qgr, groups))))
