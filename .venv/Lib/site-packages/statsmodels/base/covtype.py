# -*- coding: utf-8 -*-
"""
Created on Mon Aug 04 08:00:16 2014

Author: Josef Perktold
License: BSD-3

"""

from statsmodels.compat.python import lzip

import numpy as np

descriptions = {
    'HC0': 'Standard Errors are heteroscedasticity robust (HC0)',
    'HC1': 'Standard Errors are heteroscedasticity robust (HC1)',
    'HC2': 'Standard Errors are heteroscedasticity robust (HC2)',
    'HC3': 'Standard Errors are heteroscedasticity robust (HC3)',
    'HAC': 'Standard Errors are heteroscedasticity and autocorrelation '
           'robust (HAC) using {maxlags} lags and '
           '{correction} small sample correction',
    'fixed_scale': 'Standard Errors are based on fixed scale',
    'cluster': 'Standard Errors are robust to cluster correlation (cluster)',
    'HAC-Panel': 'Standard Errors are robust to '
                 'cluster correlation (HAC-Panel)',
    'HAC-Groupsum': 'Driscoll and Kraay Standard Errors are robust to '
                    'cluster correlation (HAC-Groupsum)',
    'none': 'Covariance matrix not calculated.',
    'approx': 'Covariance matrix calculated using numerical ({approx_type}) '
              'differentiation.',
    'OPG': 'Covariance matrix calculated using the outer product of '
           'gradients ({approx_type}).',
    'OIM': 'Covariance matrix calculated using the observed information '
           'matrix ({approx_type}) described in Harvey (1989).',
    'robust': 'Quasi-maximum likelihood covariance matrix used for '
              'robustness to some misspecifications; calculated using '
              'numerical ({approx_type}) differentiation.',
    'robust-OIM': 'Quasi-maximum likelihood covariance matrix used for '
                  'robustness to some misspecifications; calculated using the '
                  'observed information matrix ({approx_type}) described in '
                  'Harvey (1989).',
    'robust-approx': 'Quasi-maximum likelihood covariance matrix used for '
                     'robustness to some misspecifications; calculated using '
                     'numerical ({approx_type}) differentiation.',
}


def normalize_cov_type(cov_type):
    """
    Normalize the cov_type string to a canonical version

    Parameters
    ----------
    cov_type : str

    Returns
    -------
    normalized_cov_type : str
    """
    if cov_type == 'nw-panel':
        cov_type = 'hac-panel'
    if cov_type == 'nw-groupsum':
        cov_type = 'hac-groupsum'
    return cov_type


def get_robustcov_results(self, cov_type='HC1', use_t=None, **kwds):
    """create new results instance with robust covariance as default

    Parameters
    ----------
    cov_type : str
        the type of robust sandwich estimator to use. see Notes below
    use_t : bool
        If true, then the t distribution is used for inference.
        If false, then the normal distribution is used.
    kwds : depends on cov_type
        Required or optional arguments for robust covariance calculation.
        see Notes below

    Returns
    -------
    results : results instance
        This method creates a new results instance with the requested
        robust covariance as the default covariance of the parameters.
        Inferential statistics like p-values and hypothesis tests will be
        based on this covariance matrix.

    Notes
    -----
    Warning: Some of the options and defaults in cov_kwds may be changed in a
    future version.

    The covariance keywords provide an option 'scaling_factor' to adjust the
    scaling of the covariance matrix, that is the covariance is multiplied by
    this factor if it is given and is not `None`. This allows the user to
    adjust the scaling of the covariance matrix to match other statistical
    packages.
    For example, `scaling_factor=(nobs - 1.) / (nobs - k_params)` provides a
    correction so that the robust covariance matrices match those of Stata in
    some models like GLM and discrete Models.

    The following covariance types and required or optional arguments are
    currently available:

    - 'HC0', 'HC1', 'HC2', 'HC3': heteroscedasticity robust covariance

      - no keyword arguments

    - 'HAC': heteroskedasticity-autocorrelation robust covariance

      ``maxlags`` :  integer, required
        number of lags to use

      ``kernel`` : {callable, str}, optional
        kernels currently available kernels are ['bartlett', 'uniform'],
        default is Bartlett

      ``use_correction``: bool, optional
        If true, use small sample correction

    - 'cluster': clustered covariance estimator

      ``groups`` : array_like[int], required :
        Integer-valued index of clusters or groups.

      ``use_correction``: bool, optional
        If True the sandwich covariance is calculated with a small
        sample correction.
        If False the sandwich covariance is calculated without
        small sample correction.

      ``df_correction``: bool, optional
        If True (default), then the degrees of freedom for the
        inferential statistics and hypothesis tests, such as
        pvalues, f_pvalue, conf_int, and t_test and f_test, are
        based on the number of groups minus one instead of the
        total number of observations minus the number of explanatory
        variables. `df_resid` of the results instance is also
        adjusted. When `use_t` is also True, then pvalues are
        computed using the Student's t distribution using the
        corrected values. These may differ substantially from
        p-values based on the normal is the number of groups is
        small.
        If False, then `df_resid` of the results instance is not
        adjusted.


    - 'hac-groupsum': Driscoll and Kraay, heteroscedasticity and
      autocorrelation robust covariance for panel data
      # TODO: more options needed here

      ``time`` : array_like, required
        index of time periods
      ``maxlags`` : integer, required
        number of lags to use
      ``kernel`` : {callable, str}, optional
        The available kernels are ['bartlett', 'uniform']. The default is
        Bartlett.
      ``use_correction`` : {False, 'hac', 'cluster'}, optional
        If False the the sandwich covariance is calculated without small
        sample correction. If `use_correction = 'cluster'` (default),
        then the same small sample correction as in the case of
        `covtype='cluster'` is used.
      ``df_correction`` : bool, optional
        The adjustment to df_resid, see cov_type 'cluster' above

    - 'hac-panel': heteroscedasticity and autocorrelation robust standard
      errors in panel data. The data needs to be sorted in this case, the
      time series for each panel unit or cluster need to be stacked. The
      membership to a time series of an individual or group can be either
      specified by group indicators or by increasing time periods. One of
      ``groups`` or ``time`` is required. # TODO: we need more options here

      ``groups`` : array_like[int]
        indicator for groups
      ``time`` : array_like[int]
        index of time periods
      ``maxlags`` : int, required
        number of lags to use
      ``kernel`` : {callable, str}, optional
        Available kernels are ['bartlett', 'uniform'], default
        is Bartlett
      ``use_correction`` : {False, 'hac', 'cluster'}, optional
        If False the sandwich covariance is calculated without
        small sample correction.
      ``df_correction`` : bool, optional
        Adjustment to df_resid, see cov_type 'cluster' above

    **Reminder**: ``use_correction`` in "hac-groupsum" and "hac-panel" is
    not bool, needs to be in {False, 'hac', 'cluster'}.

    .. todo:: Currently there is no check for extra or misspelled keywords,
         except in the case of cov_type `HCx`
    """

    import statsmodels.stats.sandwich_covariance as sw

    cov_type = normalize_cov_type(cov_type)

    if 'kernel' in kwds:
        kwds['weights_func'] = kwds.pop('kernel')
    if 'weights_func' in kwds and not callable(kwds['weights_func']):
        kwds['weights_func'] = sw.kernel_dict[kwds['weights_func']]

    # pop because HCx raises if any kwds
    sc_factor = kwds.pop('scaling_factor', None)

    # TODO: make separate function that returns a robust cov plus info
    use_self = kwds.pop('use_self', False)
    if use_self:
        res = self
    else:
        # this does not work for most models, use raw instance instead from fit
        res = self.__class__(self.model, self.params,
                   normalized_cov_params=self.normalized_cov_params,
                   scale=self.scale)

    res.cov_type = cov_type
    # use_t might already be defined by the class, and already set
    if use_t is None:
        use_t = self.use_t
    res.cov_kwds = {'use_t':use_t}  # store for information
    res.use_t = use_t

    adjust_df = False
    if cov_type in ['cluster', 'hac-panel', 'hac-groupsum']:
        df_correction = kwds.get('df_correction', None)
        # TODO: check also use_correction, do I need all combinations?
        if df_correction is not False: # i.e. in [None, True]:
            # user did not explicitely set it to False
            adjust_df = True

    res.cov_kwds['adjust_df'] = adjust_df

    # verify and set kwds, and calculate cov
    # TODO: this should be outsourced in a function so we can reuse it in
    #       other models
    # TODO: make it DRYer   repeated code for checking kwds
    if cov_type.upper() in ('HC0', 'HC1', 'HC2', 'HC3'):
        if kwds:
            raise ValueError('heteroscedasticity robust covariance '
                             'does not use keywords')
        res.cov_kwds['description'] = descriptions[cov_type.upper()]

        res.cov_params_default = getattr(self, 'cov_' + cov_type.upper(), None)
        if res.cov_params_default is None:
            # results classes that do not have cov_HCx attribute
            res.cov_params_default = sw.cov_white_simple(self,
                                                         use_correction=False)
    elif cov_type.lower() == 'hac':
        maxlags = kwds['maxlags']   # required?, default in cov_hac_simple
        res.cov_kwds['maxlags'] = maxlags
        weights_func = kwds.get('weights_func', sw.weights_bartlett)
        res.cov_kwds['weights_func'] = weights_func
        use_correction = kwds.get('use_correction', False)
        res.cov_kwds['use_correction'] = use_correction
        res.cov_kwds['description'] =  descriptions['HAC'].format(
            maxlags=maxlags, correction=['without', 'with'][use_correction])

        res.cov_params_default = sw.cov_hac_simple(self, nlags=maxlags,
                                             weights_func=weights_func,
                                             use_correction=use_correction)
    elif cov_type.lower() == 'cluster':
        #cluster robust standard errors, one- or two-way
        groups = kwds['groups']
        if not hasattr(groups, 'shape'):
            groups = np.asarray(groups).T

        if groups.ndim >= 2:
            groups = groups.squeeze()

        res.cov_kwds['groups'] = groups
        use_correction = kwds.get('use_correction', True)
        res.cov_kwds['use_correction'] = use_correction
        if groups.ndim == 1:
            if adjust_df:
                # need to find number of groups
                # duplicate work
                self.n_groups = n_groups = len(np.unique(groups))
            res.cov_params_default = sw.cov_cluster(self, groups,
                                             use_correction=use_correction)

        elif groups.ndim == 2:
            if hasattr(groups, 'values'):
                groups = groups.values

            if adjust_df:
                # need to find number of groups
                # duplicate work
                n_groups0 = len(np.unique(groups[:,0]))
                n_groups1 = len(np.unique(groups[:, 1]))
                self.n_groups = (n_groups0, n_groups1)
                n_groups = min(n_groups0, n_groups1) # use for adjust_df

            # Note: sw.cov_cluster_2groups has 3 returns
            res.cov_params_default = sw.cov_cluster_2groups(self, groups,
                                         use_correction=use_correction)[0]
        else:
            raise ValueError('only two groups are supported')
        res.cov_kwds['description'] = descriptions['cluster']

    elif cov_type.lower() == 'hac-panel':
        #cluster robust standard errors
        res.cov_kwds['time'] = time = kwds.get('time', None)
        res.cov_kwds['groups'] = groups = kwds.get('groups', None)
        #TODO: nlags is currently required
        #nlags = kwds.get('nlags', True)
        #res.cov_kwds['nlags'] = nlags
        #TODO: `nlags` or `maxlags`
        res.cov_kwds['maxlags'] = maxlags = kwds['maxlags']
        use_correction = kwds.get('use_correction', 'hac')
        res.cov_kwds['use_correction'] = use_correction
        weights_func = kwds.get('weights_func', sw.weights_bartlett)
        res.cov_kwds['weights_func'] = weights_func
        # TODO: clumsy time index in cov_nw_panel
        if groups is not None:
            groups = np.asarray(groups)
            tt = (np.nonzero(groups[:-1] != groups[1:])[0] + 1).tolist()
            nobs_ = len(groups)
        elif time is not None:
            # TODO: clumsy time index in cov_nw_panel
            time = np.asarray(time)
            tt = (np.nonzero(time[1:] < time[:-1])[0] + 1).tolist()
            nobs_ = len(time)
        else:
            raise ValueError('either time or groups needs to be given')
        groupidx = lzip([0] + tt, tt + [nobs_])
        self.n_groups = n_groups = len(groupidx)
        res.cov_params_default = sw.cov_nw_panel(self, maxlags, groupidx,
                                            weights_func=weights_func,
                                            use_correction=use_correction)
        res.cov_kwds['description'] = descriptions['HAC-Panel']

    elif cov_type.lower() == 'hac-groupsum':
        # Driscoll-Kraay standard errors
        res.cov_kwds['time'] = time = kwds['time']
        #TODO: nlags is currently required
        #nlags = kwds.get('nlags', True)
        #res.cov_kwds['nlags'] = nlags
        #TODO: `nlags` or `maxlags`
        res.cov_kwds['maxlags'] = maxlags = kwds['maxlags']
        use_correction = kwds.get('use_correction', 'cluster')
        res.cov_kwds['use_correction'] = use_correction
        weights_func = kwds.get('weights_func', sw.weights_bartlett)
        res.cov_kwds['weights_func'] = weights_func
        if adjust_df:
            # need to find number of groups
            tt = (np.nonzero(time[1:] < time[:-1])[0] + 1)
            self.n_groups = n_groups = len(tt) + 1
        res.cov_params_default = sw.cov_nw_groupsum(self, maxlags, time,
                                        weights_func=weights_func,
                                        use_correction=use_correction)
        res.cov_kwds['description'] = descriptions['HAC-Groupsum']
    else:
        raise ValueError('cov_type not recognized. See docstring for ' +
                         'available options and spelling')

    # generic optional factor to scale covariance

    res.cov_kwds['scaling_factor'] = sc_factor
    if sc_factor is not None:
        res.cov_params_default *= sc_factor

    if adjust_df:
        # Note: df_resid is used for scale and others, add new attribute
        res.df_resid_inference = n_groups - 1

    return res
