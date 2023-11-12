"""
Implementation of Regression on Order Statistics for imputing left-
censored (non-detect data)

Method described in *Nondetects and Data Analysis* by Dennis R.
Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
values of a dataset.

Author: Paul M. Hobson
Company: Geosyntec Consultants (Portland, OR)
Date: 2016-06-14

"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _ros_sort(df, observations, censorship, warn=False):
    """
    This function prepares a dataframe for ROS.

    It sorts ascending with
    left-censored observations first. Censored observations larger than
    the maximum uncensored observations are removed from the dataframe.

    Parameters
    ----------
    df : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    ------
    sorted_df : DataFrame
        The sorted dataframe with all columns dropped except the
        observation and censorship columns.
    """

    # separate uncensored data from censored data
    censored = df[df[censorship]].sort_values(observations, axis=0)
    uncensored = df[~df[censorship]].sort_values(observations, axis=0)

    if censored[observations].max() > uncensored[observations].max():
        censored = censored[censored[observations] <= uncensored[observations].max()]

        if warn:
            msg = ("Dropping censored observations greater than "
                   "the max uncensored observation.")
            warnings.warn(msg)

    combined = pd.concat([censored, uncensored], axis=0)
    return combined[[observations, censorship]].reset_index(drop=True)


def cohn_numbers(df, observations, censorship):
    r"""
    Computes the Cohn numbers for the detection limits in the dataset.

    The Cohn Numbers are:

        - :math:`A_j =` the number of uncensored obs above the jth
          threshold.
        - :math:`B_j =` the number of observations (cen & uncen) below
          the jth threshold.
        - :math:`C_j =` the number of censored observations at the jth
          threshold.
        - :math:`\mathrm{PE}_j =` the probability of exceeding the jth
          threshold
        - :math:`\mathrm{DL}_j =` the unique, sorted detection limits
        - :math:`\mathrm{DL}_{j+1} = \mathrm{DL}_j` shifted down a
          single index (row)

    Parameters
    ----------
    dataframe : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    cohn : DataFrame
    """

    def nuncen_above(row):
        """ A, the number of uncensored obs above the given threshold.
        """

        # index of observations above the lower_dl DL
        above = df[observations] >= row['lower_dl']

        # index of observations below the upper_dl DL
        below = df[observations] < row['upper_dl']

        # index of non-detect observations
        detect = ~df[censorship]

        # return the number of observations where all conditions are True
        return df[above & below & detect].shape[0]

    def nobs_below(row):
        """ B, the number of observations (cen & uncen) below the given
        threshold
        """

        # index of data less than the lower_dl DL
        less_than = df[observations] < row['lower_dl']

        # index of data less than or equal to the lower_dl DL
        less_thanequal = df[observations] <= row['lower_dl']

        # index of detects, non-detects
        uncensored = ~df[censorship]
        censored = df[censorship]

        # number observations less than or equal to lower_dl DL and non-detect
        LTE_censored = df[less_thanequal & censored].shape[0]

        # number of observations less than lower_dl DL and detected
        LT_uncensored = df[less_than & uncensored].shape[0]

        # return the sum
        return LTE_censored + LT_uncensored

    def ncen_equal(row):
        """ C, the number of censored observations at the given
        threshold.
        """

        censored_index = df[censorship]
        censored_data = df[observations][censored_index]
        censored_below = censored_data == row['lower_dl']
        return censored_below.sum()

    def set_upper_limit(cohn):
        """ Sets the upper_dl DL for each row of the Cohn dataframe. """
        if cohn.shape[0] > 1:
            return cohn['lower_dl'].shift(-1).fillna(value=np.inf)
        else:
            return [np.inf]

    def compute_PE(A, B):
        """ Computes the probability of excedance for each row of the
        Cohn dataframe. """
        N = len(A)
        PE = np.empty(N, dtype='float64')
        PE[-1] = 0.0
        for j in range(N-2, -1, -1):
            PE[j] = PE[j+1] + (1 - PE[j+1]) * A[j] / (A[j] + B[j])

        return PE

    # unique, sorted detection limts
    censored_data = df[censorship]
    DLs = pd.unique(df.loc[censored_data, observations])
    DLs.sort()

    # if there is a observations smaller than the minimum detection limit,
    # add that value to the array
    if DLs.shape[0] > 0:
        if df[observations].min() < DLs.min():
            DLs = np.hstack([df[observations].min(), DLs])

        # create a dataframe
        # (editted for pandas 0.14 compatibility; see commit 63f162e
        #  when `pipe` and `assign` are available)
        cohn = pd.DataFrame(DLs, columns=['lower_dl'])
        cohn.loc[:, 'upper_dl'] = set_upper_limit(cohn)
        cohn.loc[:, 'nuncen_above'] = cohn.apply(nuncen_above, axis=1)
        cohn.loc[:, 'nobs_below'] = cohn.apply(nobs_below, axis=1)
        cohn.loc[:, 'ncen_equal'] = cohn.apply(ncen_equal, axis=1)
        cohn = cohn.reindex(range(DLs.shape[0] + 1))
        cohn.loc[:, 'prob_exceedance'] = compute_PE(cohn['nuncen_above'], cohn['nobs_below'])

    else:
        dl_cols = ['lower_dl', 'upper_dl', 'nuncen_above',
                   'nobs_below', 'ncen_equal', 'prob_exceedance']
        cohn = pd.DataFrame(np.empty((0, len(dl_cols))), columns=dl_cols)

    return cohn


def _detection_limit_index(obs, cohn):
    """
    Locates the corresponding detection limit for each observation.

    Basically, creates an array of indices for the detection limits
    (Cohn numbers) corresponding to each data point.

    Parameters
    ----------
    obs : float
        A single observation from the larger dataset.

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    det_limit_index : int
        The index of the corresponding detection limit in `cohn`

    See Also
    --------
    cohn_numbers
    """

    if cohn.shape[0] > 0:
        index, = np.where(cohn['lower_dl'] <= obs)
        det_limit_index = index[-1]
    else:
        det_limit_index = 0

    return det_limit_index


def _ros_group_rank(df, dl_idx, censorship):
    """
    Ranks each observation within the data groups.

    In this case, the groups are defined by the record's detection
    limit index and censorship status.

    Parameters
    ----------
    df : DataFrame

    dl_idx : str
        Name of the column in the dataframe the index of the
        observations' corresponding detection limit in the `cohn`
        dataframe.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    ranks : ndarray
        Array of ranks for the dataset.
    """

    # (editted for pandas 0.14 compatibility; see commit 63f162e
    #  when `pipe` and `assign` are available)
    ranks = df.copy()
    ranks.loc[:, 'rank'] = 1
    ranks = (
        ranks.groupby(by=[dl_idx, censorship])['rank']
             .transform(lambda g: g.cumsum())
    )
    return ranks


def _ros_plot_pos(row, censorship, cohn):
    """
    ROS-specific plotting positions.

    Computes the plotting position for an observation based on its rank,
    censorship status, and detection limit index.

    Parameters
    ----------
    row : {Series, dict}
        Full observation (row) from a censored dataset. Requires a
        'rank', 'detection_limit', and `censorship` column.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    plotting_position : float

    See Also
    --------
    cohn_numbers
    """

    DL_index = row['det_limit_index']
    rank = row['rank']
    censored = row[censorship]

    dl_1 = cohn.iloc[DL_index]
    dl_2 = cohn.iloc[DL_index + 1]
    if censored:
        return (1 - dl_1['prob_exceedance']) * rank / (dl_1['ncen_equal']+1)
    else:
        return (1 - dl_1['prob_exceedance']) + (dl_1['prob_exceedance'] - dl_2['prob_exceedance']) * \
                rank / (dl_1['nuncen_above']+1)


def _norm_plot_pos(observations):
    """
    Computes standard normal (Gaussian) plotting positions using scipy.

    Parameters
    ----------
    observations : array_like
        Sequence of observed quantities.

    Returns
    -------
    plotting_position : array of floats
    """
    ppos, sorted_res = stats.probplot(observations, fit=False)
    return stats.norm.cdf(ppos)


def plotting_positions(df, censorship, cohn):
    """
    Compute the plotting positions for the observations.

    The ROS-specific plotting postions are based on the observations'
    rank, censorship status, and corresponding detection limit.

    Parameters
    ----------
    df : DataFrame

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    plotting_position : array of float

    See Also
    --------
    cohn_numbers
    """

    plot_pos = df.apply(lambda r: _ros_plot_pos(r, censorship, cohn), axis=1)

    # correctly sort the plotting positions of the ND data:
    ND_plotpos = plot_pos[df[censorship]]
    ND_plotpos_arr = np.require(ND_plotpos, requirements="W")
    ND_plotpos_arr.sort()
    plot_pos.loc[df[censorship].index[df[censorship]]] = ND_plotpos_arr

    return plot_pos


def _impute(df, observations, censorship, transform_in, transform_out):
    """
    Executes the basic regression on order stat (ROS) proceedure.

    Uses ROS to impute censored from the best-fit line of a
    probability plot of the uncensored values.

    Parameters
    ----------
    df : DataFrame
    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.
    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)
    transform_in, transform_out : callable
        Transformations to be applied to the data prior to fitting
        the line and after estimated values from that line. Typically,
        `np.log` and `np.exp` are used, respectively.

    Returns
    -------
    estimated : DataFrame
        A new dataframe with two new columns: "estimated" and "final".
        The "estimated" column contains of the values inferred from the
        best-fit line. The "final" column contains the estimated values
        only where the original observations were censored, and the original
        observations everwhere else.
    """

    # detect/non-detect selectors
    uncensored_mask = ~df[censorship]
    censored_mask = df[censorship]

    # fit a line to the logs of the detected data
    fit_params = stats.linregress(
        df['Zprelim'][uncensored_mask],
        transform_in(df[observations][uncensored_mask])
    )

    # pull out the slope and intercept for use later
    slope, intercept = fit_params[:2]

    # model the data based on the best-fit curve
    # (editted for pandas 0.14 compatibility; see commit 63f162e
    #  when `pipe` and `assign` are available)
    df.loc[:, 'estimated'] = transform_out(slope * df['Zprelim'][censored_mask] + intercept)
    df.loc[:, 'final'] = np.where(df[censorship], df['estimated'], df[observations])

    return df


def _do_ros(df, observations, censorship, transform_in, transform_out):
    """
    DataFrame-centric function to impute censored valies with ROS.

    Prepares a dataframe for, and then esimates the values of a censored
    dataset using Regression on Order Statistics

    Parameters
    ----------
    df : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    transform_in, transform_out : callable
        Transformations to be applied to the data prior to fitting
        the line and after estimated values from that line. Typically,
        `np.log` and `np.exp` are used, respectively.

    Returns
    -------
    estimated : DataFrame
        A new dataframe with two new columns: "estimated" and "final".
        The "estimated" column contains of the values inferred from the
        best-fit line. The "final" column contains the estimated values
        only where the original observations were censored, and the original
        observations everwhere else.
    """

    # compute the Cohn numbers
    cohn = cohn_numbers(df, observations=observations, censorship=censorship)

    # (editted for pandas 0.14 compatibility; see commit 63f162e
    #  when `pipe` and `assign` are available)
    modeled = _ros_sort(df, observations=observations, censorship=censorship)
    modeled.loc[:, 'det_limit_index'] = modeled[observations].apply(_detection_limit_index, args=(cohn,))
    modeled.loc[:, 'rank'] = _ros_group_rank(modeled, 'det_limit_index', censorship)
    modeled.loc[:, 'plot_pos'] = plotting_positions(modeled, censorship, cohn)
    modeled.loc[:, 'Zprelim'] = stats.norm.ppf(modeled['plot_pos'])

    return _impute(modeled, observations, censorship, transform_in, transform_out)


def impute_ros(observations, censorship, df=None, min_uncensored=2,
               max_fraction_censored=0.8, substitution_fraction=0.5,
               transform_in=np.log, transform_out=np.exp,
               as_array=True):
    """
    Impute censored dataset using Regression on Order Statistics (ROS).

    Method described in *Nondetects and Data Analysis* by Dennis R.
    Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
    values of a dataset. When there is insufficient non-censorded data,
    simple substitution is used.

    Parameters
    ----------
    observations : str or array-like
        Label of the column or the float array of censored observations

    censorship : str
        Label of the column or the bool array of the censorship
        status of the observations.

          * True if censored,
          * False if uncensored

    df : DataFrame, optional
        If `observations` and `censorship` are labels, this is the
        DataFrame that contains those columns.

    min_uncensored : int (default is 2)
        The minimum number of uncensored values required before ROS
        can be used to impute the censored observations. When this
        criterion is not met, simple substituion is used instead.

    max_fraction_censored : float (default is 0.8)
        The maximum fraction of censored data below which ROS can be
        used to impute the censored observations. When this fraction is
        exceeded, simple substituion is used instead.

    substitution_fraction : float (default is 0.5)
        The fraction of the detection limit to be used during simple
        substitution of the censored values.

    transform_in : callable (default is np.log)
        Transformation to be applied to the values prior to fitting a
        line to the plotting positions vs. uncensored values.

    transform_out : callable (default is np.exp)
        Transformation to be applied to the imputed censored values
        estimated from the previously computed best-fit line.

    as_array : bool (default is True)
        When True, a numpy array of the imputed observations is
        returned. Otherwise, a modified copy of the original dataframe
        with all of the intermediate calculations is returned.

    Returns
    -------
    imputed : {ndarray, DataFrame}
        The final observations where the censored values have either been
        imputed through ROS or substituted as a fraction of the
        detection limit.

    Notes
    -----
    This function requires pandas 0.14 or more recent.
    """

    # process arrays into a dataframe, if necessary
    if df is None:
        df = pd.DataFrame({'obs': observations, 'cen': censorship})
        observations = 'obs'
        censorship = 'cen'

    # basic counts/metrics of the dataset
    N_observations = df.shape[0]
    N_censored = df[censorship].astype(int).sum()
    N_uncensored = N_observations - N_censored
    fraction_censored = N_censored / N_observations

    # add plotting positions if there are no censored values
    # (editted for pandas 0.14 compatibility; see commit 63f162e
    #  when `pipe` and `assign` are available)
    if N_censored == 0:
        output = df[[observations, censorship]].copy()
        output.loc[:, 'final'] = df[observations]

    # substitute w/ fraction of the DLs if there's insufficient
    # uncensored data
    # (editted for pandas 0.14 compatibility; see commit 63f162e
    #  when `pipe` and `assign` are available)
    elif (N_uncensored < min_uncensored) or (fraction_censored > max_fraction_censored):
        output = df[[observations, censorship]].copy()
        output.loc[:, 'final'] = df[observations]
        output.loc[df[censorship], 'final'] *= substitution_fraction


    # normal ROS stuff
    else:
        output = _do_ros(df, observations, censorship, transform_in, transform_out)

    # convert to an array if necessary
    if as_array:
        output = output['final'].values

    return output
