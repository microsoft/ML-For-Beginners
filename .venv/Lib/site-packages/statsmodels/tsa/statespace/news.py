# -*- coding: utf-8 -*-
"""
News for state space models

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import FUTURE_STACK

import numpy as np
import pandas as pd

from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params


class NewsResults:
    """
    Impacts of data revisions and news on estimates of variables of interest

    Parameters
    ----------
    news_results : SimpleNamespace instance
        Results from `KalmanSmoother.news`.
    model : MLEResults
        The results object associated with the model from which the NewsResults
        was generated.
    updated : MLEResults
        The results object associated with the model containing the updated
        dataset.
    previous : MLEResults
        The results object associated with the model containing the previous
        dataset.
    impacted_variable : str, list, array, or slice, optional
        Observation variable label or slice of labels specifying particular
        impacted variables to display in output. The impacted variable(s)
        describe the variables that were *affected* by the news. If you do not
        know the labels for the variables, check the `endog_names` attribute of
        the model instance.
    tolerance : float, optional
        The numerical threshold for determining zero impact. Default is that
        any impact less than 1e-10 is assumed to be zero.
    row_labels : iterable
        Row labels (often dates) for the impacts of the revisions and news.

    Attributes
    ----------
    total_impacts : pd.DataFrame
        Updates to forecasts of impacted variables from both news and data
        revisions, E[y^i | post] - E[y^i | previous].
    update_impacts : pd.DataFrame
        Updates to forecasts of impacted variables from the news,
        E[y^i | post] - E[y^i | revisions] where y^i are the impacted variables
        of interest.
    revision_impacts : pd.DataFrame
        Updates to forecasts of impacted variables from all data revisions,
        E[y^i | revisions] - E[y^i | previous].
    news : pd.DataFrame
        The unexpected component of the updated data,
        E[y^u | post] - E[y^u | revisions] where y^u are the updated variables.
    weights : pd.DataFrame
        Weights describing the effect of news on variables of interest.
    revisions : pd.DataFrame
        The revisions between the current and previously observed data, for
        revisions for which detailed impacts were computed.
    revisions_all : pd.DataFrame
        The revisions between the current and previously observed data,
        y^r_{revised} - y^r_{previous} where y^r are the revised variables.
    revision_weights : pd.DataFrame
        Weights describing the effect of revisions on variables of interest,
        for revisions for which detailed impacts were computed.
    revision_weights_all : pd.DataFrame
        Weights describing the effect of revisions on variables of interest,
        with a new entry that includes NaNs for the revisions for which
        detailed impacts were not computed.
    update_forecasts : pd.DataFrame
        Forecasts based on the previous dataset of the variables that were
        updated, E[y^u | previous].
    update_realized : pd.DataFrame
        Actual observed data associated with the variables that were
        updated, y^u
    revisions_details_start : int
        Integer index of first period in which detailed revision impacts were
        computed.
    revision_detailed_impacts : pd.DataFrame
        Updates to forecasts of impacted variables from data revisions with
        detailed impacts, E[y^i | revisions] - E[y^i | grouped revisions].
    revision_grouped_impacts : pd.DataFrame
        Updates to forecasts of impacted variables from data revisions that
        were grouped together, E[y^i | grouped revisions] - E[y^i | previous].
    revised_prev : pd.DataFrame
        Previously observed data associated with the variables that were
        revised, for revisions for which detailed impacts were computed.
    revised_prev_all : pd.DataFrame
        Previously observed data associated with the variables that were
        revised, y^r_{previous}
    revised : pd.DataFrame
        Currently observed data associated with the variables that were
        revised, for revisions for which detailed impacts were computed.
    revised_all : pd.DataFrame
        Currently observed data associated with the variables that were
        revised, y^r_{revised}
    prev_impacted_forecasts : pd.DataFrame
        Previous forecast of the variables of interest, E[y^i | previous].
    post_impacted_forecasts : pd.DataFrame
        Forecast of the variables of interest after taking into account both
        revisions and updates, E[y^i | post].
    revisions_iloc : pd.DataFrame
        The integer locations of the data revisions in the dataset.
    revisions_ix : pd.DataFrame
        The label-based locations of the data revisions in the dataset.
    revisions_iloc_detailed : pd.DataFrame
        The integer locations of the data revisions in the dataset for which
        detailed impacts were computed.
    revisions_ix_detailed : pd.DataFrame
        The label-based locations of the data revisions in the dataset for
        which detailed impacts were computed.
    updates_iloc : pd.DataFrame
        The integer locations of the updated data points.
    updates_ix : pd.DataFrame
        The label-based locations of updated data points.
    state_index : array_like
        Index of state variables used to compute impacts.

    References
    ----------
    .. [1] Bańbura, Marta, and Michele Modugno.
           "Maximum likelihood estimation of factor models on datasets with
           arbitrary pattern of missing data."
           Journal of Applied Econometrics 29, no. 1 (2014): 133-160.
    .. [2] Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin.
           "Nowcasting."
           The Oxford Handbook of Economic Forecasting. July 8, 2011.
    .. [3] Bańbura, Marta, Domenico Giannone, Michele Modugno, and Lucrezia
           Reichlin.
           "Now-casting and the real-time data flow."
           In Handbook of economic forecasting, vol. 2, pp. 195-237.
           Elsevier, 2013.
    """
    def __init__(self, news_results, model, updated, previous,
                 impacted_variable=None, tolerance=1e-10, row_labels=None):
        # Note: `model` will be the same as one of `revised` or `previous`, but
        # we need to save it as self.model so that the `predict_dates`, which
        # were generated by the `_get_prediction_index` call, will be available
        # for use by the base wrapping code.
        self.model = model
        self.updated = updated
        self.previous = previous
        self.news_results = news_results
        self._impacted_variable = impacted_variable
        self._tolerance = tolerance
        self.row_labels = row_labels
        self.params = []  # required for `summary` to work

        self.endog_names = self.updated.model.endog_names
        self.k_endog = len(self.endog_names)

        self.n_revisions = len(self.news_results.revisions_ix)
        self.n_revisions_detailed = len(self.news_results.revisions_details)
        self.n_revisions_grouped = len(self.news_results.revisions_grouped)

        index = self.updated.model._index
        columns = np.atleast_1d(self.endog_names)

        # E[y^i | post]
        self.post_impacted_forecasts = pd.DataFrame(
            news_results.post_impacted_forecasts.T,
            index=self.row_labels, columns=columns).rename_axis(
                index='impact date', columns='impacted variable')
        # E[y^i | previous]
        self.prev_impacted_forecasts = pd.DataFrame(
            news_results.prev_impacted_forecasts.T,
            index=self.row_labels, columns=columns).rename_axis(
                index='impact date', columns='impacted variable')
        # E[y^i | post] - E[y^i | revisions]
        self.update_impacts = pd.DataFrame(
            news_results.update_impacts,
            index=self.row_labels, columns=columns).rename_axis(
                index='impact date', columns='impacted variable')
        # E[y^i | revisions] - E[y^i | grouped revisions]
        self.revision_detailed_impacts = pd.DataFrame(
            news_results.revision_detailed_impacts,
            index=self.row_labels,
            columns=columns,
            dtype=float,
        ).rename_axis(index="impact date", columns="impacted variable")
        # E[y^i | revisions] - E[y^i | previous]
        self.revision_impacts = pd.DataFrame(
            news_results.revision_impacts,
            index=self.row_labels,
            columns=columns,
            dtype=float,
        ).rename_axis(index="impact date", columns="impacted variable")
        # E[y^i | grouped revisions] - E[y^i | previous]
        self.revision_grouped_impacts = (
            self.revision_impacts
            - self.revision_detailed_impacts.fillna(0))
        if self.n_revisions_grouped == 0:
            self.revision_grouped_impacts.loc[:] = 0

        # E[y^i | post] - E[y^i | previous]
        self.total_impacts = (self.post_impacted_forecasts -
                              self.prev_impacted_forecasts)

        # Indices of revisions and updates
        self.revisions_details_start = news_results.revisions_details_start

        self.revisions_iloc = pd.DataFrame(
            list(zip(*news_results.revisions_ix)),
            index=['revision date', 'revised variable']).T
        iloc = self.revisions_iloc
        if len(iloc) > 0:
            self.revisions_ix = pd.DataFrame({
                'revision date': index[iloc['revision date']],
                'revised variable': columns[iloc['revised variable']]})
        else:
            self.revisions_ix = iloc.copy()

        mask = iloc['revision date'] >= self.revisions_details_start
        self.revisions_iloc_detailed = self.revisions_iloc[mask]
        self.revisions_ix_detailed = self.revisions_ix[mask]

        self.updates_iloc = pd.DataFrame(
            list(zip(*news_results.updates_ix)),
            index=['update date', 'updated variable']).T
        iloc = self.updates_iloc
        if len(iloc) > 0:
            self.updates_ix = pd.DataFrame({
                'update date': index[iloc['update date']],
                'updated variable': columns[iloc['updated variable']]})
        else:
            self.updates_ix = iloc.copy()

        # Index of the state variables used
        self.state_index = news_results.state_index

        # Wrap forecasts and forecasts errors
        r_ix_all = pd.MultiIndex.from_arrays([
            self.revisions_ix['revision date'],
            self.revisions_ix['revised variable']])
        r_ix = pd.MultiIndex.from_arrays([
            self.revisions_ix_detailed['revision date'],
            self.revisions_ix_detailed['revised variable']])
        u_ix = pd.MultiIndex.from_arrays([
            self.updates_ix['update date'],
            self.updates_ix['updated variable']])

        # E[y^u | post] - E[y^u | revisions]
        if news_results.news is None:
            self.news = pd.Series([], index=u_ix, name='news',
                                  dtype=model.params.dtype)
        else:
            self.news = pd.Series(news_results.news, index=u_ix, name='news')
        # Revisions to data (y^r_{revised} - y^r_{previous})
        if news_results.revisions_all is None:
            self.revisions_all = pd.Series([], index=r_ix_all, name='revision',
                                           dtype=model.params.dtype)
        else:
            self.revisions_all = pd.Series(news_results.revisions_all,
                                           index=r_ix_all, name='revision')
        # Revisions to data (y^r_{revised} - y^r_{previous}) for which detailed
        # impacts were computed
        if news_results.revisions is None:
            self.revisions = pd.Series([], index=r_ix, name='revision',
                                       dtype=model.params.dtype)
        else:
            self.revisions = pd.Series(news_results.revisions,
                                       index=r_ix, name='revision')
        # E[y^u | revised]
        if news_results.update_forecasts is None:
            self.update_forecasts = pd.Series([], index=u_ix,
                                              dtype=model.params.dtype)
        else:
            self.update_forecasts = pd.Series(
                news_results.update_forecasts, index=u_ix)
        # y^r_{revised}
        if news_results.revised_all is None:
            self.revised_all = pd.Series([], index=r_ix_all,
                                         dtype=model.params.dtype,
                                         name='revised')
        else:
            self.revised_all = pd.Series(news_results.revised_all,
                                         index=r_ix_all, name='revised')
        # y^r_{revised} for which detailed impacts were computed
        if news_results.revised is None:
            self.revised = pd.Series([], index=r_ix, dtype=model.params.dtype,
                                     name='revised')
        else:
            self.revised = pd.Series(news_results.revised, index=r_ix,
                                     name='revised')
        # y^r_{previous}
        if news_results.revised_prev_all is None:
            self.revised_prev_all = pd.Series([], index=r_ix_all,
                                              dtype=model.params.dtype)
        else:
            self.revised_prev_all = pd.Series(
                news_results.revised_prev_all, index=r_ix_all)
        # y^r_{previous} for which detailed impacts were computed
        if news_results.revised_prev is None:
            self.revised_prev = pd.Series([], index=r_ix,
                                          dtype=model.params.dtype)
        else:
            self.revised_prev = pd.Series(
                news_results.revised_prev, index=r_ix)
        # y^u
        if news_results.update_realized is None:
            self.update_realized = pd.Series([], index=u_ix,
                                             dtype=model.params.dtype)
        else:
            self.update_realized = pd.Series(
                news_results.update_realized, index=u_ix)
        cols = pd.MultiIndex.from_product([self.row_labels, columns])
        # reshaped version of gain matrix E[y A'] E[A A']^{-1}
        if len(self.updates_iloc):
            weights = news_results.gain.reshape(
                len(cols), len(u_ix))
        else:
            weights = np.zeros((len(cols), len(u_ix)))
        self.weights = pd.DataFrame(weights, index=cols, columns=u_ix).T
        self.weights.columns.names = ['impact date', 'impacted variable']

        # reshaped version of revision_weights
        if self.n_revisions_detailed > 0:
            revision_weights = news_results.revision_weights.reshape(
                len(cols), len(r_ix))
        else:
            revision_weights = np.zeros((len(cols), len(r_ix)))
        self.revision_weights = pd.DataFrame(
            revision_weights, index=cols, columns=r_ix).T
        self.revision_weights.columns.names = [
            'impact date', 'impacted variable']

        self.revision_weights_all = self.revision_weights.reindex(
            self.revised_all.index)

    @property
    def impacted_variable(self):
        return self._impacted_variable

    @impacted_variable.setter
    def impacted_variable(self, value):
        self._impacted_variable = value

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    @property
    def data_revisions(self):
        """
        Revisions to data points that existed in the previous dataset

        Returns
        -------
        data_revisions : pd.DataFrame
            Index is as MultiIndex consisting of `revision date` and
            `revised variable`. The columns are:

            - `observed (prev)`: the value of the data as it was observed
              in the previous dataset.
            - `revised`: the revised value of the data, as it is observed
              in the new dataset
            - `detailed impacts computed`: whether or not detailed impacts have
              been computed in these NewsResults for this revision

        See also
        --------
        data_updates
        """
        # Save revisions data
        data = pd.concat([
            self.revised_all.rename('revised'),
            self.revised_prev_all.rename('observed (prev)')
        ], axis=1).sort_index()
        data['detailed impacts computed'] = (
            self.revised_all.index.isin(self.revised.index))
        return data

    @property
    def data_updates(self):
        """
        Updated data; new entries that did not exist in the previous dataset

        Returns
        -------
        data_updates : pd.DataFrame
            Index is as MultiIndex consisting of `update date` and
            `updated variable`. The columns are:

            - `forecast (prev)`: the previous forecast of the new entry,
              based on the information available in the previous dataset
              (recall that for these updated data points, the previous dataset
              had no observed value for them at all)
            - `observed`: the value of the new entry, as it is observed in the
              new dataset

        See also
        --------
        data_revisions
        """
        data = pd.concat([
            self.update_realized.rename('observed'),
            self.update_forecasts.rename('forecast (prev)')
        ], axis=1).sort_index()
        return data

    @property
    def details_by_impact(self):
        """
        Details of forecast revisions from news, organized by impacts first

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted
            - `update date`: the date of the data update, that results in
              `news` that impacts the forecast of variables of interest
            - `updated variable`: the variable being updated, that results in
              `news` that impacts the forecast of variables of interest

            The columns are:

            - `forecast (prev)`: the previous forecast of the new entry,
              based on the information available in the previous dataset
            - `observed`: the value of the new entry, as it is observed in the
              new dataset
            - `news`: the news associated with the update (this is just the
              forecast error: `observed` - `forecast (prev)`)
            - `weight`: the weight describing how the `news` effects the
              forecast of the variable of interest
            - `impact`: the impact of the `news` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `news` associated with each updated datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        revisions. That information can be found in the `impacts` or
        `revision_details_by_impact` tables.

        This form of the details table is organized so that the impacted
        dates / variables are first in the index. This is convenient for
        slicing by impacted variables / dates to view the details of data
        updates for a particular variable or date.

        However, since the `forecast (prev)` and `observed` columns have a lot
        of duplication, printing the entire table gives a result that is less
        easy to parse than that produced by the `details_by_update` property.
        `details_by_update` contains the same information but is organized to
        be more convenient for displaying the entire table of detailed updates.
        At the same time, `details_by_update` is less convenient for
        subsetting.

        See Also
        --------
        details_by_update
        revision_details_by_update
        impacts
        """
        s = self.weights.stack(level=[0, 1], **FUTURE_STACK)
        df = s.rename('weight').to_frame()
        if len(self.updates_iloc):
            df['forecast (prev)'] = self.update_forecasts
            df['observed'] = self.update_realized
            df['news'] = self.news
            df['impact'] = df['news'] * df['weight']
        else:
            df['forecast (prev)'] = []
            df['observed'] = []
            df['news'] = []
            df['impact'] = []
        df = df[['observed', 'forecast (prev)', 'news', 'weight', 'impact']]
        df = df.reorder_levels([2, 3, 0, 1]).sort_index()

        if self.impacted_variable is not None and len(df) > 0:
            df = df.loc[np.s_[:, self.impacted_variable], :]

        mask = np.abs(df['impact']) > self.tolerance
        return df[mask]

    @property
    def _revision_grouped_impacts(self):
        s = self.revision_grouped_impacts.stack(**FUTURE_STACK)
        df = s.rename('impact').to_frame()
        df = df.reindex(['revision date', 'revised variable', 'impact'],
                        axis=1)
        if self.revisions_details_start > 0:
            df['revision date'] = (
                self.updated.model._index[self.revisions_details_start - 1])
            df['revised variable'] = 'all prior revisions'
        df = (df.set_index(['revision date', 'revised variable'], append=True)
                .reorder_levels([2, 3, 0, 1]))
        return df

    @property
    def revision_details_by_impact(self):
        """
        Details of forecast revisions from revised data, organized by impacts

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted
            - `revision date`: the date of the data revision, that results in
              `revision` that impacts the forecast of variables of interest
            - `revised variable`: the variable being revised, that results in
              `news` that impacts the forecast of variables of interest

            The columns are:

            - `observed (prev)`: the previous value of the observation, as it
              was given in the previous dataset
            - `revised`: the value of the revised entry, as it is observed in
              the new dataset
            - `revision`: the revision (this is `revised` - `observed (prev)`)
            - `weight`: the weight describing how the `revision` effects the
              forecast of the variable of interest
            - `impact`: the impact of the `revision` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `revision` associated with each revised datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        new datapoints. That information can be found in the
        `impacts` or `details_by_impact` tables.

        Grouped impacts are shown in this table, with a "revision date" equal
        to the last period prior to which detailed revisions were computed and
        with "revised variable" set to the string "all prior revisions". For
        these rows, all columns except "impact" will be set to NaNs.

        This form of the details table is organized so that the impacted
        dates / variables are first in the index. This is convenient for
        slicing by impacted variables / dates to view the details of data
        updates for a particular variable or date.

        However, since the `observed (prev)` and `revised` columns have a lot
        of duplication, printing the entire table gives a result that is less
        easy to parse than that produced by the `details_by_revision` property.
        `details_by_revision` contains the same information but is organized to
        be more convenient for displaying the entire table of detailed
        revisions. At the same time, `details_by_revision` is less convenient
        for subsetting.

        See Also
        --------
        details_by_revision
        details_by_impact
        impacts
        """
        weights = self.revision_weights.stack(level=[0, 1], **FUTURE_STACK)
        df = pd.concat([
            self.revised.reindex(weights.index),
            self.revised_prev.rename('observed (prev)').reindex(weights.index),
            self.revisions.reindex(weights.index),
            weights.rename('weight'),
            (self.revisions.reindex(weights.index) * weights).rename('impact'),
        ], axis=1)

        if self.n_revisions_grouped > 0:
            df = pd.concat([df, self._revision_grouped_impacts])
            # Explicitly set names for compatibility with pandas=1.2.5
            df.index = df.index.set_names(
                ['revision date', 'revised variable',
                 'impact date', 'impacted variable'])

        df = df.reorder_levels([2, 3, 0, 1]).sort_index()

        if self.impacted_variable is not None and len(df) > 0:
            df = df.loc[np.s_[:, self.impacted_variable], :]

        mask = np.abs(df['impact']) > self.tolerance
        return df[mask]

    @property
    def details_by_update(self):
        """
        Details of forecast revisions from news, organized by updates first

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `update date`: the date of the data update, that results in
              `news` that impacts the forecast of variables of interest
            - `updated variable`: the variable being updated, that results in
              `news` that impacts the forecast of variables of interest
            - `forecast (prev)`: the previous forecast of the new entry,
              based on the information available in the previous dataset
            - `observed`: the value of the new entry, as it is observed in the
              new dataset
            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted

            The columns are:

            - `news`: the news associated with the update (this is just the
              forecast error: `observed` - `forecast (prev)`)
            - `weight`: the weight describing how the `news` affects the
              forecast of the variable of interest
            - `impact`: the impact of the `news` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `news` associated with each updated datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        revisions. That information can be found in the `impacts` table.

        This form of the details table is organized so that the updated
        dates / variables are first in the index, and in this table the index
        also contains the forecasts and observed values of the updates. This is
        convenient for displaying the entire table of detailed updates because
        it allows sparsifying duplicate entries.

        However, since it includes forecasts and observed values in the index
        of the table, it is not convenient for subsetting by the variable of
        interest. Instead, the `details_by_impact` property is organized to
        make slicing by impacted variables / dates easy. This allows, for
        example, viewing the details of data updates on a particular variable
        or date of interest.

        See Also
        --------
        details_by_impact
        impacts
        """
        s = self.weights.stack(level=[0, 1], **FUTURE_STACK)
        df = s.rename('weight').to_frame()
        if len(self.updates_iloc):
            df['forecast (prev)'] = self.update_forecasts
            df['observed'] = self.update_realized
            df['news'] = self.news
            df['impact'] = df['news'] * df['weight']
        else:
            df['forecast (prev)'] = []
            df['observed'] = []
            df['news'] = []
            df['impact'] = []
        df = df[['forecast (prev)', 'observed', 'news',
                 'weight', 'impact']]
        df = df.reset_index()
        keys = ['update date', 'updated variable', 'observed',
                'forecast (prev)', 'impact date', 'impacted variable']
        df.index = pd.MultiIndex.from_arrays([df[key] for key in keys])
        details = df.drop(keys, axis=1).sort_index()

        if self.impacted_variable is not None and len(df) > 0:
            details = details.loc[
                np.s_[:, :, :, :, :, self.impacted_variable], :]

        mask = np.abs(details['impact']) > self.tolerance
        return details[mask]

    @property
    def revision_details_by_update(self):
        """
        Details of forecast revisions from revisions, organized by updates

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `revision date`: the date of the data revision, that results in
              `revision` that impacts the forecast of variables of interest
            - `revised variable`: the variable being revised, that results in
              `news` that impacts the forecast of variables of interest
            - `observed (prev)`: the previous value of the observation, as it
              was given in the previous dataset
            - `revised`: the value of the revised entry, as it is observed in
              the new dataset
            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted

            The columns are:

            - `revision`: the revision (this is `revised` - `observed (prev)`)
            - `weight`: the weight describing how the `revision` affects the
              forecast of the variable of interest
            - `impact`: the impact of the `revision` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `revision` associated with each revised datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        new datapoints, see `details_by_update` instead.

        Grouped impacts are shown in this table, with a "revision date" equal
        to the last period prior to which detailed revisions were computed and
        with "revised variable" set to the string "all prior revisions". For
        these rows, all columns except "impact" will be set to NaNs.

        This form of the details table is organized so that the revision
        dates / variables are first in the index, and in this table the index
        also contains the previously observed and revised values. This is
        convenient for displaying the entire table of detailed revisions
        because it allows sparsifying duplicate entries.

        However, since it includes previous observations and revisions in the
        index of the table, it is not convenient for subsetting by the variable
        of interest. Instead, the `revision_details_by_impact` property is
        organized to make slicing by impacted variables / dates easy. This
        allows, for example, viewing the details of data revisions on a
        particular variable or date of interest.

        See Also
        --------
        details_by_impact
        impacts
        """
        weights = self.revision_weights.stack(level=[0, 1], **FUTURE_STACK)

        df = pd.concat([
            self.revised_prev.rename('observed (prev)').reindex(weights.index),
            self.revised.reindex(weights.index),
            self.revisions.reindex(weights.index),
            weights.rename('weight'),
            (self.revisions.reindex(weights.index) * weights).rename('impact'),
        ], axis=1)

        if self.n_revisions_grouped > 0:
            df = pd.concat([df, self._revision_grouped_impacts])
            # Explicitly set names for compatibility with pandas=1.2.5
            df.index = df.index.set_names(
                ['revision date', 'revised variable',
                 'impact date', 'impacted variable'])

        details = (df.set_index(['observed (prev)', 'revised'], append=True)
                     .reorder_levels([
                         'revision date', 'revised variable', 'revised',
                         'observed (prev)', 'impact date',
                         'impacted variable'])
                     .sort_index())

        if self.impacted_variable is not None and len(df) > 0:
            details = details.loc[
                np.s_[:, :, :, :, :, self.impacted_variable], :]

        mask = np.abs(details['impact']) > self.tolerance
        return details[mask]

    @property
    def impacts(self):
        """
        Impacts from news and revisions on all dates / variables of interest

        Returns
        -------
        impacts : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted

            The columns are:

            - `estimate (prev)`: the previous estimate / forecast of the
              date / variable of interest.
            - `impact of revisions`: the impact of all data revisions on
              the estimate of the date / variable of interest.
            - `impact of news`: the impact of all news on the estimate of
              the date / variable of interest.
            - `total impact`: the total impact of both revisions and news on
              the estimate of the date / variable of interest.
            - `estimate (new)`: the new estimate / forecast of the
              date / variable of interest after taking into account the effects
              of the revisions and news.

        Notes
        -----
        This table decomposes updated forecasts of variables of interest into
        the overall effect from revisions and news.

        This table does not break down the detail by the updated
        dates / variables. That information can be found in the
        `details_by_impact` `details_by_update` tables.

        See Also
        --------
        details_by_impact
        details_by_update
        """
        # Summary of impacts
        impacts = pd.concat([
            self.prev_impacted_forecasts.unstack().rename('estimate (prev)'),
            self.revision_impacts.unstack().rename('impact of revisions'),
            self.update_impacts.unstack().rename('impact of news'),
            self.post_impacted_forecasts.unstack().rename('estimate (new)')],
            axis=1)
        impacts['impact of revisions'] = (
            impacts['impact of revisions'].astype(float).fillna(0))
        impacts['impact of news'] = (
            impacts['impact of news'].astype(float).fillna(0))
        impacts['total impact'] = (impacts['impact of revisions'] +
                                   impacts['impact of news'])
        impacts = impacts.reorder_levels([1, 0]).sort_index()
        impacts.index.names = ['impact date', 'impacted variable']
        impacts = impacts[['estimate (prev)', 'impact of revisions',
                           'impact of news', 'total impact', 'estimate (new)']]

        if self.impacted_variable is not None:
            impacts = impacts.loc[np.s_[:, self.impacted_variable], :]

        tmp = np.abs(impacts[['impact of revisions', 'impact of news']])
        mask = (tmp > self.tolerance).any(axis=1)

        return impacts[mask]

    def summary_impacts(self, impact_date=None, impacted_variable=None,
                        groupby='impact date', show_revisions_columns=None,
                        sparsify=True, float_format='%.2f'):
        """
        Create summary table with detailed impacts from news; by date, variable

        Parameters
        ----------
        impact_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            impact periods to display. The impact date(s) describe the periods
            in which impacted variables were *affected* by the news. If this
            argument is given, the output table will only show this impact date
            or dates. Note that this argument is passed to the Pandas `loc`
            accessor, and so it should correspond to the labels of the model's
            index. If the model was created with data in a list or numpy array,
            then these labels will be zero-indexes observation integers.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            impacted variables to display. The impacted variable(s) describe
            the variables that were *affected* by the news. If you do not know
            the labels for the variables, check the `endog_names` attribute of
            the model instance.
        groupby : {impact date, impacted date}
            The primary variable for grouping results in the impacts table. The
            default is to group by update date.
        show_revisions_columns : bool, optional
            If set to False, the impacts table will not show the impacts from
            data revisions or the total impacts. Default is to show the
            revisions and totals columns if any revisions were made and
            otherwise to hide them.
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.
        float_format : str, optional
            Formatter format string syntax for converting numbers to strings.
            Default is '%.2f'.

        Returns
        -------
        impacts_table : SimpleTable
            Table describing total impacts from both revisions and news. See
            the documentation for the `impacts` attribute for more details
            about the index and columns.

        See Also
        --------
        impacts
        """
        # Squeeze for univariate models
        if impacted_variable is None and self.k_endog == 1:
            impacted_variable = self.endog_names[0]

        # Default is to only show the revisions columns if there were any
        # revisions (otherwise it would just be a column of zeros)
        if show_revisions_columns is None:
            show_revisions_columns = self.n_revisions > 0

        # Select only the variables / dates of interest
        s = list(np.s_[:, :])
        if impact_date is not None:
            s[0] = np.s_[impact_date]
        if impacted_variable is not None:
            s[1] = np.s_[impacted_variable]
        s = tuple(s)
        impacts = self.impacts.loc[s, :]

        # Make the first index level the groupby level
        groupby = groupby.lower()
        if groupby in ['impacted variable', 'impacted_variable']:
            impacts.index = impacts.index.swaplevel(1, 0)
        elif groupby not in ['impact date', 'impact_date']:
            raise ValueError('Invalid groupby for impacts table. Valid options'
                             ' are "impact date" or "impacted variable".'
                             f'Got "{groupby}".')
        impacts = impacts.sort_index()

        # Drop the non-groupby level if there's only one value
        tmp_index = impacts.index.remove_unused_levels()
        k_vars = len(tmp_index.levels[1])
        removed_level = None
        if sparsify and k_vars == 1:
            name = tmp_index.names[1]
            value = tmp_index.levels[1][0]
            removed_level = f'{name} = {value}'
            impacts.index = tmp_index.droplevel(1)
            impacts = impacts.applymap(
                lambda num: '' if pd.isnull(num) else float_format % num)
            impacts = impacts.reset_index()
            impacts.iloc[:, 0] = impacts.iloc[:, 0].map(str)
        else:
            impacts = impacts.reset_index()
            impacts.iloc[:, :2] = impacts.iloc[:, :2].applymap(str)
            impacts.iloc[:, 2:] = impacts.iloc[:, 2:].applymap(
                lambda num: '' if pd.isnull(num) else float_format % num)

        # Sparsify the groupby column
        if sparsify and groupby in impacts:
            mask = impacts[groupby] == impacts[groupby].shift(1)
            tmp = impacts.loc[mask, groupby]
            if len(tmp) > 0:
                impacts.loc[mask, groupby] = ''

        # Drop revisions and totals columns if applicable
        if not show_revisions_columns:
            impacts.drop(['impact of revisions', 'total impact'], axis=1,
                         inplace=True)

        params_data = impacts.values
        params_header = impacts.columns.tolist()
        params_stubs = None

        title = 'Impacts'
        if removed_level is not None:
            join = 'on' if groupby == 'date' else 'for'
            title += f' {join} [{removed_level}]'
        impacts_table = SimpleTable(
            params_data, params_header, params_stubs,
            txt_fmt=fmt_params, title=title)

        return impacts_table

    def summary_details(self, source='news', impact_date=None,
                        impacted_variable=None, update_date=None,
                        updated_variable=None, groupby='update date',
                        sparsify=True, float_format='%.2f',
                        multiple_tables=False):
        """
        Create summary table with detailed impacts; by date, variable

        Parameters
        ----------
        source : {news, revisions}
            The source of impacts to summarize. Default is "news".
        impact_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            impact periods to display. The impact date(s) describe the periods
            in which impacted variables were *affected* by the news. If this
            argument is given, the output table will only show this impact date
            or dates. Note that this argument is passed to the Pandas `loc`
            accessor, and so it should correspond to the labels of the model's
            index. If the model was created with data in a list or numpy array,
            then these labels will be zero-indexes observation integers.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            impacted variables to display. The impacted variable(s) describe
            the variables that were *affected* by the news. If you do not know
            the labels for the variables, check the `endog_names` attribute of
            the model instance.
        update_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            updated periods to display. The updated date(s) describe the
            periods in which the new data points were available that generated
            the news). See the note on `impact_date` for details about what
            these labels are.
        updated_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            updated variables to display. The updated variable(s) describe the
            variables that were *affected* by the news. If you do not know the
            labels for the variables, check the `endog_names` attribute of the
            model instance.
        groupby : {update date, updated date, impact date, impacted date}
            The primary variable for grouping results in the details table. The
            default is to group by update date.
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.
        float_format : str, optional
            Formatter format string syntax for converting numbers to strings.
            Default is '%.2f'.
        multiple_tables : bool, optional
            If set to True, this function will return a list of tables, one
            table for each of the unique `groupby` levels. Default is False,
            in which case this function returns a single table.

        Returns
        -------
        details_table : SimpleTable or list of SimpleTable
            Table or list of tables describing how the news from each update
            (i.e. news from a particular variable / date) translates into
            changes to the forecasts of each impacted variable variable / date.

            This table contains information about the updates and about the
            impacts. Updates are newly observed datapoints that were not
            available in the previous results set. Each update leads to news,
            and the news may cause changes in the forecasts of the impacted
            variables. The amount that a particular piece of news (from an
            update to some variable at some date) impacts a variable at some
            date depends on weights that can be computed from the model
            results.

            The data contained in this table that refer to updates are:

            - `update date` : The date at which a new datapoint was added.
            - `updated variable` : The variable for which a new datapoint was
              added.
            - `forecast (prev)` : The value that had been forecast by the
              previous model for the given updated variable and date.
            - `observed` : The observed value of the new datapoint.
            - `news` : The news is the difference between the observed value
              and the previously forecast value for a given updated variable
              and date.

            The data contained in this table that refer to impacts are:

            - `impact date` : A date associated with an impact.
            - `impacted variable` : A variable that was impacted by the news.
            - `weight` : The weight of news from a given `update date` and
              `update variable` on a given `impacted variable` at a given
              `impact date`.
            - `impact` : The revision to the smoothed estimate / forecast of
              the impacted variable at the impact date based specifically on
              the news generated by the `updated variable` at the
              `update date`.

        See Also
        --------
        details_by_impact
        details_by_update
        """
        # Squeeze for univariate models
        if self.k_endog == 1:
            if impacted_variable is None:
                impacted_variable = self.endog_names[0]
            if updated_variable is None:
                updated_variable = self.endog_names[0]

        # Select only the variables / dates of interest
        s = list(np.s_[:, :, :, :, :, :])
        if impact_date is not None:
            s[0] = np.s_[impact_date]
        if impacted_variable is not None:
            s[1] = np.s_[impacted_variable]
        if update_date is not None:
            s[2] = np.s_[update_date]
        if updated_variable is not None:
            s[3] = np.s_[updated_variable]
        s = tuple(s)

        if source == 'news':
            details = self.details_by_impact.loc[s, :]
            columns = {
                'current': 'observed',
                'prev': 'forecast (prev)',
                'update date': 'update date',
                'updated variable': 'updated variable',
                'news': 'news',
            }
        elif source == 'revisions':
            details = self.revision_details_by_impact.loc[s, :]
            columns = {
                'current': 'revised',
                'prev': 'observed (prev)',
                'update date': 'revision date',
                'updated variable': 'revised variable',
                'news': 'revision',
            }
        else:
            raise ValueError(f'Invalid `source`: {source}. Must be "news" or'
                             ' "revisions".')

        # Make the first index level the groupby level
        groupby = groupby.lower().replace('_', ' ')
        groupby_overall = 'impact'
        levels_order = [0, 1, 2, 3]
        if groupby == 'update date':
            levels_order = [2, 3, 0, 1]
            groupby_overall = 'update'
        elif groupby == 'updated variable':
            levels_order = [3, 2, 1, 0]
            groupby_overall = 'update'
        elif groupby == 'impacted variable':
            levels_order = [1, 0, 3, 2]
        elif groupby != 'impact date':
            raise ValueError('Invalid groupby for details table. Valid options'
                             ' are "update date", "updated variable",'
                             ' "impact date",or "impacted variable".'
                             f' Got "{groupby}".')
        details.index = (details.index.reorder_levels(levels_order)
                                      .remove_unused_levels())
        details = details.sort_index()

        # If our overall group-by is `update`, move forecast (prev) and
        # observed into the index
        base_levels = [0, 1, 2, 3]
        if groupby_overall == 'update':
            details.set_index([columns['current'], columns['prev']],
                              append=True, inplace=True)
            details.index = details.index.reorder_levels([0, 1, 4, 5, 2, 3])
            base_levels = [0, 1, 4, 5]

        # Drop the non-groupby levels if there's only one value
        tmp_index = details.index.remove_unused_levels()
        n_levels = len(tmp_index.levels)
        k_level_values = [len(tmp_index.levels[i]) for i in range(n_levels)]
        removed_levels = []
        if sparsify:
            for i in sorted(base_levels)[::-1][:-1]:
                if k_level_values[i] == 1:
                    name = tmp_index.names[i]
                    value = tmp_index.levels[i][0]
                    can_drop = (
                        (name == columns['update date']
                            and update_date is not None) or
                        (name == columns['updated variable']
                            and updated_variable is not None) or
                        (name == 'impact date'
                            and impact_date is not None) or
                        (name == 'impacted variable'
                            and (impacted_variable is not None or
                                 self.impacted_variable is not None)))
                    if can_drop or not multiple_tables:
                        removed_levels.insert(0, f'{name} = {value}')
                        details.index = tmp_index = tmp_index.droplevel(i)

        # Move everything to columns
        details = details.reset_index()

        # Function for formatting numbers
        def str_format(num, mark_ones=False, mark_zeroes=False):
            if pd.isnull(num):
                out = ''
            elif mark_ones and np.abs(1 - num) < self.tolerance:
                out = '1.0'
            elif mark_zeroes and np.abs(num) < self.tolerance:
                out = '0'
            else:
                out = float_format % num
            return out

        # Function to create the table
        def create_table(details, removed_levels):
            # Convert everything to strings
            for key in [columns['current'], columns['prev'], columns['news'],
                        'weight', 'impact']:
                if key in details:
                    args = (
                        # mark_ones
                        True if key in ['weight'] else False,
                        # mark_zeroes
                        True if key in ['weight', 'impact'] else False)
                    details[key] = details[key].apply(str_format, args=args)
            for key in [columns['update date'], 'impact date']:
                if key in details:
                    details[key] = details[key].apply(str)

            # Sparsify index columns
            if sparsify:
                sparsify_cols = [columns['update date'],
                                 columns['updated variable'], 'impact date',
                                 'impacted variable']
                data_cols = [columns['current'], columns['prev']]
                if groupby_overall == 'update':
                    # Put data columns first, since we need to do an additional
                    # check based on the other columns before sparsifying
                    sparsify_cols = data_cols + sparsify_cols

                for key in sparsify_cols:
                    if key in details:
                        mask = details[key] == details[key].shift(1)
                        if key in data_cols:
                            if columns['update date'] in details:
                                tmp = details[columns['update date']]
                                mask &= tmp == tmp.shift(1)
                            if columns['updated variable'] in details:
                                tmp = details[columns['updated variable']]
                                mask &= tmp == tmp.shift(1)
                        details.loc[mask, key] = ''

            params_data = details.values
            params_header = [str(x) for x in details.columns.tolist()]
            params_stubs = None

            title = f"Details of {source}"
            if len(removed_levels):
                title += ' for [' + ', '.join(removed_levels) + ']'
            return SimpleTable(params_data, params_header, params_stubs,
                               txt_fmt=fmt_params, title=title)

        if multiple_tables:
            details_table = []
            for item in details[columns[groupby]].unique():
                mask = details[columns[groupby]] == item
                item_details = details[mask].drop(columns[groupby], axis=1)
                item_removed_levels = (
                    [f'{columns[groupby]} = {item}'] + removed_levels)
                details_table.append(create_table(item_details,
                                                  item_removed_levels))
        else:
            details_table = create_table(details, removed_levels)

        return details_table

    def summary_revisions(self, sparsify=True):
        """
        Create summary table showing revisions to the previous results' data

        Parameters
        ----------
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.

        Returns
        -------
        revisions_table : SimpleTable
            Table showing revisions to the previous results' data. Columns are:

            - `revision date` : date associated with a revised data point
            - `revised variable` : variable that was revised at `revision date`
            - `observed (prev)` : the observed value prior to the revision
            - `revised` : the new value after the revision
            - `revision` : the new value after the revision
            - `detailed impacts computed` : whether detailed impacts were
              computed for this revision
        """
        data = pd.merge(
            self.data_revisions, self.revisions_all, left_index=True,
            right_index=True).sort_index().reset_index()
        data = data[['revision date', 'revised variable', 'observed (prev)',
                     'revision', 'detailed impacts computed']]
        data[['revision date', 'revised variable']] = (
            data[['revision date', 'revised variable']].applymap(str))
        data.iloc[:, 2:-1] = data.iloc[:, 2:-1].applymap(
            lambda num: '' if pd.isnull(num) else '%.2f' % num)

        # Sparsify the date column
        if sparsify:
            mask = data['revision date'] == data['revision date'].shift(1)
            data.loc[mask, 'revision date'] = ''

        params_data = data.values
        params_header = data.columns.tolist()
        params_stubs = None

        title = 'Revisions to dataset:'
        revisions_table = SimpleTable(
            params_data, params_header, params_stubs,
            txt_fmt=fmt_params, title=title)

        return revisions_table

    def summary_news(self, sparsify=True):
        """
        Create summary table showing news from new data since previous results

        Parameters
        ----------
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.

        Returns
        -------
        updates_table : SimpleTable
            Table showing new datapoints that were not in the previous results'
            data. Columns are:

            - `update date` : date associated with a new data point.
            - `updated variable` : variable for which new data was added at
              `update date`.
            - `forecast (prev)` : the forecast value for the updated variable
              at the update date in the previous results object (i.e. prior to
              the data being available).
            - `observed` : the observed value of the new datapoint.

        See Also
        --------
        data_updates
        """
        data = pd.merge(
            self.data_updates, self.news, left_index=True,
            right_index=True).sort_index().reset_index()
        data[['update date', 'updated variable']] = (
            data[['update date', 'updated variable']].applymap(str))
        data.iloc[:, 2:] = data.iloc[:, 2:].applymap(
            lambda num: '' if pd.isnull(num) else '%.2f' % num)

        # Sparsify the date column
        if sparsify:
            mask = data['update date'] == data['update date'].shift(1)
            data.loc[mask, 'update date'] = ''

        params_data = data.values
        params_header = data.columns.tolist()
        params_stubs = None

        title = 'News from updated observations:'
        updates_table = SimpleTable(
            params_data, params_header, params_stubs,
            txt_fmt=fmt_params, title=title)

        return updates_table

    def summary(self, impact_date=None, impacted_variable=None,
                update_date=None, updated_variable=None,
                revision_date=None, revised_variable=None,
                impacts_groupby='impact date', details_groupby='update date',
                show_revisions_columns=None, sparsify=True,
                include_details_tables=None, include_revisions_tables=False,
                float_format='%.2f'):
        """
        Create summary tables describing news and impacts

        Parameters
        ----------
        impact_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            impact periods to display. The impact date(s) describe the periods
            in which impacted variables were *affected* by the news. If this
            argument is given, the impact and details tables will only show
            this impact date or dates. Note that this argument is passed to the
            Pandas `loc` accessor, and so it should correspond to the labels of
            the model's index. If the model was created with data in a list or
            numpy array, then these labels will be zero-indexes observation
            integers.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            impacted variables to display. The impacted variable(s) describe
            the variables that were *affected* by the news. If you do not know
            the labels for the variables, check the `endog_names` attribute of
            the model instance.
        update_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            updated periods to display. The updated date(s) describe the
            periods in which the new data points were available that generated
            the news). See the note on `impact_date` for details about what
            these labels are.
        updated_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            updated variables to display. The updated variable(s) describe the
            variables that newly added in the updated dataset and which
            generated the news. If you do not know the labels for the
            variables, check the `endog_names` attribute of the model instance.
        revision_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            revision periods to display. The revision date(s) describe the
            periods in which the data points were revised. See the note on
            `impact_date` for details about what these labels are.
        revised_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            revised variables to display. The updated variable(s) describe the
            variables that were *revised*. If you do not know the labels for
            the variables, check the `endog_names` attribute of the model
            instance.
        impacts_groupby : {impact date, impacted date}
            The primary variable for grouping results in the impacts table. The
            default is to group by update date.
        details_groupby : str
            One of "update date", "updated date", "impact date", or
            "impacted date". The primary variable for grouping results in the
            details table. Only used if the details tables are included. The
            default is to group by update date.
        show_revisions_columns : bool, optional
            If set to False, the impacts table will not show the impacts from
            data revisions or the total impacts. Default is to show the
            revisions and totals columns if any revisions were made and
            otherwise to hide them.
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.
        include_details_tables : bool, optional
            If set to True, the summary will show tables describing the details
            of how news from specific updates translate into specific impacts.
            These tables can be very long, particularly in cases where there
            were many updates and in multivariate models. The default is to
            show detailed tables only for univariate models.
        include_revisions_tables : bool, optional
            If set to True, the summary will show tables describing the
            revisions and updates that lead to impacts on variables of
            interest.
        float_format : str, optional
            Formatter format string syntax for converting numbers to strings.
            Default is '%.2f'.

        Returns
        -------
        summary_tables : Summary
            Summary tables describing news and impacts. Basic tables include:

            - A table with general information about the sample.
            - A table describing the impacts of revisions and news.
            - Tables describing revisions in the dataset since the previous
              results set (unless `include_revisions_tables=False`).

            In univariate models or if `include_details_tables=True`, one or
            more tables will additionally be included describing the details
            of how news from specific updates translate into specific impacts.

        See Also
        --------
        summary_impacts
        summary_details
        summary_revisions
        summary_updates
        """
        # Default for include_details_tables
        if include_details_tables is None:
            include_details_tables = (self.k_endog == 1)

        # Model specification results
        model = self.model.model
        title = 'News'

        def get_sample(model):
            if model._index_dates:
                mask = ~np.isnan(model.endog).all(axis=1)
                ix = model._index[mask]
                d = ix[0]
                sample = ['%s' % d]
                d = ix[-1]
                sample += ['- ' + '%s' % d]
            else:
                sample = [str(0), ' - ' + str(model.nobs)]

            return sample
        previous_sample = get_sample(self.previous.model)
        revised_sample = get_sample(self.updated.model)

        # Standardize the model name as a list of str
        model_name = model.__class__.__name__

        # Top summary table
        top_left = [('Model:', [model_name]),
                    ('Date:', None),
                    ('Time:', None)]
        if self.state_index is not None:
            k_states_used = len(self.state_index)
            if k_states_used != self.model.model.k_states:
                top_left.append(('# of included states:', [k_states_used]))

        top_right = [
            ('Original sample:', [previous_sample[0]]),
            ('', [previous_sample[1]]),
            ('Update through:', [revised_sample[1][2:]]),
            ('# of revisions:', [len(self.revisions_ix)]),
            ('# of new datapoints:', [len(self.updates_ix)])]

        summary = Summary()
        self.model.endog_names = self.model.model.endog_names
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)
        table_ix = 1

        # Impact table
        summary.tables.insert(table_ix, self.summary_impacts(
            impact_date=impact_date, impacted_variable=impacted_variable,
            groupby=impacts_groupby,
            show_revisions_columns=show_revisions_columns, sparsify=sparsify,
            float_format=float_format))
        table_ix += 1

        # News table
        if len(self.updates_iloc) > 0:
            summary.tables.insert(
                table_ix, self.summary_news(sparsify=sparsify))
            table_ix += 1

        # Detail tables
        multiple_tables = (self.k_endog > 1)
        details_tables = self.summary_details(
            source='news',
            impact_date=impact_date, impacted_variable=impacted_variable,
            update_date=update_date, updated_variable=updated_variable,
            groupby=details_groupby, sparsify=sparsify,
            float_format=float_format, multiple_tables=multiple_tables)
        if not multiple_tables:
            details_tables = [details_tables]

        if include_details_tables:
            for table in details_tables:
                summary.tables.insert(table_ix, table)
                table_ix += 1

        # Revisions
        if include_revisions_tables and self.n_revisions > 0:
            summary.tables.insert(
                table_ix, self.summary_revisions(sparsify=sparsify))
            table_ix += 1

            # Revision detail tables
            revision_details_tables = self.summary_details(
                source='revisions',
                impact_date=impact_date, impacted_variable=impacted_variable,
                update_date=revision_date, updated_variable=revised_variable,
                groupby=details_groupby, sparsify=sparsify,
                float_format=float_format, multiple_tables=multiple_tables)
            if not multiple_tables:
                revision_details_tables = [revision_details_tables]

            if include_details_tables:
                for table in revision_details_tables:
                    summary.tables.insert(table_ix, table)
                    table_ix += 1

        return summary

    def get_details(self, include_revisions=True, include_updates=True):
        details = []
        if include_updates:
            details.append(self.details_by_impact.rename(
                columns={'forecast (prev)': 'previous'}))
        if include_revisions:
            tmp = self.revision_details_by_impact.rename_axis(
                index={'revision date': 'update date',
                       'revised variable': 'updated variable'})
            tmp = tmp.rename(columns={'revised': 'observed',
                                      'observed (prev)': 'previous',
                                      'revision': 'news'})
            details.append(tmp)
        if not (include_updates or include_revisions):
            details.append(self.details_by_impact.rename(
                columns={'forecast (prev)': 'previous'}).iloc[:0])

        return pd.concat(details)

    def get_impacts(self, groupby=None, include_revisions=True,
                    include_updates=True):
        details = self.get_details(include_revisions=include_revisions,
                                   include_updates=include_updates)

        impacts = details['impact'].unstack(['impact date',
                                             'impacted variable'])

        if groupby is not None:
            impacts = (impacts.unstack('update date')
                              .groupby(groupby).sum(min_count=1)
                              .stack('update date')
                              .swaplevel()
                              .sort_index())

        return impacts
