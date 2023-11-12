# -*- coding: utf-8 -*-
"""
Dynamic factor model.

Author: Chad Fulton
License: BSD-3
"""
from collections import OrderedDict
from warnings import warn

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
    companion_matrix, is_invertible, constrain_stationary_univariate,
    constrain_stationary_multivariate, unconstrain_stationary_univariate,
    unconstrain_stationary_multivariate)
from statsmodels.tsa.statespace.kalman_smoother import (
    SMOOTHER_STATE, SMOOTHER_STATE_COV, SMOOTHER_STATE_AUTOCOV)
from statsmodels.base.data import PandasData

from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params


class FactorBlock(dict):
    """
    Helper class for describing and indexing a block of factors.

    Parameters
    ----------
    factor_names : tuple of str
        Tuple of factor names in the block (in the order that they will appear
        in the state vector).
    factor_order : int
        Order of the vector autoregression governing the factor block dynamics.
    endog_factor_map : pd.DataFrame
        Mapping from endog variable names to factor names.
    state_offset : int
        Offset of this factor block in the state vector.
    has_endog_Q : bool
        Flag if the model contains quarterly data.

    Notes
    -----
    The goal of this class is, in particular, to make it easier to retrieve
    indexes of subsets of the state vector that are associated with a
    particular block of factors.

    - `factors_ix` is a matrix of indices, with rows corresponding to factors
      in the block and columns corresponding to lags
    - `factors` is vec(factors_ix) (i.e. it stacks columns, so that it is
      `factors_ix.ravel(order='F')`). Thinking about a VAR system, the first
       k*p elements correspond to the equation for the first variable. The next
       k*p elements correspond to the equation for the second variable, and so
       on. It contains all of the lags in the state vector, which is max(5, p)
    - `factors_ar` is the subset of `factors` that have nonzero coefficients,
      so it contains lags up to p.
    - `factors_L1` only contains the first lag of the factors
    - `factors_L1_5` contains the first - fifth lags of the factors

    """

    def __init__(self, factor_names, factor_order, endog_factor_map,
                 state_offset, k_endog_Q):
        self.factor_names = factor_names
        self.k_factors = len(self.factor_names)
        self.factor_order = factor_order
        self.endog_factor_map = endog_factor_map.loc[:, factor_names]
        self.state_offset = state_offset
        self.k_endog_Q = k_endog_Q
        if self.k_endog_Q > 0:
            self._factor_order = max(5, self.factor_order)
        else:
            self._factor_order = self.factor_order
        self.k_states = self.k_factors * self._factor_order

        # Save items
        self['factors'] = self.factors
        self['factors_ar'] = self.factors_ar
        self['factors_ix'] = self.factors_ix
        self['factors_L1'] = self.factors_L1
        self['factors_L1_5'] = self.factors_L1_5

    @property
    def factors_ix(self):
        """Factor state index array, shaped (k_factors, lags)."""
        # i.e. the position in the state vector of the second lag of the third
        # factor is factors_ix[2, 1]
        # ravel(order='F') gives e.g (f0.L1, f1.L1, f0.L2, f1.L2, f0.L3, ...)
        # while
        # ravel(order='C') gives e.g (f0.L1, f0.L2, f0.L3, f1.L1, f1.L2, ...)
        o = self.state_offset
        return np.reshape(o + np.arange(self.k_factors * self._factor_order),
                          (self._factor_order, self.k_factors)).T

    @property
    def factors(self):
        """Factors and all lags in the state vector (max(5, p))."""
        # Note that this is equivalent to factors_ix with ravel(order='F')
        o = self.state_offset
        return np.s_[o:o + self.k_factors * self._factor_order]

    @property
    def factors_ar(self):
        """Factors and all lags used in the factor autoregression (p)."""
        o = self.state_offset
        return np.s_[o:o + self.k_factors * self.factor_order]

    @property
    def factors_L1(self):
        """Factors (first block / lag only)."""
        o = self.state_offset
        return np.s_[o:o + self.k_factors]

    @property
    def factors_L1_5(self):
        """Factors plus four lags."""
        o = self.state_offset
        return np.s_[o:o + self.k_factors * 5]


class DynamicFactorMQStates(dict):
    """
    Helper class for describing and indexing the state vector.

    Parameters
    ----------
    k_endog_M : int
        Number of monthly (or non-time-specific, if k_endog_Q=0) variables.
    k_endog_Q : int
        Number of quarterly variables.
    endog_names : list
        Names of the endogenous variables.
    factors : int, list, or dict
        Integer giving the number of (global) factors, a list with the names of
        (global) factors, or a dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        If this is an integer, then the factor names will be 0, 1, ....
    factor_orders : int or dict
        Integer describing the order of the vector autoregression (VAR)
        governing all factor block dynamics or dictionary with:

        - keys : factor name or tuples of factor names in a block
        - values : integer describing the VAR order for that factor block

        If a dictionary, this defines the order of the factor blocks in the
        state vector. Otherwise, factors are ordered so that factors that load
        on more variables come first (and then alphabetically, to break ties).
    factor_multiplicities : int or dict
        This argument provides a convenient way to specify multiple factors
        that load identically on variables. For example, one may want two
        "global" factors (factors that load on all variables) that evolve
        jointly according to a VAR. One could specify two global factors in the
        `factors` argument and specify that they are in the same block in the
        `factor_orders` argument, but it is easier to specify a single global
        factor in the `factors` argument, and set the order in the
        `factor_orders` argument, and then set the factor multiplicity to 2.

        This argument must be an integer describing the factor multiplicity for
        all factors or dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block
    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process. If False, the idiosyncratic component is instead
        modeled as white noise.

    Attributes
    ----------
    k_endog : int
        Total number of endogenous variables.
    k_states : int
        Total number of state variables (those associated with the factors and
        those associated with the idiosyncratic disturbances).
    k_posdef : int
        Total number of state disturbance terms (those associated with the
        factors and those associated with the idiosyncratic disturbances).
    k_endog_M : int
        Number of monthly (or non-time-specific, if k_endog_Q=0) variables.
    k_endog_Q : int
        Number of quarterly variables.
    k_factors : int
        Total number of factors. Note that factor multiplicities will have
        already been expanded.
    k_states_factors : int
        The number of state variables associated with factors (includes both
        factors and lags of factors included in the state vector).
    k_posdef_factors : int
        The number of state disturbance terms associated with factors.
    k_states_idio : int
        Total number of state variables associated with idiosyncratic
        disturbances.
    k_posdef_idio : int
        Total number of state disturbance terms associated with idiosyncratic
        disturbances.
    k_states_idio_M : int
        The number of state variables associated with idiosyncratic
        disturbances for monthly (or non-time-specific if there are no
        quarterly variables) variables. If the disturbances are AR(1), then
        this will be equal to `k_endog_M`, otherwise it will be equal to zero.
    k_states_idio_Q : int
        The number of state variables associated with idiosyncratic
        disturbances for quarterly variables. This will always be equal to
        `k_endog_Q * 5`, even if the disturbances are not AR(1).
    k_posdef_idio_M : int
        The number of state disturbance terms associated with idiosyncratic
        disturbances for monthly (or non-time-specific if there are no
        quarterly variables) variables. If the disturbances are AR(1), then
        this will be equal to `k_endog_M`, otherwise it will be equal to zero.
    k_posdef_idio_Q : int
        The number of state disturbance terms associated with idiosyncratic
        disturbances for quarterly variables. This will always be equal to
        `k_endog_Q`, even if the disturbances are not AR(1).
    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process.
    factor_blocks : list of FactorBlock
        List of `FactorBlock` helper instances for each factor block.
    factor_names : list of str
        List of factor names.
    factors : dict
        Dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        Note that factor multiplicities will have already been expanded.
    factor_orders : dict
        Dictionary with:

        - keys : tuple of factor names
        - values : integer describing autoregression order

        Note that factor multiplicities will have already been expanded.
    max_factor_order : int
        Maximum autoregression order across all factor blocks.
    factor_block_orders : pd.Series
        Series containing lag orders, with the factor block (a tuple of factor
        names) as the index.
    factor_multiplicities : dict
        Dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block
    endog_factor_map : dict
        Dictionary with:

        - keys : endog name
        - values : list of factor names
    loading_counts : pd.Series
        Series containing number of endogenous variables loading on each
        factor, with the factor name as the index.
    block_loading_counts : dict
        Dictionary with:

        - keys : tuple of factor names
        - values : average number of endogenous variables loading on the block
          (note that average is over the factors in the block)

    Notes
    -----
    The goal of this class is, in particular, to make it easier to retrieve
    indexes of subsets of the state vector.

    Note that the ordering of the factor blocks in the state vector is
    determined by the `factor_orders` argument if a dictionary. Otherwise,
    factors are ordered so that factors that load on more variables come first
    (and then alphabetically, to break ties).

    - `factors_L1` is an array with the indexes of first lag of the factors
      from each block. Ordered first by block, and then by lag.
    - `factors_L1_5` is an array with the indexes contains the first - fifth
      lags of the factors from each block. Ordered first by block, and then by
      lag.
    - `factors_L1_5_ix` is an array shaped (5, k_factors) with the indexes
      of the first - fifth lags of the factors from each block.
    - `idio_ar_L1` is an array with the indexes of the first lag of the
      idiosyncratic AR states, both monthly (if appliable) and quarterly.
    - `idio_ar_M` is a slice with the indexes of the idiosyncratic disturbance
      states for the monthly (or non-time-specific if there are no quarterly
      variables) variables. It is an empty slice if
      `idiosyncratic_ar1 = False`.
    - `idio_ar_Q` is a slice with the indexes of the idiosyncratic disturbance
      states and all lags, for the quarterly variables. It is an empty slice if
      there are no quarterly variable.
    - `idio_ar_Q_ix` is an array shaped (k_endog_Q, 5) with the indexes of the
      first - fifth lags of the idiosyncratic disturbance states for the
      quarterly variables.
    - `endog_factor_iloc` is a list of lists, with entries for each endogenous
      variable. The entry for variable `i`, `endog_factor_iloc[i]` is a list of
      indexes of the factors that variable `i` loads on. This does not include
      any lags, but it can be used with e.g. `factors_L1_5_ix` to get lags.

    """

    def __init__(self, k_endog_M, k_endog_Q, endog_names, factors,
                 factor_orders, factor_multiplicities, idiosyncratic_ar1):
        # Save model parameterization
        self.k_endog_M = k_endog_M
        self.k_endog_Q = k_endog_Q
        self.k_endog = self.k_endog_M + self.k_endog_Q
        self.idiosyncratic_ar1 = idiosyncratic_ar1

        # Validate factor-related inputs
        factors_is_int = np.issubdtype(type(factors), np.integer)
        factors_is_list = isinstance(factors, (list, tuple))
        orders_is_int = np.issubdtype(type(factor_orders), np.integer)
        if factor_multiplicities is None:
            factor_multiplicities = 1
        mult_is_int = np.issubdtype(type(factor_multiplicities), np.integer)

        if not (factors_is_int or factors_is_list or
                isinstance(factors, dict)):
            raise ValueError('`factors` argument must an integer number of'
                             ' factors, a list of global factor names, or a'
                             ' dictionary, mapping observed variables to'
                             ' factors.')
        if not (orders_is_int or isinstance(factor_orders, dict)):
            raise ValueError('`factor_orders` argument must either be an'
                             ' integer or a dictionary.')
        if not (mult_is_int or isinstance(factor_multiplicities, dict)):
            raise ValueError('`factor_multiplicities` argument must either be'
                             ' an integer or a dictionary.')

        # Expand integers
        # If `factors` is an integer, we assume that it denotes the number of
        # global factors (factors that load on each variable)
        if factors_is_int or factors_is_list:
            # Validate this here for a more informative error message
            if ((factors_is_int and factors == 0) or
                    (factors_is_list and len(factors) == 0)):
                raise ValueError('The model must contain at least one factor.')

            if factors_is_list:
                factor_names = list(factors)
            else:
                factor_names = [f'{i}' for i in range(factors)]
            factors = {name: factor_names[:] for name in endog_names}
        _factor_names = []
        for val in factors.values():
            _factor_names.extend(val)
        factor_names = set(_factor_names)
        if orders_is_int:
            factor_orders = {factor_name: factor_orders
                             for factor_name in factor_names}
        if mult_is_int:
            factor_multiplicities = {factor_name: factor_multiplicities
                                     for factor_name in factor_names}

        # Apply the factor multiplicities
        factors, factor_orders = self._apply_factor_multiplicities(
            factors, factor_orders, factor_multiplicities)

        # Save the (potentially expanded) variables
        self.factors = factors
        self.factor_orders = factor_orders
        self.factor_multiplicities = factor_multiplicities

        # Get the mapping between endog and factors
        self.endog_factor_map = self._construct_endog_factor_map(
            factors, endog_names)
        self.k_factors = self.endog_factor_map.shape[1]

        # Validate number of factors
        # TODO: could do more extensive validation here.
        if self.k_factors > self.k_endog_M:
            raise ValueError(f'Number of factors ({self.k_factors}) cannot be'
                             ' greater than the number of monthly endogenous'
                             f' variables ({self.k_endog_M}).')

        # Get `loading_counts`: factor -> # endog loading on the factor
        self.loading_counts = (
            self.endog_factor_map.sum(axis=0).rename('count')
                .reset_index().sort_values(['count', 'factor'],
                                           ascending=[False, True])
                .set_index('factor'))
        # `block_loading_counts`: block -> average of (# loading on factor)
        # across each factor in the block
        block_loading_counts = {
            block: np.atleast_1d(
                self.loading_counts.loc[list(block), 'count']).mean(axis=0)
            for block in factor_orders.keys()}
        ix = pd.Index(block_loading_counts.keys(), tupleize_cols=False,
                      name='block')
        self.block_loading_counts = pd.Series(
            list(block_loading_counts.values()),
            index=ix, name='count').to_frame().sort_values(
                ['count', 'block'], ascending=[False, True])['count']

        # Get the mapping between factor blocks and VAR order

        # `factor_block_orders`: pd.Series of factor block -> lag order
        ix = pd.Index(factor_orders.keys(), tupleize_cols=False, name='block')
        self.factor_block_orders = pd.Series(
            list(factor_orders.values()), index=ix, name='order')

        # If the `factor_orders` variable was an integer, then it did not
        # define an ordering for the factor blocks. In this case, we use the
        # loading counts to do so. This ensures that e.g. global factors are
        # listed first.
        if orders_is_int:
            keys = self.block_loading_counts.keys()
            self.factor_block_orders = self.factor_block_orders.loc[keys]
            self.factor_block_orders.index.name = 'block'

        # Define factor_names based on factor_block_orders (instead of on those
        # from `endog_factor_map`) to (a) make sure that factors are allocated
        # to only one block, and (b) order the factor names to be consistent
        # with the block definitions.
        factor_names = pd.Series(
            np.concatenate(list(self.factor_block_orders.index)))
        missing = [name for name in self.endog_factor_map.columns
                   if name not in factor_names.tolist()]
        if len(missing):
            ix = pd.Index([(factor_name,) for factor_name in missing],
                          tupleize_cols=False, name='block')
            default_block_orders = pd.Series(np.ones(len(ix), dtype=int),
                                             index=ix, name='order')
            self.factor_block_orders = (
                self.factor_block_orders.append(default_block_orders))
            factor_names = pd.Series(
                np.concatenate(list(self.factor_block_orders.index)))
        duplicates = factor_names.duplicated()
        if duplicates.any():
            duplicate_names = set(factor_names[duplicates])
            raise ValueError('Each factor can be assigned to at most one'
                             ' block of factors in `factor_orders`.'
                             f' Duplicate entries for {duplicate_names}')
        self.factor_names = factor_names.tolist()
        self.max_factor_order = np.max(self.factor_block_orders)

        # Re-order the columns of the endog factor mapping to reflect the
        # orderings of endog_names and factor_names
        self.endog_factor_map = (
            self.endog_factor_map.loc[endog_names, factor_names])

        # Create factor block helpers, and get factor-related state and posdef
        # dimensions
        self.k_states_factors = 0
        self.k_posdef_factors = 0
        state_offset = 0
        self.factor_blocks = []
        for factor_names, factor_order in self.factor_block_orders.items():
            block = FactorBlock(factor_names, factor_order,
                                self.endog_factor_map, state_offset,
                                self.k_endog_Q)
            self.k_states_factors += block.k_states
            self.k_posdef_factors += block.k_factors
            state_offset += block.k_states

            self.factor_blocks.append(block)

        # Idiosyncratic state dimensions
        self.k_states_idio_M = self.k_endog_M if idiosyncratic_ar1 else 0
        self.k_states_idio_Q = self.k_endog_Q * 5
        self.k_states_idio = self.k_states_idio_M + self.k_states_idio_Q

        # Idiosyncratic posdef dimensions
        self.k_posdef_idio_M = self.k_endog_M if self.idiosyncratic_ar1 else 0
        self.k_posdef_idio_Q = self.k_endog_Q
        self.k_posdef_idio = self.k_posdef_idio_M + self.k_posdef_idio_Q

        # Total states, posdef
        self.k_states = self.k_states_factors + self.k_states_idio
        self.k_posdef = self.k_posdef_factors + self.k_posdef_idio

        # Cache
        self._endog_factor_iloc = None

    def _apply_factor_multiplicities(self, factors, factor_orders,
                                     factor_multiplicities):
        """
        Expand `factors` and `factor_orders` to account for factor multiplity.

        For example, if there is a `global` factor with multiplicity 2, then
        this method expands that into `global.1` and `global.2` in both the
        `factors` and `factor_orders` dictionaries.

        Parameters
        ----------
        factors : dict
            Dictionary of {endog_name: list of factor names}
        factor_orders : dict
            Dictionary of {tuple of factor names: factor order}
        factor_multiplicities : dict
            Dictionary of {factor name: factor multiplicity}

        Returns
        -------
        new_factors : dict
            Dictionary of {endog_name: list of factor names}, with factor names
            expanded to incorporate multiplicities.
        new_factors : dict
            Dictionary of {tuple of factor names: factor order}, with factor
            names in each tuple expanded to incorporate multiplicities.
        """
        # Expand the factors to account for the multiplicities
        new_factors = {}
        for endog_name, factors_list in factors.items():
            new_factor_list = []
            for factor_name in factors_list:
                n = factor_multiplicities.get(factor_name, 1)
                if n > 1:
                    new_factor_list += [f'{factor_name}.{i + 1}'
                                        for i in range(n)]
                else:
                    new_factor_list.append(factor_name)
            new_factors[endog_name] = new_factor_list

        # Expand the factor orders to account for the multiplicities
        new_factor_orders = {}
        for block, factor_order in factor_orders.items():
            if not isinstance(block, tuple):
                block = (block,)
            new_block = []
            for factor_name in block:
                n = factor_multiplicities.get(factor_name, 1)
                if n > 1:
                    new_block += [f'{factor_name}.{i + 1}'
                                  for i in range(n)]
                else:
                    new_block += [factor_name]
            new_factor_orders[tuple(new_block)] = factor_order

        return new_factors, new_factor_orders

    def _construct_endog_factor_map(self, factors, endog_names):
        """
        Construct mapping of observed variables to factors.

        Parameters
        ----------
        factors : dict
            Dictionary of {endog_name: list of factor names}
        endog_names : list of str
            List of the names of the observed variables.

        Returns
        -------
        endog_factor_map : pd.DataFrame
            Boolean dataframe with `endog_names` as the index and the factor
            names (computed from the `factors` input) as the columns. Each cell
            is True if the associated factor is allowed to load on the
            associated observed variable.

        """
        # Validate that all entries in the factors dictionary have associated
        # factors
        missing = []
        for key, value in factors.items():
            if not isinstance(value, (list, tuple)) or len(value) == 0:
                missing.append(key)
        if len(missing):
            raise ValueError('Each observed variable must be mapped to at'
                             ' least one factor in the `factors` dictionary.'
                             f' Variables missing factors are: {missing}.')

        # Validate that we have been told about the factors for each endog
        # variable. This is because it doesn't make sense to include an
        # observed variable that doesn't load on any factor
        missing = set(endog_names).difference(set(factors.keys()))
        if len(missing):
            raise ValueError('If a `factors` dictionary is provided, then'
                             ' it must include entries for each observed'
                             f' variable. Missing variables are: {missing}.')

        # Figure out the set of factor names
        # (0 is just a dummy value for the dict - we just do it this way to
        # collect the keys, in order, without duplicates.)
        factor_names = {}
        for key, value in factors.items():
            if isinstance(value, str):
                factor_names[value] = 0
            else:
                factor_names.update({v: 0 for v in value})
        factor_names = list(factor_names.keys())
        k_factors = len(factor_names)

        endog_factor_map = pd.DataFrame(
            np.zeros((self.k_endog, k_factors), dtype=bool),
            index=pd.Index(endog_names, name='endog'),
            columns=pd.Index(factor_names, name='factor'))
        for key, value in factors.items():
            endog_factor_map.loc[key, value] = True

        return endog_factor_map

    @property
    def factors_L1(self):
        """Factors."""
        ix = np.arange(self.k_states_factors)
        iloc = tuple(ix[block.factors_L1] for block in self.factor_blocks)
        return np.concatenate(iloc)

    @property
    def factors_L1_5_ix(self):
        """Factors plus any lags, index shaped (5, k_factors)."""
        ix = np.arange(self.k_states_factors)
        iloc = []
        for block in self.factor_blocks:
            iloc.append(ix[block.factors_L1_5].reshape(5, block.k_factors))
        return np.concatenate(iloc, axis=1)

    @property
    def idio_ar_L1(self):
        """Idiosyncratic AR states, (first block / lag only)."""
        ix1 = self.k_states_factors
        if self.idiosyncratic_ar1:
            ix2 = ix1 + self.k_endog
        else:
            ix2 = ix1 + self.k_endog_Q
        return np.s_[ix1:ix2]

    @property
    def idio_ar_M(self):
        """Idiosyncratic AR states for monthly variables."""
        ix1 = self.k_states_factors
        ix2 = ix1
        if self.idiosyncratic_ar1:
            ix2 += self.k_endog_M
        return np.s_[ix1:ix2]

    @property
    def idio_ar_Q(self):
        """Idiosyncratic AR states and all lags for quarterly variables."""
        # Note that this is equivalent to idio_ar_Q_ix with ravel(order='F')
        ix1 = self.k_states_factors
        if self.idiosyncratic_ar1:
            ix1 += self.k_endog_M
        ix2 = ix1 + self.k_endog_Q * 5
        return np.s_[ix1:ix2]

    @property
    def idio_ar_Q_ix(self):
        """Idiosyncratic AR (quarterly) state index, (k_endog_Q, lags)."""
        # i.e. the position in the state vector of the second lag of the third
        # quarterly variable is idio_ar_Q_ix[2, 1]
        # ravel(order='F') gives e.g (y1.L1, y2.L1, y1.L2, y2.L3, y1.L3, ...)
        # while
        # ravel(order='C') gives e.g (y1.L1, y1.L2, y1.L3, y2.L1, y2.L2, ...)
        start = self.k_states_factors
        if self.idiosyncratic_ar1:
            start += self.k_endog_M
        return (start + np.reshape(
                np.arange(5 * self.k_endog_Q), (5, self.k_endog_Q)).T)

    @property
    def endog_factor_iloc(self):
        """List of list of int, factor indexes for each observed variable."""
        # i.e. endog_factor_iloc[i] is a list of integer locations of the
        # factors that load on the ith observed variable
        if self._endog_factor_iloc is None:
            ilocs = []
            for i in range(self.k_endog):
                ilocs.append(np.where(self.endog_factor_map.iloc[i])[0])
            self._endog_factor_iloc = ilocs
        return self._endog_factor_iloc

    def __getitem__(self, key):
        """
        Use square brackets to access index / slice elements.

        This is convenient in highlighting the indexing / slice quality of
        these attributes in the code below.
        """
        if key in ['factors_L1', 'factors_L1_5_ix', 'idio_ar_L1', 'idio_ar_M',
                   'idio_ar_Q', 'idio_ar_Q_ix']:
            return getattr(self, key)
        else:
            raise KeyError(key)


class DynamicFactorMQ(mlemodel.MLEModel):
    r"""
    Dynamic factor model with EM algorithm; option for monthly/quarterly data.

    Implementation of the dynamic factor model of Bańbura and Modugno (2014)
    ([1]_) and Bańbura, Giannone, and Reichlin (2011) ([2]_). Uses the EM
    algorithm for parameter fitting, and so can accommodate a large number of
    left-hand-side variables. Specifications can include any collection of
    blocks of factors, including different factor autoregression orders, and
    can include AR(1) processes for idiosyncratic disturbances. Can
    incorporate monthly/quarterly mixed frequency data along the lines of
    Mariano and Murasawa (2011) ([4]_). A special case of this model is the
    Nowcasting model of Bok et al. (2017) ([3]_). Moreover, this model can be
    used to compute the news associated with updated data releases.

    Parameters
    ----------
    endog : array_like
        Observed time-series process :math:`y`. See the "Notes" section for
        details on how to set up a model with monthly/quarterly mixed frequency
        data.
    k_endog_monthly : int, optional
        If specifying a monthly/quarterly mixed frequency model in which the
        provided `endog` dataset contains both the monthly and quarterly data,
        this variable should be used to indicate how many of the variables
        are monthly. Note that when using the `k_endog_monthly` argument, the
        columns with monthly variables in `endog` should be ordered first, and
        the columns with quarterly variables should come afterwards. See the
        "Notes" section for details on how to set up a model with
        monthly/quarterly mixed frequency data.
    factors : int, list, or dict, optional
        Integer giving the number of (global) factors, a list with the names of
        (global) factors, or a dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        If this is an integer, then the factor names will be 0, 1, .... The
        default is a single factor that loads on all variables. Note that there
        cannot be more factors specified than there are monthly variables.
    factor_orders : int or dict, optional
        Integer describing the order of the vector autoregression (VAR)
        governing all factor block dynamics or dictionary with:

        - keys : factor name or tuples of factor names in a block
        - values : integer describing the VAR order for that factor block

        If a dictionary, this defines the order of the factor blocks in the
        state vector. Otherwise, factors are ordered so that factors that load
        on more variables come first (and then alphabetically, to break ties).
    factor_multiplicities : int or dict, optional
        This argument provides a convenient way to specify multiple factors
        that load identically on variables. For example, one may want two
        "global" factors (factors that load on all variables) that evolve
        jointly according to a VAR. One could specify two global factors in the
        `factors` argument and specify that they are in the same block in the
        `factor_orders` argument, but it is easier to specify a single global
        factor in the `factors` argument, and set the order in the
        `factor_orders` argument, and then set the factor multiplicity to 2.

        This argument must be an integer describing the factor multiplicity for
        all factors or dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block

    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process. If False, the idiosyncratic component is instead
        modeled as white noise.
    standardize : bool or tuple, optional
        If a boolean, whether or not to standardize each endogenous variable to
        have mean zero and standard deviation 1 before fitting the model. See
        "Notes" for details about how this option works with postestimation
        output. If a tuple (usually only used internally), then the tuple must
        have length 2, with each element containing a Pandas series with index
        equal to the names of the endogenous variables. The first element
        should contain the mean values and the second element should contain
        the standard deviations. Default is True.
    endog_quarterly : pandas.Series or pandas.DataFrame
        Observed quarterly variables. If provided, must be a Pandas Series or
        DataFrame with a DatetimeIndex or PeriodIndex at the quarterly
        frequency. See the "Notes" section for details on how to set up a model
        with monthly/quarterly mixed frequency data.
    init_t0 : bool, optional
        If True, this option initializes the Kalman filter with the
        distribution for :math:`\alpha_0` rather than :math:`\alpha_1`. See
        the "Notes" section for more details. This option is rarely used except
        for testing. Default is False.
    obs_cov_diag : bool, optional
        If True and if `idiosyncratic_ar1 is True`, then this option puts small
        positive values in the observation disturbance covariance matrix. This
        is not required for estimation and is rarely used except for testing.
        (It is sometimes used to prevent numerical errors, for example those
        associated with a positive semi-definite forecast error covariance
        matrix at the first time step when using EM initialization, but state
        space models in Statsmodels switch to the univariate approach in those
        cases, and so do not need to use this trick). Default is False.

    Notes
    -----
    The basic model is:

    .. math::

        y_t & = \Lambda f_t + \epsilon_t \\
        f_t & = A_1 f_{t-1} + \dots + A_p f_{t-p} + u_t

    where:

    - :math:`y_t` is observed data at time t
    - :math:`\epsilon_t` is idiosyncratic disturbance at time t (see below for
      details, including modeling serial correlation in this term)
    - :math:`f_t` is the unobserved factor at time t
    - :math:`u_t \sim N(0, Q)` is the factor disturbance at time t

    and:

    - :math:`\Lambda` is referred to as the matrix of factor loadings
    - :math:`A_i` are matrices of autoregression coefficients

    Furthermore, we allow the idiosyncratic disturbances to be serially
    correlated, so that, if `idiosyncratic_ar1=True`,
    :math:`\epsilon_{i,t} = \rho_i \epsilon_{i,t-1} + e_{i,t}`, where
    :math:`e_{i,t} \sim N(0, \sigma_i^2)`. If `idiosyncratic_ar1=False`,
    then we instead have :math:`\epsilon_{i,t} = e_{i,t}`.

    This basic setup can be found in [1]_, [2]_, [3]_, and [4]_.

    We allow for two generalizations of this model:

    1. Following [2]_, we allow multiple "blocks" of factors, which are
       independent from the other blocks of factors. Different blocks can be
       set to load on different subsets of the observed variables, and can be
       specified with different lag orders.
    2. Following [4]_ and [2]_, we allow mixed frequency models in which both
       monthly and quarterly data are used. See the section on "Mixed frequency
       models", below, for more details.

    Additional notes:

    - The observed data may contain arbitrary patterns of missing entries.

    **EM algorithm**

    This model contains a potentially very large number of parameters, and it
    can be difficult and take a prohibitively long time to numerically optimize
    the likelihood function using quasi-Newton methods. Instead, the default
    fitting method in this model uses the EM algorithm, as detailed in [1]_.
    As a result, the model can accommodate datasets with hundreds of
    observed variables.

    **Mixed frequency data**

    This model can handle mixed frequency data in two ways. In this section,
    we only briefly describe this, and refer readers to [2]_ and [4]_ for all
    details.

    First, because there can be arbitrary patterns of missing data in the
    observed vector, one can simply include lower frequency variables as
    observed in a particular higher frequency period, and missing otherwise.
    For example, in a monthly model, one could include quarterly data as
    occurring on the third month of each quarter. To use this method, one
    simply needs to combine the data into a single dataset at the higher
    frequency that can be passed to this model as the `endog` argument.
    However, depending on the type of variables used in the analysis and the
    assumptions about the data generating process, this approach may not be
    valid.

    For example, suppose that we are interested in the growth rate of real GDP,
    which is measured at a quarterly frequency. If the basic factor model is
    specified at a monthly frequency, then the quarterly growth rate in the
    third month of each quarter -- which is what we actually observe -- is
    approximated by a particular weighted average of unobserved monthly growth
    rates. We need to take this particular weight moving average into account
    in constructing our model, and this is what the second approach does.

    The second approach follows [2]_ and [4]_ in constructing a state space
    form to explicitly model the quarterly growth rates in terms of the
    unobserved monthly growth rates. To use this approach, there are two
    methods:

    1. Combine the monthly and quarterly data into a single dataset at the
       monthly frequency, with the monthly data in the first columns and the
       quarterly data in the last columns. Pass this dataset to the model as
       the `endog` argument and give the number of the variables that are
       monthly as the `k_endog_monthly` argument.
    2. Construct a monthly dataset as a Pandas DataFrame with a DatetimeIndex
       or PeriodIndex at the monthly frequency and separately construct a
       quarterly dataset as a Pandas DataFrame with a DatetimeIndex or
       PeriodIndex at the quarterly frequency. Pass the monthly DataFrame to
       the model as the `endog` argument and pass the quarterly DataFrame to
       the model as the `endog_quarterly` argument.

    Note that this only incorporates one particular type of mixed frequency
    data. See also Banbura et al. (2013). "Now-Casting and the Real-Time Data
    Flow." for discussion about other types of mixed frequency data that are
    not supported by this framework.

    **Nowcasting and the news**

    Through its support for monthly/quarterly mixed frequency data, this model
    can allow for the nowcasting of quarterly variables based on monthly
    observations. In particular, [2]_ and [3]_ use this model to construct
    nowcasts of real GDP and analyze the impacts of "the news", derived from
    incoming data on a real-time basis. This latter functionality can be
    accessed through the `news` method of the results object.

    **Standardizing data**

    As is often the case in formulating a dynamic factor model, we do not
    explicitly account for the mean of each observed variable. Instead, the
    default behavior is to standardize each variable prior to estimation. Thus
    if :math:`y_t` are the given observed data, the dynamic factor model is
    actually estimated on the standardized data defined by:

    .. math::

        x_{i, t} = (y_{i, t} - \bar y_i) / s_i

    where :math:`\bar y_i` is the sample mean and :math:`s_i` is the sample
    standard deviation.

    By default, if standardization is applied prior to estimation, results such
    as in-sample predictions, out-of-sample forecasts, and the computation of
    the "news"  are reported in the scale of the original data (i.e. the model
    output has the reverse transformation applied before it is returned to the
    user).

    Standardization can be disabled by passing `standardization=False` to the
    model constructor.

    **Identification of factors and loadings**

    The estimated factors and the factor loadings in this model are only
    identified up to an invertible transformation. As described in (the working
    paper version of) [2]_, while it is possible to impose normalizations to
    achieve identification, the EM algorithm does will converge regardless.
    Moreover, for nowcasting and forecasting purposes, identification is not
    required. This model does not impose any normalization to identify the
    factors and the factor loadings.

    **Miscellaneous**

    There are two arguments available in the model constructor that are rarely
    used but which deserve a brief mention: `init_t0` and `obs_cov_diag`. These
    arguments are provided to allow exactly matching the output of other
    packages that have slight differences in how the underlying state space
    model is set up / applied.

    - `init_t0`: state space models in Statsmodels follow Durbin and Koopman in
      initializing the model with :math:`\alpha_1 \sim N(a_1, P_1)`. Other
      implementations sometimes initialize instead with
      :math:`\alpha_0 \sim N(a_0, P_0)`. We can accommodate this by prepending
      a row of NaNs to the observed dataset.
    - `obs_cov_diag`: the state space form in [1]_ incorporates non-zero (but
      very small) diagonal elements for the observation disturbance covariance
      matrix.

    Examples
    --------
    Constructing and fitting a `DynamicFactorMQ` model.

    >>> data = sm.datasets.macrodata.load_pandas().data.iloc[-100:]
    >>> data.index = pd.period_range(start='1984Q4', end='2009Q3', freq='Q')
    >>> endog = data[['infl', 'tbilrate']].resample('M').last()
    >>> endog_Q = np.log(data[['realgdp', 'realcons']]).diff().iloc[1:] * 400

    **Basic usage**

    In the simplest case, passing only the `endog` argument results in a model
    with a single factor that follows an AR(1) process. Note that because we
    are not also providing an `endog_quarterly` dataset, `endog` can be a numpy
    array or Pandas DataFrame with any index (it does not have to be monthly).

    The `summary` method can be useful in checking the model specification.

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 1 factors in 1 blocks   # of factors:                    1
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ========================
    Dep. variable          0
    ------------------------
             infl          X
         tbilrate          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
    =====================

    **Factors**

    With `factors=2`, there will be two independent factors that will each
    evolve according to separate AR(1) processes.

    >>> mod = sm.tsa.DynamicFactorMQ(endog, factors=2)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 2 blocks   # of factors:                    2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable          0          1
    -----------------------------------
             infl          X          X
         tbilrate          X          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
             1          1
    =====================

    **Factor multiplicities**

    By instead specifying `factor_multiplicities=2`, we would still have two
    factors, but they would be dependent and would evolve jointly according
    to a VAR(1) process.

    >>> mod = sm.tsa.DynamicFactorMQ(endog, factor_multiplicities=2)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 1 blocks   # of factors:                    2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable        0.1        0.2
    -----------------------------------
             infl         X          X
         tbilrate         X          X
        Factor blocks:
    =====================
         block      order
    ---------------------
      0.1, 0.2          1
    =====================

    **Factor orders**

    In either of the above cases, we could extend the order of the (vector)
    autoregressions by using the `factor_orders` argument. For example, the
    below model would contain two independent factors that each evolve
    according to a separate AR(2) process:

    >>> mod = sm.tsa.DynamicFactorMQ(endog, factors=2, factor_orders=2)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 2 blocks   # of factors:                    2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable          0          1
    -----------------------------------
             infl          X          X
         tbilrate          X          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          2
             1          2
    =====================

    **Serial correlation in the idiosyncratic disturbances**

    By default, the model allows each idiosyncratic disturbance terms to evolve
    according to an AR(1) process. If preferred, they can instead be specified
    to be serially independent by passing `ididosyncratic_ar1=False`.

    >>> mod = sm.tsa.DynamicFactorMQ(endog, idiosyncratic_ar1=False)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 1 factors in 1 blocks   # of factors:                    1
                    + iid idiosyncratic   Idiosyncratic disturbances:    iid
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ========================
    Dep. variable          0
    ------------------------
             infl          X
         tbilrate          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
    =====================

    *Monthly / Quarterly mixed frequency*

    To specify a monthly / quarterly mixed frequency model see the (Notes
    section for more details about these models):

    >>> mod = sm.tsa.DynamicFactorMQ(endog, endog_quarterly=endog_Q)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 1 factors in 1 blocks   # of quarterly variables:        2
                + Mixed frequency (M/Q)   # of factors:                    1
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ========================
    Dep. variable          0
    ------------------------
             infl          X
         tbilrate          X
          realgdp          X
         realcons          X
        Factor blocks:
    =====================
         block      order
    ---------------------
             0          1
    =====================

    *Customize observed variable / factor loadings*

    To specify that certain that certain observed variables only load on
    certain factors, it is possible to pass a dictionary to the `factors`
    argument.

    >>> factors = {'infl': ['global']
    ...            'tbilrate': ['global']
    ...            'realgdp': ['global', 'real']
    ...            'realcons': ['global', 'real']}
    >>> mod = sm.tsa.DynamicFactorMQ(endog, endog_quarterly=endog_Q)
    >>> print(mod.summary())
                        Model Specification: Dynamic Factor Model
    ==========================================================================
    Model:         Dynamic Factor Model   # of monthly variables:          2
                + 2 factors in 2 blocks   # of quarterly variables:        2
                + Mixed frequency (M/Q)   # of factor blocks:              2
                  + AR(1) idiosyncratic   Idiosyncratic disturbances:  AR(1)
    Sample:                     1984-10   Standardize variables:        True
                              - 2009-09
    Observed variables / factor loadings
    ===================================
    Dep. variable     global       real
    -----------------------------------
             infl       X
         tbilrate       X
          realgdp       X           X
         realcons       X           X
        Factor blocks:
    =====================
         block      order
    ---------------------
        global          1
          real          1
    =====================

    **Fitting parameters**

    To fit the model, use the `fit` method. This method uses the EM algorithm
    by default.

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit()
    >>> print(res.summary())
                              Dynamic Factor Results
    ==========================================================================
    Dep. Variable:      ['infl', 'tbilrate']   No. Observations:         300
    Model:              Dynamic Factor Model   Log Likelihood       -127.909
                     + 1 factors in 1 blocks   AIC                   271.817
                       + AR(1) idiosyncratic   BIC                   301.447
    Date:                   Tue, 04 Aug 2020   HQIC                  283.675
    Time:                           15:59:11   EM Iterations              83
    Sample:                       10-31-1984
                                - 09-30-2009
    Covariance Type:            Not computed
                        Observation equation:
    ==============================================================
    Factor loadings:          0    idiosyncratic: AR(1)       var.
    --------------------------------------------------------------
                infl      -0.67                    0.39       0.73
            tbilrate      -0.63                    0.99       0.01
           Transition: Factor block 0
    =======================================
                     L1.0    error variance
    ---------------------------------------
             0       0.98              0.01
    =======================================
    Warnings:
    [1] Covariance matrix not calculated.

    *Displaying iteration progress*

    To display information about the EM iterations, use the `disp` argument.

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit(disp=10)
    EM start iterations, llf=-291.21
    EM iteration 10, llf=-157.17, convergence criterion=0.053801
    EM iteration 20, llf=-128.99, convergence criterion=0.0035545
    EM iteration 30, llf=-127.97, convergence criterion=0.00010224
    EM iteration 40, llf=-127.93, convergence criterion=1.3281e-05
    EM iteration 50, llf=-127.92, convergence criterion=5.4725e-06
    EM iteration 60, llf=-127.91, convergence criterion=2.8665e-06
    EM iteration 70, llf=-127.91, convergence criterion=1.6999e-06
    EM iteration 80, llf=-127.91, convergence criterion=1.1085e-06
    EM converged at iteration 83, llf=-127.91,
       convergence criterion=9.9004e-07 < tolerance=1e-06

    **Results: forecasting, impulse responses, and more**

    One the model is fitted, there are a number of methods available from the
    results object. Some examples include:

    *Forecasting*

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit()
    >>> print(res.forecast(steps=5))
                 infl  tbilrate
    2009-10  1.784169  0.260401
    2009-11  1.735848  0.305981
    2009-12  1.730674  0.350968
    2010-01  1.742110  0.395369
    2010-02  1.759786  0.439194

    *Impulse responses*

    >>> mod = sm.tsa.DynamicFactorMQ(endog)
    >>> res = mod.fit()
    >>> print(res.impulse_responses(steps=5))
           infl  tbilrate
    0 -1.511956 -1.341498
    1 -1.483172 -1.315960
    2 -1.454937 -1.290908
    3 -1.427240 -1.266333
    4 -1.400069 -1.242226
    5 -1.373416 -1.218578

    For other available methods (including in-sample prediction, simulation of
    time series, extending the results to incorporate new data, and the news),
    see the documentation for state space models.

    References
    ----------
    .. [1] Bańbura, Marta, and Michele Modugno.
           "Maximum likelihood estimation of factor models on datasets with
           arbitrary pattern of missing data."
           Journal of Applied Econometrics 29, no. 1 (2014): 133-160.
    .. [2] Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin.
           "Nowcasting."
           The Oxford Handbook of Economic Forecasting. July 8, 2011.
    .. [3] Bok, Brandyn, Daniele Caratelli, Domenico Giannone,
           Argia M. Sbordone, and Andrea Tambalotti. 2018.
           "Macroeconomic Nowcasting and Forecasting with Big Data."
           Annual Review of Economics 10 (1): 615-43.
           https://doi.org/10.1146/annurev-economics-080217-053214.
    .. [4] Mariano, Roberto S., and Yasutomo Murasawa.
           "A coincident index, common factors, and monthly real GDP."
           Oxford Bulletin of Economics and Statistics 72, no. 1 (2010): 27-46.

    """

    def __init__(self, endog, k_endog_monthly=None, factors=1, factor_orders=1,
                 factor_multiplicities=None, idiosyncratic_ar1=True,
                 standardize=True, endog_quarterly=None, init_t0=False,
                 obs_cov_diag=False, **kwargs):
        # Handle endog variables
        if endog_quarterly is not None:
            if k_endog_monthly is not None:
                raise ValueError('If `endog_quarterly` is specified, then'
                                 ' `endog` must contain only monthly'
                                 ' variables, and so `k_endog_monthly` cannot'
                                 ' be specified since it will be inferred from'
                                 ' the shape of `endog`.')
            endog, k_endog_monthly = self.construct_endog(
                endog, endog_quarterly)
        endog_is_pandas = _is_using_pandas(endog, None)

        if endog_is_pandas:
            if isinstance(endog, pd.Series):
                endog = endog.to_frame()
        else:
            if np.ndim(endog) < 2:
                endog = np.atleast_2d(endog).T

        if k_endog_monthly is None:
            k_endog_monthly = endog.shape[1]

        if endog_is_pandas:
            endog_names = endog.columns.tolist()
        else:
            if endog.shape[1] == 1:
                endog_names = ['y']
            else:
                endog_names = [f'y{i + 1}' for i in range(endog.shape[1])]

        self.k_endog_M = int_like(k_endog_monthly, 'k_endog_monthly')
        self.k_endog_Q = endog.shape[1] - self.k_endog_M

        # Compute helper for handling factors / state indexing
        s = self._s = DynamicFactorMQStates(
            self.k_endog_M, self.k_endog_Q, endog_names, factors,
            factor_orders, factor_multiplicities, idiosyncratic_ar1)

        # Save parameterization
        self.factors = factors
        self.factor_orders = factor_orders
        self.factor_multiplicities = factor_multiplicities

        self.endog_factor_map = self._s.endog_factor_map
        self.factor_block_orders = self._s.factor_block_orders
        self.factor_names = self._s.factor_names
        self.k_factors = self._s.k_factors
        self.k_factor_blocks = len(self.factor_block_orders)
        self.max_factor_order = self._s.max_factor_order
        self.idiosyncratic_ar1 = idiosyncratic_ar1
        self.init_t0 = init_t0
        self.obs_cov_diag = obs_cov_diag

        if self.init_t0:
            # TODO: test each of these options
            if endog_is_pandas:
                ix = pd.period_range(endog.index[0] - 1, endog.index[-1],
                                     freq=endog.index.freq)
                endog = endog.reindex(ix)
            else:
                endog = np.c_[[np.nan] * endog.shape[1], endog.T].T

        # Standardize endog, if requested
        # Note: endog_mean and endog_std will always each be 1-dimensional with
        # length equal to the number of endog variables
        if isinstance(standardize, tuple) and len(standardize) == 2:
            endog_mean, endog_std = standardize

            # Validate the input
            n = endog.shape[1]
            if (isinstance(endog_mean, pd.Series) and not
                    endog_mean.index.equals(pd.Index(endog_names))):
                raise ValueError('Invalid value passed for `standardize`:'
                                 ' if a Pandas Series, must have index'
                                 f' {endog_names}. Got {endog_mean.index}.')
            else:
                endog_mean = np.atleast_1d(endog_mean)
            if (isinstance(endog_std, pd.Series) and not
                    endog_std.index.equals(pd.Index(endog_names))):
                raise ValueError('Invalid value passed for `standardize`:'
                                 ' if a Pandas Series, must have index'
                                 f' {endog_names}. Got {endog_std.index}.')
            else:
                endog_std = np.atleast_1d(endog_std)

            if (np.shape(endog_mean) != (n,) or np.shape(endog_std) != (n,)):
                raise ValueError('Invalid value passed for `standardize`: each'
                                 f' element must be shaped ({n},).')
            standardize = True

            # Make sure we have Pandas if endog is Pandas
            if endog_is_pandas:
                endog_mean = pd.Series(endog_mean, index=endog_names)
                endog_std = pd.Series(endog_std, index=endog_names)

        elif standardize in [1, True]:
            endog_mean = endog.mean(axis=0)
            endog_std = endog.std(axis=0)
        elif standardize in [0, False]:
            endog_mean = np.zeros(endog.shape[1])
            endog_std = np.ones(endog.shape[1])
        else:
            raise ValueError('Invalid value passed for `standardize`.')
        self._endog_mean = endog_mean
        self._endog_std = endog_std
        self.standardize = standardize
        if np.any(self._endog_std < 1e-10):
            ix = np.where(self._endog_std < 1e-10)
            names = np.array(endog_names)[ix[0]].tolist()
            raise ValueError('Constant variable(s) found in observed'
                             ' variables, but constants cannot be included'
                             f' in this model. These variables are: {names}.')

        if self.standardize:
            endog = (endog - self._endog_mean) / self._endog_std

        # Observation / states slices
        o = self._o = {
            'M': np.s_[:self.k_endog_M],
            'Q': np.s_[self.k_endog_M:]}

        # Construct the basic state space representation
        super().__init__(endog, k_states=s.k_states, k_posdef=s.k_posdef,
                         **kwargs)

        # Revert the standardization for orig_endog
        if self.standardize:
            self.data.orig_endog = (
                self.data.orig_endog * self._endog_std + self._endog_mean)

        # State initialization
        # Note: we could just initialize the entire thing as stationary, but
        # doing each block separately should be faster and avoid numerical
        # issues
        if 'initialization' not in kwargs:
            self.ssm.initialize(self._default_initialization())

        # Fixed components of the state space representation

        # > design
        if self.idiosyncratic_ar1:
            self['design', o['M'], s['idio_ar_M']] = np.eye(self.k_endog_M)
        multipliers = [1, 2, 3, 2, 1]
        for i in range(len(multipliers)):
            m = multipliers[i]
            self['design', o['Q'], s['idio_ar_Q_ix'][:, i]] = (
                m * np.eye(self.k_endog_Q))

        # > obs cov
        if self.obs_cov_diag:
            self['obs_cov'] = np.eye(self.k_endog) * 1e-4

        # > transition
        for block in s.factor_blocks:
            if block.k_factors == 1:
                tmp = 0
            else:
                tmp = np.zeros((block.k_factors, block.k_factors))
            self['transition', block['factors'], block['factors']] = (
                companion_matrix([1] + [tmp] * block._factor_order).T)
        if self.k_endog_Q == 1:
            tmp = 0
        else:
            tmp = np.zeros((self.k_endog_Q, self.k_endog_Q))
        self['transition', s['idio_ar_Q'], s['idio_ar_Q']] = (
            companion_matrix([1] + [tmp] * 5).T)

        # > selection
        ix1 = ix2 = 0
        for block in s.factor_blocks:
            ix2 += block.k_factors
            self['selection', block['factors_ix'][:, 0], ix1:ix2] = (
                np.eye(block.k_factors))
            ix1 = ix2
        if self.idiosyncratic_ar1:
            ix2 = ix1 + self.k_endog_M
            self['selection', s['idio_ar_M'], ix1:ix2] = np.eye(self.k_endog_M)
            ix1 = ix2

        ix2 = ix1 + self.k_endog_Q
        self['selection', s['idio_ar_Q_ix'][:, 0], ix1:ix2] = (
            np.eye(self.k_endog_Q))

        # Parameters
        self.params = OrderedDict([
            ('loadings', np.sum(self.endog_factor_map.values)),
            ('factor_ar', np.sum([block.k_factors**2 * block.factor_order
                                  for block in s.factor_blocks])),
            ('factor_cov', np.sum([block.k_factors * (block.k_factors + 1) // 2
                                   for block in s.factor_blocks])),
            ('idiosyncratic_ar1',
                self.k_endog if self.idiosyncratic_ar1 else 0),
            ('idiosyncratic_var', self.k_endog)])
        self.k_params = np.sum(list(self.params.values()))

        # Parameter slices
        ix = np.split(np.arange(self.k_params),
                      np.cumsum(list(self.params.values()))[:-1])
        self._p = dict(zip(self.params.keys(), ix))

        # Cache
        self._loading_constraints = {}

        # Initialization kwarg keys, e.g. for cloning
        self._init_keys += [
            'factors', 'factor_orders', 'factor_multiplicities',
            'idiosyncratic_ar1', 'standardize', 'init_t0',
            'obs_cov_diag'] + list(kwargs.keys())

    @classmethod
    def construct_endog(cls, endog_monthly, endog_quarterly):
        """
        Construct a combined dataset from separate monthly and quarterly data.

        Parameters
        ----------
        endog_monthly : array_like
            Monthly dataset. If a quarterly dataset is given, then this must
            be a Pandas object with a PeriodIndex or DatetimeIndex at a monthly
            frequency.
        endog_quarterly : array_like or None
            Quarterly dataset. If not None, then this must be a Pandas object
            with a PeriodIndex or DatetimeIndex at a quarterly frequency.

        Returns
        -------
        endog : array_like
            If both endog_monthly and endog_quarterly were given, this is a
            Pandas DataFrame with a PeriodIndex at the monthly frequency, with
            all of the columns from `endog_monthly` ordered first and the
            columns from `endog_quarterly` ordered afterwards. Otherwise it is
            simply the input `endog_monthly` dataset.
        k_endog_monthly : int
            The number of monthly variables (which are ordered first) in the
            returned `endog` dataset.
        """
        # Create combined dataset
        if endog_quarterly is not None:
            # Validate endog_monthly
            base_msg = ('If given both monthly and quarterly data'
                        ' then the monthly dataset must be a Pandas'
                        ' object with a date index at a monthly frequency.')
            if not isinstance(endog_monthly, (pd.Series, pd.DataFrame)):
                raise ValueError('Given monthly dataset is not a'
                                 ' Pandas object. ' + base_msg)
            elif endog_monthly.index.inferred_type not in ("datetime64",
                                                           "period"):
                raise ValueError('Given monthly dataset has an'
                                 ' index with non-date values. ' + base_msg)
            elif not getattr(endog_monthly.index, 'freqstr', 'N')[0] == 'M':
                freqstr = getattr(endog_monthly.index, 'freqstr', 'None')
                raise ValueError('Index of given monthly dataset has a'
                                 ' non-monthly frequency (to check this,'
                                 ' examine the `freqstr` attribute of the'
                                 ' index of the dataset - it should start with'
                                 ' M if it is monthly).'
                                 f' Got {freqstr}. ' + base_msg)

            # Validate endog_quarterly
            base_msg = ('If a quarterly dataset is given, then it must be a'
                        ' Pandas object with a date index at a quarterly'
                        ' frequency.')
            if not isinstance(endog_quarterly, (pd.Series, pd.DataFrame)):
                raise ValueError('Given quarterly dataset is not a'
                                 ' Pandas object. ' + base_msg)
            elif endog_quarterly.index.inferred_type not in ("datetime64",
                                                             "period"):
                raise ValueError('Given quarterly dataset has an'
                                 ' index with non-date values. ' + base_msg)
            elif not getattr(endog_quarterly.index, 'freqstr', 'N')[0] == 'Q':
                freqstr = getattr(endog_quarterly.index, 'freqstr', 'None')
                raise ValueError('Index of given quarterly dataset'
                                 ' has a non-quarterly frequency (to check'
                                 ' this, examine the `freqstr` attribute of'
                                 ' the index of the dataset - it should start'
                                 ' with Q if it is quarterly).'
                                 f' Got {freqstr}. ' + base_msg)

            # Convert to PeriodIndex, if applicable
            if hasattr(endog_monthly.index, 'to_period'):
                endog_monthly = endog_monthly.to_period('M')
            if hasattr(endog_quarterly.index, 'to_period'):
                endog_quarterly = endog_quarterly.to_period('Q')

            # Combine the datasets
            endog = pd.concat([
                endog_monthly,
                endog_quarterly.resample('M', convention='end').first()],
                axis=1)

            # Make sure we didn't accidentally get duplicate column names
            column_counts = endog.columns.value_counts()
            if column_counts.max() > 1:
                columns = endog.columns.values.astype(object)
                for name in column_counts.index:
                    count = column_counts.loc[name]
                    if count == 1:
                        continue
                    mask = columns == name
                    columns[mask] = [f'{name}{i + 1}' for i in range(count)]
                endog.columns = columns
        else:
            endog = endog_monthly.copy()
        shape = endog_monthly.shape
        k_endog_monthly = shape[1] if len(shape) == 2 else 1

        return endog, k_endog_monthly

    def clone(self, endog, k_endog_monthly=None, endog_quarterly=None,
              retain_standardization=False, **kwargs):
        """
        Clone state space model with new data and optionally new specification.

        Parameters
        ----------
        endog : array_like
            The observed time-series process :math:`y`
        k_endog_monthly : int, optional
            If specifying a monthly/quarterly mixed frequency model in which
            the provided `endog` dataset contains both the monthly and
            quarterly data, this variable should be used to indicate how many
            of the variables are monthly.
        endog_quarterly : array_like, optional
            Observations of quarterly variables. If provided, must be a
            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at
            the quarterly frequency.
        kwargs
            Keyword arguments to pass to the new model class to change the
            model specification.

        Returns
        -------
        model : DynamicFactorMQ instance
        """
        if retain_standardization and self.standardize:
            kwargs['standardize'] = (self._endog_mean, self._endog_std)
        mod = self._clone_from_init_kwds(
            endog, k_endog_monthly=k_endog_monthly,
            endog_quarterly=endog_quarterly, **kwargs)
        return mod

    @property
    def _res_classes(self):
        return {'fit': (DynamicFactorMQResults, mlemodel.MLEResultsWrapper)}

    def _default_initialization(self):
        s = self._s
        init = initialization.Initialization(self.k_states)
        for block in s.factor_blocks:
            init.set(block['factors'], 'stationary')
        if self.idiosyncratic_ar1:
            for i in range(s['idio_ar_M'].start, s['idio_ar_M'].stop):
                init.set(i, 'stationary')
        init.set(s['idio_ar_Q'], 'stationary')
        return init

    def _get_endog_names(self, truncate=None, as_string=None):
        if truncate is None:
            truncate = False if as_string is False or self.k_endog == 1 else 24
        if as_string is False and truncate is not False:
            raise ValueError('Can only truncate endog names if they'
                             ' are returned as a string.')
        if as_string is None:
            as_string = truncate is not False

        # The base `endog_names` property is only a list if there are at least
        # two variables; often, we need it to be a list
        endog_names = self.endog_names
        if not isinstance(endog_names, list):
            endog_names = [endog_names]

        if as_string:
            endog_names = [str(name) for name in endog_names]

        if truncate is not False:
            n = truncate
            endog_names = [name if len(name) <= n else name[:n] + '...'
                           for name in endog_names]

        return endog_names

    @property
    def _model_name(self):
        model_name = [
            'Dynamic Factor Model',
            f'{self.k_factors} factors in {self.k_factor_blocks} blocks']
        if self.k_endog_Q > 0:
            model_name.append('Mixed frequency (M/Q)')

        error_type = 'AR(1)' if self.idiosyncratic_ar1 else 'iid'
        model_name.append(f'{error_type} idiosyncratic')

        return model_name

    def summary(self, truncate_endog_names=None):
        """
        Create a summary table describing the model.

        Parameters
        ----------
        truncate_endog_names : int, optional
            The number of characters to show for names of observed variables.
            Default is 24 if there is more than one observed variable, or
            an unlimited number of there is only one.
        """
        # Get endog names
        endog_names = self._get_endog_names(truncate=truncate_endog_names,
                                            as_string=True)

        title = 'Model Specification: Dynamic Factor Model'

        if self._index_dates:
            ix = self._index
            d = ix[0]
            sample = ['%s' % d]
            d = ix[-1]
            sample += ['- ' + '%s' % d]
        else:
            sample = [str(0), ' - ' + str(self.nobs)]

        # Standardize the model name as a list of str
        model_name = self._model_name

        # - Top summary table ------------------------------------------------
        top_left = []
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [
            ('Sample:', [sample[0]]),
            ('', [sample[1]])]

        top_right = []
        if self.k_endog_Q > 0:
            top_right += [
                ('# of monthly variables:', [self.k_endog_M]),
                ('# of quarterly variables:', [self.k_endog_Q])]
        else:
            top_right += [('# of observed variables:', [self.k_endog])]
        if self.k_factor_blocks == 1:
            top_right += [('# of factors:', [self.k_factors])]
        else:
            top_right += [('# of factor blocks:', [self.k_factor_blocks])]
        top_right += [('Idiosyncratic disturbances:',
                       ['AR(1)' if self.idiosyncratic_ar1 else 'iid']),
                      ('Standardize variables:', [self.standardize])]

        summary = Summary()
        self.model = self
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)
        table_ix = 1
        del self.model

        # - Endog / factor map -----------------------------------------------
        data = self.endog_factor_map.replace({True: 'X', False: ''})
        data.index = endog_names
        try:
            items = data.items()
        except AttributeError:
            # Remove after pandas 1.5 is minimum
            items = data.iteritems()
        for name, col in items:
            data[name] = data[name] + (' ' * (len(name) // 2))
        data.index.name = 'Dep. variable'
        data = data.reset_index()

        params_data = data.values
        params_header = data.columns.map(str).tolist()
        params_stubs = None

        title = 'Observed variables / factor loadings'
        table = SimpleTable(
            params_data, params_header, params_stubs,
            txt_fmt=fmt_params, title=title)

        summary.tables.insert(table_ix, table)
        table_ix += 1

        # - Factor blocks summary table --------------------------------------
        data = self.factor_block_orders.reset_index()
        data['block'] = data['block'].map(
            lambda factor_names: ', '.join(factor_names))
        try:
            data[['order']] = data[['order']].map(str)
        except AttributeError:
            data[['order']] = data[['order']].applymap(str)

        params_data = data.values
        params_header = data.columns.map(str).tolist()
        params_stubs = None

        title = 'Factor blocks:'
        table = SimpleTable(
            params_data, params_header, params_stubs,
            txt_fmt=fmt_params, title=title)

        summary.tables.insert(table_ix, table)
        table_ix += 1

        return summary

    def __str__(self):
        """Summary tables showing model specification."""
        return str(self.summary())

    @property
    def state_names(self):
        """(list of str) List of human readable names for unobserved states."""
        # Factors
        state_names = []
        for block in self._s.factor_blocks:
            state_names += [f'{name}' for name in block.factor_names[:]]
            for s in range(1, block._factor_order):
                state_names += [f'L{s}.{name}'
                                for name in block.factor_names]

        # Monthly error
        endog_names = self._get_endog_names()
        if self.idiosyncratic_ar1:
            endog_names_M = endog_names[self._o['M']]
            state_names += [f'eps_M.{name}' for name in endog_names_M]
        endog_names_Q = endog_names[self._o['Q']]

        # Quarterly error
        state_names += [f'eps_Q.{name}' for name in endog_names_Q]
        for s in range(1, 5):
            state_names += [f'L{s}.eps_Q.{name}' for name in endog_names_Q]
        return state_names

    @property
    def param_names(self):
        """(list of str) List of human readable parameter names."""
        param_names = []
        # Loadings
        # So that Lambda = params[ix].reshape(self.k_endog, self.k_factors)
        # (where Lambda stacks Lambda_M and Lambda_Q)
        endog_names = self._get_endog_names(as_string=False)
        for endog_name in endog_names:
            for block in self._s.factor_blocks:
                for factor_name in block.factor_names:
                    if self.endog_factor_map.loc[endog_name, factor_name]:
                        param_names.append(
                            f'loading.{factor_name}->{endog_name}')

        # Factor VAR
        for block in self._s.factor_blocks:
            for to_factor in block.factor_names:
                param_names += [f'L{i}.{from_factor}->{to_factor}'
                                for i in range(1, block.factor_order + 1)
                                for from_factor in block.factor_names]

        # Factor covariance
        for i in range(len(self._s.factor_blocks)):
            block = self._s.factor_blocks[i]
            param_names += [f'fb({i}).cov.chol[{j + 1},{k + 1}]'
                            for j in range(block.k_factors)
                            for k in range(j + 1)]

        # Error AR(1)
        if self.idiosyncratic_ar1:
            endog_names_M = endog_names[self._o['M']]
            param_names += [f'L1.eps_M.{name}' for name in endog_names_M]

            endog_names_Q = endog_names[self._o['Q']]
            param_names += [f'L1.eps_Q.{name}' for name in endog_names_Q]

        # Error innovation variances
        param_names += [f'sigma2.{name}' for name in endog_names]

        return param_names

    @property
    def start_params(self):
        """(array) Starting parameters for maximum likelihood estimation."""
        params = np.zeros(self.k_params, dtype=np.float64)

        # (1) estimate factors one at a time, where the first step uses
        # PCA on all `endog` variables that load on the first factor, and
        # subsequent steps use residuals from the previous steps.
        # TODO: what about factors that only load on quarterly variables?
        endog_factor_map_M = self.endog_factor_map.iloc[:self.k_endog_M]
        factors = []
        endog = np.require(
            pd.DataFrame(self.endog).interpolate().fillna(method='backfill'),
            requirements="W"
        )
        for name in self.factor_names:
            # Try to retrieve this from monthly variables, which is most
            # consistent
            endog_ix = np.where(endog_factor_map_M.loc[:, name])[0]
            # But fall back to quarterly if necessary
            if len(endog_ix) == 0:
                endog_ix = np.where(self.endog_factor_map.loc[:, name])[0]
            factor_endog = endog[:, endog_ix]

            res_pca = PCA(factor_endog, ncomp=1, method='eig', normalize=False)
            factors.append(res_pca.factors)
            endog[:, endog_ix] -= res_pca.projection
        factors = np.concatenate(factors, axis=1)

        # (2) Estimate coefficients for each endog, one at a time (OLS for
        # monthly variables, restricted OLS for quarterly). Also, compute
        # residuals.
        loadings = []
        resid = []
        for i in range(self.k_endog_M):
            factor_ix = self._s.endog_factor_iloc[i]
            factor_exog = factors[:, factor_ix]
            mod_ols = OLS(self.endog[:, i], exog=factor_exog, missing='drop')
            res_ols = mod_ols.fit()
            loadings += res_ols.params.tolist()
            resid.append(res_ols.resid)
        for i in range(self.k_endog_M, self.k_endog):
            factor_ix = self._s.endog_factor_iloc[i]
            factor_exog = lagmat(factors[:, factor_ix], 4, original='in')
            mod_glm = GLM(self.endog[:, i], factor_exog, missing='drop')
            res_glm = mod_glm.fit_constrained(self.loading_constraints(i))
            loadings += res_glm.params[:len(factor_ix)].tolist()
            resid.append(res_glm.resid_response)
        params[self._p['loadings']] = loadings

        # (3) For each factor block, use an AR or VAR model to get coefficients
        # and covariance estimate
        # Factor transitions
        stationary = True

        factor_ar = []
        factor_cov = []
        i = 0
        for block in self._s.factor_blocks:
            factors_endog = factors[:, i:i + block.k_factors]
            i += block.k_factors

            if block.factor_order == 0:
                continue

            if block.k_factors == 1:
                mod_factors = SARIMAX(factors_endog,
                                      order=(block.factor_order, 0, 0))
                sp = mod_factors.start_params
                block_factor_ar = sp[:-1]
                block_factor_cov = sp[-1:]

                coefficient_matrices = mod_factors.start_params[:-1]
            elif block.k_factors > 1:
                mod_factors = VAR(factors_endog)
                res_factors = mod_factors.fit(
                    maxlags=block.factor_order, ic=None, trend='n')

                block_factor_ar = res_factors.params.T.ravel()
                L = np.linalg.cholesky(res_factors.sigma_u)
                block_factor_cov = L[np.tril_indices_from(L)]

                coefficient_matrices = np.transpose(
                    np.reshape(block_factor_ar,
                               (block.k_factors, block.k_factors,
                                block.factor_order)), (2, 0, 1))

            # Test for stationarity
            stationary = is_invertible([1] + list(-coefficient_matrices))

            # Check for stationarity
            if not stationary:
                warn('Non-stationary starting factor autoregressive'
                     ' parameters found for factor block'
                     f' {block.factor_names}. Using zeros as starting'
                     ' parameters.')
                block_factor_ar[:] = 0
                cov_factor = np.diag(factors_endog.std(axis=0))
                block_factor_cov = (
                    cov_factor[np.tril_indices(block.k_factors)])
            factor_ar += block_factor_ar.tolist()
            factor_cov += block_factor_cov.tolist()
        params[self._p['factor_ar']] = factor_ar
        params[self._p['factor_cov']] = factor_cov

        # (4) Use residuals from step (2) to estimate the idiosyncratic
        # component
        # Idiosyncratic component
        if self.idiosyncratic_ar1:
            idio_ar1 = []
            idio_var = []
            for i in range(self.k_endog_M):
                mod_idio = SARIMAX(resid[i], order=(1, 0, 0), trend='c')
                sp = mod_idio.start_params
                idio_ar1.append(np.clip(sp[1], -0.99, 0.99))
                idio_var.append(np.clip(sp[-1], 1e-5, np.inf))
            for i in range(self.k_endog_M, self.k_endog):
                y = self.endog[:, i].copy()
                y[~np.isnan(y)] = resid[i]
                mod_idio = QuarterlyAR1(y)
                res_idio = mod_idio.fit(maxiter=10, return_params=True,
                                        disp=False)
                res_idio = mod_idio.fit_em(res_idio, maxiter=5,
                                           return_params=True)
                idio_ar1.append(np.clip(res_idio[0], -0.99, 0.99))
                idio_var.append(np.clip(res_idio[1], 1e-5, np.inf))
            params[self._p['idiosyncratic_ar1']] = idio_ar1
            params[self._p['idiosyncratic_var']] = idio_var
        else:
            idio_var = [np.var(resid[i]) for i in range(self.k_endog_M)]
            for i in range(self.k_endog_M, self.k_endog):
                y = self.endog[:, i].copy()
                y[~np.isnan(y)] = resid[i]
                mod_idio = QuarterlyAR1(y)
                res_idio = mod_idio.fit(return_params=True, disp=False)
                idio_var.append(np.clip(res_idio[1], 1e-5, np.inf))
            params[self._p['idiosyncratic_var']] = idio_var

        return params

    def transform_params(self, unconstrained):
        """
        Transform parameters from optimizer space to model space.

        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation.

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.
        """
        constrained = unconstrained.copy()

        # Stationary factor VAR
        unconstrained_factor_ar = unconstrained[self._p['factor_ar']]
        constrained_factor_ar = []
        i = 0
        for block in self._s.factor_blocks:
            length = block.k_factors**2 * block.factor_order
            tmp_coeff = np.reshape(
                unconstrained_factor_ar[i:i + length],
                (block.k_factors, block.k_factors * block.factor_order))
            tmp_cov = np.eye(block.k_factors)
            tmp_coeff, _ = constrain_stationary_multivariate(tmp_coeff,
                                                             tmp_cov)
            constrained_factor_ar += tmp_coeff.ravel().tolist()
            i += length
        constrained[self._p['factor_ar']] = constrained_factor_ar

        # Stationary idiosyncratic AR(1)
        if self.idiosyncratic_ar1:
            idio_ar1 = unconstrained[self._p['idiosyncratic_ar1']]
            constrained[self._p['idiosyncratic_ar1']] = [
                constrain_stationary_univariate(idio_ar1[i:i + 1])[0]
                for i in range(self.k_endog)]

        # Positive idiosyncratic variances
        constrained[self._p['idiosyncratic_var']] = (
            constrained[self._p['idiosyncratic_var']]**2)

        return constrained

    def untransform_params(self, constrained):
        """
        Transform parameters from model space to optimizer space.

        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        unconstrained = constrained.copy()

        # Stationary factor VAR
        constrained_factor_ar = constrained[self._p['factor_ar']]
        unconstrained_factor_ar = []
        i = 0
        for block in self._s.factor_blocks:
            length = block.k_factors**2 * block.factor_order
            tmp_coeff = np.reshape(
                constrained_factor_ar[i:i + length],
                (block.k_factors, block.k_factors * block.factor_order))
            tmp_cov = np.eye(block.k_factors)
            tmp_coeff, _ = unconstrain_stationary_multivariate(tmp_coeff,
                                                               tmp_cov)
            unconstrained_factor_ar += tmp_coeff.ravel().tolist()
            i += length
        unconstrained[self._p['factor_ar']] = unconstrained_factor_ar

        # Stationary idiosyncratic AR(1)
        if self.idiosyncratic_ar1:
            idio_ar1 = constrained[self._p['idiosyncratic_ar1']]
            unconstrained[self._p['idiosyncratic_ar1']] = [
                unconstrain_stationary_univariate(idio_ar1[i:i + 1])[0]
                for i in range(self.k_endog)]

        # Positive idiosyncratic variances
        unconstrained[self._p['idiosyncratic_var']] = (
            unconstrained[self._p['idiosyncratic_var']]**0.5)

        return unconstrained

    def update(self, params, **kwargs):
        """
        Update the parameters of the model.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True.

        """
        params = super().update(params, **kwargs)

        # Local copies
        o = self._o
        s = self._s
        p = self._p

        # Loadings
        loadings = params[p['loadings']]
        start = 0
        for i in range(self.k_endog_M):
            iloc = self._s.endog_factor_iloc[i]
            k_factors = len(iloc)
            factor_ix = s['factors_L1'][iloc]
            self['design', i, factor_ix] = loadings[start:start + k_factors]
            start += k_factors
        multipliers = np.array([1, 2, 3, 2, 1])[:, None]
        for i in range(self.k_endog_M, self.k_endog):
            iloc = self._s.endog_factor_iloc[i]
            k_factors = len(iloc)
            factor_ix = s['factors_L1_5_ix'][:, iloc]
            self['design', i, factor_ix.ravel()] = np.ravel(
                loadings[start:start + k_factors] * multipliers)
            start += k_factors

        # Factor VAR
        factor_ar = params[p['factor_ar']]
        start = 0
        for block in s.factor_blocks:
            k_params = block.k_factors**2 * block.factor_order
            A = np.reshape(
                factor_ar[start:start + k_params],
                (block.k_factors, block.k_factors * block.factor_order))
            start += k_params
            self['transition', block['factors_L1'], block['factors_ar']] = A

        # Factor covariance
        factor_cov = params[p['factor_cov']]
        start = 0
        ix1 = 0
        for block in s.factor_blocks:
            k_params = block.k_factors * (block.k_factors + 1) // 2
            L = np.zeros((block.k_factors, block.k_factors),
                         dtype=params.dtype)
            L[np.tril_indices_from(L)] = factor_cov[start:start + k_params]
            start += k_params
            Q = L @ L.T
            ix2 = ix1 + block.k_factors
            self['state_cov', ix1:ix2, ix1:ix2] = Q
            ix1 = ix2

        # Error AR(1)
        if self.idiosyncratic_ar1:
            alpha = np.diag(params[p['idiosyncratic_ar1']])
            self['transition', s['idio_ar_L1'], s['idio_ar_L1']] = alpha

        # Error variances
        if self.idiosyncratic_ar1:
            self['state_cov', self.k_factors:, self.k_factors:] = (
                np.diag(params[p['idiosyncratic_var']]))
        else:
            idio_var = params[p['idiosyncratic_var']]
            self['obs_cov', o['M'], o['M']] = np.diag(idio_var[o['M']])
            self['state_cov', self.k_factors:, self.k_factors:] = (
                np.diag(idio_var[o['Q']]))

    @property
    def loglike_constant(self):
        """
        Constant term in the joint log-likelihood function.

        Useful in facilitating comparisons to other packages that exclude the
        constant from the log-likelihood computation.
        """
        return -0.5 * (1 - np.isnan(self.endog)).sum() * np.log(2 * np.pi)

    def loading_constraints(self, i):
        r"""
        Matrix formulation of quarterly variables' factor loading constraints.

        Parameters
        ----------
        i : int
            Index of the `endog` variable to compute constraints for.

        Returns
        -------
        R : array (k_constraints, k_factors * 5)
        q : array (k_constraints,)

        Notes
        -----
        If the factors were known, then the factor loadings for the ith
        quarterly variable would be computed by a linear regression of the form

        y_i = A_i' f + B_i' L1.f + C_i' L2.f + D_i' L3.f + E_i' L4.f

        where:

        - f is (k_i x 1) and collects all of the factors that load on y_i
        - L{j}.f is (k_i x 1) and collects the jth lag of each factor
        - A_i, ..., E_i are (k_i x 1) and collect factor loadings

        As the observed variable is quarterly while the factors are monthly, we
        want to restrict the estimated regression coefficients to be:

        y_i = A_i f + 2 A_i L1.f + 3 A_i L2.f + 2 A_i L3.f + A_i L4.f

        Stack the unconstrained coefficients: \Lambda_i = [A_i' B_i' ... E_i']'

        Then the constraints can be written as follows, for l = 1, ..., k_i

        - 2 A_{i,l} - B_{i,l} = 0
        - 3 A_{i,l} - C_{i,l} = 0
        - 2 A_{i,l} - D_{i,l} = 0
        - A_{i,l} - E_{i,l} = 0

        So that k_constraints = 4 * k_i. In matrix form the constraints are:

        .. math::

            R \Lambda_i = q

        where :math:`\Lambda_i` is shaped `(k_i * 5,)`, :math:`R` is shaped
        `(k_constraints, k_i * 5)`, and :math:`q` is shaped `(k_constraints,)`.


        For example, for the case that k_i = 2, we can write:

        |  2 0   -1  0    0  0    0  0    0  0  |   | A_{i,1} |     | 0 |
        |  0 2    0 -1    0  0    0  0    0  0  |   | A_{i,2} |     | 0 |
        |  3 0    0  0   -1  0    0  0    0  0  |   | B_{i,1} |     | 0 |
        |  0 3    0  0    0 -1    0  0    0  0  |   | B_{i,2} |     | 0 |
        |  2 0    0  0    0  0   -1  0    0  0  |   | C_{i,1} |  =  | 0 |
        |  0 2    0  0    0  0    0 -1    0  0  |   | C_{i,2} |     | 0 |
        |  1 0    0  0    0  0    0  0   -1  0  |   | D_{i,1} |     | 0 |
        |  0 1    0  0    0  0    0  0    0 -1  |   | D_{i,2} |     | 0 |
                                                    | E_{i,1} |     | 0 |
                                                    | E_{i,2} |     | 0 |

        """
        if i < self.k_endog_M:
            raise ValueError('No constraints for monthly variables.')
        if i not in self._loading_constraints:
            k_factors = self.endog_factor_map.iloc[i].sum()

            R = np.zeros((k_factors * 4, k_factors * 5))
            q = np.zeros(R.shape[0])

            # Let R = [R_1 R_2]
            # Then R_1 is multiples of the identity matrix
            multipliers = np.array([1, 2, 3, 2, 1])
            R[:, :k_factors] = np.reshape(
                (multipliers[1:] * np.eye(k_factors)[..., None]).T,
                (k_factors * 4, k_factors))

            # And R_2 is the identity
            R[:, k_factors:] = np.diag([-1] * (k_factors * 4))

            self._loading_constraints[i] = (R, q)
        return self._loading_constraints[i]

    def fit(self, start_params=None, transformed=True, includes_fixed=False,
            cov_type='none', cov_kwds=None, method='em', maxiter=500,
            tolerance=1e-6, em_initialization=True, mstep_method=None,
            full_output=1, disp=False, callback=None, return_params=False,
            optim_score=None, optim_complex_step=None, optim_hessian=None,
            flags=None, low_memory=False, llf_decrease_action='revert',
            llf_decrease_tolerance=1e-4, **kwargs):
        """
        Fits the model by maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presence of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.

            Default is 'none', since computing this matrix can be very slow
            when there are a large number of parameters.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : bool, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : bool, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'em' for the EM algorithm
            - 'newton' for Newton-Raphson
            - 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        tolerance : float, optional
            Tolerance to use for convergence checking when using the EM
            algorithm. To set the tolerance for other methods, pass
            the optimizer-specific keyword argument(s).
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        optim_score : {'harvey', 'approx'} or None, optional
            The method by which the score vector is calculated. 'harvey' uses
            the method from Harvey (1989), 'approx' uses either finite
            difference or complex step differentiation depending upon the
            value of `optim_complex_step`, and None uses the built-in gradient
            approximation of the optimizer. Default is None. This keyword is
            only relevant if the optimization method uses the score.
        optim_complex_step : bool, optional
            Whether or not to use complex step differentiation when
            approximating the score; if False, finite difference approximation
            is used. Default is True. This keyword is only relevant if
            `optim_score` is set to 'harvey' or 'approx'.
        optim_hessian : {'opg','oim','approx'}, optional
            The method by which the Hessian is numerically approximated. 'opg'
            uses outer product of gradients, 'oim' uses the information
            matrix formula from Harvey (1989), and 'approx' uses numerical
            approximation. This keyword is only relevant if the
            optimization method uses the Hessian matrix.
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including smoothed results and in-sample
            prediction), although out-of-sample forecasting is possible.
            Note that this option is not available when using the EM algorithm
            (which is the default for this model). Default is False.
        llf_decrease_action : {'ignore', 'warn', 'revert'}, optional
            Action to take if the log-likelihood decreases in an EM iteration.
            'ignore' continues the iterations, 'warn' issues a warning but
            continues the iterations, while 'revert' ends the iterations and
            returns the result from the last good iteration. Default is 'warn'.
        llf_decrease_tolerance : float, optional
            Minimum size of the log-likelihood decrease required to trigger a
            warning or to end the EM iterations. Setting this value slightly
            larger than zero allows small decreases in the log-likelihood that
            may be caused by numerical issues. If set to zero, then any
            decrease will trigger the `llf_decrease_action`. Default is 1e-4.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        MLEResults

        See Also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        statsmodels.tsa.statespace.mlemodel.MLEResults
        """
        if method == 'em':
            return self.fit_em(
                start_params=start_params, transformed=transformed,
                cov_type=cov_type, cov_kwds=cov_kwds, maxiter=maxiter,
                tolerance=tolerance, em_initialization=em_initialization,
                mstep_method=mstep_method, full_output=full_output, disp=disp,
                return_params=return_params, low_memory=low_memory,
                llf_decrease_action=llf_decrease_action,
                llf_decrease_tolerance=llf_decrease_tolerance, **kwargs)
        else:
            return super().fit(
                start_params=start_params, transformed=transformed,
                includes_fixed=includes_fixed, cov_type=cov_type,
                cov_kwds=cov_kwds, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp,
                callback=callback, return_params=return_params,
                optim_score=optim_score,
                optim_complex_step=optim_complex_step,
                optim_hessian=optim_hessian, flags=flags,
                low_memory=low_memory, **kwargs)

    def fit_em(self, start_params=None, transformed=True, cov_type='none',
               cov_kwds=None, maxiter=500, tolerance=1e-6, disp=False,
               em_initialization=True, mstep_method=None, full_output=True,
               return_params=False, low_memory=False,
               llf_decrease_action='revert', llf_decrease_tolerance=1e-4):
        """
        Fits the model by maximum likelihood via the EM algorithm.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is to use `DynamicFactorMQ.start_params`.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presence of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.

            Default is 'none', since computing this matrix can be very slow
            when there are a large number of parameters.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : bool, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : bool, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        maxiter : int, optional
            The maximum number of EM iterations to perform.
        tolerance : float, optional
            Parameter governing convergence of the EM algorithm. The
            `tolerance` is the minimum relative increase in the likelihood
            for which convergence will be declared. A smaller value for the
            `tolerance` will typically yield more precise parameter estimates,
            but will typically require more EM iterations. Default is 1e-6.
        disp : int or bool, optional
            Controls printing of EM iteration progress. If an integer, progress
            is printed at every `disp` iterations. A value of True is
            interpreted as the value of 1. Default is False (nothing will be
            printed).
        em_initialization : bool, optional
            Whether or not to also update the Kalman filter initialization
            using the EM algorithm. Default is True.
        mstep_method : {None, 'missing', 'nonmissing'}, optional
            The EM algorithm maximization step. If there are no NaN values
            in the dataset, this can be set to "nonmissing" (which is slightly
            faster) or "missing", otherwise it must be "missing". Default is
            "nonmissing" if there are no NaN values or "missing" if there are.
        full_output : bool, optional
            Set to True to have all available output from EM iterations in
            the Results object's mle_retvals attribute.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        low_memory : bool, optional
            This option cannot be used with the EM algorithm and will raise an
            error if set to True. Default is False.
        llf_decrease_action : {'ignore', 'warn', 'revert'}, optional
            Action to take if the log-likelihood decreases in an EM iteration.
            'ignore' continues the iterations, 'warn' issues a warning but
            continues the iterations, while 'revert' ends the iterations and
            returns the result from the last good iteration. Default is 'warn'.
        llf_decrease_tolerance : float, optional
            Minimum size of the log-likelihood decrease required to trigger a
            warning or to end the EM iterations. Setting this value slightly
            larger than zero allows small decreases in the log-likelihood that
            may be caused by numerical issues. If set to zero, then any
            decrease will trigger the `llf_decrease_action`. Default is 1e-4.

        Returns
        -------
        DynamicFactorMQResults

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEModel.fit
        statsmodels.tsa.statespace.mlemodel.MLEResults
        """
        if self._has_fixed_params:
            raise NotImplementedError('Cannot fit using the EM algorithm while'
                                      ' holding some parameters fixed.')
        if low_memory:
            raise ValueError('Cannot fit using the EM algorithm when using'
                             ' low_memory option.')

        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)

        if not transformed:
            start_params = self.transform_params(start_params)

        llf_decrease_action = string_like(
            llf_decrease_action, 'llf_decrease_action',
            options=['ignore', 'warn', 'revert'])

        disp = int(disp)

        # Perform expectation-maximization
        s = self._s
        llf = []
        params = [start_params]
        init = None
        inits = [self.ssm.initialization]
        i = 0
        delta = 0
        terminate = False
        # init_stationary = None if em_initialization else True
        while i < maxiter and not terminate and (i < 1 or (delta > tolerance)):
            out = self._em_iteration(params[-1], init=init,
                                     mstep_method=mstep_method)
            new_llf = out[0].llf_obs.sum()

            # If we are not using EM initialization, then we need to check for
            # non-stationary parameters
            if not em_initialization:
                self.update(out[1])
                switch_init = []
                T = self['transition']
                init = self.ssm.initialization
                iloc = np.arange(self.k_states)

                # We may only have global initialization if we have no
                # quarterly variables and idiosyncratic_ar1=False
                if self.k_endog_Q == 0 and not self.idiosyncratic_ar1:
                    block = s.factor_blocks[0]
                    if init.initialization_type == 'stationary':
                        Tb = T[block['factors'], block['factors']]
                        if not np.all(np.linalg.eigvals(Tb) < (1 - 1e-10)):
                            init.set(block['factors'], 'diffuse')
                            switch_init.append(
                                'factor block:'
                                f' {tuple(block.factor_names)}')
                else:
                    # Factor blocks
                    for block in s.factor_blocks:
                        b = tuple(iloc[block['factors']])
                        init_type = init.blocks[b].initialization_type
                        if init_type == 'stationary':
                            Tb = T[block['factors'], block['factors']]
                            if not np.all(np.linalg.eigvals(Tb) < (1 - 1e-10)):
                                init.set(block['factors'], 'diffuse')
                                switch_init.append(
                                    'factor block:'
                                    f' {tuple(block.factor_names)}')

                if self.idiosyncratic_ar1:
                    endog_names = self._get_endog_names(as_string=True)
                    # Monthly variables
                    for j in range(s['idio_ar_M'].start, s['idio_ar_M'].stop):
                        init_type = init.blocks[(j,)].initialization_type
                        if init_type == 'stationary':
                            if not np.abs(T[j, j]) < (1 - 1e-10):
                                init.set(j, 'diffuse')
                                name = endog_names[j - s['idio_ar_M'].start]
                                switch_init.append(
                                    'idiosyncratic AR(1) for monthly'
                                    f' variable: {name}')

                    # Quarterly variables
                    if self.k_endog_Q > 0:
                        b = tuple(iloc[s['idio_ar_Q']])
                        init_type = init.blocks[b].initialization_type
                        if init_type == 'stationary':
                            Tb = T[s['idio_ar_Q'], s['idio_ar_Q']]
                            if not np.all(np.linalg.eigvals(Tb) < (1 - 1e-10)):
                                init.set(s['idio_ar_Q'], 'diffuse')
                                switch_init.append(
                                    'idiosyncratic AR(1) for the'
                                    ' block of quarterly variables')

                if len(switch_init) > 0:
                    warn('Non-stationary parameters found at EM iteration'
                         f' {i + 1}, which is not compatible with'
                         ' stationary initialization. Initialization was'
                         ' switched to diffuse for the following: '
                         f' {switch_init}, and fitting was restarted.')
                    results = self.fit_em(
                        start_params=params[-1], transformed=transformed,
                        cov_type=cov_type, cov_kwds=cov_kwds,
                        maxiter=maxiter, tolerance=tolerance,
                        em_initialization=em_initialization,
                        mstep_method=mstep_method, full_output=full_output,
                        disp=disp, return_params=return_params,
                        low_memory=low_memory,
                        llf_decrease_action=llf_decrease_action,
                        llf_decrease_tolerance=llf_decrease_tolerance)
                    self.ssm.initialize(self._default_initialization())
                    return results

            # Check for decrease in the log-likelihood
            # Note: allow a little numerical error before declaring a decrease
            llf_decrease = (
                i > 0 and (new_llf - llf[-1]) < -llf_decrease_tolerance)

            if llf_decrease_action == 'revert' and llf_decrease:
                warn(f'Log-likelihood decreased at EM iteration {i + 1}.'
                     f' Reverting to the results from EM iteration {i}'
                     ' (prior to the decrease) and returning the solution.')
                # Terminated iteration
                i -= 1
                terminate = True
            else:
                if llf_decrease_action == 'warn' and llf_decrease:
                    warn(f'Log-likelihood decreased at EM iteration {i + 1},'
                         ' which can indicate numerical issues.')
                llf.append(new_llf)
                params.append(out[1])
                if em_initialization:
                    init = initialization.Initialization(
                        self.k_states, 'known',
                        constant=out[0].smoothed_state[..., 0],
                        stationary_cov=out[0].smoothed_state_cov[..., 0])
                    inits.append(init)
                if i > 0:
                    delta = (2 * np.abs(llf[-1] - llf[-2]) /
                             (np.abs(llf[-1]) + np.abs(llf[-2])))
                else:
                    delta = np.inf

                # If `disp` is not False, display the first iteration
                if disp and i == 0:
                    print(f'EM start iterations, llf={llf[-1]:.5g}')
                # Print output every `disp` observations
                elif disp and ((i + 1) % disp) == 0:
                    print(f'EM iteration {i + 1}, llf={llf[-1]:.5g},'
                          f' convergence criterion={delta:.5g}')

            # Advance the iteration counter
            i += 1

        # Check for convergence
        not_converged = (i == maxiter and delta > tolerance)

        # If no convergence without explicit termination, warn users
        if not_converged:
            warn(f'EM reached maximum number of iterations ({maxiter}),'
                 f' without achieving convergence: llf={llf[-1]:.5g},'
                 f' convergence criterion={delta:.5g}'
                 f' (while specified tolerance was {tolerance:.5g})')

        # If `disp` is not False, display the final iteration
        if disp:
            if terminate:
                print(f'EM terminated at iteration {i}, llf={llf[-1]:.5g},'
                      f' convergence criterion={delta:.5g}'
                      f' (while specified tolerance was {tolerance:.5g})')
            elif not_converged:
                print(f'EM reached maximum number of iterations ({maxiter}),'
                      f' without achieving convergence: llf={llf[-1]:.5g},'
                      f' convergence criterion={delta:.5g}'
                      f' (while specified tolerance was {tolerance:.5g})')
            else:
                print(f'EM converged at iteration {i}, llf={llf[-1]:.5g},'
                      f' convergence criterion={delta:.5g}'
                      f' < tolerance={tolerance:.5g}')

        # Just return the fitted parameters if requested
        if return_params:
            result = params[-1]
        # Otherwise construct the results class if desired
        else:
            if em_initialization:
                base_init = self.ssm.initialization
                self.ssm.initialization = init
            # Note that because we are using params[-1], we are actually using
            # the results from one additional iteration compared to the
            # iteration at which we declared convergence.
            result = self.smooth(params[-1], transformed=True,
                                 cov_type=cov_type, cov_kwds=cov_kwds)
            if em_initialization:
                self.ssm.initialization = base_init

            # Save the output
            if full_output:
                llf.append(result.llf)
                em_retvals = Bunch(**{'params': np.array(params),
                                      'llf': np.array(llf),
                                      'iter': i,
                                      'inits': inits})
                em_settings = Bunch(**{'method': 'em',
                                       'tolerance': tolerance,
                                       'maxiter': maxiter})
            else:
                em_retvals = None
                em_settings = None

            result._results.mle_retvals = em_retvals
            result._results.mle_settings = em_settings

        return result

    def _em_iteration(self, params0, init=None, mstep_method=None):
        """EM iteration."""
        # (E)xpectation step
        res = self._em_expectation_step(params0, init=init)

        # (M)aximization step
        params1 = self._em_maximization_step(res, params0,
                                             mstep_method=mstep_method)

        return res, params1

    def _em_expectation_step(self, params0, init=None):
        """EM expectation step."""
        # (E)xpectation step
        self.update(params0)
        # Re-initialize state, if new initialization is given
        if init is not None:
            base_init = self.ssm.initialization
            self.ssm.initialization = init
        # Perform smoothing, only saving what is required
        res = self.ssm.smooth(
            SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_STATE_AUTOCOV,
            update_filter=False)
        res.llf_obs = np.array(
            self.ssm._kalman_filter.loglikelihood, copy=True)
        # Reset initialization
        if init is not None:
            self.ssm.initialization = base_init

        return res

    def _em_maximization_step(self, res, params0, mstep_method=None):
        """EM maximization step."""
        s = self._s

        a = res.smoothed_state.T[..., None]
        cov_a = res.smoothed_state_cov.transpose(2, 0, 1)
        acov_a = res.smoothed_state_autocov.transpose(2, 0, 1)

        # E[a_t a_t'], t = 0, ..., T
        Eaa = cov_a.copy() + np.matmul(a, a.transpose(0, 2, 1))
        # E[a_t a_{t-1}'], t = 1, ..., T
        Eaa1 = acov_a[:-1] + np.matmul(a[1:], a[:-1].transpose(0, 2, 1))

        # Observation equation
        has_missing = np.any(res.nmissing)
        if mstep_method is None:
            mstep_method = 'missing' if has_missing else 'nonmissing'
        mstep_method = mstep_method.lower()
        if mstep_method == 'nonmissing' and has_missing:
            raise ValueError('Cannot use EM algorithm option'
                             ' `mstep_method="nonmissing"` with missing data.')

        if mstep_method == 'nonmissing':
            func = self._em_maximization_obs_nonmissing
        elif mstep_method == 'missing':
            func = self._em_maximization_obs_missing
        else:
            raise ValueError('Invalid maximization step method: "%s".'
                             % mstep_method)
        # TODO: compute H is pretty slow
        Lambda, H = func(res, Eaa, a, compute_H=(not self.idiosyncratic_ar1))

        # Factor VAR and covariance
        factor_ar = []
        factor_cov = []
        for b in s.factor_blocks:
            A = Eaa[:-1, b['factors_ar'], b['factors_ar']].sum(axis=0)
            B = Eaa1[:, b['factors_L1'], b['factors_ar']].sum(axis=0)
            C = Eaa[1:, b['factors_L1'], b['factors_L1']].sum(axis=0)
            nobs = Eaa.shape[0] - 1

            # want: x = B A^{-1}, so solve: x A = B or solve: A' x' = B'
            try:
                f_A = cho_solve(cho_factor(A), B.T).T
            except LinAlgError:
                # Fall back to general solver if there are problems with
                # postive-definiteness
                f_A = np.linalg.solve(A, B.T).T

            f_Q = (C - f_A @ B.T) / nobs
            factor_ar += f_A.ravel().tolist()
            factor_cov += (
                np.linalg.cholesky(f_Q)[np.tril_indices_from(f_Q)].tolist())

        # Idiosyncratic AR(1) and variances
        if self.idiosyncratic_ar1:
            ix = s['idio_ar_L1']

            Ad = Eaa[:-1, ix, ix].sum(axis=0).diagonal()
            Bd = Eaa1[:, ix, ix].sum(axis=0).diagonal()
            Cd = Eaa[1:, ix, ix].sum(axis=0).diagonal()
            nobs = Eaa.shape[0] - 1

            alpha = Bd / Ad
            sigma2 = (Cd - alpha * Bd) / nobs
        else:
            ix = s['idio_ar_L1']
            C = Eaa[:, ix, ix].sum(axis=0)
            sigma2 = np.r_[H.diagonal()[self._o['M']],
                           C.diagonal() / Eaa.shape[0]]

        # Save parameters
        params1 = np.zeros_like(params0)
        loadings = []
        for i in range(self.k_endog):
            iloc = self._s.endog_factor_iloc[i]
            factor_ix = s['factors_L1'][iloc]
            loadings += Lambda[i, factor_ix].tolist()
        params1[self._p['loadings']] = loadings
        params1[self._p['factor_ar']] = factor_ar
        params1[self._p['factor_cov']] = factor_cov
        if self.idiosyncratic_ar1:
            params1[self._p['idiosyncratic_ar1']] = alpha
        params1[self._p['idiosyncratic_var']] = sigma2

        return params1

    def _em_maximization_obs_nonmissing(self, res, Eaa, a, compute_H=False):
        """EM maximization step, observation equation without missing data."""
        s = self._s
        dtype = Eaa.dtype

        # Observation equation (non-missing)
        # Note: we only compute loadings for monthly variables because
        # quarterly variables will always have missing entries, so we would
        # never choose this method in that case
        k = s.k_states_factors
        Lambda = np.zeros((self.k_endog, k), dtype=dtype)
        for i in range(self.k_endog):
            y = self.endog[:, i:i + 1]
            iloc = self._s.endog_factor_iloc[i]
            factor_ix = s['factors_L1'][iloc]

            ix = (np.s_[:],) + np.ix_(factor_ix, factor_ix)
            A = Eaa[ix].sum(axis=0)
            B = y.T @ a[:, factor_ix, 0]
            if self.idiosyncratic_ar1:
                ix1 = s.k_states_factors + i
                ix2 = ix1 + 1
                B -= Eaa[:, ix1:ix2, factor_ix].sum(axis=0)

            # want: x = B A^{-1}, so solve: x A = B or solve: A' x' = B'
            try:
                Lambda[i, factor_ix] = cho_solve(cho_factor(A), B.T).T
            except LinAlgError:
                # Fall back to general solver if there are problems with
                # postive-definiteness
                Lambda[i, factor_ix] = np.linalg.solve(A, B.T).T

        # Compute new obs cov
        # Note: this is unnecessary if `idiosyncratic_ar1=True`.
        # This is written in a slightly more general way than
        # Banbura and Modugno (2014), equation (7); see instead equation (13)
        # of Wu et al. (1996)
        # "An algorithm for estimating parameters of state-space models"
        if compute_H:
            Z = self['design'].copy()
            Z[:, :k] = Lambda
            BL = self.endog.T @ a[..., 0] @ Z.T
            C = self.endog.T @ self.endog

            H = (C + -BL - BL.T + Z @ Eaa.sum(axis=0) @ Z.T) / self.nobs
        else:
            H = np.zeros((self.k_endog, self.k_endog), dtype=dtype) * np.nan

        return Lambda, H

    def _em_maximization_obs_missing(self, res, Eaa, a, compute_H=False):
        """EM maximization step, observation equation with missing data."""
        s = self._s
        dtype = Eaa.dtype

        # Observation equation (missing)
        k = s.k_states_factors
        Lambda = np.zeros((self.k_endog, k), dtype=dtype)

        W = (1 - res.missing.T)
        mask = W.astype(bool)

        # Compute design for monthly
        # Note: the relevant A changes for each i
        for i in range(self.k_endog_M):
            iloc = self._s.endog_factor_iloc[i]
            factor_ix = s['factors_L1'][iloc]

            m = mask[:, i]
            yt = self.endog[m, i:i + 1]

            ix = np.ix_(m, factor_ix, factor_ix)
            Ai = Eaa[ix].sum(axis=0)
            Bi = yt.T @ a[np.ix_(m, factor_ix)][..., 0]
            if self.idiosyncratic_ar1:
                ix1 = s.k_states_factors + i
                ix2 = ix1 + 1
                Bi -= Eaa[m, ix1:ix2][..., factor_ix].sum(axis=0)
            # want: x = B A^{-1}, so solve: x A = B or solve: A' x' = B'
            try:
                Lambda[i, factor_ix] = cho_solve(cho_factor(Ai), Bi.T).T
            except LinAlgError:
                # Fall back to general solver if there are problems with
                # postive-definiteness
                Lambda[i, factor_ix] = np.linalg.solve(Ai, Bi.T).T

        # Compute unrestricted design for quarterly
        # See Banbura at al. (2011), where this is described in Appendix C,
        # between equations (13) and (14).
        if self.k_endog_Q > 0:
            # Note: the relevant A changes for each i
            multipliers = np.array([1, 2, 3, 2, 1])[:, None]
            for i in range(self.k_endog_M, self.k_endog):
                iloc = self._s.endog_factor_iloc[i]
                factor_ix = s['factors_L1_5_ix'][:, iloc].ravel().tolist()

                R, _ = self.loading_constraints(i)

                iQ = i - self.k_endog_M
                m = mask[:, i]
                yt = self.endog[m, i:i + 1]
                ix = np.ix_(m, factor_ix, factor_ix)
                Ai = Eaa[ix].sum(axis=0)
                BiQ = yt.T @ a[np.ix_(m, factor_ix)][..., 0]
                if self.idiosyncratic_ar1:
                    ix = (np.s_[:],) + np.ix_(s['idio_ar_Q_ix'][iQ], factor_ix)
                    Eepsf = Eaa[ix]
                    BiQ -= (multipliers * Eepsf[m].sum(axis=0)).sum(axis=0)

                # Note that there was a typo in Banbura et al. (2011) for
                # the formula applying the restrictions. In their notation,
                # they show (C D C')^{-1} while it should be (C D^{-1} C')^{-1}
                # Note: in reality, this is:
                # unrestricted - Aii @ R.T @ RARi @ (R @ unrestricted - q)
                # where the restrictions are defined as: R @ unrestricted = q
                # However, here q = 0, so we can simplify.
                try:
                    L_and_lower = cho_factor(Ai)
                    # x = BQ A^{-1}, or x A = BQ, so solve A' x' = (BQ)'
                    unrestricted = cho_solve(L_and_lower, BiQ.T).T[0]
                    AiiRT = cho_solve(L_and_lower, R.T)

                    L_and_lower = cho_factor(R @ AiiRT)
                    RAiiRTiR = cho_solve(L_and_lower, R)
                    restricted = unrestricted - AiiRT @ RAiiRTiR @ unrestricted
                except LinAlgError:
                    # Fall back to slower method if there are problems with
                    # postive-definiteness
                    Aii = np.linalg.inv(Ai)
                    unrestricted = (BiQ @ Aii)[0]
                    RARi = np.linalg.inv(R @ Aii @ R.T)
                    restricted = (unrestricted -
                                  Aii @ R.T @ RARi @ R @ unrestricted)
                Lambda[i, factor_ix] = restricted

        # Compute new obs cov
        # Note: this is unnecessary if `idiosyncratic_ar1=True`.
        # See Banbura and Modugno (2014), equation (12)
        # This does not literally follow their formula, e.g. multiplying by the
        # W_t selection matrices, because those formulas require loops that are
        # relatively slow. The formulation here is vectorized.
        if compute_H:
            Z = self['design'].copy()
            Z[:, :Lambda.shape[1]] = Lambda

            y = np.nan_to_num(self.endog)
            C = y.T @ y
            W = W[..., None]
            IW = 1 - W

            WL = W * Z
            WLT = WL.transpose(0, 2, 1)
            BL = y[..., None] @ a.transpose(0, 2, 1) @ WLT
            A = Eaa

            BLT = BL.transpose(0, 2, 1)
            IWT = IW.transpose(0, 2, 1)

            H = (C + (-BL - BLT + WL @ A @ WLT +
                      IW * self['obs_cov'] * IWT).sum(axis=0)) / self.nobs
        else:
            H = np.zeros((self.k_endog, self.k_endog), dtype=dtype) * np.nan

        return Lambda, H

    def smooth(self, params, transformed=True, includes_fixed=False,
               complex_step=False, cov_type='none', cov_kwds=None,
               return_ssm=False, results_class=None,
               results_wrapper_class=None, **kwargs):
        """
        Kalman smoothing.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : bool,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object. Default is None.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        return super().smooth(
            params, transformed=transformed, includes_fixed=includes_fixed,
            complex_step=complex_step, cov_type=cov_type, cov_kwds=cov_kwds,
            return_ssm=return_ssm, results_class=results_class,
            results_wrapper_class=results_wrapper_class, **kwargs)

    def filter(self, params, transformed=True, includes_fixed=False,
               complex_step=False, cov_type='none', cov_kwds=None,
               return_ssm=False, results_class=None,
               results_wrapper_class=None, low_memory=False, **kwargs):
        """
        Kalman filtering.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : bool,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object. Default is 'none'.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including in-sample prediction), although
            out-of-sample forecasting is possible. Default is False.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        return super().filter(
            params, transformed=transformed, includes_fixed=includes_fixed,
            complex_step=complex_step, cov_type=cov_type, cov_kwds=cov_kwds,
            return_ssm=return_ssm, results_class=results_class,
            results_wrapper_class=results_wrapper_class, **kwargs)

    def simulate(self, params, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None, anchor=None,
                 repetitions=None, exog=None, extend_model=None,
                 extend_kwargs=None, transformed=True, includes_fixed=False,
                 original_scale=True, **kwargs):
        r"""
        Simulate a new time series following the state space model.

        Parameters
        ----------
        params : array_like
            Array of parameters to use in constructing the state space
            representation to use when simulating.
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number of observations.
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the initial state vector to use in
            simulation, which should be shaped (`k_states` x 1), where
            `k_states` is the same as in the state space model. If unspecified,
            but the model has been initialized, then that initialization is
            used. This must be specified if `anchor` is anything other than
            "start" or 0 (or else you can use the `simulate` method on a
            results object rather than on the model object).
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample, and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        original_scale : bool, optional
            If the model specification standardized the data, whether or not
            to return simulations in the original scale of the data (i.e.
            before it was standardized by the model). Default is True.

        Returns
        -------
        simulated_obs : ndarray
            An array of simulated observations. If `repetitions=None`, then it
            will be shaped (nsimulations x k_endog) or (nsimulations,) if
            `k_endog=1`. Otherwise it will be shaped
            (nsimulations x k_endog x repetitions). If the model was given
            Pandas input then the output will be a Pandas object. If
            `k_endog > 1` and `repetitions` is not None, then the output will
            be a Pandas DataFrame that has a MultiIndex for the columns, with
            the first level containing the names of the `endog` variables and
            the second level containing the repetition number.
        """
        # Get usual simulations (in the possibly-standardized scale)
        sim = super().simulate(
            params, nsimulations, measurement_shocks=measurement_shocks,
            state_shocks=state_shocks, initial_state=initial_state,
            anchor=anchor, repetitions=repetitions, exog=exog,
            extend_model=extend_model, extend_kwargs=extend_kwargs,
            transformed=transformed, includes_fixed=includes_fixed, **kwargs)

        # If applicable, convert predictions back to original space
        if self.standardize and original_scale:
            use_pandas = isinstance(self.data, PandasData)
            shape = sim.shape

            if use_pandas:
                # pd.Series (k_endog=1, replications=None)
                if len(shape) == 1:
                    sim = sim * self._endog_std[0] + self._endog_mean[0]
                # pd.DataFrame (k_endog > 1, replications=None)
                # [or]
                # pd.DataFrame with MultiIndex (replications > 0)
                elif len(shape) == 2:
                    sim = (sim.multiply(self._endog_std, axis=1, level=0)
                              .add(self._endog_mean, axis=1, level=0))
            else:
                # 1-dim array (k_endog=1, replications=None)
                if len(shape) == 1:
                    sim = sim * self._endog_std + self._endog_mean
                # 2-dim array (k_endog > 1, replications=None)
                elif len(shape) == 2:
                    sim = sim * self._endog_std + self._endog_mean
                # 3-dim array with MultiIndex (replications > 0)
                else:
                    # Get arrays into the form that can be used for
                    # broadcasting
                    std = np.atleast_2d(self._endog_std)[..., None]
                    mean = np.atleast_2d(self._endog_mean)[..., None]
                    sim = sim * std + mean

        return sim

    def impulse_responses(self, params, steps=1, impulse=0,
                          orthogonalized=False, cumulative=False, anchor=None,
                          exog=None, extend_model=None, extend_kwargs=None,
                          transformed=True, includes_fixed=False,
                          original_scale=True, **kwargs):
        """
        Impulse response function.

        Parameters
        ----------
        params : array_like
            Array of model parameters.
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 1. Note that for time-invariant models, the initial
            impulse is not counted as a step, so if `steps=1`, the output will
            have 2 entries.
        impulse : int or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1`. Alternatively, a custom impulse vector may be
            provided; must be shaped `k_posdef x 1`.
        orthogonalized : bool, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : bool, optional
            Whether or not to return cumulative impulse responses. Default is
            False.
        anchor : int, str, or datetime, optional
            Time point within the sample for the state innovation impulse. Type
            depends on the index of the given `endog` in the model. Two special
            cases are the strings 'start' and 'end', which refer to setting the
            impulse at the first and last points of the sample, respectively.
            Integer values can run from 0 to `nobs - 1`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        exog : array_like, optional
            New observations of exogenous regressors for our-of-sample periods,
            if applicable.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        original_scale : bool, optional
            If the model specification standardized the data, whether or not
            to return impulse responses in the original scale of the data (i.e.
            before it was standardized by the model). Default is True.
        **kwargs
            If the model has time-varying design or transition matrices and the
            combination of `anchor` and `steps` implies creating impulse
            responses for the out-of-sample period, then these matrices must
            have updated values provided for the out-of-sample steps. For
            example, if `design` is a time-varying component, `nobs` is 10,
            `anchor=1`, and `steps` is 15, a (`k_endog` x `k_states` x 7)
            matrix must be provided with the new design matrix values.

        Returns
        -------
        impulse_responses : ndarray
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. For a time-invariant model, the
            impulse responses are given for `steps + 1` elements (this gives
            the "initial impulse" followed by `steps` responses for the
            important cases of VAR and SARIMAX models), while for time-varying
            models the impulse responses are only given for `steps` elements
            (to avoid having to unexpectedly provide updated time-varying
            matrices).

        """
        # Get usual simulations (in the possibly-standardized scale)
        irfs = super().impulse_responses(
            params, steps=steps, impulse=impulse,
            orthogonalized=orthogonalized, cumulative=cumulative,
            anchor=anchor, exog=exog, extend_model=extend_model,
            extend_kwargs=extend_kwargs, transformed=transformed,
            includes_fixed=includes_fixed, **kwargs)

        # If applicable, convert predictions back to original space
        if self.standardize and original_scale:
            use_pandas = isinstance(self.data, PandasData)
            shape = irfs.shape

            if use_pandas:
                # pd.Series (k_endog=1, replications=None)
                if len(shape) == 1:
                    irfs = irfs * self._endog_std[0]
                # pd.DataFrame (k_endog > 1)
                # [or]
                # pd.DataFrame with MultiIndex (replications > 0)
                elif len(shape) == 2:
                    irfs = irfs.multiply(self._endog_std, axis=1, level=0)
            else:
                # 1-dim array (k_endog=1)
                if len(shape) == 1:
                    irfs = irfs * self._endog_std
                # 2-dim array (k_endog > 1)
                elif len(shape) == 2:
                    irfs = irfs * self._endog_std

        return irfs


class DynamicFactorMQResults(mlemodel.MLEResults):
    """
    Results from fitting a dynamic factor model
    """
    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super(DynamicFactorMQResults, self).__init__(
            model, params, filter_results, cov_type, **kwargs)

    @property
    def factors(self):
        """
        Estimates of unobserved factors.

        Returns
        -------
        out : Bunch
            Has the following attributes shown in Notes.

        Notes
        -----
        The output is a bunch of the following format:

        - `filtered`: a time series array with the filtered estimate of
          the component
        - `filtered_cov`: a time series array with the filtered estimate of
          the variance/covariance of the component
        - `smoothed`: a time series array with the smoothed estimate of
          the component
        - `smoothed_cov`: a time series array with the smoothed estimate of
          the variance/covariance of the component
        - `offset`: an integer giving the offset in the state vector where
          this component begins
        """
        out = None
        if self.model.k_factors > 0:
            iloc = self.model._s.factors_L1
            ix = np.array(self.model.state_names)[iloc].tolist()
            out = Bunch(
                filtered=self.states.filtered.loc[:, ix],
                filtered_cov=self.states.filtered_cov.loc[np.s_[ix, :], ix],
                smoothed=None, smoothed_cov=None)
            if self.smoothed_state is not None:
                out.smoothed = self.states.smoothed.loc[:, ix]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = (
                    self.states.smoothed_cov.loc[np.s_[ix, :], ix])
        return out

    def get_coefficients_of_determination(self, method='individual',
                                          which=None):
        """
        Get coefficients of determination (R-squared) for variables / factors.

        Parameters
        ----------
        method : {'individual', 'joint', 'cumulative'}, optional
            The type of R-squared values to generate. "individual" plots
            the R-squared of each variable on each factor; "joint" plots the
            R-squared of each variable on each factor that it loads on;
            "cumulative" plots the successive R-squared values as each
            additional factor is added to the regression, for each variable.
            Default is 'individual'.
        which: {None, 'filtered', 'smoothed'}, optional
            Whether to compute R-squared values based on filtered or smoothed
            estimates of the factors. Default is 'smoothed' if smoothed results
            are available and 'filtered' otherwise.

        Returns
        -------
        rsquared : pd.DataFrame or pd.Series
            The R-squared values from regressions of observed variables on
            one or more of the factors. If method='individual' or
            method='cumulative', this will be a Pandas DataFrame with observed
            variables as the index and factors as the columns . If
            method='joint', will be a Pandas Series with observed variables as
            the index.

        See Also
        --------
        plot_coefficients_of_determination
        coefficients_of_determination
        """
        from statsmodels.tools import add_constant

        method = string_like(method, 'method', options=['individual', 'joint',
                                                        'cumulative'])
        if which is None:
            which = 'filtered' if self.smoothed_state is None else 'smoothed'

        k_endog = self.model.k_endog
        k_factors = self.model.k_factors
        ef_map = self.model._s.endog_factor_map
        endog_names = self.model.endog_names
        factor_names = self.model.factor_names

        if method == 'individual':
            coefficients = np.zeros((k_endog, k_factors))

            for i in range(k_factors):
                exog = add_constant(self.factors[which].iloc[:, i])
                for j in range(k_endog):
                    if ef_map.iloc[j, i]:
                        endog = self.filter_results.endog[j]
                        coefficients[j, i] = (
                            OLS(endog, exog, missing='drop').fit().rsquared)
                    else:
                        coefficients[j, i] = np.nan

            coefficients = pd.DataFrame(coefficients, index=endog_names,
                                        columns=factor_names)
        elif method == 'joint':
            coefficients = np.zeros((k_endog,))
            exog = add_constant(self.factors[which])
            for j in range(k_endog):
                endog = self.filter_results.endog[j]
                ix = np.r_[True, ef_map.iloc[j]].tolist()
                X = exog.loc[:, ix]
                coefficients[j] = (
                    OLS(endog, X, missing='drop').fit().rsquared)
            coefficients = pd.Series(coefficients, index=endog_names)
        elif method == 'cumulative':
            coefficients = np.zeros((k_endog, k_factors))
            exog = add_constant(self.factors[which])
            for j in range(k_endog):
                endog = self.filter_results.endog[j]

                for i in range(k_factors):
                    if self.model._s.endog_factor_map.iloc[j, i]:
                        ix = np.r_[True, ef_map.iloc[j, :i + 1],
                                   [False] * (k_factors - i - 1)]
                        X = exog.loc[:, ix.astype(bool).tolist()]
                        coefficients[j, i] = (
                            OLS(endog, X, missing='drop').fit().rsquared)
                    else:
                        coefficients[j, i] = np.nan
            coefficients = pd.DataFrame(coefficients, index=endog_names,
                                        columns=factor_names)

        return coefficients

    @cache_readonly
    def coefficients_of_determination(self):
        """
        Individual coefficients of determination (:math:`R^2`).

        Coefficients of determination (:math:`R^2`) from regressions of
        endogenous variables on individual estimated factors.

        Returns
        -------
        coefficients_of_determination : ndarray
            A `k_endog` x `k_factors` array, where
            `coefficients_of_determination[i, j]` represents the :math:`R^2`
            value from a regression of factor `j` and a constant on endogenous
            variable `i`.

        Notes
        -----
        Although it can be difficult to interpret the estimated factor loadings
        and factors, it is often helpful to use the coefficients of
        determination from univariate regressions to assess the importance of
        each factor in explaining the variation in each endogenous variable.

        In models with many variables and factors, this can sometimes lend
        interpretation to the factors (for example sometimes one factor will
        load primarily on real variables and another on nominal variables).

        See Also
        --------
        get_coefficients_of_determination
        plot_coefficients_of_determination
        """
        return self.get_coefficients_of_determination(method='individual')

    def plot_coefficients_of_determination(self, method='individual',
                                           which=None, endog_labels=None,
                                           fig=None, figsize=None):
        """
        Plot coefficients of determination (R-squared) for variables / factors.

        Parameters
        ----------
        method : {'individual', 'joint', 'cumulative'}, optional
            The type of R-squared values to generate. "individual" plots
            the R-squared of each variable on each factor; "joint" plots the
            R-squared of each variable on each factor that it loads on;
            "cumulative" plots the successive R-squared values as each
            additional factor is added to the regression, for each variable.
            Default is 'individual'.
        which: {None, 'filtered', 'smoothed'}, optional
            Whether to compute R-squared values based on filtered or smoothed
            estimates of the factors. Default is 'smoothed' if smoothed results
            are available and 'filtered' otherwise.
        endog_labels : bool, optional
            Whether or not to label the endogenous variables along the x-axis
            of the plots. Default is to include labels if there are 5 or fewer
            endogenous variables.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        The endogenous variables are arranged along the x-axis according to
        their position in the model's `endog` array.

        See Also
        --------
        get_coefficients_of_determination
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)

        method = string_like(method, 'method', options=['individual', 'joint',
                                                        'cumulative'])

        # Should we label endogenous variables?
        if endog_labels is None:
            endog_labels = self.model.k_endog <= 5

        # Plot the coefficients of determination
        rsquared = self.get_coefficients_of_determination(method=method,
                                                          which=which)

        if method in ['individual', 'cumulative']:
            plot_idx = 1
            for factor_name, coeffs in rsquared.T.iterrows():
                # Create the new axis
                ax = fig.add_subplot(self.model.k_factors, 1, plot_idx)
                ax.set_ylim((0, 1))
                ax.set(title=f'{factor_name}', ylabel=r'$R^2$')

                coeffs.plot(ax=ax, kind='bar')
                if plot_idx < len(rsquared.columns) or not endog_labels:
                    ax.xaxis.set_ticklabels([])

                plot_idx += 1
        elif method == 'joint':
            ax = fig.add_subplot(1, 1, 1)
            ax.set_ylim((0, 1))
            ax.set(title=r'$R^2$ - regression on all loaded factors',
                   ylabel=r'$R^2$')
            rsquared.plot(ax=ax, kind='bar')
            if not endog_labels:
                ax.xaxis.set_ticklabels([])

        return fig

    def get_prediction(self, start=None, end=None, dynamic=False,
                       information_set='predicted', signal_only=False,
                       original_scale=True, index=None, exog=None,
                       extend_model=None, extend_kwargs=None, **kwargs):
        r"""
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        information_set : str, optional
            The information set to condition each prediction on. Default is
            "predicted", which computes predictions of period t values
            conditional on observed data through period t-1; these are
            one-step-ahead predictions, and correspond with the typical
            `fittedvalues` results attribute. Alternatives are "filtered",
            which computes predictions of period t values conditional on
            observed data through period t, and "smoothed", which computes
            predictions of period t values conditional on the entire dataset
            (including also future observations t+1, t+2, ...).
        signal_only : bool, optional
            Whether to compute forecasts of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \alpha_t + \varepsilon_t`, and the "signal"
            component is then :math:`Z \alpha_t`. If this argument is set to
            True, then forecasts of the "signal" :math:`Z \alpha_t` will be
            returned. Otherwise, the default is for forecasts of :math:`y_t`
            to be returned.
        original_scale : bool, optional
            If the model specification standardized the data, whether or not
            to return predictions in the original scale of the data (i.e.
            before it was standardized by the model). Default is True.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : ndarray
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        # Get usual predictions (in the possibly-standardized scale)
        res = super().get_prediction(start=start, end=end, dynamic=dynamic,
                                     information_set=information_set,
                                     signal_only=signal_only,
                                     index=index, exog=exog,
                                     extend_model=extend_model,
                                     extend_kwargs=extend_kwargs, **kwargs)

        # If applicable, convert predictions back to original space
        if self.model.standardize and original_scale:
            prediction_results = res.prediction_results
            k_endog, _ = prediction_results.endog.shape

            mean = np.array(self.model._endog_mean)
            std = np.array(self.model._endog_std)

            if self.model.k_endog > 1:
                mean = mean[None, :]
                std = std[None, :]

            res._results._predicted_mean = (
                res._results._predicted_mean * std + mean)

            if k_endog == 1:
                res._results._var_pred_mean *= std**2
            else:
                res._results._var_pred_mean = (
                    std * res._results._var_pred_mean * std.T)

        return res

    def news(self, comparison, impact_date=None, impacted_variable=None,
             start=None, end=None, periods=None, exog=None,
             comparison_type=None, state_index=None, return_raw=False,
             tolerance=1e-10, endog_quarterly=None, original_scale=True,
             **kwargs):
        """
        Compute impacts from updated data (news and revisions).

        Parameters
        ----------
        comparison : array_like or MLEResults
            An updated dataset with updated and/or revised data from which the
            news can be computed, or an updated or previous results object
            to use in computing the news.
        impact_date : int, str, or datetime, optional
            A single specific period of impacts from news and revisions to
            compute. Can also be a date string to parse or a datetime type.
            This argument cannot be used in combination with `start`, `end`, or
            `periods`. Default is the first out-of-sample observation.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying that only
            specific impacted variables should be shown in the News output. The
            impacted variable(s) describe the variables that were *affected* by
            the news. If you do not know the labels for the variables, check
            the `endog_names` attribute of the model instance.
        start : int, str, or datetime, optional
            The first period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        end : int, str, or datetime, optional
            The last period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        periods : int, optional
            The number of periods of impacts from news and revisions to
            compute.
        exog : array_like, optional
            Array of exogenous regressors for the out-of-sample period, if
            applicable.
        comparison_type : {None, 'previous', 'updated'}
            This denotes whether the `comparison` argument represents a
            *previous* results object or dataset or an *updated* results object
            or dataset. If not specified, then an attempt is made to determine
            the comparison type.
        state_index : array_like or "common", optional
            An optional index specifying a subset of states to use when
            constructing the impacts of revisions and news. For example, if
            `state_index=[0, 1]` is passed, then only the impacts to the
            observed variables arising from the impacts to the first two
            states will be returned. If the string "common" is passed and the
            model includes idiosyncratic AR(1) components, news will only be
            computed based on the common states. Default is to use all states.
        return_raw : bool, optional
            Whether or not to return only the specific output or a full
            results object. Default is to return a full results object.
        tolerance : float, optional
            The numerical threshold for determining zero impact. Default is
            that any impact less than 1e-10 is assumed to be zero.
        endog_quarterly : array_like, optional
            New observations of quarterly variables, if `comparison` was
            provided as an updated monthly dataset. If this argument is
            provided, it must be a Pandas Series or DataFrame with a
            DatetimeIndex or PeriodIndex at the quarterly frequency.

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
        if state_index == 'common':
            state_index = (
                np.arange(self.model.k_states - self.model.k_endog))

        news_results = super().news(
            comparison, impact_date=impact_date,
            impacted_variable=impacted_variable, start=start, end=end,
            periods=periods, exog=exog, comparison_type=comparison_type,
            state_index=state_index, return_raw=return_raw,
            tolerance=tolerance, endog_quarterly=endog_quarterly, **kwargs)

        # If we have standardized the data, we may want to report the news in
        # the original scale. If so, we need to modify the data to "undo" the
        # standardization.
        if not return_raw and self.model.standardize and original_scale:
            endog_mean = self.model._endog_mean
            endog_std = self.model._endog_std

            # Don't need to add in the mean for the impacts, since they are
            # the difference of two forecasts
            news_results.total_impacts = (
                news_results.total_impacts * endog_std)
            news_results.update_impacts = (
                news_results.update_impacts * endog_std)
            if news_results.revision_impacts is not None:
                news_results.revision_impacts = (
                    news_results.revision_impacts * endog_std)

            # Update forecasts
            for name in ['prev_impacted_forecasts', 'news', 'revisions',
                         'update_realized', 'update_forecasts',
                         'revised', 'revised_prev', 'post_impacted_forecasts']:
                dta = getattr(news_results, name)

                # for pd.Series, dta.multiply(...) and (sometimes) dta.add(...)
                # remove the name attribute; save it now so that we can add it
                # back in
                orig_name = None
                if hasattr(dta, 'name'):
                    orig_name = dta.name

                dta = dta.multiply(endog_std, level=1)

                if name not in ['news', 'revisions']:
                    dta = dta.add(endog_mean, level=1)

                # add back in the name attribute if it was removed
                if orig_name is not None:
                    dta.name = orig_name

                setattr(news_results, name, dta)

            # For the weights: rows correspond to update (date, variable) and
            # columns correspond to the impacted variable.
            # 1. Because we have modified the updates (realized, forecasts, and
            #    forecast errors) to be in the scale of the original updated
            #    variable, we need to essentially reverse that change for each
            #    row of the weights by dividing by the standard deviation of
            #    that row's updated variable
            # 2. Because we want the impacts to be in the scale of the original
            #    impacted variable, we need to multiply each column by the
            #    standard deviation of that column's impacted variable
            news_results.weights = (
                news_results.weights.divide(endog_std, axis=0, level=1)
                                    .multiply(endog_std, axis=1, level=1))
            news_results.revision_weights = (
                news_results.revision_weights
                            .divide(endog_std, axis=0, level=1)
                            .multiply(endog_std, axis=1, level=1))

        return news_results

    def get_smoothed_decomposition(self, decomposition_of='smoothed_state',
                                   state_index=None, original_scale=True):
        r"""
        Decompose smoothed output into contributions from observations

        Parameters
        ----------
        decomposition_of : {"smoothed_state", "smoothed_signal"}
            The object to perform a decomposition of. If it is set to
            "smoothed_state", then the elements of the smoothed state vector
            are decomposed into the contributions of each observation. If it
            is set to "smoothed_signal", then the predictions of the
            observation vector based on the smoothed state vector are
            decomposed. Default is "smoothed_state".
        state_index : array_like, optional
            An optional index specifying a subset of states to use when
            constructing the decomposition of the "smoothed_signal". For
            example, if `state_index=[0, 1]` is passed, then only the
            contributions of observed variables to the smoothed signal arising
            from the first two states will be returned. Note that if not all
            states are used, the contributions will not sum to the smoothed
            signal. Default is to use all states.
        original_scale : bool, optional
            If the model specification standardized the data, whether or not
            to return simulations in the original scale of the data (i.e.
            before it was standardized by the model). Default is True.

        Returns
        -------
        data_contributions : pd.DataFrame
            Contributions of observations to the decomposed object. If the
            smoothed state is being decomposed, then `data_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `variable_from x date_from`. If the
            smoothed signal is being decomposed, then `data_contributions` is
            shaped `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `variable_from x date_from`.
        obs_intercept_contributions : pd.DataFrame
            Contributions of the observation intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `obs_intercept_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `obs_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `obs_intercept_contributions` is shaped
            `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `obs_intercept_from x date_from`.
        state_intercept_contributions : pd.DataFrame
            Contributions of the state intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `state_intercept_contributions` is
            shaped `(k_states x nobs, k_states x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `state_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `state_intercept_contributions` is shaped
            `(k_endog x nobs, k_states x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `state_intercept_from x date_from`.
        prior_contributions : pd.DataFrame
            Contributions of the prior to the decomposed object. If the
            smoothed state is being decomposed, then `prior_contributions` is
            shaped `(nobs x k_states, k_states)`, with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and columns
            corresponding to elements of the prior mean (aka "initial state").
            If the smoothed signal is being decomposed, then
            `prior_contributions` is shaped `(nobs x k_endog, k_states)`,
            with a `pd.MultiIndex` index corresponding to
            `variable_to x date_to` and columns corresponding to elements of
            the prior mean.

        Notes
        -----
        Denote the smoothed state at time :math:`t` by :math:`\alpha_t`. Then
        the smoothed signal is :math:`Z_t \alpha_t`, where :math:`Z_t` is the
        design matrix operative at time :math:`t`.
        """
        # De-meaning the data is like putting the mean into the observation
        # intercept. To compute the decomposition correctly in the original
        # scale, we need to account for this, so we fill in the observation
        # intercept temporarily
        if self.model.standardize and original_scale:
            cache_obs_intercept = self.model['obs_intercept']
            self.model['obs_intercept'] = self.model._endog_mean

        # Compute the contributions
        (data_contributions, obs_intercept_contributions,
         state_intercept_contributions, prior_contributions) = (
            super().get_smoothed_decomposition(
                decomposition_of=decomposition_of, state_index=state_index))

        # Replace the original observation intercept
        if self.model.standardize and original_scale:
            self.model['obs_intercept'] = cache_obs_intercept

        # Reverse the effect of dividing by the standard deviation
        if (decomposition_of == 'smoothed_signal'
                and self.model.standardize and original_scale):
            endog_std = self.model._endog_std

            data_contributions = (
                data_contributions.multiply(endog_std, axis=0, level=0))
            obs_intercept_contributions = (
                obs_intercept_contributions.multiply(
                    endog_std, axis=0, level=0))
            state_intercept_contributions = (
                state_intercept_contributions.multiply(
                    endog_std, axis=0, level=0))
            prior_contributions = (
                prior_contributions.multiply(endog_std, axis=0, level=0))

        return (data_contributions, obs_intercept_contributions,
                state_intercept_contributions, prior_contributions)

    def append(self, endog, endog_quarterly=None, refit=False, fit_kwargs=None,
               copy_initialization=True, retain_standardization=True,
               **kwargs):
        """
        Recreate the results object with new data appended to original data.

        Creates a new result object applied to a dataset that is created by
        appending new data to the end of the model's original data. The new
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        endog_quarterly : array_like, optional
            New observations of quarterly variables. If provided, must be a
            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at
            the quarterly frequency.
        refit : bool, optional
            Whether to re-fit the parameters, based on the combined dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /
            `smooth`.
        copy_initialization : bool, optional
            Whether or not to copy the initialization from the current results
            set to the new model. Default is True.
        retain_standardization : bool, optional
            Whether or not to use the mean and standard deviations that were
            used to standardize the data in the current model in the new model.
            Default is True.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results from both the
            original dataset and the new dataset.

        Notes
        -----
        The `endog` and `exog` arguments to this method must be formatted in
        the same way (e.g. Pandas Series versus Numpy array) as were the
        `endog` and `exog` arrays passed to the original model.

        The `endog` (and, if applicable, `endog_quarterly`) arguments to this
        method should consist of new observations that occurred directly after
        the last element of `endog`. For any other kind of dataset, see the
        `apply` method.

        This method will apply filtering to all of the original data as well
        as to the new data. To apply filtering only to the new data (which
        can be much faster if the original dataset is large), see the `extend`
        method.

        See Also
        --------
        extend
        apply
        """
        # Construct the combined dataset, if necessary
        endog, k_endog_monthly = DynamicFactorMQ.construct_endog(
            endog, endog_quarterly)

        # Check for compatible dimensions
        k_endog = endog.shape[1] if len(endog.shape) == 2 else 1
        if (k_endog_monthly != self.model.k_endog_M or
                k_endog != self.model.k_endog):
            raise ValueError('Cannot append data of a different dimension to'
                             ' a model.')

        kwargs['k_endog_monthly'] = k_endog_monthly

        return super().append(
            endog, refit=refit, fit_kwargs=fit_kwargs,
            copy_initialization=copy_initialization,
            retain_standardization=retain_standardization, **kwargs)

    def extend(self, endog, endog_quarterly=None, fit_kwargs=None,
               retain_standardization=True, **kwargs):
        """
        Recreate the results object for new data that extends original data.

        Creates a new result object applied to a new dataset that is assumed to
        follow directly from the end of the model's original data. The new
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        endog_quarterly : array_like, optional
            New observations of quarterly variables. If provided, must be a
            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at
            the quarterly frequency.
        fit_kwargs : dict, optional
            Keyword arguments to pass to `filter` or `smooth`.
        retain_standardization : bool, optional
            Whether or not to use the mean and standard deviations that were
            used to standardize the data in the current model in the new model.
            Default is True.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results only for the new
            dataset.

        See Also
        --------
        append
        apply

        Notes
        -----
        The `endog` argument to this method should consist of new observations
        that occurred directly after the last element of the model's original
        `endog` array. For any other kind of dataset, see the `apply` method.

        This method will apply filtering only to the new data provided by the
        `endog` argument, which can be much faster than re-filtering the entire
        dataset. However, the returned results object will only have results
        for the new data. To retrieve results for both the new data and the
        original data, see the `append` method.
        """
        # Construct the combined dataset, if necessary
        endog, k_endog_monthly = DynamicFactorMQ.construct_endog(
            endog, endog_quarterly)

        # Check for compatible dimensions
        k_endog = endog.shape[1] if len(endog.shape) == 2 else 1
        if (k_endog_monthly != self.model.k_endog_M or
                k_endog != self.model.k_endog):
            raise ValueError('Cannot append data of a different dimension to'
                             ' a model.')

        kwargs['k_endog_monthly'] = k_endog_monthly
        return super().extend(
            endog, fit_kwargs=fit_kwargs,
            retain_standardization=retain_standardization, **kwargs)

    def apply(self, endog, k_endog_monthly=None, endog_quarterly=None,
              refit=False, fit_kwargs=None, copy_initialization=False,
              retain_standardization=True, **kwargs):
        """
        Apply the fitted parameters to new data unrelated to the original data.

        Creates a new result object using the current fitted parameters,
        applied to a completely new dataset that is assumed to be unrelated to
        the model's original data. The new results can then be used for
        analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        k_endog_monthly : int, optional
            If specifying a monthly/quarterly mixed frequency model in which
            the provided `endog` dataset contains both the monthly and
            quarterly data, this variable should be used to indicate how many
            of the variables are monthly.
        endog_quarterly : array_like, optional
            New observations of quarterly variables. If provided, must be a
            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at
            the quarterly frequency.
        refit : bool, optional
            Whether to re-fit the parameters, using the new dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /
            `smooth`.
        copy_initialization : bool, optional
            Whether or not to copy the initialization from the current results
            set to the new model. Default is False.
        retain_standardization : bool, optional
            Whether or not to use the mean and standard deviations that were
            used to standardize the data in the current model in the new model.
            Default is True.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results only for the new
            dataset.

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEResults.append
        statsmodels.tsa.statespace.mlemodel.MLEResults.apply

        Notes
        -----
        The `endog` argument to this method should consist of new observations
        that are not necessarily related to the original model's `endog`
        dataset. For observations that continue that original dataset by follow
        directly after its last element, see the `append` and `extend` methods.
        """
        mod = self.model.clone(endog, k_endog_monthly=k_endog_monthly,
                               endog_quarterly=endog_quarterly,
                               retain_standardization=retain_standardization,
                               **kwargs)
        if copy_initialization:
            init = initialization.Initialization.from_results(
                self.filter_results)
            mod.ssm.initialization = init

        res = self._apply(mod, refit=refit, fit_kwargs=fit_kwargs)

        return res

    def summary(self, alpha=.05, start=None, title=None, model_name=None,
                display_params=True, display_diagnostics=False,
                display_params_as_list=False, truncate_endog_names=None,
                display_max_endog=3):
        """
        Summarize the Model.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        title : str, optional
            The title used for the summary table.
        model_name : str, optional
            The name of the model used. Default is to use model class name.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        mod = self.model

        # Default title / model name
        if title is None:
            title = 'Dynamic Factor Results'
        if model_name is None:
            model_name = self.model._model_name

        # Get endog names
        endog_names = self.model._get_endog_names(
            truncate=truncate_endog_names)

        # Get extra elements for top summary table
        extra_top_left = None
        extra_top_right = []
        mle_retvals = getattr(self, 'mle_retvals', None)
        mle_settings = getattr(self, 'mle_settings', None)
        if mle_settings is not None and mle_settings.method == 'em':
            extra_top_right += [('EM Iterations', [f'{mle_retvals.iter}'])]

        # Get the basic summary tables
        summary = super().summary(
            alpha=alpha, start=start, title=title, model_name=model_name,
            display_params=(display_params and display_params_as_list),
            display_diagnostics=display_diagnostics,
            truncate_endog_names=truncate_endog_names,
            display_max_endog=display_max_endog,
            extra_top_left=extra_top_left, extra_top_right=extra_top_right)

        # Get tables of parameters
        table_ix = 1
        if not display_params_as_list:

            # Observation equation table
            data = pd.DataFrame(
                self.filter_results.design[:, mod._s['factors_L1'], 0],
                index=endog_names, columns=mod.factor_names)
            try:
                data = data.map(lambda s: '%.2f' % s)
            except AttributeError:
                data = data.applymap(lambda s: '%.2f' % s)

            # Idiosyncratic terms
            # data['   '] = '   '
            k_idio = 1
            if mod.idiosyncratic_ar1:
                data['   idiosyncratic: AR(1)'] = (
                    self.params[mod._p['idiosyncratic_ar1']])
                k_idio += 1
            data['var.'] = self.params[mod._p['idiosyncratic_var']]
            try:
                data.iloc[:, -k_idio:] = data.iloc[:, -k_idio:].map(
                    lambda s: f'{s:.2f}')
            except AttributeError:
                data.iloc[:, -k_idio:] = data.iloc[:, -k_idio:].applymap(
                    lambda s: f'{s:.2f}')

            data.index.name = 'Factor loadings:'

            # Clear entries for non-loading factors
            base_iloc = np.arange(mod.k_factors)
            for i in range(mod.k_endog):
                iloc = [j for j in base_iloc
                        if j not in mod._s.endog_factor_iloc[i]]
                data.iloc[i, iloc] = '.'

            data = data.reset_index()

            # Build the table
            params_data = data.values
            params_header = data.columns.tolist()
            params_stubs = None

            title = 'Observation equation:'
            table = SimpleTable(
                params_data, params_header, params_stubs,
                txt_fmt=fmt_params, title=title)
            summary.tables.insert(table_ix, table)
            table_ix += 1

            # Factor transitions
            ix1 = 0
            ix2 = 0
            for i in range(len(mod._s.factor_blocks)):
                block = mod._s.factor_blocks[i]
                ix2 += block.k_factors

                T = self.filter_results.transition
                lag_names = []
                for j in range(block.factor_order):
                    lag_names += [f'L{j + 1}.{name}'
                                  for name in block.factor_names]
                data = pd.DataFrame(T[block.factors_L1, block.factors_ar, 0],
                                    index=block.factor_names,
                                    columns=lag_names)
                data.index.name = ''
                try:
                    data = data.map(lambda s: '%.2f' % s)
                except AttributeError:
                    data = data.applymap(lambda s: '%.2f' % s)

                Q = self.filter_results.state_cov
                # data[' '] = ''
                if block.k_factors == 1:
                    data['   error variance'] = Q[ix1, ix1]
                else:
                    data['   error covariance'] = block.factor_names
                    for j in range(block.k_factors):
                        data[block.factor_names[j]] = Q[ix1:ix2, ix1 + j]
                try:
                    formatted_vals = data.iloc[:, -block.k_factors:].map(
                        lambda s: f'{s:.2f}'
                    )
                except AttributeError:
                    formatted_vals = data.iloc[:, -block.k_factors:].applymap(
                        lambda s: f'{s:.2f}'
                    )
                data.iloc[:, -block.k_factors:] = formatted_vals

                data = data.reset_index()

                params_data = data.values
                params_header = data.columns.tolist()
                params_stubs = None

                title = f'Transition: Factor block {i}'
                table = SimpleTable(
                    params_data, params_header, params_stubs,
                    txt_fmt=fmt_params, title=title)
                summary.tables.insert(table_ix, table)
                table_ix += 1

                ix1 = ix2

        return summary
