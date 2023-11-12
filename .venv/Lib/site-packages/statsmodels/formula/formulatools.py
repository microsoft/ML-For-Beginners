import statsmodels.tools.data as data_util
from patsy import dmatrices, NAAction
import numpy as np

# if users want to pass in a different formula framework, they can
# add their handler here. how to do it interactively?

# this is a mutable object, so editing it should show up in the below
formula_handler = {}


class NAAction(NAAction):
    # monkey-patch so we can handle missing values in 'extra' arrays later
    def _handle_NA_drop(self, values, is_NAs, origins):
        total_mask = np.zeros(is_NAs[0].shape[0], dtype=bool)
        for is_NA in is_NAs:
            total_mask |= is_NA
        good_mask = ~total_mask
        self.missing_mask = total_mask
        # "..." to handle 1- versus 2-dim indexing
        return [v[good_mask, ...] for v in values]


def handle_formula_data(Y, X, formula, depth=0, missing='drop'):
    """
    Returns endog, exog, and the model specification from arrays and formula.

    Parameters
    ----------
    Y : array_like
        Either endog (the LHS) of a model specification or all of the data.
        Y must define __getitem__ for now.
    X : array_like
        Either exog or None. If all the data for the formula is provided in
        Y then you must explicitly set X to None.
    formula : str or patsy.model_desc
        You can pass a handler by import formula_handler and adding a
        key-value pair where the key is the formula object class and
        the value is a function that returns endog, exog, formula object.

    Returns
    -------
    endog : array_like
        Should preserve the input type of Y,X.
    exog : array_like
        Should preserve the input type of Y,X. Could be None.
    """
    # half ass attempt to handle other formula objects
    if isinstance(formula, tuple(formula_handler.keys())):
        return formula_handler[type(formula)]

    na_action = NAAction(on_NA=missing)

    if X is not None:
        if data_util._is_using_pandas(Y, X):
            result = dmatrices(formula, (Y, X), depth,
                               return_type='dataframe', NA_action=na_action)
        else:
            result = dmatrices(formula, (Y, X), depth,
                               return_type='dataframe', NA_action=na_action)
    else:
        if data_util._is_using_pandas(Y, None):
            result = dmatrices(formula, Y, depth, return_type='dataframe',
                               NA_action=na_action)
        else:
            result = dmatrices(formula, Y, depth, return_type='dataframe',
                               NA_action=na_action)

    # if missing == 'raise' there's not missing_mask
    missing_mask = getattr(na_action, 'missing_mask', None)
    if not np.any(missing_mask):
        missing_mask = None
    if len(result) > 1:  # have RHS design
        design_info = result[1].design_info  # detach it from DataFrame
    else:
        design_info = None
    # NOTE: is there ever a case where we'd need LHS design_info?
    return result, missing_mask, design_info


def _remove_intercept_patsy(terms):
    """
    Remove intercept from Patsy terms.
    """
    from patsy.desc import INTERCEPT
    if INTERCEPT in terms:
        terms.remove(INTERCEPT)
    return terms


def _has_intercept(design_info):
    from patsy.desc import INTERCEPT
    return INTERCEPT in design_info.terms


def _intercept_idx(design_info):
    """
    Returns boolean array index indicating which column holds the intercept.
    """
    from patsy.desc import INTERCEPT
    from numpy import array
    return array([INTERCEPT == i for i in design_info.terms])


def make_hypotheses_matrices(model_results, test_formula):
    """
    """
    from patsy.constraint import linear_constraint
    exog_names = model_results.model.exog_names
    LC = linear_constraint(test_formula, exog_names)
    return LC
