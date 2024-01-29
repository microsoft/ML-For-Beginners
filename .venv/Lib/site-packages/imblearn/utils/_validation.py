"""Utilities for input validation"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import warnings
from collections import OrderedDict
from functools import wraps
from inspect import Parameter, signature
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples

from .fixes import _is_pandas_df

SAMPLING_KIND = (
    "over-sampling",
    "under-sampling",
    "clean-sampling",
    "ensemble",
    "bypass",
)
TARGET_KIND = ("binary", "multiclass", "multilabel-indicator")


class ArraysTransformer:
    """A class to convert sampler output arrays to their original types."""

    def __init__(self, X, y):
        self.x_props = self._gets_props(X)
        self.y_props = self._gets_props(y)

    def transform(self, X, y):
        X = self._transfrom_one(X, self.x_props)
        y = self._transfrom_one(y, self.y_props)
        if self.x_props["type"].lower() == "dataframe" and self.y_props[
            "type"
        ].lower() in {"series", "dataframe"}:
            # We lost the y.index during resampling. We can safely use X.index to align
            # them.
            y.index = X.index
        return X, y

    def _gets_props(self, array):
        props = {}
        props["type"] = array.__class__.__name__
        props["columns"] = getattr(array, "columns", None)
        props["name"] = getattr(array, "name", None)
        props["dtypes"] = getattr(array, "dtypes", None)
        return props

    def _transfrom_one(self, array, props):
        type_ = props["type"].lower()
        if type_ == "list":
            ret = array.tolist()
        elif type_ == "dataframe":
            import pandas as pd

            if issparse(array):
                ret = pd.DataFrame.sparse.from_spmatrix(array, columns=props["columns"])
            else:
                ret = pd.DataFrame(array, columns=props["columns"])

            try:
                ret = ret.astype(props["dtypes"])
            except TypeError:
                # We special case the following error:
                # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1055
                # There is no easy way to have a generic workaround. Here, we detect
                # that we have a column with only null values that is datetime64
                # (resulting from the np.vstack of the resampling).
                for col in ret.columns:
                    if (
                        ret[col].isnull().all()
                        and ret[col].dtype == "datetime64[ns]"
                        and props["dtypes"][col] == "timedelta64[ns]"
                    ):
                        ret[col] = pd.to_timedelta(["NaT"] * len(ret[col]))
                # try again
                ret = ret.astype(props["dtypes"])
        elif type_ == "series":
            import pandas as pd

            ret = pd.Series(array, dtype=props["dtypes"], name=props["name"])
        else:
            ret = array
        return ret


def _is_neighbors_object(estimator):
    """Check that the estimator exposes a KNeighborsMixin-like API.

    A KNeighborsMixin-like API exposes the following methods: (i) `kneighbors`,
    (ii) `kneighbors_graph`.

    Parameters
    ----------
    estimator : object
        A scikit-learn compatible estimator.

    Returns
    -------
    is_neighbors_object : bool
        True if the estimator exposes a KNeighborsMixin-like API.
    """
    neighbors_attributes = ["kneighbors", "kneighbors_graph"]
    return all(hasattr(estimator, attr) for attr in neighbors_attributes)


def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    """Check the objects is consistent to be a k nearest neighbors.

    Several methods in `imblearn` relies on k nearest neighbors. These objects
    can be passed at initialisation as an integer or as an object that has
    KNeighborsMixin-like attributes. This utility will create or clone said
    object, ensuring it is KNeighbors-like.

    Parameters
    ----------
    nn_name : str
        The name associated to the object to raise an error if needed.

    nn_object : int or KNeighborsMixin
        The object to be checked.

    additional_neighbor : int, default=0
        Sometimes, some algorithm need an additional neighbors.

    Returns
    -------
    nn_object : KNeighborsMixin
        The k-NN object.
    """
    if isinstance(nn_object, Integral):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    # _is_neighbors_object(nn_object)
    return clone(nn_object)


def _count_class_sample(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def check_target_type(y, indicate_one_vs_all=False):
    """Check the target types to be conform to the current samplers.

    The current samplers should be compatible with ``'binary'``,
    ``'multilabel-indicator'`` and ``'multiclass'`` targets only.

    Parameters
    ----------
    y : ndarray
        The array containing the target.

    indicate_one_vs_all : bool, default=False
        Either to indicate if the targets are encoded in a one-vs-all fashion.

    Returns
    -------
    y : ndarray
        The returned target.

    is_one_vs_all : bool, optional
        Indicate if the target was originally encoded in a one-vs-all fashion.
        Only returned if ``indicate_multilabel=True``.
    """
    type_y = type_of_target(y)
    if type_y == "multilabel-indicator":
        if np.any(y.sum(axis=1) > 1):
            raise ValueError(
                "Imbalanced-learn currently supports binary, multiclass and "
                "binarized encoded multiclasss targets. Multilabel and "
                "multioutput targets are not supported."
            )
        y = y.argmax(axis=1)
    else:
        y = column_or_1d(y)

    return (y, type_y == "multilabel-indicator") if indicate_one_vs_all else y


def _sampling_strategy_all(y, sampling_type):
    """Returns sampling target by targeting all classes."""
    target_stats = _count_class_sample(y)
    if sampling_type == "over-sampling":
        n_sample_majority = max(target_stats.values())
        sampling_strategy = {
            key: n_sample_majority - value for (key, value) in target_stats.items()
        }
    elif sampling_type == "under-sampling" or sampling_type == "clean-sampling":
        n_sample_minority = min(target_stats.values())
        sampling_strategy = {key: n_sample_minority for key in target_stats.keys()}
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_majority(y, sampling_type):
    """Returns sampling target by targeting the majority class only."""
    if sampling_type == "over-sampling":
        raise ValueError(
            "'sampling_strategy'='majority' cannot be used with over-sampler."
        )
    elif sampling_type == "under-sampling" or sampling_type == "clean-sampling":
        target_stats = _count_class_sample(y)
        class_majority = max(target_stats, key=target_stats.get)
        n_sample_minority = min(target_stats.values())
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys()
            if key == class_majority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_not_majority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    majority."""
    target_stats = _count_class_sample(y)
    if sampling_type == "over-sampling":
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
    elif sampling_type == "under-sampling" or sampling_type == "clean-sampling":
        n_sample_minority = min(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys()
            if key != class_majority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_not_minority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    minority."""
    target_stats = _count_class_sample(y)
    if sampling_type == "over-sampling":
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_minority
        }
    elif sampling_type == "under-sampling" or sampling_type == "clean-sampling":
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys()
            if key != class_minority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_minority(y, sampling_type):
    """Returns sampling target by targeting the minority class only."""
    target_stats = _count_class_sample(y)
    if sampling_type == "over-sampling":
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key == class_minority
        }
    elif sampling_type == "under-sampling" or sampling_type == "clean-sampling":
        raise ValueError(
            "'sampling_strategy'='minority' cannot be used with"
            " under-sampler and clean-sampler."
        )
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_auto(y, sampling_type):
    """Returns sampling target auto for over-sampling and not-minority for
    under-sampling."""
    if sampling_type == "over-sampling":
        return _sampling_strategy_not_majority(y, sampling_type)
    elif sampling_type == "under-sampling" or sampling_type == "clean-sampling":
        return _sampling_strategy_not_minority(y, sampling_type)


def _sampling_strategy_dict(sampling_strategy, y, sampling_type):
    """Returns sampling target by converting the dictionary depending of the
    sampling."""
    target_stats = _count_class_sample(y)
    # check that all keys in sampling_strategy are also in y
    set_diff_sampling_strategy_target = set(sampling_strategy.keys()) - set(
        target_stats.keys()
    )
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError(
            f"The {set_diff_sampling_strategy_target} target class is/are not "
            f"present in the data."
        )
    # check that there is no negative number
    if any(n_samples < 0 for n_samples in sampling_strategy.values()):
        raise ValueError(
            f"The number of samples in a class cannot be negative."
            f"'sampling_strategy' contains some negative value: {sampling_strategy}"
        )
    sampling_strategy_ = {}
    if sampling_type == "over-sampling":
        max(target_stats.values())
        max(target_stats, key=target_stats.get)
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples < target_stats[class_sample]:
                raise ValueError(
                    f"With over-sampling methods, the number"
                    f" of samples in a class should be greater"
                    f" or equal to the original number of samples."
                    f" Originally, there is {target_stats[class_sample]} "
                    f"samples and {n_samples} samples are asked."
                )
            sampling_strategy_[class_sample] = n_samples - target_stats[class_sample]
    elif sampling_type == "under-sampling":
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples > target_stats[class_sample]:
                raise ValueError(
                    f"With under-sampling methods, the number of"
                    f" samples in a class should be less or equal"
                    f" to the original number of samples."
                    f" Originally, there is {target_stats[class_sample]} "
                    f"samples and {n_samples} samples are asked."
                )
            sampling_strategy_[class_sample] = n_samples
    elif sampling_type == "clean-sampling":
        raise ValueError(
            "'sampling_strategy' as a dict for cleaning methods is "
            "not supported. Please give a list of the classes to be "
            "targeted by the sampling."
        )
    else:
        raise NotImplementedError

    return sampling_strategy_


def _sampling_strategy_list(sampling_strategy, y, sampling_type):
    """With cleaning methods, sampling_strategy can be a list to target the
    class of interest."""
    if sampling_type != "clean-sampling":
        raise ValueError(
            "'sampling_strategy' cannot be a list for samplers "
            "which are not cleaning methods."
        )

    target_stats = _count_class_sample(y)
    # check that all keys in sampling_strategy are also in y
    set_diff_sampling_strategy_target = set(sampling_strategy) - set(
        target_stats.keys()
    )
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError(
            f"The {set_diff_sampling_strategy_target} target class is/are not "
            f"present in the data."
        )

    return {
        class_sample: min(target_stats.values()) for class_sample in sampling_strategy
    }


def _sampling_strategy_float(sampling_strategy, y, sampling_type):
    """Take a proportion of the majority (over-sampling) or minority
    (under-sampling) class in binary classification."""
    type_y = type_of_target(y)
    if type_y != "binary":
        raise ValueError(
            '"sampling_strategy" can be a float only when the type '
            "of target is binary. For multi-class, use a dict."
        )
    target_stats = _count_class_sample(y)
    if sampling_type == "over-sampling":
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy_ = {
            key: int(n_sample_majority * sampling_strategy - value)
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        if any([n_samples <= 0 for n_samples in sampling_strategy_.values()]):
            raise ValueError(
                "The specified ratio required to remove samples "
                "from the minority class while trying to "
                "generate new samples. Please increase the "
                "ratio."
            )
    elif sampling_type == "under-sampling":
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy_ = {
            key: int(n_sample_minority / sampling_strategy)
            for (key, value) in target_stats.items()
            if key != class_minority
        }
        if any(
            [
                n_samples > target_stats[target]
                for target, n_samples in sampling_strategy_.items()
            ]
        ):
            raise ValueError(
                "The specified ratio required to generate new "
                "sample in the majority class while trying to "
                "remove samples. Please increase the ratio."
            )
    else:
        raise ValueError(
            "'clean-sampling' methods do let the user specify the sampling ratio."
        )
    return sampling_strategy_


def check_sampling_strategy(sampling_strategy, y, sampling_type, **kwargs):
    """Sampling target validation for samplers.

    Checks that ``sampling_strategy`` is of consistent type and return a
    dictionary containing each targeted class with its corresponding
    number of sample. It is used in :class:`~imblearn.base.BaseSampler`.

    Parameters
    ----------
    sampling_strategy : float, str, dict, list or callable,
        Sampling information to sample the data set.

        - When ``float``:

            For **under-sampling methods**, it corresponds to the ratio
            :math:`\\alpha_{us}` defined by :math:`N_{rM} = \\alpha_{us}
            \\times N_{m}` where :math:`N_{rM}` and :math:`N_{m}` are the
            number of samples in the majority class after resampling and the
            number of samples in the minority class, respectively;

            For **over-sampling methods**, it correspond to the ratio
            :math:`\\alpha_{os}` defined by :math:`N_{rm} = \\alpha_{os}
            \\times N_{m}` where :math:`N_{rm}` and :math:`N_{M}` are the
            number of samples in the minority class after resampling and the
            number of samples in the majority class, respectively.

            .. warning::
               ``float`` is only available for **binary** classification. An
               error is raised for multi-class classification and with cleaning
               samplers.

        - When ``str``, specify the class targeted by the resampling. For
          **under- and over-sampling methods**, the number of samples in the
          different classes will be equalized. For **cleaning methods**, the
          number of samples will not be equal. Possible choices are:

            ``'minority'``: resample only the minority class;

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: for under-sampling methods, equivalent to ``'not
            minority'`` and for over-sampling methods, equivalent to ``'not
            majority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

          .. warning::
             ``dict`` is available for both **under- and over-sampling
             methods**. An error is raised with **cleaning methods**. Use a
             ``list`` instead.

        - When ``list``, the list contains the targeted classes. It used only
          for **cleaning methods**.

          .. warning::
             ``list`` is available for **cleaning methods**. An error is raised
             with **under- and over-sampling methods**.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

    y : ndarray of shape (n_samples,)
        The target array.

    sampling_type : {{'over-sampling', 'under-sampling', 'clean-sampling'}}
        The type of sampling. Can be either ``'over-sampling'``,
        ``'under-sampling'``, or ``'clean-sampling'``.

    **kwargs : dict
        Dictionary of additional keyword arguments to pass to
        ``sampling_strategy`` when this is a callable.

    Returns
    -------
    sampling_strategy_converted : dict
        The converted and validated sampling target. Returns a dictionary with
        the key being the class target and the value being the desired
        number of samples.
    """
    if sampling_type not in SAMPLING_KIND:
        raise ValueError(
            f"'sampling_type' should be one of {SAMPLING_KIND}. "
            f"Got '{sampling_type} instead."
        )

    if np.unique(y).size <= 1:
        raise ValueError(
            f"The target 'y' needs to have more than 1 class. "
            f"Got {np.unique(y).size} class instead"
        )

    if sampling_type in ("ensemble", "bypass"):
        return sampling_strategy

    if isinstance(sampling_strategy, str):
        if sampling_strategy not in SAMPLING_TARGET_KIND.keys():
            raise ValueError(
                f"When 'sampling_strategy' is a string, it needs"
                f" to be one of {SAMPLING_TARGET_KIND}. Got '{sampling_strategy}' "
                f"instead."
            )
        return OrderedDict(
            sorted(SAMPLING_TARGET_KIND[sampling_strategy](y, sampling_type).items())
        )
    elif isinstance(sampling_strategy, dict):
        return OrderedDict(
            sorted(_sampling_strategy_dict(sampling_strategy, y, sampling_type).items())
        )
    elif isinstance(sampling_strategy, list):
        return OrderedDict(
            sorted(_sampling_strategy_list(sampling_strategy, y, sampling_type).items())
        )
    elif isinstance(sampling_strategy, Real):
        if sampling_strategy <= 0 or sampling_strategy > 1:
            raise ValueError(
                f"When 'sampling_strategy' is a float, it should be "
                f"in the range (0, 1]. Got {sampling_strategy} instead."
            )
        return OrderedDict(
            sorted(
                _sampling_strategy_float(sampling_strategy, y, sampling_type).items()
            )
        )
    elif callable(sampling_strategy):
        sampling_strategy_ = sampling_strategy(y, **kwargs)
        return OrderedDict(
            sorted(
                _sampling_strategy_dict(sampling_strategy_, y, sampling_type).items()
            )
        )


SAMPLING_TARGET_KIND = {
    "minority": _sampling_strategy_minority,
    "majority": _sampling_strategy_majority,
    "not minority": _sampling_strategy_not_minority,
    "not majority": _sampling_strategy_not_majority,
    "all": _sampling_strategy_all,
    "auto": _sampling_strategy_auto,
}


def _deprecate_positional_args(f):
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    f : function
        function to check arguments on.
    """
    sig = signature(f)
    kwonly_args = []
    all_args = []

    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(f)
    def inner_f(*args, **kwargs):
        extra_args = len(args) - len(all_args)
        if extra_args > 0:
            # ignore first 'self' argument for instance methods
            args_msg = [
                f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            warnings.warn(
                f"Pass {', '.join(args_msg)} as keyword args. From version 0.9 "
                f"passing these as positional arguments will "
                f"result in an error",
                FutureWarning,
            )
        kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
        return f(**kwargs)

    return inner_f


def _check_X(X):
    """Check X and do not check it if a dataframe."""
    n_samples = _num_samples(X)
    if n_samples < 1:
        raise ValueError(
            f"Found array with {n_samples} sample(s) while a minimum of 1 is "
            "required."
        )
    if _is_pandas_df(X):
        return X
    return check_array(
        X, dtype=None, accept_sparse=["csr", "csc"], force_all_finite=False
    )
