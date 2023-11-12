"""Test utilities."""

# Adapted from scikit-learn
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.neighbors import KDTree
from sklearn.utils._testing import ignore_warnings


def all_estimators(
    type_filter=None,
):
    """Get a list of all estimators from imblearn.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from sklearn.

    Parameters
    ----------
    type_filter : str, list of str, or None, default=None
        Which kind of estimators should be returned. If None, no
        filter is applied and all estimators are returned.  Possible
        values are 'sampler' to get estimators only of these specific
        types, or a list of these to get the estimators that fit at
        least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    from ..base import SamplerMixin

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {"tests"}
    root = str(Path(__file__).parent.parent)
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[root], prefix="imblearn."
        ):
            mod_parts = modname.split(".")
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue
            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    # get rid of sklearn estimators which have been imported in some classes
    estimators = [c for c in estimators if "sklearn" not in c[1].__module__]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {"sampler": SamplerMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'sampler' or "
                "None, got"
                " %s." % repr(type_filter)
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


class _CustomNearestNeighbors(BaseEstimator):
    """Basic implementation of nearest neighbors not relying on scikit-learn.

    `kneighbors_graph` is ignored and `metric` does not have any impact.
    """

    def __init__(self, n_neighbors=1, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y=None):
        X = X.toarray() if sparse.issparse(X) else X
        self._kd_tree = KDTree(X)
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        n_neighbors = n_neighbors if n_neighbors is not None else self.n_neighbors
        X = X.toarray() if sparse.issparse(X) else X
        distances, indices = self._kd_tree.query(X, k=n_neighbors)
        if return_distance:
            return distances, indices
        return indices

    def kneighbors_graph(X=None, n_neighbors=None, mode="connectivity"):
        """This method is not used within imblearn but it is required for
        duck-typing."""
        pass


class _CustomClusterer(BaseEstimator):
    """Class that mimics a cluster that does not expose `cluster_centers_`."""

    def __init__(self, n_clusters=1, expose_cluster_centers=True):
        self.n_clusters = n_clusters
        self.expose_cluster_centers = expose_cluster_centers

    def fit(self, X, y=None):
        if self.expose_cluster_centers:
            self.cluster_centers_ = np.random.randn(self.n_clusters, X.shape[1])
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)
