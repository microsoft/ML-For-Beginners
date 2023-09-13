def _check_feature_names(X, feature_names=None):
    """Check feature names.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    feature_names : None or array-like of shape (n_names,), dtype=str
        Feature names to check or `None`.

    Returns
    -------
    feature_names : list of str
        Feature names validated. If `feature_names` is `None`, then a list of
        feature names is provided, i.e. the column names of a pandas dataframe
        or a generic list of feature names (e.g. `["x0", "x1", ...]`) for a
        NumPy array.
    """
    if feature_names is None:
        if hasattr(X, "columns") and hasattr(X.columns, "tolist"):
            # get the column names for a pandas dataframe
            feature_names = X.columns.tolist()
        else:
            # define a list of numbered indices for a numpy array
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    elif hasattr(feature_names, "tolist"):
        # convert numpy array or pandas index to a list
        feature_names = feature_names.tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names should not contain duplicates.")

    return feature_names


def _get_feature_index(fx, feature_names=None):
    """Get feature index.

    Parameters
    ----------
    fx : int or str
        Feature index or name.

    feature_names : list of str, default=None
        All feature names from which to search the indices.

    Returns
    -------
    idx : int
        Feature index.
    """
    if isinstance(fx, str):
        if feature_names is None:
            raise ValueError(
                f"Cannot plot partial dependence for feature {fx!r} since "
                "the list of feature names was not provided, neither as "
                "column names of a pandas data-frame nor via the feature_names "
                "parameter."
            )
        try:
            return feature_names.index(fx)
        except ValueError as e:
            raise ValueError(f"Feature {fx!r} not in feature_names") from e
    return fx
