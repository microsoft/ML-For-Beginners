from numbers import Integral, Real

import numpy as np

from ..base import OneToOneFeatureMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import type_of_target
from ..utils.validation import _check_y, check_consistent_length
from ._encoders import _BaseEncoder
from ._target_encoder_fast import _fit_encoding_fast, _fit_encoding_fast_auto_smooth


class TargetEncoder(OneToOneFeatureMixin, _BaseEncoder):
    """Target Encoder for regression and classification targets.

    Each category is encoded based on a shrunk estimate of the average target
    values for observations belonging to the category. The encoding scheme mixes
    the global target mean with the target mean conditioned on the value of the
    category. [MIC]_

    :class:`TargetEncoder` considers missing values, such as `np.nan` or `None`,
    as another category and encodes them like any other category. Categories
    that are not seen during :meth:`fit` are encoded with the target mean, i.e.
    `target_mean_`.

    Read more in the :ref:`User Guide <target_encoder>`.

    .. note::
        `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
        cross fitting scheme is used in `fit_transform` for encoding. See the
        :ref:`User Guide <target_encoder>` for details.

    .. versionadded:: 1.3

    Parameters
    ----------
    categories : "auto" or list of shape (n_features,) of array-like, default="auto"
        Categories (unique values) per feature:

        - `"auto"` : Determine categories automatically from the training data.
        - list : `categories[i]` holds the categories expected in the i-th column. The
          passed categories should not mix strings and numeric values within a single
          feature, and should be sorted in case of numeric values.

        The used categories are stored in the `categories_` fitted attribute.

    target_type : {"auto", "continuous", "binary"}, default="auto"
        Type of target.

        - `"auto"` : Type of target is inferred with
          :func:`~sklearn.utils.multiclass.type_of_target`.
        - `"continuous"` : Continuous target
        - `"binary"` : Binary target

        .. note::
            The type of target inferred with `"auto"` may not be the desired target
            type used for modeling. For example, if the target consisted of integers
            between 0 and 100, then :func:`~sklearn.utils.multiclass.type_of_target`
            will infer the target as `"multiclass"`. In this case, setting
            `target_type="continuous"` will specify the target as a regression
            problem. The `target_type_` attribute gives the target type used by the
            encoder.

    smooth : "auto" or float, default="auto"
        The amount of mixing of the target mean conditioned on the value of the
        category with the global target mean. A larger `smooth` value will put
        more weight on the global target mean.
        If `"auto"`, then `smooth` is set to an empirical Bayes estimate.

    cv : int, default=5
        Determines the number of folds in the cross fitting strategy used in
        :meth:`fit_transform`. For classification targets, `StratifiedKFold` is used
        and for continuous targets, `KFold` is used.

    shuffle : bool, default=True
        Whether to shuffle the data in :meth:`fit_transform` before splitting into
        folds. Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    encodings_ : list of shape (n_features,) of ndarray
        Encodings learnt on all of `X`.
        For feature `i`, `encodings_[i]` are the encodings matching the
        categories listed in `categories_[i]`.

    categories_ : list of shape (n_features,) of ndarray
        The categories of each feature determined during fitting or specified
        in `categories`
        (in order of the features in `X` and corresponding with the output
        of :meth:`transform`).

    target_type_ : str
        Type of target.

    target_mean_ : float
        The overall mean of the target. This value is only used in :meth:`transform`
        to encode categories.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    OrdinalEncoder : Performs an ordinal (integer) encoding of the categorical features.
        Contrary to TargetEncoder, this encoding is not supervised. Treating the
        resulting encoding as a numerical features therefore lead arbitrarily
        ordered values and therefore typically lead to lower predictive performance
        when used as preprocessing for a classifier or regressor.
    OneHotEncoder : Performs a one-hot encoding of categorical features. This
        unsupervised encoding is better suited for low cardinality categorical
        variables as it generate one new feature per unique category.

    References
    ----------
    .. [MIC] :doi:`Micci-Barreca, Daniele. "A preprocessing scheme for high-cardinality
       categorical attributes in classification and prediction problems"
       SIGKDD Explor. Newsl. 3, 1 (July 2001), 27–32. <10.1145/507533.507538>`

    Examples
    --------
    With `smooth="auto"`, the smoothing parameter is set to an empirical Bayes estimate:

    >>> import numpy as np
    >>> from sklearn.preprocessing import TargetEncoder
    >>> X = np.array([["dog"] * 20 + ["cat"] * 30 + ["snake"] * 38], dtype=object).T
    >>> y = [90.3] * 5 + [80.1] * 15 + [20.4] * 5 + [20.1] * 25 + [21.2] * 8 + [49] * 30
    >>> enc_auto = TargetEncoder(smooth="auto")
    >>> X_trans = enc_auto.fit_transform(X, y)

    >>> # A high `smooth` parameter puts more weight on global mean on the categorical
    >>> # encodings:
    >>> enc_high_smooth = TargetEncoder(smooth=5000.0).fit(X, y)
    >>> enc_high_smooth.target_mean_
    44...
    >>> enc_high_smooth.encodings_
    [array([44..., 44..., 44...])]

    >>> # On the other hand, a low `smooth` parameter puts more weight on target
    >>> # conditioned on the value of the categorical:
    >>> enc_low_smooth = TargetEncoder(smooth=1.0).fit(X, y)
    >>> enc_low_smooth.encodings_
    [array([20..., 80..., 43...])]
    """

    _parameter_constraints: dict = {
        "categories": [StrOptions({"auto"}), list],
        "target_type": [StrOptions({"auto", "continuous", "binary"})],
        "smooth": [StrOptions({"auto"}), Interval(Real, 0, None, closed="left")],
        "cv": [Interval(Integral, 2, None, closed="left")],
        "shuffle": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        categories="auto",
        target_type="auto",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=None,
    ):
        self.categories = categories
        self.smooth = smooth
        self.target_type = target_type
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the :class:`TargetEncoder` to X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : array-like of shape (n_samples,)
            The target data used to encode the categories.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        self._fit_encodings_all(X, y)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y):
        """Fit :class:`TargetEncoder` and transform X with the target encoding.

        .. note::
            `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
            cross fitting scheme is used in `fit_transform` for encoding. See the
            :ref:`User Guide <target_encoder>`. for details.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : array-like of shape (n_samples,)
            The target data used to encode the categories.

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        from ..model_selection import KFold, StratifiedKFold  # avoid circular import

        X_ordinal, X_known_mask, y, n_categories = self._fit_encodings_all(X, y)

        # The cv splitter is voluntarily restricted to *KFold to enforce non
        # overlapping validation folds, otherwise the fit_transform output will
        # not be well-specified.
        if self.target_type_ == "continuous":
            cv = KFold(self.cv, shuffle=self.shuffle, random_state=self.random_state)
        else:
            cv = StratifiedKFold(
                self.cv, shuffle=self.shuffle, random_state=self.random_state
            )

        X_out = np.empty_like(X_ordinal, dtype=np.float64)
        X_unknown_mask = ~X_known_mask

        for train_idx, test_idx in cv.split(X, y):
            X_train, y_train = X_ordinal[train_idx, :], y[train_idx]
            y_mean = np.mean(y_train)

            if self.smooth == "auto":
                y_variance = np.var(y_train)
                encodings = _fit_encoding_fast_auto_smooth(
                    X_train, y_train, n_categories, y_mean, y_variance
                )
            else:
                encodings = _fit_encoding_fast(
                    X_train, y_train, n_categories, self.smooth, y_mean
                )
            self._transform_X_ordinal(
                X_out, X_ordinal, X_unknown_mask, test_idx, encodings, y_mean
            )
        return X_out

    def transform(self, X):
        """Transform X with the target encoding.

        .. note::
            `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
            cross fitting scheme is used in `fit_transform` for encoding. See the
            :ref:`User Guide <target_encoder>`. for details.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X_ordinal, X_valid = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        X_out = np.empty_like(X_ordinal, dtype=np.float64)
        self._transform_X_ordinal(
            X_out,
            X_ordinal,
            ~X_valid,
            slice(None),
            self.encodings_,
            self.target_mean_,
        )
        return X_out

    def _fit_encodings_all(self, X, y):
        """Fit a target encoding with all the data."""
        from ..preprocessing import LabelEncoder  # avoid circular import

        check_consistent_length(X, y)
        self._fit(X, handle_unknown="ignore", force_all_finite="allow-nan")

        if self.target_type == "auto":
            accepted_target_types = ("binary", "continuous")
            inferred_type_of_target = type_of_target(y, input_name="y")
            if inferred_type_of_target not in accepted_target_types:
                raise ValueError(
                    f"Target type was inferred to be {inferred_type_of_target!r}. Only"
                    f" {accepted_target_types} are supported."
                )
            self.target_type_ = inferred_type_of_target
        else:
            self.target_type_ = self.target_type

        if self.target_type_ == "binary":
            y = LabelEncoder().fit_transform(y)
        else:  # continuous
            y = _check_y(y, y_numeric=True, estimator=self)

        self.target_mean_ = np.mean(y)

        X_ordinal, X_known_mask = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        n_categories = np.fromiter(
            (len(category_for_feature) for category_for_feature in self.categories_),
            dtype=np.int64,
            count=len(self.categories_),
        )
        if self.smooth == "auto":
            y_variance = np.var(y)
            self.encodings_ = _fit_encoding_fast_auto_smooth(
                X_ordinal, y, n_categories, self.target_mean_, y_variance
            )
        else:
            self.encodings_ = _fit_encoding_fast(
                X_ordinal, y, n_categories, self.smooth, self.target_mean_
            )

        return X_ordinal, X_known_mask, y, n_categories

    @staticmethod
    def _transform_X_ordinal(
        X_out, X_ordinal, X_unknown_mask, indices, encodings, y_mean
    ):
        """Transform X_ordinal using encodings."""
        for f_idx, encoding in enumerate(encodings):
            X_out[indices, f_idx] = encoding[X_ordinal[indices, f_idx]]
            X_out[X_unknown_mask[:, f_idx], f_idx] = y_mean

    def _more_tags(self):
        return {"requires_y": True}
