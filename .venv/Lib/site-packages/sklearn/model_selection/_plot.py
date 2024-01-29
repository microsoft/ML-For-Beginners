import warnings

import numpy as np

from ..utils import check_matplotlib_support
from ..utils._plotting import _interval_max_min_ratio, _validate_score_name
from ._validation import learning_curve, validation_curve


class _BaseCurveDisplay:
    def _plot_curve(
        self,
        x_data,
        *,
        ax=None,
        negate_score=False,
        score_name=None,
        score_type="test",
        log_scale="deprecated",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        check_matplotlib_support(f"{self.__class__.__name__}.plot")

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if negate_score:
            train_scores, test_scores = -self.train_scores, -self.test_scores
        else:
            train_scores, test_scores = self.train_scores, self.test_scores

        if std_display_style not in ("errorbar", "fill_between", None):
            raise ValueError(
                f"Unknown std_display_style: {std_display_style}. Should be one of"
                " 'errorbar', 'fill_between', or None."
            )

        if score_type not in ("test", "train", "both"):
            raise ValueError(
                f"Unknown score_type: {score_type}. Should be one of 'test', "
                "'train', or 'both'."
            )

        if score_type == "train":
            scores = {"Train": train_scores}
        elif score_type == "test":
            scores = {"Test": test_scores}
        else:  # score_type == "both"
            scores = {"Train": train_scores, "Test": test_scores}

        if std_display_style in ("fill_between", None):
            # plot the mean score
            if line_kw is None:
                line_kw = {}

            self.lines_ = []
            for line_label, score in scores.items():
                self.lines_.append(
                    *ax.plot(
                        x_data,
                        score.mean(axis=1),
                        label=line_label,
                        **line_kw,
                    )
                )
            self.errorbar_ = None
            self.fill_between_ = None  # overwritten below by fill_between

        if std_display_style == "errorbar":
            if errorbar_kw is None:
                errorbar_kw = {}

            self.errorbar_ = []
            for line_label, score in scores.items():
                self.errorbar_.append(
                    ax.errorbar(
                        x_data,
                        score.mean(axis=1),
                        score.std(axis=1),
                        label=line_label,
                        **errorbar_kw,
                    )
                )
            self.lines_, self.fill_between_ = None, None
        elif std_display_style == "fill_between":
            if fill_between_kw is None:
                fill_between_kw = {}
            default_fill_between_kw = {"alpha": 0.5}
            fill_between_kw = {**default_fill_between_kw, **fill_between_kw}

            self.fill_between_ = []
            for line_label, score in scores.items():
                self.fill_between_.append(
                    ax.fill_between(
                        x_data,
                        score.mean(axis=1) - score.std(axis=1),
                        score.mean(axis=1) + score.std(axis=1),
                        **fill_between_kw,
                    )
                )

        score_name = self.score_name if score_name is None else score_name

        ax.legend()

        # TODO(1.5): to be removed
        if log_scale != "deprecated":
            warnings.warn(
                (
                    "The `log_scale` parameter is deprecated as of version 1.3 "
                    "and will be removed in 1.5. You can use display.ax_.set_xscale "
                    "and display.ax_.set_yscale instead."
                ),
                FutureWarning,
            )
            xscale = "log" if log_scale else "linear"
        else:
            # We found that a ratio, smaller or bigger than 5, between the largest and
            # smallest gap of the x values is a good indicator to choose between linear
            # and log scale.
            if _interval_max_min_ratio(x_data) > 5:
                xscale = "symlog" if x_data.min() <= 0 else "log"
            else:
                xscale = "linear"
        ax.set_xscale(xscale)
        ax.set_ylabel(f"{score_name}")

        self.ax_ = ax
        self.figure_ = ax.figure


class LearningCurveDisplay(_BaseCurveDisplay):
    """Learning Curve visualization.

    It is recommended to use
    :meth:`~sklearn.model_selection.LearningCurveDisplay.from_estimator` to
    create a :class:`~sklearn.model_selection.LearningCurveDisplay` instance.
    All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>` for general information
    about the visualization API and
    :ref:`detailed documentation <learning_curve>` regarding the learning
    curve visualization.

    .. versionadded:: 1.2

    Parameters
    ----------
    train_sizes : ndarray of shape (n_unique_ticks,)
        Numbers of training examples that has been used to generate the
        learning curve.

    train_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on test set.

    score_name : str, default=None
        The name of the score used in `learning_curve`. It will override the name
        inferred from the `scoring` parameter. If `score` is `None`, we use `"Score"` if
        `negate_score` is `False` and `"Negative score"` otherwise. If `scoring` is a
        string or a callable, we infer the name. We replace `_` by spaces and capitalize
        the first letter. We remove `neg_` and replace it by `"Negative"` if
        `negate_score` is `False` or just remove it otherwise.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the learning curve.

    figure_ : matplotlib Figure
        Figure containing the learning curve.

    errorbar_ : list of matplotlib Artist or None
        When the `std_display_style` is `"errorbar"`, this is a list of
        `matplotlib.container.ErrorbarContainer` objects. If another style is
        used, `errorbar_` is `None`.

    lines_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.lines.Line2D` objects corresponding to the mean train and
        test scores. If another style is used, `line_` is `None`.

    fill_between_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.collections.PolyCollection` objects. If another style is
        used, `fill_between_` is `None`.

    See Also
    --------
    sklearn.model_selection.learning_curve : Compute the learning curve.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import LearningCurveDisplay, learning_curve
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> tree = DecisionTreeClassifier(random_state=0)
    >>> train_sizes, train_scores, test_scores = learning_curve(
    ...     tree, X, y)
    >>> display = LearningCurveDisplay(train_sizes=train_sizes,
    ...     train_scores=train_scores, test_scores=test_scores, score_name="Score")
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, *, train_sizes, train_scores, test_scores, score_name=None):
        self.train_sizes = train_sizes
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.score_name = score_name

    def plot(
        self,
        ax=None,
        *,
        negate_score=False,
        score_name=None,
        score_type="both",
        log_scale="deprecated",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        """Plot visualization.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        negate_score : bool, default=False
            Whether or not to negate the scores obtained through
            :func:`~sklearn.model_selection.learning_curve`. This is
            particularly useful when using the error denoted by `neg_*` in
            `scikit-learn`.

        score_name : str, default=None
            The name of the score used to decorate the y-axis of the plot. It will
            override the name inferred from the `scoring` parameter. If `score` is
            `None`, we use `"Score"` if `negate_score` is `False` and `"Negative score"`
            otherwise. If `scoring` is a string or a callable, we infer the name. We
            replace `_` by spaces and capitalize the first letter. We remove `neg_` and
            replace it by `"Negative"` if `negate_score` is
            `False` or just remove it otherwise.

        score_type : {"test", "train", "both"}, default="both"
            The type of score to plot. Can be one of `"test"`, `"train"`, or
            `"both"`.

        log_scale : bool, default="deprecated"
            Whether or not to use a logarithmic scale for the x-axis.

            .. deprecated:: 1.3
               `log_scale` is deprecated in 1.3 and will be removed in 1.5.
               Use `display.ax_.set_xscale` and `display.ax_.set_yscale` instead.

        std_display_style : {"errorbar", "fill_between"} or None, default="fill_between"
            The style used to display the score standard deviation around the
            mean score. If None, no standard deviation representation is
            displayed.

        line_kw : dict, default=None
            Additional keyword arguments passed to the `plt.plot` used to draw
            the mean score.

        fill_between_kw : dict, default=None
            Additional keyword arguments passed to the `plt.fill_between` used
            to draw the score standard deviation.

        errorbar_kw : dict, default=None
            Additional keyword arguments passed to the `plt.errorbar` used to
            draw mean score and standard deviation score.

        Returns
        -------
        display : :class:`~sklearn.model_selection.LearningCurveDisplay`
            Object that stores computed values.
        """
        self._plot_curve(
            self.train_sizes,
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            log_scale=log_scale,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )
        self.ax_.set_xlabel("Number of samples in the training set")
        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        groups=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=None,
        scoring=None,
        exploit_incremental_learning=False,
        n_jobs=None,
        pre_dispatch="all",
        verbose=0,
        shuffle=False,
        random_state=None,
        error_score=np.nan,
        fit_params=None,
        ax=None,
        negate_score=False,
        score_name=None,
        score_type="both",
        log_scale="deprecated",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        """Create a learning curve display from an estimator.

        Read more in the :ref:`User Guide <visualizations>` for general
        information about the visualization API and :ref:`detailed
        documentation <learning_curve>` regarding the learning curve
        visualization.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`GroupKFold`).

        train_sizes : array-like of shape (n_ticks,), \
                default=np.linspace(0.1, 1.0, 5)
            Relative or absolute numbers of training examples that will be used
            to generate the learning curve. If the dtype is float, it is
            regarded as a fraction of the maximum size of the training set
            (that is determined by the selected validation method), i.e. it has
            to be within (0, 1]. Otherwise it is interpreted as absolute sizes
            of the training sets. Note that for classification the number of
            samples usually have to be big enough to contain at least one
            sample from each class.

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For int/None inputs, if the estimator is a classifier and `y` is
            either binary or multiclass,
            :class:`~sklearn.model_selection.StratifiedKFold` is used. In all
            other cases, :class:`~sklearn.model_selection.KFold` is used. These
            splitters are instantiated with `shuffle=False` so the splits will
            be the same across calls.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        scoring : str or callable, default=None
            A string (see :ref:`scoring_parameter`) or
            a scorer callable object / function with signature
            `scorer(estimator, X, y)` (see :ref:`scoring`).

        exploit_incremental_learning : bool, default=False
            If the estimator supports incremental learning, this will be
            used to speed up fitting for different training set sizes.

        n_jobs : int, default=None
            Number of jobs to run in parallel. Training the estimator and
            computing the score are parallelized over the different training
            and test sets. `None` means 1 unless in a
            :obj:`joblib.parallel_backend` context. `-1` means using all
            processors. See :term:`Glossary <n_jobs>` for more details.

        pre_dispatch : int or str, default='all'
            Number of predispatched jobs for parallel execution (default is
            all). The option can reduce the allocated memory. The str can
            be an expression like '2*n_jobs'.

        verbose : int, default=0
            Controls the verbosity: the higher, the more messages.

        shuffle : bool, default=False
            Whether to shuffle training data before taking prefixes of it
            based on`train_sizes`.

        random_state : int, RandomState instance or None, default=None
            Used when `shuffle` is True. Pass an int for reproducible
            output across multiple function calls.
            See :term:`Glossary <random_state>`.

        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator
            fitting. If set to 'raise', the error is raised. If a numeric value
            is given, FitFailedWarning is raised.

        fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        negate_score : bool, default=False
            Whether or not to negate the scores obtained through
            :func:`~sklearn.model_selection.learning_curve`. This is
            particularly useful when using the error denoted by `neg_*` in
            `scikit-learn`.

        score_name : str, default=None
            The name of the score used to decorate the y-axis of the plot. It will
            override the name inferred from the `scoring` parameter. If `score` is
            `None`, we use `"Score"` if `negate_score` is `False` and `"Negative score"`
            otherwise. If `scoring` is a string or a callable, we infer the name. We
            replace `_` by spaces and capitalize the first letter. We remove `neg_` and
            replace it by `"Negative"` if `negate_score` is
            `False` or just remove it otherwise.

        score_type : {"test", "train", "both"}, default="both"
            The type of score to plot. Can be one of `"test"`, `"train"`, or
            `"both"`.

        log_scale : bool, default="deprecated"
            Whether or not to use a logarithmic scale for the x-axis.

            .. deprecated:: 1.3
               `log_scale` is deprecated in 1.3 and will be removed in 1.5.
               Use `display.ax_.xscale` and `display.ax_.yscale` instead.

        std_display_style : {"errorbar", "fill_between"} or None, default="fill_between"
            The style used to display the score standard deviation around the
            mean score. If `None`, no representation of the standard deviation
            is displayed.

        line_kw : dict, default=None
            Additional keyword arguments passed to the `plt.plot` used to draw
            the mean score.

        fill_between_kw : dict, default=None
            Additional keyword arguments passed to the `plt.fill_between` used
            to draw the score standard deviation.

        errorbar_kw : dict, default=None
            Additional keyword arguments passed to the `plt.errorbar` used to
            draw mean score and standard deviation score.

        Returns
        -------
        display : :class:`~sklearn.model_selection.LearningCurveDisplay`
            Object that stores computed values.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import LearningCurveDisplay
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> tree = DecisionTreeClassifier(random_state=0)
        >>> LearningCurveDisplay.from_estimator(tree, X, y)
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_estimator")

        score_name = _validate_score_name(score_name, scoring, negate_score)

        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            groups=groups,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            exploit_incremental_learning=exploit_incremental_learning,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
            shuffle=shuffle,
            random_state=random_state,
            error_score=error_score,
            return_times=False,
            fit_params=fit_params,
        )

        viz = cls(
            train_sizes=train_sizes,
            train_scores=train_scores,
            test_scores=test_scores,
            score_name=score_name,
        )
        return viz.plot(
            ax=ax,
            negate_score=negate_score,
            score_type=score_type,
            log_scale=log_scale,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )


class ValidationCurveDisplay(_BaseCurveDisplay):
    """Validation Curve visualization.

    It is recommended to use
    :meth:`~sklearn.model_selection.ValidationCurveDisplay.from_estimator` to
    create a :class:`~sklearn.model_selection.ValidationCurveDisplay` instance.
    All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>` for general information
    about the visualization API and :ref:`detailed documentation
    <validation_curve>` regarding the validation curve visualization.

    .. versionadded:: 1.3

    Parameters
    ----------
    param_name : str
        Name of the parameter that has been varied.

    param_range : array-like of shape (n_ticks,)
        The values of the parameter that have been evaluated.

    train_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on test set.

    score_name : str, default=None
        The name of the score used in `validation_curve`. It will override the name
        inferred from the `scoring` parameter. If `score` is `None`, we use `"Score"` if
        `negate_score` is `False` and `"Negative score"` otherwise. If `scoring` is a
        string or a callable, we infer the name. We replace `_` by spaces and capitalize
        the first letter. We remove `neg_` and replace it by `"Negative"` if
        `negate_score` is `False` or just remove it otherwise.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the validation curve.

    figure_ : matplotlib Figure
        Figure containing the validation curve.

    errorbar_ : list of matplotlib Artist or None
        When the `std_display_style` is `"errorbar"`, this is a list of
        `matplotlib.container.ErrorbarContainer` objects. If another style is
        used, `errorbar_` is `None`.

    lines_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.lines.Line2D` objects corresponding to the mean train and
        test scores. If another style is used, `line_` is `None`.

    fill_between_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.collections.PolyCollection` objects. If another style is
        used, `fill_between_` is `None`.

    See Also
    --------
    sklearn.model_selection.validation_curve : Compute the validation curve.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import ValidationCurveDisplay, validation_curve
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=1_000, random_state=0)
    >>> logistic_regression = LogisticRegression()
    >>> param_name, param_range = "C", np.logspace(-8, 3, 10)
    >>> train_scores, test_scores = validation_curve(
    ...     logistic_regression, X, y, param_name=param_name, param_range=param_range
    ... )
    >>> display = ValidationCurveDisplay(
    ...     param_name=param_name, param_range=param_range,
    ...     train_scores=train_scores, test_scores=test_scores, score_name="Score"
    ... )
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(
        self, *, param_name, param_range, train_scores, test_scores, score_name=None
    ):
        self.param_name = param_name
        self.param_range = param_range
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.score_name = score_name

    def plot(
        self,
        ax=None,
        *,
        negate_score=False,
        score_name=None,
        score_type="both",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        """Plot visualization.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        negate_score : bool, default=False
            Whether or not to negate the scores obtained through
            :func:`~sklearn.model_selection.validation_curve`. This is
            particularly useful when using the error denoted by `neg_*` in
            `scikit-learn`.

        score_name : str, default=None
            The name of the score used to decorate the y-axis of the plot. It will
            override the name inferred from the `scoring` parameter. If `score` is
            `None`, we use `"Score"` if `negate_score` is `False` and `"Negative score"`
            otherwise. If `scoring` is a string or a callable, we infer the name. We
            replace `_` by spaces and capitalize the first letter. We remove `neg_` and
            replace it by `"Negative"` if `negate_score` is
            `False` or just remove it otherwise.

        score_type : {"test", "train", "both"}, default="both"
            The type of score to plot. Can be one of `"test"`, `"train"`, or
            `"both"`.

        std_display_style : {"errorbar", "fill_between"} or None, default="fill_between"
            The style used to display the score standard deviation around the
            mean score. If None, no standard deviation representation is
            displayed.

        line_kw : dict, default=None
            Additional keyword arguments passed to the `plt.plot` used to draw
            the mean score.

        fill_between_kw : dict, default=None
            Additional keyword arguments passed to the `plt.fill_between` used
            to draw the score standard deviation.

        errorbar_kw : dict, default=None
            Additional keyword arguments passed to the `plt.errorbar` used to
            draw mean score and standard deviation score.

        Returns
        -------
        display : :class:`~sklearn.model_selection.ValidationCurveDisplay`
            Object that stores computed values.
        """
        self._plot_curve(
            self.param_range,
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            log_scale="deprecated",
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )
        self.ax_.set_xlabel(f"{self.param_name}")
        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        param_name,
        param_range,
        groups=None,
        cv=None,
        scoring=None,
        n_jobs=None,
        pre_dispatch="all",
        verbose=0,
        error_score=np.nan,
        fit_params=None,
        ax=None,
        negate_score=False,
        score_name=None,
        score_type="both",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        """Create a validation curve display from an estimator.

        Read more in the :ref:`User Guide <visualizations>` for general
        information about the visualization API and :ref:`detailed
        documentation <validation_curve>` regarding the validation curve
        visualization.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        param_name : str
            Name of the parameter that will be varied.

        param_range : array-like of shape (n_values,)
            The values of the parameter that will be evaluated.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`GroupKFold`).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For int/None inputs, if the estimator is a classifier and `y` is
            either binary or multiclass,
            :class:`~sklearn.model_selection.StratifiedKFold` is used. In all
            other cases, :class:`~sklearn.model_selection.KFold` is used. These
            splitters are instantiated with `shuffle=False` so the splits will
            be the same across calls.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        scoring : str or callable, default=None
            A string (see :ref:`scoring_parameter`) or
            a scorer callable object / function with signature
            `scorer(estimator, X, y)` (see :ref:`scoring`).

        n_jobs : int, default=None
            Number of jobs to run in parallel. Training the estimator and
            computing the score are parallelized over the different training
            and test sets. `None` means 1 unless in a
            :obj:`joblib.parallel_backend` context. `-1` means using all
            processors. See :term:`Glossary <n_jobs>` for more details.

        pre_dispatch : int or str, default='all'
            Number of predispatched jobs for parallel execution (default is
            all). The option can reduce the allocated memory. The str can
            be an expression like '2*n_jobs'.

        verbose : int, default=0
            Controls the verbosity: the higher, the more messages.

        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator
            fitting. If set to 'raise', the error is raised. If a numeric value
            is given, FitFailedWarning is raised.

        fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        negate_score : bool, default=False
            Whether or not to negate the scores obtained through
            :func:`~sklearn.model_selection.validation_curve`. This is
            particularly useful when using the error denoted by `neg_*` in
            `scikit-learn`.

        score_name : str, default=None
            The name of the score used to decorate the y-axis of the plot. It will
            override the name inferred from the `scoring` parameter. If `score` is
            `None`, we use `"Score"` if `negate_score` is `False` and `"Negative score"`
            otherwise. If `scoring` is a string or a callable, we infer the name. We
            replace `_` by spaces and capitalize the first letter. We remove `neg_` and
            replace it by `"Negative"` if `negate_score` is
            `False` or just remove it otherwise.

        score_type : {"test", "train", "both"}, default="both"
            The type of score to plot. Can be one of `"test"`, `"train"`, or
            `"both"`.

        std_display_style : {"errorbar", "fill_between"} or None, default="fill_between"
            The style used to display the score standard deviation around the
            mean score. If `None`, no representation of the standard deviation
            is displayed.

        line_kw : dict, default=None
            Additional keyword arguments passed to the `plt.plot` used to draw
            the mean score.

        fill_between_kw : dict, default=None
            Additional keyword arguments passed to the `plt.fill_between` used
            to draw the score standard deviation.

        errorbar_kw : dict, default=None
            Additional keyword arguments passed to the `plt.errorbar` used to
            draw mean score and standard deviation score.

        Returns
        -------
        display : :class:`~sklearn.model_selection.ValidationCurveDisplay`
            Object that stores computed values.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import ValidationCurveDisplay
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(n_samples=1_000, random_state=0)
        >>> logistic_regression = LogisticRegression()
        >>> param_name, param_range = "C", np.logspace(-8, 3, 10)
        >>> ValidationCurveDisplay.from_estimator(
        ...     logistic_regression, X, y, param_name=param_name,
        ...     param_range=param_range,
        ... )
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_estimator")

        score_name = _validate_score_name(score_name, scoring, negate_score)

        train_scores, test_scores = validation_curve(
            estimator,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
            error_score=error_score,
            fit_params=fit_params,
        )

        viz = cls(
            param_name=param_name,
            param_range=np.array(param_range, copy=False),
            train_scores=train_scores,
            test_scores=test_scores,
            score_name=score_name,
        )
        return viz.plot(
            ax=ax,
            negate_score=negate_score,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )
