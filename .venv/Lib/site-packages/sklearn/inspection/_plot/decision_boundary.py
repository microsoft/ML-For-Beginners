import numpy as np

from ...base import is_regressor
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_matplotlib_support
from ...utils._response import _get_response_values
from ...utils.validation import (
    _is_arraylike_not_scalar,
    _num_features,
    check_is_fitted,
)


def _check_boundary_response_method(estimator, response_method, class_of_interest):
    """Validate the response methods to be used with the fitted estimator.

    Parameters
    ----------
    estimator : object
        Fitted estimator to check.

    response_method : {'auto', 'predict_proba', 'decision_function', 'predict'}
        Specifies whether to use :term:`predict_proba`,
        :term:`decision_function`, :term:`predict` as the target response.
        If set to 'auto', the response method is tried in the following order:
        :term:`decision_function`, :term:`predict_proba`, :term:`predict`.

    class_of_interest : int, float, bool, str or None
        The class considered when plotting the decision. If the label is specified, it
        is then possible to plot the decision boundary in multiclass settings.

        .. versionadded:: 1.4

    Returns
    -------
    prediction_method : list of str or str
        The name or list of names of the response methods to use.
    """
    has_classes = hasattr(estimator, "classes_")
    if has_classes and _is_arraylike_not_scalar(estimator.classes_[0]):
        msg = "Multi-label and multi-output multi-class classifiers are not supported"
        raise ValueError(msg)

    if has_classes and len(estimator.classes_) > 2:
        if response_method not in {"auto", "predict"} and class_of_interest is None:
            msg = (
                "Multiclass classifiers are only supported when `response_method` is "
                "'predict' or 'auto'. Else you must provide `class_of_interest` to "
                "plot the decision boundary of a specific class."
            )
            raise ValueError(msg)
        prediction_method = "predict" if response_method == "auto" else response_method
    elif response_method == "auto":
        if is_regressor(estimator):
            prediction_method = "predict"
        else:
            prediction_method = ["decision_function", "predict_proba", "predict"]
    else:
        prediction_method = response_method

    return prediction_method


class DecisionBoundaryDisplay:
    """Decisions boundary visualization.

    It is recommended to use
    :func:`~sklearn.inspection.DecisionBoundaryDisplay.from_estimator`
    to create a :class:`DecisionBoundaryDisplay`. All parameters are stored as
    attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    xx0 : ndarray of shape (grid_resolution, grid_resolution)
        First output of :func:`meshgrid <numpy.meshgrid>`.

    xx1 : ndarray of shape (grid_resolution, grid_resolution)
        Second output of :func:`meshgrid <numpy.meshgrid>`.

    response : ndarray of shape (grid_resolution, grid_resolution)
        Values of the response function.

    xlabel : str, default=None
        Default label to place on x axis.

    ylabel : str, default=None
        Default label to place on y axis.

    Attributes
    ----------
    surface_ : matplotlib `QuadContourSet` or `QuadMesh`
        If `plot_method` is 'contour' or 'contourf', `surface_` is a
        :class:`QuadContourSet <matplotlib.contour.QuadContourSet>`. If
        `plot_method` is 'pcolormesh', `surface_` is a
        :class:`QuadMesh <matplotlib.collections.QuadMesh>`.

    ax_ : matplotlib Axes
        Axes with decision boundary.

    figure_ : matplotlib Figure
        Figure containing the decision boundary.

    See Also
    --------
    DecisionBoundaryDisplay.from_estimator : Plot decision boundary given an estimator.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.inspection import DecisionBoundaryDisplay
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> iris = load_iris()
    >>> feature_1, feature_2 = np.meshgrid(
    ...     np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()),
    ...     np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max())
    ... )
    >>> grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    >>> tree = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)
    >>> y_pred = np.reshape(tree.predict(grid), feature_1.shape)
    >>> display = DecisionBoundaryDisplay(
    ...     xx0=feature_1, xx1=feature_2, response=y_pred
    ... )
    >>> display.plot()
    <...>
    >>> display.ax_.scatter(
    ...     iris.data[:, 0], iris.data[:, 1], c=iris.target, edgecolor="black"
    ... )
    <...>
    >>> plt.show()
    """

    def __init__(self, *, xx0, xx1, response, xlabel=None, ylabel=None):
        self.xx0 = xx0
        self.xx1 = xx1
        self.response = response
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, plot_method="contourf", ax=None, xlabel=None, ylabel=None, **kwargs):
        """Plot visualization.

        Parameters
        ----------
        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'
            Plotting method to call when plotting the response. Please refer
            to the following matplotlib documentation for details:
            :func:`contourf <matplotlib.pyplot.contourf>`,
            :func:`contour <matplotlib.pyplot.contour>`,
            :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`.

        ax : Matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        xlabel : str, default=None
            Overwrite the x-axis label.

        ylabel : str, default=None
            Overwrite the y-axis label.

        **kwargs : dict
            Additional keyword arguments to be passed to the `plot_method`.

        Returns
        -------
        display: :class:`~sklearn.inspection.DecisionBoundaryDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support("DecisionBoundaryDisplay.plot")
        import matplotlib.pyplot as plt  # noqa

        if plot_method not in ("contourf", "contour", "pcolormesh"):
            raise ValueError(
                "plot_method must be 'contourf', 'contour', or 'pcolormesh'"
            )

        if ax is None:
            _, ax = plt.subplots()

        plot_func = getattr(ax, plot_method)
        self.surface_ = plot_func(self.xx0, self.xx1, self.response, **kwargs)

        if xlabel is not None or not ax.get_xlabel():
            xlabel = self.xlabel if xlabel is None else xlabel
            ax.set_xlabel(xlabel)
        if ylabel is not None or not ax.get_ylabel():
            ylabel = self.ylabel if ylabel is None else ylabel
            ax.set_ylabel(ylabel)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        *,
        grid_resolution=100,
        eps=1.0,
        plot_method="contourf",
        response_method="auto",
        class_of_interest=None,
        xlabel=None,
        ylabel=None,
        ax=None,
        **kwargs,
    ):
        """Plot decision boundary given an estimator.

        Read more in the :ref:`User Guide <visualizations>`.

        Parameters
        ----------
        estimator : object
            Trained estimator used to plot the decision boundary.

        X : {array-like, sparse matrix, dataframe} of shape (n_samples, 2)
            Input data that should be only 2-dimensional.

        grid_resolution : int, default=100
            Number of grid points to use for plotting decision boundary.
            Higher values will make the plot look nicer but be slower to
            render.

        eps : float, default=1.0
            Extends the minimum and maximum values of X for evaluating the
            response function.

        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'
            Plotting method to call when plotting the response. Please refer
            to the following matplotlib documentation for details:
            :func:`contourf <matplotlib.pyplot.contourf>`,
            :func:`contour <matplotlib.pyplot.contour>`,
            :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`.

        response_method : {'auto', 'predict_proba', 'decision_function', \
                'predict'}, default='auto'
            Specifies whether to use :term:`predict_proba`,
            :term:`decision_function`, :term:`predict` as the target response.
            If set to 'auto', the response method is tried in the following order:
            :term:`decision_function`, :term:`predict_proba`, :term:`predict`.
            For multiclass problems, :term:`predict` is selected when
            `response_method="auto"`.

        class_of_interest : int, float, bool or str, default=None
            The class considered when plotting the decision. If None,
            `estimator.classes_[1]` is considered as the positive class
            for binary classifiers. For multiclass classifiers, passing
            an explicit value for `class_of_interest` is mandatory.

            .. versionadded:: 1.4

        xlabel : str, default=None
            The label used for the x-axis. If `None`, an attempt is made to
            extract a label from `X` if it is a dataframe, otherwise an empty
            string is used.

        ylabel : str, default=None
            The label used for the y-axis. If `None`, an attempt is made to
            extract a label from `X` if it is a dataframe, otherwise an empty
            string is used.

        ax : Matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Additional keyword arguments to be passed to the
            `plot_method`.

        Returns
        -------
        display : :class:`~sklearn.inspection.DecisionBoundaryDisplay`
            Object that stores the result.

        See Also
        --------
        DecisionBoundaryDisplay : Decision boundary visualization.
        sklearn.metrics.ConfusionMatrixDisplay.from_estimator : Plot the
            confusion matrix given an estimator, the data, and the label.
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions : Plot the
            confusion matrix given the true and predicted labels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.inspection import DecisionBoundaryDisplay
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]
        >>> classifier = LogisticRegression().fit(X, iris.target)
        >>> disp = DecisionBoundaryDisplay.from_estimator(
        ...     classifier, X, response_method="predict",
        ...     xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],
        ...     alpha=0.5,
        ... )
        >>> disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_estimator")
        check_is_fitted(estimator)

        if not grid_resolution > 1:
            raise ValueError(
                "grid_resolution must be greater than 1. Got"
                f" {grid_resolution} instead."
            )

        if not eps >= 0:
            raise ValueError(
                f"eps must be greater than or equal to 0. Got {eps} instead."
            )

        possible_plot_methods = ("contourf", "contour", "pcolormesh")
        if plot_method not in possible_plot_methods:
            available_methods = ", ".join(possible_plot_methods)
            raise ValueError(
                f"plot_method must be one of {available_methods}. "
                f"Got {plot_method} instead."
            )

        num_features = _num_features(X)
        if num_features != 2:
            raise ValueError(
                f"n_features must be equal to 2. Got {num_features} instead."
            )

        x0, x1 = _safe_indexing(X, 0, axis=1), _safe_indexing(X, 1, axis=1)

        x0_min, x0_max = x0.min() - eps, x0.max() + eps
        x1_min, x1_max = x1.min() - eps, x1.max() + eps

        xx0, xx1 = np.meshgrid(
            np.linspace(x0_min, x0_max, grid_resolution),
            np.linspace(x1_min, x1_max, grid_resolution),
        )
        if hasattr(X, "iloc"):
            # we need to preserve the feature names and therefore get an empty dataframe
            X_grid = X.iloc[[], :].copy()
            X_grid.iloc[:, 0] = xx0.ravel()
            X_grid.iloc[:, 1] = xx1.ravel()
        else:
            X_grid = np.c_[xx0.ravel(), xx1.ravel()]

        prediction_method = _check_boundary_response_method(
            estimator, response_method, class_of_interest
        )
        try:
            response, _, response_method_used = _get_response_values(
                estimator,
                X_grid,
                response_method=prediction_method,
                pos_label=class_of_interest,
                return_response_method_used=True,
            )
        except ValueError as exc:
            if "is not a valid label" in str(exc):
                # re-raise a more informative error message since `pos_label` is unknown
                # to our user when interacting with
                # `DecisionBoundaryDisplay.from_estimator`
                raise ValueError(
                    f"class_of_interest={class_of_interest} is not a valid label: It "
                    f"should be one of {estimator.classes_}"
                ) from exc
            raise

        # convert classes predictions into integers
        if response_method_used == "predict" and hasattr(estimator, "classes_"):
            encoder = LabelEncoder()
            encoder.classes_ = estimator.classes_
            response = encoder.transform(response)

        if response.ndim != 1:
            if is_regressor(estimator):
                raise ValueError("Multi-output regressors are not supported")

            # For the multiclass case, `_get_response_values` returns the response
            # as-is. Thus, we have a column per class and we need to select the column
            # corresponding to the positive class.
            col_idx = np.flatnonzero(estimator.classes_ == class_of_interest)[0]
            response = response[:, col_idx]

        if xlabel is None:
            xlabel = X.columns[0] if hasattr(X, "columns") else ""

        if ylabel is None:
            ylabel = X.columns[1] if hasattr(X, "columns") else ""

        display = cls(
            xx0=xx0,
            xx1=xx1,
            response=response.reshape(xx0.shape),
            xlabel=xlabel,
            ylabel=ylabel,
        )
        return display.plot(ax=ax, plot_method=plot_method, **kwargs)
