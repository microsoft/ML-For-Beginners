"""
The :mod:`imblearn.pipeline` module implements utilities to build a
composite estimator, as a chain of transforms, samples and estimators.
"""
# Adapted from scikit-learn

# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
#         Christos Aridas
#         Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: BSD
import joblib
from sklearn import pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if

from .base import _ParamsValidationMixin
from .utils._param_validation import HasMethods, validate_params

__all__ = ["Pipeline", "make_pipeline"]


class Pipeline(_ParamsValidationMixin, pipeline.Pipeline):
    """Pipeline of transforms and resamples with a final estimator.

    Sequentially apply a list of transforms, sampling, and a final estimator.
    Intermediate steps of the pipeline must be transformers or resamplers,
    that is, they must implement fit, transform and sample methods.
    The samplers are only applied during fit.
    The final estimator only needs to implement fit.
    The transformers and samplers in the pipeline can be cached using
    ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing
        fit/transform/fit_resample) that are chained, in the order in which
        they are chained, with the last object an estimator.

    memory : Instance of joblib.Memory or str, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during first step `fit` method.

    See Also
    --------
    make_pipeline : Helper function to make pipeline.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`

    .. warning::
       A surprising behaviour of the `imbalanced-learn` pipeline is that it
       breaks the `scikit-learn` contract where one expects
       `estimmator.fit_transform(X, y)` to be equivalent to
       `estimator.fit(X, y).transform(X)`.

       The semantic of `fit_resample` is to be applied only during the fit
       stage. Therefore, resampling will happen when calling `fit_transform`
       while it will only happen on the `fit` stage when calling `fit` and
       `transform` separately. Practically, `fit_transform` will lead to a
       resampled dataset while `fit` and `transform` will not.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split as tts
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.neighbors import KNeighborsClassifier as KNN
    >>> from sklearn.metrics import classification_report
    >>> from imblearn.over_sampling import SMOTE
    >>> from imblearn.pipeline import Pipeline
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print(f'Original dataset shape {Counter(y)}')
    Original dataset shape Counter({1: 900, 0: 100})
    >>> pca = PCA()
    >>> smt = SMOTE(random_state=42)
    >>> knn = KNN()
    >>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
    >>> X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
    >>> pipeline.fit(X_train, y_train)
    Pipeline(...)
    >>> y_hat = pipeline.predict(X_test)
    >>> print(classification_report(y_test, y_hat))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.87      1.00      0.93        26
               1       1.00      0.98      0.99       224
    <BLANKLINE>
        accuracy                           0.98       250
       macro avg       0.93      0.99      0.96       250
    weighted avg       0.99      0.98      0.98       250
    <BLANKLINE>
    """

    _parameter_constraints: dict = {
        "steps": "no_validation",  # validated in `_validate_steps`
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }

    # BaseEstimator interface

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None or t == "passthrough":
                continue
            if not (
                hasattr(t, "fit")
                or hasattr(t, "fit_transform")
                or hasattr(t, "fit_resample")
            ) or not (hasattr(t, "transform") or hasattr(t, "fit_resample")):
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or "
                    "fit_resample (but not both) or be a string 'passthrough' "
                    "'%s' (type %s) doesn't)" % (t, type(t))
                )

            if hasattr(t, "fit_resample") and (
                hasattr(t, "fit_transform") or hasattr(t, "transform")
            ):
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or "
                    "fit_resample."
                    " '%s' implements both)" % (t)
                )

            if isinstance(t, pipeline.Pipeline):
                raise TypeError(
                    "All intermediate steps of the chain should not be Pipelines"
                )

        # We allow last estimator to be None as an identity transformation
        if (
            estimator is not None
            and estimator != "passthrough"
            and not hasattr(estimator, "fit")
        ):
            raise TypeError(
                "Last step of Pipeline should implement fit or be "
                "the string 'passthrough'. '%s' (type %s) doesn't"
                % (estimator, type(estimator))
            )

    def _iter(self, with_final=True, filter_passthrough=True, filter_resample=True):
        """Generate (idx, (name, trans)) tuples from self.steps.

        When `filter_passthrough` is `True`, 'passthrough' and None
        transformers are filtered out. When `filter_resample` is `True`,
        estimator with a method `fit_resample` are filtered out.
        """
        it = super()._iter(with_final, filter_passthrough)
        if filter_resample:
            return filter(lambda x: not hasattr(x[-1], "fit_resample"), it)
        else:
            return it

    # Estimator interface

    def _fit(self, X, y=None, **fit_params_steps):
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        if self.memory is None or isinstance(self.memory, str):
            memory = joblib.Memory(location=self.memory, verbose=0)
        else:
            memory = self.memory

        fit_transform_one_cached = memory.cache(pipeline._fit_transform_one)
        fit_resample_one_cached = memory.cache(_fit_resample_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False, filter_resample=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            try:
                # joblib >= 0.12
                mem = memory.location
            except AttributeError:
                mem = memory.cachedir
            finally:
                cloned_transformer = clone(transformer) if mem else transformer

            # Fit or load from cache the current transformer
            if hasattr(cloned_transformer, "transform") or hasattr(
                cloned_transformer, "fit_transform"
            ):
                X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    None,
                    message_clsname="Pipeline",
                    message=self._log_message(step_idx),
                    **fit_params_steps[name],
                )
            elif hasattr(cloned_transformer, "fit_resample"):
                X, y, fitted_transformer = fit_resample_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    message_clsname="Pipeline",
                    message=self._log_message(step_idx),
                    **fit_params_steps[name],
                )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transforms/samplers one after the other and
        transform/sample the data, then fit the transformed/sampled
        data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator.
        """
        self._validate_params()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_transform on
        transformed data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        self._validate_params()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, yt, **fit_params_last_step)
            else:
                return last_step.fit(Xt, yt, **fit_params_last_step).transform(Xt)

    def fit_resample(self, X, y=None, **fit_params):
        """Fit the model and sample with the final estimator.

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_resample on transformed
        data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Transformed samples.

        yt : array-like of shape (n_samples, n_transformed_features)
            Transformed target.
        """
        self._validate_params()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_resample"):
                return last_step.fit_resample(Xt, yt, **fit_params_last_step)

    @available_if(pipeline._final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """Apply `fit_predict` of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target.
        """
        self._validate_params()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][-1].fit_predict(Xt, yt, **fit_params_last_step)
        return y_pred


def _fit_resample_one(sampler, X, y, message_clsname="", message=None, **fit_params):
    with _print_elapsed_time(message_clsname, message):
        X_res, y_res = sampler.fit_resample(X, y, **fit_params)

        return X_res, y_res, sampler


@validate_params(
    {"memory": [None, str, HasMethods(["cache"])], "verbose": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def make_pipeline(*steps, memory=None, verbose=False):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators
        A list of estimators.

    memory : None, str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline
        Returns an imbalanced-learn `Pipeline` instance that handles samplers.

    See Also
    --------
    imblearn.pipeline.Pipeline : Class for creating a pipeline of
        transforms with a final estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    return Pipeline(pipeline._name_estimators(steps), memory=memory, verbose=verbose)
