"""Transform a dataset into an imbalanced dataset."""

# Authors: Dayvid Oliveira
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter
from collections.abc import Mapping

from ..under_sampling import RandomUnderSampler
from ..utils import check_sampling_strategy
from ..utils._param_validation import validate_params


@validate_params(
    {
        "X": ["array-like"],
        "y": ["array-like"],
        "sampling_strategy": [Mapping, callable, None],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def make_imbalance(
    X, y, *, sampling_strategy=None, random_state=None, verbose=False, **kwargs
):
    """Turn a dataset into an imbalanced dataset with a specific sampling strategy.

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Read more in the :ref:`User Guide <make_imbalanced>`.

    Parameters
    ----------
    X : {array-like, dataframe} of shape (n_samples, n_features)
        Matrix containing the data to be imbalanced.

    y : array-like of shape (n_samples,)
        Corresponding label for each sample in X.

    sampling_strategy : dict or callable,
        Ratio to use for resampling the data set.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    verbose : bool, default=False
        Show information regarding the sampling.

    **kwargs : dict
        Dictionary of additional keyword arguments to pass to
        ``sampling_strategy``.

    Returns
    -------
    X_resampled : {ndarray, dataframe} of shape (n_samples_new, n_features)
        The array containing the imbalanced data.

    y_resampled : ndarray of shape (n_samples_new)
        The corresponding label of `X_resampled`.

    Notes
    -----
    See
    :ref:`sphx_glr_auto_examples_applications_plot_multi_class_under_sampling.py`,
    :ref:`sphx_glr_auto_examples_datasets_plot_make_imbalance.py`, and
    :ref:`sphx_glr_auto_examples_api_plot_sampling_strategy_usage.py`.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import load_iris
    >>> from imblearn.datasets import make_imbalance

    >>> data = load_iris()
    >>> X, y = data.data, data.target
    >>> print(f'Distribution before imbalancing: {Counter(y)}')
    Distribution before imbalancing: Counter({0: 50, 1: 50, 2: 50})
    >>> X_res, y_res = make_imbalance(X, y,
    ...                               sampling_strategy={0: 10, 1: 20, 2: 30},
    ...                               random_state=42)
    >>> print(f'Distribution after imbalancing: {Counter(y_res)}')
    Distribution after imbalancing: Counter({2: 30, 1: 20, 0: 10})
    """
    target_stats = Counter(y)
    # restrict ratio to be a dict or a callable
    if isinstance(sampling_strategy, Mapping) or callable(sampling_strategy):
        sampling_strategy_ = check_sampling_strategy(
            sampling_strategy, y, "under-sampling", **kwargs
        )

    if verbose:
        print(f"The original target distribution in the dataset is: {target_stats}")
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy_,
        replacement=False,
        random_state=random_state,
    )
    X_resampled, y_resampled = rus.fit_resample(X, y)
    if verbose:
        print(f"Make the dataset imbalanced: {Counter(y_resampled)}")

    return X_resampled, y_resampled
