"""Implement generators for ``tensorflow`` which will balance the data."""

from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.utils import _safe_indexing, check_random_state

from ..under_sampling import RandomUnderSampler
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring


@Substitution(random_state=_random_state_docstring)
def balanced_batch_generator(
    X,
    y,
    *,
    sample_weight=None,
    sampler=None,
    batch_size=32,
    keep_sparse=False,
    random_state=None,
):
    """Create a balanced batch generator to train tensorflow model.

    Returns a generator --- as well as the number of step per epoch --- to
    iterate to get the mini-batches. The sampler defines the sampling strategy
    used to balance the dataset ahead of creating the batch. The sampler should
    have an attribute ``sample_indices_``.

    .. versionadded:: 0.4

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original imbalanced dataset.

    y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Associated targets.

    sample_weight : ndarray of shape (n_samples,), default=None
        Sample weight.

    sampler : sampler object, default=None
        A sampler instance which has an attribute ``sample_indices_``.
        By default, the sampler used is a
        :class:`~imblearn.under_sampling.RandomUnderSampler`.

    batch_size : int, default=32
        Number of samples per gradient update.

    keep_sparse : bool, default=False
        Either or not to conserve or not the sparsity of the input ``X``. By
        default, the returned batches will be dense.

    {random_state}

    Returns
    -------
    generator : generator of tuple
        Generate batch of data. The tuple generated are either (X_batch,
        y_batch) or (X_batch, y_batch, sampler_weight_batch).

    steps_per_epoch : int
        The number of samples per epoch.
    """

    random_state = check_random_state(random_state)
    if sampler is None:
        sampler_ = RandomUnderSampler(random_state=random_state)
    else:
        sampler_ = clone(sampler)
    sampler_.fit_resample(X, y)
    if not hasattr(sampler_, "sample_indices_"):
        raise ValueError("'sampler' needs to have an attribute 'sample_indices_'.")
    indices = sampler_.sample_indices_
    # shuffle the indices since the sampler are packing them by class
    random_state.shuffle(indices)

    def generator(X, y, sample_weight, indices, batch_size):
        while True:
            for index in range(0, len(indices), batch_size):
                X_res = _safe_indexing(X, indices[index : index + batch_size])
                y_res = _safe_indexing(y, indices[index : index + batch_size])
                if issparse(X_res) and not keep_sparse:
                    X_res = X_res.toarray()
                if sample_weight is None:
                    yield X_res, y_res
                else:
                    sw_res = _safe_indexing(
                        sample_weight, indices[index : index + batch_size]
                    )
                    yield X_res, y_res, sw_res

    return (
        generator(X, y, sample_weight, indices, batch_size),
        int(indices.size // batch_size),
    )
