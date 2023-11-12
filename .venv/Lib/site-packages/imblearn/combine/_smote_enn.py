"""Class to perform over-sampling using SMOTE and cleaning using ENN."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers

from sklearn.base import clone
from sklearn.utils import check_X_y

from ..base import BaseSampler
from ..over_sampling import SMOTE
from ..over_sampling.base import BaseOverSampler
from ..under_sampling import EditedNearestNeighbours
from ..utils import Substitution, check_target_type
from ..utils._docstring import _n_jobs_docstring, _random_state_docstring


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTEENN(BaseSampler):
    """Over-sampling using SMOTE and cleaning using ENN.

    Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.

    Read more in the :ref:`User Guide <combine>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    smote : sampler object, default=None
        The :class:`~imblearn.over_sampling.SMOTE` object to use. If not given,
        a :class:`~imblearn.over_sampling.SMOTE` object with default parameters
        will be given.

    enn : sampler object, default=None
        The :class:`~imblearn.under_sampling.EditedNearestNeighbours` object
        to use. If not given, a
        :class:`~imblearn.under_sampling.EditedNearestNeighbours` object with
        sampling strategy='all' will be given.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    smote_ : sampler object
        The validated :class:`~imblearn.over_sampling.SMOTE` instance.

    enn_ : sampler object
        The validated :class:`~imblearn.under_sampling.EditedNearestNeighbours`
        instance.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTETomek : Over-sample using SMOTE followed by under-sampling removing
        the Tomek's links.

    Notes
    -----
    The method is presented in [1]_.

    Supports multi-class resampling. Refer to SMOTE and ENN regarding the
    scheme which used.

    References
    ----------
    .. [1] G. Batista, R. C. Prati, M. C. Monard. "A study of the behavior of
       several methods for balancing machine learning training data," ACM
       Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.combine import SMOTEENN
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sme = SMOTEENN(random_state=42)
    >>> X_res, y_res = sme.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 881}})
    """

    _sampling_type = "over-sampling"

    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "smote": [SMOTE, None],
        "enn": [EditedNearestNeighbours, None],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        smote=None,
        enn=None,
        n_jobs=None,
    ):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.enn = enn
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        "Private function to validate SMOTE and ENN objects"
        if self.smote is not None:
            self.smote_ = clone(self.smote)
        else:
            self.smote_ = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        if self.enn is not None:
            self.enn_ = clone(self.enn)
        else:
            self.enn_ = EditedNearestNeighbours(
                sampling_strategy="all", n_jobs=self.n_jobs
            )

    def _fit_resample(self, X, y):
        self._validate_estimator()
        y = check_target_type(y)
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        self.sampling_strategy_ = self.sampling_strategy

        X_res, y_res = self.smote_.fit_resample(X, y)
        return self.enn_.fit_resample(X_res, y_res)
