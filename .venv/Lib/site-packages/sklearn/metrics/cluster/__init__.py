"""
The :mod:`sklearn.metrics.cluster` submodule contains evaluation metrics for
cluster analysis results. There are two forms of evaluation:

- supervised, which uses a ground truth class values for each sample.
- unsupervised, which does not and measures the 'quality' of the model itself.
"""
from ._bicluster import consensus_score
from ._supervised import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    contingency_matrix,
    entropy,
    expected_mutual_information,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    rand_score,
    v_measure_score,
)
from ._unsupervised import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

__all__ = [
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "adjusted_rand_score",
    "rand_score",
    "completeness_score",
    "pair_confusion_matrix",
    "contingency_matrix",
    "expected_mutual_information",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "mutual_info_score",
    "v_measure_score",
    "fowlkes_mallows_score",
    "entropy",
    "silhouette_samples",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "consensus_score",
]
