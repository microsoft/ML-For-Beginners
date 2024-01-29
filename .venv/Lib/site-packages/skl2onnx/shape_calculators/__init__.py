# SPDX-License-Identifier: Apache-2.0


# To register shape calculators for scikit-learn operators,
# import associated modules here.
from . import array_feature_extractor
from . import cast_op
from . import class_labels
from . import concat
from . import cross_decomposition
from . import dict_vectorizer
from . import ensemble_shapes
from . import feature_hasher
from . import flatten
from . import function_transformer
from . import gaussian_process
from . import grid_search_cv
from . import identity
from . import imputer
from . import isolation_forest
from . import kernel_pca
from . import k_bins_discretiser
from . import k_means
from . import label_binariser
from . import label_encoder
from . import linear_classifier
from . import linear_regressor
from . import local_outlier_factor
from . import mixture
from . import multioutput
from . import multiply
from . import nearest_neighbours
from . import one_hot_encoder
from . import ordinal_encoder
from . import one_vs_one_classifier
from . import one_vs_rest_classifier
from . import ovr_decision_function
from . import pipelines
from . import polynomial_features
from . import power_transformer
from . import quadratic_discriminant_analysis
from . import random_projection
from . import random_trees_embedding
from . import replace_op
from . import scaler
from . import sequence
from . import sgd_oneclass_svm
from . import svd
from . import support_vector_machines
from . import text_vectorizer
from . import tfidf_transformer
from . import voting_classifier
from . import voting_regressor
from . import zip_map

__all__ = [
    array_feature_extractor,
    cast_op,
    class_labels,
    concat,
    cross_decomposition,
    dict_vectorizer,
    ensemble_shapes,
    feature_hasher,
    flatten,
    function_transformer,
    gaussian_process,
    grid_search_cv,
    identity,
    imputer,
    isolation_forest,
    kernel_pca,
    k_bins_discretiser,
    k_means,
    label_binariser,
    label_encoder,
    linear_classifier,
    linear_regressor,
    local_outlier_factor,
    mixture,
    multioutput,
    multiply,
    nearest_neighbours,
    one_hot_encoder,
    ordinal_encoder,
    one_vs_one_classifier,
    one_vs_rest_classifier,
    ovr_decision_function,
    pipelines,
    polynomial_features,
    power_transformer,
    quadratic_discriminant_analysis,
    random_projection,
    random_trees_embedding,
    replace_op,
    scaler,
    sequence,
    sgd_oneclass_svm,
    svd,
    support_vector_machines,
    text_vectorizer,
    tfidf_transformer,
    voting_classifier,
    voting_regressor,
    zip_map,
]
