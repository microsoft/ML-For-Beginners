from importlib import import_module
from inspect import signature
from numbers import Integral, Real

import pytest

from sklearn.utils._param_validation import (
    Interval,
    InvalidParameterError,
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
)


def _get_func_info(func_module):
    module_name, func_name = func_module.rsplit(".", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    func_sig = signature(func)
    func_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    # The parameters `*args` and `**kwargs` are ignored since we cannot generate
    # constraints.
    required_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    return func, func_name, func_params, required_params


def _check_function_param_validation(
    func, func_name, func_params, required_params, parameter_constraints
):
    """Check that an informative error is raised when the value of a parameter does not
    have an appropriate type or value.
    """
    # generate valid values for the required parameters
    valid_required_params = {}
    for param_name in required_params:
        if parameter_constraints[param_name] == "no_validation":
            valid_required_params[param_name] = 1
        else:
            valid_required_params[param_name] = generate_valid_param(
                make_constraint(parameter_constraints[param_name][0])
            )

    # check that there is a constraint for each parameter
    if func_params:
        validation_params = parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(func_params)
        missing_params = set(func_params) - set(validation_params)
        err_msg = (
            "Mismatch between _parameter_constraints and the parameters of"
            f" {func_name}.\nConsider the unexpected parameters {unexpected_params} and"
            f" expected but missing parameters {missing_params}\n"
        )
        assert set(validation_params) == set(func_params), err_msg

    # this object does not have a valid type for sure for all params
    param_with_bad_type = type("BadType", (), {})()

    for param_name in func_params:
        constraints = parameter_constraints[param_name]

        if constraints == "no_validation":
            # This parameter is not validated
            continue

        # Mixing an interval of reals and an interval of integers must be avoided.
        if any(
            isinstance(constraint, Interval) and constraint.type == Integral
            for constraint in constraints
        ) and any(
            isinstance(constraint, Interval) and constraint.type == Real
            for constraint in constraints
        ):
            raise ValueError(
                f"The constraint for parameter {param_name} of {func_name} can't have a"
                " mix of intervals of Integral and Real types. Use the type"
                " RealNotInt instead of Real."
            )

        match = (
            rf"The '{param_name}' parameter of {func_name} must be .* Got .* instead."
        )

        err_msg = (
            f"{func_name} does not raise an informative error message when the "
            f"parameter {param_name} does not have a valid type. If any Python type "
            "is valid, the constraint should be 'no_validation'."
        )

        # First, check that the error is raised if param doesn't match any valid type.
        with pytest.raises(InvalidParameterError, match=match):
            func(**{**valid_required_params, param_name: param_with_bad_type})
            pytest.fail(err_msg)

        # Then, for constraints that are more than a type constraint, check that the
        # error is raised if param does match a valid type but does not match any valid
        # value for this type.
        constraints = [make_constraint(constraint) for constraint in constraints]

        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            err_msg = (
                f"{func_name} does not raise an informative error message when the "
                f"parameter {param_name} does not have a valid value.\n"
                "Constraints should be disjoint. For instance "
                "[StrOptions({'a_string'}), str] is not a acceptable set of "
                "constraint because generating an invalid string for the first "
                "constraint will always produce a valid string for the second "
                "constraint."
            )

            with pytest.raises(InvalidParameterError, match=match):
                func(**{**valid_required_params, param_name: bad_value})
                pytest.fail(err_msg)


PARAM_VALIDATION_FUNCTION_LIST = [
    "sklearn.calibration.calibration_curve",
    "sklearn.cluster.cluster_optics_dbscan",
    "sklearn.cluster.compute_optics_graph",
    "sklearn.cluster.estimate_bandwidth",
    "sklearn.cluster.kmeans_plusplus",
    "sklearn.cluster.cluster_optics_xi",
    "sklearn.cluster.ward_tree",
    "sklearn.covariance.empirical_covariance",
    "sklearn.covariance.ledoit_wolf_shrinkage",
    "sklearn.covariance.log_likelihood",
    "sklearn.covariance.shrunk_covariance",
    "sklearn.datasets.clear_data_home",
    "sklearn.datasets.dump_svmlight_file",
    "sklearn.datasets.fetch_20newsgroups",
    "sklearn.datasets.fetch_20newsgroups_vectorized",
    "sklearn.datasets.fetch_california_housing",
    "sklearn.datasets.fetch_covtype",
    "sklearn.datasets.fetch_kddcup99",
    "sklearn.datasets.fetch_lfw_pairs",
    "sklearn.datasets.fetch_lfw_people",
    "sklearn.datasets.fetch_olivetti_faces",
    "sklearn.datasets.fetch_rcv1",
    "sklearn.datasets.fetch_openml",
    "sklearn.datasets.fetch_species_distributions",
    "sklearn.datasets.get_data_home",
    "sklearn.datasets.load_breast_cancer",
    "sklearn.datasets.load_diabetes",
    "sklearn.datasets.load_digits",
    "sklearn.datasets.load_files",
    "sklearn.datasets.load_iris",
    "sklearn.datasets.load_linnerud",
    "sklearn.datasets.load_sample_image",
    "sklearn.datasets.load_svmlight_file",
    "sklearn.datasets.load_svmlight_files",
    "sklearn.datasets.load_wine",
    "sklearn.datasets.make_biclusters",
    "sklearn.datasets.make_blobs",
    "sklearn.datasets.make_checkerboard",
    "sklearn.datasets.make_circles",
    "sklearn.datasets.make_classification",
    "sklearn.datasets.make_friedman1",
    "sklearn.datasets.make_friedman2",
    "sklearn.datasets.make_friedman3",
    "sklearn.datasets.make_gaussian_quantiles",
    "sklearn.datasets.make_hastie_10_2",
    "sklearn.datasets.make_low_rank_matrix",
    "sklearn.datasets.make_moons",
    "sklearn.datasets.make_multilabel_classification",
    "sklearn.datasets.make_regression",
    "sklearn.datasets.make_s_curve",
    "sklearn.datasets.make_sparse_coded_signal",
    "sklearn.datasets.make_sparse_spd_matrix",
    "sklearn.datasets.make_sparse_uncorrelated",
    "sklearn.datasets.make_spd_matrix",
    "sklearn.datasets.make_swiss_roll",
    "sklearn.decomposition.sparse_encode",
    "sklearn.feature_extraction.grid_to_graph",
    "sklearn.feature_extraction.img_to_graph",
    "sklearn.feature_extraction.image.extract_patches_2d",
    "sklearn.feature_extraction.image.reconstruct_from_patches_2d",
    "sklearn.feature_selection.chi2",
    "sklearn.feature_selection.f_classif",
    "sklearn.feature_selection.f_regression",
    "sklearn.feature_selection.mutual_info_classif",
    "sklearn.feature_selection.mutual_info_regression",
    "sklearn.feature_selection.r_regression",
    "sklearn.inspection.partial_dependence",
    "sklearn.inspection.permutation_importance",
    "sklearn.isotonic.check_increasing",
    "sklearn.isotonic.isotonic_regression",
    "sklearn.linear_model.enet_path",
    "sklearn.linear_model.lars_path",
    "sklearn.linear_model.lars_path_gram",
    "sklearn.linear_model.lasso_path",
    "sklearn.linear_model.orthogonal_mp",
    "sklearn.linear_model.orthogonal_mp_gram",
    "sklearn.linear_model.ridge_regression",
    "sklearn.manifold.trustworthiness",
    "sklearn.metrics.accuracy_score",
    "sklearn.manifold.smacof",
    "sklearn.metrics.auc",
    "sklearn.metrics.average_precision_score",
    "sklearn.metrics.balanced_accuracy_score",
    "sklearn.metrics.brier_score_loss",
    "sklearn.metrics.calinski_harabasz_score",
    "sklearn.metrics.check_scoring",
    "sklearn.metrics.completeness_score",
    "sklearn.metrics.class_likelihood_ratios",
    "sklearn.metrics.classification_report",
    "sklearn.metrics.cluster.adjusted_mutual_info_score",
    "sklearn.metrics.cluster.contingency_matrix",
    "sklearn.metrics.cluster.entropy",
    "sklearn.metrics.cluster.fowlkes_mallows_score",
    "sklearn.metrics.cluster.homogeneity_completeness_v_measure",
    "sklearn.metrics.cluster.normalized_mutual_info_score",
    "sklearn.metrics.cluster.silhouette_samples",
    "sklearn.metrics.cluster.silhouette_score",
    "sklearn.metrics.cohen_kappa_score",
    "sklearn.metrics.confusion_matrix",
    "sklearn.metrics.consensus_score",
    "sklearn.metrics.coverage_error",
    "sklearn.metrics.d2_absolute_error_score",
    "sklearn.metrics.d2_pinball_score",
    "sklearn.metrics.d2_tweedie_score",
    "sklearn.metrics.davies_bouldin_score",
    "sklearn.metrics.dcg_score",
    "sklearn.metrics.det_curve",
    "sklearn.metrics.explained_variance_score",
    "sklearn.metrics.f1_score",
    "sklearn.metrics.fbeta_score",
    "sklearn.metrics.get_scorer",
    "sklearn.metrics.hamming_loss",
    "sklearn.metrics.hinge_loss",
    "sklearn.metrics.homogeneity_score",
    "sklearn.metrics.jaccard_score",
    "sklearn.metrics.label_ranking_average_precision_score",
    "sklearn.metrics.label_ranking_loss",
    "sklearn.metrics.log_loss",
    "sklearn.metrics.make_scorer",
    "sklearn.metrics.matthews_corrcoef",
    "sklearn.metrics.max_error",
    "sklearn.metrics.mean_absolute_error",
    "sklearn.metrics.mean_absolute_percentage_error",
    "sklearn.metrics.mean_gamma_deviance",
    "sklearn.metrics.mean_pinball_loss",
    "sklearn.metrics.mean_poisson_deviance",
    "sklearn.metrics.mean_squared_error",
    "sklearn.metrics.mean_squared_log_error",
    "sklearn.metrics.mean_tweedie_deviance",
    "sklearn.metrics.median_absolute_error",
    "sklearn.metrics.multilabel_confusion_matrix",
    "sklearn.metrics.mutual_info_score",
    "sklearn.metrics.ndcg_score",
    "sklearn.metrics.pair_confusion_matrix",
    "sklearn.metrics.adjusted_rand_score",
    "sklearn.metrics.pairwise.additive_chi2_kernel",
    "sklearn.metrics.pairwise.chi2_kernel",
    "sklearn.metrics.pairwise.cosine_distances",
    "sklearn.metrics.pairwise.cosine_similarity",
    "sklearn.metrics.pairwise.euclidean_distances",
    "sklearn.metrics.pairwise.haversine_distances",
    "sklearn.metrics.pairwise.laplacian_kernel",
    "sklearn.metrics.pairwise.linear_kernel",
    "sklearn.metrics.pairwise.manhattan_distances",
    "sklearn.metrics.pairwise.nan_euclidean_distances",
    "sklearn.metrics.pairwise.paired_cosine_distances",
    "sklearn.metrics.pairwise.paired_distances",
    "sklearn.metrics.pairwise.paired_euclidean_distances",
    "sklearn.metrics.pairwise.paired_manhattan_distances",
    "sklearn.metrics.pairwise.pairwise_distances_argmin_min",
    "sklearn.metrics.pairwise.pairwise_kernels",
    "sklearn.metrics.pairwise.polynomial_kernel",
    "sklearn.metrics.pairwise.rbf_kernel",
    "sklearn.metrics.pairwise.sigmoid_kernel",
    "sklearn.metrics.pairwise_distances",
    "sklearn.metrics.pairwise_distances_argmin",
    "sklearn.metrics.pairwise_distances_chunked",
    "sklearn.metrics.precision_recall_curve",
    "sklearn.metrics.precision_recall_fscore_support",
    "sklearn.metrics.precision_score",
    "sklearn.metrics.r2_score",
    "sklearn.metrics.rand_score",
    "sklearn.metrics.recall_score",
    "sklearn.metrics.roc_auc_score",
    "sklearn.metrics.roc_curve",
    "sklearn.metrics.root_mean_squared_error",
    "sklearn.metrics.root_mean_squared_log_error",
    "sklearn.metrics.top_k_accuracy_score",
    "sklearn.metrics.v_measure_score",
    "sklearn.metrics.zero_one_loss",
    "sklearn.model_selection.cross_val_predict",
    "sklearn.model_selection.cross_val_score",
    "sklearn.model_selection.cross_validate",
    "sklearn.model_selection.learning_curve",
    "sklearn.model_selection.permutation_test_score",
    "sklearn.model_selection.train_test_split",
    "sklearn.model_selection.validation_curve",
    "sklearn.neighbors.kneighbors_graph",
    "sklearn.neighbors.radius_neighbors_graph",
    "sklearn.neighbors.sort_graph_by_row_values",
    "sklearn.preprocessing.add_dummy_feature",
    "sklearn.preprocessing.binarize",
    "sklearn.preprocessing.label_binarize",
    "sklearn.preprocessing.normalize",
    "sklearn.preprocessing.scale",
    "sklearn.random_projection.johnson_lindenstrauss_min_dim",
    "sklearn.svm.l1_min_c",
    "sklearn.tree.export_graphviz",
    "sklearn.tree.export_text",
    "sklearn.tree.plot_tree",
    "sklearn.utils.gen_batches",
    "sklearn.utils.gen_even_slices",
    "sklearn.utils.resample",
    "sklearn.utils.safe_mask",
    "sklearn.utils.extmath.randomized_svd",
    "sklearn.utils.class_weight.compute_class_weight",
    "sklearn.utils.class_weight.compute_sample_weight",
    "sklearn.utils.graph.single_source_shortest_path_length",
]


@pytest.mark.parametrize("func_module", PARAM_VALIDATION_FUNCTION_LIST)
def test_function_param_validation(func_module):
    """Check param validation for public functions that are not wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    parameter_constraints = getattr(func, "_skl_parameter_constraints")

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )


PARAM_VALIDATION_CLASS_WRAPPER_LIST = [
    ("sklearn.cluster.affinity_propagation", "sklearn.cluster.AffinityPropagation"),
    ("sklearn.cluster.dbscan", "sklearn.cluster.DBSCAN"),
    ("sklearn.cluster.k_means", "sklearn.cluster.KMeans"),
    ("sklearn.cluster.mean_shift", "sklearn.cluster.MeanShift"),
    ("sklearn.cluster.spectral_clustering", "sklearn.cluster.SpectralClustering"),
    ("sklearn.covariance.graphical_lasso", "sklearn.covariance.GraphicalLasso"),
    ("sklearn.covariance.ledoit_wolf", "sklearn.covariance.LedoitWolf"),
    ("sklearn.covariance.oas", "sklearn.covariance.OAS"),
    ("sklearn.decomposition.dict_learning", "sklearn.decomposition.DictionaryLearning"),
    ("sklearn.decomposition.fastica", "sklearn.decomposition.FastICA"),
    ("sklearn.decomposition.non_negative_factorization", "sklearn.decomposition.NMF"),
    ("sklearn.preprocessing.maxabs_scale", "sklearn.preprocessing.MaxAbsScaler"),
    ("sklearn.preprocessing.minmax_scale", "sklearn.preprocessing.MinMaxScaler"),
    ("sklearn.preprocessing.power_transform", "sklearn.preprocessing.PowerTransformer"),
    (
        "sklearn.preprocessing.quantile_transform",
        "sklearn.preprocessing.QuantileTransformer",
    ),
    ("sklearn.preprocessing.robust_scale", "sklearn.preprocessing.RobustScaler"),
]


@pytest.mark.parametrize(
    "func_module, class_module", PARAM_VALIDATION_CLASS_WRAPPER_LIST
)
def test_class_wrapper_param_validation(func_module, class_module):
    """Check param validation for public functions that are wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    module_name, class_name = class_module.rsplit(".", 1)
    module = import_module(module_name)
    klass = getattr(module, class_name)

    parameter_constraints_func = getattr(func, "_skl_parameter_constraints")
    parameter_constraints_class = getattr(klass, "_parameter_constraints")
    parameter_constraints = {
        **parameter_constraints_class,
        **parameter_constraints_func,
    }
    parameter_constraints = {
        k: v for k, v in parameter_constraints.items() if k in func_params
    }

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )
