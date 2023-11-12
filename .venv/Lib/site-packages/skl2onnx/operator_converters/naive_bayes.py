# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_add,
    apply_cast,
    apply_div,
    apply_exp,
    apply_log,
    apply_mul,
    apply_pow,
    apply_sub,
    apply_reshape,
    apply_transpose,
)
from ..common.data_types import (
    BooleanTensorType,
    Int64TensorType,
    guess_numpy_type,
    guess_proto_type,
)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.utils_classifier import get_label_classes


def _joint_log_likelihood_bernoulli(
    scope,
    container,
    input_name,
    feature_log_prob_name,
    class_log_prior_name,
    binarize,
    feature_count,
    proto_dtype,
    sum_result_name,
):
    """
    Calculate joint log likelihood for Bernoulli Naive Bayes model.
    """
    constant_name = scope.get_unique_variable_name("constant")
    exp_result_name = scope.get_unique_variable_name("exp_result")
    sub_result_name = scope.get_unique_variable_name("sub_result")
    neg_prob_name = scope.get_unique_variable_name("neg_prob")
    sum_neg_prob_name = scope.get_unique_variable_name("sum_neg_prob")
    difference_matrix_name = scope.get_unique_variable_name("difference_matrix")
    dot_prod_name = scope.get_unique_variable_name("dot_prod")
    partial_sum_result_name = scope.get_unique_variable_name("partial_sum_result")
    # Define constant slightly greater than 1 to avoid log 0
    # scenarios when calculating log (1 - x) and x=1 in line 70
    container.add_initializer(constant_name, proto_dtype, [], [1.000000001])

    if binarize is not None:
        threshold_name = scope.get_unique_variable_name("threshold")
        condition_name = scope.get_unique_variable_name("condition")
        cast_values_name = scope.get_unique_variable_name("cast_values")
        zero_tensor_name = scope.get_unique_variable_name("zero_tensor")
        binarised_input_name = scope.get_unique_variable_name("binarised_input")
        num_features = feature_count.shape[1]

        container.add_initializer(threshold_name, proto_dtype, [1], [binarize])
        container.add_initializer(
            zero_tensor_name,
            proto_dtype,
            [1, num_features],
            np.zeros((1, num_features)).ravel(),
        )

        container.add_node(
            "Greater",
            [input_name, threshold_name],
            condition_name,
            name=scope.get_unique_operator_name("Greater"),
            op_version=9,
        )
        apply_cast(scope, condition_name, cast_values_name, container, to=proto_dtype)
        apply_add(
            scope,
            [zero_tensor_name, cast_values_name],
            binarised_input_name,
            container,
            broadcast=1,
        )
        input_name = binarised_input_name

    apply_exp(scope, feature_log_prob_name, exp_result_name, container)
    apply_sub(
        scope, [constant_name, exp_result_name], sub_result_name, container, broadcast=1
    )
    apply_log(scope, sub_result_name, neg_prob_name, container)
    if container.target_opset < 13:
        container.add_node(
            "ReduceSum",
            neg_prob_name,
            sum_neg_prob_name,
            axes=[0],
            name=scope.get_unique_operator_name("ReduceSum"),
        )
    else:
        axis_name = scope.get_unique_variable_name("axis")
        container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [0])
        container.add_node(
            "ReduceSum",
            [neg_prob_name, axis_name],
            sum_neg_prob_name,
            name=scope.get_unique_operator_name("ReduceSum"),
        )
    apply_sub(
        scope, [feature_log_prob_name, neg_prob_name], difference_matrix_name, container
    )
    container.add_node(
        "MatMul",
        [input_name, difference_matrix_name],
        dot_prod_name,
        name=scope.get_unique_operator_name("MatMul"),
    )

    apply_add(
        scope, [dot_prod_name, sum_neg_prob_name], partial_sum_result_name, container
    )
    apply_add(
        scope,
        [partial_sum_result_name, class_log_prior_name],
        sum_result_name,
        container,
    )
    return sum_result_name


def _joint_log_likelihood_gaussian(
    scope, container, input_name, model, proto_dtype, sum_result_name
):
    """
    Calculate joint log likelihood for Gaussian Naive Bayes model.
    """
    features = model.theta_.shape[1]
    jointi = np.log(model.class_prior_)
    var_sigma = model.var_ if hasattr(model, "var_") else model.sigma_
    sigma_sum_log = -0.5 * np.sum(np.log(2.0 * np.pi * var_sigma), axis=1)
    theta_name = scope.get_unique_variable_name("theta")
    sigma_name = scope.get_unique_variable_name("sigma")
    sigma_sum_log_name = scope.get_unique_variable_name("sigma_sum_log")
    jointi_name = scope.get_unique_variable_name("jointi")
    exponent_name = scope.get_unique_variable_name("exponent")
    prod_operand_name = scope.get_unique_variable_name("prod_operand")
    reshaped_input_name = scope.get_unique_variable_name("reshaped_input")
    subtracted_input_name = scope.get_unique_variable_name("subtracted_input")
    pow_result_name = scope.get_unique_variable_name("pow_result")
    div_result_name = scope.get_unique_variable_name("div_result")
    reduced_sum_name = scope.get_unique_variable_name("reduced_sum")
    mul_result_name = scope.get_unique_variable_name("mul_result")
    part_log_likelihood_name = scope.get_unique_variable_name("part_log_likelihood")

    theta = model.theta_.reshape((1, -1, features))
    sigma = var_sigma.reshape((1, -1, features))

    container.add_initializer(theta_name, proto_dtype, theta.shape, theta.ravel())
    container.add_initializer(sigma_name, proto_dtype, sigma.shape, sigma.ravel())
    container.add_initializer(jointi_name, proto_dtype, [1, jointi.shape[0]], jointi)
    container.add_initializer(
        sigma_sum_log_name,
        proto_dtype,
        [1, sigma_sum_log.shape[0]],
        sigma_sum_log.ravel(),
    )
    container.add_initializer(exponent_name, proto_dtype, [], [2])
    container.add_initializer(prod_operand_name, proto_dtype, [], [0.5])

    apply_reshape(
        scope,
        input_name,
        reshaped_input_name,
        container,
        desired_shape=[-1, 1, features],
    )
    apply_sub(
        scope,
        [reshaped_input_name, theta_name],
        subtracted_input_name,
        container,
        broadcast=1,
    )
    apply_pow(
        scope,
        [subtracted_input_name, exponent_name],
        pow_result_name,
        container,
        broadcast=1,
    )
    apply_div(
        scope, [pow_result_name, sigma_name], div_result_name, container, broadcast=1
    )
    if container.target_opset < 13:
        container.add_node(
            "ReduceSum",
            div_result_name,
            reduced_sum_name,
            axes=[2],
            keepdims=0,
            name=scope.get_unique_operator_name("ReduceSum"),
        )
    else:
        axis_name = scope.get_unique_variable_name("axis")
        container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [2])
        container.add_node(
            "ReduceSum",
            [div_result_name, axis_name],
            reduced_sum_name,
            keepdims=0,
            name=scope.get_unique_operator_name("ReduceSum"),
        )
    apply_mul(
        scope,
        [reduced_sum_name, prod_operand_name],
        mul_result_name,
        container,
        broadcast=1,
    )
    apply_sub(
        scope,
        [sigma_sum_log_name, mul_result_name],
        part_log_likelihood_name,
        container,
        broadcast=1,
    )
    apply_add(
        scope,
        [jointi_name, part_log_likelihood_name],
        sum_result_name,
        container,
        broadcast=1,
    )
    return sum_result_name


def _joint_log_likelihood_categorical(
    scope, container, input_name, model, sum_result_name
):
    """
    Calculate joint log likelihood for Categorical Naive Bayes model.
    """
    jll_list = []
    class_log_prior_name = scope.get_unique_variable_name("class_log_prior")
    summation_jll_name = scope.get_unique_variable_name("summation_jll")

    container.add_initializer(
        class_log_prior_name,
        onnx_proto.TensorProto.FLOAT,
        model.class_log_prior_.shape,
        model.class_log_prior_,
    )

    n_features = (
        model.n_features_in_ if hasattr(model, "n_features_in_") else model.n_features_
    )

    for i in range(n_features):
        feature_index_name = scope.get_unique_variable_name("feature_index")
        indices_name = scope.get_unique_variable_name("indices")
        cast_indices_name = scope.get_unique_variable_name("cast_indices")
        feature_log_proba_name = scope.get_unique_variable_name("feature_log_proba")
        jll_name = scope.get_unique_variable_name("jll")
        transposed_jll_name = scope.get_unique_variable_name("transposed_jll")

        container.add_initializer(
            feature_index_name, onnx_proto.TensorProto.INT64, [], [i]
        )
        container.add_initializer(
            feature_log_proba_name,
            onnx_proto.TensorProto.FLOAT,
            model.feature_log_prob_[i].shape,
            model.feature_log_prob_[i].ravel(),
        )

        container.add_node(
            "ArrayFeatureExtractor",
            [input_name, feature_index_name],
            indices_name,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
        )
        apply_cast(
            scope,
            indices_name,
            cast_indices_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
        container.add_node(
            "ArrayFeatureExtractor",
            [feature_log_proba_name, cast_indices_name],
            jll_name,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
        )
        apply_transpose(scope, jll_name, transposed_jll_name, container, perm=(1, 0))
        jll_list.append(transposed_jll_name)
    container.add_node(
        "Sum", jll_list, summation_jll_name, name=scope.get_unique_operator_name("Sum")
    )
    apply_add(
        scope,
        [summation_jll_name, class_log_prior_name],
        sum_result_name,
        container,
        broadcast=1,
    )
    return sum_result_name


def convert_sklearn_naive_bayes(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    # Computational graph:
    #
    # Note: In the following graph, variable names are in lower case
    # characters only and operator names are in upper case characters.
    # We borrow operator names from the official ONNX spec:
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of instances
    # N: Number of features
    # C: Number of classes
    # input(or x): input
    # output(or y): output (There are two paths for producing output, one for
    #               string labels and the other one for int labels)
    # output_probability: class probabilties
    # feature_log_prob: Empirical log probability of features given a
    #                   class, P(x_i|y)
    # class_log_prior: Smoothed empirical log probability for each class
    #
    # Multinomial NB
    # Equation:
    #   y = argmax (class_log_prior + X . feature_log_prob^T)
    #
    # Graph:
    #
    #   input [M, N] -> MATMUL <- feature_log_prob.T [N, C]
    #                    |
    #                    V
    #        matmul_result [M, C] -> CAST <- proto_dtype
    #                                |
    #                                V
    #                    cast_result [M, C] -> ADD <- class_log_prior [1, C]
    #                                          |
    #                        .-----------------'
    #                        |
    #                        V
    # sum_result [M, C] -> ARGMAX -> argmax_output [M, 1]
    #                        |
    #                        V
    # classes [C] -----> ARRAYFEATUREEXTRACTOR
    #                        |
    #                        V               (string labels)
    # array_feature_extractor_result [M, 1] --------------------------.
    #           (int labels) |                                        |
    #                        V                                        |
    #              CAST(to=proto_dtype)              |
    #                        |                                        |
    #                        V                                        |
    #                  cast2_result [M, 1]                            |
    #                        |                                        |
    #                        V                                        |
    # output_shape [1] -> RESHAPE                                     |
    #                        |                                        |
    #                        V                                        V
    #                       reshaped_result [M,]            .-----RESHAPE
    #                                   |                   |
    #                                   V                   V
    #  (to=onnx_proto.TensorProto.INT64)CAST --------> output [M,]
    #
    # Bernoulli NB
    # Equation:
    #   y = argmax (class_log_prior + \sum neg_prob
    #               + X . (feature_log_prob - neg_prob))
    #   neg_prob = log( 1 - e ^ feature_log_prob)
    #
    #   Graph:
    #
    #           .---------------------------------------------------------.
    #           |                                                         |
    #  feature_log_prob.T [N, C] -> EXP -> exp_result [N, C]              |
    #                                      |                              |
    #               .----------------------'                              |
    #               |                                                     |
    #               V                                                     V
    #  constant -> SUB -> sub_result [N, C] -> LOG -> neg_prob [N, C] -> SUB
    #                                                   |                 |
    #                                         .---------'       .---------'
    #                                         |                 |
    #                                         V                 V
    #  .----------- sum_neg_prob [1, C] <- REDUCE_SUM  difference_matrix [N, C]
    #  |                                                        |
    #  |                     .----------------------------------'
    #  |                     |
    #  |                     V
    #  |    input [M, N] -> MATMUL -> dot_product [M, C]
    #  |                                       |
    #  |                                       V
    #  '------------------------------------> ADD
    #                                          |
    #                                          V
    #  class_log_prior [1, C] -> ADD <- partial_sum_result [M, C]
    #                            |
    #                            V
    #                   sum_result [M, C] -> ARGMAX -> argmax_output [M, 1]
    #                                                            |
    #                                                            V
    #                              classes [C] -------> ARRAYFEATUREEXTRACTOR
    #                                                            |
    #                       .------------------------------------'
    #                       |
    #                       V                (string labels)
    #  array_feature_extractor_result [M, 1] ----------------.
    #          (int labels) |                                |
    #                       V                                |
    #   CAST(to=proto_dtype)                |
    #                       |                                |
    #                       V                                |
    #                cast2_result [M, 1]                     |
    #                       |                                |
    #                       V                                |
    #  output_shape [1] -> RESHAPE                           |
    #                          |                             |
    #                          V                             V
    #                       reshaped_result [M,]           RESHAPE
    #                                   |                    |
    #                                   V                    |
    # (to=onnx_proto.TensorProto.INT64)CAST -> output [M,] <-'
    #
    #
    # If model's binarize attribute is not null, then input of
    # Bernoulli NB is produced by the following graph:
    #
    #    input [M, N] -> GREATER <- threshold [1]
    #       |              |
    #       |              V
    #       |       condition [M, N] -> CAST(to=proto_dtype)
    #       |                             |
    #       |                             V
    #       |                          cast_values [M, N]
    #       |                                   |
    #       V                                   V
    #   CONSTANT_LIKE -> zero_tensor [M, N] -> ADD
    #                                           |
    #                                           V
    #               input [M, N] <- binarised_input [M, N]
    #
    # Sub-graph for probability calculation common to both Multinomial
    # and Bernoulli Naive Bayes
    #
    #  sum_result [M, C] -> REDUCELOGSUMEXP -> reduce_log_sum_exp_result [M,]
    #         |                                                   |
    #         |                                                   V
    #         |                           log_prob_shape [2] -> RESHAPE
    #         |                                                   |
    #         '------------> SUB <-- reshaped_log_prob [M, 1] <---'
    #                         |
    #                         V
    #                     log_prob [M, C] -> EXP -> prob_tensor [M, C] -.
    #                                                                   |
    #         output_probability [M, C] <- ZIPMAP <---------------------'
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    float_dtype = dtype
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    nb_op = operator.raw_operator
    classes = get_label_classes(scope, nb_op)
    output_shape = (-1,)

    sum_result_name = scope.get_unique_variable_name("sum_result")
    argmax_output_name = scope.get_unique_variable_name("argmax_output")
    cast2_result_name = scope.get_unique_variable_name("cast2_result")
    reshaped_result_name = scope.get_unique_variable_name("reshaped_result")
    classes_name = scope.get_unique_variable_name("classes")
    reduce_log_sum_exp_result_name = scope.get_unique_variable_name(
        "reduce_log_sum_exp_result"
    )
    log_prob_name = scope.get_unique_variable_name("log_prob")
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        "array_feature_extractor_result"
    )

    class_type = onnx_proto.TensorProto.STRING
    if np.issubdtype(classes.dtype, np.floating) or classes.dtype == np.bool_:
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype(np.int32)
    elif np.issubdtype(classes.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode("utf-8") for s in classes])

    container.add_initializer(classes_name, class_type, classes.shape, classes)

    if operator.type not in ("SklearnCategoricalNB", "SklearnGaussianNB"):
        class_log_prior_name = scope.get_unique_variable_name("class_log_prior")
        feature_log_prob_name = scope.get_unique_variable_name("feature_log_prob")

        class_log_prior = nb_op.class_log_prior_.astype(float_dtype).reshape((1, -1))
        feature_log_prob = nb_op.feature_log_prob_.T.astype(float_dtype)

        container.add_initializer(
            feature_log_prob_name,
            proto_dtype,
            feature_log_prob.shape,
            feature_log_prob.flatten(),
        )
        container.add_initializer(
            class_log_prior_name,
            proto_dtype,
            class_log_prior.shape,
            class_log_prior.flatten(),
        )

    input_name = operator.inputs[0].full_name
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")

        apply_cast(
            scope, operator.input_full_names, cast_input_name, container, to=proto_dtype
        )
        input_name = cast_input_name

    if operator.type == "SklearnBernoulliNB":
        sum_result_name = _joint_log_likelihood_bernoulli(
            scope,
            container,
            input_name,
            feature_log_prob_name,
            class_log_prior_name,
            nb_op.binarize,
            nb_op.feature_count_,
            proto_dtype,
            sum_result_name,
        )
    elif operator.type == "SklearnGaussianNB":
        sum_result_name = _joint_log_likelihood_gaussian(
            scope, container, input_name, nb_op, proto_dtype, sum_result_name
        )
    elif operator.type == "SklearnCategoricalNB":
        sum_result_name = _joint_log_likelihood_categorical(
            scope, container, input_name, nb_op, sum_result_name
        )
    else:
        # MultinomialNB or ComplementNB
        matmul_result_name = (
            scope.get_unique_variable_name("matmul_result")
            if operator.type == "SklearnMultinomialNB" or len(classes) == 1
            else sum_result_name
        )

        container.add_node(
            "MatMul",
            [input_name, feature_log_prob_name],
            matmul_result_name,
            name=scope.get_unique_operator_name("MatMul"),
        )
        if operator.type == "SklearnMultinomialNB" or len(classes) == 1:
            apply_add(
                scope,
                [matmul_result_name, class_log_prior_name],
                sum_result_name,
                container,
                broadcast=1,
            )

    container.add_node(
        "ArgMax",
        sum_result_name,
        argmax_output_name,
        name=scope.get_unique_operator_name("ArgMax"),
        axis=1,
    )

    # Calculation of class probability
    log_prob_shape = [-1, 1]

    reshaped_log_prob_name = scope.get_unique_variable_name("reshaped_log_prob")

    if container.target_opset >= 18:
        axis_name = scope.get_unique_variable_name("axis")
        container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [1])
        container.add_node(
            "ReduceLogSumExp",
            [sum_result_name, axis_name],
            reduce_log_sum_exp_result_name,
            name=scope.get_unique_operator_name("ReduceLogSumExp"),
            keepdims=0,
        )
    else:
        container.add_node(
            "ReduceLogSumExp",
            sum_result_name,
            reduce_log_sum_exp_result_name,
            name=scope.get_unique_operator_name("ReduceLogSumExp"),
            axes=[1],
            keepdims=0,
        )
    apply_reshape(
        scope,
        reduce_log_sum_exp_result_name,
        reshaped_log_prob_name,
        container,
        desired_shape=log_prob_shape,
    )
    apply_sub(
        scope,
        [sum_result_name, reshaped_log_prob_name],
        log_prob_name,
        container,
        broadcast=1,
    )
    apply_exp(scope, log_prob_name, operator.outputs[1].full_name, container)
    container.add_node(
        "ArrayFeatureExtractor",
        [classes_name, argmax_output_name],
        array_feature_extractor_result_name,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
    )
    # Reshape op does not seem to handle INT64 tensor even though it is
    # listed as one of the supported types in the doc, so Cast was
    # required here.
    if class_type == onnx_proto.TensorProto.INT32:
        apply_cast(
            scope,
            array_feature_extractor_result_name,
            cast2_result_name,
            container,
            to=proto_dtype,
        )
        apply_reshape(
            scope,
            cast2_result_name,
            reshaped_result_name,
            container,
            desired_shape=output_shape,
        )
        apply_cast(
            scope,
            reshaped_result_name,
            operator.outputs[0].full_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    else:  # string labels
        apply_reshape(
            scope,
            array_feature_extractor_result_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=output_shape,
        )


register_converter(
    "SklearnBernoulliNB",
    convert_sklearn_naive_bayes,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
register_converter(
    "SklearnCategoricalNB",
    convert_sklearn_naive_bayes,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
register_converter(
    "SklearnComplementNB",
    convert_sklearn_naive_bayes,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
register_converter(
    "SklearnGaussianNB",
    convert_sklearn_naive_bayes,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
register_converter(
    "SklearnMultinomialNB",
    convert_sklearn_naive_bayes,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
