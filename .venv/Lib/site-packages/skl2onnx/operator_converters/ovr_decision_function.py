# SPDX-License-Identifier: Apache-2.0

from ..common._apply_operation import (
    apply_concat,
    apply_abs,
    apply_add,
    apply_mul,
    apply_div,
)

try:
    from ..common._apply_operation import apply_less
except ImportError:
    # onnxconverter-common is too old
    apply_less = None
from ..common.data_types import guess_proto_type
from ..common._registration import register_converter
from ..proto import onnx_proto
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_ovr_decision_function(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    # Applies _ovr_decision_function.
    # See https://github.com/scikit-learn/scikit-learn/blob/
    # master/sklearn/utils/multiclass.py#L407:
    # ::
    #     def _ovr_decision_function(predictions, confidences, n_classes):
    #
    #         n_samples = predictions.shape[0]
    #         votes = np.zeros((n_samples, n_classes))
    #         sum_of_confidences = np.zeros((n_samples, n_classes))
    #         k = 0
    #         for i in range(n_classes):
    #             for j in range(i + 1, n_classes):
    #                 sum_of_confidences[:, i] -= confidences[:, k]
    #                 sum_of_confidences[:, j] += confidences[:, k]
    #                 votes[predictions[:, k] == 0, i] += 1
    #                 votes[predictions[:, k] == 1, j] += 1
    #                 k += 1
    #         transformed_confidences = (
    #             sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1)))
    #         return votes + transformed_confidences
    proto_dtype = guess_proto_type(operator.inputs[1].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator

    cst3 = scope.get_unique_variable_name("cst3")
    container.add_initializer(cst3, proto_dtype, [], [3])
    cst1 = scope.get_unique_variable_name("cst1")
    container.add_initializer(cst1, proto_dtype, [], [1])

    iprediction = operator.inputs[0].full_name
    score_name = operator.inputs[1].full_name

    n_classes = len(op.classes_)
    sumc_name = [
        scope.get_unique_variable_name("svcsumc_%d" % i) for i in range(n_classes)
    ]
    vote_name = [
        scope.get_unique_variable_name("svcvote_%d" % i) for i in range(n_classes)
    ]
    sumc_add = {n: [] for n in sumc_name}
    vote_add = {n: [] for n in vote_name}
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            ind = scope.get_unique_variable_name("Cind_%d" % k)
            container.add_initializer(ind, onnx_proto.TensorProto.INT64, [], [k])

            # confidences
            ext = scope.get_unique_variable_name("Csvc_%d" % k)
            container.add_node(
                "ArrayFeatureExtractor", [score_name, ind], ext, op_domain="ai.onnx.ml"
            )
            sumc_add[sumc_name[j]].append(ext)

            neg = scope.get_unique_variable_name("Cneg_%d" % k)
            container.add_node("Neg", ext, neg, op_domain="", op_version=6)
            sumc_add[sumc_name[i]].append(neg)

            # votes
            ext = scope.get_unique_variable_name("Vsvcv_%d" % k)
            container.add_node(
                "ArrayFeatureExtractor", [iprediction, ind], ext, op_domain="ai.onnx.ml"
            )
            vote_add[vote_name[j]].append(ext)

            neg = scope.get_unique_variable_name("Vnegv_%d" % k)
            container.add_node("Neg", ext, neg, op_domain="", op_version=6)
            neg1 = scope.get_unique_variable_name("Vnegv1_%d" % k)
            apply_add(
                scope,
                [neg, cst1],
                neg1,
                container,
                broadcast=1,
                operator_name="AddCl_%d_%d" % (i, j),
            )
            vote_add[vote_name[i]].append(neg1)

            # next
            k += 1

    for k, v in sumc_add.items():
        name = scope.get_unique_operator_name("Sum")
        container.add_node("Sum", v, k, op_domain="", name=name, op_version=8)
    for k, v in vote_add.items():
        name = scope.get_unique_operator_name("Sum")
        container.add_node("Sum", v, k, op_domain="", name=name, op_version=8)

    conc = scope.get_unique_variable_name("Csvcconc")
    apply_concat(scope, sumc_name, conc, container, axis=1)
    conc_vote = scope.get_unique_variable_name("Vsvcconcv")
    apply_concat(scope, vote_name, conc_vote, container, axis=1)

    conc_abs = scope.get_unique_variable_name("Cabs")
    apply_abs(scope, conc, conc_abs, container)

    conc_abs1 = scope.get_unique_variable_name("Cconc_abs1")
    apply_add(
        scope,
        [conc_abs, cst1],
        conc_abs1,
        container,
        broadcast=1,
        operator_name="AddF0",
    )
    conc_abs3 = scope.get_unique_variable_name("Cconc_abs3")
    apply_mul(scope, [conc_abs1, cst3], conc_abs3, container, broadcast=1)

    final = scope.get_unique_variable_name("Csvcfinal")
    apply_div(scope, [conc, conc_abs3], final, container, broadcast=0)

    output_name = operator.outputs[0].full_name
    apply_add(
        scope,
        [conc_vote, final],
        output_name,
        container,
        broadcast=0,
        operator_name="AddF1",
    )


register_converter("SklearnOVRDecisionFunction", convert_sklearn_ovr_decision_function)
