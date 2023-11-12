# SPDX-License-Identifier: Apache-2.0

import numpy as np

try:
    from sklearn.ensemble._iforest import _average_path_length
except ImportError:
    # scikit-learn < 0.22
    from sklearn.ensemble.iforest import _average_path_length
from ..common._registration import register_converter
from ..common.data_types import (
    BooleanTensorType,
    Int64TensorType,
    guess_numpy_type,
    guess_proto_type,
)
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
)
from ..proto import onnx_proto
from ..algebra.onnx_ops import (
    OnnxTreeEnsembleRegressor_1,
    OnnxLog,
    OnnxCast,
    OnnxLess,
    OnnxLabelEncoder,
    OnnxMul,
    OnnxGreater,
    OnnxAdd,
    OnnxDiv,
    OnnxSum,
    OnnxNeg,
    OnnxReshapeApi13,
    OnnxEqual,
    OnnxPow,
    OnnxGather,
    OnnxMax,
)


def convert_sklearn_isolation_forest(
    scope,
    operator,
    container,
    op_type="TreeEnsembleRegressor",
    op_domain="ai.onnx.ml",
    op_version=1,
):
    op = operator.raw_operator
    outputs = operator.outputs
    opv = container.target_opset
    opvml = container.target_opset_any_domain("ai.onnx.ml")
    options = container.get_options(op, dict(score_samples=None))
    if opvml < 2:
        raise RuntimeError(
            "This converter requires at least opset 2 for " "domain 'ai.onnx.ml'."
        )

    input_name = operator.inputs[0]
    dtype = guess_numpy_type(operator.inputs[0].type)
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        input_name = OnnxCast(input_name, to=proto_dtype, op_version=opv)

    if op._max_features != operator.inputs[0].type.shape[1]:
        raise RuntimeError(
            "Converter for IsolationForest does not support the case when "
            "_max_features={} != number of given features {}.".format(
                op._max_features, operator.inputs[0].type.shape[1]
            )
        )

    # decision_path
    scores = []
    for i, (tree, features) in enumerate(zip(op.estimators_, op.estimators_features_)):
        # X_subset = X[:, features]
        gather = OnnxGather(
            input_name, features.astype(np.int64), axis=1, op_version=opv
        )

        attrs = get_default_tree_regressor_attribute_pairs()
        attrs["n_targets"] = 1
        add_tree_to_attribute_pairs(
            attrs, False, tree.tree_, 0, 1.0, 0, False, True, dtype=dtype
        )

        # tree leave
        attrs["n_targets"] = 1
        attrs["post_transform"] = "NONE"
        attrs["target_ids"] = [0 for _ in attrs["target_ids"]]
        attrs["target_weights"] = [float(_) for _ in attrs["target_nodeids"]]
        leave = OnnxTreeEnsembleRegressor_1(gather, op_version=1, **attrs)

        # tree - retrieve node_sample
        labels = _build_labels(tree.tree_, output="node_sample")
        ordered = list(sorted(labels.items()))
        values = [float(_[1]) for _ in ordered]
        if any(map(lambda i: int(i[0]) != i[0], ordered)):
            keys = [float(_[0]) for _ in ordered]
            node_sample = OnnxReshapeApi13(
                OnnxLabelEncoder(
                    leave, op_version=opvml, keys_floats=keys, values_floats=values
                ),
                np.array([-1, 1], dtype=np.int64),
                op_version=opv,
            )
        else:
            keys = [int(_[0]) for _ in ordered]
            values = [float(_[1]) for _ in ordered]
            node_sample = OnnxReshapeApi13(
                OnnxLabelEncoder(
                    OnnxCast(leave, op_version=opv, to=onnx_proto.TensorProto.INT64),
                    op_version=opvml,
                    keys_int64s=keys,
                    values_floats=values,
                ),
                np.array([-1, 1], dtype=np.int64),
                op_version=opv,
            )
        node_sample.set_onnx_name_prefix("node_sample%d" % i)

        # tree - retrieve path_length
        labels = _build_labels(tree.tree_, output="path_length")
        ordered = list(sorted(labels.items()))
        values = [float(_[1]) for _ in ordered]
        if any(map(lambda i: int(i[0]) != i[0], ordered)):
            keys = [float(_[0]) for _ in ordered]
            values = [float(_[1]) for _ in ordered]
            path_length = OnnxReshapeApi13(
                OnnxLabelEncoder(
                    leave, op_version=opvml, keys_floats=keys, values_floats=values
                ),
                np.array([-1, 1], dtype=np.int64),
                op_version=opv,
            )
        else:
            keys = [int(_[0]) for _ in ordered]
            path_length = OnnxReshapeApi13(
                OnnxLabelEncoder(
                    OnnxCast(leave, op_version=opv, to=onnx_proto.TensorProto.INT64),
                    op_version=opvml,
                    keys_int64s=keys,
                    values_floats=values,
                ),
                np.array([-1, 1], dtype=np.int64),
                op_version=opv,
            )
        path_length.set_onnx_name_prefix("path_length%d" % i)

        # score
        eq2 = OnnxCast(
            OnnxEqual(node_sample, np.array([2], dtype=np.float32), op_version=opv),
            to=proto_dtype,
            op_version=opv,
        )
        eq2.set_onnx_name_prefix("eq2_%d" % i)

        # 2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)

        eqp2p = OnnxCast(
            OnnxGreater(node_sample, np.array([2], dtype=np.float32), op_version=opv),
            to=proto_dtype,
            op_version=opv,
        )
        eqp2p.set_onnx_name_prefix("plus2_%d" % i)

        eqp2ps = OnnxMul(eqp2p, node_sample, op_version=opv)
        eqp2ps.set_onnx_name_prefix("eqp2ps%d" % i)

        eqp2ps_1 = OnnxAdd(eqp2ps, np.array([-1], dtype=dtype), op_version=opv)

        eqp2p_m1 = OnnxMax(eqp2ps_1, np.array([1], dtype=dtype), op_version=opv)
        eqp2p_m1.set_onnx_name_prefix("eqp2p_m1_%d" % i)

        eqp_log = OnnxMul(
            OnnxAdd(
                OnnxLog(eqp2p_m1, op_version=opv),
                np.array([np.euler_gamma], dtype=dtype),
                op_version=opv,
            ),
            np.array([2], dtype=dtype),
            op_version=opv,
        )
        eqp_log.set_onnx_name_prefix("eqp_log%d" % i)

        # - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]

        eqp2p_m0 = OnnxMax(eqp2ps_1, np.array([0], dtype=dtype), op_version=opv)
        eqp2p_m0.set_onnx_name_prefix("eqp2p_m1_%d" % i)

        eqp_ns = OnnxMul(
            OnnxDiv(
                eqp2p_m0,
                OnnxMax(eqp2ps, np.array([1], dtype=dtype), op_version=opv),
                op_version=opv,
            ),
            np.array([-2], dtype=dtype),
            op_version=opv,
        )
        eqp_ns.set_onnx_name_prefix("eqp_ns%d" % i)

        # np.ravel(node_indicator.sum(axis=1))
        # + _average_path_length(n_samples_leaf)
        # - 1.0
        av_path_length_log = OnnxMul(
            OnnxAdd(eqp_log, eqp_ns, op_version=opv), eqp2p, op_version=opv
        )
        av_path_length_log.set_onnx_name_prefix("avlog%d" % i)
        av_path_length = OnnxAdd(eq2, av_path_length_log, op_version=opv)
        av_path_length.set_onnx_name_prefix("avpl%d" % i)

        depth = OnnxAdd(
            OnnxAdd(path_length, av_path_length, op_version=opv),
            np.array([-1], dtype=dtype),
            op_version=opv,
        )
        depth.set_onnx_name_prefix("depth%d" % i)
        scores.append(depth)

    cst = len(op.estimators_) * _average_path_length([op.max_samples_])
    depths = OnnxDiv(
        OnnxSum(*scores, op_version=opv), np.array([cst], dtype=dtype), op_version=opv
    )

    # decision_function
    output_names = outputs[2].full_name if options["score_samples"] else None
    score_samples = OnnxNeg(
        OnnxPow(
            np.array([2], dtype=dtype), OnnxNeg(depths, op_version=opv), op_version=opv
        ),
        op_version=opv,
        output_names=output_names,
    )

    decision = OnnxAdd(
        score_samples,
        np.array([-op.offset_], dtype=dtype),
        op_version=opv,
        output_names=outputs[1].full_name,
    )
    decision.set_onnx_name_prefix("dec")

    less = OnnxLess(decision, np.array([0], dtype=dtype), op_version=opv)
    predict = OnnxAdd(
        OnnxMul(
            OnnxCast(less, op_version=opv, to=onnx_proto.TensorProto.INT64),
            np.array([-2], dtype=np.int64),
            op_version=opv,
        ),
        np.array([1], dtype=np.int64),
        op_version=opv,
        output_names=outputs[0].full_name,
    )
    predict.set_onnx_name_prefix("predict")

    predict.add_to(scope, container)
    less.add_to(scope, container)
    if options["score_samples"]:
        score_samples.add_to(scope, container)


def _build_labels(tree, output):
    def _recursive_build_labels(index, current):
        current[index] = True
        if tree.children_left[index] == -1:
            yield (index, current.copy())
        else:
            for it in _recursive_build_labels(tree.children_left[index], current):
                yield it
            for it in _recursive_build_labels(tree.children_right[index], current):
                yield it
        current[index] = False

    paths = {}
    current = {}

    if output == "path_length":
        for leave_index, path in _recursive_build_labels(0, current):
            spath = {}
            for nodeid, b in path.items():
                if b:
                    spath[nodeid] = 1
            paths[leave_index] = sum(spath.values())
    elif output == "node_sample":
        for leave_index, path in _recursive_build_labels(0, current):
            spath = {}
            for nodeid, b in path.items():
                if b:
                    spath[nodeid] = tree.n_node_samples[nodeid]
            paths[leave_index] = spath[leave_index]
    else:
        raise RuntimeError("Unknown method '%s'." % output)
    return paths


register_converter(
    "SklearnIsolationForest",
    convert_sklearn_isolation_forest,
    options={"score_samples": [True, False]},
)
