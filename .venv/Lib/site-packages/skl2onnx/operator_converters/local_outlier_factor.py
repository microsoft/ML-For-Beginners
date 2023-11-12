# SPDX-License-Identifier: Apache-2.0

import warnings
import numpy as np
from onnx import TensorProto
from ..common._registration import register_converter
from ..common.data_types import (
    BooleanTensorType,
    Int64TensorType,
    guess_numpy_type,
    guess_proto_type,
)
from ..algebra.onnx_ops import (
    OnnxCast,
    OnnxLess,
    OnnxMul,
    OnnxAdd,
    OnnxDiv,
    OnnxGather,
    OnnxReduceMeanApi18,
    OnnxMax,
    OnnxSqueezeApi11,
)
from .nearest_neighbours import onnx_nearest_neighbors_indices_k


def convert_sklearn_local_outlier_factor(
    scope,
    operator,
    container,
    op_type="TreeEnsembleRegressor",
    op_domain="ai.onnx.ml",
    op_version=1,
):
    op = operator.raw_operator
    if not op.novelty:
        raise RuntimeError(
            "The converter only converts the model %r is novelty is True." "" % type(op)
        )
    outputs = operator.outputs
    opv = container.target_opset
    options = container.get_options(op, dict(score_samples=None, optim=None))

    X = operator.inputs[0]
    dtype = guess_numpy_type(operator.inputs[0].type)
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        X = OnnxCast(X, to=proto_dtype, op_version=opv)

    metric = op.effective_metric_ if hasattr(op, "effective_metric_") else op.metric
    neighb = op._fit_X.astype(dtype)
    k = op.n_neighbors_
    kwargs = {}
    if op.p != 2:
        if options["optim"] == "cdist":
            warnings.warn(
                "Option p=%r may not be compatible with the runtime. "
                "See https://github.com/microsoft/onnxruntime/blob/master/"
                "docs/ContribOperators.md#com.microsoft.CDist."
            )
        kwargs["p"] = op.p

    top_k, dist = onnx_nearest_neighbors_indices_k(
        X,
        neighb,
        k,
        metric,
        dtype=dtype,
        op_version=opv,
        keep_distances=True,
        optim=options.get("optim", None),
        **kwargs
    )

    # dist_k = self._distances_fit_X_[neighbors_indices, self.n_neighbors_ - 1]
    # reach_dist_array = np.maximum(distances_X, dist_k)
    dist_k_ = OnnxGather(op._distances_fit_X_.astype(dtype), top_k, op_version=opv)
    dist_k = OnnxSqueezeApi11(
        OnnxGather(
            dist_k_,
            np.array([op.n_neighbors_ - 1], dtype=np.int64),
            axis=2,
            op_version=opv,
        ),
        axes=[2],
        op_version=opv,
    )
    dist_k.set_onnx_name_prefix("dist_k")
    reach_dist_array = OnnxMax(
        OnnxMul(dist, np.array([-1], dtype=dtype), op_version=opv),
        dist_k,
        op_version=opv,
    )

    # X_lrd=  return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)
    X_lrd = OnnxDiv(
        np.array([1], dtype=dtype),
        OnnxAdd(
            OnnxReduceMeanApi18(reach_dist_array, axes=[1], op_version=opv, keepdims=1),
            np.array([1e-10], dtype=dtype),
            op_version=opv,
        ),
        op_version=opv,
    )
    X_lrd.set_onnx_name_prefix("X_lrd")

    # lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]
    lrd_ratios_array = OnnxDiv(
        OnnxGather(op._lrd.astype(dtype), top_k, op_version=opv), X_lrd, op_version=opv
    )
    lrd_ratios_array.set_onnx_name_prefix("lrd_ratios_array")

    # -np.mean(lrd_ratios_array, axis=1)
    if options["score_samples"]:
        output_names_score_samples = [outputs[2]]
    else:
        output_names_score_samples = None
    score_samples = OnnxReduceMeanApi18(lrd_ratios_array, axes=[1], op_version=opv)
    score_samples.set_onnx_name_prefix("score_samples")
    score_samples_neg = OnnxMul(
        score_samples,
        np.array([-1], dtype=dtype),
        op_version=opv,
        output_names=output_names_score_samples,
    )
    final = OnnxAdd(
        score_samples_neg,
        np.array([-op.offset_], dtype=dtype),
        op_version=opv,
        output_names=[outputs[1]],
    )

    # labels
    # is_inlier = np.ones(X.shape[0], dtype=int)
    # is_inlier[self.decision_function(X) < 0] = -1

    predict = OnnxAdd(
        OnnxMul(
            OnnxCast(
                OnnxLess(final, np.array([0], dtype=dtype), op_version=opv),
                to=TensorProto.INT64,
                op_version=opv,
            ),
            np.array([-2], dtype=np.int64),
            op_version=opv,
        ),
        np.array([1], dtype=np.int64),
        op_version=opv,
        output_names=outputs[0].full_name,
    )
    predict.set_onnx_name_prefix("predict")

    predict.add_to(scope, container)
    final.add_to(scope, container)
    if options["score_samples"]:
        score_samples_neg.add_to(scope, container)


register_converter(
    "SklearnLocalOutlierFactor",
    convert_sklearn_local_outlier_factor,
    options={"score_samples": [True, False], "optim": [None, "cdist"]},
)
