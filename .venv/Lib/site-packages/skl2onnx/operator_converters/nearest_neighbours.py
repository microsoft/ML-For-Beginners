# SPDX-License-Identifier: Apache-2.0


import numpy as np
from onnx.helper import make_tensor
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxArgMax,
    OnnxArrayFeatureExtractor,
    OnnxCast,
    OnnxConcat,
    OnnxDiv,
    OnnxEqual,
    OnnxFlatten,
    OnnxIdentity,
    OnnxLess,
    OnnxMatMul,
    OnnxMax,
    OnnxMul,
    OnnxNeg,
    OnnxNot,
    OnnxReciprocal,
    OnnxReduceMeanApi18,
    OnnxReduceSumApi11,
    OnnxReshapeApi13,
    OnnxShape,
    OnnxSqueezeApi11,
    OnnxSub,
    OnnxTopK_1,
    OnnxTranspose,
)

try:
    from ..algebra.onnx_ops import (
        OnnxConstantOfShape,
        OnnxCumSum,
        OnnxIsNaN,
        OnnxWhere,
    )
except ImportError:
    OnnxConstantOfShape = None
    OnnxCumSum = None
    OnnxIsNaN = None
    OnnxWhere = None
try:
    from ..algebra.onnx_ops import OnnxTopK_10
except ImportError:
    OnnxTopK_10 = None
try:
    from ..algebra.onnx_ops import OnnxTopK_11
except ImportError:
    OnnxTopK_11 = None
from ..algebra.complex_functions import onnx_cdist, _onnx_cdist_sqeuclidean
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import (
    Int64TensorType,
    DoubleTensorType,
    guess_numpy_type,
    guess_proto_type,
)
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto
from ._gp_kernels import py_make_float_array


def onnx_nearest_neighbors_indices_k(
    X,
    Y,
    k,
    metric="euclidean",
    dtype=None,
    op_version=None,
    keep_distances=False,
    optim=None,
    **kwargs
):
    """
    Retrieves the nearest neigbours *ONNX*.
    :param X: features or *OnnxOperatorMixin*
    :param Y: neighbours or *OnnxOperatorMixin*
    :param k: number of neighbours to retrieve
    :param metric: requires metric
    :param dtype: numerical type
    :param op_version: opset version
    :param keep_distance: returns the distances as well (second position)
    :param optim: implements specific optimisations,
        ``'cdist'`` replaces *Scan* operator by operator *CDist*
    :param kwargs: additional parameters for function @see fn onnx_cdist
    :return: top indices, top distances
    """
    kwargs_dist = {k: v for k, v in kwargs.items() if k == "p"}
    kwargs_topk = {k: v for k, v in kwargs.items() if k != "p"}
    if optim == "cdist":
        from skl2onnx.algebra.custom_ops import OnnxCDist

        dist = OnnxCDist(X, Y, metric=metric, op_version=op_version, **kwargs_dist)
    elif optim is None:
        dim_in = Y.shape[1] if hasattr(Y, "shape") else None
        dim_out = Y.shape[0] if hasattr(Y, "shape") else None
        dist = onnx_cdist(
            X,
            Y,
            metric=metric,
            dtype=dtype,
            op_version=op_version,
            dim_in=dim_in,
            dim_out=dim_out,
            **kwargs_dist
        )
    else:
        raise ValueError("Unknown optimisation '{}'.".format(optim))
    if op_version < 10:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_1(neg_dist, k=k, op_version=1, **kwargs_topk)
    elif op_version < 11:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_10(
            neg_dist, np.array([k], dtype=np.int64), op_version=10, **kwargs_topk
        )
    else:
        node = OnnxTopK_11(
            dist,
            np.array([k], dtype=np.int64),
            largest=0,
            sorted=1,
            op_version=11,
            **kwargs_topk
        )
        if keep_distances:
            return (
                node[1],
                OnnxMul(node[0], np.array([-1], dtype=dtype), op_version=op_version),
            )
    if keep_distances:
        return (node[1], node[0])
    return node[1]


def onnx_nearest_neighbors_indices_radius(
    X,
    Y,
    radius,
    metric="euclidean",
    dtype=None,
    op_version=None,
    keep_distances=False,
    optim=None,
    proto_dtype=None,
    **kwargs
):
    """
    Retrieves the nearest neigbours *ONNX*.
    :param X: features or *OnnxOperatorMixin*
    :param Y: neighbours or *OnnxOperatorMixin*
    :param radius: radius
    :param metric: requires metric
    :param dtype: numerical type
    :param op_version: opset version
    :param keep_distance: returns the distances as well (second position)
    :param optim: implements specific optimisations,
        ``'cdist'`` replaces *Scan* operator by operator *CDist*
    :param kwargs: additional parameters for function @see fn onnx_cdist
    :return: 3 squares matrices, indices or -1, distance or 0,
        based on the fact that the distance is below the radius,
        binary weights
    """
    opv = op_version
    if optim == "cdist":
        from skl2onnx.algebra.custom_ops import OnnxCDist

        dist = OnnxCDist(X, Y, metric=metric, op_version=op_version, **kwargs)
    elif optim is None:
        dim_in = Y.shape[1] if hasattr(Y, "shape") else None
        dim_out = Y.shape[0] if hasattr(Y, "shape") else None
        dist = onnx_cdist(
            X,
            Y,
            metric=metric,
            dtype=dtype,
            op_version=op_version,
            dim_in=dim_in,
            dim_out=dim_out,
            **kwargs
        )
    else:
        raise ValueError("Unknown optimisation '{}'.".format(optim))

    less = OnnxLess(dist, np.array([radius], dtype=dtype), op_version=opv)
    less.set_onnx_name_prefix("cond")
    shape = OnnxShape(dist, op_version=opv)
    zero = OnnxCast(
        OnnxConstantOfShape(shape, op_version=opv), op_version=opv, to=proto_dtype
    )
    tensor_value = py_make_float_array(-1, dtype=np.float32, as_tensor=True)
    minus = OnnxCast(
        OnnxConstantOfShape(shape, op_version=opv, value=tensor_value),
        op_version=opv,
        to=onnx_proto.TensorProto.INT64,
    )
    minus_range = OnnxAdd(
        OnnxNeg(
            OnnxCumSum(minus, np.array([1], dtype=np.int64), op_version=opv),
            op_version=opv,
        ),
        minus,
        op_version=opv,
    )
    minus_range.set_onnx_name_prefix("arange")

    dist_only = OnnxWhere(less, dist, zero, op_version=opv)
    dist_only.set_onnx_name_prefix("nndist")
    indices = OnnxWhere(less, minus_range, minus, op_version=opv)
    indices.set_onnx_name_prefix("nnind")
    binary = OnnxCast(less, to=proto_dtype, op_version=opv)
    binary.set_onnx_name_prefix("nnbin")
    return indices, dist_only, binary


def _convert_nearest_neighbors(operator, container, k=None, radius=None):
    """
    Common parts to regressor and classifier. Let's denote
    *N* as the number of observations, *k*
    the number of neighbours. It returns
    the following intermediate results:

    top_indices: [N, k] (int64), best indices for
        every observation
    top_distances: [N, k] (dtype), float distances
        for every observation, it can be None
        if the weights are uniform
    top_labels: [N, k] (label type), labels
        associated to every top index
    weights: [N, k] (dtype), if top_distances is not None,
        returns weights
    norm: [N] (dtype), if top_distances is not None,
        returns normalized weights
    axis: 1 if there is one dimension only, 2 if
        this is a multi-regression or a multi classification
    """
    X = operator.inputs[0]
    op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    proto_type = guess_proto_type(X.type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT

    if isinstance(X.type, Int64TensorType):
        X = OnnxCast(X, to=proto_type, op_version=opv)

    options = container.get_options(op, dict(optim=None))

    single_reg = (
        not hasattr(op, "_y")
        or len(op._y.shape) == 1
        or len(op._y.shape) == 2
        and op._y.shape[1] == 1
    )
    ndim = 1 if single_reg else op._y.shape[1]

    metric = op.effective_metric_ if hasattr(op, "effective_metric_") else op.metric
    neighb = op._fit_X.astype(dtype)

    if (
        hasattr(op, "n_neighbors")
        and op.n_neighbors is not None
        and hasattr(op, "radius")
        and op.radius is not None
    ):
        raise RuntimeError(
            "The model defines radius and n_neighbors at the "
            "same time ({} and {}). "
            "This case is not supported.".format(op.radius, op.n_neighbors)
        )

    if hasattr(op, "n_neighbors") and op.n_neighbors is not None:
        k = op.n_neighbors if k is None else k
        radius = None
    elif hasattr(op, "radius") and op.radius is not None:
        k = None
        radius = op.radius if radius is None else radius
    else:
        raise RuntimeError("Cannot convert class '{}'.".format(op.__class__.__name__))

    training_labels = op._y if hasattr(op, "_y") else None
    distance_kwargs = {}
    if metric == "minkowski":
        if op.p != 2:
            distance_kwargs["p"] = op.p
        else:
            metric = "euclidean"

    weights = op.weights if hasattr(op, "weights") else "distance"
    binary = None
    if weights == "uniform" and radius is None:
        top_indices = onnx_nearest_neighbors_indices_k(
            X,
            neighb,
            k,
            metric=metric,
            dtype=dtype,
            op_version=opv,
            optim=options.get("optim", None),
            **distance_kwargs
        )
        top_distances = None
    elif radius is not None:
        three = onnx_nearest_neighbors_indices_radius(
            X,
            neighb,
            radius,
            metric=metric,
            dtype=dtype,
            op_version=opv,
            keep_distances=True,
            proto_dtype=proto_type,
            optim=options.get("optim", None),
            **distance_kwargs
        )
        top_indices, top_distances, binary = three
    elif weights == "distance":
        top_indices, top_distances = onnx_nearest_neighbors_indices_k(
            X,
            neighb,
            k,
            metric=metric,
            dtype=dtype,
            op_version=opv,
            keep_distances=True,
            optim=options.get("optim", None),
            **distance_kwargs
        )
    else:
        raise RuntimeError(
            "Unable to convert KNeighborsRegressor when weights is callable."
        )

    if training_labels is not None:
        if ndim > 1:
            training_labels = training_labels.T
            axis = 2
        else:
            training_labels = training_labels.ravel()
            axis = 1
        if opv >= 9:
            kor = k if k is not None else training_labels.shape[-1]
            if ndim > 1:
                shape = np.array([ndim, -1, kor], dtype=np.int64)
            else:
                shape = np.array([-1, kor], dtype=np.int64)
        else:
            raise RuntimeError(
                "Conversion of a KNeighborsRegressor for multi regression "
                "requires opset >= 9."
            )

        if training_labels.dtype == np.int32:
            training_labels = training_labels.astype(np.int64)
        flattened = OnnxFlatten(top_indices, op_version=opv)
        extracted = OnnxArrayFeatureExtractor(
            training_labels, flattened, op_version=opv
        )
        reshaped = OnnxReshapeApi13(extracted, shape, op_version=opv)

        if ndim > 1:
            reshaped = OnnxTranspose(reshaped, op_version=opv, perm=[1, 0, 2])
        reshaped.set_onnx_name_prefix("knny")

    else:
        reshaped = None
        axis = 1

    if binary is not None:
        if op.weights == "uniform":
            wei = binary
        else:
            modified = OnnxMax(
                top_distances, np.array([1e-6], dtype=dtype), op_version=opv
            )
            wei = OnnxMul(
                binary, OnnxReciprocal(modified, op_version=opv), op_version=opv
            )
        norm = OnnxReduceSumApi11(wei, op_version=opv, axes=[1], keepdims=0)
    elif top_distances is not None:
        modified = OnnxMax(top_distances, np.array([1e-6], dtype=dtype), op_version=opv)
        wei = OnnxReciprocal(modified, op_version=opv)
        norm = OnnxReduceSumApi11(wei, op_version=opv, axes=[1], keepdims=0)
    else:
        norm = None
        wei = None

    if wei is not None:
        wei.set_onnx_name_prefix("wei")
    if norm is not None:
        norm.set_onnx_name_prefix("norm")
    return top_indices, top_distances, reshaped, wei, norm, axis


def convert_nearest_neighbors_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNeighborsRegressor* into *ONNX*.
    The converted model may return different predictions depending
    on how the runtime select the topk element.
    *scikit-learn* uses function `argpartition
    <https://docs.scipy.org/doc/numpy/reference/
    generated/numpy.argpartition.html>`_ which keeps the
    original order of the elements.
    """
    many = _convert_nearest_neighbors(operator, container)
    _, top_distances, reshaped, wei, norm, axis = many
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT

    opv = container.target_opset
    out = operator.outputs

    reshaped_cast = OnnxCast(reshaped, to=proto_type, op_version=opv)
    if top_distances is not None:
        # Multi-target
        if (
            hasattr(operator.raw_operator, "_y")
            and len(operator.raw_operator._y.shape) > 1
            and operator.raw_operator._y.shape[1] > 1
        ):
            rs = OnnxTranspose(reshaped_cast, perm=[1, 0, 2], op_version=opv)
            weighted_rs = OnnxMul(rs, wei, op_version=opv)
            weighted = OnnxTranspose(weighted_rs, perm=[1, 0, 2], op_version=opv)

            if OnnxIsNaN is not None:
                # This steps sometimes produces nan (bug in onnxuntime)
                # They are replaced by null values.
                isnan = OnnxIsNaN(weighted, op_version=opv)
                shape = OnnxShape(weighted, op_version=opv)
                csts0 = OnnxConstantOfShape(shape, op_version=opv)
                weighted = OnnxWhere(isnan, csts0, weighted, op_version=opv)
                # Back to original plan.

            res = OnnxReduceSumApi11(weighted, axes=[axis], op_version=opv, keepdims=0)
            norm2 = OnnxReshapeApi13(
                norm, np.array([-1, 1], dtype=np.int64), op_version=opv
            )
            res = OnnxDiv(res, norm2, op_version=opv, output_names=out)
        else:
            weighted = OnnxMul(reshaped_cast, wei, op_version=opv)
            res = OnnxReduceSumApi11(weighted, axes=[axis], op_version=opv, keepdims=0)
            res.set_onnx_name_prefix("final")
            if opv >= 12:
                shape = OnnxShape(res, op_version=opv)
                norm = OnnxReshapeApi13(norm, shape, op_version=opv)
                norm.set_onnx_name_prefix("normr")
            res = OnnxDiv(res, norm, op_version=opv)
            res = OnnxReshapeApi13(
                res, np.array([-1, 1], dtype=np.int64), output_names=out, op_version=opv
            )
    else:
        if (
            hasattr(operator.raw_operator, "_y")
            and len(np.squeeze(operator.raw_operator._y).shape) == 1
        ):
            keepdims = 1
        elif operator.raw_operator.n_neighbors == 1:
            keepdims = 0
        else:
            keepdims = 0
        res = OnnxReduceMeanApi18(
            reshaped_cast,
            axes=[axis],
            op_version=opv,
            keepdims=keepdims,
            output_names=out,
        )
    res.add_to(scope, container)


def get_proba_and_label(
    container, nb_classes, reshaped, wei, axis, opv, proto_type, keep_axis=True
):
    """
    This function calculates the label by choosing majority label
    amongst the nearest neighbours.
    """
    conc = []
    for cl in range(nb_classes):
        cst = np.array([cl], dtype=np.int64)
        mat_cast = OnnxCast(
            OnnxEqual(reshaped, cst, op_version=opv), op_version=opv, to=proto_type
        )
        if wei is not None:
            if not keep_axis:
                mat_cast = OnnxSqueezeApi11(mat_cast, axes=[-1], op_version=opv)
            mat_cast = OnnxMul(mat_cast, wei, op_version=opv)
        wh = OnnxReduceSumApi11(mat_cast, axes=[1], op_version=opv)
        conc.append(wh)
    all_together = OnnxConcat(*conc, axis=1, op_version=opv)
    sum_prob = OnnxReduceSumApi11(all_together, axes=[1], op_version=opv, keepdims=1)
    res = OnnxArgMax(all_together, axis=axis, op_version=opv, keepdims=0)
    return all_together, sum_prob, res


def convert_nearest_neighbors_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNeighborsClassifier* into *ONNX*.
    The converted model may return different predictions depending
    on how the runtime select the topk element.
    *scikit-learn* uses function `argpartition
    <https://docs.scipy.org/doc/numpy/reference/
    generated/numpy.argpartition.html>`_ which keeps the
    original order of the elements.
    """
    many = _convert_nearest_neighbors(operator, container)
    _, __, reshaped, wei, ___, axis = many

    opv = container.target_opset
    out = operator.outputs
    op = operator.raw_operator
    nb_classes = len(op.classes_)
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT

    if axis == 0:
        raise RuntimeError(
            "Binary classification not implemented in scikit-learn. "
            "Check this code is not reused for other libraries."
        )
    classes = get_label_classes(scope, op)
    if hasattr(classes, "dtype") and (
        np.issubdtype(classes.dtype, np.floating) or classes.dtype == np.bool_
    ):
        classes = classes.astype(np.int32)
        is_integer = True
    elif isinstance(classes[0], (int, np.int32, np.int64)):
        is_integer = True
    else:
        is_integer = False
    if isinstance(op.classes_, list) and isinstance(op.classes_[0], np.ndarray):
        # Multi-label
        out_labels, out_probas = [], []
        for index, cur_class in enumerate(op.classes_):
            transpose_result = OnnxTranspose(reshaped, op_version=opv, perm=[0, 2, 1])
            extracted_name = OnnxArrayFeatureExtractor(
                transpose_result, np.array([index], dtype=np.int64), op_version=opv
            )
            extracted_name.set_onnx_name_prefix("tr%d" % index)
            all_together, sum_prob, res = get_proba_and_label(
                container,
                len(cur_class),
                extracted_name,
                wei,
                1,
                opv,
                proto_type,
                keep_axis=False,
            )
            probas = OnnxDiv(all_together, sum_prob, op_version=opv)
            res_name = OnnxArrayFeatureExtractor(cur_class, res, op_version=opv)
            res_name.set_onnx_name_prefix("div%d" % index)
            reshaped_labels = OnnxReshapeApi13(
                res_name, np.array([-1, 1], dtype=np.int64), op_version=opv
            )
            reshaped_probas = OnnxReshapeApi13(
                probas,
                np.array([1, -1, len(cur_class)], dtype=np.int64),
                op_version=opv,
            )
            out_labels.append(reshaped_labels)
            out_probas.append(reshaped_probas)
        concatenated_labels = OnnxConcat(*out_labels, axis=1, op_version=opv)
        final_proba = OnnxConcat(
            *out_probas, axis=0, output_names=out[1:], op_version=opv
        )
        final_label = OnnxCast(
            concatenated_labels,
            to=onnx_proto.TensorProto.INT64,
            output_names=out[:1],
            op_version=opv,
        )
        final_label.add_to(scope, container)
        final_proba.add_to(scope, container)
    else:
        all_together, sum_prob, res = get_proba_and_label(
            container, nb_classes, reshaped, wei, axis, opv, proto_type
        )
        probas = OnnxDiv(all_together, sum_prob, op_version=opv, output_names=out[1:])
        probas.set_onnx_name_prefix("bprob")
        res_name = OnnxArrayFeatureExtractor(classes, res, op_version=opv)
        if is_integer:
            res_name = OnnxCast(
                res_name, to=onnx_proto.TensorProto.INT64, op_version=opv
            )
        out_labels = OnnxReshapeApi13(
            res_name,
            np.array([-1], dtype=np.int64),
            output_names=out[:1],
            op_version=opv,
        )
        out_labels.set_onnx_name_prefix("blab")
        out_labels.add_to(scope, container)
        probas.add_to(scope, container)


def convert_nearest_neighbors_transform(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *NearestNeighbors* into *ONNX*.
    """
    many = _convert_nearest_neighbors(operator, container)
    top_indices, top_distances = many[:2]
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32

    out = operator.outputs

    ind = OnnxIdentity(
        top_indices, output_names=out[:1], op_version=container.target_opset
    )
    dist = OnnxMul(
        top_distances,
        np.array([-1], dtype=dtype),
        output_names=out[1:],
        op_version=container.target_opset,
    )

    dist.add_to(scope, container)
    ind.add_to(scope, container)


def convert_k_neighbours_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNeighborsTransformer* into *ONNX*.
    """
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    transformer_op = operator.raw_operator
    op_version = container.target_opset
    k = (
        transformer_op.n_neighbors + 1
        if transformer_op.mode == "distance"
        else transformer_op.n_neighbors
    )
    out = operator.outputs

    many = _convert_nearest_neighbors(operator, container, k=k)
    top_indices, top_dist = many[:2]
    top_dist = (
        OnnxReshapeApi13(
            OnnxMul(top_dist, np.array([-1], dtype=dtype), op_version=op_version),
            np.array([-1, 1, k], dtype=np.int64),
            op_version=op_version,
        )
        if transformer_op.mode == "distance"
        else None
    )
    fit_samples_indices = np.array(
        np.arange(transformer_op.n_samples_fit_).reshape((1, -1, 1)), dtype=np.int64
    )
    reshaped_ind = OnnxReshapeApi13(
        top_indices, np.array([-1, 1, k], dtype=np.int64), op_version=op_version
    )
    comparison_res = OnnxCast(
        OnnxEqual(fit_samples_indices, reshaped_ind, op_version=op_version),
        op_version=op_version,
        to=proto_type,
    )
    if top_dist:
        comparison_res = OnnxMul(comparison_res, top_dist, op_version=op_version)
    res = OnnxReduceSumApi11(
        comparison_res,
        op_version=op_version,
        axes=[2],
        keepdims=0,
        output_names=out[:1],
    )
    res.add_to(scope, container)


def _nan_euclidean_distance(
    container, model, input_name, op_version, optim, dtype, proto_type
):
    training_data = model._fit_X.astype(dtype)
    shape = OnnxShape(input_name, op_version=op_version)
    zero = OnnxConstantOfShape(
        shape, value=make_tensor("value", proto_type, (1,), [0]), op_version=op_version
    )
    missing_input_name = OnnxIsNaN(input_name, op_version=op_version)
    masked_input_name = OnnxWhere(
        missing_input_name, zero, input_name, op_version=op_version
    )
    missing_y = np.isnan(training_data)
    training_data[missing_y] = 0
    d_in = training_data.shape[1] if hasattr(training_data, "shape") else None
    d_out = training_data.shape[0] if hasattr(training_data, "shape") else None

    if optim is None:
        dist = _onnx_cdist_sqeuclidean(
            masked_input_name,
            training_data,
            dtype=dtype,
            op_version=container.target_opset,
            dim_in=d_in,
            dim_out=d_out,
        )
    elif optim == "cdist":
        from skl2onnx.algebra.custom_ops import OnnxCDist

        dist = OnnxCDist(
            masked_input_name,
            training_data,
            metric="sqeuclidean",
            op_version=container.target_opset,
        )
    else:
        raise RuntimeError("Unexpected optimization '{}'.".format(optim))
    dist1 = OnnxMatMul(
        OnnxMul(masked_input_name, masked_input_name, op_version=op_version),
        missing_y.T.astype(dtype),
        op_version=op_version,
    )
    dist2 = OnnxMatMul(
        OnnxCast(missing_input_name, to=proto_type, op_version=op_version),
        (training_data * training_data).T.astype(dtype),
        op_version=op_version,
    )
    distances = OnnxSub(
        dist, OnnxAdd(dist1, dist2, op_version=op_version), op_version=op_version
    )
    present_x = OnnxSub(
        np.array([1], dtype=dtype),
        OnnxCast(missing_input_name, to=proto_type, op_version=op_version),
        op_version=op_version,
    )
    present_y = (1.0 - missing_y).astype(dtype)
    present_count = OnnxMatMul(
        present_x, present_y.T.astype(dtype), op_version=op_version
    )
    present_count = OnnxMax(
        np.array([1], dtype=dtype), present_count, op_version=op_version
    )
    dist = OnnxDiv(distances, present_count, op_version=op_version)
    return (
        OnnxMul(dist, np.array([d_in], dtype=dtype), op_version=op_version),
        missing_input_name,
    )


def _nearest_neighbours(
    container, model, input_name, op_version, optim, dtype, proto_type, **kwargs
):
    dist, missing_input_name = _nan_euclidean_distance(
        container, model, input_name, op_version, optim, dtype, proto_type
    )
    if op_version < 10:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_1(neg_dist, k=model.n_neighbors, op_version=1, **kwargs)
    elif op_version < 11:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_10(
            neg_dist,
            np.array([model.n_neighbors], dtype=np.int64),
            op_version=10,
            **kwargs
        )
    else:
        node = OnnxTopK_11(
            dist,
            np.array([model.n_neighbors], dtype=np.int64),
            largest=0,
            sorted=1,
            op_version=11,
            **kwargs
        )
    return node[1], missing_input_name


def convert_knn_imputer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNNImputer* into *ONNX*.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT
    knn_op = operator.raw_operator
    if knn_op.metric != "nan_euclidean":
        raise RuntimeError("Unable to convert KNNImputer when metric is callable.")
    if knn_op.weights not in ("uniform", "distance"):
        raise RuntimeError("Unable to convert KNNImputer when weights is callable.")
    if knn_op.weights == "distance":
        raise NotImplementedError(
            "KNNImputer with distance as metric is not supported, "
            "you may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )
    options = container.get_options(knn_op, dict(optim=None))
    op_version = container.target_opset
    input_name = operator.inputs[0]
    training_data = knn_op._fit_X.astype(dtype)
    training_data[np.isnan(training_data)] = 0
    out = operator.outputs
    top_indices, missing_input_name = _nearest_neighbours(
        container, knn_op, input_name, op_version, options["optim"], dtype, proto_type
    )
    flattened = OnnxFlatten(top_indices, op_version=op_version)
    extracted = OnnxArrayFeatureExtractor(
        training_data.T, flattened, op_version=op_version
    )
    reshaped = OnnxReshapeApi13(
        extracted,
        np.array([training_data.shape[1], -1, knn_op.n_neighbors], dtype=np.int64),
        op_version=op_version,
    )
    transpose_result = OnnxTranspose(reshaped, op_version=op_version, perm=[1, 2, 0])
    reduced = OnnxReduceSumApi11(
        transpose_result, op_version=op_version, axes=[1], keepdims=0
    )
    cast_res = OnnxCast(
        OnnxCast(
            transpose_result, to=onnx_proto.TensorProto.BOOL, op_version=op_version
        ),
        to=proto_type,
        op_version=op_version,
    )
    deno = OnnxReduceSumApi11(cast_res, op_version=op_version, axes=[1], keepdims=0)
    deno_updated = OnnxAdd(
        deno,
        OnnxCast(
            OnnxNot(
                OnnxCast(deno, to=onnx_proto.TensorProto.BOOL, op_version=op_version),
                op_version=op_version,
            ),
            to=proto_type,
            op_version=op_version,
        ),
        op_version=op_version,
    )
    imputed_out = OnnxWhere(
        missing_input_name,
        OnnxDiv(reduced, deno_updated, op_version=op_version),
        input_name,
        output_names=out[:1],
        op_version=op_version,
    )
    imputed_out.add_to(scope, container)


def convert_nca(scope: Scope, operator: Operator, container: ModelComponentContainer):
    """
    Converts *NeighborhoodComponentsAnalysis* into *ONNX*.
    """
    X = operator.inputs[0]
    nca_op = operator.raw_operator
    op_version = container.target_opset
    out = operator.outputs
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    components = nca_op.components_.T.astype(dtype)

    if isinstance(X.type, Int64TensorType):
        X = OnnxCast(X, to=onnx_proto.TensorProto.FLOAT, op_version=op_version)
    elif isinstance(X.type, DoubleTensorType):
        components = OnnxCast(
            components, to=onnx_proto.TensorProto.DOUBLE, op_version=op_version
        )
    else:
        components = components.astype(dtype)
    res = OnnxMatMul(X, components, output_names=out[:1], op_version=op_version)
    res.add_to(scope, container)


register_converter(
    "SklearnKNeighborsClassifier",
    convert_nearest_neighbors_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "raw_scores": [True, False],
        "output_class_labels": [False, True],
        "optim": [None, "cdist"],
    },
)
register_converter(
    "SklearnRadiusNeighborsClassifier",
    convert_nearest_neighbors_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "raw_scores": [True, False],
        "output_class_labels": [False, True],
        "optim": [None, "cdist"],
    },
)
register_converter(
    "SklearnKNeighborsRegressor",
    convert_nearest_neighbors_regressor,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnRadiusNeighborsRegressor",
    convert_nearest_neighbors_regressor,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnKNeighborsTransformer",
    convert_k_neighbours_transformer,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnNearestNeighbors",
    convert_nearest_neighbors_transform,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnKNNImputer", convert_knn_imputer, options={"optim": [None, "cdist"]}
)
register_converter("SklearnNeighborhoodComponentsAnalysis", convert_nca)
