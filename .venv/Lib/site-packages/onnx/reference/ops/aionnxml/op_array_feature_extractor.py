# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


def _array_feature_extrator(data, indices):  # type: ignore
    """
    Implementation of operator *ArrayFeatureExtractor*.
    """
    if len(indices.shape) == 2 and indices.shape[0] == 1:
        index = indices.ravel().tolist()
        add = len(index)
    elif len(indices.shape) == 1:
        index = indices.tolist()
        add = len(index)
    else:
        add = 1
        for s in indices.shape:
            add *= s
        index = indices.ravel().tolist()
    if len(data.shape) == 1:
        new_shape = (1, add)
    else:
        new_shape = [*data.shape[:-1], add]
    try:
        tem = data[..., index]
    except IndexError as e:
        raise RuntimeError(f"data.shape={data.shape}, indices={indices}") from e
    res = tem.reshape(new_shape)
    return res


class ArrayFeatureExtractor(OpRunAiOnnxMl):
    def _run(self, data, indices):  # type: ignore
        """
        Runtime for operator *ArrayFeatureExtractor*.
        .. warning::
            ONNX specifications may be imprecise in some cases.
            When the input data is a vector (one dimension),
            the output has still two like a matrix with one row.
            The implementation follows what onnxruntime does in
            `array_feature_extractor.cc
            <https://github.com/microsoft/onnxruntime/blob/main/
            onnxruntime/core/providers/cpu/ml/array_feature_extractor.cc#L84>`_.
        """
        res = _array_feature_extrator(data, indices)
        return (res,)
