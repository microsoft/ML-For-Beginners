# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import (
    FloatType,
    Int64Type,
    StringType,
    TensorType,
    DoubleType,
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
    DoubleTensorType,
)
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_concat(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    for i in range(len(operator.inputs)):
        if len(operator.inputs[i].type.shape) != 2:
            operator.outputs[0].type.shape = [None, None]
            return

    N = operator.inputs[0].get_first_dimension()
    C = 0
    seen_types = []
    for i in operator.inputs:
        if C is not None:
            if isinstance(i.type, TensorType):
                if i.type.shape[1] is None:
                    C = None
                else:
                    C += i.type.shape[1]
            elif isinstance(i.type, (Int64Type, FloatType, StringType, DoubleType)):
                C += 1
            else:
                C = None
        if i.type not in seen_types:
            seen_types.append(i.type)

    def more_generic(t1, t2):
        if isinstance(t1, TensorType):
            if not isinstance(t2, TensorType):
                raise RuntimeError(
                    "Cannot merge columns with types {} and {}."
                    "Inputs:\n{}\nOutputs:\n{}".format(
                        t1, t2, operator.inputs, operator.outputs
                    )
                )
            for ts in [
                StringTensorType,
                DoubleTensorType,
                FloatTensorType,
                Int64TensorType,
                BooleanTensorType,
            ]:
                if isinstance(t1, ts) or isinstance(t2, ts):
                    return ts
            raise RuntimeError(
                "Cannot merge columns with types {} and {}."
                "Inputs:\n{}\nOutputs:\n{}".format(
                    t1, t2, operator.inputs, operator.outputs
                )
            )
        raise NotImplementedError(
            "Columns must be tensors."
            "Inputs:\n{}\nOutputs:\n{}".format(operator.inputs, operator.outputs)
        )

    # Let's determine the resulting type
    final_type = None
    for seen in seen_types:
        if final_type is None:
            final_type = seen
        elif seen.to_onnx_type() != final_type.to_onnx_type():
            merged_type = more_generic(final_type, seen)
            if isinstance(seen, merged_type):
                final_type = seen

    if final_type is None:
        raise NotImplementedError(
            "Columns must be tensors.\n"
            "- Inputs: {}\n- Outputs: {}\n- types: {}"
            "".format(operator.inputs, operator.outputs, seen_types)
        )
    if final_type != operator.outputs[0].type:
        operator.outputs[0].type = type(final_type)([N, C])
    else:
        operator.outputs[0].type.shape = [N, C]


register_shape_calculator("SklearnConcat", calculate_sklearn_concat)
register_shape_calculator("SklearnGenericUnivariateSelect", calculate_sklearn_concat)
register_shape_calculator("SklearnRFE", calculate_sklearn_concat)
register_shape_calculator("SklearnRFECV", calculate_sklearn_concat)
register_shape_calculator("SklearnSelectFdr", calculate_sklearn_concat)
register_shape_calculator("SklearnSelectFpr", calculate_sklearn_concat)
register_shape_calculator("SklearnSelectFromModel", calculate_sklearn_concat)
register_shape_calculator("SklearnSelectFwe", calculate_sklearn_concat)
register_shape_calculator("SklearnSelectKBest", calculate_sklearn_concat)
register_shape_calculator("SklearnSelectPercentile", calculate_sklearn_concat)
register_shape_calculator("SklearnVarianceThreshold", calculate_sklearn_concat)
