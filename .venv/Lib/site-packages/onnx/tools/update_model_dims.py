# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Set

import onnx.checker
from onnx import ModelProto, ValueInfoProto


def update_inputs_outputs_dims(
    model: ModelProto,
    input_dims: Dict[str, List[Any]],
    output_dims: Dict[str, List[Any]],
) -> ModelProto:
    """
    This function updates the dimension sizes of the model's inputs and outputs to the values
    provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
    will be set for that dimension.

    Example. if we have the following shape for inputs and outputs:

    * shape(input_1) = ('b', 3, 'w', 'h')
    * shape(input_2) = ('b', 4)
    * shape(output)  = ('b', 'd', 5)

    The parameters can be provided as:

    ::

        input_dims = {
            "input_1": ['b', 3, 'w', 'h'],
            "input_2": ['b', 4],
        }
        output_dims = {
            "output": ['b', -1, 5]
        }

    Putting it together:

    ::

        model = onnx.load('model.onnx')
        updated_model = update_inputs_outputs_dims(model, input_dims, output_dims)
        onnx.save(updated_model, 'model.onnx')
    """
    dim_param_set: Set[str] = set()

    def init_dim_param_set(
        dim_param_set: Set[str], value_infos: List[ValueInfoProto]
    ) -> None:
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField("dim_param"):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

    def update_dim(tensor: ValueInfoProto, dim: Any, j: int, name: str) -> None:
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
                    raise ValueError(
                        f"Unable to set dimension value to {dim} for axis {j} of {name}. Contradicts existing dimension value {dim_proto.dim_value}."
                    )
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + "_" + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError(
                        f"Unable to generate unique dim_param for axis {j} of {name}. Please manually provide a dim_param value."
                    )
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError(
                f"Only int or str is accepted as dimension value, incorrect type: {type(dim)}"
            )

    for input_ in model.graph.input:
        input_name = input_.name
        input_dim_arr = input_dims[input_name]
        for j, dim in enumerate(input_dim_arr):
            update_dim(input_, dim, j, input_name)

    for output in model.graph.output:
        output_name = output.name
        output_dim_arr = output_dims[output_name]
        for j, dim in enumerate(output_dim_arr):
            update_dim(output, dim, j, output_name)

    onnx.checker.check_model(model)
    return model
