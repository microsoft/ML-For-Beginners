# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator


def calculate_sklearn_zipmap(operator):
    if len(operator.inputs) != len(operator.outputs) or len(operator.inputs) not in (
        1,
        2,
    ):
        raise RuntimeError(
            "SklearnZipMap expects the same number of inputs and outputs."
        )
    if len(operator.inputs) == 2:
        operator.outputs[0].type = operator.inputs[0].type.__class__(
            operator.inputs[0].type.shape
        )
        if operator.outputs[1].type is not None:
            operator.outputs[1].type.element_type.value_type = operator.inputs[
                1
            ].type.__class__([])


def calculate_sklearn_zipmap_columns(operator):
    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = operator.inputs[0].type.__class__(
        operator.inputs[0].type.shape
    )
    for i in range(1, len(operator.outputs)):
        operator.outputs[i].type.shape = [N]


register_shape_calculator("SklearnZipMap", calculate_sklearn_zipmap)
register_shape_calculator("SklearnZipMapColumns", calculate_sklearn_zipmap_columns)
