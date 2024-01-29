# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..common._apply_operation import apply_cast, apply_concat, apply_reshape
from ..common.data_types import (
    Int64TensorType,
    StringTensorType,
    Int32TensorType,
    FloatTensorType,
    DoubleTensorType,
)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto


def convert_sklearn_one_hot_encoder(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *OneHotEncoder* into ONNX.
    It supports multiple inputs of types
    string or int64.
    """
    ohe_op = operator.raw_operator

    if len(operator.inputs) > 1:
        all_shapes = [inp.type.shape[1] for inp in operator.inputs]
        if any(map(lambda x: not isinstance(x, int) or x < 1, all_shapes)):
            raise RuntimeError(
                "Shapes must be known when OneHotEncoder is converted. "
                "There are {} inputs with the following number of columns "
                "{}.".format(len(operator.inputs), all_shapes)
            )
        total = sum(all_shapes)
        if total != len(ohe_op.categories_):
            raise RuntimeError(
                "Mismatch between the number of sets of categories {} and "
                "the total number of inputs columns {}.".format(
                    len(ohe_op.categories_), total
                )
            )

        enum_cats = []
        index_inputs = 0

        for index, cats in enumerate(ohe_op.categories_):
            while sum(all_shapes[: index_inputs + 1]) <= index:
                index_inputs += 1
            index_in_input = index - sum(all_shapes[:index_inputs])

            inp = operator.inputs[index_inputs]
            if not isinstance(
                inp.type,
                (
                    Int64TensorType,
                    StringTensorType,
                    Int32TensorType,
                    FloatTensorType,
                    DoubleTensorType,
                ),
            ):
                raise NotImplementedError(
                    "{} input datatype not yet supported. "
                    "You may raise an issue at "
                    "https://github.com/onnx/sklearn-onnx/issues"
                    "".format(type(inp.type))
                )

            if all_shapes[index_inputs] == 1:
                assert index_in_input == 0
                afeat = False
            else:
                afeat = True
            enum_cats.append((afeat, index_in_input, inp.full_name, cats, inp.type))
    else:
        inp = operator.inputs[0]
        enum_cats = [
            (True, i, inp.full_name, cats, inp.type)
            for i, cats in enumerate(ohe_op.categories_)
        ]

    result, categories_len = [], 0
    for index, enum_c in enumerate(enum_cats):
        afeat, index_in, name, categories, inp_type = enum_c
        container.debug(
            "[conv.OneHotEncoder] cat %r/%r name=%r type=%r",
            index + 1,
            len(enum_cats),
            name,
            inp_type,
        )
        if len(categories) == 0:
            continue
        if afeat:
            index_name = scope.get_unique_variable_name(name + str(index_in))
            container.add_initializer(
                index_name, onnx_proto.TensorProto.INT64, [1], [index_in]
            )
            out_name = scope.get_unique_variable_name(name + str(index_in))
            container.add_node(
                "Gather",
                [name, index_name],
                out_name,
                axis=1,
                name=scope.get_unique_operator_name("Gather"),
            )
            name = out_name

        attrs = {"name": scope.get_unique_operator_name("OneHotEncoder")}
        attrs["zeros"] = 1 if ohe_op.handle_unknown == "ignore" else 0

        if isinstance(inp_type, (Int64TensorType, Int32TensorType)):
            attrs["cats_int64s"] = categories.astype(np.int64)
        elif isinstance(inp_type, StringTensorType):
            attrs["cats_strings"] = np.array(
                [str(s).encode("utf-8") for s in categories]
            )
        elif isinstance(inp_type, (FloatTensorType, DoubleTensorType)):
            # The converter checks that categories can be casted into
            # integers. String is not allowed here.
            # Input type is casted into int64.
            for c in categories:
                try:
                    ci = int(c)
                except TypeError:
                    raise RuntimeError(
                        "Category '{}' cannot be casted into int.".format(c)
                    )
                if ci != c:
                    raise RuntimeError(
                        "Category %r is not an int64. "
                        "The converter only supports string and int64 "
                        "categories not %r." % (c, type(c))
                    )
            attrs["cats_int64s"] = categories.astype(np.int64)
        else:
            raise RuntimeError(
                "Input type {} is not supported for OneHotEncoder. "
                "Ideally, it should either be integer or strings.".format(inp_type)
            )

        ohe_output = scope.get_unique_variable_name(name + "out")

        if "cats_int64s" in attrs:
            # Let's cast this input in int64.
            cast_feature = scope.get_unique_variable_name(name + "cast")
            apply_cast(
                scope, name, cast_feature, container, to=onnx_proto.TensorProto.INT64
            )
            name = cast_feature

        container.add_node(
            "OneHotEncoder", name, ohe_output, op_domain="ai.onnx.ml", **attrs
        )
        if (
            hasattr(ohe_op, "drop_idx_")
            and ohe_op.drop_idx_ is not None
            and ohe_op.drop_idx_[index] is not None
        ):
            extracted_outputs_name = scope.get_unique_variable_name("extracted_outputs")
            indices_to_keep_name = scope.get_unique_variable_name("indices_to_keep")
            indices_to_keep = np.delete(
                np.arange(len(categories)), ohe_op.drop_idx_[index]
            )
            container.add_initializer(
                indices_to_keep_name,
                onnx_proto.TensorProto.INT64,
                indices_to_keep.shape,
                indices_to_keep,
            )
            container.add_node(
                "Gather",
                [ohe_output, indices_to_keep_name],
                extracted_outputs_name,
                axis=-1,
                name=scope.get_unique_operator_name("Gather"),
            )
            ohe_output, categories = extracted_outputs_name, indices_to_keep

        result.append(ohe_output)
        categories_len += len(categories)

    concat_result_name = scope.get_unique_variable_name("concat_result")
    apply_concat(scope, result, concat_result_name, container, axis=-1)

    reshape_input = concat_result_name
    if np.issubdtype(ohe_op.dtype, np.signedinteger):
        reshape_input = scope.get_unique_variable_name("cast")
        apply_cast(
            scope,
            concat_result_name,
            reshape_input,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    apply_reshape(
        scope,
        reshape_input,
        operator.output_full_names,
        container,
        desired_shape=(-1, categories_len),
    )


register_converter("SklearnOneHotEncoder", convert_sklearn_one_hot_encoder)
