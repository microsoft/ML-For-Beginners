# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

# This file contains some high-level APIs for applying operations on variables specified by names. We should try our
# best to use those functions because they can produce ONNX operators according to the ONNX version specified in the
# `container` argument. Notice that those function behaviors are defined in a way very similar to ONNX-1.2.

import numpy as np
from onnx import onnx_pb as onnx_proto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


def _create_name_or_use_existing_one(scope, op_type, name):
    if name is None:
        return scope.get_unique_operator_name(op_type)
    else:
        return name


def _apply_unary_operation(scope, op_type, input_name, output_name, container, operator_name, **attrs):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs['name'] = name
    if container.target_opset < 6:
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node(op_type, input_name, output_name, op_version=op_version, **attrs)


def _apply_basic_numerical_operation(scope, op_type, input_names, output_name, container, operator_name,
                                     axis, broadcast):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs = {}
    if container.target_opset < 7:
        # Before ONNX-1.2 (opset 7), broadcasting behavior is Caffe2-like.
        if axis is not None:
            attrs['axis'] = axis
        if broadcast is not None:
            attrs['broadcast'] = broadcast

        if container.target_opset < 6:
            attrs['consumed_inputs'] = [0, 0]
            op_version = 1
        else:
            op_version = 6
    else:
        # Since ONNX-1.2 (opset 7), broadcasting behavior is Numpy-like, so we don't need to specify any attributes
        op_version = 7

    container.add_node(op_type, input_names, output_name, op_version=op_version, name=name, **attrs)


def _apply_pointwise_operation(scope, op_type, input_names, output_name, container, operator_name):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)
    attrs = {}

    if container.target_opset < 6:
        attrs['consumed_inputs'] = [0] * len(input_names)
        op_version = 1
    elif container.target_opset < 8:
        op_version = 6
    else:
        if container.target_opset < 12 or op_type == 'Mean':
            op_version = 8
        else:
            op_version = 12

    container.add_node(op_type, input_names, output_name, op_version=op_version, name=name, **attrs)


def apply_abs(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Abs', input_name, output_name, container, operator_name=operator_name)


def apply_add(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    _apply_basic_numerical_operation(scope, 'Add', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_argmax(scope, input_name, output_name, container, operator_name=None, axis=0, keepdims=1,
                 select_last_index=0):
    name = _create_name_or_use_existing_one(scope, 'ArgMax', operator_name)
    attrs = {'axis': axis, 'keepdims': keepdims}
    if container.target_opset < 11:
        op_version = 1
    elif container.target_opset < 12:
        op_version = 11
    else:
        op_version = 12
        attrs['select_last_index'] = select_last_index
    container.add_node('ArgMax', input_name, output_name, op_version=op_version, name=name, **attrs)


def apply_argmin(scope, input_name, output_name, container, operator_name=None, axis=0, keepdims=1,
                 select_last_index=0):
    name = _create_name_or_use_existing_one(scope, 'ArgMin', operator_name)
    attrs = {'axis': axis, 'keepdims': keepdims}
    if container.target_opset < 11:
        op_version = 1
    elif container.target_opset < 12:
        op_version = 11
    else:
        op_version = 12
        attrs['select_last_index'] = select_last_index
    container.add_node('ArgMin', input_name, output_name, op_version=op_version, name=name, **attrs)


def apply_affine(scope, input_name, output_name, container, operator_name=None, alpha=1., beta=0.):
    if container.target_opset < 9:
        op_type = 'Affine'
        name = _create_name_or_use_existing_one(scope, 'Affine', operator_name)
        attrs = {'name': name, 'alpha': alpha, 'beta': beta}
        container.add_node(op_type, input_name, output_name, **attrs)
    else:
        name = _create_name_or_use_existing_one(scope, 'Affine', operator_name)
        # Define a and b.
        aName = scope.get_unique_variable_name(name + '_alpha')
        container.add_initializer(aName, onnx_proto.TensorProto.FLOAT, [1], [alpha])
        bName = scope.get_unique_variable_name(name + '_beta')
        container.add_initializer(bName, onnx_proto.TensorProto.FLOAT, [1], [beta])

        # Compute Z = a * X, where X is the original input.
        zName = scope.get_unique_variable_name(name + '_scaled')
        apply_mul(scope, [aName, input_name], zName, container)

        # Compute Y = Z + b, where Y is the final output.
        apply_add(scope, [zName, bName], output_name, container)


def apply_batch_norm(scope, input_names, output_names, container, operator_name=None,
                     epsilon=None, is_test=None, momentum=None, spatial=None):
    name = _create_name_or_use_existing_one(scope, 'BatchNormalization', operator_name)
    attrs = {'name': name, 'epsilon': epsilon, 'momentum': momentum}

    if container.target_opset < 9:
        attrs['spatial'] = spatial
    if container.target_opset < 7:
        attrs['is_test'] = is_test

    if container.target_opset < 6:
        attrs['consumed_inputs'] = [0] * len(input_names)
        if len(input_names) > 3:
            attrs['consumed_inputs'][3] = 1
        if len(input_names) > 4:
            attrs['consumed_inputs'][4] = 2
        op_version = 1
    elif container.target_opset < 7:
        op_version = 6
    elif container.target_opset < 9:
        op_version = 7
    else:
        op_version = 9

    container.add_node('BatchNormalization', input_names, output_names, op_version=op_version, **attrs)


def apply_cast(scope, input_name, output_name, container, operator_name=None, to=None):
    '''
    :param to: enum defined in ONNX TensorProto.DataType, for example, TensorProto.FLOAT and TensorProto.INT64.
    '''
    name = _create_name_or_use_existing_one(scope, 'Cast', operator_name)
    attrs = {'name': name}

    d = onnx_proto.TensorProto.DataType.DESCRIPTOR
    allowed_type_name_and_type_enum_pairs = {v.number: k for k, v in d.values_by_name.items()}
    if to not in allowed_type_name_and_type_enum_pairs:
        raise ValueError('Attribute "to" must be one of %s' % allowed_type_name_and_type_enum_pairs.keys())

    if container.target_opset < 9:
        if to in [onnx_proto.TensorProto.STRING, onnx_proto.TensorProto.COMPLEX64, onnx_proto.TensorProto.COMPLEX128]:
            raise ValueError('Attribute "to" cannot correspond to a String or Complex TensorProto type.')

        if container.target_opset < 6:
            # Convert enum to string, for example, TensorProto.INT64 to 'INT64'
            attrs['to'] = allowed_type_name_and_type_enum_pairs[to]
            op_version = 1
        else:
            # Enum, for example, TensorProto.INT64
            attrs['to'] = to
            op_version = 6
    else:
        # Enum value, for example, TensorProto.INT64
        # String casting is supported in opset 9
        if to in [onnx_proto.TensorProto.COMPLEX64, onnx_proto.TensorProto.COMPLEX128]:
            raise ValueError('Attribute "to" cannot correspond to a Complex TensorProto type.')
        attrs['to'] = to
        op_version = 9

    container.add_node('Cast', input_name, output_name, op_version=op_version, **attrs)


def apply_clip(scope, input_name, output_name, container, operator_name=None, max=None, min=None):
    name = _create_name_or_use_existing_one(scope, 'Clip', operator_name)
    attrs = {'name': name}

    if container.target_opset < 11:
        if max is not None:
            attrs['max'] = float(max)
        if min is not None:
            attrs['min'] = float(min)

        if container.target_opset < 6:
            attrs['consumed_inputs'] = [0]
            op_version = 1
        else:
            op_version = 6

        container.add_node('Clip', input_name, output_name, op_version=op_version, **attrs)
    else:
        if container.target_opset < 12:
            op_version = 11
        else:
            op_version = 12
        if min is None and max is not None:
            raise RuntimeError("Operator 'Clip': min must be specified if max is.")
        inputs = [input_name]

        if min is not None:
            if isinstance(min, (np.ndarray, float, int)):
                # add initializer
                if isinstance(min, np.ndarray):
                    if len(min.shape) == 0:
                        min = [min]
                    elif min.shape == (1,):
                        min = list(min[0]) if hasattr(min[0], '__iter__') else list(min)
                    else:
                        raise RuntimeError("min must be an array of one element.")
                else:
                    min = [min]

                # container in sklearn-onnx stores the computation type in
                # container.dtype.
                min_name = scope.get_unique_variable_name('clip_min')
                if op_version < 12:
                    min = np.array(min, dtype=getattr(container, 'dtype', np.float32))
                    container.add_initializer(min_name, getattr(container, 'proto_dtype',
                                                                onnx_proto.TensorProto.FLOAT), [], [min[0]])
                else:
                    min = np.array(min)
                    container.add_initializer(min_name, NP_TYPE_TO_TENSOR_TYPE[min.dtype], [], [min[0]])
                min = min_name
            if isinstance(min, str):
                inputs.append(min)
            else:
                raise RuntimeError("Parameter 'min' must be a string or a float.")

        if max is not None:
            if min is None:
                raise RuntimeError("Parameter 'min' must be specified if 'max' is.")
            if isinstance(max, (np.ndarray, float, int)):
                # add initializer
                if isinstance(max, np.ndarray):
                    if len(max.shape) == 0:
                        max = [max]
                    elif max.shape == (1,):
                        max = list(max[0]) if hasattr(max[0], '__iter__') else list(max)
                    else:
                        raise RuntimeError("max must be an array of one element.")
                else:
                    max = [max]

                max_name = scope.get_unique_variable_name('clip_max')
                if op_version < 12:
                    max = np.array(max, dtype=getattr(container, 'dtype', np.float32))
                    container.add_initializer(max_name, getattr(container, 'proto_dtype',
                                                                onnx_proto.TensorProto.FLOAT), [], [max[0]])
                else:
                    max = np.array(max)
                    container.add_initializer(max_name, NP_TYPE_TO_TENSOR_TYPE[max.dtype], [], [max[0]])
                max = max_name
            if isinstance(max, str):
                inputs.append(max)
            else:
                raise RuntimeError("Parameter 'max' must be a string or a float.")

        container.add_node('Clip', inputs, output_name, op_version=op_version,
                           **attrs)


def apply_concat(scope, input_names, output_name, container, operator_name=None, axis=0):
    name = _create_name_or_use_existing_one(scope, 'Concat', operator_name)

    if container.target_opset < 4:
        op_version = 1
    elif container.target_opset < 11:
        op_version = 4
    else:
        op_version = 11

    container.add_node('Concat', input_names, output_name, op_version=op_version, name=name, axis=axis)


def apply_constant(scope, output_name, container, operator_name=None, value=None):
    name = _create_name_or_use_existing_one(scope, 'Constant', operator_name)

    if value is None:
        raise ValueError('Attribute "value" is a required argument.')

    if container.target_opset < 9:
        op_version = 1
    elif container.target_opset < 11:
        op_version = 9
    elif container.target_opset < 12:
        op_version = 11
    else:
        op_version = 12

    if op_version < 12:
        attrs = {'name': name, 'value': value}
    else:
        if isinstance(value, float):
            attrs = {'name': name, 'value_float': value}
        elif isinstance(value, int):
            attrs = {'name': name, 'value_int': value}
        elif isinstance(value, str):
            attrs = {'name': name, 'value_string': value}
        else:
            attrs = {'name': name, 'value': value}

    container.add_node('Constant', [], output_name, op_version=op_version, **attrs)


def apply_constant2(scope, input_names, output_name, container, operator_name=None, value=None):
    assert len(input_names) == 0  # only a placeholder to standardize the argument list.
    return apply_constant(scope, output_name, container, operator_name, value)


def apply_constant_of_shape(scope, input_names, output_name, container, operator_name=None, value=None):
    name = _create_name_or_use_existing_one(scope, 'ConstantOfShape', operator_name)
    container.add_node('ConstantOfShape', input_names, output_name, name=name, op_version=9, value=value)


def apply_conv(scope, input_names, output_name, container, operator_name=None, **attrs):
    name = _create_name_or_use_existing_one(scope, 'Conv', operator_name)

    if container.target_opset < 11:
        op_version = 1
    else:
        op_version = 11

    container.add_node('Conv', input_names, output_name, name=name, op_version=op_version, **attrs)


def apply_crop_height_width(scope, input_name, output_name, container, operator_name=None,
                            top_border=0, bottom_border=0, left_border=0, right_border=0):
    name = scope.get_unique_operator_name('CropHeightWidth')
    if container.target_opset < 9:
        # If operator set < 9, we can use the experimental Crop in ONNX.
        attrs = {'name': name, 'border': [left_border, top_border, right_border, bottom_border]}
        container.add_node('Crop', input_name, output_name, **attrs)
    else:
        # The experimental Crop in ONNX is removed after operator set 9, so we
        # switch to ONNX DynamicSlice operator.

        # CoreML only crops H- and W-axes.
        axes = [2, 3]
        axes_name = scope.get_unique_variable_name(name + '_axes')
        container.add_initializer(axes_name, onnx_proto.TensorProto.INT64,
                                  [len(axes)], axes)

        # Number of cropped pixels is the starting index of the remained region.
        starts = [top_border, left_border]
        starts_name = scope.get_unique_variable_name(name + '_starts')
        container.add_initializer(starts_name, onnx_proto.TensorProto.INT64,
                                  [len(starts)], starts)

        # First we assume no cropping is needed at the end of those axes.
        # We will change this right below depending on Crop's configuration.
        ends = [np.iinfo(np.int64).max] * 2

        # Crop n pixel means the end index (exclusive) is -n. Note that indexing
        # system is zero-based.
        if bottom_border > 0:
            ends[0] = -bottom_border
        if right_border > 0:
            ends[1] = -right_border

        # Add the adjusted ends.
        ends_name = scope.get_unique_variable_name(name + '_ends')
        container.add_initializer(ends_name, onnx_proto.TensorProto.INT64,
                                  [len(ends)], ends)

        # Collect all input names as a list because DynamicSlice has multiple inputs.
        input_list = [input_name, starts_name, ends_name, axes_name]
        container.add_node('DynamicSlice', input_list, output_name, op_version=9)


def apply_div(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    _apply_basic_numerical_operation(scope, 'Div', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_elu(scope, input_name, output_name, container, operator_name=None, alpha=1.0):
    _apply_unary_operation(scope, 'Elu', input_name, output_name, container, operator_name, alpha=alpha)


def apply_equal(scope, input_names, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'equal', operator_name)
    if container.target_opset < 7:
        op_version = 1
    elif container.target_opset < 9:
        op_version = 7
    else:
        op_version = 9
    container.add_node('Equal', input_names, output_name, name=name, op_version=op_version)


def apply_exp(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Exp', input_name, output_name, container, operator_name=operator_name)


def apply_floor(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Floor', input_name, output_name, container, operator_name=operator_name)


def apply_flatten(scope, input_name, output_name, container, operator_name=None, axis=1):
    name = _create_name_or_use_existing_one(scope, 'Flatten', operator_name)
    if container.target_opset < 9:
        op_version = 1
    elif container.target_opset < 11:
        op_version = 9
    else:
        op_version = 11
    container.add_node('Flatten', input_name, output_name, name=name, op_version=op_version, axis=axis)


def apply_gather(scope, input_names, output_name, container, operator_name=None, axis=0):
    name = _create_name_or_use_existing_one(scope, 'Gather', operator_name)
    if container.target_opset < 11:
        op_version = 1
    else:
        op_version = 11

    container.add_node('Gather', input_names, output_name, name=name, op_version=op_version, axis=axis)


def apply_gemm(scope, input_name, output_name, container, operator_name=None, alpha=1.0, beta=1.0,
               transA=0, transB=0):
    """
    Applies operator `gemm <https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm>`.
    """
    name = _create_name_or_use_existing_one(scope, 'Gemm', operator_name)
    attrs = {'alpha': alpha, 'beta': beta, 'transA': transA, 'transB': transB}
    if container.target_opset < 5:
        attrs['op_version'] = 1
        attrs['broadcast'] = 1
    elif container.target_opset < 7:
        attrs['op_version'] = 6
        attrs['broadcast'] = 1
    elif container.target_opset < 11:
        attrs['op_version'] = 7
    else:
        attrs['op_version'] = 11

    container.add_node('Gemm', input_name, output_name, name=name, **attrs)


def apply_greater(scope, input_names, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Greater', operator_name)
    if container.target_opset < 7:
        op_version = 1
    elif container.target_opset < 9:
        op_version = 7
    else:
        op_version = 9

    container.add_node('Greater', input_names, output_name, name=name, op_version=op_version)


def _convert_compare_equal(scope, input_names, output_name, container, operator_name, tf_op_string, onnx_op_string_rev,
                           onnx_op_string):
    if container.target_opset < 7:
        raise ValueError(tf_op_string + " op is not supported for opset < 7")
    elif container.target_opset < 9:
        op_version = 7
    elif container.target_opset < 12:
        op_version = 9
    else:
        op_version = 12
    name = _create_name_or_use_existing_one(scope, tf_op_string, operator_name)
    if op_version < 9:
        compare_input_0 = scope.get_unique_variable_name(name + '_input_0_cast')
        container.add_node('Cast', [input_names[0]], compare_input_0, name=name + '_input_0_cast', to=1)
        compare_input_1 = scope.get_unique_variable_name(name + '_input_1_cast')
        container.add_node('Cast', [input_names[1]], compare_input_1, name=name + '_input_1_cast', to=1)
        less_out = scope.get_unique_variable_name(name + '_less_out')
        container.add_node(onnx_op_string_rev, [compare_input_0, compare_input_1], less_out,
                           name=name + '_' + onnx_op_string_rev.lower(),
                           op_version=op_version)
        container.add_node('Not', less_out, output_name, name=name + '_not')
    elif op_version < 12:
        compare_node = scope.get_unique_variable_name(name + '_compare_node')
        container.add_node(onnx_op_string_rev, input_names, compare_node,
                           name=name + '_' + onnx_op_string_rev.lower(),
                           op_version=op_version)
        container.add_node('Not', [compare_node], output_name, name=name)
    else:
        container.add_node(onnx_op_string, input_names, output_name,
                           name=name + '_' + onnx_op_string_rev.lower(), op_version=op_version)


def apply_greater_or_equal(scope, input_names, output_name, container, operator_name=None):
    _convert_compare_equal(scope, input_names, output_name, container, operator_name, 'GreaterEqual', 'Less',
                           'GreaterOrEqual')


def apply_less_or_equal(scope, input_names, output_name, container, operator_name=None):
    _convert_compare_equal(scope, input_names, output_name, container, operator_name, 'LessEqual', 'Greater',
                           'LessOrEqual')


def apply_gru(scope, input_names, output_names, container, operator_name=None, output_seq=0, reset_after=0, **attrs):
    name = _create_name_or_use_existing_one(scope, 'GRU', operator_name)
    if container.target_opset < 3:
        op_version = 1
        attrs['output_sequence'] = 1 if output_seq else 0
    else:
        attrs['linear_before_reset'] = 1 if reset_after else 0
        if container.target_opset <= 5:
            attrs['output_sequence'] = 1 if output_seq else 0
            op_version = 3
        elif container.target_opset <= 13:
            op_version = 7
        else:
            op_version = 14

    container.add_node('GRU', input_names, output_names, name=name, op_version=op_version, **attrs)


def apply_hard_sigmoid(scope, input_name, output_name, container, operator_name=None, alpha=None, beta=None):
    _apply_unary_operation(scope, 'HardSigmoid', input_name, output_name, container, operator_name,
                           alpha=alpha, beta=beta)


def apply_identity(scope, input_name, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Identity', operator_name)
    container.add_node('Identity', input_name, output_name, name=name)


def apply_instance_norm(scope, input_names, output_name, container, operator_name=None, epsilon=1e-5):
    name = _create_name_or_use_existing_one(scope, 'InstanceNormalization', operator_name)
    attrs = {'name': name, 'epsilon': epsilon}

    if container.target_opset < 2:
        attrs['consumed_inputs'] = [0] * len(input_names)
        op_version = 1
    else:
        op_version = 6

    container.add_node('InstanceNormalization', input_names, output_name, op_version=op_version, **attrs)


def apply_inverse(scope, input_name, output_name, container, operator_name=None):
    if container.target_opset < 12:
        raise ValueError("tf op MatrixInverse is not supported for opset < 12")
    else:
        op_version = 12
    name = _create_name_or_use_existing_one(scope, 'Inverse', operator_name)
    container.add_node('Inverse', input_name, output_name, name=name, op_version=op_version)


def apply_leaky_relu(scope, input_name, output_name, container, operator_name=None, alpha=0.01):
    _apply_unary_operation(scope, 'LeakyRelu', input_name, output_name, container, operator_name, alpha=alpha)


def apply_less(scope, input_names, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Less', operator_name)
    if container.target_opset < 7:
        op_version = 1
    elif container.target_opset < 9:
        op_version = 7
    else:
        op_version = 9

    container.add_node('Less', input_names, output_name, name=name, op_version=op_version)


def apply_log(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Log', input_name, output_name, container, operator_name=operator_name)


def apply_lstm(scope, input_names, output_names, container, operator_name=None, output_seq=0, **attrs):
    name = _create_name_or_use_existing_one(scope, 'LSTM', operator_name)
    if container.target_opset <= 6:
        attrs['output_sequence'] = 1 if output_seq else 0
        op_version = 1
    elif container.target_opset <= 13:
        op_version = 7
    else:
        op_version = 14
    container.add_node('LSTM', input_names, output_names, name=name, op_version=op_version, **attrs)


def apply_matmul(scope, input_names, output_name, container, operator_name=None):
    op_type = 'MatMul'
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)
    if container.target_opset <= 9:
        op_version = 1
    else:
        op_version = 9
    container.add_node(op_type, input_names, output_name, op_version=op_version, name=name)


def apply_max(scope, input_names, output_name, container, operator_name=None):
    _apply_pointwise_operation(scope, 'Max', input_names, output_name, container, operator_name)


def apply_mean(scope, input_names, output_name, container, operator_name=None):
    _apply_pointwise_operation(scope, 'Mean', input_names, output_name, container, operator_name)


def apply_min(scope, input_names, output_name, container, operator_name=None):
    _apply_pointwise_operation(scope, 'Min', input_names, output_name, container, operator_name)


def apply_mul(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    _apply_basic_numerical_operation(scope, 'Mul', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_neg(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Neg', input_name, output_name, container, operator_name)


def apply_normalization(scope, input_name, output_name, container, operator_name=None, axis=1, p=2):
    name = _create_name_or_use_existing_one(scope, 'LpNormalization', operator_name)
    container.add_node('LpNormalization', input_name, output_name, name=name, p=p, axis=axis)


def apply_not_op(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Not', input_name, output_name, container, operator_name)


def apply_pad(scope, input_name, output_name, container, operator_name=None, mode=None, pads=None, value=None,
              onnx_type=onnx_proto.TensorProto.FLOAT):
    name = _create_name_or_use_existing_one(scope, 'Pad', operator_name)
    attrs = {'name': name}
    inputs = input_name if isinstance(input_name, list) else [input_name]

    if mode is not None:
        attrs['mode'] = mode

    if container.target_opset < 11:
        if isinstance(pads, str):
            raise ValueError("Dynamic pad is not supported for opset < 11.")
        if value is not None:
            attrs['value'] = value
        if container.target_opset < 2:
            attrs['paddings'] = pads
            op_version = 1
        else:
            attrs['pads'] = pads
            op_version = 2
    else:
        op_version = 11
        if isinstance(pads, str):
            inputs.append(pads)
        else:
            pads_name = scope.get_unique_variable_name(name + '_pads')
            container.add_initializer(pads_name, onnx_proto.TensorProto.INT64, [len(pads)], pads)
            inputs.append(pads_name)
        if value is not None:
            value_name = scope.get_unique_variable_name(name + '_value')
            container.add_initializer(value_name, onnx_type, [], [value])
            inputs.append(value_name)

    container.add_node('Pad', inputs, output_name, op_version=op_version, **attrs)


def apply_parametric_softplus(scope, input_name, output_name, container, operator_name=None, alpha=None, beta=None):
    if alpha is None:
        alpha = [1.0]
    if beta is None:
        beta = [0.]

    name = _create_name_or_use_existing_one(scope, 'ParametricSoftplus', operator_name)
    if container.target_opset < 9:
        if len(alpha) != 1 or len(beta) != 1:
            raise ValueError('alpha and beta must be 1-element lists')
        op_type = 'ParametricSoftplus'
        attrs = {'name': name, 'alpha': alpha[0], 'beta': beta[0]}
        container.add_node(op_type, input_name, output_name, **attrs)
    else:
        # Define three scalars: a, b, 1.
        aName = scope.get_unique_variable_name(name + '_alpha')
        aShape = [len(alpha)] if len(alpha) == 1 else [len(alpha), 1, 1]
        container.add_initializer(aName, onnx_proto.TensorProto.FLOAT, aShape, alpha)
        bShape = [len(beta)] if len(beta) == 1 else [len(beta), 1, 1]
        bName = scope.get_unique_variable_name(name + '_beta')
        container.add_initializer(bName, onnx_proto.TensorProto.FLOAT, bShape, beta)
        oneName = scope.get_unique_variable_name(name + '_one')
        container.add_initializer(oneName, onnx_proto.TensorProto.FLOAT, [1], [1.])

        # c = b * x
        cName = scope.get_unique_variable_name(name + '_c')
        apply_mul(scope, [input_name, bName], cName, container)

        # d = exp(c)
        dName = scope.get_unique_variable_name(name + '_d')
        apply_exp(scope, cName, dName, container)

        # e = 1 + d
        eName = scope.get_unique_variable_name(name + '_e')
        apply_add(scope, [dName, oneName], eName, container)

        # f = log(e)
        fName = scope.get_unique_variable_name(name + '_f')
        apply_log(scope, eName, fName, container)

        # g = a * f
        apply_mul(scope, [fName, aName], output_name, container)


def apply_pow(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    name = _create_name_or_use_existing_one(scope, 'Pow', operator_name)

    attrs = {'name': name}
    if container.target_opset < 7:
        # Before ONNX-1.2, broadcasting behavior is Caffe2-like.
        if axis is not None:
            attrs['axis'] = axis
        if broadcast is not None:
            attrs['broadcast'] = broadcast
        op_version = 1
    elif container.target_opset < 12:
        # Since ONNX-1.2, broadcasting behavior is Numpy-like, so we don't need to specify any attributes
        op_version = 7
    else:
        op_version = 12

    container.add_node('Pow', input_names, output_name, op_version=op_version, **attrs)


def apply_prelu(scope, input_name, output_name, container, operator_name=None, slope=None):
    name = _create_name_or_use_existing_one(scope, 'PRelu', operator_name)
    slope_tensor_name = scope.get_unique_variable_name('slope')
    s_shape = slope.shape
    if container.target_opset < 7:
        s_shape = [len(slope.flatten())]
    container.add_initializer(slope_tensor_name, onnx_proto.TensorProto.FLOAT, s_shape, slope.flatten())

    if container.target_opset < 6:
        container.add_node('PRelu', [input_name, slope_tensor_name], output_name, op_version=1, name=name,
                           consumed_inputs=[0, 0])
    else:
        if container.target_opset < 7:
            op_version = 6
        elif container.target_opset < 9:
            op_version = 7
        else:
            # opset 9 supports unidirectional broadcasting
            op_version = 9

        container.add_node('PRelu', [input_name, slope_tensor_name], output_name, op_version=op_version, name=name)


def apply_range(scope, input_name, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Range', operator_name)
    container.add_node('Range', input_name, output_name, op_version=11, name=name)


def apply_reciprocal(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Reciprocal', input_name, output_name, container, operator_name=operator_name)


# Some old ORT supports axis < 0 case, so put rank=0 as default.
def apply_reducesum(scope, input_name, output_name, container, operator_name=None, axes=None, keepdims=1, rank=0):
    name = _create_name_or_use_existing_one(scope, 'ReduceSum', operator_name)
    if axes is None:
        axes = []
    if container.target_opset < 13:
        if container.target_opset < 11:
            op_version = 1
            axes = [axis if axis >= 0 else axis + rank for axis in axes]
        else:
            op_version = 11
        container.add_node('ReduceSum', input_name, output_name, name=name,
                           op_version=op_version, axes=axes, keepdims=keepdims)
    else:
        if not isinstance(input_name, list):
            input_name = [input_name]
        op_version = 13
        if isinstance(axes, str):
            container.add_node('ReduceSum', input_name + [axes], output_name,
                               op_version=op_version, name=name, keepdims=keepdims)
        elif axes is None or len(axes) == 0:
            container.add_node('ReduceSum', input_name, output_name,
                               op_version=op_version, name=name, keepdims=keepdims)
        else:
            axes_name = scope.get_unique_variable_name(name + '_reducesum')
            container.add_initializer(axes_name, onnx_proto.TensorProto.INT64, [len(axes)], axes)
            container.add_node('ReduceSum', input_name + [axes_name], output_name,
                               op_version=op_version, name=name, keepdims=keepdims)


def apply_relu(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Relu', input_name, output_name, container, operator_name)


def apply_relu_6(scope, input_name, output_name, container, operator_name=None, zero_value=0.0):
    name_relu = _create_name_or_use_existing_one(scope, 'relu', operator_name)
    name_relu_op = _create_name_or_use_existing_one(scope, 'relu6', operator_name)
    apply_relu(scope, input_name, name_relu, container, name_relu_op+'_relu')
    apply_clip(scope, name_relu, output_name, container, name_relu_op + '_clip', zero_value+6, zero_value)


def apply_reshape(scope, input_name, output_name, container, operator_name=None, desired_shape=None):
    if not isinstance(desired_shape, str) and len(list(i for i in desired_shape if i is not None and i < 0)) > 1:
        raise ValueError('There can only be one -1 in the targeted shape of a Reshape but got %s' % desired_shape)

    name = _create_name_or_use_existing_one(scope, 'Reshape', operator_name)

    if container.target_opset < 5:
        container.add_node('Reshape', input_name, output_name, op_version=1, name=name, shape=desired_shape,
                           consumed_inputs=[0])
    else:
        if isinstance(desired_shape, str):
            desired_shape_name = desired_shape
        else:
            desired_shape_name = scope.get_unique_variable_name('shape_tensor')
            container.add_initializer(desired_shape_name, onnx_proto.TensorProto.INT64, [len(desired_shape)],
                                      desired_shape)

        # Create ONNX Reshape operator
        if isinstance(input_name, list):
            input_name.append(desired_shape_name)
        else:
            input_name = [input_name, desired_shape_name]
        container.add_node('Reshape', input_name, output_name, op_version=5, name=name)


def apply_resize(scope, input_name, output_name, container, operator_name=None, mode='nearest',
                 coordinate_transformation_mode='asymmetric', scales=None):
    '''
    :param mode: "nearest" or "linear"
    :param scales: a float tensor for scaling (upsampling or downsampling) all input dimensions
    '''
    name = _create_name_or_use_existing_one(scope, 'Resize', operator_name)
    attrs = {'name': name}
    attrs['mode'] = mode.lower()

    inputs = [input_name]

    if container.target_opset < 11:
        op_version = 10
    else:
        op_version = 11
        roi_tensor_name = scope.get_unique_variable_name(name + '_roi')
        roi = [0.0] * len(scales) + [1.0] * len(scales)
        container.add_initializer(roi_tensor_name, onnx_proto.TensorProto.FLOAT, [2 * len(scales)], roi)
        inputs.append(roi_tensor_name)
        attrs['coordinate_transformation_mode'] = coordinate_transformation_mode
        if attrs['mode'] == 'nearest':
            attrs['nearest_mode'] = 'floor'

    scales_tensor_name = scope.get_unique_variable_name(name + '_scales')
    container.add_initializer(scales_tensor_name, onnx_proto.TensorProto.FLOAT, [len(scales)], scales)
    inputs.append(scales_tensor_name)
    container.add_node('Resize', inputs, output_name, op_version=op_version, **attrs)


def apply_rnn(scope, input_names, output_names, container, operator_name=None, output_seq=0, **attrs):
    name = _create_name_or_use_existing_one(scope, 'RNN', operator_name)
    if container.target_opset <= 6:
        attrs['output_sequence'] = 1 if output_seq else 0
        op_version = 1
    elif container.target_opset <= 13:
        op_version = 7
    else:
        op_version = 14
    container.add_node('RNN', input_names, output_names, name=name, op_version=op_version, **attrs)


def apply_shape(scope, input_name, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Shape', operator_name)
    container.add_node('Shape', input_name, output_name, name=name, op_version=1)


def apply_sigmoid(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Sigmoid', input_name, output_name, container, operator_name)


def apply_softsign(scope, input_name, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Softsign', operator_name)
    container.add_node('Softsign', input_name, output_name, name=name, op_version=1)


# See alpha and gamma at https://github.com/keras-team/keras/blob/master/keras/activations.py#L80-L81
def apply_selu(scope, input_name, output_name, container, operator_name=None, alpha=1.673263, gamma=1.050701):
    _apply_unary_operation(scope, 'Selu', input_name, output_name, container, operator_name, alpha=alpha, gamma=gamma)


def apply_softmax(scope, input_name, output_name, container, operator_name=None, axis=None):
    name = _create_name_or_use_existing_one(scope, 'Softmax', operator_name)
    if axis is None:
        axis = 1 if container.target_opset < 13 else -1
    container.add_node('Softmax', input_name, output_name, name=name, axis=axis)


def apply_scaled_tanh(scope, input_name, output_name, container, operator_name=None, alpha=None, beta=None):
    if alpha is None:
        alpha = [1.0]
    if beta is None:
        beta = [1.0]
    if len(alpha) != 1 or len(beta) != 1:
        raise ValueError('alpha and beta must be 1-element lists')

    name = _create_name_or_use_existing_one(scope, 'ScaledTanh', operator_name)
    if container.target_opset < 9:
        attrs = {'name': name, 'alpha': alpha[0], 'beta': beta[0]}
        container.add_node('ScaledTanh', input_name, output_name, **attrs)
    else:
        # Define scalar a, initialize with parameter alpha.
        aName = scope.get_unique_variable_name(name + '_alpha')
        aShape = [len(alpha)] if len(alpha) == 1 else [len(alpha), 1, 1]
        container.add_initializer(aName, onnx_proto.TensorProto.FLOAT, aShape, alpha)

        # Define scalar b, initialize with parameter beta.
        bShape = [len(beta)] if len(beta) == 1 else [len(beta), 1, 1]
        bName = scope.get_unique_variable_name(name + '_beta')
        container.add_initializer(bName, onnx_proto.TensorProto.FLOAT, bShape, beta)

        # c = b * x
        cName = scope.get_unique_variable_name(name + '_c')
        apply_mul(scope, [input_name, bName], cName, container)

        # d = tanh(c)
        dName = scope.get_unique_variable_name(name + '_d')
        apply_tanh(scope, cName, dName, container)

        # output = a * d
        apply_mul(scope, [aName, dName], output_name, container)


def apply_slice(scope, input_name, output_name, container, starts, ends,
                axes=None, steps=None, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Slice', operator_name)

    if container.target_opset < 10:
        if axes is None:
            container.add_node('Slice', input_name, output_name, name=name,
                               starts=starts, ends=ends, op_version=1)
        else:
            container.add_node('Slice', input_name, output_name, name=name,
                               starts=starts, ends=ends, axes=axes, op_version=1)
    else:
        if container.target_opset == 10:
            op_version = 10
        else:
            op_version = 11
        inputs = input_name if isinstance(input_name, list) else [input_name]
        if isinstance(starts, str):
            starts_name = starts
        else:
            starts_name = scope.get_unique_variable_name('starts')
            container.add_initializer(starts_name, onnx_proto.TensorProto.INT64,
                                      [len(starts)], starts)

        if isinstance(ends, str):
            ends_name = ends
        else:
            ends_name = scope.get_unique_variable_name('ends')
            container.add_initializer(ends_name, onnx_proto.TensorProto.INT64,
                                      [len(ends)], ends)

        inputs.append(starts_name)
        inputs.append(ends_name)
        if axes:
            if isinstance(axes, str):
                axes_name = axes
            else:
                axes_name = scope.get_unique_variable_name('axes')
                container.add_initializer(axes_name, onnx_proto.TensorProto.INT64,
                                          [len(axes)], axes)
            inputs.append(axes_name)
        if steps:
            if not axes:
                inputs.append('')
            if isinstance(steps, str):
                steps_name = steps
            else:
                steps_name = scope.get_unique_variable_name('steps')
                container.add_initializer(steps_name, onnx_proto.TensorProto.INT64,
                                          [len(steps)], steps)
            inputs.append(steps_name)
        container.add_node('Slice', inputs, output_name, name=name,
                           op_version=op_version)


def apply_slice2(scope, input_name, output_name, container,
                 operator_name=None, starts=None, ends=None, axes=None, steps=None):
    assert starts is not None, 'the starts in slice op cannot be None'
    assert ends is not None, 'the ends in slice op cannot be None'
    return apply_slice(scope, input_name, output_name, container, starts, ends,
                       axes, steps, operator_name)


def apply_split(scope, input_name, output_names, container, operator_name=None, split=None, axis=0):
    name = _create_name_or_use_existing_one(scope, 'Split', operator_name)
    if container.target_opset <= 1:
        op_version = 1
    elif container.target_opset < 11:
        op_version = 2
    elif container.target_opset < 13:
        op_version = 11
    else:
        op_version = 13

    attrs = {'name': name}
    if split is not None:
        if container.target_opset < 13:
            attrs['split'] = split
        else:
            if not isinstance(input_name, list):
                input_name = [input_name]
            if isinstance(split, str):
                split_name = split
            else:
                split_name = scope.get_unique_variable_name(name + '_split')
                container.add_initializer(split_name, onnx_proto.TensorProto.INT64, [len(split)], split)
            input_name = input_name + [split_name]

    if axis is not None:
        attrs['axis'] = axis

    container.add_node('Split', input_name, output_names, op_version=op_version, **attrs)


def apply_sqrt(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Sqrt', input_name, output_name, container, operator_name=operator_name)


def _apply_squeeze_unsqueeze(scope, input_name, output_name, container, squeeze_str, operator_name=None, axes=None,
                             rank=0):
    name = _create_name_or_use_existing_one(scope, squeeze_str, operator_name)
    if container.target_opset < 13:
        if container.target_opset < 11:
            op_version = 1
            axes = [axis if axis >= 0 else axis + rank for axis in axes]
        else:
            op_version = 11
        container.add_node(squeeze_str, input_name, output_name, name=name, op_version=op_version, axes=axes)
    else:
        op_version = 13
        if not isinstance(input_name, list):
            input_name = [input_name]
        if isinstance(axes, str):
            container.add_node(squeeze_str, input_name + [axes], output_name, op_version=op_version, name=name)
        elif len(axes) == 0:
            container.add_node(squeeze_str, input_name, output_name, op_version=op_version, name=name)
        else:
            axes_name = scope.get_unique_variable_name(name + '_axes')
            container.add_initializer(axes_name, onnx_proto.TensorProto.INT64, [len(axes)], axes)
            container.add_node(squeeze_str, input_name + [axes_name], output_name, op_version=op_version, name=name)


def apply_squeeze(scope, input_name, output_name, container, operator_name=None, axes=None, rank=0):
    if axes is None:
        axes = []
    _apply_squeeze_unsqueeze(scope, input_name, output_name, container, 'Squeeze', operator_name, axes, rank)


def apply_sub(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=0):
    _apply_basic_numerical_operation(scope, 'Sub', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_sum(scope, input_names, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'Sum', operator_name)
    if container.target_opset < 6:
        op_version = 1
    else:
        op_version = 6
    container.add_node('Sum', input_names, output_name, op_version=op_version, name=name)


def apply_tanh(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Tanh', input_name, output_name, container, operator_name)


def apply_thresholded_relu(scope, input_name, output_name, container, operator_name=None, alpha=None):
    if alpha is None:
        alpha = [1.0]

    name = _create_name_or_use_existing_one(scope, 'ThresholdedRelu', operator_name)
    attrs = {'name': name, 'alpha': alpha[0]}
    if container.target_opset < 10:
        # ThresholdedRelu graduated from an experimental op to a full op in opset 10
        # onnxruntime maintains support in the ONNX domain for ThresholdedRelu as a contrib op
        attrs['op_domain'] = "ai.onnx"
        op_version = 1
    else:
        op_version = 10
    container.add_node('ThresholdedRelu', input_name, output_name, op_version=op_version, **attrs)


def apply_tile(scope, input_name, output_name, container, operator_name=None, repeats=None):
    name = _create_name_or_use_existing_one(scope, 'Tile', operator_name)

    if repeats is None or (not isinstance(repeats, str) and all(repeat_count == 1 for repeat_count in repeats)):
        container.add_node('Identity', input_name, output_name, name=name)
        return

    if container.target_opset < 6:
        intermediate_input_name = input_name
        intermediate_output_name = None
        if isinstance(repeats, str):
            raise ValueError('repeats cannot be string type before opset 6')

        for axis, repeat_count in enumerate(repeats):
            if repeat_count == 1:
                continue

            # Create the 2nd input of Tile
            tile_tensor_name = scope.get_unique_variable_name(name + '_tile')
            container.add_initializer(tile_tensor_name, onnx_proto.TensorProto.FLOAT, [1], [float(repeat_count)])

            # Create the 3rd input of Tile
            axis_tensor_name = scope.get_unique_variable_name(name + '_axis')
            container.add_initializer(axis_tensor_name, onnx_proto.TensorProto.FLOAT, [1], [float(axis)])

            # Create tile for duplicating along one axis. After ONNX-1.2, we can duplicate along multiple axes, so we
            # don't have to iterate through all axes.
            intermediate_output_name = scope.get_unique_variable_name(name + '_input')
            container.add_node('Tile', [intermediate_input_name, tile_tensor_name, axis_tensor_name],
                               intermediate_output_name, name=name)

            # Use the output produced by this round as the input in the next iteration
            intermediate_input_name = intermediate_output_name

            # Create a new name for next Tile
            name = scope.get_unique_operator_name('Tile')

        # Use the last Tile name for the name of an Identity
        container.add_node('Identity', intermediate_output_name, output_name, op_version=1, name=name)
    else:
        # ONNX-1.2 has a new Tile and we use it here
        if isinstance(repeats, str):
            container.add_node('Tile', input_name + [repeats], output_name, op_version=6, name=name)
        else:
            repeat_tensor_name = scope.get_unique_variable_name(name + '_repeats')
            container.add_initializer(repeat_tensor_name, onnx_proto.TensorProto.INT64, [len(repeats)], repeats)
            container.add_node('Tile', [input_name, repeat_tensor_name], output_name, op_version=6, name=name)


def apply_topk(scope, input_name, output_names, container, k, operator_name=None):
    name = _create_name_or_use_existing_one(scope, 'TopK', operator_name)

    if container.target_opset < 10:
        if isinstance(k, str):
            raise ValueError('topk k cannot be string type before opset 10')
        container.add_node('TopK', input_name, output_names, name=name, k=k, op_version=1)
    else:
        if container.target_opset == 10:
            op_version = 10
        else:
            op_version = 11

        if isinstance(k, str):
            k_value_name = k
        else:
            k_value_name = scope.get_unique_variable_name('k_value')
            container.add_initializer(k_value_name, onnx_proto.TensorProto.INT64, [1], [k])
        container.add_node('TopK', input_name + [k_value_name], output_names, name=name, op_version=op_version)


def apply_transpose(scope, input_name, output_name, container, operator_name=None, perm=None):
    name = _create_name_or_use_existing_one(scope, 'Transpose', operator_name)
    container.add_node('Transpose', input_name, output_name, name=name, perm=perm)


def apply_upsample(scope, input_name, output_name, container, operator_name=None, mode='nearest',
                   coordinate_transformation_mode='asymmetric', scales=None):
    '''
    :param mode: nearest or linear
    :param scales: an integer list of scaling-up rate of all input dimensions
    '''
    if container.target_opset < 10:
        name = _create_name_or_use_existing_one(scope, 'Upsample', operator_name)
        inputs = [input_name]
        attrs = {'name': name}
        if container.target_opset < 7:
            if len(scales) != 4:
                raise ValueError('Need to specify a 4-element list the the scales of N-, C-, H-, and W-axes')
            attrs['height_scale'] = float(scales[2])
            attrs['width_scale'] = float(scales[3])
            attrs['mode'] = mode.upper()
            op_version = 1
        else:
            attrs['mode'] = mode.lower()
            if container.target_opset < 9:
                attrs['scales'] = list(map(float, scales))
                op_version = 7
            else:
                # scales moved from attribute to input in opset 9
                scales_tensor_name = scope.get_unique_variable_name(name + '_scales')
                container.add_initializer(scales_tensor_name, onnx_proto.TensorProto.FLOAT, [len(scales)], scales)
                inputs = [input_name, scales_tensor_name]
                op_version = 9

        container.add_node('Upsample', inputs, output_name, op_version=op_version, **attrs)
    else:
        # Upsample op is deprecated in ONNX opset 10
        # We implement Upsample through Resize instead
        apply_resize(scope, input_name, output_name, container, operator_name, mode, coordinate_transformation_mode,
                     scales)


def apply_unsqueeze(scope, input_name, output_name, container, operator_name=None, axes=None, rank=0):
    if axes is None:
        axes = [0]
    _apply_squeeze_unsqueeze(scope, input_name, output_name, container, 'Unsqueeze', operator_name, axes, rank)
