# coding=utf-8
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from onnx import onnx_pb as onnx_proto, helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from . import onnx_ops


class _OperatorNameContext:
    _history = []

    def __init__(self, oopb, basename):
        self.basename = basename
        self.oopb = oopb

    def __enter__(self):
        assert self.oopb.basename is None, "The previous context doesn't quit"
        self.oopb.basename = self.basename
        if len(_OperatorNameContext._history) > 0:
            self.oopb.upper_ctx = _OperatorNameContext._history[-1]
        _OperatorNameContext._history.append(self)
        return self.oopb

    def __exit__(self, type, value, traceback):
        assert self is _OperatorNameContext._history.pop()
        self.oopb.basename = None


class OnnxOperatorBuilder:
    def __init__(self, container, scope):
        self._container = container
        self._scope = scope
        self.basename = None
        # TODO: not all OnnxOperatorBuilder invocation is via as_default...
        # ... temporarily enable this for onnx_fx
        self.upper_ctx = None
        self.int32 = onnx_proto.TensorProto.INT32
        self.int64 = onnx_proto.TensorProto.INT64
        self.float = onnx_proto.TensorProto.FLOAT
        self.float16 = onnx_proto.TensorProto.FLOAT16
        self.double = onnx_proto.TensorProto.DOUBLE
        self.bool = onnx_proto.TensorProto.BOOL

    def as_default(self, basename):
        return _OperatorNameContext(self, basename)

    @property
    def upper_context(self):
        return self.upper_ctx

    def _process_inputs(self, inputs, name):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        ox_inputs = []
        for i_ in inputs:
            ox_n = i_
            if isinstance(i_, np.ndarray):
                ox_n = self._scope.get_unique_variable_name(name + '_i')
                self._container.add_initializer(
                    ox_n,
                    NP_TYPE_TO_TENSOR_TYPE[i_.dtype],
                    i_.shape,
                    i_.flatten()
                )
            elif isinstance(i_, (tuple, list)):
                ox_n = self._scope.get_unique_variable_name(name + i_[0])
                self._container.add_initializer(
                    ox_n,
                    i_[1],
                    i_[2].shape,
                    i_[2].flatten()
                )
            elif isinstance(ox_n, str):
                pass
            else:
                raise ValueError(
                    'Unknown type for ONNX initializer: {}'.format(type(ox_n)))
            ox_inputs.append(ox_n)

        return ox_inputs

    def _process_outputs(self, outputs, name):
        if outputs is None:
            ox_outputs = 1
        else:
            ox_outputs = outputs
        if isinstance(ox_outputs, int):
            ox_outputs = [self._scope.get_unique_variable_name(
                name + str(i_)) for i_ in range(ox_outputs)]
        elif isinstance(ox_outputs, (list, tuple)):
            pass
        else:
            raise ValueError(
                'Unknown type for outputs: {}'.format(type(ox_outputs)))
        return ox_outputs

    def _generate_name(self, type_or_func, name):
        base_name = (self.basename if self.basename else '') + '_'
        if name is not None:
            long_name = base_name + name
        else:
            if isinstance(type_or_func, str):
                suffix = type_or_func.lower()
            else:
                suffix = type_or_func.__name__[len('apply_'):]
            long_name = base_name + suffix
        return long_name

    @staticmethod
    def value_to_ndarray(value):
        if isinstance(value, (int, float, bool)):
            ty = np.int64
            if isinstance(value, float):
                ty = np.float32
            elif isinstance(value, bool):
                ty = np.bool_
            else:
                pass
            value = np.array(value).astype(ty)
        return value

    def _value_to_tensor(self, value, name, atleast_1d=False):
        value = type(self).value_to_ndarray(value)
        if isinstance(value, np.ndarray):
            if atleast_1d:
                value = np.atleast_1d(value)  # e.g. constant_of_shape() needs this
            lst = value.flatten().tolist()
            value = helper.make_tensor(name, NP_TYPE_TO_TENSOR_TYPE[value.dtype], value.shape, lst)
        return value

    def add_node(self, op_type, inputs, name=None, outputs=None, op_domain='', op_version=None, **attrs):
        if op_version is None:
            op_version = self._container.target_opset
        name = self._generate_name(op_type, name)
        ox_inputs = self._process_inputs(inputs, name)
        ox_outputs = self._process_outputs(outputs, name)
        self._container.add_node(op_type, ox_inputs, ox_outputs, op_domain, op_version,
                                 name=self._scope.get_unique_operator_name(name), **attrs)
        return ox_outputs[0] if outputs is None else ox_outputs

    def add_node_with_output(self, op_type, inputs, outputs, name, op_domain='', op_version=None, **attrs):
        if op_version is None:
            op_version = self._container.target_opset
        ox_inputs = self._process_inputs(inputs, name)
        self._container.add_node(
            op_type, ox_inputs, outputs, op_domain, op_version, name=name, **attrs)
        return outputs

    def apply_op(self, apply_func, inputs, name=None, outputs=None, **attrs):
        name = self._generate_name(apply_func, name)
        ox_inputs = self._process_inputs(inputs, name)
        ox_outputs = self._process_outputs(outputs, name)
        apply_func(self._scope, ox_inputs, ox_outputs, self._container,
                   operator_name=self._scope.get_unique_operator_name(name), **attrs)
        return ox_outputs[0] if outputs is None else ox_outputs

    def constant(self, name=None, value=None, outputs=None):
        c_name = self._scope.get_unique_variable_name(name or 'c')
        c_value = self._value_to_tensor(value, c_name)
        return self.apply_op(onnx_ops.apply_constant2, [], name, outputs, value=c_value)

    def constant_of_shape(self, inputs, name=None, outputs=None, value=None):
        if not isinstance(value, (int, float, bool)):
            raise ValueError("constant_of_shape requires 'value' to be a scalar")
        c_name = self._scope.get_unique_variable_name(name or 'cos')
        c_value = self._value_to_tensor(value, c_name, atleast_1d=True)
        return self.apply_op(onnx_ops.apply_constant_of_shape, inputs, name, outputs, value=c_value)

    def slice(self, inputs, name=None, outputs=None, starts=None, ends=None, axes=None, steps=None):
        return self.apply_op(onnx_ops.apply_slice2, inputs, name, outputs,
                             starts=starts, ends=ends, axes=axes, steps=steps)

    def loop(self, trip_count, cond, body, inputs, outputs, name=None):
        name = self._generate_name('loop', name)
        trip_count = '' if trip_count is None else trip_count
        cond_name = '' if cond is None else cond
        ox_inputs = self._process_inputs(inputs, name)
        ox_inputs = [trip_count, cond_name] + ox_inputs
        ox_outputs = outputs
        self._container.add_node(
            'Loop', ox_inputs, ox_outputs, op_version=1, name=name, body=body)
        return ox_outputs

    # !!!!CODE-AUTOGEN!!!! #
    # The following code was generated by ../update_ops.py

    def abs(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_abs, inputs, name, outputs)

    def add(self, inputs, name=None, outputs=None, axis=None, broadcast=None):
        return self.apply_op(onnx_ops.apply_add, inputs, name, outputs, axis=axis, broadcast=broadcast)

    def argmax(self, inputs, name=None, outputs=None, axis=0, keepdims=1, select_last_index=0):
        return self.apply_op(onnx_ops.apply_argmax, inputs, name, outputs, axis=axis, keepdims=keepdims,
                             select_last_index=select_last_index)

    def argmin(self, inputs, name=None, outputs=None, axis=0, keepdims=1, select_last_index=0):
        return self.apply_op(onnx_ops.apply_argmin, inputs, name, outputs, axis=axis, keepdims=keepdims,
                             select_last_index=select_last_index)

    def affine(self, inputs, name=None, outputs=None, alpha=1.0, beta=0.0):
        return self.apply_op(onnx_ops.apply_affine, inputs, name, outputs, alpha=alpha, beta=beta)

    def batch_norm(self, inputs, name=None, outputs=None, epsilon=None, is_test=None, momentum=None, spatial=None):
        return self.apply_op(onnx_ops.apply_batch_norm, inputs, name, outputs, epsilon=epsilon, is_test=is_test,
                             momentum=momentum, spatial=spatial)

    def cast(self, inputs, name=None, outputs=None, to=None):
        return self.apply_op(onnx_ops.apply_cast, inputs, name, outputs, to=to)

    def clip(self, inputs, name=None, outputs=None, max=None, min=None):
        return self.apply_op(onnx_ops.apply_clip, inputs, name, outputs, max=max, min=min)

    def concat(self, inputs, name=None, outputs=None, axis=0):
        return self.apply_op(onnx_ops.apply_concat, inputs, name, outputs, axis=axis)

    def conv(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_conv, inputs, name, outputs)

    def crop_height_width(self, inputs, name=None, outputs=None, top_border=0, bottom_border=0, left_border=0,
                          right_border=0):
        return self.apply_op(onnx_ops.apply_crop_height_width, inputs, name, outputs, top_border=top_border,
                             bottom_border=bottom_border, left_border=left_border, right_border=right_border)

    def div(self, inputs, name=None, outputs=None, axis=None, broadcast=None):
        return self.apply_op(onnx_ops.apply_div, inputs, name, outputs, axis=axis, broadcast=broadcast)

    def elu(self, inputs, name=None, outputs=None, alpha=1.0):
        return self.apply_op(onnx_ops.apply_elu, inputs, name, outputs, alpha=alpha)

    def equal(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_equal, inputs, name, outputs)

    def exp(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_exp, inputs, name, outputs)

    def floor(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_floor, inputs, name, outputs)

    def flatten(self, inputs, name=None, outputs=None, axis=1):
        return self.apply_op(onnx_ops.apply_flatten, inputs, name, outputs, axis=axis)

    def gather(self, inputs, name=None, outputs=None, axis=0):
        return self.apply_op(onnx_ops.apply_gather, inputs, name, outputs, axis=axis)

    def gemm(self, inputs, name=None, outputs=None, alpha=1.0, beta=1.0, transA=0, transB=0):
        return self.apply_op(onnx_ops.apply_gemm, inputs, name, outputs, alpha=alpha, beta=beta, transA=transA,
                             transB=transB)

    def greater(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_greater, inputs, name, outputs)

    def greater_or_equal(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_greater_or_equal, inputs, name, outputs)

    def less_or_equal(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_less_or_equal, inputs, name, outputs)

    def gru(self, inputs, name=None, outputs=None, output_seq=0, reset_after=0):
        return self.apply_op(onnx_ops.apply_gru, inputs, name, outputs, output_seq=output_seq, reset_after=reset_after)

    def hard_sigmoid(self, inputs, name=None, outputs=None, alpha=None, beta=None):
        return self.apply_op(onnx_ops.apply_hard_sigmoid, inputs, name, outputs, alpha=alpha, beta=beta)

    def identity(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_identity, inputs, name, outputs)

    def instance_norm(self, inputs, name=None, outputs=None, epsilon=1e-05):
        return self.apply_op(onnx_ops.apply_instance_norm, inputs, name, outputs, epsilon=epsilon)

    def inverse(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_inverse, inputs, name, outputs)

    def leaky_relu(self, inputs, name=None, outputs=None, alpha=0.01):
        return self.apply_op(onnx_ops.apply_leaky_relu, inputs, name, outputs, alpha=alpha)

    def less(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_less, inputs, name, outputs)

    def log(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_log, inputs, name, outputs)

    def lstm(self, inputs, name=None, outputs=None, output_seq=0):
        return self.apply_op(onnx_ops.apply_lstm, inputs, name, outputs, output_seq=output_seq)

    def matmul(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_matmul, inputs, name, outputs)

    def max(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_max, inputs, name, outputs)

    def mean(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_mean, inputs, name, outputs)

    def min(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_min, inputs, name, outputs)

    def mul(self, inputs, name=None, outputs=None, axis=None, broadcast=None):
        return self.apply_op(onnx_ops.apply_mul, inputs, name, outputs, axis=axis, broadcast=broadcast)

    def neg(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_neg, inputs, name, outputs)

    def normalization(self, inputs, name=None, outputs=None, axis=1, p=2):
        return self.apply_op(onnx_ops.apply_normalization, inputs, name, outputs, axis=axis, p=p)

    def not_op(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_not_op, inputs, name, outputs)

    def pad(self, inputs, name=None, outputs=None, mode=None, pads=None, value=None, onnx_type=1):
        return self.apply_op(onnx_ops.apply_pad, inputs, name, outputs, mode=mode, pads=pads, value=value,
                             onnx_type=onnx_type)

    def parametric_softplus(self, inputs, name=None, outputs=None, alpha=None, beta=None):
        return self.apply_op(onnx_ops.apply_parametric_softplus, inputs, name, outputs, alpha=alpha, beta=beta)

    def pow(self, inputs, name=None, outputs=None, axis=None, broadcast=None):
        return self.apply_op(onnx_ops.apply_pow, inputs, name, outputs, axis=axis, broadcast=broadcast)

    def prelu(self, inputs, name=None, outputs=None, slope=None):
        return self.apply_op(onnx_ops.apply_prelu, inputs, name, outputs, slope=slope)

    def range(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_range, inputs, name, outputs)

    def reciprocal(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_reciprocal, inputs, name, outputs)

    def reducesum(self, inputs, name=None, outputs=None, axes=None, keepdims=1, rank=0):
        return self.apply_op(onnx_ops.apply_reducesum, inputs, name, outputs, axes=axes, keepdims=keepdims, rank=rank)

    def relu(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_relu, inputs, name, outputs)

    def relu_6(self, inputs, name=None, outputs=None, zero_value=0.0):
        return self.apply_op(onnx_ops.apply_relu_6, inputs, name, outputs, zero_value=zero_value)

    def reshape(self, inputs, name=None, outputs=None, desired_shape=None):
        return self.apply_op(onnx_ops.apply_reshape, inputs, name, outputs, desired_shape=desired_shape)

    def resize(self, inputs, name=None, outputs=None, mode='nearest', coordinate_transformation_mode='asymmetric',
               scales=None):
        return self.apply_op(onnx_ops.apply_resize, inputs, name, outputs, mode=mode,
                             coordinate_transformation_mode=coordinate_transformation_mode, scales=scales)

    def rnn(self, inputs, name=None, outputs=None, output_seq=0):
        return self.apply_op(onnx_ops.apply_rnn, inputs, name, outputs, output_seq=output_seq)

    def shape(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_shape, inputs, name, outputs)

    def sigmoid(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_sigmoid, inputs, name, outputs)

    def softsign(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_softsign, inputs, name, outputs)

    def selu(self, inputs, name=None, outputs=None, alpha=1.673263, gamma=1.050701):
        return self.apply_op(onnx_ops.apply_selu, inputs, name, outputs, alpha=alpha, gamma=gamma)

    def softmax(self, inputs, name=None, outputs=None, axis=None):
        return self.apply_op(onnx_ops.apply_softmax, inputs, name, outputs, axis=axis)

    def scaled_tanh(self, inputs, name=None, outputs=None, alpha=None, beta=None):
        return self.apply_op(onnx_ops.apply_scaled_tanh, inputs, name, outputs, alpha=alpha, beta=beta)

    def slice2(self, inputs, name=None, outputs=None, starts=None, ends=None, axes=None, steps=None):
        return self.apply_op(onnx_ops.apply_slice2, inputs, name, outputs, starts=starts, ends=ends, axes=axes,
                             steps=steps)

    def split(self, inputs, name=None, outputs=None, split=None, axis=0):
        return self.apply_op(onnx_ops.apply_split, inputs, name, outputs, split=split, axis=axis)

    def sqrt(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_sqrt, inputs, name, outputs)

    def squeeze(self, inputs, name=None, outputs=None, axes=None, rank=0):
        return self.apply_op(onnx_ops.apply_squeeze, inputs, name, outputs, axes=axes, rank=rank)

    def sub(self, inputs, name=None, outputs=None, axis=None, broadcast=0):
        return self.apply_op(onnx_ops.apply_sub, inputs, name, outputs, axis=axis, broadcast=broadcast)

    def sum(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_sum, inputs, name, outputs)

    def tanh(self, inputs, name=None, outputs=None):
        return self.apply_op(onnx_ops.apply_tanh, inputs, name, outputs)

    def thresholded_relu(self, inputs, name=None, outputs=None, alpha=None):
        return self.apply_op(onnx_ops.apply_thresholded_relu, inputs, name, outputs, alpha=alpha)

    def tile(self, inputs, name=None, outputs=None, repeats=None):
        return self.apply_op(onnx_ops.apply_tile, inputs, name, outputs, repeats=repeats)

    def transpose(self, inputs, name=None, outputs=None, perm=None):
        return self.apply_op(onnx_ops.apply_transpose, inputs, name, outputs, perm=perm)

    def upsample(self, inputs, name=None, outputs=None, mode='nearest', coordinate_transformation_mode='asymmetric',
                 scales=None):
        return self.apply_op(onnx_ops.apply_upsample, inputs, name, outputs, mode=mode,
                             coordinate_transformation_mode=coordinate_transformation_mode, scales=scales)

    def unsqueeze(self, inputs, name=None, outputs=None, axes=None, rank=0):
        return self.apply_op(onnx_ops.apply_unsqueeze, inputs, name, outputs, axes=axes, rank=rank)
