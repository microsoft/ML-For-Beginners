# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import copy
import onnx
import logging
import numpy as np
from onnx import onnx_pb as onnx_proto
from inspect import getfullargspec

from . import onnx_ops
from .registration import register_converter
from .topology import Topology, convert_topology
from .oopb import OnnxOperatorBuilder
from .onnx_ex import OPSET_TO_IR_VERSION, get_maximum_opset_supported
from .data_types import (DoubleTensorType, FloatTensorType,
                         Int64TensorType, Int32TensorType, BooleanTensorType,
                         Complex64TensorType, Complex128TensorType, StringTensorType)

_logger = logging.getLogger(__name__)


class GraphFunctionType:
    D = DoubleTensorType
    F = FloatTensorType
    I = Int64TensorType  # noqa: E741 ambiguous variable name 'I'
    B = BooleanTensorType
    I32 = Int32TensorType
    C64 = Complex64TensorType
    C128 = Complex128TensorType
    S = StringTensorType

    d = DoubleTensorType(shape=[])
    f = FloatTensorType(shape=[])
    i = Int64TensorType(shape=[])
    i32 = Int32TensorType(shape=[])
    b = BooleanTensorType(shape=[])
    c64 = Complex64TensorType(shape=[])
    c128 = Complex64TensorType(shape=[])
    s = StringTensorType(shape=[])


def _get_python_function_arguments(f):
    """
    Helper to get the parameter names and annotations of a Python function.
    """
    # Note that we only return non-optional arguments (we assume that any optional args are not specified).
    # This allows to, e.g., accept max(a, b, *more, name='') as a binary function
    param_specs = getfullargspec(f)
    annotations = param_specs.annotations
    arg_names = param_specs.args
    # "if this tuple has n elements, they correspond to the last n elements listed in args"
    defaults = param_specs.defaults
    if defaults:
        # we allow Function(functions with default arguments),
        # but those args will always have default values since CNTK Functions do not support this
        arg_names = arg_names[:-len(defaults)]
    return arg_names, annotations


class _SimpleRawModelContainer(object):
    def __init__(self, inputs, outputs):
        self.input_names = inputs
        self.output_names = outputs


class Graph:
    """
    The ONNX Graph build from a python function, or load it from an ONNX object
    """
    opset = get_maximum_opset_supported()

    inference_runtime = None

    def __init__(self, name):
        self._name = name
        self._oxml = None
        self._onnxfunc = None
        self._onnxfunc_args = []
        self._onnxfunc_kwargs = {}

    @property
    def oxml(self):
        return self.to_model()

    @property
    def name(self):
        return self._name

    def to_model(self):
        if self._oxml is None:
            self._defer_building()
        return self._oxml

    def _bind(self, oxml, inputs, outputs):
        ox_graph = oxml.graph
        initializer_set = {
            initializer.name for initializer in ox_graph.initializer}
        model_inputs = [
            input.name for input in ox_graph.input if input.name not in initializer_set]
        model_outputs = [output.name for output in ox_graph.output]
        if inputs is None:
            inputs = model_inputs
        else:
            assert {input for input in inputs} == {
                input for input in
                model_inputs}, "User-specified set of inputs does not match actual set"
        if outputs is None:
            outputs = model_outputs
        else:
            assert {output for output in outputs} == {
                output for output in
                model_outputs}, "User-specified set of outputs does not match actual set"
        self._oxml = oxml
        self._inputs = inputs
        self._outputs = outputs

    @staticmethod
    def _create(func, *args, **kwargs):
        name = kwargs.get('name', None)
        f_name = name or func.__name__
        graph = Graph(f_name)
        graph._onnxfunc = func
        graph._onnxfunc_args = args
        graph._onnxfunc_kwargs = kwargs
        return graph

    @staticmethod
    def trace(*args, **kwargs):
        """
        This is the decorator. Example:
        @Graph.trace(outputs="logits")
        def model(source_sequence):
            ...
        """
        if len(args) > 0 and hasattr(args[0], '__call__'):  # first arg is function
            return Graph._create(args[0])
        else:
            return lambda f: Graph._create(f, *args, **kwargs)

    @staticmethod
    def _to_list(element_or_list):
        if element_or_list is None:
            return []

        return element_or_list if isinstance(
            element_or_list, (list, tuple)) else [element_or_list]

    @staticmethod
    def _on_conversion(scope, operator, container):
        with OnnxOperatorBuilderX(container, scope).as_default(operator.full_name) as ox:  # type: OnnxOperatorBuilderX
            f = operator.raw_operator
            # if ox.upper_context is not None:
            #     container.enable_optimizer = False  # optimizer on a subgraph is not supported yet.
            container.enable_optimizer = False
            fn_inputs = [ox.arg(arg_name) for arg_name in operator.input_full_names]
            f_outputs = f(*fn_inputs)
            outputs = operator.output_full_names
            if outputs:
                if isinstance(f_outputs, Tensor):
                    f_outputs = ox.identity([f_outputs], outputs=outputs)
                    n_outputs = 1
                else:
                    f_outputs = [ox.identity([f_output], outputs=[output_name])
                                 for f_output, output_name in zip(f_outputs, outputs)]
                    n_outputs = len(f_outputs)
            assert n_outputs == len(
                outputs), "Function {}() returned {} but {} were declared".format(
                operator.full_name, n_outputs, len(outputs))

    @staticmethod
    def _enforce_opset_version(oxml):
        for im_ in oxml.opset_import:
            if (im_.domain == '' or im_.domain == 'ai.onnx') and \
                    im_.version != Graph.opset:
                im_.version = Graph.opset
                opv = Graph.opset
                irv = OPSET_TO_IR_VERSION.get(opv, onnx_proto.IR_VERSION)
                oxml.ir_version = irv
                _logger.warning('The maximum opset needed by this model is updated to %d.' % Graph.opset)

    def _build_graph(self, f, input_types=None, output_types=None, outputs=None):
        input_types = Graph._to_list(input_types)
        output_types = Graph._to_list(output_types)
        outputs = Graph._to_list(outputs)
        if not outputs:
            outputs.append(f.__name__)

        f_name = self._name
        arg_names, _ = _get_python_function_arguments(f)
        raw_model = _SimpleRawModelContainer(arg_names, outputs)
        topo = Topology(raw_model)
        top_level = topo.declare_scope(f_name)
        graph_opname = f_name
        op_whole = top_level.declare_local_operator(graph_opname, f)
        register_converter(op_whole.type, type(self)._on_conversion, overwrite=True)
        for i_ in raw_model.input_names:
            vi_ = top_level.get_local_variable_or_declare_one(i_,
                                                              FloatTensorType(shape=[None]) if not input_types else
                                                              input_types[
                                                                  arg_names.index(i_)])
            op_whole.inputs.append(vi_)
        for o_ in raw_model.output_names:
            vo_ = top_level.get_local_variable_or_declare_one(o_,
                                                              FloatTensorType(shape=[None]) if not output_types else
                                                              output_types[outputs.index(o_)])
            op_whole.outputs.append(vo_)

        oxml = convert_topology(topo, f_name, "onnx.fn: {}".format(f_name), target_opset=Graph.opset)
        type(self)._enforce_opset_version(oxml)
        self._bind(oxml, arg_names, outputs)
        return self

    @staticmethod
    def _map_function_arguments(params, params_set, *args, **kwargs):
        """
        Helper to determine the argument map for use with various call operations.
        Returns a dictionary from parameters to whatever arguments are passed.
        Accepted are both positional and keyword arguments.
        This mimics Python's argument interpretation, except that keyword arguments are not optional.
        """
        # start with positional arguments
        arg_map = dict(zip(params, args))

        # now look up keyword arguments
        if len(kwargs) != 0:
            for name, arg in kwargs.items():  # keyword args are matched by name
                if name not in params_set:
                    raise TypeError("got an unexpected keyword argument '%s'" % name)
                if name in arg_map:
                    raise SyntaxError("got multiple values for argument '%s'" % name)
                arg_map[name] = arg  # add kw argument to dict
        assert len(arg_map) == len(params)

        return arg_map

    def _defer_building(self):
        if self._oxml is None:
            self._build_graph(self._onnxfunc, *self._onnxfunc_args, **self._onnxfunc_kwargs)

    def _argument_map(self, *args, **kwargs):
        """
        Determines the {placeholder: variable} map for use with various call operations
        Returns a dictionary from this function's placeholders to whatever arguments are passed.
        Accepted are both positional and keyword arguments.
        This mimics Python's argument interpretation, except that keyword arguments are not optional
        (there is no concept of default value).
        """
        params = self._inputs
        if len(args) + len(kwargs) != len(params):
            raise TypeError("Graph invocation expected {} arguments, got {}".format(
                len(params), len(args) + len(kwargs)))
        params_set = {arg for arg in params}
        return Graph._map_function_arguments(params, params_set, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._defer_building()
        # parse argument list and map to the function's input
        arg_map = self._argument_map(*args, **kwargs)
        # determine whether this is eval() or clone()
        is_symbolic = any(isinstance(arg, Tensor) for arg in arg_map.values())
        if is_symbolic:
            first_arg = next(iter(arg_map.values()))
            ox = first_arg.ox
            output_map = {output: None for output in self._outputs}
            # @TODO: outputs missing
            return ox.apply_invoke_inline(self.oxml.graph, arg_map, output_map)
        else:
            assert Graph.inference_runtime is not None, "No inference runtime has been provided."
            # evaluate with real values
            kwargs = {name: OnnxOperatorBuilderX.value_to_ndarray(val) for name, val in arg_map.items()}
            return Graph.inference_runtime(self.oxml, kwargs)

    def save(self, path):
        onnx.save_model(self.oxml, path)

    @staticmethod
    def load(path_or_model, name=None, inputs=None, outputs=None):
        """
        Construct a Graph object by loading an ONNX model.
        """
        oxml = onnx.load_model(path_or_model) if isinstance(path_or_model, str) else path_or_model
        for opset_import in oxml.opset_import:
            if opset_import.domain == '':
                if Graph.opset < opset_import.version:
                    raise RuntimeError("Graph opset < model opset: Graph opset = " + str(Graph.opset)
                                       + ", model opset = " + str(opset_import.version))
                elif Graph.opset > opset_import.version:
                    Graph._enforce_opset_version(oxml)
        g = Graph(name or oxml.graph.name)
        g._bind(oxml, inputs=inputs, outputs=outputs)
        return g


class Tensor(object):
    def __init__(self, tensor_name: str, ox):
        self.name = tensor_name
        self.ox = ox

    def _to_binary_tensor_args(self, other):
        # convert self, other to [self, other], but if either is a number, convert that to a constant
        x, y = self, other
        if isinstance(y, (int, float, bool, np.ndarray)):
            y = self.ox.constant(value=y)
        elif isinstance(x, (int, float, bool, np.ndarray)):
            x = self.ox.constant(value=x)
        return [x, y]

    def __add__(self, other):
        return self.ox.add(self._to_binary_tensor_args(other))

    def __sub__(self, other):
        return self.ox.sub(self._to_binary_tensor_args(other))

    def __mul__(self, other):
        return self.ox.mul(self._to_binary_tensor_args(other))

    def __div__(self, other):
        return self.ox.div(self._to_binary_tensor_args(other))

    def __pow__(self, other):
        return self.ox.pow(self._to_binary_tensor_args(other))

    def __matmul__(self, other):
        return self.ox.matmul(self._to_binary_tensor_args(other))

    def __lt__(self, other):
        return self.ox.less(self._to_binary_tensor_args(other))

    def __le__(self, other):
        return self.ox.less_or_equal(self._to_binary_tensor_args(other))

    def __eq__(self, other):
        return self.ox.equal(self._to_binary_tensor_args(other))

    def __ne__(self, other):  # ONNX has no NotEqual
        return self.ox.not_op([self.ox.equal(self._to_binary_tensor_args(other))])

    def __gt__(self, other):
        return self.ox.greater(self._to_binary_tensor_args(other))

    def __ge__(self, other):
        return self.ox.greater_or_equal(self._to_binary_tensor_args(other))

    def __neg__(self):
        return self.ox.neg([self])

    def __not__(self):
        return self.ox.not_op([self])

    def __getitem__(self, indices):
        # normalize indices to tuples of slices
        # Formats encountered:
        #  - a single int
        #  - a tuple of (int or slice)
        if not isinstance(indices, (tuple, list)):  # single item: make it a tuple
            indices = (indices,)
        squeeze = [axis for axis, index in enumerate(indices) if
                   isinstance(index, int)]  # which axes had a single index?
        indices = tuple(
            index if isinstance(index, slice) else slice(index, index + 1 if index != -1 else None, 1) for index in
            indices)  # make all tuple items of type Slice
        bs, es, ss, ds = [], [], [], []
        INT_MAX = 2 ** 63 - 1
        for axis, index in enumerate(indices):
            if not isinstance(index, slice):
                raise ValueError("Index expected")
            if index.start is None and index.stop is None:  # [:] can be skipped
                continue
            b, e, s = index.start, index.stop, index.step
            bs.append(b if b is not None else 0)
            es.append(e if e is not None else INT_MAX)
            ss.append(s if s is not None else 1)
            ds.append(axis)
        res = self.ox.slice(self, starts=bs, ends=es, axes=ds, steps=ss)
        if squeeze:  # single index means we must drop the axis
            res = self.ox.squeeze([res], axes=squeeze)
        return res

    _all_ops = [op_name[6:] for op_name in dir(onnx_ops) if op_name.startswith("apply_")] + \
               ['shape', 'constant_of_shape', 'range', 'slice', 'equal']  # these are temporarily declared here

    def __getattribute__(self, attr):
        """
        A little hack that allows to call unary operators in a chaining fashion,
        e.g. x.shape() instead of ox.shape(x).
        """
        if attr in Tensor._all_ops:
            f = self.ox.__getattribute__(attr)

            def call_it(*args, **kwargs):
                assert len(args) == 0, "In chaining expressions, only keyword args are allowed"
                assert "inputs" not in kwargs, "Chaining expressions do not currently support additional inputs"
                return f(self, *args, **kwargs)

            return call_it
        else:
            return object.__getattribute__(self, attr)


class OnnxOperatorBuilderX(OnnxOperatorBuilder):
    def _output_names_to_tensors(self, outputs):
        if isinstance(outputs, str):
            return Tensor(outputs, self)
        else:
            return tuple(self._output_names_to_tensors(output) for output in outputs)

    def _tensors_to_input_names(self, inputs):
        if isinstance(inputs, Tensor):
            return inputs.name
        else:
            return [self._tensors_to_input_names(input) for input in inputs]

    def apply_op(self, apply_func_or_op_type, inputs, name=None, outputs=None, **attrs):  # override!
        inputs = self._tensors_to_input_names(inputs)
        if isinstance(apply_func_or_op_type, str):
            return self._output_names_to_tensors(
                super().add_node(apply_func_or_op_type, inputs, name=name, outputs=outputs, **attrs))
        else:
            return self._output_names_to_tensors(
                super().apply_op(apply_func_or_op_type, inputs, name=name, outputs=outputs, **attrs))

    def apply_invoke_inline(self, ox_graph, input_map, output_map):
        input_map = dict(input_map)
        output_map = dict(output_map)
        # input_map:  [name in graph] -> actual input Tensor
        # output_map: [name in graph] -> desired name for the result, or None
        f_name = "invoke_inline_" + ox_graph.name
        for graph_output in output_map.keys():  # @TODO: use proper comprehensions
            output_map[graph_output] = self._process_outputs(
                output_map[graph_output], name=f_name)[0]
        outputs = list(output_map.values())  # remember these; these are the outputs of this invocation
        if len(outputs) == 1:
            outputs = outputs[0]  # single output
        for graph_input in input_map.keys():
            input_map[graph_input] = self._process_inputs(
                [input_map[graph_input].name], name=f_name)[0]
        _logger.debug(f_name, input_map, output_map)

        existing_node_names = {item.name: item for item in self._container.nodes}
        existing_initializer_names = {item.name: item for item in self._container.initializers}
        existing_value_infos = {item.name: item for item in self._container.value_info}

        # collect all outputs from the graph we are expanding, so that we can map them to unique names
        # @TODO: This will also map some code that may be shared later on. Leave that to the optimizer.
        node_map = dict()
        for node in ox_graph.node:
            if not node.input:  # leaves do not need to be mapped; they can just get uniq'ed
                continue
            for output in node.output:
                if output in output_map:  # this is an actual output that already has been mapped
                    continue
                uniq_name = onnx_ops._create_name_or_use_existing_one(self._scope, self._generate_name(output, None),
                                                                      None)
                output_map[output] = uniq_name
            uniq_node_name = onnx_ops._create_name_or_use_existing_one(self._scope,
                                                                       self._generate_name(node.name, None), None)
            node_map[output] = uniq_node_name

        def map_tensors(args, arg_map):
            for i in range(len(args)):
                if args[i] in arg_map:
                    _logger.debug("Remapping", args[i], "to", arg_map[args[i]])
                    args[i] = arg_map[args[i]]

        for node in ox_graph.node:
            node = copy.deepcopy(node)  # since we patch, we must clone it first
            map_tensors(node.input, input_map)  # patch the input references to the function arguments
            map_tensors(node.output, output_map)  # rename the outputs to unique ones
            map_tensors(node.input, output_map)  # outputs may be inputs to other nodes in this graph
            if node.name in node_map:
                node.name = node_map[node.name]
            if node.name in existing_node_names:
                str_node = str(node)
                str_other = str(existing_node_names[node.name])
                if str_node != str_other:
                    # must be the same, otherwise we have inconsistent dups, e.g. in input models
                    _logger.info("Duplicate node name with inconsistent nodes:\n", node, "vs:\n",
                                 existing_node_names[node.name])
                    assert str_node == str_other
                continue
            self._container.nodes.append(node)
        for initializer in ox_graph.initializer:
            if initializer.name in existing_initializer_names:  # @TODO: check if they are the same
                _logger.info("Duplicate initializer name skipped:", initializer.name)
                continue
            if initializer.name in output_map:  # technically, the whole function could be a lonely initializer
                # _logger.debug("Replacing:", initializer.name, initializer.shape)
                initializer = copy.deepcopy(initializer)
                initializer.name = output_map[initializer.name]
            # _logger.debug(initializer.name)
            self._container.initializers.append(initializer)
        for value_info in ox_graph.value_info:
            if value_info.name in existing_value_infos:  # @TODO: check if they are the same
                _logger.info("Duplicate value_info name skipped:", value_info.name)
                continue
            # @TODO: Not sure what must be mapped, and how
            _logger.debug(value_info)
            self._container.value_info.append(value_info)
        return self._output_names_to_tensors(outputs)  # note: outputs is either a string or a list of strings

    def arg(self, name):
        """
        Use this to create a function argument
        """
        return Tensor(name, self)

    # @TODO?: Should we follow the conventions in the spec? loop(self, count, cond, inputs, body)
    def loop(self, count, cond, body, inputs, outputs=None, name=None):
        inputs = self._tensors_to_input_names(inputs)
        count = None if count is None else self._tensors_to_input_names(count)
        if cond is not None:
            cond = self._tensors_to_input_names(cond)
        else:
            cond = self._tensors_to_input_names(self.constant(value=np.array([True]), name="cf"))
        sub_graph = body.oxml.graph
        return self._output_names_to_tensors(super().loop(count, cond, sub_graph,
                                                          inputs=inputs,
                                                          outputs=outputs,  # @TODO: unique output names
                                                          name=name))
