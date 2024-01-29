# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from onnx import load
from onnx.defs import onnx_opset_version
from onnx.onnx_pb import FunctionProto, GraphProto, ModelProto, NodeProto, TypeProto
from onnx.reference.op_run import (
    OpFunctionContextDependant,
    OpRun,
    OpRunExpand,
    RuntimeContextError,
    to_array_extended,
)
from onnx.reference.ops_optimized import optimized_operators


class ReferenceEvaluator:
    """
    Computes the outputs of an ONNX proto
    (`ModelProto`, `FunctionProto`, `GraphProto`, `NodeProto`).
    This is a pure python implementation of ONNX specifications.
    Mismatches may remain between the official specifications and the implementation here.
    In the case of such a mismatch, the official spec overrides this implementation.

    :param proto: :class:`onnx.ModelProto`, :class:`onnx.GraphProto`,
        :class:`onnx.FunctionProto`, :class:`onnx.NodeProto`,
        filename or bytes
    :param verbose: display intermediate results
        on the standard output during the execution
    :param opsets: if *proto* is an instance of *GraphProto*,
        opsets must be defined by a dictionary of
    :param functions: known onnx functions
    :param new_ops: this runtime can be used to test the implementations
        of new operators, *new_ops* is a list of classes
        derived from :class:`OpRun <onnx.reference.op_run.OpRun>`,
        every class must define the static attribute `domain`,
        there may be multiple implementations for the same operator,
        the first one in the list is used.
    :param optimized: some operators have two implementations,
        a naive one corresponding to definition of the mathematical
        definition of the operator, another one more efficient.
        This is the case for operator Conv. The naive version is ten times
        slower than the optimized one using a decomposition
        into *Conv = im2col + Gemm*. If True, all optimized
        kernels are added in `new_ops` and are used instead of the
        inner implementation if list *new_ops* does not already contain
        one.

    The class maps every node to its associated implementation.
    When a subgraph of a function is met,
    it uses this class to execute the subgraph or the function.
    Next example shows how to run `ReferenceEvaluator` with an onnx model
    stored in file `model.onnx`.

    ::

        import numpy as np
        from onnx.reference import ReferenceEvaluator

        X = np.array(...)
        sess = ReferenceEvaluator("model.onnx")
        results = sess.run(None, {"X": X})
        print(results[0])  # display the first result

    Parameter *verbose* may be used to show intermediate results.

    ::

        import numpy as np
        from onnx.reference import ReferenceEvaluator

        X = np.array(...)
        sess = ReferenceEvaluator("model.onnx", verbose=1)
        results = sess.run(None, {"X": X})
        print(results[0])  # display the first result

    The class can use any implementation available in folder
    `ops <https://github.com/onnx/onnx/tree/main/onnx/reference/ops>`_.
    Adding an implementation requires two changes. The first one is
    the implementation itself. Any existing node can be used as a template.
    The second is one line in file `_op_list.py
    <https://github.com/onnx/onnx/tree/main/onnx/reference/ops/_op_list.py>`_
    to import the file and let the reference evaluator know it exists.

    This class can also be used to test an implementation of
    a custom operator. Let's assume this new operator
    is `InvAlpha` from domain `custom`. The implementation
    must take place in a class inheriting from
    :class:`OpRun <onnx.reference.op_run.OpRun>`.
    It must also define attribute `op_domain`.
    Here is an example which computes :math:`\\frac{1}{X + \\alpha}`.

    .. exec_code::

        from onnx.reference.op_run import OpRun

        class InvAlpha(OpRun):

            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore
                # None must be the default value, it is automatically
                # replaced by class OpRun with either the default value
                # specified in the NodeProto or an attribute value defined
                # in a `FunctionProto`.
                return (1 / (x + alpha),)

    `alpha` is an attribute. It can be defined by the onnx node or
    be defined by the function using this node. It is safe to assume
    that attributes are known at the same time as the input.
    Class `ReferenceEvaluator` must know about this new implementation
    and this can be done by specified argument *new_ops*.

    ::

        sess = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        got = sess.run(None, {"X": x})[0]

    A specific node can be simply evaluated.

    .. exec_code::

        import numpy as np
        from onnx.reference.ops._op_list import Celu

        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Celu.eval(x, alpha=0.5)
        print(y)

    This can also be expressed as:

    .. exec_code::

        import numpy as np
        from onnx.reference.ops import load_op

        Celu = load_op("", "Celu")  # domain is ""
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Celu.eval(x, alpha=0.5)
        print(y)

    It is possible to overwrite an existing operator.
    The class name must be the same. The domain does not have
    to be specified for the default domain. However, by default,
    class `OpRun` will load the most recent for this operator.
    It can be explicitly specified by adding static attribute
    `op_schema` of type :class:`OpSchema
    <onnx.onnx_cpp2py_export.defs.OpSchema>`.

    ::

        from onnx.reference.op_run.op_conv import Conv as _Conv

        class Conv(_Conv):

            op_schema = instance_of_OpSchema()

            def _run(self, ...):
                ...

        An operator may be different in a later opset. In that case,
        a new implementation needs to be registered. `Pad_11`, `Pad_18`.
        `Pad_11` is the implementation chose for opset in [11, 17].
        `Pad_18` is selected for any greater opset. Both classes must be
        imported into file `_op_list.py` to register their existence to the
        runtime.

        An operator may have a reference implementation such as `CastLike`
        and still be defined as a function. By default, the reference implementation
        is used. This behaviour can be changed by adding a class to the list
        of overwritten operators. It must inherit from :class:`OpRunExpand`.

        ::

            from onnx.reference.op_run import OpRunExpand

            class CastLike(OpRunExpand):
                op_domain = ""

            ref = ReferenceEvaluator(model, new_ops=[CastLike])
            # ...

            This mechanism is used in unit test to check the function
            implementation a schema may define.
    """

    def __init__(  # type: ignore
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union["ReferenceEvaluator", FunctionProto]]] = None,  # type: ignore
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
        optimized: bool = True,
    ):
        if optimized:
            if new_ops is None:
                new_ops = optimized_operators.copy()
            else:
                set_new_ops = set(new_ops)
                for op in optimized_operators:
                    if op not in set_new_ops:
                        new_ops.append(op)
        self.output_types_ = None
        self.input_types_ = None
        if isinstance(proto, str):
            with open(proto, "rb") as f:
                proto = load(f)
        elif isinstance(proto, bytes):
            proto = load(BytesIO(proto))
        self.proto_ = proto
        self.functions_: Dict[Tuple[str, str], ReferenceEvaluator] = {}
        self.attributes_: List[str] = []
        if isinstance(proto, ModelProto):
            self.onnx_graph_ = proto.graph
            self.opsets_ = {d.domain: d.version for d in proto.opset_import}
            if opsets is not None:
                raise ValueError("opsets must be None if proto is ModelProto.")
            if functions is not None:
                raise ValueError("functions must be None if proto is ModelProto.")
            functions = proto.functions  # type: ignore[assignment]
        elif isinstance(proto, GraphProto):
            self.onnx_graph_ = proto
            if not isinstance(opsets, dict):
                raise ValueError("opsets must be a dictionary if proto is GraphProto.")
            self.opsets_ = opsets
        elif isinstance(proto, FunctionProto):
            self.onnx_graph_ = None  # type: ignore
            self.opsets_ = {d.domain: d.version for d in proto.opset_import}
            if opsets is not None:
                raise ValueError("opsets must be None if proto is FunctionProto.")
            self.attributes_ = list(proto.attribute)
        elif isinstance(proto, NodeProto):
            self.onnx_graph_ = None  # type: ignore
            self.opsets_ = {
                proto.domain: 1 if proto.domain != "" else onnx_opset_version()
            }
        else:
            raise TypeError(f"Unexpected type {type(proto)} for proto.")
        if self.onnx_graph_:
            self.input_names_ = [i.name for i in self.onnx_graph_.input]
            self.input_types_ = [i.type for i in self.onnx_graph_.input]
            self.output_names_ = [o.name for o in self.onnx_graph_.output]
            self.output_types_ = [i.type for i in self.onnx_graph_.output]
            self.inits_ = list(self.onnx_graph_.initializer) + list(
                self.onnx_graph_.sparse_initializer  # type: ignore
            )
            self.nodes_ = self.onnx_graph_.node
            all_types = {i.name: i.type for i in self.onnx_graph_.input}
            if hasattr(self.proto_, "value_info"):
                for shape_type in self.proto_.value_info:
                    all_types[shape_type.name] = shape_type.type
            self.all_types_ = all_types
        else:
            self.input_names_ = list(proto.input)
            self.output_names_ = list(proto.output)
            self.inits_ = []
            if isinstance(proto, NodeProto):
                self.nodes_ = [proto]  # type: ignore[assignment]
            else:
                self.nodes_ = proto.node
        if functions is not None:
            for f in functions:  # type: ignore
                if isinstance(f, FunctionProto):
                    existing_functions = list(self.functions_.values())
                    self.functions_[f.domain, f.name] = ReferenceEvaluator(
                        f, verbose=verbose, functions=existing_functions
                    )
                elif isinstance(f, ReferenceEvaluator):
                    onx = f.proto_  # type: ignore
                    self.functions_[onx.domain, onx.name] = f
                else:
                    raise TypeError(f"Unexpected type {type(f)!r} for a function.")
        self.verbose = verbose
        self.new_ops_: Dict[Tuple[str, str], OpRun] = {}
        if new_ops is not None:
            for cl in new_ops:
                if not hasattr(cl, "op_domain"):
                    raise AttributeError(
                        f"Class {cl} must define attribute 'op_domain'."
                    )
                if not issubclass(cl, OpRun):  # type: ignore
                    raise TypeError(f"Class {cl} must inherit from OpRun (in new_ops).")
                key = cl.op_domain, cl.__name__  # type: ignore
                if key in self.new_ops_:
                    # Already an implementation, the first one is used.
                    continue
                self.new_ops_[key] = cl
        self._init()

    def _log_arg(self, a: Any) -> Any:
        if isinstance(a, (str, int, float)):
            return a
        if isinstance(a, np.ndarray):
            if self.verbose < 4:  # noqa: PLR2004
                return f"{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
            elements = a.ravel().tolist()
            if len(elements) > 5:  # noqa: PLR2004
                elements = elements[:5]
                return f"{a.dtype}:{a.shape}:{','.join(map(str, elements))}..."
            return f"{a.dtype}:{a.shape}:{elements}"
        if hasattr(a, "append"):
            return ", ".join(map(self._log_arg, a))
        return a

    def _log(self, level: int, pattern: str, *args: List[Any]) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))

    @property
    def input_names(self):  # type: ignore
        "Returns the input names."
        return self.input_names_

    @property
    def input_types(self):  # type: ignore
        "Returns the input types if any specified."
        return self.input_types_

    @property
    def output_names(self):  # type: ignore
        "Returns the output names."
        return self.output_names_

    @property
    def output_types(self):  # type: ignore
        "Returns the output types."
        return self.output_types_

    @property
    def opsets(self):  # type: ignore
        "Returns the opsets."
        return self.opsets_

    @property
    def has_linked_attribute(self):
        """
        Checks if the graph has a linked attribute (= an attribute whose value is defined
        by a function attribute.
        """
        return any(node.has_linked_attribute for node in self.rt_nodes_)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.input_names)}) -> {', '.join(self.output_names)}"

    def get_result_types(self, name: str, exc: bool = True) -> Any:
        if self.all_types_ is None:
            raise RuntimeError(
                f"Unable to return type for name {name!r}. Run shape_inference first."
            )
        if name not in self.all_types_:
            if exc:
                raise RuntimeError(
                    f"Unable to return type for name {name!r}, it was not found in {sorted(self.all_types_)}."
                )
            return None
        return self.all_types_[name]

    def _init(self) -> None:
        """
        Loads the implementation for every node in the graph.
        """
        self.rt_inits_ = {}
        self.rt_nodes_ = []
        for init in self.inits_:
            self.rt_inits_[init.name] = to_array_extended(init)  # type: ignore[union-attr,arg-type]
        run_params = {
            "log": lambda pattern, *args: self._log(10, pattern, *args),
            "opsets": self.opsets,
            "verbose": self.verbose,
            "new_ops": self.new_ops_,
        }
        if self.input_types_:
            all_types = {i.name: i.type for i in self.onnx_graph_.input}
            if hasattr(self.proto_, "value_info"):
                for shape_type in self.proto_.value_info:
                    all_types[shape_type.name] = shape_type.type
            self.all_types_ = all_types
        else:
            self.all_types_ = None  # type: ignore

        for node in self.nodes_:
            try:
                cl = self._load_impl(node)
            except RuntimeContextError as e:
                # A node has a context dependent implementation.
                # Shape inference must be run to get the input types.
                if self.all_types_:
                    it = [self.get_result_types(i, exc=False) for i in node.input]
                    if None in it:
                        # One input does not exist. It must be done while executing the graph.
                        cl = lambda *args, parent=self: OpFunctionContextDependant(  # noqa: E731
                            *args, parent=parent
                        )
                    else:
                        cl = self._load_impl(node, it)  # type: ignore
                else:
                    raise RuntimeContextError(
                        f"No implementation was found for node type {node.op_type!r} from domain {node.domain!r}. "
                        f"If this node has a context dependent implementation, you should run function infer_shapes "
                        f"before calling ReferenceEvaluator."
                    ) from e
            try:
                inst = cl(node, run_params)
            except TypeError as e:
                raise TypeError(
                    f"Unable to instantiate class {cl!r} with "
                    f"run_params={run_params} and node={node}."
                ) from e
            self.rt_nodes_.append(inst)

    def _load_impl(  # noqa: PLR0911
        self, node: NodeProto, input_types: Optional[TypeProto] = None
    ) -> Any:
        """
        Loads the implementation for a specified runtime.
        """
        if node.domain not in self.opsets:
            raise RuntimeError(
                f"Domain {node.domain!r} (node type: {node.op_type!r}) "
                f"is not specified. Known opsets: {self.opsets!r}."
            )
        version = self.opsets[node.domain]
        key = node.domain, node.op_type
        expand = False
        if key in self.new_ops_:
            # This operator has a custom implementation.
            # This mechanism can be used to implement a custom onnx node
            # or to overwrite an existing one.
            cl = self.new_ops_[key]
            if not issubclass(cl, OpRunExpand):
                return cl
            # It must be replaced by its implementation defined in its schema.
            expand = True

        if node.domain == "":
            from onnx.reference.ops import load_op

            try:
                return load_op(node.domain, node.op_type, version, expand=expand)
            except RuntimeContextError:
                if input_types is None:
                    raise
                return load_op(
                    node.domain,
                    node.op_type,
                    version,
                    node=node,
                    input_types=input_types,  # type: ignore[arg-type]
                    expand=expand,
                )

        if expand:
            raise NotImplementedError(
                f"Expanding an operator with its function definition "
                f"is only implemented for the main opset. Remove operator "
                f"{node.domain},{node.op_type} from the list of inlined operator."
            )
        if node.domain == "ai.onnx.preview.training":
            from onnx.reference.ops.aionnx_preview_training import load_op as load_op_pt

            return load_op_pt(node.domain, node.op_type, version)

        if node.domain == "experimental":
            from onnx.reference.ops.experimental import load_op as load_op_exp

            return load_op_exp(node.domain, node.op_type, version)

        if node.domain == "ai.onnx.ml":
            from onnx.reference.ops.aionnxml import load_op as load_op_ml

            return load_op_ml(node.domain, node.op_type, version)

        # It has to be a function.
        if key in self.functions_:
            from onnx.reference.ops import load_op

            impl = self.functions_[key]
            return load_op(node.domain, node.op_type, version, custom=impl)
        raise NotImplementedError(
            f"Node type {node.op_type!r} from domain {node.domain!r} "
            f"is unknown, known functions: {sorted(self.functions_)}."
        )

    def run(self, output_names, feed_inputs: Dict[str, Any], attributes: Optional[Dict[str, Any]] = None):  # type: ignore
        """
        Executes the onnx model.

        :param output_names: requested outputs by names,
            None for all
        :param feed_inputs: dictionary `{ input name: input value }`
        :param attributes: attributes value if the instance runs a FunctionProto
        :return: list of requested outputs
        """
        if output_names is None:
            output_names = self.output_names
        if isinstance(self.proto_, FunctionProto) and attributes is None:
            raise TypeError()

        # step 1: inputs and initializers
        results = {"": None}  # optional input
        results.update(self.rt_inits_)  # type: ignore[arg-type]
        results.update(feed_inputs)
        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)  # type: ignore[arg-type]
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)  # type: ignore[arg-type]

        # step 2: execute nodes
        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            inputs = [results[i] for i in node.input]
            linked_attributes = {}
            if node.has_linked_attribute and attributes:
                linked_attributes["linked_attributes"] = attributes
            if node.need_context():
                outputs = node.run(*inputs, context=results, **linked_attributes)
            else:
                outputs = node.run(*inputs, **linked_attributes)
            for name, value in zip(node.output, outputs):
                if isinstance(value, tuple):
                    raise TypeError(
                        f"Unexected type {type(value)} for output {name!r}."
                    )
                self._log(2, " + %s: %s", name, value)  # type: ignore[arg-type]
                results[name] = value

        # return the results
        list_results: List[Any] = []
        for name in output_names:
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output name {name!r} in {sorted(results)}, proto is\n{self.proto_}"
                )
            list_results.append(results[name])
        return list_results
