# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0415,R0912

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from onnx import TensorProto
from onnx.defs import get_all_schemas_with_history, get_schema, onnx_opset_version
from onnx.helper import make_node
from onnx.numpy_helper import to_array
from onnx.onnx_pb import AttributeProto, GraphProto, NodeProto, TypeProto
from onnx.reference.custom_element_types import (
    bfloat16,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)


def _split_class_name(name):  # type: ignore
    if "_" in name:
        prefix, vers = name.rsplit("_", maxsplit=1)
        try:
            v = int(vers)
        except ValueError:
            return name, None
        return prefix, v
    return name, None


class RuntimeTypeError(RuntimeError):
    """
    Raised when a type of a variable is unexpected.
    """


class RuntimeContextError(RuntimeError):
    """
    Raised when the context is missing but an context dependent implementation is defined for an operator.
    """


class RuntimeImplementationError(NotImplementedError):
    """
    Raised when no implementation was found for an operator.
    """


class DefaultNone:
    """
    Default value for parameters when the parameter is not set
    but the operator has a default behavior for it.
    """


class RefAttrName:
    """
    Implements a link between a parameter of a function
    and an attribute in node.

    :param name: name of the input
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}({self.name!r})"


def _build_schemas() -> Dict[str, type]:
    res: Dict[str, type] = {}
    for schema in get_all_schemas_with_history():
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.domain != res[schema.name].domain:  # type: ignore
                raise NotImplementedError(
                    f"This function assumes every operator has a unique name {schema.name!r} "  # type: ignore
                    f"even accross multiple domains {schema.domain!r} and {res[schema.name].domain!r}."  # type: ignore
                )
            if schema.since_version > res[schema.name].since_version:  # type: ignore
                # We keep the most recent one.
                res[schema.name] = schema  # type: ignore
        else:
            res[schema.name] = schema  # type: ignore
        res[schema.name + "_" + str(schema.since_version)] = schema  # type: ignore
    return res


_schemas = _build_schemas()


class OnnxType:
    def __init__(self, type_proto: TypeProto):
        if not isinstance(type_proto, TypeProto):
            raise TypeError(f"type_proto {type(type_proto)} must be of type TypeProto.")
        self.type_proto = type_proto

    def __repr__(self) -> str:
        return f"OnnxType({self.type_proto!r})"


class SparseTensor:
    """
    Simple representation of a sparse tensor.
    It is based on numpy but does not require scipy.
    """

    def __init__(
        self, values: np.ndarray, indices: np.ndarray, shape: Tuple[int]
    ) -> None:
        self.values = values
        self.indices = indices
        self.shape = shape

    @property
    def dtype(self) -> Any:
        return self.values.dtype


def to_sparse_tensor(att: AttributeProto) -> SparseTensor:
    """
    Hosts a sparse tensor.
    """
    shape = tuple(d for d in att.dims)  # type: ignore[attr-defined]
    return SparseTensor(to_array(att.values), to_array(att.indices), shape)  # type: ignore


def to_array_extended(tensor: TensorProto) -> np.ndarray:
    """
    Similar to :func:`to_array` but deals with bfloat16,
    float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz.
    """
    elem_type = tensor.data_type
    if elem_type == TensorProto.BFLOAT16:
        data = tensor.int32_data
        shape = tuple(tensor.dims)
        y = np.empty(shape, dtype=bfloat16).ravel()
        for i, d in enumerate(data):
            y[i] = d
        return y.reshape(shape)

    if elem_type in (
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
    ):
        m = {
            TensorProto.FLOAT8E4M3FN: float8e4m3fn,
            TensorProto.FLOAT8E4M3FNUZ: float8e4m3fnuz,
            TensorProto.FLOAT8E5M2: float8e5m2,
            TensorProto.FLOAT8E5M2FNUZ: float8e5m2fnuz,
        }

        data = tensor.int32_data
        shape = tuple(tensor.dims)
        y = np.empty(shape, dtype=m[elem_type]).ravel()  # type: ignore[index]
        for i, d in enumerate(data):
            y[i] = d
        return y.reshape(shape)
    return to_array(tensor)


class Graph:
    __slots__ = ["g"]

    def __init__(self, g: GraphProto) -> None:
        self.g = g


class OpRun(ABC):
    """
    Ancestor to all operators in this subfolder.

    :param onnx_node: `onnx` node
    :param run_params: additional parameters such as `verbose`, `opsets`
        (it can be more than one if the operator has a subgraph),
        `log` for a logging function
    :param schema: operator schema
    """

    op_domain = ""

    _attribute_conversion_functions = {
        AttributeProto.FLOAT: lambda att: np.float32(att.f),
        AttributeProto.FLOATS: lambda att: [np.float32(f) for f in att.floats],
        AttributeProto.GRAPH: lambda att: Graph(att.g),
        AttributeProto.GRAPHS: lambda att: [Graph(g) for g in att.graphs],
        AttributeProto.INT: lambda att: int(att.i),
        AttributeProto.INTS: lambda att: [int(i) for i in att.ints],
        AttributeProto.SPARSE_TENSOR: lambda att: to_sparse_tensor(att.sparse_tensor),
        AttributeProto.SPARSE_TENSORS: lambda att: [
            to_sparse_tensor(t) for t in att.sparse_tensors
        ],
        AttributeProto.STRING: lambda att: att.s.decode("utf-8"),
        AttributeProto.STRINGS: lambda att: [s.decode("utf-8") for s in att.strings],
        AttributeProto.TENSOR: lambda att: to_array_extended(att.t),
        AttributeProto.TENSORS: lambda att: [to_array_extended(t) for t in att.tensors],
        AttributeProto.TYPE_PROTO: lambda att: OnnxType(att.tp),
        AttributeProto.TYPE_PROTOS: lambda att: [OnnxType(t) for t in att.type_protos],
    }

    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        if not isinstance(run_params, dict):
            raise TypeError(f"run_params must be a dictionary not {type(run_params)}.")
        for att in ["opsets", "new_ops"]:
            if att not in run_params:
                raise RuntimeError(
                    f"Attribute {att!r} must be in run_params, only "
                    f"{list(sorted(run_params))} was found."
                )
        if "log" not in run_params:
            raise KeyError("run_params must contains key 'log'.")
        self.onnx_node = onnx_node
        self.run_params = run_params
        if schema is None:
            if hasattr(self.__class__, "op_schema"):
                self._schema = self.__class__.op_schema
            elif self.__class__.__name__ in _schemas:
                self._schema = _schemas[self.__class__.__name__]
            elif onnx_node.op_type in _schemas:
                self._schema = _schemas[onnx_node.op_type]
            else:
                self._schema = None  # type: ignore
        else:
            self._schema = schema
        self.has_subgraph = False
        self._load_attributes()

    def _log(self, pattern, *args):  # type: ignore
        self.run_params["log"](pattern, *args)

    def _extract_attribute_value(
        self, att: AttributeProto, ref_att: Optional[AttributeProto] = None
    ) -> Any:
        """
        Converts an attribute value into a python value.
        """
        if att.type == AttributeProto.GRAPH:
            from onnx.reference.reference_evaluator import (
                ReferenceEvaluator,  # type: ignore
            )

            new_ops = self.run_params.get("new_ops", None)
            return ReferenceEvaluator(
                att.g,
                opsets=self.run_params["opsets"],
                verbose=max(0, self.run_params.get("verbose", 0) - 2),
                new_ops=None if new_ops is None else new_ops.values(),
            )
        if att.type in OpRun._attribute_conversion_functions:
            return OpRun._attribute_conversion_functions[att.type](att)  # type: ignore
        if ref_att is None:
            raise AttributeError(
                f"Unable to convert attribute {att.name!r} type {att.type!r} "
                f"from node type {self.onnx_node.op_type!r}, "
                f"domain {self.onnx_node.domain!r}\n{att}."
            )
        raise AttributeError(
            f"Unable to convert default value for {ref_att.name!r} type {att.type!r} "
            f"from node type {self.onnx_node.op_type!r}, "
            f"domain {self.onnx_node.domain!r}\n{att}\n{ref_att}."
        )

    @staticmethod
    def _evaluate_subgraph(context, value, attributes):
        return value.run(None, context or {}, attributes=attributes)

    def _load_attributes(self) -> None:
        "Checks and loads attributes."
        self.has_linked_attribute = False
        added_attributes = []
        for att in self.onnx_node.attribute:
            name = att.name
            if att.ref_attr_name:
                value = RefAttrName(att.ref_attr_name)
                self.has_linked_attribute = True
            else:
                value = self._extract_attribute_value(att)
            setattr(self, name, value)
            added_attributes.append(name)
            if att.type == AttributeProto.GRAPH:
                self.has_subgraph = True
                self.has_linked_attribute |= value.has_linked_attribute  # type: ignore
                setattr(
                    self,
                    f"_run_{att.name}",
                    lambda context, value=value, attributes=None: OpRun._evaluate_subgraph(
                        context, value, attributes
                    ),
                )

        if self._schema and self.onnx_node.op_type not in {"Constant"}:
            for k, v in self._schema.attributes.items():  # type: ignore
                if not hasattr(self, k):
                    if getattr(v, "required", True):
                        raise RuntimeError(
                            f"Attribute {k!r} is expected based on ONNX specifications "
                            f"for node {self.onnx_node.op_type!r}."
                        )
                    if hasattr(v, "default_value"):
                        if v.default_value.type == 0:
                            # default value is undefined, it depends on the inputs
                            value = None  # type: ignore
                        else:
                            value = self._extract_attribute_value(v.default_value, v)
                        setattr(self, k, value)
                        added_attributes.append(k)
        self.attributes_names_ = set(added_attributes)

    @staticmethod
    def implicit_inputs(graph: GraphProto) -> List[str]:
        """
        Returns all varibles not registered as inputs and not produced by
        an node inside the graph. This inputs are part of the context
        existing in the graph calling this one.
        """
        if not isinstance(graph, GraphProto):
            raise TypeError(f"Unexpected type {type(graph)!r}.")
        local = set()
        known = set()
        for init in graph.initializer:
            known.add(init.name)
        for sparse_init in graph.sparse_initializer:
            known.add(sparse_init.name)  # type: ignore
        for inp in graph.input:
            known.add(inp.name)
        for node in graph.node:
            for o in node.output:
                known.add(o)
            for i in node.input:
                if i not in known:
                    local.add(i)
        return list(local)

    @property
    def input(self) -> Iterable[str]:
        "Returns node attribute `input`."
        return self.onnx_node.input  # type: ignore

    @property
    def output(self) -> Iterable[str]:
        "Returns node attribute `output`."
        return self.onnx_node.output  # type: ignore

    @property
    def op_type(self) -> str:
        "Returns node attribute `op_type`."
        return self.onnx_node.op_type  # type: ignore

    @property
    def domain(self) -> str:
        "Returns node attribute `domain`."
        return self.onnx_node.domain  # type: ignore

    def need_context(self) -> bool:
        """
        Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Scan, If, Loop).
        The default answer is `False`.
        """
        return False

    def __str__(self) -> str:
        atts = [self.__class__.__name__ + "(", f"    op_type={self.onnx_node.op_type}"]
        for k, v in sorted(self.__dict__.items()):
            if k in {"desc", "onnx_node"}:
                continue
            if "a" <= k[0] <= "z" and k[-1] != "_":
                atts.append(f"    {k}={v},")
        atts.append(")")
        return "\n".join(atts)

    @abstractmethod
    def _run(self, *args, **kwargs):  # type: ignore
        """
        Should be overwritten.

        :param args: operator inputs
        :param kwargs: optional inputs and overriden attributes,
            an attribute may be overridden if it belongs to a function,
            in this case, the same instance of OpRun can be called
            with different values of the same attribute.
        :return: outputs
        """
        raise NotImplementedError(
            f"Method '_run' must be overwritten for operator {self.__class__.__name__!r}."
        )

    def run(self, *args, linked_attributes=None, context=None):  # type: ignore
        """
        Calls method ``_run``, catches exceptions,
        displays a longer error message.

        :param args: inputs
        :param linked_attributes: used if this has an attriute linked
            to the attribute of the function it belongs to
        :param context: if this node is part of the subgraph, `context`
            is a dictionary with the values this node may use
        :return: tuple of results
        """
        if self.need_context():
            if context is None:
                raise RuntimeError(
                    f"This node if type {type(self)} needs context to be filled."
                )
        elif context is not None:
            raise RuntimeError(
                f"This node if type {type(self)} does not need any contextbut one is given."
            )
        if self.has_linked_attribute and linked_attributes is None:
            raise ValueError(
                f"This node {type(self)} has linked attributes but None are given in parameter 'linked_attributes'."
            )
        if not self.has_linked_attribute and linked_attributes is not None:
            raise ValueError(
                f"This node {type(self)} has no linked attribute but some are given in parameter "
                f"'linked_attributes' {set(linked_attributes)}."
            )
        overridden_attributes = {}
        if self.has_linked_attribute:
            if linked_attributes is None:
                raise AttributeError(
                    f"One attribute is linked but no linked value is provided, "
                    f"in class {type(self)}."
                )
            for att in self.attributes_names_:
                v = getattr(self, att)
                if isinstance(v, RefAttrName):
                    if v.name not in linked_attributes:
                        raise ValueError(
                            f"Unable to find a value for linked attribute {att!r} in {linked_attributes!r} "
                            f"in node {type(self)}."
                        )
                    overridden_attributes[att] = linked_attributes[v.name]

        self._log("-- begin %s.run(%d inputs)", self.__class__.__name__, len(args))
        kwargs = {}
        for att in self.attributes_names_:
            if att in overridden_attributes:
                continue
            if not hasattr(self, att):
                raise NameError(
                    f"Attribute {att!r} is missing in operator {self.__class__.__name__!r}."
                )
            kwargs[att] = getattr(self, att)
        if self.has_subgraph:
            if self.has_linked_attribute and len(linked_attributes) == 0:
                raise RuntimeError(
                    f"A subgraph has linked attribute but none was given to {type(self)}."
                )
            kwargs["attributes"] = linked_attributes
        if context is not None:
            kwargs["context"] = context
        try:
            if len(overridden_attributes) > 0:
                res = self._run(*args, **overridden_attributes, **kwargs)
            else:
                res = self._run(*args, **kwargs)
        except (TypeError, AttributeError) as e:
            raise TypeError(
                f"Issues with types {[type(_) for _ in args]} and attributes "
                f"{list(sorted(kwargs))} and linked attributes={list(sorted(overridden_attributes))} "
                f"(operator {self.__class__.__name__!r})."
            ) from e
        self._log("-- done %s.run -> %d outputs", self.__class__.__name__, len(res))
        if not isinstance(res, tuple):
            raise TypeError(
                f"Method '_run' of class {self.__class__.__name__!r} does not return a tuple but {type(res)}."
            )
        if len(res) == 0:
            raise ValueError(
                f"Method '_run' of class {self.__class__.__name__!r} does not return any result."
            )
        if any(map(lambda t: isinstance(t, tuple), res)):
            dtypes = [type(t) for t in res]
            raise TypeError(
                f"One of the results returned by method '_run' of class {self.__class__.__name__!r} "
                f"is a tuple, this is no onnx correponding type (Map, List, Tensor, SparseTensor). "
                f"All returned types: {dtypes!r}."
            )
        return res

    @classmethod
    def make_node(
        cls,
        n_inputs: Optional[int] = None,
        n_outputs: Optional[int] = None,
        **kwargs: Any,
    ) -> NodeProto:  # type: ignore
        """
        Creates an ONNX node for this class based on the given information.

        :param n_inputs: number of inputs (default is defined by the operator schema)
        :param n_outputs: number of outputs (default is defined by the operator schema)
        :param verbose: verbosity
        :param kwargs: node attributes
        :return: NodeProto

        Method :meth:`eval <onnx.reference.op_run.OpRun.eval>` creates an onnx node
        returned by method :meth:`make_node <onnx.reference.op_run.OpRun.make_node>`.

        .. exec_code::

            import numpy as np
            from onnx.reference.ops._op_list import Celu

            onnx_node = Celu.make_node(alpha=0.5)
            print(onnx_node)
        """
        domain = cls.op_domain
        schema = None
        if n_inputs is None:
            if schema is None:
                schema = get_schema(cls.__name__, onnx_opset_version(), domain)
            n_inputs = schema.min_input
        if n_outputs is None:
            if schema is None:
                schema = get_schema(cls.__name__, onnx_opset_version(), domain)
            n_outputs = schema.min_output

        names_in = [f"x{i}" for i in range(n_inputs)]
        names_out = [f"y{i}" for i in range(n_outputs)]
        node = make_node(cls.__name__, names_in, names_out, **kwargs)
        return node

    @classmethod
    def create(
        cls,
        n_inputs: Optional[int] = None,
        n_outputs: Optional[int] = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Instantiates this class based on the given information.

        :param n_inputs: number of inputs (default is defined by the operator schema)
        :param n_outputs: number of outputs (default is defined by the operator schema)
        :param verbose: verbosity
        :param kwargs: node attributes
        :return: NodeProto
        """

        def log_function(pattern: str, *args: Any) -> None:
            if verbose > 1:
                print(pattern % tuple(args))

        node = cls.make_node(n_inputs, n_outputs, **kwargs)
        run_params = dict(
            verbose=verbose,
            log=log_function,
            new_ops=None,
            opsets={"": onnx_opset_version()},
        )
        cl = cls(node, run_params)
        return cl

    @classmethod
    def eval(
        cls,
        *args: List[Any],
        n_outputs: Optional[int] = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Any:  # type: ignore
        """
        Evaluates this operator.

        :param args: inputs
        :param n_outputs: number of outputs (default is defined by the operator schema)
        :param verbose: verbosity
        :param kwargs: node attributes
        :return: NodeProto
        """
        inst = cls.create(len(args), n_outputs=n_outputs, verbose=verbose, **kwargs)
        res = inst.run(*args)
        if len(res) == 1:
            return res[0]
        return res


class OpFunction(OpRun):
    """
    Runs a custom function.
    """

    def __init__(self, onnx_node: NodeProto, log_function: Any, impl: Any = None):
        if impl is None:
            raise RuntimeError(
                f"impl cannot be None for node type {onnx_node.op_type!r} "
                f"from domain {onnx_node.domain!r}."
            )
        OpRun.__init__(self, onnx_node, log_function)
        self.impl_ = impl
        # The function implementation is the same whenever the function is called
        # but the attributes may be different at every call.
        self.attributes_ = {
            name: getattr(self, name) for name in self.impl_.attributes_
        }

    def _run(self, *inputs, **kwargs):  # type: ignore # pylint: disable=W0221
        if len(self.impl_.input_names) != len(inputs):
            raise RuntimeError(
                f"Mismatch lengths between the number of inputs {len(inputs)} "
                f"and the expected number of inputs {len(self.impl_.inputs)} "
                f"for node {self.op_type!r} from domain {self.domain!r}."
            )
        feeds = dict(zip(self.impl_.input_names, inputs))
        attributes = self.attributes_.copy()
        attributes.update(kwargs)
        results = self.impl_.run(None, feeds, attributes=attributes)
        if len(self.impl_.output_names) != len(results):
            raise RuntimeError(
                f"Mismatch lengths between the number of outputs {len(results)} "
                f"and the expected number of outputs {len(self.impl_.output_names)} "
                f"for node {self.op_type!r} from domain {self.domain!r}."
            )
        return tuple(results)
