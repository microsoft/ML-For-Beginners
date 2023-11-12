# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0302,R0912
import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
    Any,
    Callable,
    Dict,
    KeysView,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import google.protobuf.message
import numpy as np

from onnx import (
    IR_VERSION,
    AttributeProto,
    FunctionProto,
    GraphProto,
    MapProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    OptionalProto,
    SequenceProto,
    SparseTensorProto,
    TensorProto,
    TensorShapeProto,
    TrainingInfoProto,
    TypeProto,
    ValueInfoProto,
    defs,
    mapping,
)

VersionRowType = Union[Tuple[str, int, int, int], Tuple[str, int, int, int, int]]
VersionTableType = List[VersionRowType]
AssignmentBindingType = List[Tuple[str, str]]

# This is a copy of the documented version in https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
# Both must be updated whenever a new version of ONNX is released.
VERSION_TABLE: VersionTableType = [
    # Release-version, IR version, ai.onnx version, ai.onnx.ml version, (optional) ai.onnx.training version
    ("1.0", 3, 1, 1),
    ("1.1", 3, 5, 1),
    ("1.1.2", 3, 6, 1),
    ("1.2", 3, 7, 1),
    ("1.3", 3, 8, 1),
    ("1.4.1", 4, 9, 1),
    ("1.5.0", 5, 10, 1),
    ("1.6.0", 6, 11, 2),
    ("1.7.0", 7, 12, 2, 1),
    ("1.8.0", 7, 13, 2, 1),
    ("1.8.1", 7, 13, 2, 1),
    ("1.9.0", 7, 14, 2, 1),
    ("1.10.0", 8, 15, 2, 1),
    ("1.10.1", 8, 15, 2, 1),
    ("1.10.2", 8, 15, 2, 1),
    ("1.11.0", 8, 16, 3, 1),
    ("1.12.0", 8, 17, 3, 1),
    ("1.13.0", 8, 18, 3, 1),
    ("1.13.1", 8, 18, 3, 1),
    ("1.14.0", 9, 19, 3, 1),
    ("1.14.1", 9, 19, 3, 1),
]

VersionMapType = Dict[Tuple[str, int], int]


def create_op_set_id_version_map(table: VersionTableType) -> VersionMapType:
    """create a map from (opset-domain, opset-version) to ir-version from above table"""
    result: VersionMapType = {}

    def process(release_version: str, ir_version: int, *args: Any) -> None:
        del release_version  # Unused
        for pair in zip(["ai.onnx", "ai.onnx.ml", "ai.onnx.training"], args):
            if pair not in result:
                result[pair] = ir_version
                if pair[0] == "ai.onnx.training":
                    result["ai.onnx.preview.training", pair[1]] = ir_version

    for row in table:
        process(*row)
    return result


OP_SET_ID_VERSION_MAP = create_op_set_id_version_map(VERSION_TABLE)


def find_min_ir_version_for(
    opsetidlist: List[OperatorSetIdProto], ignore_unknown: bool = False
) -> int:
    """Given list of opset ids, determine minimum IR version required.

    Arguments:
        opsetidlist (List[OperatorSetIdProto]): The list of OperatorSetIdProto
        ignore_unknown (bool): If True, ignore unknown domain and return default min version for that domain.
    Returns:
        The minimum IR version required (integer)
    """
    default_min_version = 3

    def find_min(domain: Union[str, None], version: int) -> int:
        key = (domain if domain else "ai.onnx", version)
        if key in OP_SET_ID_VERSION_MAP:
            return OP_SET_ID_VERSION_MAP[key]
        if ignore_unknown:
            return default_min_version
        raise ValueError("Unsupported opset-version.")

    if opsetidlist:
        return max(find_min(x.domain, x.version) for x in opsetidlist)
    return default_min_version  # if no opsets specified


def make_node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    domain: Optional[str] = None,
    **kwargs: Any,
) -> NodeProto:
    """Construct a NodeProto.

    Arguments:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for NodeProto
        doc_string (string, default None): optional documentation string for NodeProto
        domain (string, default None): optional domain for NodeProto.
            If it's None, we will just use default domain (which is empty)
        **kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.
    Returns:
        NodeProto
    """

    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if kwargs:
        node.attribute.extend(
            make_attribute(key, value)
            for key, value in sorted(kwargs.items())
            if value is not None
        )
    return node


def make_operatorsetid(
    domain: str,
    version: int,
) -> OperatorSetIdProto:
    """Construct an OperatorSetIdProto.

    Arguments:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    Returns:
        OperatorSetIdProto
    """
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = domain
    operatorsetid.version = version
    return operatorsetid


def make_graph(
    nodes: Sequence[NodeProto],
    name: str,
    inputs: Sequence[ValueInfoProto],
    outputs: Sequence[ValueInfoProto],
    initializer: Optional[Sequence[TensorProto]] = None,
    doc_string: Optional[str] = None,
    value_info: Optional[Sequence[ValueInfoProto]] = None,
    sparse_initializer: Optional[Sequence[SparseTensorProto]] = None,
) -> GraphProto:
    """Construct a GraphProto

    Arguments:
        nodes: list of NodeProto
        name (string): graph name
        inputs: list of ValueInfoProto
        outputs: list of ValueInfoProto
        initializer: list of TensorProto
        doc_string (string): graph documentation
        value_info: list of ValueInfoProto
        sparse_initializer: list of SparseTensorProto
    Returns:
        GraphProto
    """
    if initializer is None:
        initializer = []
    if sparse_initializer is None:
        sparse_initializer = []
    if value_info is None:
        value_info = []
    graph = GraphProto()
    graph.node.extend(nodes)
    graph.name = name
    graph.input.extend(inputs)
    graph.output.extend(outputs)
    graph.initializer.extend(initializer)
    graph.sparse_initializer.extend(sparse_initializer)
    graph.value_info.extend(value_info)
    if doc_string:
        graph.doc_string = doc_string
    return graph


def make_opsetid(domain: str, version: int) -> OperatorSetIdProto:
    """Construct an OperatorSetIdProto.

    Arguments:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    Returns:
        OperatorSetIdProto
    """
    opsetid = OperatorSetIdProto()
    opsetid.domain = domain
    opsetid.version = version
    return opsetid


def make_function(
    domain: str,
    fname: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    nodes: Sequence[NodeProto],
    opset_imports: Sequence[OperatorSetIdProto],
    attributes: Optional[Sequence[str]] = None,
    attribute_protos: Optional[Sequence[AttributeProto]] = None,
    doc_string: Optional[str] = None,
) -> FunctionProto:
    if attributes is None:
        attributes = []
    if attribute_protos is None:
        attribute_protos = []
    f = FunctionProto()
    f.domain = domain
    f.name = fname
    f.input.extend(inputs)
    f.output.extend(outputs)
    f.node.extend(nodes)
    f.opset_import.extend(opset_imports)
    f.attribute.extend(attributes)
    f.attribute_proto.extend(attribute_protos)
    if doc_string:
        f.doc_string = doc_string
    return f


def make_model(graph: GraphProto, **kwargs: Any) -> ModelProto:
    """Construct a ModelProto

    Arguments:
        graph (GraphProto): *make_graph* returns
        **kwargs: any attribute to add to the returned instance
    Returns:
        ModelProto
    """
    model = ModelProto()
    # Touch model.ir_version so it is stored as the version from which it is
    # generated.
    model.ir_version = IR_VERSION
    model.graph.CopyFrom(graph)

    opset_imports: Optional[Sequence[OperatorSetIdProto]] = None
    opset_imports = kwargs.pop("opset_imports", None)  # type: ignore
    if opset_imports is not None:
        model.opset_import.extend(opset_imports)
    else:
        # Default import
        imp = model.opset_import.add()
        imp.version = defs.onnx_opset_version()

    functions: Optional[Sequence[FunctionProto]] = None
    functions = kwargs.pop("functions", None)  # type: ignore
    if functions is not None:
        model.functions.extend(functions)

    for k, v in kwargs.items():
        # TODO: Does this work with repeated fields?
        setattr(model, k, v)
    return model


# An extension of make_model that infers an IR_VERSION for the model,
# if not specified, using a best-effort-basis.
def make_model_gen_version(graph: GraphProto, **kwargs: Any) -> ModelProto:
    ir_version_field = "ir_version"
    if ir_version_field not in kwargs:
        opset_imports_field = "opset_imports"
        imports = kwargs[opset_imports_field] if opset_imports_field in kwargs else []
        kwargs[ir_version_field] = find_min_ir_version_for(imports)
    return make_model(graph, **kwargs)


def set_model_props(model: ModelProto, dict_value: Dict[str, str]) -> None:
    del model.metadata_props[:]
    for k, v in dict_value.items():
        entry = model.metadata_props.add()
        entry.key = k
        entry.value = v
        # model.metadata_properties.append(entry)


def split_complex_to_pairs(ca: Sequence[np.complex64]) -> Sequence[int]:
    return [
        (ca[i // 2].real if (i % 2 == 0) else ca[i // 2].imag)  # type: ignore[misc]
        for i in range(len(ca) * 2)
    ]


# convert a float32 value to a bfloat16 (as int)
# By default, this conversion rounds-to-nearest-even and supports NaN
# Setting `truncate` to True enables a simpler conversion. In this mode the
# conversion is performed by simply dropping the 2 least significant bytes of
# the significand. In this mode an error of up to 1 bit may be introduced and
# preservation of NaN values is not be guaranteed.
def float32_to_bfloat16(fval: float, truncate: bool = False) -> int:
    ival = int.from_bytes(struct.pack("<f", fval), "little")
    if truncate:
        return ival >> 16
    # NaN requires at least 1 significand bit set
    if isnan(fval):
        return 0x7FC0  # sign=0, exp=all-ones, sig=0b1000000
    # drop bottom 16-bits
    # round remaining bits using round-to-nearest-even
    rounded = ((ival >> 16) & 1) + 0x7FFF
    return (ival + rounded) >> 16


def float32_to_float8e4m3(  # pylint: disable=too-many-statements
    fval: float,
    scale: float = 1.0,
    fn: bool = True,
    uz: bool = False,
    saturate: bool = True,
) -> int:
    """
    Convert a float32 value to a float8, e4m3 (as int).

    :param fval: float to convert
    :param scale: scale, divide *fval* by *scale* before casting it
    :param fn: no infinite values
    :param uz: no negative zero
    :param saturate: if True, any value out of range included inf becomes the maximum value,
        otherwise, it becomes NaN. The description of operator Cast fully describes the
        differences.
    :return: converted float

    See :ref:`onnx-detail-float8` for technical details.
    """
    if not fn:
        raise NotImplementedError(
            "float32_to_float8e4m3 not implemented with fn=False."
        )
    x = fval / scale
    b = int.from_bytes(struct.pack("<f", np.float32(x)), "little")
    ret = (b & 0x80000000) >> 24  # sign
    if uz:
        if (b & 0x7FC00000) == 0x7FC00000:
            return 0x80
        if np.isinf(x):
            if saturate:
                return ret | 0x7F
            return 0x80
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 116:
                pass
            elif e < 117:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    ret += 1
            elif e < 120:  # 127 - 8 + 1
                d = 119 - e
                ret |= 1 << (2 - d)
                ret |= m >> (21 + d)
                if (m >> (20 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 135:  # 127 + 8
                ex = e - 119  # 127 - 8
                if ex == 0:
                    ret |= 0x4
                    ret |= m >> 21
                else:
                    ret |= ex << 3
                    ret |= m >> 20
                if m & 0x80000:
                    if (ret & 0x7F) < 0x7F:
                        # rounding
                        ret += 1
                    elif not saturate:
                        return 0x80
            elif saturate:
                ret |= 0x7F  # 01111110
            else:
                ret = 0x80
        elif m == 0:
            # -0
            ret = 0
        return int(ret)
    else:
        if (b & 0x7FC00000) == 0x7FC00000:
            return 0x7F | ret
        if np.isinf(x):
            if saturate:
                return ret | 126
            return 0x7F | ret
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 117:
                pass
            elif e < 118:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    ret += 1
            elif e < 121:  # 127 - 7 + 1
                d = 120 - e
                ret |= 1 << (2 - d)
                ret |= m >> (21 + d)
                if (m >> (20 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 136:  # 127 + 8 + 1
                ex = e - 120  # 127 - 7
                if ex == 0:
                    ret |= 0x4
                    ret |= m >> 21
                else:
                    ret |= ex << 3
                    ret |= m >> 20
                    if (ret & 0x7F) == 0x7F:
                        ret &= 0xFE
                if (m & 0x80000) and (
                    (m & 0x100000) or (m & 0x7C000)
                ):  # round to nearest even
                    if (ret & 0x7F) < 0x7E:
                        # rounding
                        ret += 1
                    elif not saturate:
                        ret |= 0x7F
            elif saturate:
                ret |= 126  # 01111110
            else:
                ret |= 0x7F
        return int(ret)


def float32_to_float8e5m2(  # pylint: disable=too-many-statements
    fval: float,
    scale: float = 1.0,
    fn: bool = False,
    uz: bool = False,
    saturate: bool = True,
) -> int:
    """
    Convert a float32 value to a float8, e5m2 (as int).

    :param fval: float to convert
    :param scale: scale, divide *fval* by *scale* before casting it
    :param fn: no infinite values
    :param uz: no negative zero
    :param saturate: if True, any value out of range included inf becomes the maximum value,
        otherwise, it becomes NaN. The description of operator Cast fully describes the
        differences.
    :return: converted float
    """
    x = fval / scale
    b = int.from_bytes(struct.pack("<f", np.float32(x)), "little")
    ret = (b & 0x80000000) >> 24  # sign

    if fn and uz:
        if (b & 0x7FC00000) == 0x7FC00000:
            return 0x80
        if (b & 0x7FFFFFFF) == 0x7F800000:
            # inf
            if saturate:
                return ret | 0x7F
            return 0x80
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 109:
                pass
            elif e < 110:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    # may be unused
                    ret += 1
            elif e < 112:  # 127 - 16 + 1
                d = 111 - e
                ret |= 1 << (1 - d)
                ret |= m >> (22 + d)
                if (m >> (21 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 143:  # 127 + 15 + 1
                ex = e - 111  # 127 - 16
                ret |= ex << 2
                ret |= m >> 21
                if m & 0x100000:
                    if (ret & 0x7F) < 0x7F:
                        # rounding
                        ret += 1
                    elif not saturate:
                        ret = 0x80
            elif e == 255 and m == 0:  # inf
                ret = 0x80
            elif saturate:
                ret |= 0x7F  # last possible number
            else:
                ret = 0x80
        elif m == 0:
            # -0
            ret = 0
        return int(ret)
    elif not fn and not uz:
        if (b & 0x7FC00000) == 0x7FC00000:
            return 0x7F | ret
        if np.isinf(x):
            if saturate:
                return 0x7B | ret
            return 0x7C | ret
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 110:
                pass
            elif e < 111:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    # may be unused
                    ret += 1
            elif e < 113:  # 127 - 15 + 1
                d = 112 - e
                ret |= 1 << (1 - d)
                ret |= m >> (22 + d)
                if (m >> (21 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 143:  # 127 + 15 + 1
                ex = e - 112  # 127 - 15
                ret |= ex << 2
                ret |= m >> 21
                if (m & 0x100000) and (
                    (m & 0xFFFFF) or (m & 0x200000)
                ):  # round to nearest even
                    if (ret & 0x7F) < 0x7B:
                        # rounding
                        ret += 1
                    elif saturate:
                        ret |= 0x7B
                    else:
                        ret |= 0x7C
            elif saturate:
                ret |= 0x7B
            else:
                ret |= 0x7C
        return int(ret)
    else:
        raise NotImplementedError("fn and uz must be both False or True.")


def make_tensor(
    name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = False
) -> TensorProto:
    """
    Make a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.

    Arguments:
        name (string): tensor name
        data_type (int): a value such as onnx.TensorProto.FLOAT
        dims (List[int]): shape
        vals: values
        raw (bool): if True, vals contains the serialized content of the tensor,
            otherwise, vals should be a list of values of the type defined by *data_type*

    Returns:
        TensorProto
    """
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name

    if data_type == TensorProto.STRING and raw:
        raise TypeError("Can not use raw_data to store string type.")

    np_dtype = tensor_dtype_to_np_dtype(data_type)

    # Check number of vals specified equals tensor size
    expected_size = 1
    if raw:
        # NumPy doesn't have BFLOAT16. TENSOR_TYPE_TO_NP_TYPE maps it to float32,
        # which has the wrong itemsize.
        if data_type == TensorProto.BFLOAT16:
            expected_size = 2
        elif data_type in (
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        ):
            expected_size = 1
        else:
            expected_size = np_dtype.itemsize

    if (
        type(vals) is np.ndarray  # pylint: disable=unidiomatic-typecheck
        and len(vals.shape) > 1
    ):
        vals = vals.flatten()
    for d in dims:
        expected_size *= d

    if len(vals) != expected_size:
        raise ValueError(
            f"Number of values does not match tensor's size. Expected {expected_size}, but it is {len(vals)}. "
        )

    if raw:
        tensor.raw_data = vals
    else:
        if data_type in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
            vals = split_complex_to_pairs(vals)
        elif data_type == TensorProto.FLOAT16:
            vals = (
                np.array(vals).astype(np_dtype).view(dtype=np.uint16).flatten().tolist()
            )
        elif data_type in (
            TensorProto.BFLOAT16,
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        ):
            fcast = {
                TensorProto.BFLOAT16: float32_to_bfloat16,
                TensorProto.FLOAT8E4M3FN: float32_to_float8e4m3,
                TensorProto.FLOAT8E4M3FNUZ: lambda *args: float32_to_float8e4m3(  # type: ignore[misc]
                    *args, uz=True
                ),
                TensorProto.FLOAT8E5M2: float32_to_float8e5m2,
                TensorProto.FLOAT8E5M2FNUZ: lambda *args: float32_to_float8e5m2(  # type: ignore[misc]
                    *args, fn=True, uz=True
                ),
            }[
                data_type  # type: ignore[index]
            ]
            vals = list(
                map(  # type: ignore[call-overload]
                    fcast,
                    np.array(vals).astype(np_dtype).flatten().tolist(),
                )
            )
        elif data_type == TensorProto.BOOL:
            vals = np.array(vals).astype(int)
        field = tensor_dtype_to_field(data_type)
        getattr(tensor, field).extend(vals)
    tensor.dims.extend(dims)
    return tensor


def make_sparse_tensor(
    values: TensorProto, indices: TensorProto, dims: Sequence[int]
) -> SparseTensorProto:
    """Construct a SparseTensorProto

    Arguments:
        values (TensorProto): the values
        indices (TensorProto): the indices
        dims: the shape

    Returns:
        SparseTensorProto
    """
    sparse = SparseTensorProto()
    sparse.values.CopyFrom(values)
    sparse.indices.CopyFrom(indices)
    sparse.dims.extend(dims)
    return sparse


def make_sequence(
    name: str,
    elem_type: SequenceProto.DataType,
    values: Sequence[Any],
) -> SequenceProto:
    """
    Make a Sequence with specified value arguments.
    """
    sequence = SequenceProto()
    sequence.name = name
    sequence.elem_type = elem_type

    if elem_type == SequenceProto.UNDEFINED:
        return sequence
    if elem_type == SequenceProto.TENSOR:
        attribute = sequence.tensor_values
    elif elem_type == SequenceProto.SPARSE_TENSOR:
        attribute = sequence.sparse_tensor_values  # type: ignore[assignment]
    elif elem_type == SequenceProto.SEQUENCE:
        attribute = sequence.sequence_values  # type: ignore[assignment]
    elif elem_type == SequenceProto.MAP:
        attribute = sequence.map_values  # type: ignore[assignment]
    elif elem_type == OptionalProto.OPTIONAL:
        attribute = sequence.optional_values  # type: ignore[assignment]
    else:
        raise TypeError("The element type in the input sequence is not supported.")

    attribute.extend(values)
    return sequence


def make_map(
    name: str, key_type: int, keys: List[Any], values: SequenceProto
) -> MapProto:
    """
    Make a Map with specified key-value pair arguments.

    Criteria for conversion:
    - Keys and Values must have the same number of elements
    - Every key in keys must be of the same type
    - Every value in values must be of the same type
    """
    map_proto = MapProto()
    valid_key_int_types = [
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    ]
    map_proto.name = name
    map_proto.key_type = key_type
    if key_type == TensorProto.STRING:
        map_proto.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map_proto.keys.extend(keys)
    map_proto.values.CopyFrom(values)
    return map_proto


def make_optional(
    name: str,
    elem_type: OptionalProto.DataType,
    value: Optional[Any],
) -> OptionalProto:
    """
    Make an Optional with specified value arguments.
    """
    optional = OptionalProto()
    optional.name = name
    optional.elem_type = elem_type

    if elem_type == OptionalProto.UNDEFINED:
        return optional
    if elem_type == OptionalProto.TENSOR:
        attribute = optional.tensor_value
    elif elem_type == OptionalProto.SPARSE_TENSOR:
        attribute = optional.sparse_tensor_value  # type: ignore[assignment]
    elif elem_type == OptionalProto.SEQUENCE:
        attribute = optional.sequence_value  # type: ignore[assignment]
    elif elem_type == OptionalProto.MAP:
        attribute = optional.map_value  # type: ignore[assignment]
    elif elem_type == OptionalProto.OPTIONAL:
        attribute = optional.optional_value  # type: ignore[assignment]
    else:
        raise TypeError("The element type in the input optional is not supported.")

    attribute.CopyFrom(value)  # type: ignore[arg-type]
    return optional


def _to_bytes(value: Union[str, bytes]) -> bytes:
    """Coerce a string (or bytes) value into UTF-8 bytes."""
    return value if isinstance(value, bytes) else value.encode("utf-8")


def make_attribute(  # pylint: disable=too-many-statements
    key: str, value: Any, doc_string: Optional[str] = None
) -> AttributeProto:
    """Makes an AttributeProto based on the value type."""
    attr = AttributeProto()
    attr.name = key
    if doc_string:
        attr.doc_string = doc_string

    # Singular cases
    if isinstance(value, numbers.Integral):
        attr.i = int(value)
        attr.type = AttributeProto.INT
    elif isinstance(value, numbers.Real):
        attr.f = float(value)
        attr.type = AttributeProto.FLOAT
    elif isinstance(value, (str, bytes)):
        # Encode strings into utf-8
        attr.s = _to_bytes(value)
        attr.type = AttributeProto.STRING
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
        attr.type = AttributeProto.TENSOR
    elif isinstance(value, SparseTensorProto):
        attr.sparse_tensor.CopyFrom(value)
        attr.type = AttributeProto.SPARSE_TENSOR
    elif isinstance(value, GraphProto):
        attr.g.CopyFrom(value)
        attr.type = AttributeProto.GRAPH
    elif isinstance(value, TypeProto):
        attr.tp.CopyFrom(value)
        attr.type = AttributeProto.TYPE_PROTO
    # Iterable cases
    elif isinstance(value, collections.abc.Iterable):
        value = list(value)
        types = {type(v) for v in value}
        if all(issubclass(t, numbers.Integral) for t in types):
            attr.ints.extend(value)
            attr.type = AttributeProto.INTS
        elif all(issubclass(t, numbers.Real) for t in types):
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
        elif all(issubclass(t, (str, bytes)) for t in types):
            attr.strings.extend(_to_bytes(v) for v in value)
            attr.type = AttributeProto.STRINGS
        elif all(issubclass(t, TensorProto) for t in types):
            attr.tensors.extend(value)
            attr.type = AttributeProto.TENSORS
        elif all(issubclass(t, SparseTensorProto) for t in types):
            attr.sparse_tensors.extend(value)
            attr.type = AttributeProto.SPARSE_TENSORS
        elif all(issubclass(t, GraphProto) for t in types):
            attr.graphs.extend(value)
            attr.type = AttributeProto.GRAPHS
        elif all(issubclass(t, TypeProto) for t in types):
            attr.type_protos.extend(value)
            attr.type = AttributeProto.TYPE_PROTOS
        else:
            raise ValueError(
                "Could not infer the attribute type from the elements of the passed Iterable value."
            )
    else:
        raise TypeError(f"'{value}' is not an accepted attribute value.")
    return attr


def make_attribute_ref(
    name: str, attr_type: AttributeProto.AttributeType, doc_string: Optional[str] = None
) -> AttributeProto:
    """Make an AttributeProto holding a reference to the parent function's attribute of given name and type."""
    attr = AttributeProto()
    attr.name = name
    attr.type = attr_type
    if doc_string:
        attr.doc_string = doc_string
    return attr


def get_attribute_value(attr: AttributeProto) -> Any:
    if attr.ref_attr_name:
        raise ValueError(f"Cannot get value of reference attribute: {attr}")
    if attr.type == AttributeProto.FLOAT:
        return attr.f
    if attr.type == AttributeProto.INT:
        return attr.i
    if attr.type == AttributeProto.STRING:
        return attr.s
    if attr.type == AttributeProto.TENSOR:
        return attr.t
    if attr.type == AttributeProto.SPARSE_TENSOR:
        return attr.sparse_tensor
    if attr.type == AttributeProto.GRAPH:
        return attr.g
    if attr.type == AttributeProto.TYPE_PROTO:
        return attr.tp
    if attr.type == AttributeProto.FLOATS:
        return list(attr.floats)
    if attr.type == AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == AttributeProto.STRINGS:
        return list(attr.strings)
    if attr.type == AttributeProto.TENSORS:
        return list(attr.tensors)
    if attr.type == AttributeProto.SPARSE_TENSORS:
        return list(attr.sparse_tensors)
    if attr.type == AttributeProto.GRAPHS:
        return list(attr.graphs)
    if attr.type == AttributeProto.TYPE_PROTOS:
        return list(attr.type_protos)
    raise ValueError(f"Unsupported ONNX attribute: {attr}")


def get_node_attr_value(node: NodeProto, attr_name: str) -> Any:
    matching = [x for x in node.attribute if x.name == attr_name]
    if len(matching) > 1:
        raise ValueError(f"Node has multiple attributes with name {attr_name}")
    if len(matching) < 1:
        raise ValueError(f"Node has no attribute with name {attr_name}")
    return get_attribute_value(matching[0])


def make_empty_tensor_value_info(name: str) -> ValueInfoProto:
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    return value_info_proto


def make_tensor_type_proto(
    elem_type: int,
    shape: Optional[Sequence[Union[str, int, None]]],
    shape_denotation: Optional[List[str]] = None,
) -> TypeProto:
    """Makes a Tensor TypeProto based on the data type and shape."""

    type_proto = TypeProto()
    tensor_type_proto = type_proto.tensor_type
    tensor_type_proto.elem_type = elem_type
    tensor_shape_proto = tensor_type_proto.shape

    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        tensor_shape_proto.dim.extend([])

        if shape_denotation and len(shape_denotation) != len(shape):
            raise ValueError(
                "Invalid shape_denotation. Must be of the same length as shape."
            )

        for i, d in enumerate(shape):
            dim = tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(
                    f"Invalid item in shape: {d}. Needs to be of int or str."
                )

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return type_proto


def make_tensor_value_info(
    name: str,
    elem_type: int,
    shape: Optional[Sequence[Union[str, int, None]]],
    doc_string: str = "",
    shape_denotation: Optional[List[str]] = None,
) -> ValueInfoProto:
    """Makes a ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = make_tensor_type_proto(elem_type, shape, shape_denotation)
    value_info_proto.type.CopyFrom(tensor_type_proto)
    return value_info_proto


def make_sparse_tensor_type_proto(
    elem_type: int,
    shape: Optional[Sequence[Union[str, int, None]]],
    shape_denotation: Optional[List[str]] = None,
) -> TypeProto:
    """Makes a SparseTensor TypeProto based on the data type and shape."""

    type_proto = TypeProto()
    sparse_tensor_type_proto = type_proto.sparse_tensor_type
    sparse_tensor_type_proto.elem_type = elem_type
    sparse_tensor_shape_proto = sparse_tensor_type_proto.shape

    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        sparse_tensor_shape_proto.dim.extend([])

        if shape_denotation and len(shape_denotation) != len(shape):
            raise ValueError(
                "Invalid shape_denotation. Must be of the same length as shape."
            )

        for i, d in enumerate(shape):
            dim = sparse_tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(
                    f"Invalid item in shape: {d}. Needs to be of int or text."
                )

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return type_proto


def make_sparse_tensor_value_info(
    name: str,
    elem_type: int,
    shape: Optional[Sequence[Union[str, int, None]]],
    doc_string: str = "",
    shape_denotation: Optional[List[str]] = None,
) -> ValueInfoProto:
    """Makes a SparseTensor ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    sparse_tensor_type_proto = make_sparse_tensor_type_proto(
        elem_type, shape, shape_denotation
    )
    value_info_proto.type.sparse_tensor_type.CopyFrom(
        sparse_tensor_type_proto.sparse_tensor_type
    )
    return value_info_proto


def make_sequence_type_proto(
    inner_type_proto: TypeProto,
) -> TypeProto:
    """Makes a sequence TypeProto."""
    type_proto = TypeProto()
    type_proto.sequence_type.elem_type.CopyFrom(inner_type_proto)
    return type_proto


def make_optional_type_proto(
    inner_type_proto: TypeProto,
) -> TypeProto:
    """Makes an optional TypeProto."""
    type_proto = TypeProto()
    type_proto.optional_type.elem_type.CopyFrom(inner_type_proto)
    return type_proto


def make_value_info(
    name: str,
    type_proto: TypeProto,
    doc_string: str = "",
) -> ValueInfoProto:
    """Makes a ValueInfoProto with the given type_proto."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    value_info_proto.type.CopyFrom(type_proto)
    return value_info_proto


def _sanitize_str(s: Union[str, bytes]) -> str:
    if isinstance(s, str):
        sanitized = s
    elif isinstance(s, bytes):
        sanitized = s.decode("utf-8", errors="ignore")
    else:
        sanitized = str(s)
    if len(sanitized) < 64:
        return sanitized
    return sanitized[:64] + f"...<+len={(len(sanitized) - 64)}>"


def make_tensor_sequence_value_info(
    name: str,
    elem_type: int,
    shape: Optional[Sequence[Union[str, int, None]]],
    doc_string: str = "",
    elem_shape_denotation: Optional[List[str]] = None,
) -> ValueInfoProto:
    """Makes a Sequence[Tensors] ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = make_tensor_type_proto(elem_type, shape, elem_shape_denotation)
    sequence_type_proto = make_sequence_type_proto(tensor_type_proto)
    value_info_proto.type.sequence_type.CopyFrom(sequence_type_proto.sequence_type)

    return value_info_proto


def printable_attribute(
    attr: AttributeProto, subgraphs: bool = False
) -> Union[str, Tuple[str, List[GraphProto]]]:
    content = []
    content.append(attr.name)
    content.append("=")

    def str_float(f: float) -> str:
        # NB: Different Python versions print different numbers of trailing
        # decimals, specifying this explicitly keeps it consistent for all
        # versions
        return f"{f:.15g}"

    def str_int(i: int) -> str:
        return str(i)

    _T = TypeVar("_T")

    def str_list(str_elem: Callable[[_T], str], xs: Sequence[_T]) -> str:
        return "[" + ", ".join(map(str_elem, xs)) + "]"

    # for now, this logic should continue to work as long as we are running on a proto3
    # implementation. If/when we switch to proto3, we will need to use attr.type

    # To support printing subgraphs, if we find a graph attribute, print out
    # its name here and pass the graph itself up to the caller for later
    # printing.
    graphs = []
    if attr.HasField("f"):
        content.append(str_float(attr.f))
    elif attr.HasField("i"):
        content.append(str_int(attr.i))
    elif attr.HasField("s"):
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        content.append(repr(_sanitize_str(attr.s)))
    elif attr.HasField("t"):
        if len(attr.t.dims) > 0:
            content.append("<Tensor>")
        else:
            # special case to print scalars
            field = tensor_dtype_to_field(attr.t.data_type)
            content.append(f"<Scalar Tensor {str(getattr(attr.t, field))}>")
    elif attr.HasField("g"):
        content.append(f"<graph {attr.g.name}>")
        graphs.append(attr.g)
    elif attr.HasField("tp"):
        content.append(f"<Type Proto {attr.tp}>")
    elif attr.floats:
        content.append(str_list(str_float, attr.floats))
    elif attr.ints:
        content.append(str_list(str_int, attr.ints))
    elif attr.strings:
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        content.append(str(list(map(_sanitize_str, attr.strings))))
    elif attr.tensors:
        content.append("[<Tensor>, ...]")
    elif attr.type_protos:
        content.append("[")
        for i, tp in enumerate(attr.type_protos):
            comma = "," if i != len(attr.type_protos) - 1 else ""
            content.append(f"<Type Proto {tp}>{comma}")
        content.append("]")
    elif attr.graphs:
        content.append("[")
        for i, g in enumerate(attr.graphs):
            comma = "," if i != len(attr.graphs) - 1 else ""
            content.append(f"<graph {g.name}>{comma}")
        content.append("]")
        graphs.extend(attr.graphs)
    else:
        content.append("<Unknown>")
    if subgraphs:
        return " ".join(content), graphs
    return " ".join(content)


def printable_dim(dim: TensorShapeProto.Dimension) -> str:
    which = dim.WhichOneof("value")
    if which is None:
        raise TypeError(f"which cannot be {None}.")
    return str(getattr(dim, which))


def printable_type(t: TypeProto) -> str:
    if t.WhichOneof("value") == "tensor_type":
        s = TensorProto.DataType.Name(t.tensor_type.elem_type)
        if t.tensor_type.HasField("shape"):
            if len(t.tensor_type.shape.dim):
                s += str(", " + "x".join(map(printable_dim, t.tensor_type.shape.dim)))
            else:
                s += ", scalar"
        return s
    if t.WhichOneof("value") is None:
        return ""
    return f"Unknown type {t.WhichOneof('value')}"


def printable_value_info(v: ValueInfoProto) -> str:
    s = f"%{v.name}"
    if v.type:
        s = f"{s}[{printable_type(v.type)}]"
    return s


def printable_tensor_proto(t: TensorProto) -> str:
    s = f"%{t.name}["
    s += TensorProto.DataType.Name(t.data_type)
    if t.dims is not None:
        if len(t.dims):
            s += str(", " + "x".join(map(str, t.dims)))
        else:
            s += ", scalar"
    s += "]"
    return s


def printable_node(
    node: NodeProto, prefix: str = "", subgraphs: bool = False
) -> Union[str, Tuple[str, List[GraphProto]]]:
    content = []
    if len(node.output):
        content.append(", ".join([f"%{name}" for name in node.output]))
        content.append("=")
    # To deal with nested graphs
    graphs: List[GraphProto] = []
    printed_attrs = []
    for attr in node.attribute:
        if subgraphs:
            printed_attr_subgraphs = printable_attribute(attr, subgraphs)
            if not isinstance(printed_attr_subgraphs[1], list):
                raise TypeError(
                    f"printed_attr_subgraphs[1] must be an instance of {list}."
                )
            graphs.extend(printed_attr_subgraphs[1])
            printed_attrs.append(printed_attr_subgraphs[0])
        else:
            printed = printable_attribute(attr)
            if not isinstance(printed, str):
                raise TypeError(f"printed must be an instance of {str}.")
            printed_attrs.append(printed)
    printed_attributes = ", ".join(sorted(printed_attrs))
    printed_inputs = ", ".join([f"%{name}" for name in node.input])
    if node.attribute:
        content.append(f"{node.op_type}[{printed_attributes}]({printed_inputs})")
    else:
        content.append(f"{node.op_type}({printed_inputs})")
    if subgraphs:
        return prefix + " ".join(content), graphs
    return prefix + " ".join(content)


def printable_graph(graph: GraphProto, prefix: str = "") -> str:
    """
    Display a GraphProto as a string.

    Arguments:
        graph (GraphProto): the graph to display
        prefix (string): prefix of every line

    Returns:
        string
    """
    content = []
    indent = prefix + "  "
    # header
    header = ["graph", graph.name]
    initializers = {t.name for t in graph.initializer}
    if len(graph.input):
        header.append("(")
        in_strs = []  # required inputs
        in_with_init_strs = (
            []
        )  # optional inputs with initializer providing default value
        for inp in graph.input:
            if inp.name not in initializers:
                in_strs.append(printable_value_info(inp))
            else:
                in_with_init_strs.append(printable_value_info(inp))
        if in_strs:
            content.append(prefix + " ".join(header))
            header = []
            for line in in_strs:
                content.append(prefix + "  " + line)
        header.append(")")

        if in_with_init_strs:
            header.append("optional inputs with matching initializers (")
            content.append(prefix + " ".join(header))
            header = []
            for line in in_with_init_strs:
                content.append(prefix + "  " + line)
            header.append(")")

        # from IR 4 onwards an initializer is not required to have a matching graph input
        # so output the name, type and shape of those as well
        if len(in_with_init_strs) < len(initializers):
            graph_inputs = {i.name for i in graph.input}
            init_strs = [
                printable_tensor_proto(i)
                for i in graph.initializer
                if i.name not in graph_inputs
            ]
            header.append("initializers (")
            content.append(prefix + " ".join(header))
            header = []
            for line in init_strs:
                content.append(prefix + "  " + line)
            header.append(")")

    header.append("{")
    content.append(prefix + " ".join(header))
    graphs: List[GraphProto] = []
    # body
    for node in graph.node:
        contents_subgraphs = printable_node(node, indent, subgraphs=True)
        if not isinstance(contents_subgraphs[1], list):
            raise TypeError(f"contents_subgraphs[1] must be an instance of {list}.")
        content.append(contents_subgraphs[0])
        graphs.extend(contents_subgraphs[1])
    # tail
    tail = ["return"]
    if len(graph.output):
        tail.append(", ".join([f"%{out.name}" for out in graph.output]))
    content.append(indent + " ".join(tail))
    # closing bracket
    content.append(prefix + "}")
    for g in graphs:
        content.append("\n" + printable_graph(g))
    return "\n".join(content)


def strip_doc_string(proto: google.protobuf.message.Message) -> None:
    """
    Empties `doc_string` field on any nested protobuf messages
    """
    if not isinstance(proto, google.protobuf.message.Message):
        raise TypeError(
            f"proto must be an instance of {google.protobuf.message.Message}."
        )
    for descriptor in proto.DESCRIPTOR.fields:
        if descriptor.name == "doc_string":
            proto.ClearField(descriptor.name)
        elif descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                for x in getattr(proto, descriptor.name):
                    strip_doc_string(x)
            elif proto.HasField(descriptor.name):
                strip_doc_string(getattr(proto, descriptor.name))


def make_training_info(
    algorithm: GraphProto,
    algorithm_bindings: AssignmentBindingType,
    initialization: Optional[GraphProto],
    initialization_bindings: Optional[AssignmentBindingType],
) -> TrainingInfoProto:
    training_info = TrainingInfoProto()
    training_info.algorithm.CopyFrom(algorithm)
    for k, v in algorithm_bindings:
        binding = training_info.update_binding.add()
        binding.key = k
        binding.value = v

    if initialization:
        training_info.initialization.CopyFrom(initialization)
    if initialization_bindings:
        for k, v in initialization_bindings:
            binding = training_info.initialization_binding.add()
            binding.key = k
            binding.value = v

    return training_info


# Following functions are used for mapping
def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """
    Convert a TensorProto's data_type to corresponding numpy dtype. It can be used while making tensor.

    :param tensor_dtype: TensorProto's data_type
    :return: numpy's data_type
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].np_dtype


def tensor_dtype_to_storage_tensor_dtype(tensor_dtype: int) -> int:
    """
    Convert a TensorProto's data_type to corresponding data_type for storage.

    :param tensor_dtype: TensorProto's data_type
    :return: data_type for storage
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].storage_dtype


def tensor_dtype_to_string(tensor_dtype: int) -> str:
    """
    Get the name of given TensorProto's data_type.

    :param tensor_dtype: TensorProto's data_type
    :return: the name of data_type
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].name


def tensor_dtype_to_field(tensor_dtype: int) -> str:
    """
    Convert a TensorProto's data_type to corresponding field name for storage. It can be used while making tensors.

    :param tensor_dtype: TensorProto's data_type
    :return: field name
    """
    return mapping._STORAGE_TENSOR_TYPE_TO_FIELD[  # pylint: disable=protected-access
        mapping.TENSOR_TYPE_MAP[tensor_dtype].storage_dtype
    ]


def np_dtype_to_tensor_dtype(np_dtype: np.dtype) -> int:
    """
    Convert a numpy's dtype to corresponding tensor type. It can be used while converting numpy arrays to tensors.

    :param np_dtype: numpy's data_type
    :return: TensorsProto's data_type
    """
    return cast(
        int,
        mapping._NP_TYPE_TO_TENSOR_TYPE[np_dtype],  # pylint: disable=protected-access
    )


def get_all_tensor_dtypes() -> KeysView[int]:
    """
    Get all tensor types from TensorProto.

    :return: all tensor types from TensorProto
    """
    return mapping.TENSOR_TYPE_MAP.keys()
