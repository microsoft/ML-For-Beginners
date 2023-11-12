# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# isort:skip_file
import os
import typing
from typing import Union, IO, Optional, TypeVar, Any

import google.protobuf.message

from onnx.onnx_cpp2py_export import ONNX_ML
from onnx.external_data_helper import (
    load_external_data_for_model,
    write_external_data_tensors,
    convert_model_to_external_data,
)
from onnx.onnx_pb import (
    AttributeProto,
    EXPERIMENTAL,
    FunctionProto,
    GraphProto,
    IR_VERSION,
    IR_VERSION_2017_10_10,
    IR_VERSION_2017_10_30,
    IR_VERSION_2017_11_3,
    IR_VERSION_2019_1_22,
    IR_VERSION_2019_3_18,
    IR_VERSION_2019_9_19,
    IR_VERSION_2020_5_8,
    IR_VERSION_2021_7_30,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    OperatorStatus,
    STABLE,
    SparseTensorProto,
    StringStringEntryProto,
    TensorAnnotation,
    TensorProto,
    TensorShapeProto,
    TrainingInfoProto,
    TypeProto,
    ValueInfoProto,
    Version,
)
from onnx.onnx_operators_pb import OperatorProto, OperatorSetProto
from onnx.onnx_data_pb import MapProto, OptionalProto, SequenceProto
from onnx.version import version as __version__

# Import common subpackages so they're available when you 'import onnx'
from onnx import (
    checker,
    compose,
    defs,
    gen_proto,
    helper,
    hub,
    mapping,
    numpy_helper,
    parser,
    printer,
    shape_inference,
    utils,
    version_converter,
)


def _load_bytes(f: Union[IO[bytes], str]) -> bytes:
    if hasattr(f, "read") and callable(typing.cast(IO[bytes], f).read):
        content = typing.cast(IO[bytes], f).read()
    else:
        with open(typing.cast(str, f), "rb") as readable:
            content = readable.read()
    return content


def _save_bytes(content: bytes, f: Union[IO[bytes], str]) -> None:
    if hasattr(f, "write") and callable(typing.cast(IO[bytes], f).write):
        typing.cast(IO[bytes], f).write(content)
    else:
        with open(typing.cast(str, f), "wb") as writable:
            writable.write(content)


def _get_file_path(f: Union[IO[bytes], str]) -> Optional[str]:
    if isinstance(f, str):
        return os.path.abspath(f)
    if hasattr(f, "name"):
        return os.path.abspath(f.name)
    return None


def _serialize(proto: Union[bytes, google.protobuf.message.Message]) -> bytes:
    """Serialize a in-memory proto to bytes.

    Args:
        proto: An in-memory proto, such as a ModelProto, TensorProto, etc

    Returns:
        Serialized proto in bytes.
    """
    if isinstance(proto, bytes):
        return proto
    if hasattr(proto, "SerializeToString") and callable(proto.SerializeToString):
        try:
            result = proto.SerializeToString()
        except ValueError as e:
            if proto.ByteSize() >= checker.MAXIMUM_PROTOBUF:
                raise ValueError(
                    "The proto size is larger than the 2 GB limit. "
                    "Please use save_as_external_data to save tensors separately from the model file."
                ) from e
            raise
        return result  # type: ignore
    raise TypeError(
        f"No SerializeToString method is detected. Neither proto is a str.\ntype is {type(proto)}"
    )


_Proto = TypeVar("_Proto", bound=google.protobuf.message.Message)


def _deserialize(s: bytes, proto: _Proto) -> _Proto:
    """Parse bytes into a in-memory proto.

    Args:
        s: bytes containing serialized proto
        proto: a in-memory proto object

    Returns:
        The proto instance filled in by `s`.

    Raises:
        TypeError: if `proto` is not a protobuf message.
    """
    if not isinstance(s, bytes):
        raise TypeError(f"Parameter 's' must be bytes, but got type: {type(s)}")

    if not (hasattr(proto, "ParseFromString") and callable(proto.ParseFromString)):
        raise TypeError(f"No ParseFromString method is detected. Type is {type(proto)}")

    decoded = typing.cast(Optional[int], proto.ParseFromString(s))
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            f"Protobuf decoding consumed too few bytes: {decoded} out of {len(s)}"
        )
    return proto


def load_model(
    f: Union[IO[bytes], str],
    format: Optional[Any] = None,  # pylint: disable=redefined-builtin
    load_external_data: bool = True,
) -> ModelProto:
    """Loads a serialized ModelProto into memory.

    Args:
        f: can be a file-like object (has "read" function) or a string containing a file name
        format: for future use
        load_external_data: Whether to load the external data.
            Set to True if the data is under the same directory of the model.
            If not, users need to call :func:`load_external_data_for_model`
            with directory to load external data from.

    Returns:
        Loaded in-memory ModelProto.
    """
    s = _load_bytes(f)
    model = load_model_from_string(s, format=format)

    if load_external_data:
        model_filepath = _get_file_path(f)
        if model_filepath:
            base_dir = os.path.dirname(model_filepath)
            load_external_data_for_model(model, base_dir)

    return model


def load_tensor(
    f: Union[IO[bytes], str],
    format: Optional[Any] = None,  # pylint: disable=redefined-builtin
) -> TensorProto:
    """Loads a serialized TensorProto into memory.

    Args:
        f: can be a file-like object (has "read" function) or a string containing a file name
        format: for future use

    Returns:
        Loaded in-memory TensorProto.
    """
    s = _load_bytes(f)
    return load_tensor_from_string(s, format=format)


def load_model_from_string(
    s: bytes,
    format: Optional[Any] = None,  # pylint: disable=redefined-builtin
) -> ModelProto:
    """Loads a binary string (bytes) that contains serialized ModelProto.

    Args:
        s: a string, which contains serialized ModelProto
        format: for future use

    Returns:
        Loaded in-memory ModelProto.
    """
    del format  # Unused
    return _deserialize(s, ModelProto())


def load_tensor_from_string(
    s: bytes,
    format: Optional[Any] = None,  # pylint: disable=redefined-builtin
) -> TensorProto:
    """Loads a binary string (bytes) that contains serialized TensorProto.

    Args:
        s: a string, which contains serialized TensorProto
        format: for future use

    Returns:
        Loaded in-memory TensorProto.
    """
    del format  # Unused
    return _deserialize(s, TensorProto())


def save_model(
    proto: Union[ModelProto, bytes],
    f: Union[IO[bytes], str],
    format: Optional[Any] = None,  # pylint: disable=redefined-builtin
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    location: Optional[str] = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    """
    Saves the ModelProto to the specified path and optionally, serialize tensors with raw data as external data before saving.

    Args:
        proto: should be a in-memory ModelProto
        f: can be a file-like object (has "write" function) or a string containing a file name format for future use
        save_as_external_data: If true, save tensors to external file(s).
        all_tensors_to_one_file: Effective only if save_as_external_data is True.
            If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name.
        location: Effective only if save_as_external_data is true.
            Specify the external file that all tensors to save to.
            If not specified, will use the model name.
        size_threshold: Effective only if save_as_external_data is True.
            Threshold for size of data. Only when tensor's data is >= the size_threshold it will be converted
            to external data. To convert every tensor with raw data to external data set size_threshold=0.
        convert_attribute: Effective only if save_as_external_data is True.
            If true, convert all tensors to external data
            If false, convert only non-attribute tensors to external data
    """
    del format  # Unused

    if isinstance(proto, bytes):
        proto = _deserialize(proto, ModelProto())

    if save_as_external_data:
        convert_model_to_external_data(
            proto, all_tensors_to_one_file, location, size_threshold, convert_attribute
        )

    model_filepath = _get_file_path(f)
    if model_filepath:
        basepath = os.path.dirname(model_filepath)
        proto = write_external_data_tensors(proto, basepath)

    serialized = _serialize(proto)
    _save_bytes(serialized, f)


def save_tensor(proto: TensorProto, f: Union[IO[bytes], str]) -> None:
    """
    Saves the TensorProto to the specified path.

    Args:
        proto: should be a in-memory TensorProto
        f: can be a file-like object (has "write" function) or a string containing a file name
        format: for future use
    """
    serialized = _serialize(proto)
    _save_bytes(serialized, f)


# For backward compatibility
load = load_model
load_from_string = load_model_from_string
save = save_model
