# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

__all__ = [
    # Constants
    "ONNX_ML",
    "IR_VERSION",
    "IR_VERSION_2017_10_10",
    "IR_VERSION_2017_10_30",
    "IR_VERSION_2017_11_3",
    "IR_VERSION_2019_1_22",
    "IR_VERSION_2019_3_18",
    "IR_VERSION_2019_9_19",
    "IR_VERSION_2020_5_8",
    "IR_VERSION_2021_7_30",
    "EXPERIMENTAL",
    "STABLE",
    # Modules
    "checker",
    "compose",
    "defs",
    "gen_proto",
    "helper",
    "hub",
    "mapping",
    "numpy_helper",
    "parser",
    "printer",
    "shape_inference",
    "utils",
    "version_converter",
    # Proto classes
    "AttributeProto",
    "FunctionProto",
    "GraphProto",
    "MapProto",
    "ModelProto",
    "NodeProto",
    "OperatorProto",
    "OperatorSetIdProto",
    "OperatorSetProto",
    "OperatorStatus",
    "OptionalProto",
    "SequenceProto",
    "SparseTensorProto",
    "StringStringEntryProto",
    "TensorAnnotation",
    "TensorProto",
    "TensorShapeProto",
    "TrainingInfoProto",
    "TypeProto",
    "ValueInfoProto",
    "Version",
    # Utility functions
    "convert_model_to_external_data",
    "load_external_data_for_model",
    "load_model_from_string",
    "load_model",
    "load_tensor_from_string",
    "load_tensor",
    "save_model",
    "save_tensor",
    "write_external_data_tensors",
]
# isort:skip_file

import os
import typing
from typing import IO, Literal, Union


from onnx import serialization
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

# Supported model formats that can be loaded from and saved to
# The literals are formats with built-in support. But we also allow users to
# register their own formats. So we allow str as well.
_SupportedFormat = Union[Literal["protobuf", "textproto"], str]
# Default serialization format
_DEFAULT_FORMAT = "protobuf"


def _load_bytes(f: IO[bytes] | str | os.PathLike) -> bytes:
    if hasattr(f, "read") and callable(typing.cast(IO[bytes], f).read):
        content = typing.cast(IO[bytes], f).read()
    else:
        f = typing.cast(Union[str, os.PathLike], f)
        with open(f, "rb") as readable:
            content = readable.read()
    return content


def _save_bytes(content: bytes, f: IO[bytes] | str | os.PathLike) -> None:
    if hasattr(f, "write") and callable(typing.cast(IO[bytes], f).write):
        typing.cast(IO[bytes], f).write(content)
    else:
        f = typing.cast(Union[str, os.PathLike], f)
        with open(f, "wb") as writable:
            writable.write(content)


def _get_file_path(f: IO[bytes] | str | os.PathLike | None) -> str | None:
    if isinstance(f, (str, os.PathLike)):
        return os.path.abspath(f)
    if hasattr(f, "name"):
        assert f is not None
        return os.path.abspath(f.name)
    return None


def _get_serializer(
    fmt: _SupportedFormat | None, f: str | os.PathLike | IO[bytes] | None = None
) -> serialization.ProtoSerializer:
    """Get the serializer for the given path and format from the serialization registry."""
    # Use fmt if it is specified
    if fmt is not None:
        return serialization.registry.get(fmt)

    if (file_path := _get_file_path(f)) is not None:
        _, ext = os.path.splitext(file_path)
        fmt = serialization.registry.get_format_from_file_extension(ext)

    # Failed to resolve format if fmt is None. Use protobuf as default
    fmt = fmt or _DEFAULT_FORMAT
    assert fmt is not None

    return serialization.registry.get(fmt)


def load_model(
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,
    load_external_data: bool = True,
) -> ModelProto:
    """Loads a serialized ModelProto into memory.

    Args:
        f: can be a file-like object (has "read" function) or a string/PathLike containing a file name
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
        load_external_data: Whether to load the external data.
            Set to True if the data is under the same directory of the model.
            If not, users need to call :func:`load_external_data_for_model`
            with directory to load external data from.

    Returns:
        Loaded in-memory ModelProto.
    """
    model = _get_serializer(format, f).deserialize_proto(_load_bytes(f), ModelProto())

    if load_external_data:
        model_filepath = _get_file_path(f)
        if model_filepath:
            base_dir = os.path.dirname(model_filepath)
            load_external_data_for_model(model, base_dir)

    return model


def load_tensor(
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,
) -> TensorProto:
    """Loads a serialized TensorProto into memory.

    Args:
        f: can be a file-like object (has "read" function) or a string/PathLike containing a file name
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.

    Returns:
        Loaded in-memory TensorProto.
    """
    return _get_serializer(format, f).deserialize_proto(_load_bytes(f), TensorProto())


def load_model_from_string(
    s: bytes | str,
    format: _SupportedFormat = _DEFAULT_FORMAT,
) -> ModelProto:
    """Loads a binary string (bytes) that contains serialized ModelProto.

    Args:
        s: a string, which contains serialized ModelProto
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.

    Returns:
        Loaded in-memory ModelProto.
    """
    return _get_serializer(format).deserialize_proto(s, ModelProto())


def load_tensor_from_string(
    s: bytes,
    format: _SupportedFormat = _DEFAULT_FORMAT,
) -> TensorProto:
    """Loads a binary string (bytes) that contains serialized TensorProto.

    Args:
        s: a string, which contains serialized TensorProto
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.

    Returns:
        Loaded in-memory TensorProto.
    """
    return _get_serializer(format).deserialize_proto(s, TensorProto())


def save_model(
    proto: ModelProto | bytes,
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,
    *,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    """
    Saves the ModelProto to the specified path and optionally, serialize tensors with raw data as external data before saving.

    Args:
        proto: should be a in-memory ModelProto
        f: can be a file-like object (has "write" function) or a string containing
        a file name or a pathlike object
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
        save_as_external_data: If true, save tensors to external file(s).
        all_tensors_to_one_file: Effective only if save_as_external_data is True.
            If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name.
        location: Effective only if save_as_external_data is true.
            Specify the external file that all tensors to save to.
            Path is relative to the model path.
            If not specified, will use the model name.
        size_threshold: Effective only if save_as_external_data is True.
            Threshold for size of data. Only when tensor's data is >= the size_threshold it will be converted
            to external data. To convert every tensor with raw data to external data set size_threshold=0.
        convert_attribute: Effective only if save_as_external_data is True.
            If true, convert all tensors to external data
            If false, convert only non-attribute tensors to external data
    """
    if isinstance(proto, bytes):
        proto = _get_serializer(_DEFAULT_FORMAT).deserialize_proto(proto, ModelProto())

    if save_as_external_data:
        convert_model_to_external_data(
            proto, all_tensors_to_one_file, location, size_threshold, convert_attribute
        )

    model_filepath = _get_file_path(f)
    if model_filepath is not None:
        basepath = os.path.dirname(model_filepath)
        proto = write_external_data_tensors(proto, basepath)

    serialized = _get_serializer(format, model_filepath).serialize_proto(proto)
    _save_bytes(serialized, f)


def save_tensor(
    proto: TensorProto,
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,
) -> None:
    """
    Saves the TensorProto to the specified path.

    Args:
        proto: should be a in-memory TensorProto
        f: can be a file-like object (has "write" function) or a string
        containing a file name or a pathlike object.
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
    """
    serialized = _get_serializer(format, f).serialize_proto(proto)
    _save_bytes(serialized, f)


# For backward compatibility
load = load_model
load_from_string = load_model_from_string
save = save_model
