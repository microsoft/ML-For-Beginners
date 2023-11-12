# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional

from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto


class ExternalDataInfo:
    def __init__(self, tensor: TensorProto) -> None:
        self.location = ""
        self.offset = None
        self.length = None
        self.checksum = None
        self.basepath = ""

        for entry in tensor.external_data:
            setattr(self, entry.key, entry.value)

        if self.offset:
            self.offset = int(self.offset)

        if self.length:
            self.length = int(self.length)


def load_external_data_for_tensor(tensor: TensorProto, base_dir: str) -> None:
    """
    Loads data from an external file for tensor.
    Ideally TensorProto should not hold any raw data but if it does it will be ignored.

    Arguments:
        tensor: a TensorProto object.
        base_dir: directory that contains the external data.
    """
    info = ExternalDataInfo(tensor)
    file_location = _sanitize_path(info.location)
    external_data_file_path = os.path.join(base_dir, file_location)

    with open(external_data_file_path, "rb") as data_file:
        if info.offset:
            data_file.seek(info.offset)

        if info.length:
            tensor.raw_data = data_file.read(info.length)
        else:
            tensor.raw_data = data_file.read()


def load_external_data_for_model(model: ModelProto, base_dir: str) -> None:
    """
    Loads external tensors into model

    Arguments:
        model: ModelProto to load external data to
        base_dir: directory that contains external data
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)
            # After loading raw_data from external_data, change the state of tensors
            tensor.data_location = TensorProto.DEFAULT
            # and remove external data
            del tensor.external_data[:]


def set_external_data(
    tensor: TensorProto,
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    checksum: Optional[str] = None,
    basepath: Optional[str] = None,
) -> None:
    if not tensor.HasField("raw_data"):
        raise ValueError(
            "Tensor "
            + tensor.name
            + "does not have raw_data field. Cannot set external data for this tensor."
        )

    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in {
        "location": location,
        "offset": int(offset) if offset is not None else None,
        "length": int(length) if length is not None else None,
        "checksum": checksum,
        "basepath": basepath,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


def convert_model_to_external_data(
    model: ModelProto,
    all_tensors_to_one_file: bool = True,
    location: Optional[str] = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    """
    Call to set all tensors with raw data as external data. This call should preceed 'save_model'.
    'save_model' saves all the tensors data as external data after calling this function.

    Arguments:
        model (ModelProto): Model to be converted.
        all_tensors_to_one_file (bool): If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name.
        location: specify the external file that all tensors to save to.
            If not specified, will use the model name.
        size_threshold: Threshold for size of data. Only when tensor's data is >= the size_threshold
            it will be converted to external data. To convert every tensor with raw data to external data set size_threshold=0.
        convert_attribute (bool): If true, convert all tensors to external data
                       If false, convert only non-attribute tensors to external data
    """
    tensors = _get_initializer_tensors(model)
    if convert_attribute:
        tensors = _get_all_tensors(model)

    if all_tensors_to_one_file:
        file_name = str(uuid.uuid1())
        if location:
            file_name = location
        for tensor in tensors:
            if (
                tensor.HasField("raw_data")
                and sys.getsizeof(tensor.raw_data) >= size_threshold
            ):
                set_external_data(tensor, file_name)
    else:
        for tensor in tensors:
            if (
                tensor.HasField("raw_data")
                and sys.getsizeof(tensor.raw_data) >= size_threshold
            ):
                tensor_location = tensor.name
                if not _is_valid_filename(tensor_location):
                    tensor_location = str(uuid.uuid1())
                set_external_data(tensor, tensor_location)


def convert_model_from_external_data(model: ModelProto) -> None:
    """
    Call to set all tensors which use external data as embedded data.
    save_model saves all the tensors data as embedded data after
    calling this function.

    Arguments:
        model (ModelProto): Model to be converted.
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            if not tensor.HasField("raw_data"):
                raise ValueError("raw_data field doesn't exist.")
            del tensor.external_data[:]
            tensor.data_location = TensorProto.DEFAULT


def save_external_data(tensor: TensorProto, base_path: str) -> None:
    """
    Writes tensor data to an external file according to information in the `external_data` field.

    Arguments:
        tensor (TensorProto): Tensor object to be serialized
        base_path: System path of a folder where tensor data is to be stored
    """
    info = ExternalDataInfo(tensor)
    external_data_file_path = os.path.join(base_path, info.location)

    # Retrieve the tensor's data from raw_data or load external file
    if not tensor.HasField("raw_data"):
        raise ValueError("raw_data field doesn't exist.")

    # Create file if it doesn't exist
    if not os.path.isfile(external_data_file_path):
        with open(external_data_file_path, "ab"):
            pass

    # Open file for reading and writing at random locations ('r+b')
    with open(external_data_file_path, "r+b") as data_file:
        data_file.seek(0, 2)
        if info.offset is not None:
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if info.offset > file_size:
                data_file.write(b"\0" * (info.offset - file_size))

            data_file.seek(info.offset)
        offset = data_file.tell()
        data_file.write(tensor.raw_data)
        set_external_data(tensor, info.location, offset, data_file.tell() - offset)


def _get_all_tensors(onnx_model_proto: ModelProto) -> Iterable[TensorProto]:
    """Scan an ONNX model for all tensors and return as an iterator."""
    return chain(
        _get_initializer_tensors(onnx_model_proto),
        _get_attribute_tensors(onnx_model_proto),
    )


def _recursive_attribute_processor(
    attribute: AttributeProto, func: Callable[[GraphProto], Iterable[TensorProto]]
) -> Iterable[TensorProto]:
    """Create an iterator through processing ONNX model attributes with functor."""
    if attribute.type == AttributeProto.GRAPH:
        yield from func(attribute.g)
    if attribute.type == AttributeProto.GRAPHS:
        for graph in attribute.graphs:
            yield from func(graph)


def _get_initializer_tensors_from_graph(
    onnx_model_proto_graph: GraphProto,
) -> Iterable[TensorProto]:
    """Create an iterator of initializer tensors from ONNX model graph."""
    yield from onnx_model_proto_graph.initializer
    for node in onnx_model_proto_graph.node:
        for attribute in node.attribute:
            yield from _recursive_attribute_processor(
                attribute, _get_initializer_tensors_from_graph
            )


def _get_initializer_tensors(onnx_model_proto: ModelProto) -> Iterable[TensorProto]:
    """Create an iterator of initializer tensors from ONNX model."""
    yield from _get_initializer_tensors_from_graph(onnx_model_proto.graph)


def _get_attribute_tensors_from_graph(
    onnx_model_proto_graph: GraphProto,
) -> Iterable[TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model graph."""
    for node in onnx_model_proto_graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            yield from _recursive_attribute_processor(
                attribute, _get_attribute_tensors_from_graph
            )


def _get_attribute_tensors(onnx_model_proto: ModelProto) -> Iterable[TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model."""
    yield from _get_attribute_tensors_from_graph(onnx_model_proto.graph)


def _sanitize_path(path: str) -> str:
    """Remove path components which would allow traversing up a directory tree from a base path.

    Note: This method is currently very basic and should be expanded.
    """
    return path.lstrip("/.")


def _is_valid_filename(filename: str) -> bool:
    """Utility to check whether the provided filename is valid."""
    exp = re.compile('^[^<>:;,?"*|/]+$')
    match = exp.match(filename)
    return bool(match)


def uses_external_data(tensor: TensorProto) -> bool:
    """Returns true if the tensor stores data in an external location."""
    return (
        tensor.HasField("data_location")
        and tensor.data_location == TensorProto.EXTERNAL
    )


def remove_external_data_field(tensor: TensorProto, field_key: str) -> None:
    """
    Removes a field from a Tensor's external_data key-value store.

    Modifies tensor object in place.

    Arguments:
        tensor (TensorProto): Tensor object from which value will be removed
        field_key (string): The key of the field to be removed
    """
    for i, field in enumerate(tensor.external_data):
        if field.key == field_key:
            del tensor.external_data[i]


def write_external_data_tensors(model: ModelProto, filepath: str) -> ModelProto:
    """
    Serializes data for all the tensors which have data location set to TensorProto.External.

    Note: This function also strips basepath information from all tensors' external_data fields.

    Arguments:
        model (ModelProto): Model object which is the source of tensors to serialize.
        filepath: System path to the directory which should be treated as base path for external data.

    Returns:
        ModelProto: The modified model object.
    """
    for tensor in _get_all_tensors(model):
        # Writing to external data happens in 2 passes:
        # 1. Tensors with raw data which pass the necessary conditions (size threshold etc) are marked for serialization
        # 2. The raw data in these tensors is serialized to a file
        # Thus serialize only if tensor has raw data and it was marked for serialization
        if uses_external_data(tensor) and tensor.HasField("raw_data"):
            save_external_data(tensor, filepath)
            tensor.ClearField("raw_data")

    return model
