# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import warnings
from .case_insensitive_dict import CaseInsensitiveDict
from onnx import onnx_pb as onnx_proto

KNOWN_METADATA_PROPS = CaseInsensitiveDict({
    'Image.BitmapPixelFormat': ['gray8', 'rgb8', 'bgr8', 'rgba8', 'bgra8'],
    'Image.ColorSpaceGamma': ['linear', 'srgb'],
    'Image.NominalPixelRange': ['nominalrange_0_255', 'normalized_0_1', 'normalized_1_1', 'nominalrange_16_235'],
})


def _validate_metadata(metadata_props):
    '''
    Validate metadata properties and possibly show warnings or throw exceptions.

    :param metadata_props: A dictionary of metadata properties,
    with property names and values (see :func:`~onnxmltools.utils.metadata_props.add_metadata_props` for examples)
    '''
    if len(CaseInsensitiveDict(metadata_props)) != len(metadata_props):
        raise RuntimeError('Duplicate metadata props found')

    for key, value in metadata_props.items():
        valid_values = KNOWN_METADATA_PROPS.get(key)
        if valid_values and value.lower() not in valid_values:
            warnings.warn('Key {} has invalid value {}. Valid values are {}'.format(key, value, valid_values))


def add_metadata_props(onnx_model, metadata_props, target_opset):
    '''
    Add metadata properties to the model. See recommended key names at:
    `Extensibility -
        Metadata <https://github.com/onnx/onnx/blob/296953db87b79c0137c5d9c1a8f26dfaa2495afc/docs/IR.md#metadata>`_ and
    `Optional Metadata <https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-metadata>`_


    :param onnx_model: ONNX model object
    :param metadata_props: A dictionary of metadata properties,
        with property names and values (example: `{ 'model_author': 'Alice', 'model_license': 'MIT' }`)
    :param target_opset: Target ONNX opset
    '''
    if target_opset < 7:
        warnings.warn('Metadata properties are not supported in targeted opset - %d' % target_opset)
        return
    _validate_metadata(metadata_props)
    new_metadata = CaseInsensitiveDict({x.key: x.value for x in onnx_model.metadata_props})
    new_metadata.update(metadata_props)
    del onnx_model.metadata_props[:]
    onnx_model.metadata_props.extend(
        onnx_proto.StringStringEntryProto(key=key, value=value)
        for key, value in metadata_props.items()
    )


def set_denotation(onnx_model, input_name, denotation, target_opset, dimension_denotation=None):
    '''
    Set input type denotation and dimension denotation.

    Type denotation is a feature in ONNX 1.2.1 that let's the model specify the content of a tensor
     (e.g. IMAGE or AUDIO).
    This information can be used by the backend. One example where it is useful is in images: Whenever data is bound to
    a tensor with type denotation IMAGE, the backend can process the data (such as transforming the color space and
    pixel format) based on model metadata properties.

    :param onnx_model: ONNX model object
    :param input_name: Name of input tensor to edit (example: `'data0'`)
    :param denotation: Input type denotation
        (`documentation <https://github.com/onnx/onnx/blob/master/docs/TypeDenotation.md#type-denotation-definition>`_)
        (example: `'IMAGE'`)
    :param target_opset: Target ONNX opset
    :param dimension_denotation: List of dimension type denotations.
        The length of the list must be the same of the number of dimensions in the tensor
    (`documentation https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md#denotation-definition>`_)
    (example: `['DATA_BATCH', 'DATA_CHANNEL', 'DATA_FEATURE', 'DATA_FEATURE']`)
    '''
    if target_opset < 7:
        warnings.warn('Denotation is not supported in targeted opset - %d' % target_opset)
        return
    for graph_input in onnx_model.graph.input:
        if graph_input.name == input_name:
            graph_input.type.denotation = denotation
            if dimension_denotation:
                dimensions = graph_input.type.tensor_type.shape.dim
                if len(dimension_denotation) != len(dimensions):
                    raise RuntimeError(
                        'Wrong number of dimensions: input "{}" has {} dimensions'.format(input_name, len(dimensions)))
                for dimension, channel_denotation in zip(dimensions, dimension_denotation):
                    dimension.denotation = channel_denotation
            return onnx_model
    raise RuntimeError('Input "{}" not found'.format(input_name))
