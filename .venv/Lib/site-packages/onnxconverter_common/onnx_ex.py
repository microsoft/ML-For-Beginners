# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

from logging import getLogger
from onnx import helper, defs as onnx_defs, onnx_pb as onnx_proto
from . import utils
from .metadata_props import add_metadata_props

DEFAULT_OPSET_NUMBER = 15  # The maximum opset supported by the converter in the code branch.
# From https://github.com/onnx/onnx/blob/master/docs/Versioning.md
OPSET_TO_IR_VERSION = {
    1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
    7: 3, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7,
    13: 7, 14: 7, 15: 8
}


def _get_main_opset_version(model):
    """
    Returns the main opset version.
    """
    for op in model.opset_import:
        if op.domain == '' or op.domain == 'ai.onnx':
            return op.version
    return None


def onnx_builtin_opset_version():
    return onnx_defs.onnx_opset_version()


def get_maximum_opset_supported():
    return min(DEFAULT_OPSET_NUMBER, onnx_builtin_opset_version())


def make_model_ex(graph, imported_opset_pairs, target_default_opset, metadata_props=None, **kwargs):
    onnx_model = helper.make_model(graph, **kwargs)

    # Merge operator sets for the same domain, the largest version number would be kept
    purified_operator_set = dict()
    for op_domain, op_version in imported_opset_pairs:
        if op_domain not in purified_operator_set:
            if op_domain == '':
                # Initializers are a subset of graph inputs for IR_VERSION <= 3 (target opset < 8).
                # Need upgrade opv since initializers are separate for IR_VERSION >= 4 to pass onnx.checker.
                if op_version < 8 and target_default_opset is not None and target_default_opset >= 8:
                    op_version = 8
            purified_operator_set[op_domain] = op_version
        else:
            purified_operator_set[op_domain] = max(purified_operator_set[op_domain], op_version)

    # Fill operator sets
    i = 0
    for op_domain, op_version in purified_operator_set.items():
        if i == 0 and len(onnx_model.opset_import) == 1:
            # Overwrite the default operator set created by helper.make_model(...)
            op_set = onnx_model.opset_import[0]
        else:
            # Just create one ONNX element in opset_import
            op_set = onnx_model.opset_import.add()
        op_set.domain = op_domain
        op_set.version = op_version
        i += 1
        if op_domain == '' or op_domain == 'ai.onnx':
            if target_default_opset < op_version:
                raise RuntimeError(('The specified opset %d is too low to convert this model, ' +
                                    'which requires at least opset %d.') % (target_default_opset, op_version))
            elif target_default_opset > op_version:
                getLogger('onnxmltools').warning('The maximum opset needed by this model is only %d.' % op_version)
            else:
                pass

    # Add extra information
    if metadata_props:
        add_metadata_props(onnx_model, metadata_props, target_default_opset)
    opv = _get_main_opset_version(onnx_model) or target_default_opset
    irv = OPSET_TO_IR_VERSION.get(opv, onnx_proto.IR_VERSION)
    onnx_model.ir_version = irv
    onnx_model.producer_name = kwargs.get('producer_name') if 'producer_name' in kwargs else utils.get_producer()
    onnx_model.producer_version = utils.get_producer_version()
    onnx_model.domain = kwargs.get('domain') if 'domain' in kwargs else utils.get_domain()
    onnx_model.model_version = utils.get_model_version()
    return onnx_model
