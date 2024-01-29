# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class AttributeHasValue(OpRun):
    def _run(  # type: ignore
        self,
        value_float=None,
        value_floats=None,
        value_graph=None,
        value_graphs=None,
        value_int=None,
        value_ints=None,
        value_sparse_tensor=None,
        value_sparse_tensors=None,
        value_string=None,
        value_strings=None,
        value_tensor=None,
        value_tensors=None,
        value_type_proto=None,
        value_type_protos=None,
    ):
        # TODO: support overridden attributes.
        for att in self.onnx_node.attribute:
            if att.name.startswith("value_"):
                return (np.array([True]),)
        return (np.array([False]),)
