# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Squeeze_1(OpRun):
    def _run(self, data, axes=None):  # type: ignore
        if isinstance(axes, np.ndarray):
            axes = tuple(axes)
        elif axes in [[], ()]:
            axes = None
        elif isinstance(axes, list):
            axes = tuple(axes)
        if isinstance(axes, (tuple, list)):
            sq = data
            for a in reversed(axes):
                sq = np.squeeze(sq, axis=a)
        else:
            sq = np.squeeze(data, axis=axes)
        return (sq,)


class Squeeze_11(Squeeze_1):
    pass


class Squeeze_13(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.axes = None

    def _run(self, data, axes=None):  # type: ignore
        if axes is not None:
            if hasattr(axes, "__iter__"):
                sq = np.squeeze(data, axis=tuple(axes))
            else:
                sq = np.squeeze(data, axis=axes)
        else:
            sq = np.squeeze(data)
        return (sq,)
