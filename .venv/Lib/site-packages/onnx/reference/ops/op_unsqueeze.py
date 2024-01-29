# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Unsqueeze_1(OpRun):
    def _run(self, data, axes=None):  # type: ignore
        if isinstance(axes, np.ndarray):
            axes = tuple(axes)
        elif axes in ([], ()):
            axes = None
        elif isinstance(axes, list):
            axes = tuple(axes)
        if isinstance(axes, (tuple, list)):
            sq = data
            for a in axes:
                sq = np.expand_dims(sq, axis=a)
        else:
            raise RuntimeError(
                "axes cannot be None for operator Unsqueeze (Unsqueeze_1)."
            )
        return (sq,)


class Unsqueeze_11(Unsqueeze_1):
    pass


class Unsqueeze_13(OpRun):
    def _run(self, data, axes=None):  # type: ignore
        if axes is not None:
            if hasattr(axes, "__iter__") and len(axes.shape) > 0:
                try:
                    sq = np.expand_dims(data, axis=tuple(axes))
                except TypeError:
                    # numpy 1.18 supports axes as a tuple
                    if len(axes) == 1:
                        sq = np.expand_dims(data, axis=tuple(axes)[0])
                    else:
                        sq = data
                        for a in reversed(axes):
                            sq = np.expand_dims(sq, axis=a)
            else:
                sq = np.expand_dims(data, axis=axes)
        else:
            raise RuntimeError(
                "axes cannot be None for operator Unsqueeze (Unsqueeze_13)."
            )
        return (sq,)
