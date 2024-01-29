# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class CenterCropPad(OpRun):
    def _run(self, input_data, shape, axes=None):  # type: ignore
        axes = axes or self.axes  # type: ignore
        input_rank = len(input_data.shape)
        if axes is None:
            axes = list(range(input_rank))
        else:
            axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
        pad_slices = [slice(0, s) for s in input_data.shape]
        crop_slices = [slice(0, s) for s in input_data.shape]
        new_shape = list(input_data.shape)
        for a, sh in zip(axes, shape):
            dim = input_data.shape[a]
            if sh == a:
                pass
            elif sh < dim:
                new_shape[a] = sh
                d = dim - sh
                if d % 2 == 0:
                    d //= 2
                    sl = slice(d, dim - d)
                else:
                    d //= 2
                    sl = slice(d, dim - d - 1)
                crop_slices[a] = sl
            else:  # sh > dim
                new_shape[a] = sh
                d = sh - dim
                if d % 2 == 0:
                    d //= 2
                    sl = slice(d, sh - d)
                else:
                    d //= 2
                    sl = slice(d, sh - d - 1)
                pad_slices[a] = sl

        res = np.zeros(tuple(new_shape), dtype=input_data.dtype)
        cropped = input_data[tuple(crop_slices)]
        res[tuple(pad_slices)] = cropped
        return (res,)
