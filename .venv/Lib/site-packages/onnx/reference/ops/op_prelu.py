# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class PRelu(OpRun):
    def _run(self, x, slope):  # type: ignore
        try:
            return (np.where(x > 0, x, x * slope).astype(x.dtype),)
        except ValueError as e:
            # Broadcast did not work according to numpy.
            # The logic is then the following, if slope has d elements,
            # the following code is looking for d in x.shape. If it is found
            # only once, x * slope is broadcasted on any other dimension.
            # Otherwise, it raises e.
            if len(slope.shape) == 1:
                dim = slope.shape[0]
                new_shape = []
                n = 0
                for d in x.shape:
                    if d == dim:
                        new_shape.append(d)
                        n += 1
                    else:
                        new_shape.append(1)
                if n == 1:
                    xs = x * slope.reshape(tuple(new_shape))
                    return (np.where(x > 0, x, xs).astype(x.dtype),)
            raise e
