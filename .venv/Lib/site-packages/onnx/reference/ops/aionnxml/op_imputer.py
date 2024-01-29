# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class Imputer(OpRunAiOnnxMl):
    def _run(  # type: ignore
        self,
        x,
        imputed_value_floats=None,
        imputed_value_int64s=None,
        replaced_value_float=None,
        replaced_value_int64=None,
    ):
        if imputed_value_floats is not None and len(imputed_value_floats) > 0:
            values = imputed_value_floats
            replace = replaced_value_float
        elif imputed_value_int64s is not None and len(imputed_value_int64s) > 0:
            values = imputed_value_int64s
            replace = replaced_value_int64
        else:
            raise ValueError("Missing are not defined.")

        if isinstance(values, list):
            values = np.array(values)
        if len(x.shape) != 2:
            raise TypeError(f"x must be a matrix but shape is {x.shape}")
        if values.shape[0] not in (x.shape[1], 1):
            raise TypeError(  # pragma: no cover
                f"Dimension mismatch {values.shape[0]} != {x.shape[1]}"
            )
        x = x.copy()
        if np.isnan(replace):
            for i in range(0, x.shape[1]):
                val = values[min(i, values.shape[0] - 1)]
                x[np.isnan(x[:, i]), i] = val
        else:
            for i in range(0, x.shape[1]):
                val = values[min(i, values.shape[0] - 1)]
                x[x[:, i] == replace, i] = val

        return (x,)
