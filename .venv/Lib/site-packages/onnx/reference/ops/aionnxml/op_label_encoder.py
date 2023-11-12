# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class LabelEncoder(OpRunAiOnnxMl):
    def _run(  # type: ignore
        self,
        x,
        default_float=None,
        default_int64=None,
        default_string=None,
        keys_floats=None,
        keys_int64s=None,
        keys_strings=None,
        values_floats=None,
        values_int64s=None,
        values_strings=None,
    ):
        keys = keys_floats or keys_int64s or keys_strings
        values = values_floats or values_int64s or values_strings
        classes = dict(zip(keys, values))
        if id(keys) == id(keys_floats):
            cast = float
        elif id(keys) == id(keys_int64s):
            cast = int  # type: ignore
        else:
            cast = str  # type: ignore
        if id(values) == id(values_floats):
            defval = default_float
            dtype = np.float32
        elif id(values) == id(values_int64s):
            defval = default_int64
            dtype = np.int64  # type: ignore
        else:
            defval = default_string
            if not isinstance(defval, str):
                defval = ""
            dtype = np.str_  # type: ignore
        shape = x.shape
        if len(x.shape) > 1:
            x = x.flatten()
        res = []
        for i in range(0, x.shape[0]):
            v = classes.get(cast(x[i]), defval)
            res.append(v)
        return (np.array(res, dtype=dtype).reshape(shape),)
