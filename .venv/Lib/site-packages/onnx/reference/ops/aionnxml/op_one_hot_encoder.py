# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class OneHotEncoder(OpRunAiOnnxMl):
    def _run(self, x, cats_int64s=None, cats_strings=None, zeros=None):  # type: ignore
        if cats_int64s is not None and len(cats_int64s) > 0:
            classes = {v: i for i, v in enumerate(cats_int64s)}
        elif len(cats_strings) > 0:
            classes = {v: i for i, v in enumerate(cats_strings)}
        else:
            raise RuntimeError("No encoding was defined.")

        shape = x.shape
        new_shape = (*shape, len(classes))
        res = np.zeros(new_shape, dtype=np.float32)
        if len(x.shape) == 1:
            for i, v in enumerate(x):
                j = classes.get(v, -1)
                if j >= 0:
                    res[i, j] = 1.0
        elif len(x.shape) == 2:
            for a, row in enumerate(x):
                for i, v in enumerate(row):
                    j = classes.get(v, -1)
                    if j >= 0:
                        res[a, i, j] = 1.0
        else:
            raise RuntimeError(f"This operator is not implemented shape {x.shape}.")

        if not zeros:
            red = res.sum(axis=len(res.shape) - 1)
            if np.min(red) == 0:
                rows = []
                for i, val in enumerate(red):
                    if val == 0:
                        rows.append({"row": i, "value": x[i]})
                        if len(rows) > 5:
                            break
                msg = "\n".join(str(_) for _ in rows)
                raise RuntimeError(
                    f"One observation did not have any defined category.\n"
                    f"classes: {classes}\nfirst rows:\n"
                    f"{msg}\nres:\n{res[:5]}\nx:\n{x[:5]}"
                )

        return (res,)
