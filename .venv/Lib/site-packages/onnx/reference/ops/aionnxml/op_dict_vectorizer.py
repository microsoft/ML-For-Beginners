# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class DictVectorizer(OpRunAiOnnxMl):
    def _run(self, x, int64_vocabulary=None, string_vocabulary=None):  # type: ignore
        if isinstance(x, (np.ndarray, list)):
            dict_labels = {}
            if int64_vocabulary:
                for i, v in enumerate(int64_vocabulary):
                    dict_labels[v] = i
            else:
                for i, v in enumerate(string_vocabulary):
                    dict_labels[v] = i
            if not dict_labels:
                raise RuntimeError(
                    "int64_vocabulary and string_vocabulary cannot be both empty."
                )

            values_list = []
            rows_list = []
            cols_list = []
            for i, row in enumerate(x):
                for k, v in row.items():
                    values_list.append(v)
                    rows_list.append(i)
                    cols_list.append(dict_labels[k])
            values = np.array(values_list)
            rows = np.array(rows_list)
            cols = np.array(cols_list)

            res = np.zeros((len(x), len(dict_labels)), dtype=values.dtype)  # type: ignore
            for r, c, v in zip(rows, cols, values):
                res[r, c] = v
            return (res,)

            # return (
            #     coo_matrix(
            #         (values, (rows, cols)), shape=(len(x), len(dict_labels))
            #     ).todense(),
            # )

        if isinstance(x, dict):
            keys = int64_vocabulary or string_vocabulary
            result = []
            for k in keys:
                result.append(x.get(k, 0))
            return (np.array(result),)

        raise TypeError(f"x must be iterable not {type(x)}.")
