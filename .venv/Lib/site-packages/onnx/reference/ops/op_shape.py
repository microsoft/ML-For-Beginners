# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple

import numpy as np

from onnx.reference.op_run import OpRun


class Shape_1(OpRun):
    def _run(self, data):  # type: ignore
        return (np.array(data.shape, dtype=np.int64),)


class Shape_15(Shape_1):
    @staticmethod
    def _interval(
        n: int, start: Optional[int], end: Optional[int]
    ) -> Optional[Tuple[int, int]]:
        if start == 0:
            if end is None or np.isnan(end):
                return None
            if end < 0:
                return (0, n + end)
            return (0, end)
        if end is None or np.isnan(end):
            return (start, n)  # type: ignore
        if end < 0:
            return (start, n + end)  # type: ignore
        return (start, end)  # type: ignore

    def _run(self, data, end=None, start=None):  # type: ignore
        ab = self._interval(len(data.shape), start=start, end=end)
        if ab is None:
            return (np.array(data.shape, dtype=np.int64),)
        return (np.array(data.shape[ab[0] : ab[1]], dtype=np.int64),)
