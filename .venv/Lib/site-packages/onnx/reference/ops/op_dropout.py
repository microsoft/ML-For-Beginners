# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState  # type: ignore

from onnx.reference.op_run import OpRun


def _dropout(
    X: np.ndarray,
    drop_probability: float = 0.5,
    seed: Optional[int] = None,
    training_mode: bool = False,
    return_mask: bool = False,
) -> Tuple[np.ndarray]:
    if drop_probability == 0 or not training_mode:
        if return_mask:
            return X, np.ones(X.shape, dtype=bool)  # type: ignore
        return (X,)

    rnd = RandomState(seed)
    mask = rnd.uniform(0, 1.0, X.shape) >= drop_probability
    scale = 1.0 / (1.0 - drop_probability)
    return (mask * X * scale, mask.astype(bool)) if return_mask else (mask * X * scale,)  # type: ignore


class DropoutBase(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)

    def _private_run(
        self,
        X: np.ndarray,
        seed: Optional[int] = None,
        ratio: float = 0.5,
        training_mode: bool = False,
    ) -> Tuple[np.ndarray]:
        return _dropout(
            X,
            ratio,
            seed=seed,  # type: ignore
            return_mask=self.n_outputs == 2,
            training_mode=training_mode,
        )


class Dropout_7(DropoutBase):
    def _run(self, X, ratio=None):  # type: ignore
        return self._private_run(X, ratio)


class Dropout_12(DropoutBase):
    def _run(self, *inputs, seed=None):  # type: ignore
        X = inputs[0]
        ratio = 0.5 if len(inputs) <= 1 else inputs[1]
        training_mode = False if len(inputs) <= 2 else inputs[2]
        return self._private_run(
            X, seed=seed, ratio=ratio, training_mode=training_mode  # type: ignore
        )
