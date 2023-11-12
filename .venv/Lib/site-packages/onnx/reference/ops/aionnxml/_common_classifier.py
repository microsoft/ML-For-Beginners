# SPDX-License-Identifier: Apache-2.0
import numpy as np


def compute_logistic(val: float) -> float:
    v = 1.0 / (1.0 + np.exp(-np.abs(val)))
    return (1.0 - v) if val < 0 else v  # type: ignore


logistic = np.vectorize(compute_logistic)


def compute_softmax_zero(values: np.ndarray) -> np.ndarray:
    """
    The function modifies the input inplace.
    """
    v_max = values.max()
    exp_neg_v_max = np.exp(-v_max)
    s = 0
    for i in range(len(values)):  # pylint: disable=C0200
        v = values[i]
        if v > 0.0000001 or v < -0.0000001:
            values[i] = np.exp(v - v_max)
        else:
            values[i] *= exp_neg_v_max
        s += values[i]
    if s == 0:
        values[:] = 0.5
    else:
        values[:] /= s
    return values


def softmax_zero(values: np.ndarray) -> np.ndarray:
    "Modifications in place."
    if len(values.shape) == 1:
        compute_softmax_zero(values)
        return values
    for row in values:
        compute_softmax_zero(row)
    return values


def softmax(values: np.ndarray) -> np.ndarray:
    "Modifications in place."
    if len(values.shape) == 2:
        v_max = values.max(axis=1, keepdims=1)  # type: ignore
        values -= v_max
        np.exp(values, out=values)
        s = values.sum(axis=1, keepdims=1)  # type: ignore
        values /= s
        return values
    v_max = values.max()
    values[:] = np.exp(values - v_max)
    this_sum = values.sum()
    values /= this_sum
    return values


def erf_inv(x: float) -> float:
    sgn = -1.0 if x < 0 else 1.0
    x = (1.0 - x) * (1 + x)
    if x == 0:
        return 0
    log = np.log(x)
    v = 2.0 / (3.14159 * 0.147) + 0.5 * log
    v2 = 1.0 / 0.147 * log
    v3 = -v + np.sqrt(v * v - v2)
    x = sgn * np.sqrt(v3)
    return x


def compute_probit(val: float) -> float:
    return 1.41421356 * erf_inv(val * 2 - 1)


probit = np.vectorize(compute_probit)


def expit(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(x.dtype)
