# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

from typing import Sequence

import numpy as np

from onnx.reference.op_run import OpRun


def _fft(x: np.ndarray, fft_length: Sequence[int], axis: int) -> np.ndarray:
    if fft_length is None:
        fft_length = [x.shape[axis]]
    try:
        ft = np.fft.fft(x, fft_length[0], axis=axis)
    except TypeError:
        # numpy 1.16.6, an array cannot be a key in the dictionary
        # fixed in numpy 1.21.5.
        ft = np.fft.fft(x, int(fft_length[0]), axis=axis)

    r = np.real(ft)
    i = np.imag(ft)
    merged = np.vstack([r[np.newaxis, ...], i[np.newaxis, ...]])
    perm = np.arange(len(merged.shape))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    tr = np.transpose(merged, list(perm))
    if tr.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected shape {tr.shape}, x.shape={x.shape} "
            f"fft_length={fft_length}."
        )
    return tr


def _cfft(
    x: np.ndarray,
    fft_length: Sequence[int],
    axis: int,
    onesided: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    if x.shape[-1] == 1:
        tmp = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        tmp = real + 1j * imag
    c = np.squeeze(tmp, -1)
    res = _fft(c, fft_length, axis=axis)
    if onesided:
        slices = [slice(0, a) for a in res.shape]
        slices[axis] = slice(0, res.shape[axis] // 2 + 1)
        res = res[tuple(slices)]  # type: ignore
    if normalize:
        if len(fft_length) == 1:
            res /= fft_length[0]
        else:
            raise NotImplementedError(
                f"normalize=True not implemented for fft_length={fft_length}."
            )
    return res


def _ifft(
    x: np.ndarray, fft_length: Sequence[int], axis: int = -1, onesided: bool = False
) -> np.ndarray:
    ft = np.fft.ifft(x, fft_length[0], axis=axis)
    r = np.real(ft)
    i = np.imag(ft)
    merged = np.vstack([r[np.newaxis, ...], i[np.newaxis, ...]])
    perm = np.arange(len(merged.shape))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    tr = np.transpose(merged, list(perm))
    if tr.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected shape {tr.shape}, x.shape={x.shape} "
            f"fft_length={fft_length}."
        )
    if onesided:
        slices = [slice(a) for a in tr.shape]
        slices[axis] = slice(0, tr.shape[axis] // 2 + 1)
        return tr[tuple(slices)]  # type: ignore
    return tr


def _cifft(
    x: np.ndarray, fft_length: Sequence[int], axis: int = -1, onesided: bool = False
) -> np.ndarray:
    if x.shape[-1] == 1:
        tmp = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        tmp = real + 1j * imag
    c = np.squeeze(tmp, -1)
    return _ifft(c, fft_length, axis=axis, onesided=onesided)


class DFT(OpRun):
    def _run(self, x, dft_length=None, axis=None, inverse=None, onesided=None):  # type: ignore
        if dft_length is None:
            dft_length = np.array([x.shape[axis]], dtype=np.int64)
        if inverse:  # type: ignore
            res = _cifft(x, dft_length, axis=axis, onesided=onesided)
        else:
            res = _cfft(x, dft_length, axis=axis, onesided=onesided)
        return (res.astype(x.dtype),)
