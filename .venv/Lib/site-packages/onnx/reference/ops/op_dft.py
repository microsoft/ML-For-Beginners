# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _fft(x: np.ndarray, fft_length: int, axis: int) -> np.ndarray:
    """Compute the FFT return the real representation of the complex result."""
    transformed = np.fft.fft(x, n=fft_length, axis=axis)
    real_frequencies = np.real(transformed)
    imaginary_frequencies = np.imag(transformed)
    return np.concatenate(
        (real_frequencies[..., np.newaxis], imaginary_frequencies[..., np.newaxis]),
        axis=-1,
    )


def _cfft(
    x: np.ndarray,
    fft_length: int,
    axis: int,
    onesided: bool,
    normalize: bool,
) -> np.ndarray:
    if x.shape[-1] == 1:
        # The input contains only the real part
        signal = x
    else:
        # The input is a real representation of a complex signal
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        signal = real + 1j * imag

    complex_signals = np.squeeze(signal, -1)
    result = _fft(complex_signals, fft_length, axis=axis)
    # Post process the result based on arguments
    if onesided:
        slices = [slice(0, a) for a in result.shape]
        slices[axis] = slice(0, result.shape[axis] // 2 + 1)
        result = result[tuple(slices)]
    if normalize:
        result /= fft_length
    return result


def _ifft(x: np.ndarray, fft_length: int, axis: int, onesided: bool) -> np.ndarray:
    signals = np.fft.ifft(x, fft_length, axis=axis)
    real_signals = np.real(signals)
    imaginary_signals = np.imag(signals)
    merged = np.concatenate(
        (real_signals[..., np.newaxis], imaginary_signals[..., np.newaxis]),
        axis=-1,
    )
    if onesided:
        slices = [slice(a) for a in merged.shape]
        slices[axis] = slice(0, merged.shape[axis] // 2 + 1)
        return merged[tuple(slices)]
    return merged


def _cifft(
    x: np.ndarray, fft_length: int, axis: int, onesided: bool = False
) -> np.ndarray:
    if x.shape[-1] == 1:
        frequencies = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        frequencies = real + 1j * imag
    complex_frequencies = np.squeeze(frequencies, -1)
    return _ifft(complex_frequencies, fft_length, axis=axis, onesided=onesided)


class DFT_17(OpRun):
    def _run(self, x: np.ndarray, dft_length: int | None = None, axis: int = 1, inverse: bool = False, onesided: bool = False) -> tuple[np.ndarray]:  # type: ignore
        # Convert to positive axis
        axis = axis % len(x.shape)
        if dft_length is None:
            dft_length = x.shape[axis]
        if inverse:  # type: ignore
            result = _cifft(x, dft_length, axis=axis, onesided=onesided)
        else:
            result = _cfft(x, dft_length, axis=axis, onesided=onesided, normalize=False)
        return (result.astype(x.dtype),)


class DFT_20(OpRun):
    def _run(self, x: np.ndarray, dft_length: int | None = None, axis: int = -2, inverse: bool = False, onesided: bool = False) -> tuple[np.ndarray]:  # type: ignore
        # Convert to positive axis
        axis = axis % len(x.shape)
        if dft_length is None:
            dft_length = x.shape[axis]
        if inverse:  # type: ignore
            result = _cifft(x, dft_length, axis=axis, onesided=onesided)
        else:
            result = _cfft(x, dft_length, axis=axis, onesided=onesided, normalize=False)
        return (result.astype(x.dtype),)
