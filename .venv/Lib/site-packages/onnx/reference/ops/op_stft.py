# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_concat_from_sequence import _concat_from_sequence
from onnx.reference.ops.op_dft import _cfft as _dft
from onnx.reference.ops.op_slice import _slice


def _concat(*args, axis=0):  # type: ignore
    return np.concatenate(args, axis=axis)


def _unsqueeze(a, axis):  # type: ignore
    try:
        return np.expand_dims(a, axis=axis)
    except TypeError:
        # numpy 1.18 supports axes as a tuple
        if len(axis) == 1:
            return np.expand_dims(a, axis=tuple(axis)[0])
        for x in reversed(axis):
            a = np.expand_dims(a, axis=x)
        return a


def _stft(x, fft_length: int, hop_length, n_frames, window, onesided=False):  # type: ignore
    """
    Applies one dimensional FFT with window weights.
    torch defines the number of frames as:
    `n_frames = 1 + (len - n_fft) // hop_length`.
    """
    last_axis = len(x.shape) - 1  # op.Sub(op.Shape(op.Shape(x)), one)
    axis = [-2]
    axis2 = [-3]
    window_size = window.shape[0]

    # building frames
    seq = []
    for fs in range(n_frames):
        begin = fs * hop_length
        end = begin + window_size
        sliced_x = _slice(x, np.array([begin]), np.array([end]), axis)  # type: ignore

        # sliced_x may be smaller
        new_dim = sliced_x.shape[-2:-1]
        missing = (window_size - new_dim[0],)
        new_shape = sliced_x.shape[:-2] + missing + sliced_x.shape[-1:]
        cst = np.zeros(new_shape, dtype=x.dtype)
        pad_sliced_x = _concat(sliced_x, cst, axis=-2)

        # same size
        un_sliced_x = _unsqueeze(pad_sliced_x, axis2)
        seq.append(un_sliced_x)

    # concatenation
    new_x = _concat_from_sequence(seq, axis=-3, new_axis=0)

    # calling weighted dft with weights=window
    shape_x = new_x.shape
    shape_x_short = shape_x[:-2]
    shape_x_short_one = tuple(1 for _ in shape_x_short)
    window_shape = (*shape_x_short_one, window_size, 1)
    weights = np.reshape(window, window_shape)
    weighted_new_x = new_x * weights

    result = _dft(
        weighted_new_x, fft_length, last_axis, onesided=onesided, normalize=False
    )

    return result


def _istft(x, fft_length: int, hop_length, window, onesided=False):  # type: ignore
    """
    Reverses of `stft`.
    """
    zero = [0]
    one = [1]
    two = [2]
    axisf = [-2]
    n_frames = x.shape[-2]
    expected_signal_len = fft_length + hop_length * (n_frames - 1)

    # building frames
    seqr = []
    seqi = []
    seqc = []
    for fs in range(n_frames):
        begin = fs
        end = fs + 1
        frame_x = np.squeeze(
            _slice(x, np.array([begin]), np.array([end]), axisf), axis=axisf[0]  # type: ignore
        )

        # ifft
        ift = _dft(frame_x, fft_length, axis=-1, onesided=onesided, normalize=True)
        n_dims = len(ift.shape)

        # real part
        n_dims_1 = n_dims - 1
        sliced = _slice(ift, np.array(zero), np.array(one), [n_dims_1])  # type: ignore
        ytmp = np.squeeze(sliced, axis=n_dims_1)
        ctmp = np.full(ytmp.shape, fill_value=1, dtype=x.dtype) * window

        shape_begin = ytmp.shape[:-1]
        n_left = fs * hop_length
        size = ytmp.shape[-1]
        n_right = expected_signal_len - (n_left + size)

        left_shape = (*shape_begin, n_left)
        right_shape = (*shape_begin, n_right)
        right = np.zeros(right_shape, dtype=x.dtype)
        left = np.zeros(left_shape, dtype=x.dtype)

        y = _concat(left, ytmp, right, axis=-1)
        yc = _concat(left, ctmp, right, axis=-1)

        # imaginary part
        sliced = _slice(ift, np.array(one), np.array(two), [n_dims_1])  # type: ignore
        itmp = np.squeeze(sliced, axis=n_dims_1)
        yi = _concat(left, itmp, right, axis=-1)

        # append
        seqr.append(_unsqueeze(y, axis=-1))
        seqi.append(_unsqueeze(yi, axis=-1))
        seqc.append(_unsqueeze(yc, axis=-1))

    # concatenation
    redr = _concat_from_sequence(seqr, axis=-1, new_axis=0)
    redi = _concat_from_sequence(seqi, axis=-1, new_axis=0)
    redc = _concat_from_sequence(seqc, axis=-1, new_axis=0)

    # unweight
    resr = redr.sum(axis=-1, keepdims=0)  # type: ignore
    resi = redi.sum(axis=-1, keepdims=0)  # type: ignore
    resc = redc.sum(axis=-1, keepdims=0)  # type: ignore
    rr = resr / resc
    ri = resi / resc

    # Make complex
    rr0 = np.expand_dims(rr, axis=0)
    ri0 = np.expand_dims(ri, axis=0)
    conc = _concat(rr0, ri0, axis=0)

    # rotation, bring first dimension to the last position
    result_shape = conc.shape
    reshaped_result = conc.reshape((2, -1))
    transposed = np.transpose(reshaped_result, (1, 0))
    other_dimensions = result_shape[1:]
    final_shape = _concat(other_dimensions, two, axis=0)
    final = transposed.reshape(final_shape)
    return final


class STFT(OpRun):
    def _run(self, x, frame_step, window=None, frame_length=None, onesided=None):  # type: ignore
        if frame_length is None:
            if window is None:
                frame_length = x.shape[-2]
            else:
                frame_length = window.shape[0]
        hop_length = frame_step
        if window is None:
            window = np.ones((frame_length,), dtype=x.dtype)
        n_frames = 1 + (x.shape[-2] - frame_length) // frame_step
        res = _stft(x, frame_length, hop_length, n_frames, window, onesided=onesided)
        return (res.astype(x.dtype),)
