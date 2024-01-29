# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.helper import tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun


class MelWeightMatrix(OpRun):
    def _run(  # type: ignore
        self,
        num_mel_bins,
        dft_length,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
        output_datatype=None,
    ):
        num_spectrogram_bins = dft_length // 2 + 1
        frequency_bins = np.arange(0, num_mel_bins + 2)

        low_frequency_mel = 2595 * np.log10(1 + lower_edge_hertz / 700)
        high_frequency_mel = 2595 * np.log10(1 + upper_edge_hertz / 700)
        mel_step = (high_frequency_mel - low_frequency_mel) / frequency_bins.shape[0]

        frequency_bins = frequency_bins * mel_step + low_frequency_mel
        frequency_bins = 700 * (np.power(10, (frequency_bins / 2595)) - 1)
        frequency_bins = ((dft_length + 1) * frequency_bins) // sample_rate
        frequency_bins = frequency_bins.astype(int)

        output = np.zeros((num_spectrogram_bins, num_mel_bins))
        output.flags.writeable = True

        for i in range(num_mel_bins):
            lower_frequency_value = frequency_bins[i]  # left
            center_frequency_point = frequency_bins[i + 1]  # center
            higher_frequency_point = frequency_bins[i + 2]  # right
            low_to_center = center_frequency_point - lower_frequency_value
            if low_to_center == 0:
                output[center_frequency_point, i] = 1
            else:
                for j in range(lower_frequency_value, center_frequency_point + 1):
                    output[j, i] = float(j - lower_frequency_value) / float(
                        low_to_center
                    )
            center_to_high = higher_frequency_point - center_frequency_point
            if center_to_high > 0:
                for j in range(center_frequency_point, higher_frequency_point):
                    output[j, i] = float(higher_frequency_point - j) / float(
                        center_to_high
                    )

        if output_datatype is None:
            output = output.astype(np.float32)
        else:
            dtype = tensor_dtype_to_np_dtype(output_datatype)
            output = output.astype(dtype)
        return (output,)
