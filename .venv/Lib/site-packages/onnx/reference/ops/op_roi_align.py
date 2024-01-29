# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from onnx.reference.op_run import OpRun


class PreCalc:
    def __init__(self, pos1=0, pos2=0, pos3=0, pos4=0, w1=0, w2=0, w3=0, w4=0):  # type: ignore
        self.pos1 = pos1
        self.pos2 = pos2
        self.pos3 = pos3
        self.pos4 = pos4
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def __repr__(self) -> str:
        return f"PreCalc({self.pos1},{self.pos2},{self.pos3},{self.pos4},{self.w1},{self.w2},{self.w3},{self.w4})"


class RoiAlign(OpRun):
    @staticmethod
    def pre_calc_for_bilinear_interpolate(  # type: ignore
        height: int,
        width: int,
        pooled_height: int,
        pooled_width: int,
        iy_upper: int,
        ix_upper: int,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h: int,
        roi_bin_grid_w: int,
        pre_calc,
    ):
        pre_calc_index = 0
        for ph in range(pooled_height):
            for pw in range(pooled_width):
                for iy in range(iy_upper):
                    yy = (
                        roi_start_h
                        + ph * bin_size_h
                        + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                    )
                    for ix in range(ix_upper):
                        xx = (
                            roi_start_w
                            + pw * bin_size_w
                            + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                        )

                        x = xx
                        y = yy
                        # deal with: inverse elements are out of feature map boundary
                        if y < -1.0 or y > height or x < -1.0 or x > width:
                            pc = pre_calc[pre_calc_index]
                            pc.pos1 = 0
                            pc.pos2 = 0
                            pc.pos3 = 0
                            pc.pos4 = 0
                            pc.w1 = 0
                            pc.w2 = 0
                            pc.w3 = 0
                            pc.w4 = 0
                            pre_calc_index += 1
                            continue

                        y = max(y, 0)
                        x = max(x, 0)

                        y_low = int(y)
                        x_low = int(x)

                        if y_low >= height - 1:
                            y_high = y_low = height - 1
                            y = y_low
                        else:
                            y_high = y_low + 1

                        if x_low >= width - 1:
                            x_high = x_low = width - 1
                            x = x_low
                        else:
                            x_high = x_low + 1

                        ly = y - y_low
                        lx = x - x_low
                        hy = 1.0 - ly
                        hx = 1.0 - lx
                        w1 = hy * hx
                        w2 = hy * lx
                        w3 = ly * hx
                        w4 = ly * lx

                        # save weights and indeces
                        pc = PreCalc()
                        pc.pos1 = y_low * width + x_low
                        pc.pos2 = y_low * width + x_high
                        pc.pos3 = y_high * width + x_low
                        pc.pos4 = y_high * width + x_high
                        pc.w1 = w1
                        pc.w2 = w2
                        pc.w3 = w3
                        pc.w4 = w4
                        pre_calc[pre_calc_index] = pc

                        pre_calc_index += 1

    @staticmethod
    def roi_align_forward(  # type: ignore
        output_shape: Tuple[int, int, int, int],
        bottom_data,
        spatial_scale,
        height: int,
        width: int,
        sampling_ratio,
        bottom_rois,
        num_roi_cols: int,
        top_data,
        mode,
        half_pixel: bool,
        batch_indices_ptr,
    ):
        n_rois = output_shape[0]
        channels = output_shape[1]
        pooled_height = output_shape[2]
        pooled_width = output_shape[3]

        # 100 is a random chosed value, need be tuned
        for n in range(n_rois):
            index_n = n * channels * pooled_width * pooled_height

            # bottom_rois
            offset_bottom_rois = n * num_roi_cols
            roi_batch_ind = batch_indices_ptr[n]

            # Do not using rounding; this implementation detail is critical.
            offset = 0.5 if half_pixel else 0.0
            roi_start_w = bottom_rois[offset_bottom_rois + 0] * spatial_scale - offset
            roi_start_h = bottom_rois[offset_bottom_rois + 1] * spatial_scale - offset
            roi_end_w = bottom_rois[offset_bottom_rois + 2] * spatial_scale - offset
            roi_end_h = bottom_rois[offset_bottom_rois + 3] * spatial_scale - offset

            roi_width = roi_end_w - roi_start_w
            roi_height = roi_end_h - roi_start_h
            if not half_pixel:
                # Force malformed ROIs to be 1x1
                roi_width = max(roi_width, 1.0)
                roi_height = max(roi_height, 1.0)

            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            # We use roi_bin_grid to sample the grid and mimic integral
            roi_bin_grid_h = (
                int(sampling_ratio)
                if sampling_ratio > 0
                else int(np.ceil(roi_height / pooled_height))
            )
            roi_bin_grid_w = (
                int(sampling_ratio)
                if sampling_ratio > 0
                else int(np.ceil(roi_width / pooled_width))
            )

            # We do average (integral) pooling inside a bin
            count = int(max(roi_bin_grid_h * roi_bin_grid_w, 1))

            # we want to precalculate indices and weights shared by all channels,
            # this is the key point of optimization
            pre_calc = [
                PreCalc()
                for i in range(
                    roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height
                )
            ]
            RoiAlign.pre_calc_for_bilinear_interpolate(
                height,
                width,
                pooled_height,
                pooled_width,
                roi_bin_grid_h,
                roi_bin_grid_w,
                roi_start_h,
                roi_start_w,
                bin_size_h,
                bin_size_w,
                roi_bin_grid_h,
                roi_bin_grid_w,
                pre_calc,
            )

            for c in range(channels):
                index_n_c = index_n + c * pooled_width * pooled_height
                # bottom_data
                offset_bottom_data = int(
                    (roi_batch_ind * channels + c) * height * width
                )
                pre_calc_index = 0

                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        index = index_n_c + ph * pooled_width + pw

                        output_val = 0.0
                        if mode == "avg":  # avg pooling
                            for _iy in range(roi_bin_grid_h):
                                for _ix in range(roi_bin_grid_w):
                                    pc = pre_calc[pre_calc_index]
                                    output_val += (
                                        pc.w1
                                        * bottom_data[offset_bottom_data + pc.pos1]
                                        + pc.w2
                                        * bottom_data[offset_bottom_data + pc.pos2]
                                        + pc.w3
                                        * bottom_data[offset_bottom_data + pc.pos3]
                                        + pc.w4
                                        * bottom_data[offset_bottom_data + pc.pos4]
                                    )

                                    pre_calc_index += 1
                            output_val /= count
                        else:  # max pooling
                            max_flag = False
                            for _iy in range(roi_bin_grid_h):
                                for _ix in range(roi_bin_grid_w):
                                    pc = pre_calc[pre_calc_index]
                                    val = max(
                                        pc.w1
                                        * bottom_data[offset_bottom_data + pc.pos1],
                                        pc.w2
                                        * bottom_data[offset_bottom_data + pc.pos2],
                                        pc.w3
                                        * bottom_data[offset_bottom_data + pc.pos3],
                                        pc.w4
                                        * bottom_data[offset_bottom_data + pc.pos4],
                                    )
                                    if not max_flag:
                                        output_val = val
                                        max_flag = True
                                    else:
                                        output_val = max(output_val, val)
                                    pre_calc_index += 1

                        top_data[index] = output_val

    def _run(  # type: ignore
        self,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode=None,
        mode=None,
        output_height=None,
        output_width=None,
        sampling_ratio=None,
        spatial_scale=None,
    ):
        coordinate_transformation_mode = coordinate_transformation_mode or self.coordinate_transformation_mode  # type: ignore
        mode = mode or self.mode  # type: ignore
        output_height = output_height or self.output_height  # type: ignore
        output_width = output_width or self.output_width  # type: ignore
        sampling_ratio = sampling_ratio or self.sampling_ratio  # type: ignore
        spatial_scale = spatial_scale or self.spatial_scale  # type: ignore

        num_channels = X.shape[1]
        num_rois = batch_indices.shape[0]
        num_roi_cols = rois.shape[1]

        y_dims = (num_rois, num_channels, output_height, output_width)
        Y = np.empty(y_dims, dtype=X.dtype).flatten()

        self.roi_align_forward(
            y_dims,
            X.flatten(),
            spatial_scale,
            X.shape[2],  # height, 3
            X.shape[3],  # width, 4
            sampling_ratio,
            rois.flatten(),
            num_roi_cols,
            Y,
            mode.lower(),
            coordinate_transformation_mode.lower() == "half_pixel",
            batch_indices.flatten(),
        )
        return (Y.reshape(y_dims).astype(X.dtype),)
