# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numbers
from typing import List

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords


class GridSample(OpRun):
    # https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/GridSampler.h#L26
    def _gs_denormalize(self, n, length: int, align_corners: bool):  # type: ignore
        # n is the normalized coordinate (float)
        # x is the unormalized coordinate (float)
        if align_corners:
            # Align to corners
            # x_min = 0
            # x_max = d-1
            # Linear mapping from [x_min, x_max] to [-1, 1]
            # Solving linear equation n = ax + b
            # a = 2/(d-1)
            # b = -1
            # n = 2/(d-1) x - 1
            # n(d-1) = 2x - (d-1)
            # x = (n+1)(d-1) / 2
            x = (n + 1) / 2.0 * (length - 1)
        else:
            # Not align to corners
            # x_min = -0.5
            # x_max = d-0.5
            # Linear mapping from [x_min, x_max] to [-1, 1]
            # Solving linear equation n = ax + b
            # a = 2/d
            # b = 1/d - 1
            # n = 2/d x + 1/d - 1
            # nd = 2x + 1 - d
            # x = (nd + d - 1) / 2
            # x = ((n + 1) d - 1) / 2
            x = ((n + 1) * length - 1) / 2.0
        return x

    def _gs_denormalize_coordinates(self, n, dims, align_corners: bool):
        x = np.zeros(len(n), dtype=np.float32)
        for i, (v, dim) in enumerate(zip(n, dims)):
            x[i] = self._gs_denormalize(n=v, length=dim, align_corners=align_corners)
        return x

    def _gs_reflect(self, x, x_min, x_max):  # type: ignore
        """
        Reflect by the near border till within the borders
        Use float for borders to avoid potential issues with integer T
        """
        fx = x
        rng = x_max - x_min
        if fx < x_min:
            dx = x_min - fx
            n = int(dx / rng)
            r = dx - n * rng
            if n % 2 == 0:
                fx = x_min + r
            else:
                fx = x_max - r
        elif fx > x_max:
            dx = fx - x_max
            n = int(dx / rng)
            r = dx - n * rng
            if n % 2 == 0:
                fx = x_max - r
            else:
                fx = x_min + r
        return fx

    def _gs_get_cubic_coeffs(self, x, coeffs):  # type: ignore
        """
        Calculate cubic convolution interpolation coefficients
        ROBERT G. KEYS https://ieeexplore.ieee.org/document/1163711
        Use float to avoid potential issues with integer.
        """
        cubic_alpha = -0.75
        x = abs(x)
        coeffs[0] = (
            (cubic_alpha * (x + 1) - 5 * cubic_alpha) * (x + 1) + 8 * cubic_alpha
        ) * (x + 1) - 4 * cubic_alpha
        coeffs[1] = ((cubic_alpha + 2) * x - (cubic_alpha + 3)) * x * x + 1
        coeffs[2] = ((cubic_alpha + 2) * (1 - x) - (cubic_alpha + 3)) * (1 - x) * (
            1 - x
        ) + 1
        coeffs[3] = (
            (cubic_alpha * (2 - x) - 5 * cubic_alpha) * (2 - x) + 8 * cubic_alpha
        ) * (2 - x) - 4 * cubic_alpha

    def _gs_get_linear_coeffs(self, x, coeffs):
        x = abs(x)
        coeffs[0] = 1 - x
        coeffs[1] = x

    def _gs_bicubic_interpolate(self, p, x, y):  # type: ignore
        v = np.empty((4,), dtype=p.dtype)
        coeffs = np.empty((4,), dtype=p.dtype)
        self._gs_get_cubic_coeffs(x, coeffs)
        for i in range(4):
            v[i] = coeffs @ p[i, :]
        self._gs_get_cubic_coeffs(y, coeffs)
        return coeffs @ v

    def _gs_cubic_interpolation_1d_with_x(self, data, x, border, padding_mode):
        v = np.empty((4,), dtype=data.dtype)
        coeffs = np.empty((4,), dtype=data.dtype)
        x_0 = int(np.floor(x))
        x_1 = x_0 + 1
        x_2 = x_0 + 2
        x_minus_1 = x_0 - 1
        self._gs_get_cubic_coeffs(x - x_0, coeffs)
        v[0] = self._pixel_at_array(
            array=data, i=x_minus_1, border=border, padding_mode=padding_mode
        )
        v[1] = self._pixel_at_array(
            array=data, i=x_0, border=border, padding_mode=padding_mode
        )
        v[2] = self._pixel_at_array(
            array=data, i=x_1, border=border, padding_mode=padding_mode
        )
        v[3] = self._pixel_at_array(
            array=data, i=x_2, border=border, padding_mode=padding_mode
        )

        return coeffs @ v

    def _gs_linear_interpolation_1d_with_x(self, data, x, border, padding_mode):
        v = np.empty((2,), dtype=data.dtype)
        coeffs = np.empty((2,), dtype=data.dtype)
        x_0 = int(np.floor(x))
        x_1 = x_0 + 1
        self._gs_get_linear_coeffs(x - x_0, coeffs)
        v[0] = self._pixel_at_array(
            array=data, i=x_0, border=border, padding_mode=padding_mode
        )
        v[1] = self._pixel_at_array(
            array=data, i=x_1, border=border, padding_mode=padding_mode
        )

        return coeffs @ v

    def _gs_linear_interpolation_nd_with_x(self, data, x, border, padding_mode):
        num_dims = data.ndim
        assert num_dims == len(x) == int(len(border) / 2)
        if num_dims == 1:
            return self._gs_linear_interpolation_1d_with_x(
                data=data, x=x[0], border=border, padding_mode=padding_mode
            )

        res1d = []
        for i in range(data.shape[0]):
            r = self._gs_linear_interpolation_nd_with_x(
                data=data[i],
                x=x[1:],
                border=list(border[1:num_dims])
                + list(border[1 + num_dims : 2 * num_dims]),
                padding_mode=padding_mode,
            )
            res1d.append(r)
        res1d = np.array(res1d)

        return self._gs_linear_interpolation_1d_with_x(
            data=res1d,
            x=x[0],
            border=[border[0], border[num_dims]],
            padding_mode=padding_mode,
        )

    def _gs_cubic_interpolation_nd_with_x(self, data, x, border, padding_mode):
        num_dims = data.ndim
        assert num_dims == len(x) == int(len(border) / 2)
        if num_dims == 1:
            return self._gs_cubic_interpolation_1d_with_x(
                data=data, x=x[0], border=border, padding_mode=padding_mode
            )

        res1d = []
        for i in range(data.shape[0]):
            r = self._gs_cubic_interpolation_nd_with_x(
                data=data[i],
                x=x[1:],
                border=list(border[1:num_dims])
                + list(border[1 + num_dims : 2 * num_dims]),
                padding_mode=padding_mode,
            )
            res1d.append(r)
        res1d = np.array(res1d)

        return self._gs_cubic_interpolation_1d_with_x(
            data=res1d,
            x=x[0],
            border=[border[0], border[num_dims]],
            padding_mode=padding_mode,
        )

    def _clamp(self, val, lo, hi):  # type: ignore
        if val < lo:
            return lo
        if val > hi:
            return hi
        return val

    def _pixel_at_ndarray(self, ndarray, x: List, border, padding_mode):  # type: ignore
        # boarder: [x_1_min, x_2_min, ..., x_1_max, x_2_max, ...]
        num_dims = ndarray.ndim
        assert num_dims == len(x) == int(len(border) / 2)
        if num_dims == 1:
            return self._pixel_at_array(
                array=ndarray, i=x[0], border=border, padding_mode=padding_mode
            )
        i = x[0]
        d = ndarray.shape[0]
        if padding_mode == "zeros":
            if i >= 0 and i < d:
                ndarray = ndarray[i]
            else:
                # Trick
                i = 0
                ndarray = np.zeros_like(ndarray[i])
        elif padding_mode == "border":
            i = self._clamp(i, 0, d - 1)
            ndarray = ndarray[i]
        else:  # padding_mode == "reflection"
            i = int(self._gs_reflect(i, border[0], border[num_dims]))
            ndarray = ndarray[i]

        return self._pixel_at_ndarray(
            ndarray=ndarray,
            x=x[1:],
            border=list(border[1:num_dims]) + list(border[1 + num_dims : 2 * num_dims]),
            padding_mode=padding_mode,
        )

    def _pixel_at_array(self, array, i: int, border, padding_mode):  # type: ignore
        assert array.ndim == 1
        d = array.shape[0]
        if padding_mode == "zeros":
            if i >= 0 and i < d:
                pixel = array[i]
            else:
                pixel = 0
        elif padding_mode == "border":
            i = self._clamp(i, 0, d - 1)
            pixel = array[i]
        else:  # padding_mode == "reflection"
            i = int(self._gs_reflect(i, border[0], border[1]))
            pixel = array[i]
        return pixel

    def _prepare_border(self, dims, align_corners: bool):
        # boarder: [x_1_min, x_2_min, ..., x_1_max, x_2_max, ...]
        num_dims = len(dims)
        borders = np.zeros(num_dims * 2)
        for i in range(num_dims):
            # min
            borders[i] = -0.5
            # max
            borders[i + num_dims] = dims[i] - 0.5
            if align_corners:
                # min
                borders[i] = 0.0
                # max
                borders[i + num_dims] = dims[i] - 1.0

        return borders

    def _cpp_std_round(self, x):
        # https://en.cppreference.com/w/cpp/numeric/math/round
        def round_single_value(v):
            if v >= 0.0:
                return np.floor(v + 0.5)
            else:
                return np.ceil(v - 0.5)

        if isinstance(x, numbers.Number):
            return round_single_value(x)
        else:
            assert x.ndim == 1
            x_rounded = np.zeros_like(x)
            for i in range(x.shape[0]):
                x_rounded[i] = round_single_value(x[i])
            x_rounded = x_rounded.astype(np.int32)
            return x_rounded

    def _run(self, X, grid, mode=None, padding_mode=None, align_corners=None):
        # This implementation supports GridSample arbitrary dimensions.

        mode = mode or self.mode  # type: ignore
        padding_mode = padding_mode or self.padding_mode  # type: ignore
        align_corners = align_corners or self.align_corners  # type: ignore

        x_dims = X.shape
        grid_dims = grid.shape
        N = x_dims[0]
        C = x_dims[1]
        y_dims = (N, C, *grid_dims[1:-1])

        if np.prod(y_dims) == 0:
            return np.array([], dtype=X.dtype)

        Y = np.empty(y_dims, dtype=X.dtype)

        for n in range(N):
            grid_data = grid[n]

            for c in range(C):
                # Because the indices in the grid_data are always in the "reverse" dimensional order.
                # To interpolate for certain positions, we either have to transpose the X_data or
                # reverse the indices.
                # In this implementation, we took the latter approach.
                X_data = X[n, c]

                num_dims = len(x_dims[2:])
                dims = x_dims[2:]

                # Prepare borders.
                border = self._prepare_border(dims, align_corners=align_corners)

                for ox in _get_all_coords(Y[n, c]):
                    # normalized coordinates.
                    nx = grid_data[tuple(ox)]
                    nx = nx[::-1]
                    # denormalized coordinates.
                    x = self._gs_denormalize_coordinates(
                        n=nx, dims=dims, align_corners=align_corners
                    )
                    if mode == "nearest":
                        # PyTorch round the index to nearest even.
                        # https://github.com/pytorch/pytorch/pull/97000
                        x = np.rint(x)
                    # https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/GridSampler.h#L142
                    for i, v in enumerate(x):
                        x_min = border[i]
                        x_max = border[i + num_dims]
                        if v < x_min or v > x_max:
                            if padding_mode == "border":
                                x[i] = self._clamp(v, 0, dims[i] - 1)
                            elif padding_mode == "reflection":
                                x[i] = self._gs_reflect(v, x_min, x_max)

                    if mode == "nearest":
                        x = x.astype(np.int32)
                        Y[n][c][tuple(ox)] = self._pixel_at_ndarray(
                            ndarray=X_data,
                            x=x,
                            border=border,
                            padding_mode=padding_mode,
                        )

                    elif mode == "linear":
                        Y[n][c][tuple(ox)] = self._gs_linear_interpolation_nd_with_x(
                            data=X_data, x=x, border=border, padding_mode=padding_mode
                        )

                    elif mode == "cubic":
                        Y[n][c][tuple(ox)] = self._gs_cubic_interpolation_nd_with_x(
                            data=X_data, x=x, border=border, padding_mode=padding_mode
                        )
                    else:
                        raise RuntimeError(
                            "GridSample interpolation only supports nearest, linear, and cubic modes."
                        )

        return (Y.astype(X.dtype),)
