# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,R0914,R0915,R1702,R1716,W0221

import numpy as np

from onnx.reference.op_run import OpRun


class GridSample(OpRun):
    def _gs_denormalize(self, n, length: int, align_corners: bool):  # type: ignore
        if align_corners:
            x = (n + 1) / 2.0 * (length - 1)
        else:
            x = ((n + 1) * length - 1) / 2.0
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

    def _gs_bicubic_interpolate(self, p, x, y):  # type: ignore
        v = np.empty((4,), dtype=p.dtype)
        coeffs = np.empty((4,), dtype=p.dtype)
        self._gs_get_cubic_coeffs(x, coeffs)
        for i in range(4):
            v[i] = coeffs @ p[i, :]
        self._gs_get_cubic_coeffs(y, coeffs)
        return coeffs @ v

    def _clamp(self, val, lo, hi):  # type: ignore
        if val < lo:
            return lo
        if val > hi:
            return hi
        return val

    def _pixel_at_grid(self, image, r: int, c: int, border, padding_mode):  # type: ignore
        H, W = image.shape
        if padding_mode == "zeros":
            if c >= 0 and c < W and r >= 0 and r < H:
                pixel = image[r, c]  # image[r * W + c]
            else:
                pixel = 0
        elif padding_mode == "border":
            c = self._clamp(c, 0, W - 1)
            r = self._clamp(r, 0, H - 1)
            pixel = image[r, c]
        else:  # padding_mode == "reflection"
            c = int(self._gs_reflect(c, border[0], border[2]))
            r = int(self._gs_reflect(r, border[1], border[3]))
            pixel = image[r, c]  # image[r * W + c]
        return pixel

    def _run(  # type: ignore
        self, X, grid, mode=None, padding_mode=None, align_corners=None
    ):
        mode = mode or self.mode  # type: ignore
        padding_mode = padding_mode or self.padding_mode  # type: ignore
        align_corners = align_corners or self.align_corners  # type: ignore

        x_dims = X.shape
        grid_dims = grid.shape

        if len(x_dims) != 4 or len(grid_dims) != 4:
            raise RuntimeError(
                f"X and grid must be 4D tensors not {len(x_dims)} or {len(grid_dims)}."
            )

        N = x_dims[0]
        C = x_dims[1]
        H_in = x_dims[2]
        W_in = x_dims[3]
        H_out = grid_dims[1]
        W_out = grid_dims[2]

        y_dims = (N, C, H_out, W_out)
        size = N * C * H_out * W_out
        if size == 0:
            return np.array([], dtype=X.dtype)

        Y = np.empty(y_dims, dtype=X.dtype)

        # Force float here to avoid possible issue in integer T case
        x_min = -0.5
        x_max = W_in - 0.5
        y_min = -0.5
        y_max = H_in - 0.5

        if align_corners:
            x_min = 0.0
            x_max = W_in - 1.0
            y_min = 0.0
            y_max = H_in - 1.0

        border = (x_min, y_min, x_max, y_max)

        for n in range(N):
            grid_data = grid[n]

            for c in range(C):
                X_data = X[n, c]

                for oy in range(H_out):
                    for ox in range(W_out):
                        gridpoint = grid_data[oy, ox]

                        nx = gridpoint[0]  # normalized location
                        ny = gridpoint[1]
                        x = self._gs_denormalize(
                            nx, W_in, align_corners
                        )  # actual location
                        y = self._gs_denormalize(ny, H_in, align_corners)

                        if mode == "nearest":
                            x = np.rint(x)  # nearbyintf(x)
                            y = np.rint(y)  # nearbyintf(y)

                        if (
                            x < x_min or x > x_max or y < y_min or y > y_max
                        ):  # out of bound
                            if padding_mode == "border":
                                # use original border in both align_corner cases
                                x = self._clamp(x, 0, W_in - 1)
                                y = self._clamp(y, 0, H_in - 1)
                            elif padding_mode == "reflection":
                                x = self._gs_reflect(x, x_min, x_max)
                                y = self._gs_reflect(y, y_min, y_max)

                        if mode == "nearest":
                            # x, y are integers in all padding modes
                            Y[n, c, oy, ox] = self._pixel_at_grid(
                                X_data, int(y), int(x), border, padding_mode
                            )
                            continue

                        if mode == "bilinear":
                            x1 = int(np.floor(x))
                            y1 = int(np.floor(y))
                            x2 = x1 + 1
                            y2 = y1 + 1

                            p11 = self._pixel_at_grid(
                                X_data, y1, x1, border, padding_mode
                            )
                            p12 = self._pixel_at_grid(
                                X_data, y1, x2, border, padding_mode
                            )
                            p21 = self._pixel_at_grid(
                                X_data, y2, x1, border, padding_mode
                            )
                            p22 = self._pixel_at_grid(
                                X_data, y2, x2, border, padding_mode
                            )

                            dx2 = x2 - x
                            dx1 = x - x1
                            dy2 = y2 - y
                            dy1 = y - y1
                            Y[n, c, oy, ox] = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (
                                dx2 * p21 + dx1 * p22
                            )

                        if mode == "bicubic":
                            x0 = int(np.floor(x)) - 1  # top-left corner of the bbox
                            y0 = int(np.floor(y)) - 1
                            p = np.empty((4, 4), dtype=X.dtype)
                            for h in range(4):
                                for w in range(4):
                                    p[h, w] = self._pixel_at_grid(
                                        X_data, h + y0, w + x0, border, padding_mode
                                    )
                            dx = x - x0 - 1
                            dy = y - y0 - 1
                            Y[n, c, oy, ox] = self._gs_bicubic_interpolate(p, dx, dy)
        return (Y.astype(X.dtype),)
