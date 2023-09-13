from __future__ import annotations

from typing import TYPE_CHECKING, cast

from contourpy import FillType, LineType
from contourpy.util.mpl_util import mpl_codes_to_offsets

if TYPE_CHECKING:
    from contourpy._contourpy import (
        CoordinateArray, FillReturn, LineReturn, LineReturn_Separate, LineReturn_SeparateCode,
    )


def filled_to_bokeh(
    filled: FillReturn,
    fill_type: FillType,
) -> tuple[list[list[CoordinateArray]], list[list[CoordinateArray]]]:
    xs: list[list[CoordinateArray]] = []
    ys: list[list[CoordinateArray]] = []
    if fill_type in (FillType.OuterOffset, FillType.ChunkCombinedOffset,
                     FillType.OuterCode, FillType.ChunkCombinedCode):
        have_codes = fill_type in (FillType.OuterCode, FillType.ChunkCombinedCode)

        for points, offsets in zip(*filled):
            if points is None:
                continue
            if have_codes:
                offsets = mpl_codes_to_offsets(offsets)
            xs.append([])  # New outer with zero or more holes.
            ys.append([])
            for i in range(len(offsets)-1):
                xys = points[offsets[i]:offsets[i+1]]
                xs[-1].append(xys[:, 0])
                ys[-1].append(xys[:, 1])
    elif fill_type in (FillType.ChunkCombinedCodeOffset, FillType.ChunkCombinedOffsetOffset):
        for points, codes_or_offsets, outer_offsets in zip(*filled):
            if points is None:
                continue
            for j in range(len(outer_offsets)-1):
                if fill_type == FillType.ChunkCombinedCodeOffset:
                    codes = codes_or_offsets[outer_offsets[j]:outer_offsets[j+1]]
                    offsets = mpl_codes_to_offsets(codes) + outer_offsets[j]
                else:
                    offsets = codes_or_offsets[outer_offsets[j]:outer_offsets[j+1]+1]
                xs.append([])  # New outer with zero or more holes.
                ys.append([])
                for k in range(len(offsets)-1):
                    xys = points[offsets[k]:offsets[k+1]]
                    xs[-1].append(xys[:, 0])
                    ys[-1].append(xys[:, 1])
    else:
        raise RuntimeError(f"Conversion of FillType {fill_type} to Bokeh is not implemented")

    return xs, ys


def lines_to_bokeh(
    lines: LineReturn,
    line_type: LineType,
) -> tuple[list[CoordinateArray], list[CoordinateArray]]:
    xs: list[CoordinateArray] = []
    ys: list[CoordinateArray] = []

    if line_type == LineType.Separate:
        if TYPE_CHECKING:
            lines = cast(LineReturn_Separate, lines)
        for line in lines:
            xs.append(line[:, 0])
            ys.append(line[:, 1])
    elif line_type == LineType.SeparateCode:
        if TYPE_CHECKING:
            lines = cast(LineReturn_SeparateCode, lines)
        for line in lines[0]:
            xs.append(line[:, 0])
            ys.append(line[:, 1])
    elif line_type in (LineType.ChunkCombinedCode, LineType.ChunkCombinedOffset):
        for points, offsets in zip(*lines):
            if points is None:
                continue
            if line_type == LineType.ChunkCombinedCode:
                offsets = mpl_codes_to_offsets(offsets)

            for i in range(len(offsets)-1):
                line = points[offsets[i]:offsets[i+1]]
                xs.append(line[:, 0])
                ys.append(line[:, 1])
    else:
        raise RuntimeError(f"Conversion of LineType {line_type} to Bokeh is not implemented")

    return xs, ys
