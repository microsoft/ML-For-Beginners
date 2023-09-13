from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.path as mpath
import numpy as np

from contourpy import FillType, LineType

if TYPE_CHECKING:
    from contourpy._contourpy import (
        CodeArray, FillReturn, LineReturn, LineReturn_Separate, OffsetArray,
    )


def filled_to_mpl_paths(filled: FillReturn, fill_type: FillType) -> list[mpath.Path]:
    if fill_type in (FillType.OuterCode, FillType.ChunkCombinedCode):
        paths = [mpath.Path(points, codes) for points, codes in zip(*filled) if points is not None]
    elif fill_type in (FillType.OuterOffset, FillType.ChunkCombinedOffset):
        paths = [mpath.Path(points, offsets_to_mpl_codes(offsets))
                 for points, offsets in zip(*filled) if points is not None]
    elif fill_type == FillType.ChunkCombinedCodeOffset:
        paths = []
        for points, codes, outer_offsets in zip(*filled):
            if points is None:
                continue
            points = np.split(points, outer_offsets[1:-1])
            codes = np.split(codes, outer_offsets[1:-1])
            paths += [mpath.Path(p, c) for p, c in zip(points, codes)]
    elif fill_type == FillType.ChunkCombinedOffsetOffset:
        paths = []
        for points, offsets, outer_offsets in zip(*filled):
            if points is None:
                continue
            for i in range(len(outer_offsets)-1):
                offs = offsets[outer_offsets[i]:outer_offsets[i+1]+1]
                pts = points[offs[0]:offs[-1]]
                paths += [mpath.Path(pts, offsets_to_mpl_codes(offs - offs[0]))]
    else:
        raise RuntimeError(f"Conversion of FillType {fill_type} to MPL Paths is not implemented")
    return paths


def lines_to_mpl_paths(lines: LineReturn, line_type: LineType) -> list[mpath.Path]:
    if line_type == LineType.Separate:
        if TYPE_CHECKING:
            lines = cast(LineReturn_Separate, lines)
        paths = []
        for line in lines:
            # Drawing as Paths so that they can be closed correctly.
            closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
            paths.append(mpath.Path(line, closed=closed))
    elif line_type in (LineType.SeparateCode, LineType.ChunkCombinedCode):
        paths = [mpath.Path(points, codes) for points, codes in zip(*lines) if points is not None]
    elif line_type == LineType.ChunkCombinedOffset:
        paths = []
        for points, offsets in zip(*lines):
            if points is None:
                continue
            for i in range(len(offsets)-1):
                line = points[offsets[i]:offsets[i+1]]
                closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
                paths.append(mpath.Path(line, closed=closed))
    else:
        raise RuntimeError(f"Conversion of LineType {line_type} to MPL Paths is not implemented")
    return paths


def mpl_codes_to_offsets(codes: CodeArray) -> OffsetArray:
    offsets = np.nonzero(codes == 1)[0].astype(np.uint32)
    offsets = np.append(offsets, len(codes))
    return offsets


def offsets_to_mpl_codes(offsets: OffsetArray) -> CodeArray:
    codes = np.full(offsets[-1]-offsets[0], 2, dtype=np.uint8)  # LINETO = 2
    codes[offsets[:-1]] = 1  # MOVETO = 1
    codes[offsets[1:]-1] = 79  # CLOSEPOLY 79
    return codes
