from __future__ import annotations

from contourpy._contourpy import FillType, LineType, ZInterp


def as_fill_type(fill_type: FillType | str) -> FillType:
    """Coerce a FillType or string value to a FillType.

    Args:
        fill_type (FillType or str): Value to convert.

    Return:
        FillType: Converted value.
    """
    if isinstance(fill_type, str):
        return FillType.__members__[fill_type]
    else:
        return fill_type


def as_line_type(line_type: LineType | str) -> LineType:
    """Coerce a LineType or string value to a LineType.

    Args:
        line_type (LineType or str): Value to convert.

    Return:
        LineType: Converted value.
    """
    if isinstance(line_type, str):
        return LineType.__members__[line_type]
    else:
        return line_type


def as_z_interp(z_interp: ZInterp | str) -> ZInterp:
    """Coerce a ZInterp or string value to a ZInterp.

    Args:
        z_interp (ZInterp or str): Value to convert.

    Return:
        ZInterp: Converted value.
    """
    if isinstance(z_interp, str):
        return ZInterp.__members__[z_interp]
    else:
        return z_interp
