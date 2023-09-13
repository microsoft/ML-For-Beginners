#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2009-2014, Mario Vilas
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice,this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Wrapper for gdi32.dll in ctypes.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

#--- Helpers ------------------------------------------------------------------

#--- Types --------------------------------------------------------------------

#--- Constants ----------------------------------------------------------------

# GDI object types
OBJ_PEN             = 1
OBJ_BRUSH           = 2
OBJ_DC              = 3
OBJ_METADC          = 4
OBJ_PAL             = 5
OBJ_FONT            = 6
OBJ_BITMAP          = 7
OBJ_REGION          = 8
OBJ_METAFILE        = 9
OBJ_MEMDC           = 10
OBJ_EXTPEN          = 11
OBJ_ENHMETADC       = 12
OBJ_ENHMETAFILE     = 13
OBJ_COLORSPACE      = 14
GDI_OBJ_LAST        = OBJ_COLORSPACE

# Ternary raster operations
SRCCOPY         = 0x00CC0020 # dest = source
SRCPAINT        = 0x00EE0086 # dest = source OR dest
SRCAND          = 0x008800C6 # dest = source AND dest
SRCINVERT       = 0x00660046 # dest = source XOR dest
SRCERASE        = 0x00440328 # dest = source AND (NOT dest)
NOTSRCCOPY      = 0x00330008 # dest = (NOT source)
NOTSRCERASE     = 0x001100A6 # dest = (NOT src) AND (NOT dest)
MERGECOPY       = 0x00C000CA # dest = (source AND pattern)
MERGEPAINT      = 0x00BB0226 # dest = (NOT source) OR dest
PATCOPY         = 0x00F00021 # dest = pattern
PATPAINT        = 0x00FB0A09 # dest = DPSnoo
PATINVERT       = 0x005A0049 # dest = pattern XOR dest
DSTINVERT       = 0x00550009 # dest = (NOT dest)
BLACKNESS       = 0x00000042 # dest = BLACK
WHITENESS       = 0x00FF0062 # dest = WHITE
NOMIRRORBITMAP  = 0x80000000 # Do not Mirror the bitmap in this call
CAPTUREBLT      = 0x40000000 # Include layered windows

# Region flags
ERROR               = 0
NULLREGION          = 1
SIMPLEREGION        = 2
COMPLEXREGION       = 3
RGN_ERROR           = ERROR

# CombineRgn() styles
RGN_AND             = 1
RGN_OR              = 2
RGN_XOR             = 3
RGN_DIFF            = 4
RGN_COPY            = 5
RGN_MIN             = RGN_AND
RGN_MAX             = RGN_COPY

# StretchBlt() modes
BLACKONWHITE        = 1
WHITEONBLACK        = 2
COLORONCOLOR        = 3
HALFTONE            = 4
MAXSTRETCHBLTMODE   = 4
STRETCH_ANDSCANS    = BLACKONWHITE
STRETCH_ORSCANS     = WHITEONBLACK
STRETCH_DELETESCANS = COLORONCOLOR
STRETCH_HALFTONE    = HALFTONE

# PolyFill() modes
ALTERNATE       = 1
WINDING         = 2
POLYFILL_LAST   = 2

# Layout orientation options
LAYOUT_RTL                         = 0x00000001 # Right to left
LAYOUT_BTT                         = 0x00000002 # Bottom to top
LAYOUT_VBH                         = 0x00000004 # Vertical before horizontal
LAYOUT_ORIENTATIONMASK             = LAYOUT_RTL + LAYOUT_BTT + LAYOUT_VBH
LAYOUT_BITMAPORIENTATIONPRESERVED  = 0x00000008

# Stock objects
WHITE_BRUSH         = 0
LTGRAY_BRUSH        = 1
GRAY_BRUSH          = 2
DKGRAY_BRUSH        = 3
BLACK_BRUSH         = 4
NULL_BRUSH          = 5
HOLLOW_BRUSH        = NULL_BRUSH
WHITE_PEN           = 6
BLACK_PEN           = 7
NULL_PEN            = 8
OEM_FIXED_FONT      = 10
ANSI_FIXED_FONT     = 11
ANSI_VAR_FONT       = 12
SYSTEM_FONT         = 13
DEVICE_DEFAULT_FONT = 14
DEFAULT_PALETTE     = 15
SYSTEM_FIXED_FONT   = 16

# Metafile functions
META_SETBKCOLOR              = 0x0201
META_SETBKMODE               = 0x0102
META_SETMAPMODE              = 0x0103
META_SETROP2                 = 0x0104
META_SETRELABS               = 0x0105
META_SETPOLYFILLMODE         = 0x0106
META_SETSTRETCHBLTMODE       = 0x0107
META_SETTEXTCHAREXTRA        = 0x0108
META_SETTEXTCOLOR            = 0x0209
META_SETTEXTJUSTIFICATION    = 0x020A
META_SETWINDOWORG            = 0x020B
META_SETWINDOWEXT            = 0x020C
META_SETVIEWPORTORG          = 0x020D
META_SETVIEWPORTEXT          = 0x020E
META_OFFSETWINDOWORG         = 0x020F
META_SCALEWINDOWEXT          = 0x0410
META_OFFSETVIEWPORTORG       = 0x0211
META_SCALEVIEWPORTEXT        = 0x0412
META_LINETO                  = 0x0213
META_MOVETO                  = 0x0214
META_EXCLUDECLIPRECT         = 0x0415
META_INTERSECTCLIPRECT       = 0x0416
META_ARC                     = 0x0817
META_ELLIPSE                 = 0x0418
META_FLOODFILL               = 0x0419
META_PIE                     = 0x081A
META_RECTANGLE               = 0x041B
META_ROUNDRECT               = 0x061C
META_PATBLT                  = 0x061D
META_SAVEDC                  = 0x001E
META_SETPIXEL                = 0x041F
META_OFFSETCLIPRGN           = 0x0220
META_TEXTOUT                 = 0x0521
META_BITBLT                  = 0x0922
META_STRETCHBLT              = 0x0B23
META_POLYGON                 = 0x0324
META_POLYLINE                = 0x0325
META_ESCAPE                  = 0x0626
META_RESTOREDC               = 0x0127
META_FILLREGION              = 0x0228
META_FRAMEREGION             = 0x0429
META_INVERTREGION            = 0x012A
META_PAINTREGION             = 0x012B
META_SELECTCLIPREGION        = 0x012C
META_SELECTOBJECT            = 0x012D
META_SETTEXTALIGN            = 0x012E
META_CHORD                   = 0x0830
META_SETMAPPERFLAGS          = 0x0231
META_EXTTEXTOUT              = 0x0a32
META_SETDIBTODEV             = 0x0d33
META_SELECTPALETTE           = 0x0234
META_REALIZEPALETTE          = 0x0035
META_ANIMATEPALETTE          = 0x0436
META_SETPALENTRIES           = 0x0037
META_POLYPOLYGON             = 0x0538
META_RESIZEPALETTE           = 0x0139
META_DIBBITBLT               = 0x0940
META_DIBSTRETCHBLT           = 0x0b41
META_DIBCREATEPATTERNBRUSH   = 0x0142
META_STRETCHDIB              = 0x0f43
META_EXTFLOODFILL            = 0x0548
META_SETLAYOUT               = 0x0149
META_DELETEOBJECT            = 0x01f0
META_CREATEPALETTE           = 0x00f7
META_CREATEPATTERNBRUSH      = 0x01F9
META_CREATEPENINDIRECT       = 0x02FA
META_CREATEFONTINDIRECT      = 0x02FB
META_CREATEBRUSHINDIRECT     = 0x02FC
META_CREATEREGION            = 0x06FF

# Metafile escape codes
NEWFRAME                     = 1
ABORTDOC                     = 2
NEXTBAND                     = 3
SETCOLORTABLE                = 4
GETCOLORTABLE                = 5
FLUSHOUTPUT                  = 6
DRAFTMODE                    = 7
QUERYESCSUPPORT              = 8
SETABORTPROC                 = 9
STARTDOC                     = 10
ENDDOC                       = 11
GETPHYSPAGESIZE              = 12
GETPRINTINGOFFSET            = 13
GETSCALINGFACTOR             = 14
MFCOMMENT                    = 15
GETPENWIDTH                  = 16
SETCOPYCOUNT                 = 17
SELECTPAPERSOURCE            = 18
DEVICEDATA                   = 19
PASSTHROUGH                  = 19
GETTECHNOLGY                 = 20
GETTECHNOLOGY                = 20
SETLINECAP                   = 21
SETLINEJOIN                  = 22
SETMITERLIMIT                = 23
BANDINFO                     = 24
DRAWPATTERNRECT              = 25
GETVECTORPENSIZE             = 26
GETVECTORBRUSHSIZE           = 27
ENABLEDUPLEX                 = 28
GETSETPAPERBINS              = 29
GETSETPRINTORIENT            = 30
ENUMPAPERBINS                = 31
SETDIBSCALING                = 32
EPSPRINTING                  = 33
ENUMPAPERMETRICS             = 34
GETSETPAPERMETRICS           = 35
POSTSCRIPT_DATA              = 37
POSTSCRIPT_IGNORE            = 38
MOUSETRAILS                  = 39
GETDEVICEUNITS               = 42
GETEXTENDEDTEXTMETRICS       = 256
GETEXTENTTABLE               = 257
GETPAIRKERNTABLE             = 258
GETTRACKKERNTABLE            = 259
EXTTEXTOUT                   = 512
GETFACENAME                  = 513
DOWNLOADFACE                 = 514
ENABLERELATIVEWIDTHS         = 768
ENABLEPAIRKERNING            = 769
SETKERNTRACK                 = 770
SETALLJUSTVALUES             = 771
SETCHARSET                   = 772
STRETCHBLT                   = 2048
METAFILE_DRIVER              = 2049
GETSETSCREENPARAMS           = 3072
QUERYDIBSUPPORT              = 3073
BEGIN_PATH                   = 4096
CLIP_TO_PATH                 = 4097
END_PATH                     = 4098
EXT_DEVICE_CAPS              = 4099
RESTORE_CTM                  = 4100
SAVE_CTM                     = 4101
SET_ARC_DIRECTION            = 4102
SET_BACKGROUND_COLOR         = 4103
SET_POLY_MODE                = 4104
SET_SCREEN_ANGLE             = 4105
SET_SPREAD                   = 4106
TRANSFORM_CTM                = 4107
SET_CLIP_BOX                 = 4108
SET_BOUNDS                   = 4109
SET_MIRROR_MODE              = 4110
OPENCHANNEL                  = 4110
DOWNLOADHEADER               = 4111
CLOSECHANNEL                 = 4112
POSTSCRIPT_PASSTHROUGH       = 4115
ENCAPSULATED_POSTSCRIPT      = 4116
POSTSCRIPT_IDENTIFY          = 4117
POSTSCRIPT_INJECTION         = 4118
CHECKJPEGFORMAT              = 4119
CHECKPNGFORMAT               = 4120
GET_PS_FEATURESETTING        = 4121
GDIPLUS_TS_QUERYVER          = 4122
GDIPLUS_TS_RECORD            = 4123
SPCLPASSTHROUGH2             = 4568

#--- Structures ---------------------------------------------------------------

# typedef struct _RECT {
#   LONG left;
#   LONG top;
#   LONG right;
#   LONG bottom;
# }RECT, *PRECT;
class RECT(Structure):
    _fields_ = [
        ('left',    LONG),
        ('top',     LONG),
        ('right',   LONG),
        ('bottom',  LONG),
    ]
PRECT  = POINTER(RECT)
LPRECT = PRECT

# typedef struct tagPOINT {
#   LONG x;
#   LONG y;
# } POINT;
class POINT(Structure):
    _fields_ = [
        ('x',   LONG),
        ('y',   LONG),
    ]
PPOINT  = POINTER(POINT)
LPPOINT = PPOINT

# typedef struct tagBITMAP {
#   LONG   bmType;
#   LONG   bmWidth;
#   LONG   bmHeight;
#   LONG   bmWidthBytes;
#   WORD   bmPlanes;
#   WORD   bmBitsPixel;
#   LPVOID bmBits;
# } BITMAP, *PBITMAP;
class BITMAP(Structure):
    _fields_ = [
        ("bmType",          LONG),
        ("bmWidth",         LONG),
        ("bmHeight",        LONG),
        ("bmWidthBytes",    LONG),
        ("bmPlanes",        WORD),
        ("bmBitsPixel",     WORD),
        ("bmBits",          LPVOID),
    ]
PBITMAP  = POINTER(BITMAP)
LPBITMAP = PBITMAP

#--- High level classes -------------------------------------------------------

#--- gdi32.dll ----------------------------------------------------------------

# HDC GetDC(
#   __in  HWND hWnd
# );
def GetDC(hWnd):
    _GetDC = windll.gdi32.GetDC
    _GetDC.argtypes = [HWND]
    _GetDC.restype  = HDC
    _GetDC.errcheck = RaiseIfZero
    return _GetDC(hWnd)

# HDC GetWindowDC(
#   __in  HWND hWnd
# );
def GetWindowDC(hWnd):
    _GetWindowDC = windll.gdi32.GetWindowDC
    _GetWindowDC.argtypes = [HWND]
    _GetWindowDC.restype  = HDC
    _GetWindowDC.errcheck = RaiseIfZero
    return _GetWindowDC(hWnd)

# int ReleaseDC(
#   __in  HWND hWnd,
#   __in  HDC hDC
# );
def ReleaseDC(hWnd, hDC):
    _ReleaseDC = windll.gdi32.ReleaseDC
    _ReleaseDC.argtypes = [HWND, HDC]
    _ReleaseDC.restype  = ctypes.c_int
    _ReleaseDC.errcheck = RaiseIfZero
    _ReleaseDC(hWnd, hDC)

# HGDIOBJ SelectObject(
#   __in  HDC hdc,
#   __in  HGDIOBJ hgdiobj
# );
def SelectObject(hdc, hgdiobj):
    _SelectObject = windll.gdi32.SelectObject
    _SelectObject.argtypes = [HDC, HGDIOBJ]
    _SelectObject.restype  = HGDIOBJ
    _SelectObject.errcheck = RaiseIfZero
    return _SelectObject(hdc, hgdiobj)

# HGDIOBJ GetStockObject(
#   __in  int fnObject
# );
def GetStockObject(fnObject):
    _GetStockObject = windll.gdi32.GetStockObject
    _GetStockObject.argtypes = [ctypes.c_int]
    _GetStockObject.restype  = HGDIOBJ
    _GetStockObject.errcheck = RaiseIfZero
    return _GetStockObject(fnObject)

# DWORD GetObjectType(
#   __in  HGDIOBJ h
# );
def GetObjectType(h):
    _GetObjectType = windll.gdi32.GetObjectType
    _GetObjectType.argtypes = [HGDIOBJ]
    _GetObjectType.restype  = DWORD
    _GetObjectType.errcheck = RaiseIfZero
    return _GetObjectType(h)

# int GetObject(
#   __in   HGDIOBJ hgdiobj,
#   __in   int cbBuffer,
#   __out  LPVOID lpvObject
# );
def GetObject(hgdiobj, cbBuffer = None, lpvObject = None):
    _GetObject = windll.gdi32.GetObject
    _GetObject.argtypes = [HGDIOBJ, ctypes.c_int, LPVOID]
    _GetObject.restype  = ctypes.c_int
    _GetObject.errcheck = RaiseIfZero

    # Both cbBuffer and lpvObject can be omitted, the correct
    # size and structure to return are automatically deduced.
    # If lpvObject is given it must be a ctypes object, not a pointer.
    # Always returns a ctypes object.

    if cbBuffer is not None:
        if lpvObject is None:
            lpvObject = ctypes.create_string_buffer("", cbBuffer)
    elif lpvObject is not None:
        cbBuffer = sizeof(lpvObject)
    else: # most likely case, both are None
        t = GetObjectType(hgdiobj)
        if   t == OBJ_PEN:
            cbBuffer  = sizeof(LOGPEN)
            lpvObject = LOGPEN()
        elif t == OBJ_BRUSH:
            cbBuffer  = sizeof(LOGBRUSH)
            lpvObject = LOGBRUSH()
        elif t == OBJ_PAL:
            cbBuffer  = _GetObject(hgdiobj, 0, None)
            lpvObject = (WORD * (cbBuffer // sizeof(WORD)))()
        elif t == OBJ_FONT:
            cbBuffer  = sizeof(LOGFONT)
            lpvObject = LOGFONT()
        elif t == OBJ_BITMAP:  # try the two possible types of bitmap
            cbBuffer  = sizeof(DIBSECTION)
            lpvObject = DIBSECTION()
            try:
                _GetObject(hgdiobj, cbBuffer, byref(lpvObject))
                return lpvObject
            except WindowsError:
                cbBuffer  = sizeof(BITMAP)
                lpvObject = BITMAP()
        elif t == OBJ_EXTPEN:
            cbBuffer  = sizeof(LOGEXTPEN)
            lpvObject = LOGEXTPEN()
        else:
            cbBuffer  = _GetObject(hgdiobj, 0, None)
            lpvObject = ctypes.create_string_buffer("", cbBuffer)
    _GetObject(hgdiobj, cbBuffer, byref(lpvObject))
    return lpvObject

# LONG GetBitmapBits(
#   __in   HBITMAP hbmp,
#   __in   LONG cbBuffer,
#   __out  LPVOID lpvBits
# );
def GetBitmapBits(hbmp):
    _GetBitmapBits = windll.gdi32.GetBitmapBits
    _GetBitmapBits.argtypes = [HBITMAP, LONG, LPVOID]
    _GetBitmapBits.restype  = LONG
    _GetBitmapBits.errcheck = RaiseIfZero

    bitmap   = GetObject(hbmp, lpvObject = BITMAP())
    cbBuffer = bitmap.bmWidthBytes * bitmap.bmHeight
    lpvBits  = ctypes.create_string_buffer("", cbBuffer)
    _GetBitmapBits(hbmp, cbBuffer, byref(lpvBits))
    return lpvBits.raw

# HBITMAP CreateBitmapIndirect(
#   __in  const BITMAP *lpbm
# );
def CreateBitmapIndirect(lpbm):
    _CreateBitmapIndirect = windll.gdi32.CreateBitmapIndirect
    _CreateBitmapIndirect.argtypes = [PBITMAP]
    _CreateBitmapIndirect.restype  = HBITMAP
    _CreateBitmapIndirect.errcheck = RaiseIfZero
    return _CreateBitmapIndirect(lpbm)

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
