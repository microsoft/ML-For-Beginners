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
Wrapper for user32.dll in ctypes.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

#--- Helpers ------------------------------------------------------------------

def MAKE_WPARAM(wParam):
    """
    Convert arguments to the WPARAM type.
    Used automatically by SendMessage, PostMessage, etc.
    You shouldn't need to call this function.
    """
    wParam = ctypes.cast(wParam, LPVOID).value
    if wParam is None:
        wParam = 0
    return wParam

def MAKE_LPARAM(lParam):
    """
    Convert arguments to the LPARAM type.
    Used automatically by SendMessage, PostMessage, etc.
    You shouldn't need to call this function.
    """
    return ctypes.cast(lParam, LPARAM)

class __WindowEnumerator (object):
    """
    Window enumerator class. Used internally by the window enumeration APIs.
    """
    def __init__(self):
        self.hwnd = list()
    def __call__(self, hwnd, lParam):
##        print hwnd  # XXX DEBUG
        self.hwnd.append(hwnd)
        return TRUE

#--- Types --------------------------------------------------------------------

WNDENUMPROC = WINFUNCTYPE(BOOL, HWND, PVOID)

#--- Constants ----------------------------------------------------------------

HWND_DESKTOP    = 0
HWND_TOP        = 1
HWND_BOTTOM     = 1
HWND_TOPMOST    = -1
HWND_NOTOPMOST  = -2
HWND_MESSAGE    = -3

# GetWindowLong / SetWindowLong
GWL_WNDPROC                          = -4
GWL_HINSTANCE                        = -6
GWL_HWNDPARENT                       = -8
GWL_ID                               = -12
GWL_STYLE                            = -16
GWL_EXSTYLE                          = -20
GWL_USERDATA                         = -21

# GetWindowLongPtr / SetWindowLongPtr
GWLP_WNDPROC                         = GWL_WNDPROC
GWLP_HINSTANCE                       = GWL_HINSTANCE
GWLP_HWNDPARENT                      = GWL_HWNDPARENT
GWLP_STYLE                           = GWL_STYLE
GWLP_EXSTYLE                         = GWL_EXSTYLE
GWLP_USERDATA                        = GWL_USERDATA
GWLP_ID                              = GWL_ID

# ShowWindow
SW_HIDE                             = 0
SW_SHOWNORMAL                       = 1
SW_NORMAL                           = 1
SW_SHOWMINIMIZED                    = 2
SW_SHOWMAXIMIZED                    = 3
SW_MAXIMIZE                         = 3
SW_SHOWNOACTIVATE                   = 4
SW_SHOW                             = 5
SW_MINIMIZE                         = 6
SW_SHOWMINNOACTIVE                  = 7
SW_SHOWNA                           = 8
SW_RESTORE                          = 9
SW_SHOWDEFAULT                      = 10
SW_FORCEMINIMIZE                    = 11

# SendMessageTimeout flags
SMTO_NORMAL                         = 0
SMTO_BLOCK                          = 1
SMTO_ABORTIFHUNG                    = 2
SMTO_NOTIMEOUTIFNOTHUNG 			= 8
SMTO_ERRORONEXIT                    = 0x20

# WINDOWPLACEMENT flags
WPF_SETMINPOSITION                  = 1
WPF_RESTORETOMAXIMIZED              = 2
WPF_ASYNCWINDOWPLACEMENT            = 4

# GetAncestor flags
GA_PARENT                           = 1
GA_ROOT                             = 2
GA_ROOTOWNER                        = 3

# GetWindow flags
GW_HWNDFIRST                        = 0
GW_HWNDLAST                         = 1
GW_HWNDNEXT                         = 2
GW_HWNDPREV                         = 3
GW_OWNER                            = 4
GW_CHILD                            = 5
GW_ENABLEDPOPUP                     = 6

#--- Window messages ----------------------------------------------------------

WM_USER                              = 0x400
WM_APP                               = 0x800

WM_NULL                              = 0
WM_CREATE                            = 1
WM_DESTROY                           = 2
WM_MOVE                              = 3
WM_SIZE                              = 5
WM_ACTIVATE                          = 6
WA_INACTIVE                          = 0
WA_ACTIVE                            = 1
WA_CLICKACTIVE                       = 2
WM_SETFOCUS                          = 7
WM_KILLFOCUS                         = 8
WM_ENABLE                            = 0x0A
WM_SETREDRAW                         = 0x0B
WM_SETTEXT                           = 0x0C
WM_GETTEXT                           = 0x0D
WM_GETTEXTLENGTH                     = 0x0E
WM_PAINT                             = 0x0F
WM_CLOSE                             = 0x10
WM_QUERYENDSESSION                   = 0x11
WM_QUIT                              = 0x12
WM_QUERYOPEN                         = 0x13
WM_ERASEBKGND                        = 0x14
WM_SYSCOLORCHANGE                    = 0x15
WM_ENDSESSION                        = 0x16
WM_SHOWWINDOW                        = 0x18
WM_WININICHANGE                      = 0x1A
WM_SETTINGCHANGE                	 = WM_WININICHANGE
WM_DEVMODECHANGE                     = 0x1B
WM_ACTIVATEAPP                       = 0x1C
WM_FONTCHANGE                        = 0x1D
WM_TIMECHANGE                        = 0x1E
WM_CANCELMODE                        = 0x1F
WM_SETCURSOR                         = 0x20
WM_MOUSEACTIVATE                     = 0x21
WM_CHILDACTIVATE                     = 0x22
WM_QUEUESYNC                         = 0x23
WM_GETMINMAXINFO                     = 0x24
WM_PAINTICON                         = 0x26
WM_ICONERASEBKGND                    = 0x27
WM_NEXTDLGCTL                        = 0x28
WM_SPOOLERSTATUS                     = 0x2A
WM_DRAWITEM                          = 0x2B
WM_MEASUREITEM                       = 0x2C
WM_DELETEITEM                        = 0x2D
WM_VKEYTOITEM                        = 0x2E
WM_CHARTOITEM                        = 0x2F
WM_SETFONT                           = 0x30
WM_GETFONT                           = 0x31
WM_SETHOTKEY                         = 0x32
WM_GETHOTKEY                         = 0x33
WM_QUERYDRAGICON                     = 0x37
WM_COMPAREITEM                       = 0x39
WM_GETOBJECT                    	 = 0x3D
WM_COMPACTING                        = 0x41
WM_OTHERWINDOWCREATED                = 0x42
WM_OTHERWINDOWDESTROYED              = 0x43
WM_COMMNOTIFY                        = 0x44

CN_RECEIVE                           = 0x1
CN_TRANSMIT                          = 0x2
CN_EVENT                             = 0x4

WM_WINDOWPOSCHANGING                 = 0x46
WM_WINDOWPOSCHANGED                  = 0x47
WM_POWER                             = 0x48

PWR_OK                               = 1
PWR_FAIL                             = -1
PWR_SUSPENDREQUEST                   = 1
PWR_SUSPENDRESUME                    = 2
PWR_CRITICALRESUME                   = 3

WM_COPYDATA                          = 0x4A
WM_CANCELJOURNAL                     = 0x4B
WM_NOTIFY                            = 0x4E
WM_INPUTLANGCHANGEREQUEST            = 0x50
WM_INPUTLANGCHANGE                   = 0x51
WM_TCARD                             = 0x52
WM_HELP                              = 0x53
WM_USERCHANGED                       = 0x54
WM_NOTIFYFORMAT                      = 0x55
WM_CONTEXTMENU                       = 0x7B
WM_STYLECHANGING                     = 0x7C
WM_STYLECHANGED                      = 0x7D
WM_DISPLAYCHANGE                     = 0x7E
WM_GETICON                           = 0x7F
WM_SETICON                           = 0x80
WM_NCCREATE                          = 0x81
WM_NCDESTROY                         = 0x82
WM_NCCALCSIZE                        = 0x83
WM_NCHITTEST                         = 0x84
WM_NCPAINT                           = 0x85
WM_NCACTIVATE                        = 0x86
WM_GETDLGCODE                        = 0x87
WM_SYNCPAINT                    	 = 0x88
WM_NCMOUSEMOVE                       = 0x0A0
WM_NCLBUTTONDOWN                     = 0x0A1
WM_NCLBUTTONUP                       = 0x0A2
WM_NCLBUTTONDBLCLK                   = 0x0A3
WM_NCRBUTTONDOWN                     = 0x0A4
WM_NCRBUTTONUP                       = 0x0A5
WM_NCRBUTTONDBLCLK                   = 0x0A6
WM_NCMBUTTONDOWN                     = 0x0A7
WM_NCMBUTTONUP                       = 0x0A8
WM_NCMBUTTONDBLCLK                   = 0x0A9
WM_KEYFIRST                          = 0x100
WM_KEYDOWN                           = 0x100
WM_KEYUP                             = 0x101
WM_CHAR                              = 0x102
WM_DEADCHAR                          = 0x103
WM_SYSKEYDOWN                        = 0x104
WM_SYSKEYUP                          = 0x105
WM_SYSCHAR                           = 0x106
WM_SYSDEADCHAR                       = 0x107
WM_KEYLAST                           = 0x108
WM_INITDIALOG                        = 0x110
WM_COMMAND                           = 0x111
WM_SYSCOMMAND                        = 0x112
WM_TIMER                             = 0x113
WM_HSCROLL                           = 0x114
WM_VSCROLL                           = 0x115
WM_INITMENU                          = 0x116
WM_INITMENUPOPUP                     = 0x117
WM_MENUSELECT                        = 0x11F
WM_MENUCHAR                          = 0x120
WM_ENTERIDLE                         = 0x121
WM_CTLCOLORMSGBOX                    = 0x132
WM_CTLCOLOREDIT                      = 0x133
WM_CTLCOLORLISTBOX                   = 0x134
WM_CTLCOLORBTN                       = 0x135
WM_CTLCOLORDLG                       = 0x136
WM_CTLCOLORSCROLLBAR                 = 0x137
WM_CTLCOLORSTATIC                    = 0x138
WM_MOUSEFIRST                        = 0x200
WM_MOUSEMOVE                         = 0x200
WM_LBUTTONDOWN                       = 0x201
WM_LBUTTONUP                         = 0x202
WM_LBUTTONDBLCLK                     = 0x203
WM_RBUTTONDOWN                       = 0x204
WM_RBUTTONUP                         = 0x205
WM_RBUTTONDBLCLK                     = 0x206
WM_MBUTTONDOWN                       = 0x207
WM_MBUTTONUP                         = 0x208
WM_MBUTTONDBLCLK                     = 0x209
WM_MOUSELAST                         = 0x209
WM_PARENTNOTIFY                      = 0x210
WM_ENTERMENULOOP                     = 0x211
WM_EXITMENULOOP                      = 0x212
WM_MDICREATE                         = 0x220
WM_MDIDESTROY                        = 0x221
WM_MDIACTIVATE                       = 0x222
WM_MDIRESTORE                        = 0x223
WM_MDINEXT                           = 0x224
WM_MDIMAXIMIZE                       = 0x225
WM_MDITILE                           = 0x226
WM_MDICASCADE                        = 0x227
WM_MDIICONARRANGE                    = 0x228
WM_MDIGETACTIVE                      = 0x229
WM_MDISETMENU                        = 0x230
WM_DROPFILES                         = 0x233
WM_MDIREFRESHMENU                    = 0x234
WM_CUT                               = 0x300
WM_COPY                              = 0x301
WM_PASTE                             = 0x302
WM_CLEAR                             = 0x303
WM_UNDO                              = 0x304
WM_RENDERFORMAT                      = 0x305
WM_RENDERALLFORMATS                  = 0x306
WM_DESTROYCLIPBOARD                  = 0x307
WM_DRAWCLIPBOARD                     = 0x308
WM_PAINTCLIPBOARD                    = 0x309
WM_VSCROLLCLIPBOARD                  = 0x30A
WM_SIZECLIPBOARD                     = 0x30B
WM_ASKCBFORMATNAME                   = 0x30C
WM_CHANGECBCHAIN                     = 0x30D
WM_HSCROLLCLIPBOARD                  = 0x30E
WM_QUERYNEWPALETTE                   = 0x30F
WM_PALETTEISCHANGING                 = 0x310
WM_PALETTECHANGED                    = 0x311
WM_HOTKEY                            = 0x312
WM_PRINT                        	 = 0x317
WM_PRINTCLIENT                       = 0x318
WM_PENWINFIRST                       = 0x380
WM_PENWINLAST                        = 0x38F

#--- Structures ---------------------------------------------------------------

# typedef struct _WINDOWPLACEMENT {
#     UINT length;
#     UINT flags;
#     UINT showCmd;
#     POINT ptMinPosition;
#     POINT ptMaxPosition;
#     RECT rcNormalPosition;
# } WINDOWPLACEMENT;
class WINDOWPLACEMENT(Structure):
    _fields_ = [
        ('length',              UINT),
        ('flags',               UINT),
        ('showCmd',             UINT),
        ('ptMinPosition',       POINT),
        ('ptMaxPosition',       POINT),
        ('rcNormalPosition',    RECT),
    ]
PWINDOWPLACEMENT  = POINTER(WINDOWPLACEMENT)
LPWINDOWPLACEMENT = PWINDOWPLACEMENT

# typedef struct tagGUITHREADINFO {
#     DWORD cbSize;
#     DWORD flags;
#     HWND hwndActive;
#     HWND hwndFocus;
#     HWND hwndCapture;
#     HWND hwndMenuOwner;
#     HWND hwndMoveSize;
#     HWND hwndCaret;
#     RECT rcCaret;
# } GUITHREADINFO, *PGUITHREADINFO;
class GUITHREADINFO(Structure):
    _fields_ = [
        ('cbSize',          DWORD),
        ('flags',           DWORD),
        ('hwndActive',      HWND),
        ('hwndFocus',       HWND),
        ('hwndCapture',     HWND),
        ('hwndMenuOwner',   HWND),
        ('hwndMoveSize',    HWND),
        ('hwndCaret',       HWND),
        ('rcCaret',         RECT),
    ]
PGUITHREADINFO  = POINTER(GUITHREADINFO)
LPGUITHREADINFO = PGUITHREADINFO

#--- High level classes -------------------------------------------------------

# Point() and Rect() are here instead of gdi32.py because they were mainly
# created to handle window coordinates rather than drawing on the screen.

# XXX not sure if these classes should be psyco-optimized,
# it may not work if the user wants to serialize them for some reason

class Point(object):
    """
    Python wrapper over the L{POINT} class.

    @type x: int
    @ivar x: Horizontal coordinate
    @type y: int
    @ivar y: Vertical coordinate
    """

    def __init__(self, x = 0, y = 0):
        """
        @see: L{POINT}
        @type  x: int
        @param x: Horizontal coordinate
        @type  y: int
        @param y: Vertical coordinate
        """
        self.x = x
        self.y = y

    def __iter__(self):
        return (self.x, self.y).__iter__()

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.x, self.y) [index]

    def __setitem__(self, index, value):
        if   index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError("index out of range")

    @property
    def _as_parameter_(self):
        """
        Compatibility with ctypes.
        Allows passing transparently a Point object to an API call.
        """
        return POINT(self.x, self.y)

    def screen_to_client(self, hWnd):
        """
        Translates window screen coordinates to client coordinates.

        @see: L{client_to_screen}, L{translate}

        @type  hWnd: int or L{HWND} or L{system.Window}
        @param hWnd: Window handle.

        @rtype:  L{Point}
        @return: New object containing the translated coordinates.
        """
        return ScreenToClient(hWnd, self)

    def client_to_screen(self, hWnd):
        """
        Translates window client coordinates to screen coordinates.

        @see: L{screen_to_client}, L{translate}

        @type  hWnd: int or L{HWND} or L{system.Window}
        @param hWnd: Window handle.

        @rtype:  L{Point}
        @return: New object containing the translated coordinates.
        """
        return ClientToScreen(hWnd, self)

    def translate(self, hWndFrom = HWND_DESKTOP, hWndTo = HWND_DESKTOP):
        """
        Translate coordinates from one window to another.

        @note: To translate multiple points it's more efficient to use the
            L{MapWindowPoints} function instead.

        @see: L{client_to_screen}, L{screen_to_client}

        @type  hWndFrom: int or L{HWND} or L{system.Window}
        @param hWndFrom: Window handle to translate from.
            Use C{HWND_DESKTOP} for screen coordinates.

        @type  hWndTo: int or L{HWND} or L{system.Window}
        @param hWndTo: Window handle to translate to.
            Use C{HWND_DESKTOP} for screen coordinates.

        @rtype:  L{Point}
        @return: New object containing the translated coordinates.
        """
        return MapWindowPoints(hWndFrom, hWndTo, [self])

class Rect(object):
    """
    Python wrapper over the L{RECT} class.

    @type   left: int
    @ivar   left: Horizontal coordinate for the top left corner.
    @type    top: int
    @ivar    top: Vertical coordinate for the top left corner.
    @type  right: int
    @ivar  right: Horizontal coordinate for the bottom right corner.
    @type bottom: int
    @ivar bottom: Vertical coordinate for the bottom right corner.

    @type  width: int
    @ivar  width: Width in pixels. Same as C{right - left}.
    @type height: int
    @ivar height: Height in pixels. Same as C{bottom - top}.
    """

    def __init__(self, left = 0, top = 0, right = 0, bottom = 0):
        """
        @see: L{RECT}
        @type    left: int
        @param   left: Horizontal coordinate for the top left corner.
        @type     top: int
        @param    top: Vertical coordinate for the top left corner.
        @type   right: int
        @param  right: Horizontal coordinate for the bottom right corner.
        @type  bottom: int
        @param bottom: Vertical coordinate for the bottom right corner.
        """
        self.left   = left
        self.top    = top
        self.right  = right
        self.bottom = bottom

    def __iter__(self):
        return (self.left, self.top, self.right, self.bottom).__iter__()

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.left, self.top, self.right, self.bottom) [index]

    def __setitem__(self, index, value):
        if   index == 0:
            self.left   = value
        elif index == 1:
            self.top    = value
        elif index == 2:
            self.right  = value
        elif index == 3:
            self.bottom = value
        else:
            raise IndexError("index out of range")

    @property
    def _as_parameter_(self):
        """
        Compatibility with ctypes.
        Allows passing transparently a Point object to an API call.
        """
        return RECT(self.left, self.top, self.right, self.bottom)

    def __get_width(self):
        return self.right - self.left

    def __get_height(self):
        return self.bottom - self.top

    def __set_width(self, value):
        self.right = value - self.left

    def __set_height(self, value):
        self.bottom = value - self.top

    width  = property(__get_width, __set_width)
    height = property(__get_height, __set_height)

    def screen_to_client(self, hWnd):
        """
        Translates window screen coordinates to client coordinates.

        @see: L{client_to_screen}, L{translate}

        @type  hWnd: int or L{HWND} or L{system.Window}
        @param hWnd: Window handle.

        @rtype:  L{Rect}
        @return: New object containing the translated coordinates.
        """
        topleft     = ScreenToClient(hWnd, (self.left,   self.top))
        bottomright = ScreenToClient(hWnd, (self.bottom, self.right))
        return Rect( topleft.x, topleft.y, bottomright.x, bottomright.y )

    def client_to_screen(self, hWnd):
        """
        Translates window client coordinates to screen coordinates.

        @see: L{screen_to_client}, L{translate}

        @type  hWnd: int or L{HWND} or L{system.Window}
        @param hWnd: Window handle.

        @rtype:  L{Rect}
        @return: New object containing the translated coordinates.
        """
        topleft     = ClientToScreen(hWnd, (self.left,   self.top))
        bottomright = ClientToScreen(hWnd, (self.bottom, self.right))
        return Rect( topleft.x, topleft.y, bottomright.x, bottomright.y )

    def translate(self, hWndFrom = HWND_DESKTOP, hWndTo = HWND_DESKTOP):
        """
        Translate coordinates from one window to another.

        @see: L{client_to_screen}, L{screen_to_client}

        @type  hWndFrom: int or L{HWND} or L{system.Window}
        @param hWndFrom: Window handle to translate from.
            Use C{HWND_DESKTOP} for screen coordinates.

        @type  hWndTo: int or L{HWND} or L{system.Window}
        @param hWndTo: Window handle to translate to.
            Use C{HWND_DESKTOP} for screen coordinates.

        @rtype:  L{Rect}
        @return: New object containing the translated coordinates.
        """
        points = [ (self.left, self.top), (self.right, self.bottom) ]
        return MapWindowPoints(hWndFrom, hWndTo, points)

class WindowPlacement(object):
    """
    Python wrapper over the L{WINDOWPLACEMENT} class.
    """

    def __init__(self, wp = None):
        """
        @type  wp: L{WindowPlacement} or L{WINDOWPLACEMENT}
        @param wp: Another window placement object.
        """

        # Initialize all properties with empty values.
        self.flags            = 0
        self.showCmd          = 0
        self.ptMinPosition    = Point()
        self.ptMaxPosition    = Point()
        self.rcNormalPosition = Rect()

        # If a window placement was given copy it's properties.
        if wp:
            self.flags            = wp.flags
            self.showCmd          = wp.showCmd
            self.ptMinPosition    = Point( wp.ptMinPosition.x, wp.ptMinPosition.y )
            self.ptMaxPosition    = Point( wp.ptMaxPosition.x, wp.ptMaxPosition.y )
            self.rcNormalPosition = Rect(
                                        wp.rcNormalPosition.left,
                                        wp.rcNormalPosition.top,
                                        wp.rcNormalPosition.right,
                                        wp.rcNormalPosition.bottom,
                                        )

    @property
    def _as_parameter_(self):
        """
        Compatibility with ctypes.
        Allows passing transparently a Point object to an API call.
        """
        wp                          = WINDOWPLACEMENT()
        wp.length                   = sizeof(wp)
        wp.flags                    = self.flags
        wp.showCmd                  = self.showCmd
        wp.ptMinPosition.x          = self.ptMinPosition.x
        wp.ptMinPosition.y          = self.ptMinPosition.y
        wp.ptMaxPosition.x          = self.ptMaxPosition.x
        wp.ptMaxPosition.y          = self.ptMaxPosition.y
        wp.rcNormalPosition.left    = self.rcNormalPosition.left
        wp.rcNormalPosition.top     = self.rcNormalPosition.top
        wp.rcNormalPosition.right   = self.rcNormalPosition.right
        wp.rcNormalPosition.bottom  = self.rcNormalPosition.bottom
        return wp

#--- user32.dll ---------------------------------------------------------------

# void WINAPI SetLastErrorEx(
#   __in  DWORD dwErrCode,
#   __in  DWORD dwType
# );
def SetLastErrorEx(dwErrCode, dwType = 0):
    _SetLastErrorEx = windll.user32.SetLastErrorEx
    _SetLastErrorEx.argtypes = [DWORD, DWORD]
    _SetLastErrorEx.restype  = None
    _SetLastErrorEx(dwErrCode, dwType)

# HWND FindWindow(
#     LPCTSTR lpClassName,
#     LPCTSTR lpWindowName
# );
def FindWindowA(lpClassName = None, lpWindowName = None):
    _FindWindowA = windll.user32.FindWindowA
    _FindWindowA.argtypes = [LPSTR, LPSTR]
    _FindWindowA.restype  = HWND

    hWnd = _FindWindowA(lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd

def FindWindowW(lpClassName = None, lpWindowName = None):
    _FindWindowW = windll.user32.FindWindowW
    _FindWindowW.argtypes = [LPWSTR, LPWSTR]
    _FindWindowW.restype  = HWND

    hWnd = _FindWindowW(lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd

FindWindow = GuessStringType(FindWindowA, FindWindowW)

# HWND WINAPI FindWindowEx(
#   __in_opt  HWND hwndParent,
#   __in_opt  HWND hwndChildAfter,
#   __in_opt  LPCTSTR lpszClass,
#   __in_opt  LPCTSTR lpszWindow
# );
def FindWindowExA(hwndParent = None, hwndChildAfter = None, lpClassName = None, lpWindowName = None):
    _FindWindowExA = windll.user32.FindWindowExA
    _FindWindowExA.argtypes = [HWND, HWND, LPSTR, LPSTR]
    _FindWindowExA.restype  = HWND

    hWnd = _FindWindowExA(hwndParent, hwndChildAfter, lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd

def FindWindowExW(hwndParent = None, hwndChildAfter = None, lpClassName = None, lpWindowName = None):
    _FindWindowExW = windll.user32.FindWindowExW
    _FindWindowExW.argtypes = [HWND, HWND, LPWSTR, LPWSTR]
    _FindWindowExW.restype  = HWND

    hWnd = _FindWindowExW(hwndParent, hwndChildAfter, lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd

FindWindowEx = GuessStringType(FindWindowExA, FindWindowExW)

# int GetClassName(
#     HWND hWnd,
#     LPTSTR lpClassName,
#     int nMaxCount
# );
def GetClassNameA(hWnd):
    _GetClassNameA = windll.user32.GetClassNameA
    _GetClassNameA.argtypes = [HWND, LPSTR, ctypes.c_int]
    _GetClassNameA.restype = ctypes.c_int

    nMaxCount = 0x1000
    dwCharSize = sizeof(CHAR)
    while 1:
        lpClassName = ctypes.create_string_buffer("", nMaxCount)
        nCount = _GetClassNameA(hWnd, lpClassName, nMaxCount)
        if nCount == 0:
            raise ctypes.WinError()
        if nCount < nMaxCount - dwCharSize:
            break
        nMaxCount += 0x1000
    return lpClassName.value

def GetClassNameW(hWnd):
    _GetClassNameW = windll.user32.GetClassNameW
    _GetClassNameW.argtypes = [HWND, LPWSTR, ctypes.c_int]
    _GetClassNameW.restype = ctypes.c_int

    nMaxCount = 0x1000
    dwCharSize = sizeof(WCHAR)
    while 1:
        lpClassName = ctypes.create_unicode_buffer(u"", nMaxCount)
        nCount = _GetClassNameW(hWnd, lpClassName, nMaxCount)
        if nCount == 0:
            raise ctypes.WinError()
        if nCount < nMaxCount - dwCharSize:
            break
        nMaxCount += 0x1000
    return lpClassName.value

GetClassName = GuessStringType(GetClassNameA, GetClassNameW)

# int WINAPI GetWindowText(
#   __in   HWND hWnd,
#   __out  LPTSTR lpString,
#   __in   int nMaxCount
# );
def GetWindowTextA(hWnd):
    _GetWindowTextA = windll.user32.GetWindowTextA
    _GetWindowTextA.argtypes = [HWND, LPSTR, ctypes.c_int]
    _GetWindowTextA.restype = ctypes.c_int

    nMaxCount = 0x1000
    dwCharSize = sizeof(CHAR)
    while 1:
        lpString = ctypes.create_string_buffer("", nMaxCount)
        nCount = _GetWindowTextA(hWnd, lpString, nMaxCount)
        if nCount == 0:
            raise ctypes.WinError()
        if nCount < nMaxCount - dwCharSize:
            break
        nMaxCount += 0x1000
    return lpString.value

def GetWindowTextW(hWnd):
    _GetWindowTextW = windll.user32.GetWindowTextW
    _GetWindowTextW.argtypes = [HWND, LPWSTR, ctypes.c_int]
    _GetWindowTextW.restype = ctypes.c_int

    nMaxCount = 0x1000
    dwCharSize = sizeof(CHAR)
    while 1:
        lpString = ctypes.create_string_buffer("", nMaxCount)
        nCount = _GetWindowTextW(hWnd, lpString, nMaxCount)
        if nCount == 0:
            raise ctypes.WinError()
        if nCount < nMaxCount - dwCharSize:
            break
        nMaxCount += 0x1000
    return lpString.value

GetWindowText = GuessStringType(GetWindowTextA, GetWindowTextW)

# BOOL WINAPI SetWindowText(
#   __in      HWND hWnd,
#   __in_opt  LPCTSTR lpString
# );
def SetWindowTextA(hWnd, lpString = None):
    _SetWindowTextA = windll.user32.SetWindowTextA
    _SetWindowTextA.argtypes = [HWND, LPSTR]
    _SetWindowTextA.restype  = bool
    _SetWindowTextA.errcheck = RaiseIfZero
    _SetWindowTextA(hWnd, lpString)

def SetWindowTextW(hWnd, lpString = None):
    _SetWindowTextW = windll.user32.SetWindowTextW
    _SetWindowTextW.argtypes = [HWND, LPWSTR]
    _SetWindowTextW.restype  = bool
    _SetWindowTextW.errcheck = RaiseIfZero
    _SetWindowTextW(hWnd, lpString)

SetWindowText = GuessStringType(SetWindowTextA, SetWindowTextW)

# LONG GetWindowLong(
#     HWND hWnd,
#     int nIndex
# );
def GetWindowLongA(hWnd, nIndex = 0):
    _GetWindowLongA = windll.user32.GetWindowLongA
    _GetWindowLongA.argtypes = [HWND, ctypes.c_int]
    _GetWindowLongA.restype  = DWORD

    SetLastError(ERROR_SUCCESS)
    retval = _GetWindowLongA(hWnd, nIndex)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval

def GetWindowLongW(hWnd, nIndex = 0):
    _GetWindowLongW = windll.user32.GetWindowLongW
    _GetWindowLongW.argtypes = [HWND, ctypes.c_int]
    _GetWindowLongW.restype  = DWORD

    SetLastError(ERROR_SUCCESS)
    retval = _GetWindowLongW(hWnd, nIndex)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval

GetWindowLong = DefaultStringType(GetWindowLongA, GetWindowLongW)

# LONG_PTR WINAPI GetWindowLongPtr(
#   _In_  HWND hWnd,
#   _In_  int nIndex
# );

if bits == 32:

    GetWindowLongPtrA = GetWindowLongA
    GetWindowLongPtrW = GetWindowLongW
    GetWindowLongPtr  = GetWindowLong

else:

    def GetWindowLongPtrA(hWnd, nIndex = 0):
        _GetWindowLongPtrA = windll.user32.GetWindowLongPtrA
        _GetWindowLongPtrA.argtypes = [HWND, ctypes.c_int]
        _GetWindowLongPtrA.restype  = SIZE_T

        SetLastError(ERROR_SUCCESS)
        retval = _GetWindowLongPtrA(hWnd, nIndex)
        if retval == 0:
            errcode = GetLastError()
            if errcode != ERROR_SUCCESS:
                raise ctypes.WinError(errcode)
        return retval

    def GetWindowLongPtrW(hWnd, nIndex = 0):
        _GetWindowLongPtrW = windll.user32.GetWindowLongPtrW
        _GetWindowLongPtrW.argtypes = [HWND, ctypes.c_int]
        _GetWindowLongPtrW.restype  = DWORD

        SetLastError(ERROR_SUCCESS)
        retval = _GetWindowLongPtrW(hWnd, nIndex)
        if retval == 0:
            errcode = GetLastError()
            if errcode != ERROR_SUCCESS:
                raise ctypes.WinError(errcode)
        return retval

    GetWindowLongPtr = DefaultStringType(GetWindowLongPtrA, GetWindowLongPtrW)

# LONG WINAPI SetWindowLong(
#   _In_  HWND hWnd,
#   _In_  int nIndex,
#   _In_  LONG dwNewLong
# );

def SetWindowLongA(hWnd, nIndex, dwNewLong):
    _SetWindowLongA = windll.user32.SetWindowLongA
    _SetWindowLongA.argtypes = [HWND, ctypes.c_int, DWORD]
    _SetWindowLongA.restype  = DWORD

    SetLastError(ERROR_SUCCESS)
    retval = _SetWindowLongA(hWnd, nIndex, dwNewLong)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval

def SetWindowLongW(hWnd, nIndex, dwNewLong):
    _SetWindowLongW = windll.user32.SetWindowLongW
    _SetWindowLongW.argtypes = [HWND, ctypes.c_int, DWORD]
    _SetWindowLongW.restype  = DWORD

    SetLastError(ERROR_SUCCESS)
    retval = _SetWindowLongW(hWnd, nIndex, dwNewLong)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval

SetWindowLong = DefaultStringType(SetWindowLongA, SetWindowLongW)

# LONG_PTR WINAPI SetWindowLongPtr(
#   _In_  HWND hWnd,
#   _In_  int nIndex,
#   _In_  LONG_PTR dwNewLong
# );

if bits == 32:

    SetWindowLongPtrA = SetWindowLongA
    SetWindowLongPtrW = SetWindowLongW
    SetWindowLongPtr  = SetWindowLong

else:

    def SetWindowLongPtrA(hWnd, nIndex, dwNewLong):
        _SetWindowLongPtrA = windll.user32.SetWindowLongPtrA
        _SetWindowLongPtrA.argtypes = [HWND, ctypes.c_int, SIZE_T]
        _SetWindowLongPtrA.restype  = SIZE_T

        SetLastError(ERROR_SUCCESS)
        retval = _SetWindowLongPtrA(hWnd, nIndex, dwNewLong)
        if retval == 0:
            errcode = GetLastError()
            if errcode != ERROR_SUCCESS:
                raise ctypes.WinError(errcode)
        return retval

    def SetWindowLongPtrW(hWnd, nIndex, dwNewLong):
        _SetWindowLongPtrW = windll.user32.SetWindowLongPtrW
        _SetWindowLongPtrW.argtypes = [HWND, ctypes.c_int, SIZE_T]
        _SetWindowLongPtrW.restype  = SIZE_T

        SetLastError(ERROR_SUCCESS)
        retval = _SetWindowLongPtrW(hWnd, nIndex, dwNewLong)
        if retval == 0:
            errcode = GetLastError()
            if errcode != ERROR_SUCCESS:
                raise ctypes.WinError(errcode)
        return retval

    SetWindowLongPtr = DefaultStringType(SetWindowLongPtrA, SetWindowLongPtrW)

# HWND GetShellWindow(VOID);
def GetShellWindow():
    _GetShellWindow = windll.user32.GetShellWindow
    _GetShellWindow.argtypes = []
    _GetShellWindow.restype  = HWND
    _GetShellWindow.errcheck = RaiseIfZero
    return _GetShellWindow()

# DWORD GetWindowThreadProcessId(
#     HWND hWnd,
#     LPDWORD lpdwProcessId
# );
def GetWindowThreadProcessId(hWnd):
    _GetWindowThreadProcessId = windll.user32.GetWindowThreadProcessId
    _GetWindowThreadProcessId.argtypes = [HWND, LPDWORD]
    _GetWindowThreadProcessId.restype  = DWORD
    _GetWindowThreadProcessId.errcheck = RaiseIfZero

    dwProcessId = DWORD(0)
    dwThreadId = _GetWindowThreadProcessId(hWnd, byref(dwProcessId))
    return (dwThreadId, dwProcessId.value)

# HWND WINAPI GetWindow(
#   __in  HWND hwnd,
#   __in  UINT uCmd
# );
def GetWindow(hWnd, uCmd):
    _GetWindow = windll.user32.GetWindow
    _GetWindow.argtypes = [HWND, UINT]
    _GetWindow.restype  = HWND

    SetLastError(ERROR_SUCCESS)
    hWndTarget = _GetWindow(hWnd, uCmd)
    if not hWndTarget:
        winerr = GetLastError()
        if winerr != ERROR_SUCCESS:
            raise ctypes.WinError(winerr)
    return hWndTarget

# HWND GetParent(
#       HWND hWnd
# );
def GetParent(hWnd):
    _GetParent = windll.user32.GetParent
    _GetParent.argtypes = [HWND]
    _GetParent.restype  = HWND

    SetLastError(ERROR_SUCCESS)
    hWndParent = _GetParent(hWnd)
    if not hWndParent:
        winerr = GetLastError()
        if winerr != ERROR_SUCCESS:
            raise ctypes.WinError(winerr)
    return hWndParent

# HWND WINAPI GetAncestor(
#   __in  HWND hwnd,
#   __in  UINT gaFlags
# );
def GetAncestor(hWnd, gaFlags = GA_PARENT):
    _GetAncestor = windll.user32.GetAncestor
    _GetAncestor.argtypes = [HWND, UINT]
    _GetAncestor.restype  = HWND

    SetLastError(ERROR_SUCCESS)
    hWndParent = _GetAncestor(hWnd, gaFlags)
    if not hWndParent:
        winerr = GetLastError()
        if winerr != ERROR_SUCCESS:
            raise ctypes.WinError(winerr)
    return hWndParent

# BOOL EnableWindow(
#     HWND hWnd,
#     BOOL bEnable
# );
def EnableWindow(hWnd, bEnable = True):
    _EnableWindow = windll.user32.EnableWindow
    _EnableWindow.argtypes = [HWND, BOOL]
    _EnableWindow.restype  = bool
    return _EnableWindow(hWnd, bool(bEnable))

# BOOL ShowWindow(
#     HWND hWnd,
#     int nCmdShow
# );
def ShowWindow(hWnd, nCmdShow = SW_SHOW):
    _ShowWindow = windll.user32.ShowWindow
    _ShowWindow.argtypes = [HWND, ctypes.c_int]
    _ShowWindow.restype  = bool
    return _ShowWindow(hWnd, nCmdShow)

# BOOL ShowWindowAsync(
#     HWND hWnd,
#     int nCmdShow
# );
def ShowWindowAsync(hWnd, nCmdShow = SW_SHOW):
    _ShowWindowAsync = windll.user32.ShowWindowAsync
    _ShowWindowAsync.argtypes = [HWND, ctypes.c_int]
    _ShowWindowAsync.restype  = bool
    return _ShowWindowAsync(hWnd, nCmdShow)

# HWND GetDesktopWindow(VOID);
def GetDesktopWindow():
    _GetDesktopWindow = windll.user32.GetDesktopWindow
    _GetDesktopWindow.argtypes = []
    _GetDesktopWindow.restype  = HWND
    _GetDesktopWindow.errcheck = RaiseIfZero
    return _GetDesktopWindow()

# HWND GetForegroundWindow(VOID);
def GetForegroundWindow():
    _GetForegroundWindow = windll.user32.GetForegroundWindow
    _GetForegroundWindow.argtypes = []
    _GetForegroundWindow.restype  = HWND
    _GetForegroundWindow.errcheck = RaiseIfZero
    return _GetForegroundWindow()

# BOOL IsWindow(
#     HWND hWnd
# );
def IsWindow(hWnd):
    _IsWindow = windll.user32.IsWindow
    _IsWindow.argtypes = [HWND]
    _IsWindow.restype  = bool
    return _IsWindow(hWnd)

# BOOL IsWindowVisible(
#     HWND hWnd
# );
def IsWindowVisible(hWnd):
    _IsWindowVisible = windll.user32.IsWindowVisible
    _IsWindowVisible.argtypes = [HWND]
    _IsWindowVisible.restype  = bool
    return _IsWindowVisible(hWnd)

# BOOL IsWindowEnabled(
#     HWND hWnd
# );
def IsWindowEnabled(hWnd):
    _IsWindowEnabled = windll.user32.IsWindowEnabled
    _IsWindowEnabled.argtypes = [HWND]
    _IsWindowEnabled.restype  = bool
    return _IsWindowEnabled(hWnd)

# BOOL IsZoomed(
#     HWND hWnd
# );
def IsZoomed(hWnd):
    _IsZoomed = windll.user32.IsZoomed
    _IsZoomed.argtypes = [HWND]
    _IsZoomed.restype  = bool
    return _IsZoomed(hWnd)

# BOOL IsIconic(
#     HWND hWnd
# );
def IsIconic(hWnd):
    _IsIconic = windll.user32.IsIconic
    _IsIconic.argtypes = [HWND]
    _IsIconic.restype  = bool
    return _IsIconic(hWnd)

# BOOL IsChild(
#     HWND hWnd
# );
def IsChild(hWnd):
    _IsChild = windll.user32.IsChild
    _IsChild.argtypes = [HWND]
    _IsChild.restype  = bool
    return _IsChild(hWnd)

# HWND WindowFromPoint(
#     POINT Point
# );
def WindowFromPoint(point):
    _WindowFromPoint = windll.user32.WindowFromPoint
    _WindowFromPoint.argtypes = [POINT]
    _WindowFromPoint.restype  = HWND
    _WindowFromPoint.errcheck = RaiseIfZero
    if isinstance(point, tuple):
        point = POINT(*point)
    return _WindowFromPoint(point)

# HWND ChildWindowFromPoint(
#     HWND hWndParent,
#     POINT Point
# );
def ChildWindowFromPoint(hWndParent, point):
    _ChildWindowFromPoint = windll.user32.ChildWindowFromPoint
    _ChildWindowFromPoint.argtypes = [HWND, POINT]
    _ChildWindowFromPoint.restype  = HWND
    _ChildWindowFromPoint.errcheck = RaiseIfZero
    if isinstance(point, tuple):
        point = POINT(*point)
    return _ChildWindowFromPoint(hWndParent, point)

#HWND RealChildWindowFromPoint(
#    HWND hwndParent,
#    POINT ptParentClientCoords
#);
def RealChildWindowFromPoint(hWndParent, ptParentClientCoords):
    _RealChildWindowFromPoint = windll.user32.RealChildWindowFromPoint
    _RealChildWindowFromPoint.argtypes = [HWND, POINT]
    _RealChildWindowFromPoint.restype  = HWND
    _RealChildWindowFromPoint.errcheck = RaiseIfZero
    if isinstance(ptParentClientCoords, tuple):
        ptParentClientCoords = POINT(*ptParentClientCoords)
    return _RealChildWindowFromPoint(hWndParent, ptParentClientCoords)

# BOOL ScreenToClient(
#   __in  HWND hWnd,
#         LPPOINT lpPoint
# );
def ScreenToClient(hWnd, lpPoint):
    _ScreenToClient = windll.user32.ScreenToClient
    _ScreenToClient.argtypes = [HWND, LPPOINT]
    _ScreenToClient.restype  = bool
    _ScreenToClient.errcheck = RaiseIfZero

    if isinstance(lpPoint, tuple):
        lpPoint = POINT(*lpPoint)
    else:
        lpPoint = POINT(lpPoint.x, lpPoint.y)
    _ScreenToClient(hWnd, byref(lpPoint))
    return Point(lpPoint.x, lpPoint.y)

# BOOL ClientToScreen(
#   HWND hWnd,
#   LPPOINT lpPoint
# );
def ClientToScreen(hWnd, lpPoint):
    _ClientToScreen = windll.user32.ClientToScreen
    _ClientToScreen.argtypes = [HWND, LPPOINT]
    _ClientToScreen.restype  = bool
    _ClientToScreen.errcheck = RaiseIfZero

    if isinstance(lpPoint, tuple):
        lpPoint = POINT(*lpPoint)
    else:
        lpPoint = POINT(lpPoint.x, lpPoint.y)
    _ClientToScreen(hWnd, byref(lpPoint))
    return Point(lpPoint.x, lpPoint.y)

# int MapWindowPoints(
#   __in     HWND hWndFrom,
#   __in     HWND hWndTo,
#   __inout  LPPOINT lpPoints,
#   __in     UINT cPoints
# );
def MapWindowPoints(hWndFrom, hWndTo, lpPoints):
    _MapWindowPoints = windll.user32.MapWindowPoints
    _MapWindowPoints.argtypes = [HWND, HWND, LPPOINT, UINT]
    _MapWindowPoints.restype  = ctypes.c_int

    cPoints  = len(lpPoints)
    lpPoints = (POINT * cPoints)(* lpPoints)
    SetLastError(ERROR_SUCCESS)
    number   = _MapWindowPoints(hWndFrom, hWndTo, byref(lpPoints), cPoints)
    if number == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    x_delta = number & 0xFFFF
    y_delta = (number >> 16) & 0xFFFF
    return x_delta, y_delta, [ (Point.x, Point.y) for Point in lpPoints ]

#BOOL SetForegroundWindow(
#    HWND hWnd
#);
def SetForegroundWindow(hWnd):
    _SetForegroundWindow = windll.user32.SetForegroundWindow
    _SetForegroundWindow.argtypes = [HWND]
    _SetForegroundWindow.restype  = bool
    _SetForegroundWindow.errcheck = RaiseIfZero
    return _SetForegroundWindow(hWnd)

# BOOL GetWindowPlacement(
#     HWND hWnd,
#     WINDOWPLACEMENT *lpwndpl
# );
def GetWindowPlacement(hWnd):
    _GetWindowPlacement = windll.user32.GetWindowPlacement
    _GetWindowPlacement.argtypes = [HWND, PWINDOWPLACEMENT]
    _GetWindowPlacement.restype  = bool
    _GetWindowPlacement.errcheck = RaiseIfZero

    lpwndpl = WINDOWPLACEMENT()
    lpwndpl.length = sizeof(lpwndpl)
    _GetWindowPlacement(hWnd, byref(lpwndpl))
    return WindowPlacement(lpwndpl)

# BOOL SetWindowPlacement(
#     HWND hWnd,
#     WINDOWPLACEMENT *lpwndpl
# );
def SetWindowPlacement(hWnd, lpwndpl):
    _SetWindowPlacement = windll.user32.SetWindowPlacement
    _SetWindowPlacement.argtypes = [HWND, PWINDOWPLACEMENT]
    _SetWindowPlacement.restype  = bool
    _SetWindowPlacement.errcheck = RaiseIfZero

    if isinstance(lpwndpl, WINDOWPLACEMENT):
        lpwndpl.length = sizeof(lpwndpl)
    _SetWindowPlacement(hWnd, byref(lpwndpl))

# BOOL WINAPI GetWindowRect(
#   __in   HWND hWnd,
#   __out  LPRECT lpRect
# );
def GetWindowRect(hWnd):
    _GetWindowRect = windll.user32.GetWindowRect
    _GetWindowRect.argtypes = [HWND, LPRECT]
    _GetWindowRect.restype  = bool
    _GetWindowRect.errcheck = RaiseIfZero

    lpRect = RECT()
    _GetWindowRect(hWnd, byref(lpRect))
    return Rect(lpRect.left, lpRect.top, lpRect.right, lpRect.bottom)

# BOOL WINAPI GetClientRect(
#   __in   HWND hWnd,
#   __out  LPRECT lpRect
# );
def GetClientRect(hWnd):
    _GetClientRect = windll.user32.GetClientRect
    _GetClientRect.argtypes = [HWND, LPRECT]
    _GetClientRect.restype  = bool
    _GetClientRect.errcheck = RaiseIfZero

    lpRect = RECT()
    _GetClientRect(hWnd, byref(lpRect))
    return Rect(lpRect.left, lpRect.top, lpRect.right, lpRect.bottom)

#BOOL MoveWindow(
#    HWND hWnd,
#    int X,
#    int Y,
#    int nWidth,
#    int nHeight,
#    BOOL bRepaint
#);
def MoveWindow(hWnd, X, Y, nWidth, nHeight, bRepaint = True):
    _MoveWindow = windll.user32.MoveWindow
    _MoveWindow.argtypes = [HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, BOOL]
    _MoveWindow.restype  = bool
    _MoveWindow.errcheck = RaiseIfZero
    _MoveWindow(hWnd, X, Y, nWidth, nHeight, bool(bRepaint))

# BOOL GetGUIThreadInfo(
#     DWORD idThread,
#     LPGUITHREADINFO lpgui
# );
def GetGUIThreadInfo(idThread):
    _GetGUIThreadInfo = windll.user32.GetGUIThreadInfo
    _GetGUIThreadInfo.argtypes = [DWORD, LPGUITHREADINFO]
    _GetGUIThreadInfo.restype  = bool
    _GetGUIThreadInfo.errcheck = RaiseIfZero

    gui = GUITHREADINFO()
    _GetGUIThreadInfo(idThread, byref(gui))
    return gui

# BOOL CALLBACK EnumWndProc(
#     HWND hwnd,
#     LPARAM lParam
# );
class __EnumWndProc (__WindowEnumerator):
    pass

# BOOL EnumWindows(
#     WNDENUMPROC lpEnumFunc,
#     LPARAM lParam
# );
def EnumWindows():
    _EnumWindows = windll.user32.EnumWindows
    _EnumWindows.argtypes = [WNDENUMPROC, LPARAM]
    _EnumWindows.restype  = bool

    EnumFunc = __EnumWndProc()
    lpEnumFunc = WNDENUMPROC(EnumFunc)
    if not _EnumWindows(lpEnumFunc, NULL):
        errcode = GetLastError()
        if errcode not in (ERROR_NO_MORE_FILES, ERROR_SUCCESS):
            raise ctypes.WinError(errcode)
    return EnumFunc.hwnd

# BOOL CALLBACK EnumThreadWndProc(
#     HWND hwnd,
#     LPARAM lParam
# );
class __EnumThreadWndProc (__WindowEnumerator):
    pass

# BOOL EnumThreadWindows(
#     DWORD dwThreadId,
#     WNDENUMPROC lpfn,
#     LPARAM lParam
# );
def EnumThreadWindows(dwThreadId):
    _EnumThreadWindows = windll.user32.EnumThreadWindows
    _EnumThreadWindows.argtypes = [DWORD, WNDENUMPROC, LPARAM]
    _EnumThreadWindows.restype  = bool

    fn = __EnumThreadWndProc()
    lpfn = WNDENUMPROC(fn)
    if not _EnumThreadWindows(dwThreadId, lpfn, NULL):
        errcode = GetLastError()
        if errcode not in (ERROR_NO_MORE_FILES, ERROR_SUCCESS):
            raise ctypes.WinError(errcode)
    return fn.hwnd

# BOOL CALLBACK EnumChildProc(
#     HWND hwnd,
#     LPARAM lParam
# );
class __EnumChildProc (__WindowEnumerator):
    pass

# BOOL EnumChildWindows(
#     HWND hWndParent,
#     WNDENUMPROC lpEnumFunc,
#     LPARAM lParam
# );
def EnumChildWindows(hWndParent = NULL):
    _EnumChildWindows = windll.user32.EnumChildWindows
    _EnumChildWindows.argtypes = [HWND, WNDENUMPROC, LPARAM]
    _EnumChildWindows.restype  = bool

    EnumFunc = __EnumChildProc()
    lpEnumFunc = WNDENUMPROC(EnumFunc)
    SetLastError(ERROR_SUCCESS)
    _EnumChildWindows(hWndParent, lpEnumFunc, NULL)
    errcode = GetLastError()
    if errcode != ERROR_SUCCESS and errcode not in (ERROR_NO_MORE_FILES, ERROR_SUCCESS):
        raise ctypes.WinError(errcode)
    return EnumFunc.hwnd

# LRESULT SendMessage(
#     HWND hWnd,
#     UINT Msg,
#     WPARAM wParam,
#     LPARAM lParam
# );
def SendMessageA(hWnd, Msg, wParam = 0, lParam = 0):
    _SendMessageA = windll.user32.SendMessageA
    _SendMessageA.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendMessageA.restype  = LRESULT

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendMessageA(hWnd, Msg, wParam, lParam)

def SendMessageW(hWnd, Msg, wParam = 0, lParam = 0):
    _SendMessageW = windll.user32.SendMessageW
    _SendMessageW.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendMessageW.restype  = LRESULT

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendMessageW(hWnd, Msg, wParam, lParam)

SendMessage = GuessStringType(SendMessageA, SendMessageW)

# BOOL PostMessage(
#     HWND hWnd,
#     UINT Msg,
#     WPARAM wParam,
#     LPARAM lParam
# );
def PostMessageA(hWnd, Msg, wParam = 0, lParam = 0):
    _PostMessageA = windll.user32.PostMessageA
    _PostMessageA.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _PostMessageA.restype  = bool
    _PostMessageA.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostMessageA(hWnd, Msg, wParam, lParam)

def PostMessageW(hWnd, Msg, wParam = 0, lParam = 0):
    _PostMessageW = windll.user32.PostMessageW
    _PostMessageW.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _PostMessageW.restype  = bool
    _PostMessageW.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostMessageW(hWnd, Msg, wParam, lParam)

PostMessage = GuessStringType(PostMessageA, PostMessageW)

# BOOL PostThreadMessage(
#     DWORD idThread,
#     UINT Msg,
#     WPARAM wParam,
#     LPARAM lParam
# );
def PostThreadMessageA(idThread, Msg, wParam = 0, lParam = 0):
    _PostThreadMessageA = windll.user32.PostThreadMessageA
    _PostThreadMessageA.argtypes = [DWORD, UINT, WPARAM, LPARAM]
    _PostThreadMessageA.restype  = bool
    _PostThreadMessageA.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostThreadMessageA(idThread, Msg, wParam, lParam)

def PostThreadMessageW(idThread, Msg, wParam = 0, lParam = 0):
    _PostThreadMessageW = windll.user32.PostThreadMessageW
    _PostThreadMessageW.argtypes = [DWORD, UINT, WPARAM, LPARAM]
    _PostThreadMessageW.restype  = bool
    _PostThreadMessageW.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostThreadMessageW(idThread, Msg, wParam, lParam)

PostThreadMessage = GuessStringType(PostThreadMessageA, PostThreadMessageW)

# LRESULT c(
#     HWND hWnd,
#     UINT Msg,
#     WPARAM wParam,
#     LPARAM lParam,
#     UINT fuFlags,
#     UINT uTimeout,
#     PDWORD_PTR lpdwResult
# );
def SendMessageTimeoutA(hWnd, Msg, wParam = 0, lParam = 0, fuFlags = 0, uTimeout = 0):
    _SendMessageTimeoutA = windll.user32.SendMessageTimeoutA
    _SendMessageTimeoutA.argtypes = [HWND, UINT, WPARAM, LPARAM, UINT, UINT, PDWORD_PTR]
    _SendMessageTimeoutA.restype  = LRESULT
    _SendMessageTimeoutA.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    dwResult = DWORD(0)
    _SendMessageTimeoutA(hWnd, Msg, wParam, lParam, fuFlags, uTimeout, byref(dwResult))
    return dwResult.value

def SendMessageTimeoutW(hWnd, Msg, wParam = 0, lParam = 0):
    _SendMessageTimeoutW = windll.user32.SendMessageTimeoutW
    _SendMessageTimeoutW.argtypes = [HWND, UINT, WPARAM, LPARAM, UINT, UINT, PDWORD_PTR]
    _SendMessageTimeoutW.restype  = LRESULT
    _SendMessageTimeoutW.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    dwResult = DWORD(0)
    _SendMessageTimeoutW(hWnd, Msg, wParam, lParam, fuFlags, uTimeout, byref(dwResult))
    return dwResult.value

SendMessageTimeout = GuessStringType(SendMessageTimeoutA, SendMessageTimeoutW)

# BOOL SendNotifyMessage(
#     HWND hWnd,
#     UINT Msg,
#     WPARAM wParam,
#     LPARAM lParam
# );
def SendNotifyMessageA(hWnd, Msg, wParam = 0, lParam = 0):
    _SendNotifyMessageA = windll.user32.SendNotifyMessageA
    _SendNotifyMessageA.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendNotifyMessageA.restype  = bool
    _SendNotifyMessageA.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _SendNotifyMessageA(hWnd, Msg, wParam, lParam)

def SendNotifyMessageW(hWnd, Msg, wParam = 0, lParam = 0):
    _SendNotifyMessageW = windll.user32.SendNotifyMessageW
    _SendNotifyMessageW.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendNotifyMessageW.restype  = bool
    _SendNotifyMessageW.errcheck = RaiseIfZero

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _SendNotifyMessageW(hWnd, Msg, wParam, lParam)

SendNotifyMessage = GuessStringType(SendNotifyMessageA, SendNotifyMessageW)

# LRESULT SendDlgItemMessage(
#     HWND hDlg,
#     int nIDDlgItem,
#     UINT Msg,
#     WPARAM wParam,
#     LPARAM lParam
# );
def SendDlgItemMessageA(hDlg, nIDDlgItem, Msg, wParam = 0, lParam = 0):
    _SendDlgItemMessageA = windll.user32.SendDlgItemMessageA
    _SendDlgItemMessageA.argtypes = [HWND, ctypes.c_int, UINT, WPARAM, LPARAM]
    _SendDlgItemMessageA.restype  = LRESULT

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendDlgItemMessageA(hDlg, nIDDlgItem, Msg, wParam, lParam)

def SendDlgItemMessageW(hDlg, nIDDlgItem, Msg, wParam = 0, lParam = 0):
    _SendDlgItemMessageW = windll.user32.SendDlgItemMessageW
    _SendDlgItemMessageW.argtypes = [HWND, ctypes.c_int, UINT, WPARAM, LPARAM]
    _SendDlgItemMessageW.restype  = LRESULT

    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendDlgItemMessageW(hDlg, nIDDlgItem, Msg, wParam, lParam)

SendDlgItemMessage = GuessStringType(SendDlgItemMessageA, SendDlgItemMessageW)

# DWORD WINAPI WaitForInputIdle(
#   _In_  HANDLE hProcess,
#   _In_  DWORD dwMilliseconds
# );
def WaitForInputIdle(hProcess, dwMilliseconds = INFINITE):
    _WaitForInputIdle = windll.user32.WaitForInputIdle
    _WaitForInputIdle.argtypes = [HANDLE, DWORD]
    _WaitForInputIdle.restype  = DWORD

    r = _WaitForInputIdle(hProcess, dwMilliseconds)
    if r == WAIT_FAILED:
        raise ctypes.WinError()
    return r

# UINT RegisterWindowMessage(
#     LPCTSTR lpString
# );
def RegisterWindowMessageA(lpString):
    _RegisterWindowMessageA = windll.user32.RegisterWindowMessageA
    _RegisterWindowMessageA.argtypes = [LPSTR]
    _RegisterWindowMessageA.restype  = UINT
    _RegisterWindowMessageA.errcheck = RaiseIfZero
    return _RegisterWindowMessageA(lpString)

def RegisterWindowMessageW(lpString):
    _RegisterWindowMessageW = windll.user32.RegisterWindowMessageW
    _RegisterWindowMessageW.argtypes = [LPWSTR]
    _RegisterWindowMessageW.restype  = UINT
    _RegisterWindowMessageW.errcheck = RaiseIfZero
    return _RegisterWindowMessageW(lpString)

RegisterWindowMessage = GuessStringType(RegisterWindowMessageA, RegisterWindowMessageW)

# UINT RegisterClipboardFormat(
#     LPCTSTR lpString
# );
def RegisterClipboardFormatA(lpString):
    _RegisterClipboardFormatA = windll.user32.RegisterClipboardFormatA
    _RegisterClipboardFormatA.argtypes = [LPSTR]
    _RegisterClipboardFormatA.restype  = UINT
    _RegisterClipboardFormatA.errcheck = RaiseIfZero
    return _RegisterClipboardFormatA(lpString)

def RegisterClipboardFormatW(lpString):
    _RegisterClipboardFormatW = windll.user32.RegisterClipboardFormatW
    _RegisterClipboardFormatW.argtypes = [LPWSTR]
    _RegisterClipboardFormatW.restype  = UINT
    _RegisterClipboardFormatW.errcheck = RaiseIfZero
    return _RegisterClipboardFormatW(lpString)

RegisterClipboardFormat = GuessStringType(RegisterClipboardFormatA, RegisterClipboardFormatW)

# HANDLE WINAPI GetProp(
#   __in  HWND hWnd,
#   __in  LPCTSTR lpString
# );
def GetPropA(hWnd, lpString):
    _GetPropA = windll.user32.GetPropA
    _GetPropA.argtypes = [HWND, LPSTR]
    _GetPropA.restype  = HANDLE
    return _GetPropA(hWnd, lpString)

def GetPropW(hWnd, lpString):
    _GetPropW = windll.user32.GetPropW
    _GetPropW.argtypes = [HWND, LPWSTR]
    _GetPropW.restype  = HANDLE
    return _GetPropW(hWnd, lpString)

GetProp = GuessStringType(GetPropA, GetPropW)

# BOOL WINAPI SetProp(
#   __in      HWND hWnd,
#   __in      LPCTSTR lpString,
#   __in_opt  HANDLE hData
# );
def SetPropA(hWnd, lpString, hData):
    _SetPropA = windll.user32.SetPropA
    _SetPropA.argtypes = [HWND, LPSTR, HANDLE]
    _SetPropA.restype  = BOOL
    _SetPropA.errcheck = RaiseIfZero
    _SetPropA(hWnd, lpString, hData)

def SetPropW(hWnd, lpString, hData):
    _SetPropW = windll.user32.SetPropW
    _SetPropW.argtypes = [HWND, LPWSTR, HANDLE]
    _SetPropW.restype  = BOOL
    _SetPropW.errcheck = RaiseIfZero
    _SetPropW(hWnd, lpString, hData)

SetProp = GuessStringType(SetPropA, SetPropW)

# HANDLE WINAPI RemoveProp(
#   __in  HWND hWnd,
#   __in  LPCTSTR lpString
# );
def RemovePropA(hWnd, lpString):
    _RemovePropA = windll.user32.RemovePropA
    _RemovePropA.argtypes = [HWND, LPSTR]
    _RemovePropA.restype  = HANDLE
    return _RemovePropA(hWnd, lpString)

def RemovePropW(hWnd, lpString):
    _RemovePropW = windll.user32.RemovePropW
    _RemovePropW.argtypes = [HWND, LPWSTR]
    _RemovePropW.restype  = HANDLE
    return _RemovePropW(hWnd, lpString)

RemoveProp = GuessStringType(RemovePropA, RemovePropW)

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
