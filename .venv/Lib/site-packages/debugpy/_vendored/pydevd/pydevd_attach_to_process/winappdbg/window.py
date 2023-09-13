#!~/.wine/drive_c/Python25/python.exe
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
Window instrumentation.

@group Instrumentation:
    Window
"""

__revision__ = "$Id$"

__all__ = ['Window']

from winappdbg import win32

# delayed imports
Process = None
Thread  = None

#==============================================================================

# Unlike Process, Thread and Module, there's no container for Window objects.
# That's because Window objects don't really store any data besides the handle.

# XXX TODO
# * implement sending fake user input (mouse and keyboard messages)
# * maybe implement low-level hooks? (they don't require a dll to be injected)

# XXX TODO
#
# Will it be possible to implement window hooks too? That requires a DLL to be
# injected in the target process. Perhaps with CPython it could be done easier,
# compiling a native extension is the safe bet, but both require having a non
# pure Python module, which is something I was trying to avoid so far.
#
# Another possibility would be to malloc some CC's in the target process and
# point the hook callback to it. We'd need to have the remote procedure call
# feature first as (I believe) the hook can't be set remotely in this case.

class Window (object):
    """
    Interface to an open window in the current desktop.

    @group Properties:
        get_handle, get_pid, get_tid,
        get_process, get_thread,
        set_process, set_thread,
        get_classname, get_style, get_extended_style,
        get_text, set_text,
        get_placement, set_placement,
        get_screen_rect, get_client_rect,
        screen_to_client, client_to_screen

    @group State:
        is_valid, is_visible, is_enabled, is_maximized, is_minimized, is_child,
        is_zoomed, is_iconic

    @group Navigation:
        get_parent, get_children, get_root, get_tree,
        get_child_at

    @group Instrumentation:
        enable, disable, show, hide, maximize, minimize, restore, move, kill

    @group Low-level access:
        send, post

    @type hWnd: int
    @ivar hWnd: Window handle.

    @type dwProcessId: int
    @ivar dwProcessId: Global ID of the process that owns this window.

    @type dwThreadId: int
    @ivar dwThreadId: Global ID of the thread that owns this window.

    @type process: L{Process}
    @ivar process: Process that owns this window.
        Use the L{get_process} method instead.

    @type thread: L{Thread}
    @ivar thread: Thread that owns this window.
        Use the L{get_thread} method instead.

    @type classname: str
    @ivar classname: Window class name.

    @type text: str
    @ivar text: Window text (caption).

    @type placement: L{win32.WindowPlacement}
    @ivar placement: Window placement in the desktop.
    """

    def __init__(self, hWnd = None, process = None, thread = None):
        """
        @type  hWnd: int or L{win32.HWND}
        @param hWnd: Window handle.

        @type  process: L{Process}
        @param process: (Optional) Process that owns this window.

        @type  thread: L{Thread}
        @param thread: (Optional) Thread that owns this window.
        """
        self.hWnd        = hWnd
        self.dwProcessId = None
        self.dwThreadId  = None
        self.set_process(process)
        self.set_thread(thread)

    @property
    def _as_parameter_(self):
        """
        Compatibility with ctypes.
        Allows passing transparently a Window object to an API call.
        """
        return self.get_handle()

    def get_handle(self):
        """
        @rtype:  int
        @return: Window handle.
        @raise ValueError: No window handle set.
        """
        if self.hWnd is None:
            raise ValueError("No window handle set!")
        return self.hWnd

    def get_pid(self):
        """
        @rtype:  int
        @return: Global ID of the process that owns this window.
        """
        if self.dwProcessId is not None:
            return self.dwProcessId
        self.__get_pid_and_tid()
        return self.dwProcessId

    def get_tid(self):
        """
        @rtype:  int
        @return: Global ID of the thread that owns this window.
        """
        if self.dwThreadId is not None:
            return self.dwThreadId
        self.__get_pid_and_tid()
        return self.dwThreadId

    def __get_pid_and_tid(self):
        "Internally used by get_pid() and get_tid()."
        self.dwThreadId, self.dwProcessId = \
                            win32.GetWindowThreadProcessId(self.get_handle())

    def __load_Process_class(self):
        global Process      # delayed import
        if Process is None:
            from winappdbg.process import Process

    def __load_Thread_class(self):
        global Thread       # delayed import
        if Thread is None:
            from winappdbg.thread import Thread

    def get_process(self):
        """
        @rtype:  L{Process}
        @return: Parent Process object.
        """
        if self.__process is not None:
            return self.__process
        self.__load_Process_class()
        self.__process = Process(self.get_pid())
        return self.__process

    def set_process(self, process = None):
        """
        Manually set the parent process. Use with care!

        @type  process: L{Process}
        @param process: (Optional) Process object. Use C{None} to autodetect.
        """
        if process is None:
            self.__process = None
        else:
            self.__load_Process_class()
            if not isinstance(process, Process):
                msg  = "Parent process must be a Process instance, "
                msg += "got %s instead" % type(process)
                raise TypeError(msg)
            self.dwProcessId = process.get_pid()
            self.__process = process

    def get_thread(self):
        """
        @rtype:  L{Thread}
        @return: Parent Thread object.
        """
        if self.__thread is not None:
            return self.__thread
        self.__load_Thread_class()
        self.__thread = Thread(self.get_tid())
        return self.__thread

    def set_thread(self, thread = None):
        """
        Manually set the thread process. Use with care!

        @type  thread: L{Thread}
        @param thread: (Optional) Thread object. Use C{None} to autodetect.
        """
        if thread is None:
            self.__thread = None
        else:
            self.__load_Thread_class()
            if not isinstance(thread, Thread):
                msg  = "Parent thread must be a Thread instance, "
                msg += "got %s instead" % type(thread)
                raise TypeError(msg)
            self.dwThreadId = thread.get_tid()
            self.__thread = thread

    def __get_window(self, hWnd):
        """
        User internally to get another Window from this one.
        It'll try to copy the parent Process and Thread references if possible.
        """
        window = Window(hWnd)
        if window.get_pid() == self.get_pid():
            window.set_process( self.get_process() )
        if window.get_tid() == self.get_tid():
            window.set_thread( self.get_thread() )
        return window

#------------------------------------------------------------------------------

    def get_classname(self):
        """
        @rtype:  str
        @return: Window class name.

        @raise WindowsError: An error occured while processing this request.
        """
        return win32.GetClassName( self.get_handle() )

    def get_style(self):
        """
        @rtype:  int
        @return: Window style mask.

        @raise WindowsError: An error occured while processing this request.
        """
        return win32.GetWindowLongPtr( self.get_handle(), win32.GWL_STYLE )

    def get_extended_style(self):
        """
        @rtype:  int
        @return: Window extended style mask.

        @raise WindowsError: An error occured while processing this request.
        """
        return win32.GetWindowLongPtr( self.get_handle(), win32.GWL_EXSTYLE )

    def get_text(self):
        """
        @see:    L{set_text}
        @rtype:  str
        @return: Window text (caption) on success, C{None} on error.
        """
        try:
            return win32.GetWindowText( self.get_handle() )
        except WindowsError:
            return None

    def set_text(self, text):
        """
        Set the window text (caption).

        @see: L{get_text}

        @type  text: str
        @param text: New window text.

        @raise WindowsError: An error occured while processing this request.
        """
        win32.SetWindowText( self.get_handle(), text )

    def get_placement(self):
        """
        Retrieve the window placement in the desktop.

        @see: L{set_placement}

        @rtype:  L{win32.WindowPlacement}
        @return: Window placement in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
        return win32.GetWindowPlacement( self.get_handle() )

    def set_placement(self, placement):
        """
        Set the window placement in the desktop.

        @see: L{get_placement}

        @type  placement: L{win32.WindowPlacement}
        @param placement: Window placement in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
        win32.SetWindowPlacement( self.get_handle(), placement )

    def get_screen_rect(self):
        """
        Get the window coordinates in the desktop.

        @rtype:  L{win32.Rect}
        @return: Rectangle occupied by the window in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
        return win32.GetWindowRect( self.get_handle() )

    def get_client_rect(self):
        """
        Get the window's client area coordinates in the desktop.

        @rtype:  L{win32.Rect}
        @return: Rectangle occupied by the window's client area in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
        cr = win32.GetClientRect( self.get_handle() )
        cr.left, cr.top     = self.client_to_screen(cr.left, cr.top)
        cr.right, cr.bottom = self.client_to_screen(cr.right, cr.bottom)
        return cr

    # XXX TODO
    # * properties x, y, width, height
    # * properties left, top, right, bottom

    process = property(get_process, set_process, doc="")
    thread = property(get_thread, set_thread, doc="")
    classname = property(get_classname, doc="")
    style = property(get_style, doc="")
    exstyle = property(get_extended_style, doc="")
    text = property(get_text, set_text, doc="")
    placement = property(get_placement, set_placement, doc="")

#------------------------------------------------------------------------------

    def client_to_screen(self, x, y):
        """
        Translates window client coordinates to screen coordinates.

        @note: This is a simplified interface to some of the functionality of
            the L{win32.Point} class.

        @see: {win32.Point.client_to_screen}

        @type  x: int
        @param x: Horizontal coordinate.
        @type  y: int
        @param y: Vertical coordinate.

        @rtype:  tuple( int, int )
        @return: Translated coordinates in a tuple (x, y).

        @raise WindowsError: An error occured while processing this request.
        """
        return tuple( win32.ClientToScreen( self.get_handle(), (x, y) ) )

    def screen_to_client(self, x, y):
        """
        Translates window screen coordinates to client coordinates.

        @note: This is a simplified interface to some of the functionality of
            the L{win32.Point} class.

        @see: {win32.Point.screen_to_client}

        @type  x: int
        @param x: Horizontal coordinate.
        @type  y: int
        @param y: Vertical coordinate.

        @rtype:  tuple( int, int )
        @return: Translated coordinates in a tuple (x, y).

        @raise WindowsError: An error occured while processing this request.
        """
        return tuple( win32.ScreenToClient( self.get_handle(), (x, y) ) )

#------------------------------------------------------------------------------

    def get_parent(self):
        """
        @see:    L{get_children}
        @rtype:  L{Window} or None
        @return: Parent window. Returns C{None} if the window has no parent.
        @raise WindowsError: An error occured while processing this request.
        """
        hWnd = win32.GetParent( self.get_handle() )
        if hWnd:
            return self.__get_window(hWnd)

    def get_children(self):
        """
        @see:    L{get_parent}
        @rtype:  list( L{Window} )
        @return: List of child windows.
        @raise WindowsError: An error occured while processing this request.
        """
        return [
                self.__get_window(hWnd) \
                for hWnd in win32.EnumChildWindows( self.get_handle() )
                ]

    def get_tree(self):
        """
        @see:    L{get_root}
        @rtype:  dict( L{Window} S{->} dict( ... ) )
        @return: Dictionary of dictionaries forming a tree of child windows.
        @raise WindowsError: An error occured while processing this request.
        """
        subtree = dict()
        for aWindow in self.get_children():
            subtree[ aWindow ] = aWindow.get_tree()
        return subtree

    def get_root(self):
        """
        @see:    L{get_tree}
        @rtype:  L{Window}
        @return: If this is a child window, return the top-level window it
            belongs to.
            If this window is already a top-level window, returns itself.
        @raise WindowsError: An error occured while processing this request.
        """
        hWnd     = self.get_handle()
        history  = set()
        hPrevWnd = hWnd
        while hWnd and hWnd not in history:
            history.add(hWnd)
            hPrevWnd = hWnd
            hWnd     = win32.GetParent(hWnd)
        if hWnd in history:
            # See: https://docs.google.com/View?id=dfqd62nk_228h28szgz
            return self
        if hPrevWnd != self.get_handle():
            return self.__get_window(hPrevWnd)
        return self

    def get_child_at(self, x, y, bAllowTransparency = True):
        """
        Get the child window located at the given coordinates. If no such
        window exists an exception is raised.

        @see: L{get_children}

        @type  x: int
        @param x: Horizontal coordinate.

        @type  y: int
        @param y: Vertical coordinate.

        @type  bAllowTransparency: bool
        @param bAllowTransparency: If C{True} transparent areas in windows are
            ignored, returning the window behind them. If C{False} transparent
            areas are treated just like any other area.

        @rtype:  L{Window}
        @return: Child window at the requested position, or C{None} if there
            is no window at those coordinates.
        """
        try:
            if bAllowTransparency:
                hWnd = win32.RealChildWindowFromPoint( self.get_handle(), (x, y) )
            else:
                hWnd = win32.ChildWindowFromPoint( self.get_handle(), (x, y) )
            if hWnd:
                return self.__get_window(hWnd)
        except WindowsError:
            pass
        return None

#------------------------------------------------------------------------------

    def is_valid(self):
        """
        @rtype:  bool
        @return: C{True} if the window handle is still valid.
        """
        return win32.IsWindow( self.get_handle() )

    def is_visible(self):
        """
        @see: {show}, {hide}
        @rtype:  bool
        @return: C{True} if the window is in a visible state.
        """
        return win32.IsWindowVisible( self.get_handle() )

    def is_enabled(self):
        """
        @see: {enable}, {disable}
        @rtype:  bool
        @return: C{True} if the window is in an enabled state.
        """
        return win32.IsWindowEnabled( self.get_handle() )

    def is_maximized(self):
        """
        @see: L{maximize}
        @rtype:  bool
        @return: C{True} if the window is maximized.
        """
        return win32.IsZoomed( self.get_handle() )

    def is_minimized(self):
        """
        @see: L{minimize}
        @rtype:  bool
        @return: C{True} if the window is minimized.
        """
        return win32.IsIconic( self.get_handle() )

    def is_child(self):
        """
        @see: L{get_parent}
        @rtype:  bool
        @return: C{True} if the window is a child window.
        """
        return win32.IsChild( self.get_handle() )

    is_zoomed = is_maximized
    is_iconic = is_minimized

#------------------------------------------------------------------------------

    def enable(self):
        """
        Enable the user input for the window.

        @see: L{disable}

        @raise WindowsError: An error occured while processing this request.
        """
        win32.EnableWindow( self.get_handle(), True )

    def disable(self):
        """
        Disable the user input for the window.

        @see: L{enable}

        @raise WindowsError: An error occured while processing this request.
        """
        win32.EnableWindow( self.get_handle(), False )

    def show(self, bAsync = True):
        """
        Make the window visible.

        @see: L{hide}

        @type  bAsync: bool
        @param bAsync: Perform the request asynchronously.

        @raise WindowsError: An error occured while processing this request.
        """
        if bAsync:
            win32.ShowWindowAsync( self.get_handle(), win32.SW_SHOW )
        else:
            win32.ShowWindow( self.get_handle(), win32.SW_SHOW )

    def hide(self, bAsync = True):
        """
        Make the window invisible.

        @see: L{show}

        @type  bAsync: bool
        @param bAsync: Perform the request asynchronously.

        @raise WindowsError: An error occured while processing this request.
        """
        if bAsync:
            win32.ShowWindowAsync( self.get_handle(), win32.SW_HIDE )
        else:
            win32.ShowWindow( self.get_handle(), win32.SW_HIDE )

    def maximize(self, bAsync = True):
        """
        Maximize the window.

        @see: L{minimize}, L{restore}

        @type  bAsync: bool
        @param bAsync: Perform the request asynchronously.

        @raise WindowsError: An error occured while processing this request.
        """
        if bAsync:
            win32.ShowWindowAsync( self.get_handle(), win32.SW_MAXIMIZE )
        else:
            win32.ShowWindow( self.get_handle(), win32.SW_MAXIMIZE )

    def minimize(self, bAsync = True):
        """
        Minimize the window.

        @see: L{maximize}, L{restore}

        @type  bAsync: bool
        @param bAsync: Perform the request asynchronously.

        @raise WindowsError: An error occured while processing this request.
        """
        if bAsync:
            win32.ShowWindowAsync( self.get_handle(), win32.SW_MINIMIZE )
        else:
            win32.ShowWindow( self.get_handle(), win32.SW_MINIMIZE )

    def restore(self, bAsync = True):
        """
        Unmaximize and unminimize the window.

        @see: L{maximize}, L{minimize}

        @type  bAsync: bool
        @param bAsync: Perform the request asynchronously.

        @raise WindowsError: An error occured while processing this request.
        """
        if bAsync:
            win32.ShowWindowAsync( self.get_handle(), win32.SW_RESTORE )
        else:
            win32.ShowWindow( self.get_handle(), win32.SW_RESTORE )

    def move(self, x = None, y = None, width = None, height = None,
                                                            bRepaint = True):
        """
        Moves and/or resizes the window.

        @note: This is request is performed syncronously.

        @type  x: int
        @param x: (Optional) New horizontal coordinate.

        @type  y: int
        @param y: (Optional) New vertical coordinate.

        @type  width: int
        @param width: (Optional) Desired window width.

        @type  height: int
        @param height: (Optional) Desired window height.

        @type  bRepaint: bool
        @param bRepaint:
            (Optional) C{True} if the window should be redrawn afterwards.

        @raise WindowsError: An error occured while processing this request.
        """
        if None in (x, y, width, height):
            rect = self.get_screen_rect()
            if x is None:
                x = rect.left
            if y is None:
                y = rect.top
            if width is None:
                width = rect.right - rect.left
            if height is None:
                height = rect.bottom - rect.top
        win32.MoveWindow(self.get_handle(), x, y, width, height, bRepaint)

    def kill(self):
        """
        Signals the program to quit.

        @note: This is an asyncronous request.

        @raise WindowsError: An error occured while processing this request.
        """
        self.post(win32.WM_QUIT)

    def send(self, uMsg, wParam = None, lParam = None, dwTimeout = None):
        """
        Send a low-level window message syncronically.

        @type  uMsg: int
        @param uMsg: Message code.

        @param wParam:
            The type and meaning of this parameter depends on the message.

        @param lParam:
            The type and meaning of this parameter depends on the message.

        @param dwTimeout: Optional timeout for the operation.
            Use C{None} to wait indefinitely.

        @rtype:  int
        @return: The meaning of the return value depends on the window message.
            Typically a value of C{0} means an error occured. You can get the
            error code by calling L{win32.GetLastError}.
        """
        if dwTimeout is None:
            return win32.SendMessage(self.get_handle(), uMsg, wParam, lParam)
        return win32.SendMessageTimeout(
            self.get_handle(), uMsg, wParam, lParam,
            win32.SMTO_ABORTIFHUNG | win32.SMTO_ERRORONEXIT, dwTimeout)

    def post(self, uMsg, wParam = None, lParam = None):
        """
        Post a low-level window message asyncronically.

        @type  uMsg: int
        @param uMsg: Message code.

        @param wParam:
            The type and meaning of this parameter depends on the message.

        @param lParam:
            The type and meaning of this parameter depends on the message.

        @raise WindowsError: An error occured while sending the message.
        """
        win32.PostMessage(self.get_handle(), uMsg, wParam, lParam)
