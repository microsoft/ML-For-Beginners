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
System settings.

@group Instrumentation:
    System
"""

from __future__ import with_statement

__revision__ = "$Id$"

__all__ = ['System']

from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
                 classproperty
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window

import sys
import os
import ctypes
import warnings

from os import path, getenv

#==============================================================================

class System (_ProcessContainer):
    """
    Interface to a batch of processes, plus some system wide settings.
    Contains a snapshot of processes.

    @group Platform settings:
        arch, bits, os, wow64, pageSize

    @group Instrumentation:
        find_window, get_window_at, get_foreground_window,
        get_desktop_window, get_shell_window

    @group Debugging:
        load_dbghelp, fix_symbol_store_path,
        request_debug_privileges, drop_debug_privileges

    @group Postmortem debugging:
        get_postmortem_debugger, set_postmortem_debugger,
        get_postmortem_exclusion_list, add_to_postmortem_exclusion_list,
        remove_from_postmortem_exclusion_list

    @group System services:
        get_services, get_active_services,
        start_service, stop_service,
        pause_service, resume_service,
        get_service_display_name, get_service_from_display_name

    @group Permissions and privileges:
        request_privileges, drop_privileges, adjust_privileges, is_admin

    @group Miscellaneous global settings:
        set_kill_on_exit_mode, read_msr, write_msr, enable_step_on_branch_mode,
        get_last_branch_location

    @type arch: str
    @cvar arch: Name of the processor architecture we're running on.
        For more details see L{win32.version._get_arch}.

    @type bits: int
    @cvar bits: Size of the machine word in bits for the current architecture.
        For more details see L{win32.version._get_bits}.

    @type os: str
    @cvar os: Name of the Windows version we're runing on.
        For more details see L{win32.version._get_os}.

    @type wow64: bool
    @cvar wow64: C{True} if the debugger is a 32 bits process running in a 64
        bits version of Windows, C{False} otherwise.

    @type pageSize: int
    @cvar pageSize: Page size in bytes. Defaults to 0x1000 but it's
        automatically updated on runtime when importing the module.

    @type registry: L{Registry}
    @cvar registry: Windows Registry for this machine.
    """

    arch  = win32.arch
    bits  = win32.bits
    os    = win32.os
    wow64 = win32.wow64

    @classproperty
    def pageSize(cls):
        pageSize = MemoryAddresses.pageSize
        cls.pageSize = pageSize
        return pageSize

    registry = Registry()

#------------------------------------------------------------------------------

    @staticmethod
    def find_window(className = None, windowName = None):
        """
        Find the first top-level window in the current desktop to match the
        given class name and/or window name. If neither are provided any
        top-level window will match.

        @see: L{get_window_at}

        @type  className: str
        @param className: (Optional) Class name of the window to find.
            If C{None} or not used any class name will match the search.

        @type  windowName: str
        @param windowName: (Optional) Caption text of the window to find.
            If C{None} or not used any caption text will match the search.

        @rtype:  L{Window} or None
        @return: A window that matches the request. There may be more matching
            windows, but this method only returns one. If no matching window
            is found, the return value is C{None}.

        @raise WindowsError: An error occured while processing this request.
        """
        # I'd love to reverse the order of the parameters
        # but that might create some confusion. :(
        hWnd = win32.FindWindow(className, windowName)
        if hWnd:
            return Window(hWnd)

    @staticmethod
    def get_window_at(x, y):
        """
        Get the window located at the given coordinates in the desktop.
        If no such window exists an exception is raised.

        @see: L{find_window}

        @type  x: int
        @param x: Horizontal coordinate.
        @type  y: int
        @param y: Vertical coordinate.

        @rtype:  L{Window}
        @return: Window at the requested position. If no such window
            exists a C{WindowsError} exception is raised.

        @raise WindowsError: An error occured while processing this request.
        """
        return Window( win32.WindowFromPoint( (x, y) ) )

    @staticmethod
    def get_foreground_window():
        """
        @rtype:  L{Window}
        @return: Returns the foreground window.
        @raise WindowsError: An error occured while processing this request.
        """
        return Window( win32.GetForegroundWindow() )

    @staticmethod
    def get_desktop_window():
        """
        @rtype:  L{Window}
        @return: Returns the desktop window.
        @raise WindowsError: An error occured while processing this request.
        """
        return Window( win32.GetDesktopWindow() )

    @staticmethod
    def get_shell_window():
        """
        @rtype:  L{Window}
        @return: Returns the shell window.
        @raise WindowsError: An error occured while processing this request.
        """
        return Window( win32.GetShellWindow() )

#------------------------------------------------------------------------------

    @classmethod
    def request_debug_privileges(cls, bIgnoreExceptions = False):
        """
        Requests debug privileges.

        This may be needed to debug processes running as SYSTEM
        (such as services) since Windows XP.

        @type  bIgnoreExceptions: bool
        @param bIgnoreExceptions: C{True} to ignore any exceptions that may be
            raised when requesting debug privileges.

        @rtype:  bool
        @return: C{True} on success, C{False} on failure.

        @raise WindowsError: Raises an exception on error, unless
            C{bIgnoreExceptions} is C{True}.
        """
        try:
            cls.request_privileges(win32.SE_DEBUG_NAME)
            return True
        except Exception:
            if not bIgnoreExceptions:
                raise
        return False

    @classmethod
    def drop_debug_privileges(cls, bIgnoreExceptions = False):
        """
        Drops debug privileges.

        This may be needed to avoid being detected
        by certain anti-debug tricks.

        @type  bIgnoreExceptions: bool
        @param bIgnoreExceptions: C{True} to ignore any exceptions that may be
            raised when dropping debug privileges.

        @rtype:  bool
        @return: C{True} on success, C{False} on failure.

        @raise WindowsError: Raises an exception on error, unless
            C{bIgnoreExceptions} is C{True}.
        """
        try:
            cls.drop_privileges(win32.SE_DEBUG_NAME)
            return True
        except Exception:
            if not bIgnoreExceptions:
                raise
        return False

    @classmethod
    def request_privileges(cls, *privileges):
        """
        Requests privileges.

        @type  privileges: int...
        @param privileges: Privileges to request.

        @raise WindowsError: Raises an exception on error.
        """
        cls.adjust_privileges(True, privileges)

    @classmethod
    def drop_privileges(cls, *privileges):
        """
        Drops privileges.

        @type  privileges: int...
        @param privileges: Privileges to drop.

        @raise WindowsError: Raises an exception on error.
        """
        cls.adjust_privileges(False, privileges)

    @staticmethod
    def adjust_privileges(state, privileges):
        """
        Requests or drops privileges.

        @type  state: bool
        @param state: C{True} to request, C{False} to drop.

        @type  privileges: list(int)
        @param privileges: Privileges to request or drop.

        @raise WindowsError: Raises an exception on error.
        """
        with win32.OpenProcessToken(win32.GetCurrentProcess(),
                                win32.TOKEN_ADJUST_PRIVILEGES) as hToken:
            NewState = ( (priv, state) for priv in privileges )
            win32.AdjustTokenPrivileges(hToken, NewState)

    @staticmethod
    def is_admin():
        """
        @rtype:  bool
        @return: C{True} if the current user as Administrator privileges,
            C{False} otherwise. Since Windows Vista and above this means if
            the current process is running with UAC elevation or not.
        """
        return win32.IsUserAnAdmin()

#------------------------------------------------------------------------------

    __binary_types = {
        win32.VFT_APP:        "application",
        win32.VFT_DLL:        "dynamic link library",
        win32.VFT_STATIC_LIB: "static link library",
        win32.VFT_FONT:       "font",
        win32.VFT_DRV:        "driver",
        win32.VFT_VXD:        "legacy driver",
    }

    __driver_types = {
        win32.VFT2_DRV_COMM:                "communications driver",
        win32.VFT2_DRV_DISPLAY:             "display driver",
        win32.VFT2_DRV_INSTALLABLE:         "installable driver",
        win32.VFT2_DRV_KEYBOARD:            "keyboard driver",
        win32.VFT2_DRV_LANGUAGE:            "language driver",
        win32.VFT2_DRV_MOUSE:               "mouse driver",
        win32.VFT2_DRV_NETWORK:             "network driver",
        win32.VFT2_DRV_PRINTER:             "printer driver",
        win32.VFT2_DRV_SOUND:               "sound driver",
        win32.VFT2_DRV_SYSTEM:              "system driver",
        win32.VFT2_DRV_VERSIONED_PRINTER:   "versioned printer driver",
    }

    __font_types = {
        win32.VFT2_FONT_RASTER:   "raster font",
        win32.VFT2_FONT_TRUETYPE: "TrueType font",
        win32.VFT2_FONT_VECTOR:   "vector font",
    }

    __months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )

    __days_of_the_week = (
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    )

    @classmethod
    def get_file_version_info(cls, filename):
        """
        Get the program version from an executable file, if available.

        @type  filename: str
        @param filename: Pathname to the executable file to query.

        @rtype: tuple(str, str, bool, bool, str, str)
        @return: Tuple with version information extracted from the executable
            file metadata, containing the following:
             - File version number (C{"major.minor"}).
             - Product version number (C{"major.minor"}).
             - C{True} for debug builds, C{False} for production builds.
             - C{True} for legacy OS builds (DOS, OS/2, Win16),
               C{False} for modern OS builds.
             - Binary file type.
               May be one of the following values:
                - "application"
                - "dynamic link library"
                - "static link library"
                - "font"
                - "raster font"
                - "TrueType font"
                - "vector font"
                - "driver"
                - "communications driver"
                - "display driver"
                - "installable driver"
                - "keyboard driver"
                - "language driver"
                - "legacy driver"
                - "mouse driver"
                - "network driver"
                - "printer driver"
                - "sound driver"
                - "system driver"
                - "versioned printer driver"
             - Binary creation timestamp.
            Any of the fields may be C{None} if not available.

        @raise WindowsError: Raises an exception on error.
        """

        # Get the file version info structure.
        pBlock = win32.GetFileVersionInfo(filename)
        pBuffer, dwLen = win32.VerQueryValue(pBlock, "\\")
        if dwLen != ctypes.sizeof(win32.VS_FIXEDFILEINFO):
            raise ctypes.WinError(win32.ERROR_BAD_LENGTH)
        pVersionInfo = ctypes.cast(pBuffer,
                                   ctypes.POINTER(win32.VS_FIXEDFILEINFO))
        VersionInfo = pVersionInfo.contents
        if VersionInfo.dwSignature != 0xFEEF04BD:
            raise ctypes.WinError(win32.ERROR_BAD_ARGUMENTS)

        # File and product versions.
        FileVersion = "%d.%d" % (VersionInfo.dwFileVersionMS,
                                 VersionInfo.dwFileVersionLS)
        ProductVersion = "%d.%d" % (VersionInfo.dwProductVersionMS,
                                    VersionInfo.dwProductVersionLS)

        # Debug build?
        if VersionInfo.dwFileFlagsMask & win32.VS_FF_DEBUG:
            DebugBuild = (VersionInfo.dwFileFlags & win32.VS_FF_DEBUG) != 0
        else:
            DebugBuild = None

        # Legacy OS build?
        LegacyBuild = (VersionInfo.dwFileOS != win32.VOS_NT_WINDOWS32)

        # File type.
        FileType = cls.__binary_types.get(VersionInfo.dwFileType)
        if VersionInfo.dwFileType == win32.VFT_DRV:
            FileType = cls.__driver_types.get(VersionInfo.dwFileSubtype)
        elif VersionInfo.dwFileType == win32.VFT_FONT:
            FileType = cls.__font_types.get(VersionInfo.dwFileSubtype)

        # Timestamp, ex: "Monday, July 7, 2013 (12:20:50.126)".
        # FIXME: how do we know the time zone?
        FileDate = (VersionInfo.dwFileDateMS << 32) + VersionInfo.dwFileDateLS
        if FileDate:
            CreationTime = win32.FileTimeToSystemTime(FileDate)
            CreationTimestamp = "%s, %s %d, %d (%d:%d:%d.%d)" % (
                cls.__days_of_the_week[CreationTime.wDayOfWeek],
                cls.__months[CreationTime.wMonth],
                CreationTime.wDay,
                CreationTime.wYear,
                CreationTime.wHour,
                CreationTime.wMinute,
                CreationTime.wSecond,
                CreationTime.wMilliseconds,
            )
        else:
            CreationTimestamp = None

        # Return the file version info.
        return (
            FileVersion,
            ProductVersion,
            DebugBuild,
            LegacyBuild,
            FileType,
            CreationTimestamp,
        )

#------------------------------------------------------------------------------

    # Locations for dbghelp.dll.
    #  Unfortunately, Microsoft started bundling WinDbg with the
    #  platform SDK, so the install directories may vary across
    #  versions and platforms.
    __dbghelp_locations = {

        # Intel 64 bits.
        win32.ARCH_AMD64: set([

            # WinDbg bundled with the SDK, version 8.0.
            path.join(
                getenv("ProgramFiles", "C:\\Program Files"),
                "Windows Kits",
                "8.0",
                "Debuggers",
                "x64",
                "dbghelp.dll"),
            path.join(
                getenv("ProgramW6432", getenv("ProgramFiles",
                                              "C:\\Program Files")),
                "Windows Kits",
                "8.0",
                "Debuggers",
                "x64",
                "dbghelp.dll"),

            # Old standalone versions of WinDbg.
            path.join(
                getenv("ProgramFiles", "C:\\Program Files"),
                "Debugging Tools for Windows (x64)",
                "dbghelp.dll"),
        ]),

        # Intel 32 bits.
        win32.ARCH_I386 : set([

            # WinDbg bundled with the SDK, version 8.0.
            path.join(
                getenv("ProgramFiles", "C:\\Program Files"),
                "Windows Kits",
                "8.0",
                "Debuggers",
                "x86",
                "dbghelp.dll"),
            path.join(
                getenv("ProgramW6432", getenv("ProgramFiles",
                                              "C:\\Program Files")),
                "Windows Kits",
                "8.0",
                "Debuggers",
                "x86",
                "dbghelp.dll"),

            # Old standalone versions of WinDbg.
            path.join(
                getenv("ProgramFiles", "C:\\Program Files"),
                "Debugging Tools for Windows (x86)",
                "dbghelp.dll"),

            # Version shipped with Windows.
            path.join(
                getenv("ProgramFiles", "C:\\Program Files"),
                "Debugging Tools for Windows (x86)",
                "dbghelp.dll"),
        ]),
    }

    @classmethod
    def load_dbghelp(cls, pathname = None):
        """
        Load the specified version of the C{dbghelp.dll} library.

        This library is shipped with the Debugging Tools for Windows, and it's
        required to load debug symbols.

        Normally you don't need to call this method, as WinAppDbg already tries
        to load the latest version automatically - but it may come in handy if
        the Debugging Tools are installed in a non standard folder.

        Example::
            from winappdbg import Debug

            def simple_debugger( argv ):

                # Instance a Debug object, passing it the event handler callback
                debug = Debug( my_event_handler )
                try:

                    # Load a specific dbghelp.dll file
                    debug.system.load_dbghelp("C:\\Some folder\\dbghelp.dll")

                    # Start a new process for debugging
                    debug.execv( argv )

                    # Wait for the debugee to finish
                    debug.loop()

                # Stop the debugger
                finally:
                    debug.stop()

        @see: U{http://msdn.microsoft.com/en-us/library/ms679294(VS.85).aspx}

        @type  pathname: str
        @param pathname:
            (Optional) Full pathname to the C{dbghelp.dll} library.
            If not provided this method will try to autodetect it.

        @rtype:  ctypes.WinDLL
        @return: Loaded instance of C{dbghelp.dll}.

        @raise NotImplementedError: This feature was not implemented for the
            current architecture.

        @raise WindowsError: An error occured while processing this request.
        """

        # If an explicit pathname was not given, search for the library.
        if not pathname:

            # Under WOW64 we'll treat AMD64 as I386.
            arch = win32.arch
            if arch == win32.ARCH_AMD64 and win32.bits == 32:
                arch = win32.ARCH_I386

            # Check if the architecture is supported.
            if not arch in cls.__dbghelp_locations:
                msg = "Architecture %s is not currently supported."
                raise NotImplementedError(msg  % arch)

            # Grab all versions of the library we can find.
            found = []
            for pathname in cls.__dbghelp_locations[arch]:
                if path.isfile(pathname):
                    try:
                        f_ver, p_ver = cls.get_file_version_info(pathname)[:2]
                    except WindowsError:
                        msg = "Failed to parse file version metadata for: %s"
                        warnings.warn(msg % pathname)
                    if not f_ver:
                        f_ver = p_ver
                    elif p_ver and p_ver > f_ver:
                        f_ver = p_ver
                    found.append( (f_ver, pathname) )

            # If we found any, use the newest version.
            if found:
                found.sort()
                pathname = found.pop()[1]

            # If we didn't find any, trust the default DLL search algorithm.
            else:
                pathname = "dbghelp.dll"

        # Load the library.
        dbghelp = ctypes.windll.LoadLibrary(pathname)

        # Set it globally as the library to be used.
        ctypes.windll.dbghelp = dbghelp

        # Return the library.
        return dbghelp

    @staticmethod
    def fix_symbol_store_path(symbol_store_path = None,
                              remote = True,
                              force = False):
        """
        Fix the symbol store path. Equivalent to the C{.symfix} command in
        Microsoft WinDbg.

        If the symbol store path environment variable hasn't been set, this
        method will provide a default one.

        @type  symbol_store_path: str or None
        @param symbol_store_path: (Optional) Symbol store path to set.

        @type  remote: bool
        @param remote: (Optional) Defines the symbol store path to set when the
            C{symbol_store_path} is C{None}.

            If C{True} the default symbol store path is set to the Microsoft
            symbol server. Debug symbols will be downloaded through HTTP.
            This gives the best results but is also quite slow.

            If C{False} the default symbol store path is set to the local
            cache only. This prevents debug symbols from being downloaded and
            is faster, but unless you've installed the debug symbols on this
            machine or downloaded them in a previous debugging session, some
            symbols may be missing.

            If the C{symbol_store_path} argument is not C{None}, this argument
            is ignored entirely.

        @type  force: bool
        @param force: (Optional) If C{True} the new symbol store path is set
            always. If C{False} the new symbol store path is only set if
            missing.

            This allows you to call this method preventively to ensure the
            symbol server is always set up correctly when running your script,
            but without messing up whatever configuration the user has.

            Example::
                from winappdbg import Debug, System

                def simple_debugger( argv ):

                    # Instance a Debug object
                    debug = Debug( MyEventHandler() )
                    try:

                        # Make sure the remote symbol store is set
                        System.fix_symbol_store_path(remote = True,
                                                      force = False)

                        # Start a new process for debugging
                        debug.execv( argv )

                        # Wait for the debugee to finish
                        debug.loop()

                    # Stop the debugger
                    finally:
                        debug.stop()

        @rtype:  str or None
        @return: The previously set symbol store path if any,
            otherwise returns C{None}.
        """
        try:
            if symbol_store_path is None:
                local_path = "C:\\SYMBOLS"
                if not path.isdir(local_path):
                    local_path = "C:\\Windows\\Symbols"
                    if not path.isdir(local_path):
                        local_path = path.abspath(".")
                if remote:
                    symbol_store_path = (
                        "cache*;SRV*"
                        + local_path +
                        "*"
                        "http://msdl.microsoft.com/download/symbols"
                    )
                else:
                    symbol_store_path = "cache*;SRV*" + local_path
            previous = os.environ.get("_NT_SYMBOL_PATH", None)
            if not previous or force:
                os.environ["_NT_SYMBOL_PATH"] = symbol_store_path
            return previous
        except Exception:
            e = sys.exc_info()[1]
            warnings.warn("Cannot fix symbol path, reason: %s" % str(e),
                          RuntimeWarning)

#------------------------------------------------------------------------------

    @staticmethod
    def set_kill_on_exit_mode(bKillOnExit = False):
        """
        Defines the behavior of the debugged processes when the debugging
        thread dies. This method only affects the calling thread.

        Works on the following platforms:

         - Microsoft Windows XP and above.
         - Wine (Windows Emulator).

        Fails on the following platforms:

         - Microsoft Windows 2000 and below.
         - ReactOS.

        @type  bKillOnExit: bool
        @param bKillOnExit: C{True} to automatically kill processes when the
            debugger thread dies. C{False} to automatically detach from
            processes when the debugger thread dies.

        @rtype:  bool
        @return: C{True} on success, C{False} on error.

        @note:
            This call will fail if a debug port was not created. That is, if
            the debugger isn't attached to at least one process. For more info
            see: U{http://msdn.microsoft.com/en-us/library/ms679307.aspx}
        """
        try:
            # won't work before calling CreateProcess or DebugActiveProcess
            win32.DebugSetProcessKillOnExit(bKillOnExit)
        except (AttributeError, WindowsError):
            return False
        return True

    @staticmethod
    def read_msr(address):
        """
        Read the contents of the specified MSR (Machine Specific Register).

        @type  address: int
        @param address: MSR to read.

        @rtype:  int
        @return: Value of the specified MSR.

        @raise WindowsError:
            Raises an exception on error.

        @raise NotImplementedError:
            Current architecture is not C{i386} or C{amd64}.

        @warning:
            It could potentially brick your machine.
            It works on my machine, but your mileage may vary.
        """
        if win32.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
            raise NotImplementedError(
                "MSR reading is only supported on i386 or amd64 processors.")
        msr         = win32.SYSDBG_MSR()
        msr.Address = address
        msr.Data    = 0
        win32.NtSystemDebugControl(win32.SysDbgReadMsr,
                                   InputBuffer  = msr,
                                   OutputBuffer = msr)
        return msr.Data

    @staticmethod
    def write_msr(address, value):
        """
        Set the contents of the specified MSR (Machine Specific Register).

        @type  address: int
        @param address: MSR to write.

        @type  value: int
        @param value: Contents to write on the MSR.

        @raise WindowsError:
            Raises an exception on error.

        @raise NotImplementedError:
            Current architecture is not C{i386} or C{amd64}.

        @warning:
            It could potentially brick your machine.
            It works on my machine, but your mileage may vary.
        """
        if win32.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
            raise NotImplementedError(
                "MSR writing is only supported on i386 or amd64 processors.")
        msr         = win32.SYSDBG_MSR()
        msr.Address = address
        msr.Data    = value
        win32.NtSystemDebugControl(win32.SysDbgWriteMsr, InputBuffer = msr)

    @classmethod
    def enable_step_on_branch_mode(cls):
        """
        When tracing, call this on every single step event
        for step on branch mode.

        @raise WindowsError:
            Raises C{ERROR_DEBUGGER_INACTIVE} if the debugger is not attached
            to least one process.

        @raise NotImplementedError:
            Current architecture is not C{i386} or C{amd64}.

        @warning:
            This method uses the processor's machine specific registers (MSR).
            It could potentially brick your machine.
            It works on my machine, but your mileage may vary.

        @note:
            It doesn't seem to work in VMWare or VirtualBox machines.
            Maybe it fails in other virtualization/emulation environments,
            no extensive testing was made so far.
        """
        cls.write_msr(DebugRegister.DebugCtlMSR,
                DebugRegister.BranchTrapFlag | DebugRegister.LastBranchRecord)

    @classmethod
    def get_last_branch_location(cls):
        """
        Returns the source and destination addresses of the last taken branch.

        @rtype: tuple( int, int )
        @return: Source and destination addresses of the last taken branch.

        @raise WindowsError:
            Raises an exception on error.

        @raise NotImplementedError:
            Current architecture is not C{i386} or C{amd64}.

        @warning:
            This method uses the processor's machine specific registers (MSR).
            It could potentially brick your machine.
            It works on my machine, but your mileage may vary.

        @note:
            It doesn't seem to work in VMWare or VirtualBox machines.
            Maybe it fails in other virtualization/emulation environments,
            no extensive testing was made so far.
        """
        LastBranchFromIP = cls.read_msr(DebugRegister.LastBranchFromIP)
        LastBranchToIP   = cls.read_msr(DebugRegister.LastBranchToIP)
        return ( LastBranchFromIP, LastBranchToIP )

#------------------------------------------------------------------------------

    @classmethod
    def get_postmortem_debugger(cls, bits = None):
        """
        Returns the postmortem debugging settings from the Registry.

        @see: L{set_postmortem_debugger}

        @type  bits: int
        @param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
            64 bits debugger. Set to {None} for the default (L{System.bits}.

        @rtype:  tuple( str, bool, int )
        @return: A tuple containing the command line string to the postmortem
            debugger, a boolean specifying if user interaction is allowed
            before attaching, and an integer specifying a user defined hotkey.
            Any member of the tuple may be C{None}.
            See L{set_postmortem_debugger} for more details.

        @raise WindowsError:
            Raises an exception on error.
        """
        if bits is None:
            bits = cls.bits
        elif bits not in (32, 64):
            raise NotImplementedError("Unknown architecture (%r bits)" % bits)

        if bits == 32 and cls.bits == 64:
            keyname = 'HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug'
        else:
            keyname = 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug'

        key = cls.registry[keyname]

        debugger = key.get('Debugger')
        auto     = key.get('Auto')
        hotkey   = key.get('UserDebuggerHotkey')

        if auto is not None:
            auto = bool(auto)

        return (debugger, auto, hotkey)

    @classmethod
    def get_postmortem_exclusion_list(cls, bits = None):
        """
        Returns the exclusion list for the postmortem debugger.

        @see: L{get_postmortem_debugger}

        @type  bits: int
        @param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
            64 bits debugger. Set to {None} for the default (L{System.bits}).

        @rtype:  list( str )
        @return: List of excluded application filenames.

        @raise WindowsError:
            Raises an exception on error.
        """
        if bits is None:
            bits = cls.bits
        elif bits not in (32, 64):
            raise NotImplementedError("Unknown architecture (%r bits)" % bits)

        if bits == 32 and cls.bits == 64:
            keyname = 'HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'
        else:
            keyname = 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'

        try:
            key = cls.registry[keyname]
        except KeyError:
            return []

        return [name for (name, enabled) in key.items() if enabled]

    @classmethod
    def set_postmortem_debugger(cls, cmdline,
                                auto = None, hotkey = None, bits = None):
        """
        Sets the postmortem debugging settings in the Registry.

        @warning: This method requires administrative rights.

        @see: L{get_postmortem_debugger}

        @type  cmdline: str
        @param cmdline: Command line to the new postmortem debugger.
            When the debugger is invoked, the first "%ld" is replaced with the
            process ID and the second "%ld" is replaced with the event handle.
            Don't forget to enclose the program filename in double quotes if
            the path contains spaces.

        @type  auto: bool
        @param auto: Set to C{True} if no user interaction is allowed, C{False}
            to prompt a confirmation dialog before attaching.
            Use C{None} to leave this value unchanged.

        @type  hotkey: int
        @param hotkey: Virtual key scan code for the user defined hotkey.
            Use C{0} to disable the hotkey.
            Use C{None} to leave this value unchanged.

        @type  bits: int
        @param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
            64 bits debugger. Set to {None} for the default (L{System.bits}).

        @rtype:  tuple( str, bool, int )
        @return: Previously defined command line and auto flag.

        @raise WindowsError:
            Raises an exception on error.
        """
        if bits is None:
            bits = cls.bits
        elif bits not in (32, 64):
            raise NotImplementedError("Unknown architecture (%r bits)" % bits)

        if bits == 32 and cls.bits == 64:
            keyname = 'HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug'
        else:
            keyname = 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug'

        key = cls.registry[keyname]

        if cmdline is not None:
            key['Debugger'] = cmdline
        if auto is not None:
            key['Auto'] = int(bool(auto))
        if hotkey is not None:
            key['UserDebuggerHotkey'] = int(hotkey)

    @classmethod
    def add_to_postmortem_exclusion_list(cls, pathname, bits = None):
        """
        Adds the given filename to the exclusion list for postmortem debugging.

        @warning: This method requires administrative rights.

        @see: L{get_postmortem_exclusion_list}

        @type  pathname: str
        @param pathname:
            Application pathname to exclude from postmortem debugging.

        @type  bits: int
        @param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
            64 bits debugger. Set to {None} for the default (L{System.bits}).

        @raise WindowsError:
            Raises an exception on error.
        """
        if bits is None:
            bits = cls.bits
        elif bits not in (32, 64):
            raise NotImplementedError("Unknown architecture (%r bits)" % bits)

        if bits == 32 and cls.bits == 64:
            keyname = 'HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'
        else:
            keyname = 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'

        try:
            key = cls.registry[keyname]
        except KeyError:
            key = cls.registry.create(keyname)

        key[pathname] = 1

    @classmethod
    def remove_from_postmortem_exclusion_list(cls, pathname, bits = None):
        """
        Removes the given filename to the exclusion list for postmortem
        debugging from the Registry.

        @warning: This method requires administrative rights.

        @warning: Don't ever delete entries you haven't created yourself!
            Some entries are set by default for your version of Windows.
            Deleting them might deadlock your system under some circumstances.

            For more details see:
            U{http://msdn.microsoft.com/en-us/library/bb204634(v=vs.85).aspx}

        @see: L{get_postmortem_exclusion_list}

        @type  pathname: str
        @param pathname: Application pathname to remove from the postmortem
            debugging exclusion list.

        @type  bits: int
        @param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
            64 bits debugger. Set to {None} for the default (L{System.bits}).

        @raise WindowsError:
            Raises an exception on error.
        """
        if bits is None:
            bits = cls.bits
        elif bits not in (32, 64):
            raise NotImplementedError("Unknown architecture (%r bits)" % bits)

        if bits == 32 and cls.bits == 64:
            keyname = 'HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'
        else:
            keyname = 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'

        try:
            key = cls.registry[keyname]
        except KeyError:
            return

        try:
            del key[pathname]
        except KeyError:
            return

#------------------------------------------------------------------------------

    @staticmethod
    def get_services():
        """
        Retrieve a list of all system services.

        @see: L{get_active_services},
            L{start_service}, L{stop_service},
            L{pause_service}, L{resume_service}

        @rtype:  list( L{win32.ServiceStatusProcessEntry} )
        @return: List of service status descriptors.
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_ENUMERATE_SERVICE
            ) as hSCManager:
                try:
                    return win32.EnumServicesStatusEx(hSCManager)
                except AttributeError:
                    return win32.EnumServicesStatus(hSCManager)

    @staticmethod
    def get_active_services():
        """
        Retrieve a list of all active system services.

        @see: L{get_services},
            L{start_service}, L{stop_service},
            L{pause_service}, L{resume_service}

        @rtype:  list( L{win32.ServiceStatusProcessEntry} )
        @return: List of service status descriptors.
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_ENUMERATE_SERVICE
        ) as hSCManager:
            return [ entry for entry in win32.EnumServicesStatusEx(hSCManager,
                        dwServiceType  = win32.SERVICE_WIN32,
                        dwServiceState = win32.SERVICE_ACTIVE) \
                    if entry.ProcessId ]

    @staticmethod
    def get_service(name):
        """
        Get the service descriptor for the given service name.

        @see: L{start_service}, L{stop_service},
            L{pause_service}, L{resume_service}

        @type  name: str
        @param name: Service unique name. You can get this value from the
            C{ServiceName} member of the service descriptors returned by
            L{get_services} or L{get_active_services}.

        @rtype:  L{win32.ServiceStatusProcess}
        @return: Service status descriptor.
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_ENUMERATE_SERVICE
        ) as hSCManager:
            with win32.OpenService(hSCManager, name,
                                   dwDesiredAccess = win32.SERVICE_QUERY_STATUS
            ) as hService:
                try:
                    return win32.QueryServiceStatusEx(hService)
                except AttributeError:
                    return win32.QueryServiceStatus(hService)

    @staticmethod
    def get_service_display_name(name):
        """
        Get the service display name for the given service name.

        @see: L{get_service}

        @type  name: str
        @param name: Service unique name. You can get this value from the
            C{ServiceName} member of the service descriptors returned by
            L{get_services} or L{get_active_services}.

        @rtype:  str
        @return: Service display name.
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_ENUMERATE_SERVICE
        ) as hSCManager:
            return win32.GetServiceDisplayName(hSCManager, name)

    @staticmethod
    def get_service_from_display_name(displayName):
        """
        Get the service unique name given its display name.

        @see: L{get_service}

        @type  displayName: str
        @param displayName: Service display name. You can get this value from
            the C{DisplayName} member of the service descriptors returned by
            L{get_services} or L{get_active_services}.

        @rtype:  str
        @return: Service unique name.
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_ENUMERATE_SERVICE
        ) as hSCManager:
            return win32.GetServiceKeyName(hSCManager, displayName)

    @staticmethod
    def start_service(name, argv = None):
        """
        Start the service given by name.

        @warn: This method requires UAC elevation in Windows Vista and above.

        @see: L{stop_service}, L{pause_service}, L{resume_service}

        @type  name: str
        @param name: Service unique name. You can get this value from the
            C{ServiceName} member of the service descriptors returned by
            L{get_services} or L{get_active_services}.
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_CONNECT
        ) as hSCManager:
            with win32.OpenService(hSCManager, name,
                                   dwDesiredAccess = win32.SERVICE_START
            ) as hService:
                win32.StartService(hService)

    @staticmethod
    def stop_service(name):
        """
        Stop the service given by name.

        @warn: This method requires UAC elevation in Windows Vista and above.

        @see: L{get_services}, L{get_active_services},
            L{start_service}, L{pause_service}, L{resume_service}
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_CONNECT
        ) as hSCManager:
            with win32.OpenService(hSCManager, name,
                                   dwDesiredAccess = win32.SERVICE_STOP
            ) as hService:
                win32.ControlService(hService, win32.SERVICE_CONTROL_STOP)

    @staticmethod
    def pause_service(name):
        """
        Pause the service given by name.

        @warn: This method requires UAC elevation in Windows Vista and above.

        @note: Not all services support this.

        @see: L{get_services}, L{get_active_services},
            L{start_service}, L{stop_service}, L{resume_service}
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_CONNECT
        ) as hSCManager:
            with win32.OpenService(hSCManager, name,
                                dwDesiredAccess = win32.SERVICE_PAUSE_CONTINUE
            ) as hService:
                win32.ControlService(hService, win32.SERVICE_CONTROL_PAUSE)

    @staticmethod
    def resume_service(name):
        """
        Resume the service given by name.

        @warn: This method requires UAC elevation in Windows Vista and above.

        @note: Not all services support this.

        @see: L{get_services}, L{get_active_services},
            L{start_service}, L{stop_service}, L{pause_service}
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_CONNECT
        ) as hSCManager:
            with win32.OpenService(hSCManager, name,
                                dwDesiredAccess = win32.SERVICE_PAUSE_CONTINUE
            ) as hService:
                win32.ControlService(hService, win32.SERVICE_CONTROL_CONTINUE)

    # TODO: create_service, delete_service
