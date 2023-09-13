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
Windows application debugging engine for Python.

by Mario Vilas (mvilas at gmail.com)

Project: U{http://sourceforge.net/projects/winappdbg/}

Web:     U{http://winappdbg.sourceforge.net/}

Blog:    U{http://breakingcode.wordpress.com}

@group Debugging:
    Debug, EventHandler, EventSift, DebugLog

@group Instrumentation:
    System, Process, Thread, Module, Window, Registry

@group Disassemblers:
    Disassembler,
    BeaEngine, DistormEngine, PyDasmEngine

@group Crash reporting:
    Crash, CrashDump, CrashDAO, CrashDictionary

@group Memory search:
    Search,
    Pattern,
    BytePattern,
    TextPattern,
    RegExpPattern,
    HexPattern

@group Debug events:
    Event,
    NoEvent,
    CreateProcessEvent,
    CreateThreadEvent,
    ExitProcessEvent,
    ExitThreadEvent,
    LoadDLLEvent,
    UnloadDLLEvent,
    OutputDebugStringEvent,
    RIPEvent,
    ExceptionEvent

@group Win32 API wrappers:
    win32, Handle, ProcessHandle, ThreadHandle, FileHandle

@group Helpers:
    HexInput, HexOutput, HexDump, Color, Table, Logger,
    PathOperations,
    MemoryAddresses,
    CustomAddressIterator,
    DataAddressIterator,
    ImageAddressIterator,
    MappedAddressIterator,
    ExecutableAddressIterator,
    ReadableAddressIterator,
    WriteableAddressIterator,
    ExecutableAndWriteableAddressIterator,
    DebugRegister,
    Regenerator

@group Warnings:
    MixedBitsWarning, BreakpointWarning, BreakpointCallbackWarning,
    EventCallbackWarning, DebugSymbolsWarning, CrashWarning

@group Deprecated classes:
    CrashContainer, CrashTable, CrashTableMSSQL,
    VolatileCrashContainer, DummyCrashContainer

@type version_number: float
@var  version_number: This WinAppDbg major and minor version,
    as a floating point number. Use this for compatibility checking.

@type version: str
@var  version: This WinAppDbg release version,
    as a printable string. Use this to show to the user.

@undocumented: plugins
"""

__revision__ = "$Id$"

# List of all public symbols
__all__ =   [
                # Library version
                'version',
                'version_number',

                # from breakpoint import *
##                'Breakpoint',
##                'CodeBreakpoint',
##                'PageBreakpoint',
##                'HardwareBreakpoint',
##                'Hook',
##                'ApiHook',
##                'BufferWatch',
                'BreakpointWarning',
                'BreakpointCallbackWarning',

                # from crash import *
                'Crash',
                'CrashWarning',
                'CrashDictionary',
                'CrashContainer',
                'CrashTable',
                'CrashTableMSSQL',
                'VolatileCrashContainer',
                'DummyCrashContainer',

                # from debug import *
                'Debug',
                'MixedBitsWarning',

                # from disasm import *
                'Disassembler',
                'BeaEngine',
                'DistormEngine',
                'PyDasmEngine',

                # from event import *
                'EventHandler',
                'EventSift',
##                'EventFactory',
##                'EventDispatcher',
                'EventCallbackWarning',
                'Event',
##                'NoEvent',
                'CreateProcessEvent',
                'CreateThreadEvent',
                'ExitProcessEvent',
                'ExitThreadEvent',
                'LoadDLLEvent',
                'UnloadDLLEvent',
                'OutputDebugStringEvent',
                'RIPEvent',
                'ExceptionEvent',

                # from interactive import *
##                'ConsoleDebugger',

                # from module import *
                'Module',
                'DebugSymbolsWarning',

                # from process import *
                'Process',

                # from system import *
                'System',

                # from search import *
                'Search',
                'Pattern',
                'BytePattern',
                'TextPattern',
                'RegExpPattern',
                'HexPattern',

                # from registry import *
                'Registry',

                # from textio import *
                'HexDump',
                'HexInput',
                'HexOutput',
                'Color',
                'Table',
                'CrashDump',
                'DebugLog',
                'Logger',

                # from thread import *
                'Thread',

                # from util import *
                'PathOperations',
                'MemoryAddresses',
                'CustomAddressIterator',
                'DataAddressIterator',
                'ImageAddressIterator',
                'MappedAddressIterator',
                'ExecutableAddressIterator',
                'ReadableAddressIterator',
                'WriteableAddressIterator',
                'ExecutableAndWriteableAddressIterator',
                'DebugRegister',

                # from window import *
                'Window',

                # import win32
                'win32',

                # from win32 import Handle, ProcessHandle, ThreadHandle, FileHandle
                'Handle',
                'ProcessHandle',
                'ThreadHandle',
                'FileHandle',
            ]

# Import all public symbols
from winappdbg.breakpoint import *
from winappdbg.crash import *
from winappdbg.debug import *
from winappdbg.disasm import *
from winappdbg.event import *
from winappdbg.interactive import *
from winappdbg.module import *
from winappdbg.process import *
from winappdbg.registry import *
from winappdbg.system import *
from winappdbg.search import *
from winappdbg.textio import *
from winappdbg.thread import *
from winappdbg.util import *
from winappdbg.window import *

import winappdbg.win32
from winappdbg.win32 import Handle, ProcessHandle, ThreadHandle, FileHandle

try:
    from sql import *
    __all__.append('CrashDAO')
except ImportError:
    import warnings
    warnings.warn("No SQL database support present (missing dependencies?)",
                  ImportWarning)

# Library version
version_number = 1.5
version = "Version %s" % version_number
