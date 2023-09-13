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
Event handling module.

@see: U{http://apps.sourceforge.net/trac/winappdbg/wiki/Debugging}

@group Debugging:
    EventHandler, EventSift

@group Debug events:
    EventFactory,
    EventDispatcher,
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

@group Warnings:
    EventCallbackWarning
"""

__revision__ = "$Id$"

__all__ = [
            # Factory of Event objects and all of it's subclasses.
            # Users should not need to instance Event objects directly.
            'EventFactory',

            # Event dispatcher used internally by the Debug class.
            'EventDispatcher',

            # Base classes for user-defined event handlers.
            'EventHandler',
            'EventSift',

            # Warning for uncaught exceptions on event callbacks.
            'EventCallbackWarning',

            # Dummy event object that can be used as a placeholder.
            # It's never returned by the EventFactory.
            'NoEvent',

            # Base class for event objects.
            'Event',

            # Event objects.
            'CreateProcessEvent',
            'CreateThreadEvent',
            'ExitProcessEvent',
            'ExitThreadEvent',
            'LoadDLLEvent',
            'UnloadDLLEvent',
            'OutputDebugStringEvent',
            'RIPEvent',
            'ExceptionEvent'
          ]

from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations

import sys
import ctypes
import warnings
import traceback

#==============================================================================

class EventCallbackWarning (RuntimeWarning):
    """
    This warning is issued when an uncaught exception was raised by a
    user-defined event handler.
    """

#==============================================================================

class Event (object):
    """
    Event object.

    @type eventMethod: str
    @cvar eventMethod:
        Method name to call when using L{EventHandler} subclasses.
        Used internally.

    @type eventName: str
    @cvar eventName:
        User-friendly name of the event.

    @type eventDescription: str
    @cvar eventDescription:
        User-friendly description of the event.

    @type debug: L{Debug}
    @ivar debug:
        Debug object that received the event.

    @type raw: L{DEBUG_EVENT}
    @ivar raw:
        Raw DEBUG_EVENT structure as used by the Win32 API.

    @type continueStatus: int
    @ivar continueStatus:
        Continue status to pass to L{win32.ContinueDebugEvent}.
    """

    eventMethod      = 'unknown_event'
    eventName        = 'Unknown event'
    eventDescription = 'A debug event of an unknown type has occured.'

    def __init__(self, debug, raw):
        """
        @type  debug: L{Debug}
        @param debug: Debug object that received the event.

        @type  raw: L{DEBUG_EVENT}
        @param raw: Raw DEBUG_EVENT structure as used by the Win32 API.
        """
        self.debug          = debug
        self.raw            = raw
        self.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED

##    @property
##    def debug(self):
##        """
##        @rtype  debug: L{Debug}
##        @return debug:
##            Debug object that received the event.
##        """
##        return self.__debug()

    def get_event_name(self):
        """
        @rtype:  str
        @return: User-friendly name of the event.
        """
        return self.eventName

    def get_event_description(self):
        """
        @rtype:  str
        @return: User-friendly description of the event.
        """
        return self.eventDescription

    def get_event_code(self):
        """
        @rtype:  int
        @return: Debug event code as defined in the Win32 API.
        """
        return self.raw.dwDebugEventCode

##    # Compatibility with version 1.0
##    # XXX to be removed in version 1.4
##    def get_code(self):
##        """
##        Alias of L{get_event_code} for backwards compatibility
##        with WinAppDbg version 1.0.
##        Will be phased out in the next version.
##
##        @rtype:  int
##        @return: Debug event code as defined in the Win32 API.
##        """
##        return self.get_event_code()

    def get_pid(self):
        """
        @see: L{get_process}

        @rtype:  int
        @return: Process global ID where the event occured.
        """
        return self.raw.dwProcessId

    def get_tid(self):
        """
        @see: L{get_thread}

        @rtype:  int
        @return: Thread global ID where the event occured.
        """
        return self.raw.dwThreadId

    def get_process(self):
        """
        @see: L{get_pid}

        @rtype:  L{Process}
        @return: Process where the event occured.
        """
        pid     = self.get_pid()
        system  = self.debug.system
        if system.has_process(pid):
            process = system.get_process(pid)
        else:
            # XXX HACK
            # The process object was missing for some reason, so make a new one.
            process = Process(pid)
            system._add_process(process)
##            process.scan_threads()    # not needed
            process.scan_modules()
        return process

    def get_thread(self):
        """
        @see: L{get_tid}

        @rtype:  L{Thread}
        @return: Thread where the event occured.
        """
        tid     = self.get_tid()
        process = self.get_process()
        if process.has_thread(tid):
            thread = process.get_thread(tid)
        else:
            # XXX HACK
            # The thread object was missing for some reason, so make a new one.
            thread = Thread(tid)
            process._add_thread(thread)
        return thread

#==============================================================================

class NoEvent (Event):
    """
    No event.

    Dummy L{Event} object that can be used as a placeholder when no debug
    event has occured yet. It's never returned by the L{EventFactory}.
    """

    eventMethod      = 'no_event'
    eventName        = 'No event'
    eventDescription = 'No debug event has occured.'

    def __init__(self, debug, raw = None):
        Event.__init__(self, debug, raw)

    def __len__(self):
        """
        Always returns C{0}, so when evaluating the object as a boolean it's
        always C{False}. This prevents L{Debug.cont} from trying to continue
        a dummy event.
        """
        return 0

    def get_event_code(self):
        return -1

    def get_pid(self):
        return -1

    def get_tid(self):
        return -1

    def get_process(self):
        return Process(self.get_pid())

    def get_thread(self):
        return Thread(self.get_tid())

#==============================================================================

class ExceptionEvent (Event):
    """
    Exception event.

    @type exceptionName: dict( int S{->} str )
    @cvar exceptionName:
        Mapping of exception constants to their names.

    @type exceptionDescription: dict( int S{->} str )
    @cvar exceptionDescription:
        Mapping of exception constants to user-friendly strings.

    @type breakpoint: L{Breakpoint}
    @ivar breakpoint:
        If the exception was caused by one of our breakpoints, this member
        contains a reference to the breakpoint object. Otherwise it's not
        defined. It should only be used from the condition or action callback
        routines, instead of the event handler.

    @type hook: L{Hook}
    @ivar hook:
        If the exception was caused by a function hook, this member contains a
        reference to the hook object. Otherwise it's not defined. It should
        only be used from the hook callback routines, instead of the event
        handler.
    """

    eventName        = 'Exception event'
    eventDescription = 'An exception was raised by the debugee.'

    __exceptionMethod = {
        win32.EXCEPTION_ACCESS_VIOLATION          : 'access_violation',
        win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED     : 'array_bounds_exceeded',
        win32.EXCEPTION_BREAKPOINT                : 'breakpoint',
        win32.EXCEPTION_DATATYPE_MISALIGNMENT     : 'datatype_misalignment',
        win32.EXCEPTION_FLT_DENORMAL_OPERAND      : 'float_denormal_operand',
        win32.EXCEPTION_FLT_DIVIDE_BY_ZERO        : 'float_divide_by_zero',
        win32.EXCEPTION_FLT_INEXACT_RESULT        : 'float_inexact_result',
        win32.EXCEPTION_FLT_INVALID_OPERATION     : 'float_invalid_operation',
        win32.EXCEPTION_FLT_OVERFLOW              : 'float_overflow',
        win32.EXCEPTION_FLT_STACK_CHECK           : 'float_stack_check',
        win32.EXCEPTION_FLT_UNDERFLOW             : 'float_underflow',
        win32.EXCEPTION_ILLEGAL_INSTRUCTION       : 'illegal_instruction',
        win32.EXCEPTION_IN_PAGE_ERROR             : 'in_page_error',
        win32.EXCEPTION_INT_DIVIDE_BY_ZERO        : 'integer_divide_by_zero',
        win32.EXCEPTION_INT_OVERFLOW              : 'integer_overflow',
        win32.EXCEPTION_INVALID_DISPOSITION       : 'invalid_disposition',
        win32.EXCEPTION_NONCONTINUABLE_EXCEPTION  : 'noncontinuable_exception',
        win32.EXCEPTION_PRIV_INSTRUCTION          : 'privileged_instruction',
        win32.EXCEPTION_SINGLE_STEP               : 'single_step',
        win32.EXCEPTION_STACK_OVERFLOW            : 'stack_overflow',
        win32.EXCEPTION_GUARD_PAGE                : 'guard_page',
        win32.EXCEPTION_INVALID_HANDLE            : 'invalid_handle',
        win32.EXCEPTION_POSSIBLE_DEADLOCK         : 'possible_deadlock',
        win32.EXCEPTION_WX86_BREAKPOINT           : 'wow64_breakpoint',
        win32.CONTROL_C_EXIT                      : 'control_c_exit',
        win32.DBG_CONTROL_C                       : 'debug_control_c',
        win32.MS_VC_EXCEPTION                     : 'ms_vc_exception',
    }

    __exceptionName = {
        win32.EXCEPTION_ACCESS_VIOLATION          : 'EXCEPTION_ACCESS_VIOLATION',
        win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED     : 'EXCEPTION_ARRAY_BOUNDS_EXCEEDED',
        win32.EXCEPTION_BREAKPOINT                : 'EXCEPTION_BREAKPOINT',
        win32.EXCEPTION_DATATYPE_MISALIGNMENT     : 'EXCEPTION_DATATYPE_MISALIGNMENT',
        win32.EXCEPTION_FLT_DENORMAL_OPERAND      : 'EXCEPTION_FLT_DENORMAL_OPERAND',
        win32.EXCEPTION_FLT_DIVIDE_BY_ZERO        : 'EXCEPTION_FLT_DIVIDE_BY_ZERO',
        win32.EXCEPTION_FLT_INEXACT_RESULT        : 'EXCEPTION_FLT_INEXACT_RESULT',
        win32.EXCEPTION_FLT_INVALID_OPERATION     : 'EXCEPTION_FLT_INVALID_OPERATION',
        win32.EXCEPTION_FLT_OVERFLOW              : 'EXCEPTION_FLT_OVERFLOW',
        win32.EXCEPTION_FLT_STACK_CHECK           : 'EXCEPTION_FLT_STACK_CHECK',
        win32.EXCEPTION_FLT_UNDERFLOW             : 'EXCEPTION_FLT_UNDERFLOW',
        win32.EXCEPTION_ILLEGAL_INSTRUCTION       : 'EXCEPTION_ILLEGAL_INSTRUCTION',
        win32.EXCEPTION_IN_PAGE_ERROR             : 'EXCEPTION_IN_PAGE_ERROR',
        win32.EXCEPTION_INT_DIVIDE_BY_ZERO        : 'EXCEPTION_INT_DIVIDE_BY_ZERO',
        win32.EXCEPTION_INT_OVERFLOW              : 'EXCEPTION_INT_OVERFLOW',
        win32.EXCEPTION_INVALID_DISPOSITION       : 'EXCEPTION_INVALID_DISPOSITION',
        win32.EXCEPTION_NONCONTINUABLE_EXCEPTION  : 'EXCEPTION_NONCONTINUABLE_EXCEPTION',
        win32.EXCEPTION_PRIV_INSTRUCTION          : 'EXCEPTION_PRIV_INSTRUCTION',
        win32.EXCEPTION_SINGLE_STEP               : 'EXCEPTION_SINGLE_STEP',
        win32.EXCEPTION_STACK_OVERFLOW            : 'EXCEPTION_STACK_OVERFLOW',
        win32.EXCEPTION_GUARD_PAGE                : 'EXCEPTION_GUARD_PAGE',
        win32.EXCEPTION_INVALID_HANDLE            : 'EXCEPTION_INVALID_HANDLE',
        win32.EXCEPTION_POSSIBLE_DEADLOCK         : 'EXCEPTION_POSSIBLE_DEADLOCK',
        win32.EXCEPTION_WX86_BREAKPOINT           : 'EXCEPTION_WX86_BREAKPOINT',
        win32.CONTROL_C_EXIT                      : 'CONTROL_C_EXIT',
        win32.DBG_CONTROL_C                       : 'DBG_CONTROL_C',
        win32.MS_VC_EXCEPTION                     : 'MS_VC_EXCEPTION',
    }

    __exceptionDescription = {
        win32.EXCEPTION_ACCESS_VIOLATION          : 'Access violation',
        win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED     : 'Array bounds exceeded',
        win32.EXCEPTION_BREAKPOINT                : 'Breakpoint',
        win32.EXCEPTION_DATATYPE_MISALIGNMENT     : 'Datatype misalignment',
        win32.EXCEPTION_FLT_DENORMAL_OPERAND      : 'Float denormal operand',
        win32.EXCEPTION_FLT_DIVIDE_BY_ZERO        : 'Float divide by zero',
        win32.EXCEPTION_FLT_INEXACT_RESULT        : 'Float inexact result',
        win32.EXCEPTION_FLT_INVALID_OPERATION     : 'Float invalid operation',
        win32.EXCEPTION_FLT_OVERFLOW              : 'Float overflow',
        win32.EXCEPTION_FLT_STACK_CHECK           : 'Float stack check',
        win32.EXCEPTION_FLT_UNDERFLOW             : 'Float underflow',
        win32.EXCEPTION_ILLEGAL_INSTRUCTION       : 'Illegal instruction',
        win32.EXCEPTION_IN_PAGE_ERROR             : 'In-page error',
        win32.EXCEPTION_INT_DIVIDE_BY_ZERO        : 'Integer divide by zero',
        win32.EXCEPTION_INT_OVERFLOW              : 'Integer overflow',
        win32.EXCEPTION_INVALID_DISPOSITION       : 'Invalid disposition',
        win32.EXCEPTION_NONCONTINUABLE_EXCEPTION  : 'Noncontinuable exception',
        win32.EXCEPTION_PRIV_INSTRUCTION          : 'Privileged instruction',
        win32.EXCEPTION_SINGLE_STEP               : 'Single step event',
        win32.EXCEPTION_STACK_OVERFLOW            : 'Stack limits overflow',
        win32.EXCEPTION_GUARD_PAGE                : 'Guard page hit',
        win32.EXCEPTION_INVALID_HANDLE            : 'Invalid handle',
        win32.EXCEPTION_POSSIBLE_DEADLOCK         : 'Possible deadlock',
        win32.EXCEPTION_WX86_BREAKPOINT           : 'WOW64 breakpoint',
        win32.CONTROL_C_EXIT                      : 'Control-C exit',
        win32.DBG_CONTROL_C                       : 'Debug Control-C',
        win32.MS_VC_EXCEPTION                     : 'Microsoft Visual C++ exception',
    }

    @property
    def eventMethod(self):
        return self.__exceptionMethod.get(
                                self.get_exception_code(), 'unknown_exception')

    def get_exception_name(self):
        """
        @rtype:  str
        @return: Name of the exception as defined by the Win32 API.
        """
        code = self.get_exception_code()
        unk  = HexDump.integer(code)
        return self.__exceptionName.get(code, unk)

    def get_exception_description(self):
        """
        @rtype:  str
        @return: User-friendly name of the exception.
        """
        code = self.get_exception_code()
        description = self.__exceptionDescription.get(code, None)
        if description is None:
            try:
                description = 'Exception code %s (%s)'
                description = description % (HexDump.integer(code),
                                             ctypes.FormatError(code))
            except OverflowError:
                description = 'Exception code %s' % HexDump.integer(code)
        return description

    def is_first_chance(self):
        """
        @rtype:  bool
        @return: C{True} for first chance exceptions, C{False} for last chance.
        """
        return self.raw.u.Exception.dwFirstChance != 0

    def is_last_chance(self):
        """
        @rtype:  bool
        @return: The opposite of L{is_first_chance}.
        """
        return not self.is_first_chance()

    def is_noncontinuable(self):
        """
        @see: U{http://msdn.microsoft.com/en-us/library/aa363082(VS.85).aspx}

        @rtype:  bool
        @return: C{True} if the exception is noncontinuable,
            C{False} otherwise.

            Attempting to continue a noncontinuable exception results in an
            EXCEPTION_NONCONTINUABLE_EXCEPTION exception to be raised.
        """
        return bool( self.raw.u.Exception.ExceptionRecord.ExceptionFlags & \
                                            win32.EXCEPTION_NONCONTINUABLE )

    def is_continuable(self):
        """
        @rtype:  bool
        @return: The opposite of L{is_noncontinuable}.
        """
        return not self.is_noncontinuable()

    def is_user_defined_exception(self):
        """
        Determines if this is an user-defined exception. User-defined
        exceptions may contain any exception code that is not system reserved.

        Often the exception code is also a valid Win32 error code, but that's
        up to the debugged application.

        @rtype:  bool
        @return: C{True} if the exception is user-defined, C{False} otherwise.
        """
        return self.get_exception_code() & 0x10000000 == 0

    def is_system_defined_exception(self):
        """
        @rtype:  bool
        @return: The opposite of L{is_user_defined_exception}.
        """
        return not self.is_user_defined_exception()

    def get_exception_code(self):
        """
        @rtype:  int
        @return: Exception code as defined by the Win32 API.
        """
        return self.raw.u.Exception.ExceptionRecord.ExceptionCode

    def get_exception_address(self):
        """
        @rtype:  int
        @return: Memory address where the exception occured.
        """
        address = self.raw.u.Exception.ExceptionRecord.ExceptionAddress
        if address is None:
            address = 0
        return address

    def get_exception_information(self, index):
        """
        @type  index: int
        @param index: Index into the exception information block.

        @rtype:  int
        @return: Exception information DWORD.
        """
        if index < 0 or index > win32.EXCEPTION_MAXIMUM_PARAMETERS:
            raise IndexError("Array index out of range: %s" % repr(index))
        info = self.raw.u.Exception.ExceptionRecord.ExceptionInformation
        value = info[index]
        if value is None:
            value = 0
        return value

    def get_exception_information_as_list(self):
        """
        @rtype:  list( int )
        @return: Exception information block.
        """
        info = self.raw.u.Exception.ExceptionRecord.ExceptionInformation
        data = list()
        for index in compat.xrange(0, win32.EXCEPTION_MAXIMUM_PARAMETERS):
            value = info[index]
            if value is None:
                value = 0
            data.append(value)
        return data

    def get_fault_type(self):
        """
        @rtype:  int
        @return: Access violation type.
            Should be one of the following constants:

             - L{win32.EXCEPTION_READ_FAULT}
             - L{win32.EXCEPTION_WRITE_FAULT}
             - L{win32.EXCEPTION_EXECUTE_FAULT}

        @note: This method is only meaningful for access violation exceptions,
            in-page memory error exceptions and guard page exceptions.

        @raise NotImplementedError: Wrong kind of exception.
        """
        if self.get_exception_code() not in (win32.EXCEPTION_ACCESS_VIOLATION,
                    win32.EXCEPTION_IN_PAGE_ERROR, win32.EXCEPTION_GUARD_PAGE):
            msg = "This method is not meaningful for %s."
            raise NotImplementedError(msg % self.get_exception_name())
        return self.get_exception_information(0)

    def get_fault_address(self):
        """
        @rtype:  int
        @return: Access violation memory address.

        @note: This method is only meaningful for access violation exceptions,
            in-page memory error exceptions and guard page exceptions.

        @raise NotImplementedError: Wrong kind of exception.
        """
        if self.get_exception_code() not in (win32.EXCEPTION_ACCESS_VIOLATION,
                    win32.EXCEPTION_IN_PAGE_ERROR, win32.EXCEPTION_GUARD_PAGE):
            msg = "This method is not meaningful for %s."
            raise NotImplementedError(msg % self.get_exception_name())
        return self.get_exception_information(1)

    def get_ntstatus_code(self):
        """
        @rtype:  int
        @return: NTSTATUS status code that caused the exception.

        @note: This method is only meaningful for in-page memory error
            exceptions.

        @raise NotImplementedError: Not an in-page memory error.
        """
        if self.get_exception_code() != win32.EXCEPTION_IN_PAGE_ERROR:
            msg = "This method is only meaningful "\
                  "for in-page memory error exceptions."
            raise NotImplementedError(msg)
        return self.get_exception_information(2)

    def is_nested(self):
        """
        @rtype:  bool
        @return: Returns C{True} if there are additional exception records
            associated with this exception. This would mean the exception
            is nested, that is, it was triggered while trying to handle
            at least one previous exception.
        """
        return bool(self.raw.u.Exception.ExceptionRecord.ExceptionRecord)

    def get_raw_exception_record_list(self):
        """
        Traverses the exception record linked list and builds a Python list.

        Nested exception records are received for nested exceptions. This
        happens when an exception is raised in the debugee while trying to
        handle a previous exception.

        @rtype:  list( L{win32.EXCEPTION_RECORD} )
        @return:
            List of raw exception record structures as used by the Win32 API.

            There is always at least one exception record, so the list is
            never empty. All other methods of this class read from the first
            exception record only, that is, the most recent exception.
        """
        # The first EXCEPTION_RECORD is contained in EXCEPTION_DEBUG_INFO.
        # The remaining EXCEPTION_RECORD structures are linked by pointers.
        nested = list()
        record = self.raw.u.Exception
        while True:
            record = record.ExceptionRecord
            if not record:
                break
            nested.append(record)
        return nested

    def get_nested_exceptions(self):
        """
        Traverses the exception record linked list and builds a Python list.

        Nested exception records are received for nested exceptions. This
        happens when an exception is raised in the debugee while trying to
        handle a previous exception.

        @rtype:  list( L{ExceptionEvent} )
        @return:
            List of ExceptionEvent objects representing each exception record
            found in this event.

            There is always at least one exception record, so the list is
            never empty. All other methods of this class read from the first
            exception record only, that is, the most recent exception.
        """
        # The list always begins with ourselves.
        # Just put a reference to "self" as the first element,
        # and start looping from the second exception record.
        nested = [ self ]
        raw = self.raw
        dwDebugEventCode = raw.dwDebugEventCode
        dwProcessId      = raw.dwProcessId
        dwThreadId       = raw.dwThreadId
        dwFirstChance    = raw.u.Exception.dwFirstChance
        record           = raw.u.Exception.ExceptionRecord
        while True:
            record = record.ExceptionRecord
            if not record:
                break
            raw = win32.DEBUG_EVENT()
            raw.dwDebugEventCode            = dwDebugEventCode
            raw.dwProcessId                 = dwProcessId
            raw.dwThreadId                  = dwThreadId
            raw.u.Exception.ExceptionRecord = record
            raw.u.Exception.dwFirstChance   = dwFirstChance
            event = EventFactory.get(self.debug, raw)
            nested.append(event)
        return nested

#==============================================================================

class CreateThreadEvent (Event):
    """
    Thread creation event.
    """

    eventMethod      = 'create_thread'
    eventName        = 'Thread creation event'
    eventDescription = 'A new thread has started.'

    def get_thread_handle(self):
        """
        @rtype:  L{ThreadHandle}
        @return: Thread handle received from the system.
            Returns C{None} if the handle is not available.
        """
        # The handle doesn't need to be closed.
        # See http://msdn.microsoft.com/en-us/library/ms681423(VS.85).aspx
        hThread = self.raw.u.CreateThread.hThread
        if hThread in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hThread = None
        else:
            hThread = ThreadHandle(hThread, False, win32.THREAD_ALL_ACCESS)
        return hThread

    def get_teb(self):
        """
        @rtype:  int
        @return: Pointer to the TEB.
        """
        return self.raw.u.CreateThread.lpThreadLocalBase

    def get_start_address(self):
        """
        @rtype:  int
        @return: Pointer to the first instruction to execute in this thread.

            Returns C{NULL} when the debugger attached to a process
            and the thread already existed.

            See U{http://msdn.microsoft.com/en-us/library/ms679295(VS.85).aspx}
        """
        return self.raw.u.CreateThread.lpStartAddress

#==============================================================================

class CreateProcessEvent (Event):
    """
    Process creation event.
    """

    eventMethod      = 'create_process'
    eventName        = 'Process creation event'
    eventDescription = 'A new process has started.'

    def get_file_handle(self):
        """
        @rtype:  L{FileHandle} or None
        @return: File handle to the main module, received from the system.
            Returns C{None} if the handle is not available.
        """
        # This handle DOES need to be closed.
        # Therefore we must cache it so it doesn't
        # get closed after the first call.
        try:
            hFile = self.__hFile
        except AttributeError:
            hFile = self.raw.u.CreateProcessInfo.hFile
            if hFile in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
                hFile = None
            else:
                hFile = FileHandle(hFile, True)
            self.__hFile = hFile
        return hFile

    def get_process_handle(self):
        """
        @rtype:  L{ProcessHandle}
        @return: Process handle received from the system.
            Returns C{None} if the handle is not available.
        """
        # The handle doesn't need to be closed.
        # See http://msdn.microsoft.com/en-us/library/ms681423(VS.85).aspx
        hProcess = self.raw.u.CreateProcessInfo.hProcess
        if hProcess in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hProcess = None
        else:
            hProcess = ProcessHandle(hProcess, False, win32.PROCESS_ALL_ACCESS)
        return hProcess

    def get_thread_handle(self):
        """
        @rtype:  L{ThreadHandle}
        @return: Thread handle received from the system.
            Returns C{None} if the handle is not available.
        """
        # The handle doesn't need to be closed.
        # See http://msdn.microsoft.com/en-us/library/ms681423(VS.85).aspx
        hThread = self.raw.u.CreateProcessInfo.hThread
        if hThread in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hThread = None
        else:
            hThread = ThreadHandle(hThread, False, win32.THREAD_ALL_ACCESS)
        return hThread

    def get_start_address(self):
        """
        @rtype:  int
        @return: Pointer to the first instruction to execute in this process.

            Returns C{NULL} when the debugger attaches to a process.

            See U{http://msdn.microsoft.com/en-us/library/ms679295(VS.85).aspx}
        """
        return self.raw.u.CreateProcessInfo.lpStartAddress

    def get_image_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        @warn: This value is taken from the PE file
            and may be incorrect because of ASLR!
        """
        # TODO try to calculate the real value when ASLR is active.
        return self.raw.u.CreateProcessInfo.lpBaseOfImage

    def get_teb(self):
        """
        @rtype:  int
        @return: Pointer to the TEB.
        """
        return self.raw.u.CreateProcessInfo.lpThreadLocalBase

    def get_debug_info(self):
        """
        @rtype:  str
        @return: Debugging information.
        """
        raw  = self.raw.u.CreateProcessInfo
        ptr  = raw.lpBaseOfImage + raw.dwDebugInfoFileOffset
        size = raw.nDebugInfoSize
        data = self.get_process().peek(ptr, size)
        if len(data) == size:
            return data
        return None

    def get_filename(self):
        """
        @rtype:  str, None
        @return: This method does it's best to retrieve the filename to
        the main module of the process. However, sometimes that's not
        possible, and C{None} is returned instead.
        """

        # Try to get the filename from the file handle.
        szFilename = None
        hFile = self.get_file_handle()
        if hFile:
            szFilename = hFile.get_filename()
        if not szFilename:

            # Try to get it from CREATE_PROCESS_DEBUG_INFO.lpImageName
            # It's NULL or *NULL most of the times, see MSDN:
            # http://msdn.microsoft.com/en-us/library/ms679286(VS.85).aspx
            aProcess = self.get_process()
            lpRemoteFilenamePtr = self.raw.u.CreateProcessInfo.lpImageName
            if lpRemoteFilenamePtr:
                lpFilename  = aProcess.peek_uint(lpRemoteFilenamePtr)
                fUnicode    = bool( self.raw.u.CreateProcessInfo.fUnicode )
                szFilename  = aProcess.peek_string(lpFilename, fUnicode)

                # XXX TODO
                # Sometimes the filename is relative (ntdll.dll, kernel32.dll).
                # It could be converted to an absolute pathname (SearchPath).

            # Try to get it from Process.get_image_name().
            if not szFilename:
                szFilename = aProcess.get_image_name()

        # Return the filename, or None on error.
        return szFilename

    def get_module_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        """
        return self.get_image_base()

    def get_module(self):
        """
        @rtype:  L{Module}
        @return: Main module of the process.
        """
        return self.get_process().get_module( self.get_module_base() )

#==============================================================================

class ExitThreadEvent (Event):
    """
    Thread termination event.
    """

    eventMethod      = 'exit_thread'
    eventName        = 'Thread termination event'
    eventDescription = 'A thread has finished executing.'

    def get_exit_code(self):
        """
        @rtype:  int
        @return: Exit code of the thread.
        """
        return self.raw.u.ExitThread.dwExitCode

#==============================================================================

class ExitProcessEvent (Event):
    """
    Process termination event.
    """

    eventMethod      = 'exit_process'
    eventName        = 'Process termination event'
    eventDescription = 'A process has finished executing.'

    def get_exit_code(self):
        """
        @rtype:  int
        @return: Exit code of the process.
        """
        return self.raw.u.ExitProcess.dwExitCode

    def get_filename(self):
        """
        @rtype:  None or str
        @return: Filename of the main module.
            C{None} if the filename is unknown.
        """
        return self.get_module().get_filename()

    def get_image_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        """
        return self.get_module_base()

    def get_module_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        """
        return self.get_module().get_base()

    def get_module(self):
        """
        @rtype:  L{Module}
        @return: Main module of the process.
        """
        return self.get_process().get_main_module()

#==============================================================================

class LoadDLLEvent (Event):
    """
    Module load event.
    """

    eventMethod      = 'load_dll'
    eventName        = 'Module load event'
    eventDescription = 'A new DLL library was loaded by the debugee.'

    def get_module_base(self):
        """
        @rtype:  int
        @return: Base address for the newly loaded DLL.
        """
        return self.raw.u.LoadDll.lpBaseOfDll

    def get_module(self):
        """
        @rtype:  L{Module}
        @return: Module object for the newly loaded DLL.
        """
        lpBaseOfDll = self.get_module_base()
        aProcess    = self.get_process()
        if aProcess.has_module(lpBaseOfDll):
            aModule = aProcess.get_module(lpBaseOfDll)
        else:
            # XXX HACK
            # For some reason the module object is missing, so make a new one.
            aModule = Module(lpBaseOfDll,
                             hFile    = self.get_file_handle(),
                             fileName = self.get_filename(),
                             process  = aProcess)
            aProcess._add_module(aModule)
        return aModule

    def get_file_handle(self):
        """
        @rtype:  L{FileHandle} or None
        @return: File handle to the newly loaded DLL received from the system.
            Returns C{None} if the handle is not available.
        """
        # This handle DOES need to be closed.
        # Therefore we must cache it so it doesn't
        # get closed after the first call.
        try:
            hFile = self.__hFile
        except AttributeError:
            hFile = self.raw.u.LoadDll.hFile
            if hFile in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
                hFile = None
            else:
                hFile = FileHandle(hFile, True)
            self.__hFile = hFile
        return hFile

    def get_filename(self):
        """
        @rtype:  str, None
        @return: This method does it's best to retrieve the filename to
        the newly loaded module. However, sometimes that's not
        possible, and C{None} is returned instead.
        """
        szFilename = None

        # Try to get it from LOAD_DLL_DEBUG_INFO.lpImageName
        # It's NULL or *NULL most of the times, see MSDN:
        # http://msdn.microsoft.com/en-us/library/ms679286(VS.85).aspx
        aProcess = self.get_process()
        lpRemoteFilenamePtr = self.raw.u.LoadDll.lpImageName
        if lpRemoteFilenamePtr:
            lpFilename  = aProcess.peek_uint(lpRemoteFilenamePtr)
            fUnicode    = bool( self.raw.u.LoadDll.fUnicode )
            szFilename  = aProcess.peek_string(lpFilename, fUnicode)
            if not szFilename:
                szFilename = None

        # Try to get the filename from the file handle.
        if not szFilename:
            hFile = self.get_file_handle()
            if hFile:
                szFilename = hFile.get_filename()

        # Return the filename, or None on error.
        return szFilename

#==============================================================================

class UnloadDLLEvent (Event):
    """
    Module unload event.
    """

    eventMethod      = 'unload_dll'
    eventName        = 'Module unload event'
    eventDescription = 'A DLL library was unloaded by the debugee.'

    def get_module_base(self):
        """
        @rtype:  int
        @return: Base address for the recently unloaded DLL.
        """
        return self.raw.u.UnloadDll.lpBaseOfDll

    def get_module(self):
        """
        @rtype:  L{Module}
        @return: Module object for the recently unloaded DLL.
        """
        lpBaseOfDll = self.get_module_base()
        aProcess    = self.get_process()
        if aProcess.has_module(lpBaseOfDll):
            aModule = aProcess.get_module(lpBaseOfDll)
        else:
            aModule = Module(lpBaseOfDll, process = aProcess)
            aProcess._add_module(aModule)
        return aModule

    def get_file_handle(self):
        """
        @rtype:  None or L{FileHandle}
        @return: File handle to the recently unloaded DLL.
            Returns C{None} if the handle is not available.
        """
        hFile = self.get_module().hFile
        if hFile in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hFile = None
        return hFile

    def get_filename(self):
        """
        @rtype:  None or str
        @return: Filename of the recently unloaded DLL.
            C{None} if the filename is unknown.
        """
        return self.get_module().get_filename()

#==============================================================================

class OutputDebugStringEvent (Event):
    """
    Debug string output event.
    """

    eventMethod      = 'output_string'
    eventName        = 'Debug string output event'
    eventDescription = 'The debugee sent a message to the debugger.'

    def get_debug_string(self):
        """
        @rtype:  str, compat.unicode
        @return: String sent by the debugee.
            It may be ANSI or Unicode and may end with a null character.
        """
        return self.get_process().peek_string(
                                self.raw.u.DebugString.lpDebugStringData,
                                bool( self.raw.u.DebugString.fUnicode ),
                                self.raw.u.DebugString.nDebugStringLength)

#==============================================================================

class RIPEvent (Event):
    """
    RIP event.
    """

    eventMethod      = 'rip'
    eventName        = 'RIP event'
    eventDescription = 'An error has occured and the process ' \
                       'can no longer be debugged.'

    def get_rip_error(self):
        """
        @rtype:  int
        @return: RIP error code as defined by the Win32 API.
        """
        return self.raw.u.RipInfo.dwError

    def get_rip_type(self):
        """
        @rtype:  int
        @return: RIP type code as defined by the Win32 API.
            May be C{0} or one of the following:
             - L{win32.SLE_ERROR}
             - L{win32.SLE_MINORERROR}
             - L{win32.SLE_WARNING}
        """
        return self.raw.u.RipInfo.dwType

#==============================================================================

class EventFactory (StaticClass):
    """
    Factory of L{Event} objects.

    @type baseEvent: L{Event}
    @cvar baseEvent:
        Base class for Event objects.
        It's used for unknown event codes.

    @type eventClasses: dict( int S{->} L{Event} )
    @cvar eventClasses:
        Dictionary that maps event codes to L{Event} subclasses.
    """

    baseEvent    = Event
    eventClasses = {
        win32.EXCEPTION_DEBUG_EVENT       : ExceptionEvent,           # 1
        win32.CREATE_THREAD_DEBUG_EVENT   : CreateThreadEvent,        # 2
        win32.CREATE_PROCESS_DEBUG_EVENT  : CreateProcessEvent,       # 3
        win32.EXIT_THREAD_DEBUG_EVENT     : ExitThreadEvent,          # 4
        win32.EXIT_PROCESS_DEBUG_EVENT    : ExitProcessEvent,         # 5
        win32.LOAD_DLL_DEBUG_EVENT        : LoadDLLEvent,             # 6
        win32.UNLOAD_DLL_DEBUG_EVENT      : UnloadDLLEvent,           # 7
        win32.OUTPUT_DEBUG_STRING_EVENT   : OutputDebugStringEvent,   # 8
        win32.RIP_EVENT                   : RIPEvent,                 # 9
    }

    @classmethod
    def get(cls, debug, raw):
        """
        @type  debug: L{Debug}
        @param debug: Debug object that received the event.

        @type  raw: L{DEBUG_EVENT}
        @param raw: Raw DEBUG_EVENT structure as used by the Win32 API.

        @rtype: L{Event}
        @returns: An Event object or one of it's subclasses,
            depending on the event type.
        """
        eventClass = cls.eventClasses.get(raw.dwDebugEventCode, cls.baseEvent)
        return eventClass(debug, raw)

#==============================================================================

class EventHandler (object):
    """
    Base class for debug event handlers.

    Your program should subclass it to implement it's own event handling.

    The constructor can be overriden as long as you call the superclass
    constructor. The special method L{__call__} B{MUST NOT} be overriden.

    The signature for event handlers is the following::

        def event_handler(self, event):

    Where B{event} is an L{Event} object.

    Each event handler is named after the event they handle.
    This is the list of all valid event handler names:

     - I{event}

       Receives an L{Event} object or an object of any of it's subclasses,
       and handles any event for which no handler was defined.

     - I{unknown_event}

       Receives an L{Event} object or an object of any of it's subclasses,
       and handles any event unknown to the debugging engine. (This is not
       likely to happen unless the Win32 debugging API is changed in future
       versions of Windows).

     - I{exception}

       Receives an L{ExceptionEvent} object and handles any exception for
       which no handler was defined. See above for exception handlers.

     - I{unknown_exception}

       Receives an L{ExceptionEvent} object and handles any exception unknown
       to the debugging engine. This usually happens for C++ exceptions, which
       are not standardized and may change from one compiler to the next.

       Currently we have partial support for C++ exceptions thrown by Microsoft
       compilers.

       Also see: U{RaiseException()
       <http://msdn.microsoft.com/en-us/library/ms680552(VS.85).aspx>}

     - I{create_thread}

       Receives a L{CreateThreadEvent} object.

     - I{create_process}

       Receives a L{CreateProcessEvent} object.

     - I{exit_thread}

       Receives a L{ExitThreadEvent} object.

     - I{exit_process}

       Receives a L{ExitProcessEvent} object.

     - I{load_dll}

       Receives a L{LoadDLLEvent} object.

     - I{unload_dll}

       Receives an L{UnloadDLLEvent} object.

     - I{output_string}

       Receives an L{OutputDebugStringEvent} object.

     - I{rip}

       Receives a L{RIPEvent} object.

    This is the list of all valid exception handler names
    (they all receive an L{ExceptionEvent} object):

     - I{access_violation}
     - I{array_bounds_exceeded}
     - I{breakpoint}
     - I{control_c_exit}
     - I{datatype_misalignment}
     - I{debug_control_c}
     - I{float_denormal_operand}
     - I{float_divide_by_zero}
     - I{float_inexact_result}
     - I{float_invalid_operation}
     - I{float_overflow}
     - I{float_stack_check}
     - I{float_underflow}
     - I{guard_page}
     - I{illegal_instruction}
     - I{in_page_error}
     - I{integer_divide_by_zero}
     - I{integer_overflow}
     - I{invalid_disposition}
     - I{invalid_handle}
     - I{ms_vc_exception}
     - I{noncontinuable_exception}
     - I{possible_deadlock}
     - I{privileged_instruction}
     - I{single_step}
     - I{stack_overflow}
     - I{wow64_breakpoint}



    @type apiHooks: dict( str S{->} list( tuple( str, int ) ) )
    @cvar apiHooks:
        Dictionary that maps module names to lists of
        tuples of ( procedure name, parameter count ).

        All procedures listed here will be hooked for calls from the debugee.
        When this happens, the corresponding event handler can be notified both
        when the procedure is entered and when it's left by the debugee.

        For example, let's hook the LoadLibraryEx() API call.
        This would be the declaration of apiHooks::

            from winappdbg import EventHandler
            from winappdbg.win32 import *

            # (...)

            class MyEventHandler (EventHandler):

                apiHook = {

                    "kernel32.dll" : (

                        #   Procedure name      Signature
                        (   "LoadLibraryEx",    (PVOID, HANDLE, DWORD) ),

                        # (more procedures can go here...)
                    ),

                    # (more libraries can go here...)
                }

                # (your method definitions go here...)

        Note that all pointer types are treated like void pointers, so your
        callback won't get the string or structure pointed to by it, but the
        remote memory address instead. This is so to prevent the ctypes library
        from being "too helpful" and trying to dereference the pointer. To get
        the actual data being pointed to, use one of the L{Process.read}
        methods.

        Now, to intercept calls to LoadLibraryEx define a method like this in
        your event handler class::

            def pre_LoadLibraryEx(self, event, ra, lpFilename, hFile, dwFlags):
                szFilename = event.get_process().peek_string(lpFilename)

                # (...)

        Note that the first parameter is always the L{Event} object, and the
        second parameter is the return address. The third parameter and above
        are the values passed to the hooked function.

        Finally, to intercept returns from calls to LoadLibraryEx define a
        method like this::

            def post_LoadLibraryEx(self, event, retval):
                # (...)

        The first parameter is the L{Event} object and the second is the
        return value from the hooked function.
    """

#------------------------------------------------------------------------------

    # Default (empty) API hooks dictionary.
    apiHooks = {}

    def __init__(self):
        """
        Class constructor. Don't forget to call it when subclassing!

        Forgetting to call the superclass constructor is a common mistake when
        you're new to Python. :)

        Example::
            class MyEventHandler (EventHandler):

                # Override the constructor to use an extra argument.
                def __init__(self, myArgument):

                    # Do something with the argument, like keeping it
                    # as an instance variable.
                    self.myVariable = myArgument

                    # Call the superclass constructor.
                    super(MyEventHandler, self).__init__()

                # The rest of your code below...
        """

        # TODO
        # All this does is set up the hooks.
        # This code should be moved to the EventDispatcher class.
        # Then the hooks can be set up at set_event_handler() instead, making
        # this class even simpler. The downside here is deciding where to store
        # the ApiHook objects.

        # Convert the tuples into instances of the ApiHook class.
        # A new dictionary must be instanced, otherwise we could also be
        #  affecting all other instances of the EventHandler.
        apiHooks = dict()
        for lib, hooks in compat.iteritems(self.apiHooks):
            hook_objs = []
            for proc, args in hooks:
                if type(args) in (int, long):
                    h = ApiHook(self, lib, proc, paramCount = args)
                else:
                    h = ApiHook(self, lib, proc,  signature = args)
                hook_objs.append(h)
            apiHooks[lib] = hook_objs
        self.__apiHooks = apiHooks

    def __get_hooks_for_dll(self, event):
        """
        Get the requested API hooks for the current DLL.

        Used by L{__hook_dll} and L{__unhook_dll}.
        """
        result = []
        if self.__apiHooks:
            path = event.get_module().get_filename()
            if path:
                lib_name = PathOperations.pathname_to_filename(path).lower()
                for hook_lib, hook_api_list in compat.iteritems(self.__apiHooks):
                    if hook_lib == lib_name:
                        result.extend(hook_api_list)
        return result

    def __hook_dll(self, event):
        """
        Hook the requested API calls (in self.apiHooks).

        This method is called automatically whenever a DLL is loaded.
        """
        debug = event.debug
        pid   = event.get_pid()
        for hook_api_stub in self.__get_hooks_for_dll(event):
            hook_api_stub.hook(debug, pid)

    def __unhook_dll(self, event):
        """
        Unhook the requested API calls (in self.apiHooks).

        This method is called automatically whenever a DLL is unloaded.
        """
        debug = event.debug
        pid   = event.get_pid()
        for hook_api_stub in self.__get_hooks_for_dll(event):
            hook_api_stub.unhook(debug, pid)

    def __call__(self, event):
        """
        Dispatch debug events.

        @warn: B{Don't override this method!}

        @type  event: L{Event}
        @param event: Event object.
        """
        try:
            code = event.get_event_code()
            if code == win32.LOAD_DLL_DEBUG_EVENT:
                self.__hook_dll(event)
            elif code == win32.UNLOAD_DLL_DEBUG_EVENT:
                self.__unhook_dll(event)
        finally:
            method = EventDispatcher.get_handler_method(self, event)
            if method is not None:
                return method(event)

#==============================================================================

# TODO
#  * Make it more generic by adding a few more callbacks.
#    That way it will be possible to make a thread sifter too.
#  * This interface feels too much like an antipattern.
#    When apiHooks is deprecated this will have to be reviewed.

class EventSift(EventHandler):
    """
    Event handler that allows you to use customized event handlers for each
    process you're attached to.

    This makes coding the event handlers much easier, because each instance
    will only "know" about one process. So you can code your event handler as
    if only one process was being debugged, but your debugger can attach to
    multiple processes.

    Example::
        from winappdbg import Debug, EventHandler, EventSift

        # This class was written assuming only one process is attached.
        # If you used it directly it would break when attaching to another
        # process, or when a child process is spawned.
        class MyEventHandler (EventHandler):

            def create_process(self, event):
                self.first = True
                self.name = event.get_process().get_filename()
                print "Attached to %s" % self.name

            def breakpoint(self, event):
                if self.first:
                    self.first = False
                    print "First breakpoint reached at %s" % self.name

            def exit_process(self, event):
                print "Detached from %s" % self.name

        # Now when debugging we use the EventSift to be able to work with
        # multiple processes while keeping our code simple. :)
        if __name__ == "__main__":
            handler = EventSift(MyEventHandler)
            #handler = MyEventHandler()  # try uncommenting this line...
            with Debug(handler) as debug:
                debug.execl("calc.exe")
                debug.execl("notepad.exe")
                debug.execl("charmap.exe")
                debug.loop()

    Subclasses of C{EventSift} can prevent specific event types from
    being forwarded by simply defining a method for it. That means your
    subclass can handle some event types globally while letting other types
    be handled on per-process basis. To forward events manually you can
    call C{self.event(event)}.

    Example::
        class MySift (EventSift):

            # Don't forward this event.
            def debug_control_c(self, event):
                pass

            # Handle this event globally without forwarding it.
            def output_string(self, event):
                print "Debug string: %s" % event.get_debug_string()

            # Handle this event globally and then forward it.
            def create_process(self, event):
                print "New process created, PID: %d" % event.get_pid()
                return self.event(event)

            # All other events will be forwarded.

    Note that overriding the C{event} method would cause no events to be
    forwarded at all. To prevent this, call the superclass implementation.

    Example::

        def we_want_to_forward_this_event(event):
            "Use whatever logic you want here..."
            # (...return True or False...)

        class MySift (EventSift):

            def event(self, event):

                # If the event matches some custom criteria...
                if we_want_to_forward_this_event(event):

                    # Forward it.
                    return super(MySift, self).event(event)

                # Otherwise, don't.

    @type cls: class
    @ivar cls:
        Event handler class. There will be one instance of this class
        per debugged process in the L{forward} dictionary.

    @type argv: list
    @ivar argv:
        Positional arguments to pass to the constructor of L{cls}.

    @type argd: list
    @ivar argd:
        Keyword arguments to pass to the constructor of L{cls}.

    @type forward: dict
    @ivar forward:
        Dictionary that maps each debugged process ID to an instance of L{cls}.
    """

    def __init__(self, cls, *argv, **argd):
        """
        Maintains an instance of your event handler for each process being
        debugged, and forwards the events of each process to each corresponding
        instance.

        @warn: If you subclass L{EventSift} and reimplement this method,
            don't forget to call the superclass constructor!

        @see: L{event}

        @type  cls: class
        @param cls: Event handler class. This must be the class itself, not an
            instance! All additional arguments passed to the constructor of
            the event forwarder will be passed on to the constructor of this
            class as well.
        """
        self.cls     = cls
        self.argv    = argv
        self.argd    = argd
        self.forward = dict()
        super(EventSift, self).__init__()

    # XXX HORRIBLE HACK
    # This makes apiHooks work in the inner handlers.
    def __call__(self, event):
        try:
            eventCode = event.get_event_code()
            if eventCode in (win32.LOAD_DLL_DEBUG_EVENT,
                             win32.LOAD_DLL_DEBUG_EVENT):
                pid = event.get_pid()
                handler = self.forward.get(pid, None)
                if handler is None:
                    handler = self.cls(*self.argv, **self.argd)
                self.forward[pid] = handler
                if isinstance(handler, EventHandler):
                    if eventCode == win32.LOAD_DLL_DEBUG_EVENT:
                        handler.__EventHandler_hook_dll(event)
                    else:
                        handler.__EventHandler_unhook_dll(event)
        finally:
            return super(EventSift, self).__call__(event)

    def event(self, event):
        """
        Forwards events to the corresponding instance of your event handler
        for this process.

        If you subclass L{EventSift} and reimplement this method, no event
        will be forwarded at all unless you call the superclass implementation.

        If your filtering is based on the event type, there's a much easier way
        to do it: just implement a handler for it.
        """
        eventCode = event.get_event_code()
        pid = event.get_pid()
        handler = self.forward.get(pid, None)
        if handler is None:
            handler = self.cls(*self.argv, **self.argd)
            if eventCode != win32.EXIT_PROCESS_DEBUG_EVENT:
                self.forward[pid] = handler
        elif eventCode == win32.EXIT_PROCESS_DEBUG_EVENT:
            del self.forward[pid]
        return handler(event)

#==============================================================================

class EventDispatcher (object):
    """
    Implements debug event dispatching capabilities.

    @group Debugging events:
        get_event_handler, set_event_handler, get_handler_method
    """

    # Maps event code constants to the names of the pre-notify routines.
    # These routines are called BEFORE the user-defined handlers.
    # Unknown codes are ignored.
    __preEventNotifyCallbackName = {
        win32.CREATE_THREAD_DEBUG_EVENT   : '_notify_create_thread',
        win32.CREATE_PROCESS_DEBUG_EVENT  : '_notify_create_process',
        win32.LOAD_DLL_DEBUG_EVENT        : '_notify_load_dll',
    }

    # Maps event code constants to the names of the post-notify routines.
    # These routines are called AFTER the user-defined handlers.
    # Unknown codes are ignored.
    __postEventNotifyCallbackName = {
        win32.EXIT_THREAD_DEBUG_EVENT     : '_notify_exit_thread',
        win32.EXIT_PROCESS_DEBUG_EVENT    : '_notify_exit_process',
        win32.UNLOAD_DLL_DEBUG_EVENT      : '_notify_unload_dll',
        win32.RIP_EVENT                   : '_notify_rip',
    }

    # Maps exception code constants to the names of the pre-notify routines.
    # These routines are called BEFORE the user-defined handlers.
    # Unknown codes are ignored.
    __preExceptionNotifyCallbackName = {
        win32.EXCEPTION_BREAKPOINT        : '_notify_breakpoint',
        win32.EXCEPTION_WX86_BREAKPOINT   : '_notify_breakpoint',
        win32.EXCEPTION_SINGLE_STEP       : '_notify_single_step',
        win32.EXCEPTION_GUARD_PAGE        : '_notify_guard_page',
        win32.DBG_CONTROL_C               : '_notify_debug_control_c',
        win32.MS_VC_EXCEPTION             : '_notify_ms_vc_exception',
    }

    # Maps exception code constants to the names of the post-notify routines.
    # These routines are called AFTER the user-defined handlers.
    # Unknown codes are ignored.
    __postExceptionNotifyCallbackName = {
    }

    def __init__(self, eventHandler = None):
        """
        Event dispatcher.

        @type  eventHandler: L{EventHandler}
        @param eventHandler: (Optional) User-defined event handler.

        @raise TypeError: The event handler is of an incorrect type.

        @note: The L{eventHandler} parameter may be any callable Python object
            (for example a function, or an instance method).
            However you'll probably find it more convenient to use an instance
            of a subclass of L{EventHandler} here.
        """
        self.set_event_handler(eventHandler)

    def get_event_handler(self):
        """
        Get the event handler.

        @see: L{set_event_handler}

        @rtype:  L{EventHandler}
        @return: Current event handler object, or C{None}.
        """
        return self.__eventHandler

    def set_event_handler(self, eventHandler):
        """
        Set the event handler.

        @warn: This is normally not needed. Use with care!

        @type  eventHandler: L{EventHandler}
        @param eventHandler: New event handler object, or C{None}.

        @rtype:  L{EventHandler}
        @return: Previous event handler object, or C{None}.

        @raise TypeError: The event handler is of an incorrect type.

        @note: The L{eventHandler} parameter may be any callable Python object
            (for example a function, or an instance method).
            However you'll probably find it more convenient to use an instance
            of a subclass of L{EventHandler} here.
        """
        if eventHandler is not None and not callable(eventHandler):
            raise TypeError("Event handler must be a callable object")
        try:
            wrong_type = issubclass(eventHandler, EventHandler)
        except TypeError:
            wrong_type = False
        if wrong_type:
            classname = str(eventHandler)
            msg  = "Event handler must be an instance of class %s"
            msg += "rather than the %s class itself. (Missing parens?)"
            msg  = msg % (classname, classname)
            raise TypeError(msg)
        try:
            previous = self.__eventHandler
        except AttributeError:
            previous = None
        self.__eventHandler = eventHandler
        return previous

    @staticmethod
    def get_handler_method(eventHandler, event, fallback=None):
        """
        Retrieves the appropriate callback method from an L{EventHandler}
        instance for the given L{Event} object.

        @type  eventHandler: L{EventHandler}
        @param eventHandler:
            Event handler object whose methods we are examining.

        @type  event: L{Event}
        @param event: Debugging event to be handled.

        @type  fallback: callable
        @param fallback: (Optional) If no suitable method is found in the
            L{EventHandler} instance, return this value.

        @rtype:  callable
        @return: Bound method that will handle the debugging event.
            Returns C{None} if no such method is defined.
        """
        eventCode = event.get_event_code()
        method = getattr(eventHandler, 'event', fallback)
        if eventCode == win32.EXCEPTION_DEBUG_EVENT:
            method = getattr(eventHandler, 'exception', method)
        method = getattr(eventHandler, event.eventMethod, method)
        return method

    def dispatch(self, event):
        """
        Sends event notifications to the L{Debug} object and
        the L{EventHandler} object provided by the user.

        The L{Debug} object will forward the notifications to it's contained
        snapshot objects (L{System}, L{Process}, L{Thread} and L{Module}) when
        appropriate.

        @warning: This method is called automatically from L{Debug.dispatch}.

        @see: L{Debug.cont}, L{Debug.loop}, L{Debug.wait}

        @type  event: L{Event}
        @param event: Event object passed to L{Debug.dispatch}.

        @raise WindowsError: Raises an exception on error.
        """
        returnValue  = None
        bCallHandler = True
        pre_handler  = None
        post_handler = None
        eventCode    = event.get_event_code()

        # Get the pre and post notification methods for exceptions.
        # If not found, the following steps take care of that.
        if eventCode == win32.EXCEPTION_DEBUG_EVENT:
            exceptionCode = event.get_exception_code()
            pre_name      = self.__preExceptionNotifyCallbackName.get(
                                                           exceptionCode, None)
            post_name     = self.__postExceptionNotifyCallbackName.get(
                                                           exceptionCode, None)
            if  pre_name     is not None:
                pre_handler  = getattr(self, pre_name,  None)
            if  post_name    is not None:
                post_handler = getattr(self, post_name, None)

        # Get the pre notification method for all other events.
        # This includes the exception event if no notify method was found
        # for this exception code.
        if pre_handler is None:
            pre_name = self.__preEventNotifyCallbackName.get(eventCode, None)
            if  pre_name is not None:
                pre_handler = getattr(self, pre_name, pre_handler)

        # Get the post notification method for all other events.
        # This includes the exception event if no notify method was found
        # for this exception code.
        if post_handler is None:
            post_name = self.__postEventNotifyCallbackName.get(eventCode, None)
            if  post_name is not None:
                post_handler = getattr(self, post_name, post_handler)

        # Call the pre-notify method only if it was defined.
        # If an exception is raised don't call the other methods.
        if pre_handler is not None:
            bCallHandler = pre_handler(event)

        # Call the user-defined event handler only if the pre-notify
        #  method was not defined, or was and it returned True.
        try:
            if bCallHandler and self.__eventHandler is not None:
                try:
                    returnValue = self.__eventHandler(event)
                except Exception:
                    e = sys.exc_info()[1]
                    msg = ("Event handler pre-callback %r"
                           " raised an exception: %s")
                    msg = msg % (self.__eventHandler, traceback.format_exc(e))
                    warnings.warn(msg, EventCallbackWarning)
                    returnValue = None

        # Call the post-notify method if defined, even if an exception is
        #  raised by the user-defined event handler.
        finally:
            if post_handler is not None:
                post_handler(event)

        # Return the value from the call to the user-defined event handler.
        # If not defined return None.
        return returnValue
