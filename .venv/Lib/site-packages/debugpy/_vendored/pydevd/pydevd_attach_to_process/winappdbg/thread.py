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
Thread instrumentation.

@group Instrumentation:
    Thread
"""

from __future__ import with_statement

__revision__ = "$Id$"

__all__ = ['Thread']

from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window

import sys
import struct
import warnings

# delayed imports
Process = None

#==============================================================================

# TODO
# + fetch special registers (MMX, XMM, 3DNow!, etc)

class Thread (object):
    """
    Interface to a thread in another process.

    @group Properties:
        get_tid, get_pid, get_process, set_process, get_exit_code, is_alive,
        get_name, set_name, get_windows, get_teb, get_teb_address, is_wow64,
        get_arch, get_bits, get_handle, open_handle, close_handle

    @group Instrumentation:
        suspend, resume, kill, wait

    @group Debugging:
        get_seh_chain_pointer, set_seh_chain_pointer,
        get_seh_chain, get_wait_chain, is_hidden

    @group Disassembly:
        disassemble, disassemble_around, disassemble_around_pc,
        disassemble_string, disassemble_instruction, disassemble_current

    @group Stack:
        get_stack_frame, get_stack_frame_range, get_stack_range,
        get_stack_trace, get_stack_trace_with_labels,
        read_stack_data, read_stack_dwords, read_stack_qwords,
        peek_stack_data, peek_stack_dwords, peek_stack_qwords,
        read_stack_structure, read_stack_frame

    @group Registers:
        get_context,
        get_register,
        get_flags, get_flag_value,
        get_pc, get_sp, get_fp,
        get_cf, get_df, get_sf, get_tf, get_zf,
        set_context,
        set_register,
        set_flags, set_flag_value,
        set_pc, set_sp, set_fp,
        set_cf, set_df, set_sf, set_tf, set_zf,
        clear_cf, clear_df, clear_sf, clear_tf, clear_zf,
        Flags

    @group Threads snapshot:
        clear

    @group Miscellaneous:
        read_code_bytes, peek_code_bytes,
        peek_pointers_in_data, peek_pointers_in_registers,
        get_linear_address, get_label_at_pc

    @type dwThreadId: int
    @ivar dwThreadId: Global thread ID. Use L{get_tid} instead.

    @type hThread: L{ThreadHandle}
    @ivar hThread: Handle to the thread. Use L{get_handle} instead.

    @type process: L{Process}
    @ivar process: Parent process object. Use L{get_process} instead.

    @type pInjectedMemory: int
    @ivar pInjectedMemory: If the thread was created by L{Process.inject_code},
        this member contains a pointer to the memory buffer for the injected
        code. Otherwise it's C{None}.

        The L{kill} method uses this member to free the buffer
        when the injected thread is killed.
    """

    def __init__(self, dwThreadId, hThread = None, process = None):
        """
        @type  dwThreadId: int
        @param dwThreadId: Global thread ID.

        @type  hThread: L{ThreadHandle}
        @param hThread: (Optional) Handle to the thread.

        @type  process: L{Process}
        @param process: (Optional) Parent Process object.
        """
        self.dwProcessId     = None
        self.dwThreadId      = dwThreadId
        self.hThread         = hThread
        self.pInjectedMemory = None
        self.set_name(None)
        self.set_process(process)

    # Not really sure if it's a good idea...
##    def __eq__(self, aThread):
##        """
##        Compare two Thread objects. The comparison is made using the IDs.
##
##        @warning:
##            If you have two Thread instances with different handles the
##            equality operator still returns C{True}, so be careful!
##
##        @type  aThread: L{Thread}
##        @param aThread: Another Thread object.
##
##        @rtype:  bool
##        @return: C{True} if the two thread IDs are equal,
##            C{False} otherwise.
##        """
##        return isinstance(aThread, Thread)           and \
##               self.get_tid() == aThread.get_tid()

    def __load_Process_class(self):
        global Process      # delayed import
        if Process is None:
            from winappdbg.process import Process

    def get_process(self):
        """
        @rtype:  L{Process}
        @return: Parent Process object.
            Returns C{None} if unknown.
        """
        if self.__process is not None:
            return self.__process
        self.__load_Process_class()
        self.__process = Process(self.get_pid())
        return self.__process

    def set_process(self, process = None):
        """
        Manually set the parent Process object. Use with care!

        @type  process: L{Process}
        @param process: (Optional) Process object. Use C{None} for no process.
        """
        if process is None:
            self.dwProcessId = None
            self.__process   = None
        else:
            self.__load_Process_class()
            if not isinstance(process, Process):
                msg  = "Parent process must be a Process instance, "
                msg += "got %s instead" % type(process)
                raise TypeError(msg)
            self.dwProcessId = process.get_pid()
            self.__process = process

    process = property(get_process, set_process, doc="")

    def get_pid(self):
        """
        @rtype:  int
        @return: Parent process global ID.

        @raise WindowsError: An error occured when calling a Win32 API function.
        @raise RuntimeError: The parent process ID can't be found.
        """
        if self.dwProcessId is None:
            if self.__process is not None:
                # Infinite loop if self.__process is None
                self.dwProcessId = self.get_process().get_pid()
            else:
                try:
                    # I wish this had been implemented before Vista...
                    # XXX TODO find the real ntdll call under this api
                    hThread = self.get_handle(
                                        win32.THREAD_QUERY_LIMITED_INFORMATION)
                    self.dwProcessId = win32.GetProcessIdOfThread(hThread)
                except AttributeError:
                    # This method is really bad :P
                    self.dwProcessId = self.__get_pid_by_scanning()
        return self.dwProcessId

    def __get_pid_by_scanning(self):
        'Internally used by get_pid().'
        dwProcessId = None
        dwThreadId = self.get_tid()
        with win32.CreateToolhelp32Snapshot(win32.TH32CS_SNAPTHREAD) as hSnapshot:
            te = win32.Thread32First(hSnapshot)
            while te is not None:
                if te.th32ThreadID == dwThreadId:
                    dwProcessId = te.th32OwnerProcessID
                    break
                te = win32.Thread32Next(hSnapshot)
        if dwProcessId is None:
            msg = "Cannot find thread ID %d in any process" % dwThreadId
            raise RuntimeError(msg)
        return dwProcessId

    def get_tid(self):
        """
        @rtype:  int
        @return: Thread global ID.
        """
        return self.dwThreadId

    def get_name(self):
        """
        @rtype:  str
        @return: Thread name, or C{None} if the thread is nameless.
        """
        return self.name

    def set_name(self, name = None):
        """
        Sets the thread's name.

        @type  name: str
        @param name: Thread name, or C{None} if the thread is nameless.
        """
        self.name = name

#------------------------------------------------------------------------------

    def open_handle(self, dwDesiredAccess = win32.THREAD_ALL_ACCESS):
        """
        Opens a new handle to the thread, closing the previous one.

        The new handle is stored in the L{hThread} property.

        @warn: Normally you should call L{get_handle} instead, since it's much
            "smarter" and tries to reuse handles and merge access rights.

        @type  dwDesiredAccess: int
        @param dwDesiredAccess: Desired access rights.
            Defaults to L{win32.THREAD_ALL_ACCESS}.
            See: U{http://msdn.microsoft.com/en-us/library/windows/desktop/ms686769(v=vs.85).aspx}

        @raise WindowsError: It's not possible to open a handle to the thread
            with the requested access rights. This tipically happens because
            the target thread belongs to system process and the debugger is not
            runnning with administrative rights.
        """
        hThread = win32.OpenThread(dwDesiredAccess, win32.FALSE, self.dwThreadId)

        # In case hThread was set to an actual handle value instead of a Handle
        # object. This shouldn't happen unless the user tinkered with it.
        if not hasattr(self.hThread, '__del__'):
            self.close_handle()

        self.hThread = hThread

    def close_handle(self):
        """
        Closes the handle to the thread.

        @note: Normally you don't need to call this method. All handles
            created by I{WinAppDbg} are automatically closed when the garbage
            collector claims them.
        """
        try:
            if hasattr(self.hThread, 'close'):
                self.hThread.close()
            elif self.hThread not in (None, win32.INVALID_HANDLE_VALUE):
                win32.CloseHandle(self.hThread)
        finally:
            self.hThread = None

    def get_handle(self, dwDesiredAccess = win32.THREAD_ALL_ACCESS):
        """
        Returns a handle to the thread with I{at least} the access rights
        requested.

        @note:
            If a handle was previously opened and has the required access
            rights, it's reused. If not, a new handle is opened with the
            combination of the old and new access rights.

        @type  dwDesiredAccess: int
        @param dwDesiredAccess: Desired access rights.
            See: U{http://msdn.microsoft.com/en-us/library/windows/desktop/ms686769(v=vs.85).aspx}

        @rtype:  ThreadHandle
        @return: Handle to the thread.

        @raise WindowsError: It's not possible to open a handle to the thread
            with the requested access rights. This tipically happens because
            the target thread belongs to system process and the debugger is not
            runnning with administrative rights.
        """
        if self.hThread in (None, win32.INVALID_HANDLE_VALUE):
            self.open_handle(dwDesiredAccess)
        else:
            dwAccess = self.hThread.dwAccess
            if (dwAccess | dwDesiredAccess) != dwAccess:
                self.open_handle(dwAccess | dwDesiredAccess)
        return self.hThread

    def clear(self):
        """
        Clears the resources held by this object.
        """
        try:
            self.set_process(None)
        finally:
            self.close_handle()

#------------------------------------------------------------------------------

    def wait(self, dwTimeout = None):
        """
        Waits for the thread to finish executing.

        @type  dwTimeout: int
        @param dwTimeout: (Optional) Timeout value in milliseconds.
            Use C{INFINITE} or C{None} for no timeout.
        """
        self.get_handle(win32.SYNCHRONIZE).wait(dwTimeout)

    def kill(self, dwExitCode = 0):
        """
        Terminates the thread execution.

        @note: If the C{lpInjectedMemory} member contains a valid pointer,
        the memory is freed.

        @type  dwExitCode: int
        @param dwExitCode: (Optional) Thread exit code.
        """
        hThread = self.get_handle(win32.THREAD_TERMINATE)
        win32.TerminateThread(hThread, dwExitCode)

        # Ugliest hack ever, won't work if many pieces of code are injected.
        # Seriously, what was I thinking? :(
        if self.pInjectedMemory is not None:
            try:
                self.get_process().free(self.pInjectedMemory)
                self.pInjectedMemory = None
            except Exception:
##                raise           # XXX DEBUG
                pass

    # XXX TODO
    # suspend() and resume() should have a counter of how many times a thread
    # was suspended, so on debugger exit they could (optionally!) be restored

    def suspend(self):
        """
        Suspends the thread execution.

        @rtype:  int
        @return: Suspend count. If zero, the thread is running.
        """
        hThread = self.get_handle(win32.THREAD_SUSPEND_RESUME)
        if self.is_wow64():
            # FIXME this will be horribly slow on XP 64
            # since it'll try to resolve a missing API every time
            try:
                return win32.Wow64SuspendThread(hThread)
            except AttributeError:
                pass
        return win32.SuspendThread(hThread)

    def resume(self):
        """
        Resumes the thread execution.

        @rtype:  int
        @return: Suspend count. If zero, the thread is running.
        """
        hThread = self.get_handle(win32.THREAD_SUSPEND_RESUME)
        return win32.ResumeThread(hThread)

    def is_alive(self):
        """
        @rtype:  bool
        @return: C{True} if the thread if currently running.
        @raise WindowsError:
            The debugger doesn't have enough privileges to perform this action.
        """
        try:
            self.wait(0)
        except WindowsError:
            e = sys.exc_info()[1]
            error = e.winerror
            if error == win32.ERROR_ACCESS_DENIED:
                raise
            return error == win32.WAIT_TIMEOUT
        return True

    def get_exit_code(self):
        """
        @rtype:  int
        @return: Thread exit code, or C{STILL_ACTIVE} if it's still alive.
        """
        if win32.THREAD_ALL_ACCESS == win32.THREAD_ALL_ACCESS_VISTA:
            dwAccess = win32.THREAD_QUERY_LIMITED_INFORMATION
        else:
            dwAccess = win32.THREAD_QUERY_INFORMATION
        return win32.GetExitCodeThread( self.get_handle(dwAccess) )

#------------------------------------------------------------------------------

    # XXX TODO
    # Support for string searches on the window captions.

    def get_windows(self):
        """
        @rtype:  list of L{Window}
        @return: Returns a list of windows handled by this thread.
        """
        try:
            process = self.get_process()
        except Exception:
            process = None
        return [
                Window( hWnd, process, self ) \
                for hWnd in win32.EnumThreadWindows( self.get_tid() )
                ]

#------------------------------------------------------------------------------

    # TODO
    # A registers cache could be implemented here.
    def get_context(self, ContextFlags = None, bSuspend = False):
        """
        Retrieves the execution context (i.e. the registers values) for this
        thread.

        @type  ContextFlags: int
        @param ContextFlags: Optional, specify which registers to retrieve.
            Defaults to C{win32.CONTEXT_ALL} which retrieves all registes
            for the current platform.

        @type  bSuspend: bool
        @param bSuspend: C{True} to automatically suspend the thread before
            getting its context, C{False} otherwise.

            Defaults to C{False} because suspending the thread during some
            debug events (like thread creation or destruction) may lead to
            strange errors.

            Note that WinAppDbg 1.4 used to suspend the thread automatically
            always. This behavior was changed in version 1.5.

        @rtype:  dict( str S{->} int )
        @return: Dictionary mapping register names to their values.

        @see: L{set_context}
        """

        # Some words on the "strange errors" that lead to the bSuspend
        # parameter. Peter Van Eeckhoutte and I were working on a fix
        # for some bugs he found in the 1.5 betas when we stumbled upon
        # what seemed to be a deadlock in the debug API that caused the
        # GetThreadContext() call never to return. Since removing the
        # call to SuspendThread() solved the problem, and a few Google
        # searches showed a handful of problems related to these two
        # APIs and Wow64 environments, I decided to break compatibility.
        #
        # Here are some pages about the weird behavior of SuspendThread:
        # http://zachsaw.blogspot.com.es/2010/11/wow64-bug-getthreadcontext-may-return.html
        # http://stackoverflow.com/questions/3444190/windows-suspendthread-doesnt-getthreadcontext-fails

        # Get the thread handle.
        dwAccess = win32.THREAD_GET_CONTEXT
        if bSuspend:
            dwAccess = dwAccess | win32.THREAD_SUSPEND_RESUME
        hThread = self.get_handle(dwAccess)

        # Suspend the thread if requested.
        if bSuspend:
            try:
                self.suspend()
            except WindowsError:
                # Threads can't be suspended when the exit process event
                # arrives, but you can still get the context.
                bSuspend = False

        # If an exception is raised, make sure the thread execution is resumed.
        try:

            if win32.bits == self.get_bits():

                # 64 bit debugger attached to 64 bit process, or
                # 32 bit debugger attached to 32 bit process.
                ctx = win32.GetThreadContext(hThread,
                                             ContextFlags = ContextFlags)

            else:
                if self.is_wow64():

                    # 64 bit debugger attached to 32 bit process.
                    if ContextFlags is not None:
                        ContextFlags &= ~win32.ContextArchMask
                        ContextFlags |=  win32.WOW64_CONTEXT_i386
                    ctx = win32.Wow64GetThreadContext(hThread, ContextFlags)

                else:

                    # 32 bit debugger attached to 64 bit process.
                    # XXX only i386/AMD64 is supported in this particular case
                    if win32.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
                        raise NotImplementedError()
                    if ContextFlags is not None:
                        ContextFlags &= ~win32.ContextArchMask
                        ContextFlags |=  win32.context_amd64.CONTEXT_AMD64
                    ctx = win32.context_amd64.GetThreadContext(hThread,
                                                 ContextFlags = ContextFlags)

        finally:

            # Resume the thread if we suspended it.
            if bSuspend:
                self.resume()

        # Return the context.
        return ctx

    def set_context(self, context, bSuspend = False):
        """
        Sets the values of the registers.

        @see: L{get_context}

        @type  context:  dict( str S{->} int )
        @param context: Dictionary mapping register names to their values.

        @type  bSuspend: bool
        @param bSuspend: C{True} to automatically suspend the thread before
            setting its context, C{False} otherwise.

            Defaults to C{False} because suspending the thread during some
            debug events (like thread creation or destruction) may lead to
            strange errors.

            Note that WinAppDbg 1.4 used to suspend the thread automatically
            always. This behavior was changed in version 1.5.
        """

        # Get the thread handle.
        dwAccess = win32.THREAD_SET_CONTEXT
        if bSuspend:
            dwAccess = dwAccess | win32.THREAD_SUSPEND_RESUME
        hThread = self.get_handle(dwAccess)

        # Suspend the thread if requested.
        if bSuspend:
            self.suspend()
            # No fix for the exit process event bug.
            # Setting the context of a dead thread is pointless anyway.

        # Set the thread context.
        try:
            if win32.bits == 64 and self.is_wow64():
                win32.Wow64SetThreadContext(hThread, context)
            else:
                win32.SetThreadContext(hThread, context)

        # Resume the thread if we suspended it.
        finally:
            if bSuspend:
                self.resume()

    def get_register(self, register):
        """
        @type  register: str
        @param register: Register name.

        @rtype:  int
        @return: Value of the requested register.
        """
        'Returns the value of a specific register.'
        context = self.get_context()
        return context[register]

    def set_register(self, register, value):
        """
        Sets the value of a specific register.

        @type  register: str
        @param register: Register name.

        @rtype:  int
        @return: Register value.
        """
        context = self.get_context()
        context[register] = value
        self.set_context(context)

#------------------------------------------------------------------------------

    # TODO: a metaclass would do a better job instead of checking the platform
    #       during module import, also would support mixing 32 and 64 bits

    if win32.arch in (win32.ARCH_I386, win32.ARCH_AMD64):

        def get_pc(self):
            """
            @rtype:  int
            @return: Value of the program counter register.
            """
            context = self.get_context(win32.CONTEXT_CONTROL)
            return context.pc

        def set_pc(self, pc):
            """
            Sets the value of the program counter register.

            @type  pc: int
            @param pc: Value of the program counter register.
            """
            context = self.get_context(win32.CONTEXT_CONTROL)
            context.pc = pc
            self.set_context(context)

        def get_sp(self):
            """
            @rtype:  int
            @return: Value of the stack pointer register.
            """
            context = self.get_context(win32.CONTEXT_CONTROL)
            return context.sp

        def set_sp(self, sp):
            """
            Sets the value of the stack pointer register.

            @type  sp: int
            @param sp: Value of the stack pointer register.
            """
            context = self.get_context(win32.CONTEXT_CONTROL)
            context.sp = sp
            self.set_context(context)

        def get_fp(self):
            """
            @rtype:  int
            @return: Value of the frame pointer register.
            """
            flags = win32.CONTEXT_CONTROL | win32.CONTEXT_INTEGER
            context = self.get_context(flags)
            return context.fp

        def set_fp(self, fp):
            """
            Sets the value of the frame pointer register.

            @type  fp: int
            @param fp: Value of the frame pointer register.
            """
            flags = win32.CONTEXT_CONTROL | win32.CONTEXT_INTEGER
            context = self.get_context(flags)
            context.fp = fp
            self.set_context(context)

#------------------------------------------------------------------------------

    if win32.arch in (win32.ARCH_I386, win32.ARCH_AMD64):

        class Flags (object):
            'Commonly used processor flags'
            Overflow    = 0x800
            Direction   = 0x400
            Interrupts  = 0x200
            Trap        = 0x100
            Sign        = 0x80
            Zero        = 0x40
            # 0x20 ???
            Auxiliary   = 0x10
            # 0x8 ???
            Parity      = 0x4
            # 0x2 ???
            Carry       = 0x1

        def get_flags(self, FlagMask = 0xFFFFFFFF):
            """
            @type  FlagMask: int
            @param FlagMask: (Optional) Bitwise-AND mask.

            @rtype:  int
            @return: Flags register contents, optionally masking out some bits.
            """
            context = self.get_context(win32.CONTEXT_CONTROL)
            return context['EFlags'] & FlagMask

        def set_flags(self, eflags, FlagMask = 0xFFFFFFFF):
            """
            Sets the flags register, optionally masking some bits.

            @type  eflags: int
            @param eflags: Flags register contents.

            @type  FlagMask: int
            @param FlagMask: (Optional) Bitwise-AND mask.
            """
            context = self.get_context(win32.CONTEXT_CONTROL)
            context['EFlags'] = (context['EFlags'] & FlagMask) | eflags
            self.set_context(context)

        def get_flag_value(self, FlagBit):
            """
            @type  FlagBit: int
            @param FlagBit: One of the L{Flags}.

            @rtype:  bool
            @return: Boolean value of the requested flag.
            """
            return bool( self.get_flags(FlagBit) )

        def set_flag_value(self, FlagBit, FlagValue):
            """
            Sets a single flag, leaving the others intact.

            @type  FlagBit: int
            @param FlagBit: One of the L{Flags}.

            @type  FlagValue: bool
            @param FlagValue: Boolean value of the flag.
            """
            if FlagValue:
                eflags = FlagBit
            else:
                eflags = 0
            FlagMask = 0xFFFFFFFF ^ FlagBit
            self.set_flags(eflags, FlagMask)

        def get_zf(self):
            """
            @rtype:  bool
            @return: Boolean value of the Zero flag.
            """
            return self.get_flag_value(self.Flags.Zero)

        def get_cf(self):
            """
            @rtype:  bool
            @return: Boolean value of the Carry flag.
            """
            return self.get_flag_value(self.Flags.Carry)

        def get_sf(self):
            """
            @rtype:  bool
            @return: Boolean value of the Sign flag.
            """
            return self.get_flag_value(self.Flags.Sign)

        def get_df(self):
            """
            @rtype:  bool
            @return: Boolean value of the Direction flag.
            """
            return self.get_flag_value(self.Flags.Direction)

        def get_tf(self):
            """
            @rtype:  bool
            @return: Boolean value of the Trap flag.
            """
            return self.get_flag_value(self.Flags.Trap)

        def clear_zf(self):
            'Clears the Zero flag.'
            self.set_flag_value(self.Flags.Zero, False)

        def clear_cf(self):
            'Clears the Carry flag.'
            self.set_flag_value(self.Flags.Carry, False)

        def clear_sf(self):
            'Clears the Sign flag.'
            self.set_flag_value(self.Flags.Sign, False)

        def clear_df(self):
            'Clears the Direction flag.'
            self.set_flag_value(self.Flags.Direction, False)

        def clear_tf(self):
            'Clears the Trap flag.'
            self.set_flag_value(self.Flags.Trap, False)

        def set_zf(self):
            'Sets the Zero flag.'
            self.set_flag_value(self.Flags.Zero, True)

        def set_cf(self):
            'Sets the Carry flag.'
            self.set_flag_value(self.Flags.Carry, True)

        def set_sf(self):
            'Sets the Sign flag.'
            self.set_flag_value(self.Flags.Sign, True)

        def set_df(self):
            'Sets the Direction flag.'
            self.set_flag_value(self.Flags.Direction, True)

        def set_tf(self):
            'Sets the Trap flag.'
            self.set_flag_value(self.Flags.Trap, True)

#------------------------------------------------------------------------------

    def is_wow64(self):
        """
        Determines if the thread is running under WOW64.

        @rtype:  bool
        @return:
            C{True} if the thread is running under WOW64. That is, it belongs
            to a 32-bit application running in a 64-bit Windows.

            C{False} if the thread belongs to either a 32-bit application
            running in a 32-bit Windows, or a 64-bit application running in a
            64-bit Windows.

        @raise WindowsError: On error an exception is raised.

        @see: U{http://msdn.microsoft.com/en-us/library/aa384249(VS.85).aspx}
        """
        try:
            wow64 = self.__wow64
        except AttributeError:
            if (win32.bits == 32 and not win32.wow64):
                wow64 = False
            else:
                wow64 = self.get_process().is_wow64()
            self.__wow64 = wow64
        return wow64

    def get_arch(self):
        """
        @rtype:  str
        @return: The architecture in which this thread believes to be running.
            For example, if running a 32 bit binary in a 64 bit machine, the
            architecture returned by this method will be L{win32.ARCH_I386},
            but the value of L{System.arch} will be L{win32.ARCH_AMD64}.
        """
        if win32.bits == 32 and not win32.wow64:
            return win32.arch
        return self.get_process().get_arch()

    def get_bits(self):
        """
        @rtype:  str
        @return: The number of bits in which this thread believes to be
            running. For example, if running a 32 bit binary in a 64 bit
            machine, the number of bits returned by this method will be C{32},
            but the value of L{System.arch} will be C{64}.
        """
        if win32.bits == 32 and not win32.wow64:
            return 32
        return self.get_process().get_bits()

    def is_hidden(self):
        """
        Determines if the thread has been hidden from debuggers.

        Some binary packers hide their own threads to thwart debugging.

        @rtype:  bool
        @return: C{True} if the thread is hidden from debuggers.
            This means the thread's execution won't be stopped for debug
            events, and thus said events won't be sent to the debugger.
        """
        return win32.NtQueryInformationThread(
                    self.get_handle(),      # XXX what permissions do I need?
                    win32.ThreadHideFromDebugger)

    def get_teb(self):
        """
        Returns a copy of the TEB.
        To dereference pointers in it call L{Process.read_structure}.

        @rtype:  L{TEB}
        @return: TEB structure.
        @raise WindowsError: An exception is raised on error.
        """
        return self.get_process().read_structure( self.get_teb_address(),
                                                  win32.TEB )

    def get_teb_address(self):
        """
        Returns a remote pointer to the TEB.

        @rtype:  int
        @return: Remote pointer to the L{TEB} structure.
        @raise WindowsError: An exception is raised on error.
        """
        try:
            return self._teb_ptr
        except AttributeError:
            try:
                hThread = self.get_handle(win32.THREAD_QUERY_INFORMATION)
                tbi = win32.NtQueryInformationThread( hThread,
                                                win32.ThreadBasicInformation)
                address = tbi.TebBaseAddress
            except WindowsError:
                address = self.get_linear_address('SegFs', 0)   # fs:[0]
                if not address:
                    raise
            self._teb_ptr = address
            return address

    def get_linear_address(self, segment, address):
        """
        Translates segment-relative addresses to linear addresses.

        Linear addresses can be used to access a process memory,
        calling L{Process.read} and L{Process.write}.

        @type  segment: str
        @param segment: Segment register name.

        @type  address: int
        @param address: Segment relative memory address.

        @rtype:  int
        @return: Linear memory address.

        @raise ValueError: Address is too large for selector.

        @raise WindowsError:
            The current architecture does not support selectors.
            Selectors only exist in x86-based systems.
        """
        hThread  = self.get_handle(win32.THREAD_QUERY_INFORMATION)
        selector = self.get_register(segment)
        ldt      = win32.GetThreadSelectorEntry(hThread, selector)
        BaseLow  = ldt.BaseLow
        BaseMid  = ldt.HighWord.Bytes.BaseMid << 16
        BaseHi   = ldt.HighWord.Bytes.BaseHi  << 24
        Base     = BaseLow | BaseMid | BaseHi
        LimitLow = ldt.LimitLow
        LimitHi  = ldt.HighWord.Bits.LimitHi  << 16
        Limit    = LimitLow | LimitHi
        if address > Limit:
            msg = "Address %s too large for segment %s (selector %d)"
            msg = msg % (HexDump.address(address, self.get_bits()),
                         segment, selector)
            raise ValueError(msg)
        return Base + address

    def get_label_at_pc(self):
        """
        @rtype:  str
        @return: Label that points to the instruction currently being executed.
        """
        return self.get_process().get_label_at_address( self.get_pc() )

    def get_seh_chain_pointer(self):
        """
        Get the pointer to the first structured exception handler block.

        @rtype:  int
        @return: Remote pointer to the first block of the structured exception
            handlers linked list. If the list is empty, the returned value is
            C{0xFFFFFFFF}.

        @raise NotImplementedError:
            This method is only supported in 32 bits versions of Windows.
        """
        if win32.arch != win32.ARCH_I386:
            raise NotImplementedError(
                "SEH chain parsing is only supported in 32-bit Windows.")

        process = self.get_process()
        address = self.get_linear_address( 'SegFs', 0 )
        return process.read_pointer( address )

    def set_seh_chain_pointer(self, value):
        """
        Change the pointer to the first structured exception handler block.

        @type  value: int
        @param value: Value of the remote pointer to the first block of the
            structured exception handlers linked list. To disable SEH set the
            value C{0xFFFFFFFF}.

        @raise NotImplementedError:
            This method is only supported in 32 bits versions of Windows.
        """
        if win32.arch != win32.ARCH_I386:
            raise NotImplementedError(
                "SEH chain parsing is only supported in 32-bit Windows.")

        process = self.get_process()
        address = self.get_linear_address( 'SegFs', 0 )
        process.write_pointer( address, value )

    def get_seh_chain(self):
        """
        @rtype:  list of tuple( int, int )
        @return: List of structured exception handlers.
            Each SEH is represented as a tuple of two addresses:
                - Address of this SEH block
                - Address of the SEH callback function
            Do not confuse this with the contents of the SEH block itself,
            where the first member is a pointer to the B{next} block instead.

        @raise NotImplementedError:
            This method is only supported in 32 bits versions of Windows.
        """
        seh_chain = list()
        try:
            process = self.get_process()
            seh = self.get_seh_chain_pointer()
            while seh != 0xFFFFFFFF:
                seh_func = process.read_pointer( seh + 4 )
                seh_chain.append( (seh, seh_func) )
                seh = process.read_pointer( seh )
        except WindowsError:
            seh_chain.append( (seh, None) )
        return seh_chain

    def get_wait_chain(self):
        """
        @rtype:
            tuple of (
            list of L{win32.WaitChainNodeInfo} structures,
            bool)
        @return:
            Wait chain for the thread.
            The boolean indicates if there's a cycle in the chain (a deadlock).
        @raise AttributeError:
            This method is only suppported in Windows Vista and above.
        @see:
            U{http://msdn.microsoft.com/en-us/library/ms681622%28VS.85%29.aspx}
        """
        with win32.OpenThreadWaitChainSession() as hWct:
            return win32.GetThreadWaitChain(hWct, ThreadId = self.get_tid())

    def get_stack_range(self):
        """
        @rtype:  tuple( int, int )
        @return: Stack beginning and end pointers, in memory addresses order.
            That is, the first pointer is the stack top, and the second pointer
            is the stack bottom, since the stack grows towards lower memory
            addresses.
        @raise   WindowsError: Raises an exception on error.
        """
        # TODO use teb.DeallocationStack too (max. possible stack size)
        teb = self.get_teb()
        tib = teb.NtTib
        return ( tib.StackLimit, tib.StackBase )    # top, bottom

    def __get_stack_trace(self, depth = 16, bUseLabels = True,
                                                           bMakePretty = True):
        """
        Tries to get a stack trace for the current function using the debug
        helper API (dbghelp.dll).

        @type  depth: int
        @param depth: Maximum depth of stack trace.

        @type  bUseLabels: bool
        @param bUseLabels: C{True} to use labels, C{False} to use addresses.

        @type  bMakePretty: bool
        @param bMakePretty:
            C{True} for user readable labels,
            C{False} for labels that can be passed to L{Process.resolve_label}.

            "Pretty" labels look better when producing output for the user to
            read, while pure labels are more useful programatically.

        @rtype:  tuple of tuple( int, int, str )
        @return: Stack trace of the thread as a tuple of
            ( return address, frame pointer address, module filename )
            when C{bUseLabels} is C{True}, or a tuple of
            ( return address, frame pointer label )
            when C{bUseLabels} is C{False}.

        @raise WindowsError: Raises an exception on error.
        """

        aProcess = self.get_process()
        arch = aProcess.get_arch()
        bits = aProcess.get_bits()

        if arch == win32.ARCH_I386:
            MachineType = win32.IMAGE_FILE_MACHINE_I386
        elif arch == win32.ARCH_AMD64:
            MachineType = win32.IMAGE_FILE_MACHINE_AMD64
        elif arch == win32.ARCH_IA64:
            MachineType = win32.IMAGE_FILE_MACHINE_IA64
        else:
            msg = "Stack walking is not available for this architecture: %s"
            raise NotImplementedError(msg % arch)

        hProcess = aProcess.get_handle( win32.PROCESS_VM_READ |
                                        win32.PROCESS_QUERY_INFORMATION )
        hThread  = self.get_handle( win32.THREAD_GET_CONTEXT |
                                    win32.THREAD_QUERY_INFORMATION )

        StackFrame = win32.STACKFRAME64()
        StackFrame.AddrPC    = win32.ADDRESS64( self.get_pc() )
        StackFrame.AddrFrame = win32.ADDRESS64( self.get_fp() )
        StackFrame.AddrStack = win32.ADDRESS64( self.get_sp() )

        trace = list()
        while win32.StackWalk64(MachineType, hProcess, hThread, StackFrame):
            if depth <= 0:
                break
            fp = StackFrame.AddrFrame.Offset
            ra = aProcess.peek_pointer(fp + 4)
            if ra == 0:
                break
            lib = aProcess.get_module_at_address(ra)
            if lib is None:
                lib = ""
            else:
                if lib.fileName:
                    lib = lib.fileName
                else:
                    lib = "%s" % HexDump.address(lib.lpBaseOfDll, bits)
            if bUseLabels:
                label = aProcess.get_label_at_address(ra)
                if bMakePretty:
                    label = '%s (%s)' % (HexDump.address(ra, bits), label)
                trace.append( (fp, label) )
            else:
                trace.append( (fp, ra, lib) )
            fp = aProcess.peek_pointer(fp)
        return tuple(trace)

    def __get_stack_trace_manually(self, depth = 16, bUseLabels = True,
                                                           bMakePretty = True):
        """
        Tries to get a stack trace for the current function.
        Only works for functions with standard prologue and epilogue.

        @type  depth: int
        @param depth: Maximum depth of stack trace.

        @type  bUseLabels: bool
        @param bUseLabels: C{True} to use labels, C{False} to use addresses.

        @type  bMakePretty: bool
        @param bMakePretty:
            C{True} for user readable labels,
            C{False} for labels that can be passed to L{Process.resolve_label}.

            "Pretty" labels look better when producing output for the user to
            read, while pure labels are more useful programatically.

        @rtype:  tuple of tuple( int, int, str )
        @return: Stack trace of the thread as a tuple of
            ( return address, frame pointer address, module filename )
            when C{bUseLabels} is C{True}, or a tuple of
            ( return address, frame pointer label )
            when C{bUseLabels} is C{False}.

        @raise WindowsError: Raises an exception on error.
        """
        aProcess = self.get_process()
        st, sb   = self.get_stack_range()   # top, bottom
        fp       = self.get_fp()
        trace    = list()
        if aProcess.get_module_count() == 0:
            aProcess.scan_modules()
        bits = aProcess.get_bits()
        while depth > 0:
            if fp == 0:
                break
            if not st <= fp < sb:
                break
            ra  = aProcess.peek_pointer(fp + 4)
            if ra == 0:
                break
            lib = aProcess.get_module_at_address(ra)
            if lib is None:
                lib = ""
            else:
                if lib.fileName:
                    lib = lib.fileName
                else:
                    lib = "%s" % HexDump.address(lib.lpBaseOfDll, bits)
            if bUseLabels:
                label = aProcess.get_label_at_address(ra)
                if bMakePretty:
                    label = '%s (%s)' % (HexDump.address(ra, bits), label)
                trace.append( (fp, label) )
            else:
                trace.append( (fp, ra, lib) )
            fp = aProcess.peek_pointer(fp)
        return tuple(trace)

    def get_stack_trace(self, depth = 16):
        """
        Tries to get a stack trace for the current function.
        Only works for functions with standard prologue and epilogue.

        @type  depth: int
        @param depth: Maximum depth of stack trace.

        @rtype:  tuple of tuple( int, int, str )
        @return: Stack trace of the thread as a tuple of
            ( return address, frame pointer address, module filename ).

        @raise WindowsError: Raises an exception on error.
        """
        try:
            trace = self.__get_stack_trace(depth, False)
        except Exception:
            import traceback
            traceback.print_exc()
            trace = ()
        if not trace:
            trace = self.__get_stack_trace_manually(depth, False)
        return trace

    def get_stack_trace_with_labels(self, depth = 16, bMakePretty = True):
        """
        Tries to get a stack trace for the current function.
        Only works for functions with standard prologue and epilogue.

        @type  depth: int
        @param depth: Maximum depth of stack trace.

        @type  bMakePretty: bool
        @param bMakePretty:
            C{True} for user readable labels,
            C{False} for labels that can be passed to L{Process.resolve_label}.

            "Pretty" labels look better when producing output for the user to
            read, while pure labels are more useful programatically.

        @rtype:  tuple of tuple( int, int, str )
        @return: Stack trace of the thread as a tuple of
            ( return address, frame pointer label ).

        @raise WindowsError: Raises an exception on error.
        """
        try:
            trace = self.__get_stack_trace(depth, True, bMakePretty)
        except Exception:
            trace = ()
        if not trace:
            trace = self.__get_stack_trace_manually(depth, True, bMakePretty)
        return trace

    def get_stack_frame_range(self):
        """
        Returns the starting and ending addresses of the stack frame.
        Only works for functions with standard prologue and epilogue.

        @rtype:  tuple( int, int )
        @return: Stack frame range.
            May not be accurate, depending on the compiler used.

        @raise RuntimeError: The stack frame is invalid,
            or the function doesn't have a standard prologue
            and epilogue.

        @raise WindowsError: An error occured when getting the thread context.
        """
        st, sb   = self.get_stack_range()   # top, bottom
        sp       = self.get_sp()
        fp       = self.get_fp()
        size     = fp - sp
        if not st <= sp < sb:
            raise RuntimeError('Stack pointer lies outside the stack')
        if not st <= fp < sb:
            raise RuntimeError('Frame pointer lies outside the stack')
        if sp > fp:
            raise RuntimeError('No valid stack frame found')
        return (sp, fp)

    def get_stack_frame(self, max_size = None):
        """
        Reads the contents of the current stack frame.
        Only works for functions with standard prologue and epilogue.

        @type  max_size: int
        @param max_size: (Optional) Maximum amount of bytes to read.

        @rtype:  str
        @return: Stack frame data.
            May not be accurate, depending on the compiler used.
            May return an empty string.

        @raise RuntimeError: The stack frame is invalid,
            or the function doesn't have a standard prologue
            and epilogue.

        @raise WindowsError: An error occured when getting the thread context
            or reading data from the process memory.
        """
        sp, fp   = self.get_stack_frame_range()
        size     = fp - sp
        if max_size and size > max_size:
            size = max_size
        return self.get_process().peek(sp, size)

    def read_stack_data(self, size = 128, offset = 0):
        """
        Reads the contents of the top of the stack.

        @type  size: int
        @param size: Number of bytes to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  str
        @return: Stack data.

        @raise WindowsError: Could not read the requested data.
        """
        aProcess = self.get_process()
        return aProcess.read(self.get_sp() + offset, size)

    def peek_stack_data(self, size = 128, offset = 0):
        """
        Tries to read the contents of the top of the stack.

        @type  size: int
        @param size: Number of bytes to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  str
        @return: Stack data.
            Returned data may be less than the requested size.
        """
        aProcess = self.get_process()
        return aProcess.peek(self.get_sp() + offset, size)

    def read_stack_dwords(self, count, offset = 0):
        """
        Reads DWORDs from the top of the stack.

        @type  count: int
        @param count: Number of DWORDs to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  tuple( int... )
        @return: Tuple of integers read from the stack.

        @raise WindowsError: Could not read the requested data.
        """
        if count > 0:
            stackData = self.read_stack_data(count * 4, offset)
            return struct.unpack('<'+('L'*count), stackData)
        return ()

    def peek_stack_dwords(self, count, offset = 0):
        """
        Tries to read DWORDs from the top of the stack.

        @type  count: int
        @param count: Number of DWORDs to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  tuple( int... )
        @return: Tuple of integers read from the stack.
            May be less than the requested number of DWORDs.
        """
        stackData = self.peek_stack_data(count * 4, offset)
        if len(stackData) & 3:
            stackData = stackData[:-len(stackData) & 3]
        if not stackData:
            return ()
        return struct.unpack('<'+('L'*count), stackData)

    def read_stack_qwords(self, count, offset = 0):
        """
        Reads QWORDs from the top of the stack.

        @type  count: int
        @param count: Number of QWORDs to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  tuple( int... )
        @return: Tuple of integers read from the stack.

        @raise WindowsError: Could not read the requested data.
        """
        stackData = self.read_stack_data(count * 8, offset)
        return struct.unpack('<'+('Q'*count), stackData)

    def peek_stack_qwords(self, count, offset = 0):
        """
        Tries to read QWORDs from the top of the stack.

        @type  count: int
        @param count: Number of QWORDs to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  tuple( int... )
        @return: Tuple of integers read from the stack.
            May be less than the requested number of QWORDs.
        """
        stackData = self.peek_stack_data(count * 8, offset)
        if len(stackData) & 7:
            stackData = stackData[:-len(stackData) & 7]
        if not stackData:
            return ()
        return struct.unpack('<'+('Q'*count), stackData)

    def read_stack_structure(self, structure, offset = 0):
        """
        Reads the given structure at the top of the stack.

        @type  structure: ctypes.Structure
        @param structure: Structure of the data to read from the stack.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.
            The stack pointer is the same returned by the L{get_sp} method.

        @rtype:  tuple
        @return: Tuple of elements read from the stack. The type of each
            element matches the types in the stack frame structure.
        """
        aProcess  = self.get_process()
        stackData = aProcess.read_structure(self.get_sp() + offset, structure)
        return tuple([ stackData.__getattribute__(name)
                       for (name, type) in stackData._fields_ ])

    def read_stack_frame(self, structure, offset = 0):
        """
        Reads the stack frame of the thread.

        @type  structure: ctypes.Structure
        @param structure: Structure of the stack frame.

        @type  offset: int
        @param offset: Offset from the frame pointer to begin reading.
            The frame pointer is the same returned by the L{get_fp} method.

        @rtype:  tuple
        @return: Tuple of elements read from the stack frame. The type of each
            element matches the types in the stack frame structure.
        """
        aProcess  = self.get_process()
        stackData = aProcess.read_structure(self.get_fp() + offset, structure)
        return tuple([ stackData.__getattribute__(name)
                       for (name, type) in stackData._fields_ ])

    def read_code_bytes(self, size = 128, offset = 0):
        """
        Tries to read some bytes of the code currently being executed.

        @type  size: int
        @param size: Number of bytes to read.

        @type  offset: int
        @param offset: Offset from the program counter to begin reading.

        @rtype:  str
        @return: Bytes read from the process memory.

        @raise WindowsError: Could not read the requested data.
        """
        return self.get_process().read(self.get_pc() + offset, size)

    def peek_code_bytes(self, size = 128, offset = 0):
        """
        Tries to read some bytes of the code currently being executed.

        @type  size: int
        @param size: Number of bytes to read.

        @type  offset: int
        @param offset: Offset from the program counter to begin reading.

        @rtype:  str
        @return: Bytes read from the process memory.
            May be less than the requested number of bytes.
        """
        return self.get_process().peek(self.get_pc() + offset, size)

    def peek_pointers_in_registers(self, peekSize = 16, context = None):
        """
        Tries to guess which values in the registers are valid pointers,
        and reads some data from them.

        @type  peekSize: int
        @param peekSize: Number of bytes to read from each pointer found.

        @type  context: dict( str S{->} int )
        @param context: (Optional)
            Dictionary mapping register names to their values.
            If not given, the current thread context will be used.

        @rtype:  dict( str S{->} str )
        @return: Dictionary mapping register names to the data they point to.
        """
        peekable_registers = (
            'Eax', 'Ebx', 'Ecx', 'Edx', 'Esi', 'Edi', 'Ebp'
        )
        if not context:
            context = self.get_context(win32.CONTEXT_CONTROL | \
                                       win32.CONTEXT_INTEGER)
        aProcess    = self.get_process()
        data        = dict()
        for (reg_name, reg_value) in compat.iteritems(context):
            if reg_name not in peekable_registers:
                continue
##            if reg_name == 'Ebp':
##                stack_begin, stack_end = self.get_stack_range()
##                print hex(stack_end), hex(reg_value), hex(stack_begin)
##                if stack_begin and stack_end and stack_end < stack_begin and \
##                   stack_begin <= reg_value <= stack_end:
##                      continue
            reg_data = aProcess.peek(reg_value, peekSize)
            if reg_data:
                data[reg_name] = reg_data
        return data

    # TODO
    # try to avoid reading the same page twice by caching it
    def peek_pointers_in_data(self, data, peekSize = 16, peekStep = 1):
        """
        Tries to guess which values in the given data are valid pointers,
        and reads some data from them.

        @type  data: str
        @param data: Binary data to find pointers in.

        @type  peekSize: int
        @param peekSize: Number of bytes to read from each pointer found.

        @type  peekStep: int
        @param peekStep: Expected data alignment.
            Tipically you specify 1 when data alignment is unknown,
            or 4 when you expect data to be DWORD aligned.
            Any other value may be specified.

        @rtype:  dict( str S{->} str )
        @return: Dictionary mapping stack offsets to the data they point to.
        """
        aProcess = self.get_process()
        return aProcess.peek_pointers_in_data(data, peekSize, peekStep)

#------------------------------------------------------------------------------

    # TODO
    # The disassemble_around and disassemble_around_pc methods
    # should take as parameter instruction counts rather than sizes

    def disassemble_string(self, lpAddress, code):
        """
        Disassemble instructions from a block of binary code.

        @type  lpAddress: int
        @param lpAddress: Memory address where the code was read from.

        @type  code: str
        @param code: Binary code to disassemble.

        @rtype:  list of tuple( long, int, str, str )
        @return: List of tuples. Each tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
        aProcess = self.get_process()
        return aProcess.disassemble_string(lpAddress, code)

    def disassemble(self, lpAddress, dwSize):
        """
        Disassemble instructions from the address space of the process.

        @type  lpAddress: int
        @param lpAddress: Memory address where to read the code from.

        @type  dwSize: int
        @param dwSize: Size of binary code to disassemble.

        @rtype:  list of tuple( long, int, str, str )
        @return: List of tuples. Each tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
        aProcess = self.get_process()
        return aProcess.disassemble(lpAddress, dwSize)

    def disassemble_around(self, lpAddress, dwSize = 64):
        """
        Disassemble around the given address.

        @type  lpAddress: int
        @param lpAddress: Memory address where to read the code from.

        @type  dwSize: int
        @param dwSize: Delta offset.
            Code will be read from lpAddress - dwSize to lpAddress + dwSize.

        @rtype:  list of tuple( long, int, str, str )
        @return: List of tuples. Each tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
        aProcess = self.get_process()
        return aProcess.disassemble_around(lpAddress, dwSize)

    def disassemble_around_pc(self, dwSize = 64):
        """
        Disassemble around the program counter of the given thread.

        @type  dwSize: int
        @param dwSize: Delta offset.
            Code will be read from pc - dwSize to pc + dwSize.

        @rtype:  list of tuple( long, int, str, str )
        @return: List of tuples. Each tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
        aProcess = self.get_process()
        return aProcess.disassemble_around(self.get_pc(), dwSize)

    def disassemble_instruction(self, lpAddress):
        """
        Disassemble the instruction at the given memory address.

        @type  lpAddress: int
        @param lpAddress: Memory address where to read the code from.

        @rtype:  tuple( long, int, str, str )
        @return: The tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
        aProcess = self.get_process()
        return aProcess.disassemble(lpAddress, 15)[0]

    def disassemble_current(self):
        """
        Disassemble the instruction at the program counter of the given thread.

        @rtype:  tuple( long, int, str, str )
        @return: The tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
        return self.disassemble_instruction( self.get_pc() )

#==============================================================================

class _ThreadContainer (object):
    """
    Encapsulates the capability to contain Thread objects.

    @group Instrumentation:
        start_thread

    @group Threads snapshot:
        scan_threads,
        get_thread, get_thread_count, get_thread_ids,
        has_thread, iter_threads, iter_thread_ids,
        find_threads_by_name, get_windows,
        clear_threads, clear_dead_threads, close_thread_handles
    """

    def __init__(self):
        self.__threadDict = dict()

    def __initialize_snapshot(self):
        """
        Private method to automatically initialize the snapshot
        when you try to use it without calling any of the scan_*
        methods first. You don't need to call this yourself.
        """
        if not self.__threadDict:
            self.scan_threads()

    def __contains__(self, anObject):
        """
        @type  anObject: L{Thread}, int
        @param anObject:
             - C{int}: Global ID of the thread to look for.
             - C{Thread}: Thread object to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains
            a L{Thread} object with the same ID.
        """
        if isinstance(anObject, Thread):
            anObject = anObject.dwThreadId
        return self.has_thread(anObject)

    def __iter__(self):
        """
        @see:    L{iter_threads}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Thread} objects in this snapshot.
        """
        return self.iter_threads()

    def __len__(self):
        """
        @see:    L{get_thread_count}
        @rtype:  int
        @return: Count of L{Thread} objects in this snapshot.
        """
        return self.get_thread_count()

    def has_thread(self, dwThreadId):
        """
        @type  dwThreadId: int
        @param dwThreadId: Global ID of the thread to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains a
            L{Thread} object with the given global ID.
        """
        self.__initialize_snapshot()
        return dwThreadId in self.__threadDict

    def get_thread(self, dwThreadId):
        """
        @type  dwThreadId: int
        @param dwThreadId: Global ID of the thread to look for.

        @rtype:  L{Thread}
        @return: Thread object with the given global ID.
        """
        self.__initialize_snapshot()
        if dwThreadId not in self.__threadDict:
            msg = "Unknown thread ID: %d" % dwThreadId
            raise KeyError(msg)
        return self.__threadDict[dwThreadId]

    def iter_thread_ids(self):
        """
        @see:    L{iter_threads}
        @rtype:  dictionary-keyiterator
        @return: Iterator of global thread IDs in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.iterkeys(self.__threadDict)

    def iter_threads(self):
        """
        @see:    L{iter_thread_ids}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Thread} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.itervalues(self.__threadDict)

    def get_thread_ids(self):
        """
        @rtype:  list( int )
        @return: List of global thread IDs in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.keys(self.__threadDict)

    def get_thread_count(self):
        """
        @rtype:  int
        @return: Count of L{Thread} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return len(self.__threadDict)

#------------------------------------------------------------------------------

    def find_threads_by_name(self, name, bExactMatch = True):
        """
        Find threads by name, using different search methods.

        @type  name: str, None
        @param name: Name to look for. Use C{None} to find nameless threads.

        @type  bExactMatch: bool
        @param bExactMatch: C{True} if the name must be
            B{exactly} as given, C{False} if the name can be
            loosely matched.

            This parameter is ignored when C{name} is C{None}.

        @rtype:  list( L{Thread} )
        @return: All threads matching the given name.
        """
        found_threads = list()

        # Find threads with no name.
        if name is None:
            for aThread in self.iter_threads():
                if aThread.get_name() is None:
                    found_threads.append(aThread)

        # Find threads matching the given name exactly.
        elif bExactMatch:
            for aThread in self.iter_threads():
                if aThread.get_name() == name:
                    found_threads.append(aThread)

        # Find threads whose names match the given substring.
        else:
            for aThread in self.iter_threads():
                t_name = aThread.get_name()
                if t_name is not None and name in t_name:
                    found_threads.append(aThread)

        return found_threads

#------------------------------------------------------------------------------

    # XXX TODO
    # Support for string searches on the window captions.

    def get_windows(self):
        """
        @rtype:  list of L{Window}
        @return: Returns a list of windows handled by this process.
        """
        window_list = list()
        for thread in self.iter_threads():
            window_list.extend( thread.get_windows() )
        return window_list

#------------------------------------------------------------------------------

    def start_thread(self, lpStartAddress, lpParameter=0,  bSuspended = False):
        """
        Remotely creates a new thread in the process.

        @type  lpStartAddress: int
        @param lpStartAddress: Start address for the new thread.

        @type  lpParameter: int
        @param lpParameter: Optional argument for the new thread.

        @type  bSuspended: bool
        @param bSuspended: C{True} if the new thread should be suspended.
            In that case use L{Thread.resume} to start execution.
        """
        if bSuspended:
            dwCreationFlags = win32.CREATE_SUSPENDED
        else:
            dwCreationFlags = 0
        hProcess = self.get_handle( win32.PROCESS_CREATE_THREAD     |
                                    win32.PROCESS_QUERY_INFORMATION |
                                    win32.PROCESS_VM_OPERATION      |
                                    win32.PROCESS_VM_WRITE          |
                                    win32.PROCESS_VM_READ           )
        hThread, dwThreadId = win32.CreateRemoteThread(
                hProcess, 0, 0, lpStartAddress, lpParameter, dwCreationFlags)
        aThread = Thread(dwThreadId, hThread, self)
        self._add_thread(aThread)
        return aThread

#------------------------------------------------------------------------------

    # TODO
    # maybe put all the toolhelp code into their own set of classes?
    #
    # XXX this method musn't end up calling __initialize_snapshot by accident!
    def scan_threads(self):
        """
        Populates the snapshot with running threads.
        """

        # Ignore special process IDs.
        # PID 0: System Idle Process. Also has a special meaning to the
        #        toolhelp APIs (current process).
        # PID 4: System Integrity Group. See this forum post for more info:
        #        http://tinyurl.com/ycza8jo
        #        (points to social.technet.microsoft.com)
        #        Only on XP and above
        # PID 8: System (?) only in Windows 2000 and below AFAIK.
        #        It's probably the same as PID 4 in XP and above.
        dwProcessId = self.get_pid()
        if dwProcessId in (0, 4, 8):
            return

##        dead_tids   = set( self.get_thread_ids() ) # XXX triggers a scan
        dead_tids   = self._get_thread_ids()
        dwProcessId = self.get_pid()
        hSnapshot   = win32.CreateToolhelp32Snapshot(win32.TH32CS_SNAPTHREAD,
                                                                 dwProcessId)
        try:
            te = win32.Thread32First(hSnapshot)
            while te is not None:
                if te.th32OwnerProcessID == dwProcessId:
                    dwThreadId = te.th32ThreadID
                    if dwThreadId in dead_tids:
                        dead_tids.remove(dwThreadId)
##                    if not self.has_thread(dwThreadId): # XXX triggers a scan
                    if not self._has_thread_id(dwThreadId):
                        aThread = Thread(dwThreadId, process = self)
                        self._add_thread(aThread)
                te = win32.Thread32Next(hSnapshot)
        finally:
            win32.CloseHandle(hSnapshot)
        for tid in dead_tids:
            self._del_thread(tid)

    def clear_dead_threads(self):
        """
        Remove Thread objects from the snapshot
        referring to threads no longer running.
        """
        for tid in self.get_thread_ids():
            aThread = self.get_thread(tid)
            if not aThread.is_alive():
                self._del_thread(aThread)

    def clear_threads(self):
        """
        Clears the threads snapshot.
        """
        for aThread in compat.itervalues(self.__threadDict):
            aThread.clear()
        self.__threadDict = dict()

    def close_thread_handles(self):
        """
        Closes all open handles to threads in the snapshot.
        """
        for aThread in self.iter_threads():
            try:
                aThread.close_handle()
            except Exception:
                try:
                    e = sys.exc_info()[1]
                    msg = "Cannot close thread handle %s, reason: %s"
                    msg %= (aThread.hThread.value, str(e))
                    warnings.warn(msg)
                except Exception:
                    pass

#------------------------------------------------------------------------------

    # XXX _notify_* methods should not trigger a scan

    def _add_thread(self, aThread):
        """
        Private method to add a thread object to the snapshot.

        @type  aThread: L{Thread}
        @param aThread: Thread object.
        """
##        if not isinstance(aThread, Thread):
##            if hasattr(aThread, '__class__'):
##                typename = aThread.__class__.__name__
##            else:
##                typename = str(type(aThread))
##            msg = "Expected Thread, got %s instead" % typename
##            raise TypeError(msg)
        dwThreadId = aThread.dwThreadId
##        if dwThreadId in self.__threadDict:
##            msg = "Already have a Thread object with ID %d" % dwThreadId
##            raise KeyError(msg)
        aThread.set_process(self)
        self.__threadDict[dwThreadId] = aThread

    def _del_thread(self, dwThreadId):
        """
        Private method to remove a thread object from the snapshot.

        @type  dwThreadId: int
        @param dwThreadId: Global thread ID.
        """
        try:
            aThread = self.__threadDict[dwThreadId]
            del self.__threadDict[dwThreadId]
        except KeyError:
            aThread = None
            msg = "Unknown thread ID %d" % dwThreadId
            warnings.warn(msg, RuntimeWarning)
        if aThread:
            aThread.clear()     # remove circular references

    def _has_thread_id(self, dwThreadId):
        """
        Private method to test for a thread in the snapshot without triggering
        an automatic scan.
        """
        return dwThreadId in self.__threadDict

    def _get_thread_ids(self):
        """
        Private method to get the list of thread IDs currently in the snapshot
        without triggering an automatic scan.
        """
        return compat.keys(self.__threadDict)

    def __add_created_thread(self, event):
        """
        Private method to automatically add new thread objects from debug events.

        @type  event: L{Event}
        @param event: Event object.
        """
        dwThreadId  = event.get_tid()
        hThread     = event.get_thread_handle()
##        if not self.has_thread(dwThreadId):   # XXX this would trigger a scan
        if not self._has_thread_id(dwThreadId):
            aThread = Thread(dwThreadId, hThread, self)
            teb_ptr = event.get_teb()   # remember the TEB pointer
            if teb_ptr:
                aThread._teb_ptr = teb_ptr
            self._add_thread(aThread)
        #else:
        #    aThread = self.get_thread(dwThreadId)
        #    if hThread != win32.INVALID_HANDLE_VALUE:
        #        aThread.hThread = hThread   # may have more privileges

    def _notify_create_process(self, event):
        """
        Notify the creation of the main thread of this process.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{CreateProcessEvent}
        @param event: Create process event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        self.__add_created_thread(event)
        return True

    def _notify_create_thread(self, event):
        """
        Notify the creation of a new thread in this process.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{CreateThreadEvent}
        @param event: Create thread event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        self.__add_created_thread(event)
        return True

    def _notify_exit_thread(self, event):
        """
        Notify the termination of a thread.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{ExitThreadEvent}
        @param event: Exit thread event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        dwThreadId = event.get_tid()
##        if self.has_thread(dwThreadId):   # XXX this would trigger a scan
        if self._has_thread_id(dwThreadId):
            self._del_thread(dwThreadId)
        return True
