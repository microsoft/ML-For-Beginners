#!~/.wine/drive_c/Python25/python.exe
# -*- coding: utf-8 -*-

# Acknowledgements:
#  Nicolas Economou, for his command line debugger on which this is inspired.
#  http://tinyurl.com/nicolaseconomou

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
Interactive debugging console.

@group Debugging:
    ConsoleDebugger

@group Exceptions:
    CmdError
"""

from __future__ import with_statement

__revision__ = "$Id$"

__all__ = [ 'ConsoleDebugger', 'CmdError' ]

# TODO document this module with docstrings.
# TODO command to set a last error breakpoint.
# TODO command to show available plugins.

from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.util import PathOperations
from winappdbg.event import EventHandler, NoEvent
from winappdbg.textio import HexInput, HexOutput, HexDump, CrashDump, DebugLog

import os
import sys
import code
import time
import warnings
import traceback

# too many variables named "cmd" to have a module by the same name :P
from cmd import Cmd

# lazy imports
readline = None

#==============================================================================


class DummyEvent (NoEvent):
    "Dummy event object used internally by L{ConsoleDebugger}."

    def get_pid(self):
        return self._pid

    def get_tid(self):
        return self._tid

    def get_process(self):
        return self._process

    def get_thread(self):
        return self._thread

#==============================================================================


class CmdError (Exception):
    """
    Exception raised when a command parsing error occurs.
    Used internally by L{ConsoleDebugger}.
    """

#==============================================================================


class ConsoleDebugger (Cmd, EventHandler):
    """
    Interactive console debugger.

    @see: L{Debug.interactive}
    """

#------------------------------------------------------------------------------
# Class variables

    # Exception to raise when an error occurs executing a command.
    command_error_exception = CmdError

    # Milliseconds to wait for debug events in the main loop.
    dwMilliseconds = 100

    # History file name.
    history_file = '.winappdbg_history'

    # Confirm before quitting?
    confirm_quit = True

    # Valid plugin name characters.
    valid_plugin_name_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXY' \
                              'abcdefghijklmnopqrstuvwxy' \
                              '012345678'                 \
                              '_'

    # Names of the registers.
    segment_names = ('cs', 'ds', 'es', 'fs', 'gs')

    register_alias_64_to_32 = {
        'eax':'Rax', 'ebx':'Rbx', 'ecx':'Rcx', 'edx':'Rdx',
        'eip':'Rip', 'ebp':'Rbp', 'esp':'Rsp', 'esi':'Rsi', 'edi':'Rdi'
    }
    register_alias_64_to_16 = { 'ax':'Rax', 'bx':'Rbx', 'cx':'Rcx', 'dx':'Rdx' }
    register_alias_64_to_8_low = { 'al':'Rax', 'bl':'Rbx', 'cl':'Rcx', 'dl':'Rdx' }
    register_alias_64_to_8_high = { 'ah':'Rax', 'bh':'Rbx', 'ch':'Rcx', 'dh':'Rdx' }
    register_alias_32_to_16 = { 'ax':'Eax', 'bx':'Ebx', 'cx':'Ecx', 'dx':'Edx' }
    register_alias_32_to_8_low = { 'al':'Eax', 'bl':'Ebx', 'cl':'Ecx', 'dl':'Edx' }
    register_alias_32_to_8_high = { 'ah':'Eax', 'bh':'Ebx', 'ch':'Ecx', 'dh':'Edx' }

    register_aliases_full_32 = list(segment_names)
    register_aliases_full_32.extend(compat.iterkeys(register_alias_32_to_16))
    register_aliases_full_32.extend(compat.iterkeys(register_alias_32_to_8_low))
    register_aliases_full_32.extend(compat.iterkeys(register_alias_32_to_8_high))
    register_aliases_full_32 = tuple(register_aliases_full_32)

    register_aliases_full_64 = list(segment_names)
    register_aliases_full_64.extend(compat.iterkeys(register_alias_64_to_32))
    register_aliases_full_64.extend(compat.iterkeys(register_alias_64_to_16))
    register_aliases_full_64.extend(compat.iterkeys(register_alias_64_to_8_low))
    register_aliases_full_64.extend(compat.iterkeys(register_alias_64_to_8_high))
    register_aliases_full_64 = tuple(register_aliases_full_64)

    # Names of the control flow instructions.
    jump_instructions = (
        'jmp', 'jecxz', 'jcxz',
        'ja', 'jnbe', 'jae', 'jnb', 'jb', 'jnae', 'jbe', 'jna', 'jc', 'je',
        'jz', 'jnc', 'jne', 'jnz', 'jnp', 'jpo', 'jp', 'jpe', 'jg', 'jnle',
        'jge', 'jnl', 'jl', 'jnge', 'jle', 'jng', 'jno', 'jns', 'jo', 'js'
    )
    call_instructions = ('call', 'ret', 'retn')
    loop_instructions = ('loop', 'loopz', 'loopnz', 'loope', 'loopne')
    control_flow_instructions = call_instructions + loop_instructions + \
                                jump_instructions

#------------------------------------------------------------------------------
# Instance variables

    def __init__(self):
        """
        Interactive console debugger.

        @see: L{Debug.interactive}
        """
        Cmd.__init__(self)
        EventHandler.__init__(self)

        # Quit the debugger when True.
        self.debuggerExit = False

        # Full path to the history file.
        self.history_file_full_path = None

        # Last executed command.
        self.__lastcmd = ""

#------------------------------------------------------------------------------
# Debugger

    # Use this Debug object.
    def start_using_debugger(self, debug):

        # Clear the previous Debug object.
        self.stop_using_debugger()

        # Keep the Debug object.
        self.debug = debug

        # Set ourselves as the event handler for the debugger.
        self.prevHandler = debug.set_event_handler(self)

    # Stop using the Debug object given by start_using_debugger().
    # Circular references must be removed, or the destructors never get called.
    def stop_using_debugger(self):
        if hasattr(self, 'debug'):
            debug = self.debug
            debug.set_event_handler(self.prevHandler)
            del self.prevHandler
            del self.debug
            return debug
        return None

    # Destroy the Debug object.
    def destroy_debugger(self, autodetach=True):
        debug = self.stop_using_debugger()
        if debug is not None:
            if not autodetach:
                debug.kill_all(bIgnoreExceptions=True)
                debug.lastEvent = None
            debug.stop()
        del debug

    @property
    def lastEvent(self):
        return self.debug.lastEvent

    def set_fake_last_event(self, process):
        if self.lastEvent is None:
            self.debug.lastEvent = DummyEvent(self.debug)
            self.debug.lastEvent._process = process
            self.debug.lastEvent._thread = process.get_thread(
                                                process.get_thread_ids()[0])
            self.debug.lastEvent._pid = process.get_pid()
            self.debug.lastEvent._tid = self.lastEvent._thread.get_tid()

#------------------------------------------------------------------------------
# Input

# TODO
# * try to guess breakpoints when insufficient data is given
# * child Cmd instances will have to be used for other prompts, for example
#   when assembling or editing memory - it may also be a good idea to think
#   if it's possible to make the main Cmd instance also a child, instead of
#   the debugger itself - probably the same goes for the EventHandler, maybe
#   it can be used as a contained object rather than a parent class.

    # Join a token list into an argument string.
    def join_tokens(self, token_list):
        return self.debug.system.argv_to_cmdline(token_list)

    # Split an argument string into a token list.
    def split_tokens(self, arg, min_count=0, max_count=None):
        token_list = self.debug.system.cmdline_to_argv(arg)
        if len(token_list) < min_count:
            raise CmdError("missing parameters.")
        if max_count and len(token_list) > max_count:
            raise CmdError("too many parameters.")
        return token_list

    # Token is a thread ID or name.
    def input_thread(self, token):
        targets = self.input_thread_list([token])
        if len(targets) == 0:
            raise CmdError("missing thread name or ID")
        if len(targets) > 1:
            msg = "more than one thread with that name:\n"
            for tid in targets:
                msg += "\t%d\n" % tid
            msg = msg[:-len("\n")]
            raise CmdError(msg)
        return targets[0]

    # Token list is a list of thread IDs or names.
    def input_thread_list(self, token_list):
        targets = set()
        system = self.debug.system
        for token in token_list:
            try:
                tid = self.input_integer(token)
                if not system.has_thread(tid):
                    raise CmdError("thread not found (%d)" % tid)
                targets.add(tid)
            except ValueError:
                found = set()
                for process in system.iter_processes():
                    found.update(system.find_threads_by_name(token))
                if not found:
                    raise CmdError("thread not found (%s)" % token)
                for thread in found:
                    targets.add(thread.get_tid())
        targets = list(targets)
        targets.sort()
        return targets

    # Token is a process ID or name.
    def input_process(self, token):
        targets = self.input_process_list([token])
        if len(targets) == 0:
            raise CmdError("missing process name or ID")
        if len(targets) > 1:
            msg = "more than one process with that name:\n"
            for pid in targets:
                msg += "\t%d\n" % pid
            msg = msg[:-len("\n")]
            raise CmdError(msg)
        return targets[0]

    # Token list is a list of process IDs or names.
    def input_process_list(self, token_list):
        targets = set()
        system = self.debug.system
        for token in token_list:
            try:
                pid = self.input_integer(token)
                if not system.has_process(pid):
                    raise CmdError("process not found (%d)" % pid)
                targets.add(pid)
            except ValueError:
                found = system.find_processes_by_filename(token)
                if not found:
                    raise CmdError("process not found (%s)" % token)
                for (process, _) in found:
                    targets.add(process.get_pid())
        targets = list(targets)
        targets.sort()
        return targets

    # Token is a command line to execute.
    def input_command_line(self, command_line):
        argv = self.debug.system.cmdline_to_argv(command_line)
        if not argv:
            raise CmdError("missing command line to execute")
        fname = argv[0]
        if not os.path.exists(fname):
            try:
                fname, _ = win32.SearchPath(None, fname, '.exe')
            except WindowsError:
                raise CmdError("file not found: %s" % fname)
            argv[0] = fname
            command_line = self.debug.system.argv_to_cmdline(argv)
        return command_line

    # Token is an integer.
    # Only hexadecimal format is supported.
    def input_hexadecimal_integer(self, token):
        return int(token, 0x10)

    # Token is an integer.
    # It can be in any supported format.
    def input_integer(self, token):
        return HexInput.integer(token)
# #    input_integer = input_hexadecimal_integer

    # Token is an address.
    # The address can be a integer, a label or a register.
    def input_address(self, token, pid=None, tid=None):
        address = None
        if self.is_register(token):
            if tid is None:
                if self.lastEvent is None or pid != self.lastEvent.get_pid():
                    msg = "can't resolve register (%s) for unknown thread"
                    raise CmdError(msg % token)
                tid = self.lastEvent.get_tid()
            address = self.input_register(token, tid)
        if address is None:
            try:
                address = self.input_hexadecimal_integer(token)
            except ValueError:
                if pid is None:
                    if self.lastEvent is None:
                        raise CmdError("no current process set")
                    process = self.lastEvent.get_process()
                elif self.lastEvent is not None and pid == self.lastEvent.get_pid():
                    process = self.lastEvent.get_process()
                else:
                    try:
                        process = self.debug.system.get_process(pid)
                    except KeyError:
                        raise CmdError("process not found (%d)" % pid)
                try:
                    address = process.resolve_label(token)
                except Exception:
                    raise CmdError("unknown address (%s)" % token)
        return address

    # Token is an address range, or a single address.
    # The addresses can be integers, labels or registers.
    def input_address_range(self, token_list, pid=None, tid=None):
        if len(token_list) == 2:
            token_1, token_2 = token_list
            address = self.input_address(token_1, pid, tid)
            try:
                size = self.input_integer(token_2)
            except ValueError:
                raise CmdError("bad address range: %s %s" % (token_1, token_2))
        elif len(token_list) == 1:
            token = token_list[0]
            if '-' in token:
                try:
                    token_1, token_2 = token.split('-')
                except Exception:
                    raise CmdError("bad address range: %s" % token)
                address = self.input_address(token_1, pid, tid)
                size = self.input_address(token_2, pid, tid) - address
            else:
                address = self.input_address(token, pid, tid)
                size = None
        return address, size

    # XXX TODO
    # Support non-integer registers here.
    def is_register(self, token):
        if win32.arch == 'i386':
            if token in self.register_aliases_full_32:
                return True
            token = token.title()
            for (name, typ) in win32.CONTEXT._fields_:
                if name == token:
                    return win32.sizeof(typ) == win32.sizeof(win32.DWORD)
        elif win32.arch == 'amd64':
            if token in self.register_aliases_full_64:
                return True
            token = token.title()
            for (name, typ) in win32.CONTEXT._fields_:
                if name == token:
                    return win32.sizeof(typ) == win32.sizeof(win32.DWORD64)
        return False

    # The token is a register name.
    # Returns None if no register name is matched.
    def input_register(self, token, tid=None):
        if tid is None:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            thread = self.lastEvent.get_thread()
        else:
            thread = self.debug.system.get_thread(tid)
        ctx = thread.get_context()

        token = token.lower()
        title = token.title()

        if title in ctx:
            return ctx.get(title)  # eax -> Eax

        if ctx.arch == 'i386':

            if token in self.segment_names:
                return ctx.get('Seg%s' % title)  # cs -> SegCs

            if token in self.register_alias_32_to_16:
                return ctx.get(self.register_alias_32_to_16[token]) & 0xFFFF

            if token in self.register_alias_32_to_8_low:
                return ctx.get(self.register_alias_32_to_8_low[token]) & 0xFF

            if token in self.register_alias_32_to_8_high:
                return (ctx.get(self.register_alias_32_to_8_high[token]) & 0xFF00) >> 8

        elif ctx.arch == 'amd64':

            if token in self.segment_names:
                return ctx.get('Seg%s' % title)  # cs -> SegCs

            if token in self.register_alias_64_to_32:
                return ctx.get(self.register_alias_64_to_32[token]) & 0xFFFFFFFF

            if token in self.register_alias_64_to_16:
                return ctx.get(self.register_alias_64_to_16[token]) & 0xFFFF

            if token in self.register_alias_64_to_8_low:
                return ctx.get(self.register_alias_64_to_8_low[token]) & 0xFF

            if token in self.register_alias_64_to_8_high:
                return (ctx.get(self.register_alias_64_to_8_high[token]) & 0xFF00) >> 8

        return None

    # Token list contains an address or address range.
    # The prefix is also parsed looking for process and thread IDs.
    def input_full_address_range(self, token_list):
        pid, tid = self.get_process_and_thread_ids_from_prefix()
        address, size = self.input_address_range(token_list, pid, tid)
        return pid, tid, address, size

    # Token list contains a breakpoint.
    def input_breakpoint(self, token_list):
        pid, tid, address, size = self.input_full_address_range(token_list)
        if not self.debug.is_debugee(pid):
            raise CmdError("target process is not being debugged")
        return pid, tid, address, size

    # Token list contains a memory address, and optional size and process.
    # Sets the results as the default for the next display command.
    def input_display(self, token_list, default_size=64):
        pid, tid, address, size = self.input_full_address_range(token_list)
        if not size:
            size = default_size
        next_address = HexOutput.integer(address + size)
        self.default_display_target = next_address
        return pid, tid, address, size

#------------------------------------------------------------------------------
# Output

    # Tell the user a module was loaded.
    def print_module_load(self, event):
        mod = event.get_module()
        base = mod.get_base()
        name = mod.get_filename()
        if not name:
            name = ''
        msg = "Loaded module (%s) %s"
        msg = msg % (HexDump.address(base), name)
        print(msg)

    # Tell the user a module was unloaded.
    def print_module_unload(self, event):
        mod = event.get_module()
        base = mod.get_base()
        name = mod.get_filename()
        if not name:
            name = ''
        msg = "Unloaded module (%s) %s"
        msg = msg % (HexDump.address(base), name)
        print(msg)

    # Tell the user a process was started.
    def print_process_start(self, event):
        pid = event.get_pid()
        start = event.get_start_address()
        if start:
            start = HexOutput.address(start)
            print("Started process %d at %s" % (pid, start))
        else:
            print("Attached to process %d" % pid)

    # Tell the user a thread was started.
    def print_thread_start(self, event):
        tid = event.get_tid()
        start = event.get_start_address()
        if start:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                start = event.get_process().get_label_at_address(start)
            print("Started thread %d at %s" % (tid, start))
        else:
            print("Attached to thread %d" % tid)

    # Tell the user a process has finished.
    def print_process_end(self, event):
        pid = event.get_pid()
        code = event.get_exit_code()
        print("Process %d terminated, exit code %d" % (pid, code))

    # Tell the user a thread has finished.
    def print_thread_end(self, event):
        tid = event.get_tid()
        code = event.get_exit_code()
        print("Thread %d terminated, exit code %d" % (tid, code))

    # Print(debug strings.
    def print_debug_string(self, event):
        tid = event.get_tid()
        string = event.get_debug_string()
        print("Thread %d says: %r" % (tid, string))

    # Inform the user of any other debugging event.
    def print_event(self, event):
        code = HexDump.integer(event.get_event_code())
        name = event.get_event_name()
        desc = event.get_event_description()
        if code in desc:
            print('')
            print("%s: %s" % (name, desc))
        else:
            print('')
            print("%s (%s): %s" % (name, code, desc))
        self.print_event_location(event)

    # Stop on exceptions and prompt for commands.
    def print_exception(self, event):
        address = HexDump.address(event.get_exception_address())
        code = HexDump.integer(event.get_exception_code())
        desc = event.get_exception_description()
        if event.is_first_chance():
            chance = 'first'
        else:
            chance = 'second'
        if code in desc:
            msg = "%s at address %s (%s chance)" % (desc, address, chance)
        else:
            msg = "%s (%s) at address %s (%s chance)" % (desc, code, address, chance)
        print('')
        print(msg)
        self.print_event_location(event)

    # Show the current location in the code.
    def print_event_location(self, event):
        process = event.get_process()
        thread = event.get_thread()
        self.print_current_location(process, thread)

    # Show the current location in the code.
    def print_breakpoint_location(self, event):
        process = event.get_process()
        thread = event.get_thread()
        pc = event.get_exception_address()
        self.print_current_location(process, thread, pc)

    # Show the current location in any process and thread.
    def print_current_location(self, process=None, thread=None, pc=None):
        if not process:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            process = self.lastEvent.get_process()
        if not thread:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            thread = self.lastEvent.get_thread()
        thread.suspend()
        try:
            if pc is None:
                pc = thread.get_pc()
            ctx = thread.get_context()
        finally:
            thread.resume()
        label = process.get_label_at_address(pc)
        try:
            disasm = process.disassemble(pc, 15)
        except WindowsError:
            disasm = None
        except NotImplementedError:
            disasm = None
        print('')
        print(CrashDump.dump_registers(ctx),)
        print("%s:" % label)
        if disasm:
            print(CrashDump.dump_code_line(disasm[0], pc, bShowDump=True))
        else:
            try:
                data = process.peek(pc, 15)
            except Exception:
                data = None
            if data:
                print('%s: %s' % (HexDump.address(pc), HexDump.hexblock_byte(data)))
            else:
                print('%s: ???' % HexDump.address(pc))

    # Display memory contents using a given method.
    def print_memory_display(self, arg, method):
        if not arg:
            arg = self.default_display_target
        token_list = self.split_tokens(arg, 1, 2)
        pid, tid, address, size = self.input_display(token_list)
        label = self.get_process(pid).get_label_at_address(address)
        data = self.read_memory(address, size, pid)
        if data:
            print("%s:" % label)
            print(method(data, address),)

#------------------------------------------------------------------------------
# Debugging

    # Get the process ID from the prefix or the last event.
    def get_process_id_from_prefix(self):
        if self.cmdprefix:
            pid = self.input_process(self.cmdprefix)
        else:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            pid = self.lastEvent.get_pid()
        return pid

    # Get the thread ID from the prefix or the last event.
    def get_thread_id_from_prefix(self):
        if self.cmdprefix:
            tid = self.input_thread(self.cmdprefix)
        else:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            tid = self.lastEvent.get_tid()
        return tid

    # Get the process from the prefix or the last event.
    def get_process_from_prefix(self):
        pid = self.get_process_id_from_prefix()
        return self.get_process(pid)

    # Get the thread from the prefix or the last event.
    def get_thread_from_prefix(self):
        tid = self.get_thread_id_from_prefix()
        return self.get_thread(tid)

    # Get the process and thread IDs from the prefix or the last event.
    def get_process_and_thread_ids_from_prefix(self):
        if self.cmdprefix:
            try:
                pid = self.input_process(self.cmdprefix)
                tid = None
            except CmdError:
                try:
                    tid = self.input_thread(self.cmdprefix)
                    pid = self.debug.system.get_thread(tid).get_pid()
                except CmdError:
                    msg = "unknown process or thread (%s)" % self.cmdprefix
                    raise CmdError(msg)
        else:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            pid = self.lastEvent.get_pid()
            tid = self.lastEvent.get_tid()
        return pid, tid

    # Get the process and thread from the prefix or the last event.
    def get_process_and_thread_from_prefix(self):
        pid, tid = self.get_process_and_thread_ids_from_prefix()
        process = self.get_process(pid)
        thread = self.get_thread(tid)
        return process, thread

    # Get the process object.
    def get_process(self, pid=None):
        if pid is None:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            process = self.lastEvent.get_process()
        elif self.lastEvent is not None and pid == self.lastEvent.get_pid():
            process = self.lastEvent.get_process()
        else:
            try:
                process = self.debug.system.get_process(pid)
            except KeyError:
                raise CmdError("process not found (%d)" % pid)
        return process

    # Get the thread object.
    def get_thread(self, tid=None):
        if tid is None:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            thread = self.lastEvent.get_thread()
        elif self.lastEvent is not None and tid == self.lastEvent.get_tid():
            thread = self.lastEvent.get_thread()
        else:
            try:
                thread = self.debug.system.get_thread(tid)
            except KeyError:
                raise CmdError("thread not found (%d)" % tid)
        return thread

    # Read the process memory.
    def read_memory(self, address, size, pid=None):
        process = self.get_process(pid)
        try:
            data = process.peek(address, size)
        except WindowsError:
            orig_address = HexOutput.integer(address)
            next_address = HexOutput.integer(address + size)
            msg = "error reading process %d, from %s to %s (%d bytes)"
            msg = msg % (pid, orig_address, next_address, size)
            raise CmdError(msg)
        return data

    # Write the process memory.
    def write_memory(self, address, data, pid=None):
        process = self.get_process(pid)
        try:
            process.write(address, data)
        except WindowsError:
            size = len(data)
            orig_address = HexOutput.integer(address)
            next_address = HexOutput.integer(address + size)
            msg = "error reading process %d, from %s to %s (%d bytes)"
            msg = msg % (pid, orig_address, next_address, size)
            raise CmdError(msg)

    # Change a register value.
    def change_register(self, register, value, tid=None):

        # Get the thread.
        if tid is None:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            thread = self.lastEvent.get_thread()
        else:
            try:
                thread = self.debug.system.get_thread(tid)
            except KeyError:
                raise CmdError("thread not found (%d)" % tid)

        # Convert the value to integer type.
        try:
            value = self.input_integer(value)
        except ValueError:
            pid = thread.get_pid()
            value = self.input_address(value, pid, tid)

        # Suspend the thread.
        # The finally clause ensures the thread is resumed before returning.
        thread.suspend()
        try:

            # Get the current context.
            ctx = thread.get_context()

            # Register name matching is case insensitive.
            register = register.lower()

            # Integer 32 bits registers.
            if register in self.register_names:
                register = register.title()  # eax -> Eax

            # Segment (16 bit) registers.
            if register in self.segment_names:
                register = 'Seg%s' % register.title()  # cs -> SegCs
                value = value & 0x0000FFFF

            # Integer 16 bits registers.
            if register in self.register_alias_16:
                register = self.register_alias_16[register]
                previous = ctx.get(register) & 0xFFFF0000
                value = (value & 0x0000FFFF) | previous

            # Integer 8 bits registers (low part).
            if register in self.register_alias_8_low:
                register = self.register_alias_8_low[register]
                previous = ctx.get(register) % 0xFFFFFF00
                value = (value & 0x000000FF) | previous

            # Integer 8 bits registers (high part).
            if register in self.register_alias_8_high:
                register = self.register_alias_8_high[register]
                previous = ctx.get(register) % 0xFFFF00FF
                value = ((value & 0x000000FF) << 8) | previous

            # Set the new context.
            ctx.__setitem__(register, value)
            thread.set_context(ctx)

        # Resume the thread.
        finally:
            thread.resume()

    # Very crude way to find data within the process memory.
    # TODO: Perhaps pfind.py can be integrated here instead.
    def find_in_memory(self, query, process):
        for mbi in process.get_memory_map():
            if mbi.State != win32.MEM_COMMIT or mbi.Protect & win32.PAGE_GUARD:
                continue
            address = mbi.BaseAddress
            size = mbi.RegionSize
            try:
                data = process.read(address, size)
            except WindowsError:
                msg = "*** Warning: read error at address %s"
                msg = msg % HexDump.address(address)
                print(msg)
            width = min(len(query), 16)
            p = data.find(query)
            while p >= 0:
                q = p + len(query)
                d = data[ p: min(q, p + width) ]
                h = HexDump.hexline(d, width=width)
                a = HexDump.address(address + p)
                print("%s: %s" % (a, h))
                p = data.find(query, q)

    # Kill a process.
    def kill_process(self, pid):
        process = self.debug.system.get_process(pid)
        try:
            process.kill()
            if self.debug.is_debugee(pid):
                self.debug.detach(pid)
            print("Killed process (%d)" % pid)
        except Exception:
            print("Error trying to kill process (%d)" % pid)

    # Kill a thread.
    def kill_thread(self, tid):
        thread = self.debug.system.get_thread(tid)
        try:
            thread.kill()
            process = thread.get_process()
            pid = process.get_pid()
            if self.debug.is_debugee(pid) and not process.is_alive():
                self.debug.detach(pid)
            print("Killed thread (%d)" % tid)
        except Exception:
            print("Error trying to kill thread (%d)" % tid)

#------------------------------------------------------------------------------
# Command prompt input

    # Prompt the user for commands.
    def prompt_user(self):
        while not self.debuggerExit:
            try:
                self.cmdloop()
                break
            except CmdError:
                e = sys.exc_info()[1]
                print("*** Error: %s" % str(e))
            except Exception:
                traceback.print_exc()
# #                self.debuggerExit = True

    # Prompt the user for a YES/NO kind of question.
    def ask_user(self, msg, prompt="Are you sure? (y/N): "):
        print(msg)
        answer = raw_input(prompt)
        answer = answer.strip()[:1].lower()
        return answer == 'y'

    # Autocomplete the given command when not ambiguous.
    # Convert it to lowercase (so commands are seen as case insensitive).
    def autocomplete(self, cmd):
        cmd = cmd.lower()
        completed = self.completenames(cmd)
        if len(completed) == 1:
            cmd = completed[0]
        return cmd

    # Get the help text for the given list of command methods.
    # Note it's NOT a list of commands, but a list of actual method names.
    # Each line of text is stripped and all lines are sorted.
    # Repeated text lines are removed.
    # Returns a single, possibly multiline, string.
    def get_help(self, commands):
        msg = set()
        for name in commands:
            if name != 'do_help':
                try:
                    doc = getattr(self, name).__doc__.split('\n')
                except Exception:
                    return ("No help available when Python"
                             " is run with the -OO switch.")
                for x in doc:
                    x = x.strip()
                    if x:
                        msg.add('  %s' % x)
        msg = list(msg)
        msg.sort()
        msg = '\n'.join(msg)
        return msg

    # Parse the prefix and remove it from the command line.
    def split_prefix(self, line):
        prefix = None
        if line.startswith('~'):
            pos = line.find(' ')
            if pos == 1:
                pos = line.find(' ', pos + 1)
            if not pos < 0:
                prefix = line[ 1: pos ].strip()
                line = line[ pos: ].strip()
        return prefix, line

#------------------------------------------------------------------------------
# Cmd() hacks

    # Header for help page.
    doc_header = 'Available commands (type help * or help <command>)'

# #    # Read and write directly to stdin and stdout.
# #    # This prevents the use of raw_input and print.
# #    use_rawinput = False

    @property
    def prompt(self):
        if self.lastEvent:
            pid = self.lastEvent.get_pid()
            tid = self.lastEvent.get_tid()
            if self.debug.is_debugee(pid):
# #                return '~%d(%d)> ' % (tid, pid)
                return '%d:%d> ' % (pid, tid)
        return '> '

    # Return a sorted list of method names.
    # Only returns the methods that implement commands.
    def get_names(self):
        names = Cmd.get_names(self)
        names = [ x for x in set(names) if x.startswith('do_') ]
        names.sort()
        return names

    # Automatically autocomplete commands, even if Tab wasn't pressed.
    # The prefix is removed from the line and stored in self.cmdprefix.
    # Also implement the commands that consist of a symbol character.
    def parseline(self, line):
        self.cmdprefix, line = self.split_prefix(line)
        line = line.strip()
        if line:
            if line[0] == '.':
                line = 'plugin ' + line[1:]
            elif line[0] == '#':
                line = 'python ' + line[1:]
        cmd, arg, line = Cmd.parseline(self, line)
        if cmd:
            cmd = self.autocomplete(cmd)
        return cmd, arg, line

# #    # Don't repeat the last executed command.
# #    def emptyline(self):
# #        pass

    # Reset the defaults for some commands.
    def preloop(self):
        self.default_disasm_target = 'eip'
        self.default_display_target = 'eip'
        self.last_display_command = self.do_db

    # Put the prefix back in the command line.
    def get_lastcmd(self):
        return self.__lastcmd

    def set_lastcmd(self, lastcmd):
        if self.cmdprefix:
            lastcmd = '~%s %s' % (self.cmdprefix, lastcmd)
        self.__lastcmd = lastcmd

    lastcmd = property(get_lastcmd, set_lastcmd)

    # Quit the command prompt if the debuggerExit flag is on.
    def postcmd(self, stop, line):
        return stop or self.debuggerExit

#------------------------------------------------------------------------------
# Commands

    # Each command contains a docstring with it's help text.
    # The help text consist of independent text lines,
    # where each line shows a command and it's parameters.
    # Each command method has the help message for itself and all it's aliases.
    # Only the docstring for the "help" command is shown as-is.

    # NOTE: Command methods MUST be all lowercase!

    # Extended help command.
    def do_help(self, arg):
        """
        ? - show the list of available commands
        ? * - show help for all commands
        ? <command> [command...] - show help for the given command(s)
        help - show the list of available commands
        help * - show help for all commands
        help <command> [command...] - show help for the given command(s)
        """
        if not arg:
            Cmd.do_help(self, arg)
        elif arg in ('?', 'help'):
            # An easter egg :)
            print("  Help! I need somebody...")
            print("  Help! Not just anybody...")
            print("  Help! You know, I need someone...")
            print("  Heeelp!")
        else:
            if arg == '*':
                commands = self.get_names()
                commands = [ x for x in commands if x.startswith('do_') ]
            else:
                commands = set()
                for x in arg.split(' '):
                    x = x.strip()
                    if x:
                        for n in self.completenames(x):
                            commands.add('do_%s' % n)
                commands = list(commands)
                commands.sort()
            print(self.get_help(commands))

    def do_shell(self, arg):
        """
        ! - spawn a system shell
        shell - spawn a system shell
        ! <command> [arguments...] - execute a single shell command
        shell <command> [arguments...] - execute a single shell command
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")

        # Try to use the environment to locate cmd.exe.
        # If not found, it's usually OK to just use the filename,
        # since cmd.exe is one of those "magic" programs that
        # can be automatically found by CreateProcess.
        shell = os.getenv('ComSpec', 'cmd.exe')

        # When given a command, run it and return.
        # When no command is given, spawn a shell.
        if arg:
            arg = '%s /c %s' % (shell, arg)
        else:
            arg = shell
        process = self.debug.system.start_process(arg, bConsole=True)
        process.wait()

    # This hack fixes a bug in Python, the interpreter console is closing the
    # stdin pipe when calling the exit() function (Ctrl+Z seems to work fine).
    class _PythonExit(object):

        def __repr__(self):
            return "Use exit() or Ctrl-Z plus Return to exit"

        def __call__(self):
            raise SystemExit()

    _python_exit = _PythonExit()

    # Spawns a Python shell with some handy local variables and the winappdbg
    # module already imported. Also the console banner is improved.
    def _spawn_python_shell(self, arg):
        import winappdbg
        banner = ('Python %s on %s\nType "help", "copyright", '
                 '"credits" or "license" for more information.\n')
        platform = winappdbg.version.lower()
        platform = 'WinAppDbg %s' % platform
        banner = banner % (sys.version, platform)
        local = {}
        local.update(__builtins__)
        local.update({
            '__name__': '__console__',
            '__doc__': None,
            'exit': self._python_exit,
            'self': self,
            'arg': arg,
            'winappdbg': winappdbg,
        })
        try:
            code.interact(banner=banner, local=local)
        except SystemExit:
            # We need to catch it so it doesn't kill our program.
            pass

    def do_python(self, arg):
        """
        # - spawn a python interpreter
        python - spawn a python interpreter
        # <statement> - execute a single python statement
        python <statement> - execute a single python statement
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")

        # When given a Python statement, execute it directly.
        if arg:
            try:
                compat.exec_(arg, globals(), locals())
            except Exception:
                traceback.print_exc()

        # When no statement is given, spawn a Python interpreter.
        else:
            try:
                self._spawn_python_shell(arg)
            except Exception:
                e = sys.exc_info()[1]
                raise CmdError(
                    "unhandled exception when running Python console: %s" % e)

    def do_quit(self, arg):
        """
        quit - close the debugging session
        q - close the debugging session
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        if arg:
            raise CmdError("too many arguments")
        if self.confirm_quit:
            count = self.debug.get_debugee_count()
            if count > 0:
                if count == 1:
                    msg = "There's a program still running."
                else:
                    msg = "There are %s programs still running." % count
                if not self.ask_user(msg):
                    return False
        self.debuggerExit = True
        return True

    do_q = do_quit

    def do_attach(self, arg):
        """
        attach <target> [target...] - attach to the given process(es)
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        targets = self.input_process_list(self.split_tokens(arg, 1))
        if not targets:
            print("Error: missing parameters")
        else:
            debug = self.debug
            for pid in targets:
                try:
                    debug.attach(pid)
                    print("Attached to process (%d)" % pid)
                except Exception:
                    print("Error: can't attach to process (%d)" % pid)

    def do_detach(self, arg):
        """
        [~process] detach - detach from the current process
        detach - detach from the current process
        detach <target> [target...] - detach from the given process(es)
        """
        debug = self.debug
        token_list = self.split_tokens(arg)
        if self.cmdprefix:
            token_list.insert(0, self.cmdprefix)
        targets = self.input_process_list(token_list)
        if not targets:
            if self.lastEvent is None:
                raise CmdError("no current process set")
            targets = [ self.lastEvent.get_pid() ]
        for pid in targets:
            try:
                debug.detach(pid)
                print("Detached from process (%d)" % pid)
            except Exception:
                print("Error: can't detach from process (%d)" % pid)

    def do_windowed(self, arg):
        """
        windowed <target> [arguments...] - run a windowed program for debugging
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        cmdline = self.input_command_line(arg)
        try:
            process = self.debug.execl(arg,
                                                bConsole=False,
                                                 bFollow=self.options.follow)
            print("Spawned process (%d)" % process.get_pid())
        except Exception:
            raise CmdError("can't execute")
        self.set_fake_last_event(process)

    def do_console(self, arg):
        """
        console <target> [arguments...] - run a console program for debugging
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        cmdline = self.input_command_line(arg)
        try:
            process = self.debug.execl(arg,
                                                bConsole=True,
                                                 bFollow=self.options.follow)
            print("Spawned process (%d)" % process.get_pid())
        except Exception:
            raise CmdError("can't execute")
        self.set_fake_last_event(process)

    def do_continue(self, arg):
        """
        continue - continue execution
        g - continue execution
        go - continue execution
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        if arg:
            raise CmdError("too many arguments")
        if self.debug.get_debugee_count() > 0:
            return True

    do_g = do_continue
    do_go = do_continue

    def do_gh(self, arg):
        """
        gh - go with exception handled
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        if arg:
            raise CmdError("too many arguments")
        if self.lastEvent:
            self.lastEvent.continueStatus = win32.DBG_EXCEPTION_HANDLED
        return self.do_go(arg)

    def do_gn(self, arg):
        """
        gn - go with exception not handled
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        if arg:
            raise CmdError("too many arguments")
        if self.lastEvent:
            self.lastEvent.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED
        return self.do_go(arg)

    def do_refresh(self, arg):
        """
        refresh - refresh the list of running processes and threads
        [~process] refresh - refresh the list of running threads
        """
        if arg:
            raise CmdError("too many arguments")
        if self.cmdprefix:
            process = self.get_process_from_prefix()
            process.scan()
        else:
            self.debug.system.scan()

    def do_processlist(self, arg):
        """
        pl - show the processes being debugged
        processlist - show the processes being debugged
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        if arg:
            raise CmdError("too many arguments")
        system = self.debug.system
        pid_list = self.debug.get_debugee_pids()
        if pid_list:
            print("Process ID   File name")
            for pid in pid_list:
                if   pid == 0:
                    filename = "System Idle Process"
                elif pid == 4:
                    filename = "System"
                else:
                    filename = system.get_process(pid).get_filename()
                    filename = PathOperations.pathname_to_filename(filename)
                print("%-12d %s" % (pid, filename))

    do_pl = do_processlist

    def do_threadlist(self, arg):
        """
        tl - show the threads being debugged
        threadlist - show the threads being debugged
        """
        if arg:
            raise CmdError("too many arguments")
        if self.cmdprefix:
            process = self.get_process_from_prefix()
            for thread in process.iter_threads():
                tid = thread.get_tid()
                name = thread.get_name()
                print("%-12d %s" % (tid, name))
        else:
            system = self.debug.system
            pid_list = self.debug.get_debugee_pids()
            if pid_list:
                print("Thread ID    Thread name")
                for pid in pid_list:
                    process = system.get_process(pid)
                    for thread in process.iter_threads():
                        tid = thread.get_tid()
                        name = thread.get_name()
                        print("%-12d %s" % (tid, name))

    do_tl = do_threadlist

    def do_kill(self, arg):
        """
        [~process] kill - kill a process
        [~thread] kill - kill a thread
        kill - kill the current process
        kill * - kill all debugged processes
        kill <processes and/or threads...> - kill the given processes and threads
        """
        if arg:
            if arg == '*':
                target_pids = self.debug.get_debugee_pids()
                target_tids = list()
            else:
                target_pids = set()
                target_tids = set()
                if self.cmdprefix:
                    pid, tid = self.get_process_and_thread_ids_from_prefix()
                    if tid is None:
                        target_tids.add(tid)
                    else:
                        target_pids.add(pid)
                for token in self.split_tokens(arg):
                    try:
                        pid = self.input_process(token)
                        target_pids.add(pid)
                    except CmdError:
                        try:
                            tid = self.input_process(token)
                            target_pids.add(pid)
                        except CmdError:
                            msg = "unknown process or thread (%s)" % token
                            raise CmdError(msg)
                target_pids = list(target_pids)
                target_tids = list(target_tids)
                target_pids.sort()
                target_tids.sort()
            msg = "You are about to kill %d processes and %d threads."
            msg = msg % (len(target_pids), len(target_tids))
            if self.ask_user(msg):
                for pid in target_pids:
                    self.kill_process(pid)
                for tid in target_tids:
                    self.kill_thread(tid)
        else:
            if self.cmdprefix:
                pid, tid = self.get_process_and_thread_ids_from_prefix()
                if tid is None:
                    if self.lastEvent is not None and pid == self.lastEvent.get_pid():
                        msg = "You are about to kill the current process."
                    else:
                        msg = "You are about to kill process %d." % pid
                    if self.ask_user(msg):
                        self.kill_process(pid)
                else:
                    if self.lastEvent is not None and tid == self.lastEvent.get_tid():
                        msg = "You are about to kill the current thread."
                    else:
                        msg = "You are about to kill thread %d." % tid
                    if self.ask_user(msg):
                        self.kill_thread(tid)
            else:
                if self.lastEvent is None:
                    raise CmdError("no current process set")
                pid = self.lastEvent.get_pid()
                if self.ask_user("You are about to kill the current process."):
                    self.kill_process(pid)

    # TODO: create hidden threads using undocumented API calls.
    def do_modload(self, arg):
        """
        [~process] modload <filename.dll> - load a DLL module
        """
        filename = self.split_tokens(arg, 1, 1)[0]
        process = self.get_process_from_prefix()
        try:
            process.inject_dll(filename, bWait=False)
        except RuntimeError:
            print("Can't inject module: %r" % filename)

    # TODO: modunload

    def do_stack(self, arg):
        """
        [~thread] k - show the stack trace
        [~thread] stack - show the stack trace
        """
        if arg:  # XXX TODO add depth parameter
            raise CmdError("too many arguments")
        pid, tid = self.get_process_and_thread_ids_from_prefix()
        process = self.get_process(pid)
        thread = process.get_thread(tid)
        try:
            stack_trace = thread.get_stack_trace_with_labels()
            if stack_trace:
                print(CrashDump.dump_stack_trace_with_labels(stack_trace),)
            else:
                print("No stack trace available for thread (%d)" % tid)
        except WindowsError:
            print("Can't get stack trace for thread (%d)" % tid)

    do_k = do_stack

    def do_break(self, arg):
        """
        break - force a debug break in all debugees
        break <process> [process...] - force a debug break
        """
        debug = self.debug
        system = debug.system
        targets = self.input_process_list(self.split_tokens(arg))
        if not targets:
            targets = debug.get_debugee_pids()
            targets.sort()
        if self.lastEvent:
            current = self.lastEvent.get_pid()
        else:
            current = None
        for pid in targets:
            if pid != current and debug.is_debugee(pid):
                process = system.get_process(pid)
                try:
                    process.debug_break()
                except WindowsError:
                    print("Can't force a debug break on process (%d)")

    def do_step(self, arg):
        """
        p - step on the current assembly instruction
        next - step on the current assembly instruction
        step - step on the current assembly instruction
        """
        if self.cmdprefix:
            raise CmdError("prefix not allowed")
        if self.lastEvent is None:
            raise CmdError("no current process set")
        if arg:  # XXX this check is to be removed
            raise CmdError("too many arguments")
        pid = self.lastEvent.get_pid()
        thread = self.lastEvent.get_thread()
        pc = thread.get_pc()
        code = thread.disassemble(pc, 16)[0]
        size = code[1]
        opcode = code[2].lower()
        if ' ' in opcode:
            opcode = opcode[: opcode.find(' ') ]
        if opcode in self.jump_instructions or opcode in ('int', 'ret', 'retn'):
            return self.do_trace(arg)
        address = pc + size
# #        print(hex(pc), hex(address), size   # XXX DEBUG
        self.debug.stalk_at(pid, address)
        return True

    do_p = do_step
    do_next = do_step

    def do_trace(self, arg):
        """
        t - trace at the current assembly instruction
        trace - trace at the current assembly instruction
        """
        if arg:  # XXX this check is to be removed
            raise CmdError("too many arguments")
        if self.lastEvent is None:
            raise CmdError("no current thread set")
        self.lastEvent.get_thread().set_tf()
        return True

    do_t = do_trace

    def do_bp(self, arg):
        """
        [~process] bp <address> - set a code breakpoint
        """
        pid = self.get_process_id_from_prefix()
        if not self.debug.is_debugee(pid):
            raise CmdError("target process is not being debugged")
        process = self.get_process(pid)
        token_list = self.split_tokens(arg, 1, 1)
        try:
            address = self.input_address(token_list[0], pid)
            deferred = False
        except Exception:
            address = token_list[0]
            deferred = True
        if not address:
            address = token_list[0]
            deferred = True
        self.debug.break_at(pid, address)
        if deferred:
            print("Deferred breakpoint set at %s" % address)
        else:
            print("Breakpoint set at %s" % address)

    def do_ba(self, arg):
        """
        [~thread] ba <a|w|e> <1|2|4|8> <address> - set hardware breakpoint
        """
        debug = self.debug
        thread = self.get_thread_from_prefix()
        pid = thread.get_pid()
        tid = thread.get_tid()
        if not debug.is_debugee(pid):
            raise CmdError("target thread is not being debugged")
        token_list = self.split_tokens(arg, 3, 3)
        access = token_list[0].lower()
        size = token_list[1]
        address = token_list[2]
        if   access == 'a':
            access = debug.BP_BREAK_ON_ACCESS
        elif access == 'w':
            access = debug.BP_BREAK_ON_WRITE
        elif access == 'e':
            access = debug.BP_BREAK_ON_EXECUTION
        else:
            raise CmdError("bad access type: %s" % token_list[0])
        if   size == '1':
            size = debug.BP_WATCH_BYTE
        elif size == '2':
            size = debug.BP_WATCH_WORD
        elif size == '4':
            size = debug.BP_WATCH_DWORD
        elif size == '8':
            size = debug.BP_WATCH_QWORD
        else:
            raise CmdError("bad breakpoint size: %s" % size)
        thread = self.get_thread_from_prefix()
        tid = thread.get_tid()
        pid = thread.get_pid()
        if not debug.is_debugee(pid):
            raise CmdError("target process is not being debugged")
        address = self.input_address(address, pid)
        if debug.has_hardware_breakpoint(tid, address):
            debug.erase_hardware_breakpoint(tid, address)
        debug.define_hardware_breakpoint(tid, address, access, size)
        debug.enable_hardware_breakpoint(tid, address)

    def do_bm(self, arg):
        """
        [~process] bm <address-address> - set memory breakpoint
        """
        pid = self.get_process_id_from_prefix()
        if not self.debug.is_debugee(pid):
            raise CmdError("target process is not being debugged")
        process = self.get_process(pid)
        token_list = self.split_tokens(arg, 1, 2)
        address, size = self.input_address_range(token_list[0], pid)
        self.debug.watch_buffer(pid, address, size)

    def do_bl(self, arg):
        """
        bl - list the breakpoints for the current process
        bl * - list the breakpoints for all processes
        [~process] bl - list the breakpoints for the given process
        bl <process> [process...] - list the breakpoints for each given process
        """
        debug = self.debug
        if arg == '*':
            if self.cmdprefix:
                raise CmdError("prefix not supported")
            breakpoints = debug.get_debugee_pids()
        else:
            targets = self.input_process_list(self.split_tokens(arg))
            if self.cmdprefix:
                targets.insert(0, self.input_process(self.cmdprefix))
            if not targets:
                if self.lastEvent is None:
                    raise CmdError("no current process is set")
                targets = [ self.lastEvent.get_pid() ]
        for pid in targets:
            bplist = debug.get_process_code_breakpoints(pid)
            printed_process_banner = False
            if bplist:
                if not printed_process_banner:
                    print("Process %d:" % pid)
                    printed_process_banner = True
                for bp in bplist:
                    address = repr(bp)[1:-1].replace('remote address ', '')
                    print("  %s" % address)
            dbplist = debug.get_process_deferred_code_breakpoints(pid)
            if dbplist:
                if not printed_process_banner:
                    print("Process %d:" % pid)
                    printed_process_banner = True
                for (label, action, oneshot) in dbplist:
                    if oneshot:
                        address = "  Deferred unconditional one-shot" \
                              " code breakpoint at %s"
                    else:
                        address = "  Deferred unconditional" \
                              " code breakpoint at %s"
                    address = address % label
                    print("  %s" % address)
            bplist = debug.get_process_page_breakpoints(pid)
            if bplist:
                if not printed_process_banner:
                    print("Process %d:" % pid)
                    printed_process_banner = True
                for bp in bplist:
                    address = repr(bp)[1:-1].replace('remote address ', '')
                    print("  %s" % address)
            for tid in debug.system.get_process(pid).iter_thread_ids():
                bplist = debug.get_thread_hardware_breakpoints(tid)
                if bplist:
                    print("Thread %d:" % tid)
                    for bp in bplist:
                        address = repr(bp)[1:-1].replace('remote address ', '')
                        print("  %s" % address)

    def do_bo(self, arg):
        """
        [~process] bo <address> - make a code breakpoint one-shot
        [~thread] bo <address> - make a hardware breakpoint one-shot
        [~process] bo <address-address> - make a memory breakpoint one-shot
        [~process] bo <address> <size> - make a memory breakpoint one-shot
        """
        token_list = self.split_tokens(arg, 1, 2)
        pid, tid, address, size = self.input_breakpoint(token_list)
        debug = self.debug
        found = False
        if size is None:
            if tid is not None:
                if debug.has_hardware_breakpoint(tid, address):
                    debug.enable_one_shot_hardware_breakpoint(tid, address)
                    found = True
            if pid is not None:
                if debug.has_code_breakpoint(pid, address):
                    debug.enable_one_shot_code_breakpoint(pid, address)
                    found = True
        else:
            if debug.has_page_breakpoint(pid, address):
                debug.enable_one_shot_page_breakpoint(pid, address)
                found = True
        if not found:
            print("Error: breakpoint not found.")

    def do_be(self, arg):
        """
        [~process] be <address> - enable a code breakpoint
        [~thread] be <address> - enable a hardware breakpoint
        [~process] be <address-address> - enable a memory breakpoint
        [~process] be <address> <size> - enable a memory breakpoint
        """
        token_list = self.split_tokens(arg, 1, 2)
        pid, tid, address, size = self.input_breakpoint(token_list)
        debug = self.debug
        found = False
        if size is None:
            if tid is not None:
                if debug.has_hardware_breakpoint(tid, address):
                    debug.enable_hardware_breakpoint(tid, address)
                    found = True
            if pid is not None:
                if debug.has_code_breakpoint(pid, address):
                    debug.enable_code_breakpoint(pid, address)
                    found = True
        else:
            if debug.has_page_breakpoint(pid, address):
                debug.enable_page_breakpoint(pid, address)
                found = True
        if not found:
            print("Error: breakpoint not found.")

    def do_bd(self, arg):
        """
        [~process] bd <address> - disable a code breakpoint
        [~thread] bd <address> - disable a hardware breakpoint
        [~process] bd <address-address> - disable a memory breakpoint
        [~process] bd <address> <size> - disable a memory breakpoint
        """
        token_list = self.split_tokens(arg, 1, 2)
        pid, tid, address, size = self.input_breakpoint(token_list)
        debug = self.debug
        found = False
        if size is None:
            if tid is not None:
                if debug.has_hardware_breakpoint(tid, address):
                    debug.disable_hardware_breakpoint(tid, address)
                    found = True
            if pid is not None:
                if debug.has_code_breakpoint(pid, address):
                    debug.disable_code_breakpoint(pid, address)
                    found = True
        else:
            if debug.has_page_breakpoint(pid, address):
                debug.disable_page_breakpoint(pid, address)
                found = True
        if not found:
            print("Error: breakpoint not found.")

    def do_bc(self, arg):
        """
        [~process] bc <address> - clear a code breakpoint
        [~thread] bc <address> - clear a hardware breakpoint
        [~process] bc <address-address> - clear a memory breakpoint
        [~process] bc <address> <size> - clear a memory breakpoint
        """
        token_list = self.split_tokens(arg, 1, 2)
        pid, tid, address, size = self.input_breakpoint(token_list)
        debug = self.debug
        found = False
        if size is None:
            if tid is not None:
                if debug.has_hardware_breakpoint(tid, address):
                    debug.dont_watch_variable(tid, address)
                    found = True
            if pid is not None:
                if debug.has_code_breakpoint(pid, address):
                    debug.dont_break_at(pid, address)
                    found = True
        else:
            if debug.has_page_breakpoint(pid, address):
                debug.dont_watch_buffer(pid, address, size)
                found = True
        if not found:
            print("Error: breakpoint not found.")

    def do_disassemble(self, arg):
        """
        [~thread] u [register] - show code disassembly
        [~process] u [address] - show code disassembly
        [~thread] disassemble [register] - show code disassembly
        [~process] disassemble [address] - show code disassembly
        """
        if not arg:
            arg = self.default_disasm_target
        token_list = self.split_tokens(arg, 1, 1)
        pid, tid = self.get_process_and_thread_ids_from_prefix()
        process = self.get_process(pid)
        address = self.input_address(token_list[0], pid, tid)
        try:
            code = process.disassemble(address, 15 * 8)[:8]
        except Exception:
            msg = "can't disassemble address %s"
            msg = msg % HexDump.address(address)
            raise CmdError(msg)
        if code:
            label = process.get_label_at_address(address)
            last_code = code[-1]
            next_address = last_code[0] + last_code[1]
            next_address = HexOutput.integer(next_address)
            self.default_disasm_target = next_address
            print("%s:" % label)
# #            print(CrashDump.dump_code(code))
            for line in code:
                print(CrashDump.dump_code_line(line, bShowDump=False))

    do_u = do_disassemble

    def do_search(self, arg):
        """
        [~process] s [address-address] <search string>
        [~process] search [address-address] <search string>
        """
        token_list = self.split_tokens(arg, 1, 3)
        pid, tid = self.get_process_and_thread_ids_from_prefix()
        process = self.get_process(pid)
        if len(token_list) == 1:
            pattern = token_list[0]
            minAddr = None
            maxAddr = None
        else:
            pattern = token_list[-1]
            addr, size = self.input_address_range(token_list[:-1], pid, tid)
            minAddr = addr
            maxAddr = addr + size
        iter = process.search_bytes(pattern)
        if process.get_bits() == 32:
            addr_width = 8
        else:
            addr_width = 16
        # TODO: need a prettier output here!
        for addr in iter:
            print(HexDump.address(addr, addr_width))

    do_s = do_search

    def do_searchhex(self, arg):
        """
        [~process] sh [address-address] <hexadecimal pattern>
        [~process] searchhex [address-address] <hexadecimal pattern>
        """
        token_list = self.split_tokens(arg, 1, 3)
        pid, tid = self.get_process_and_thread_ids_from_prefix()
        process = self.get_process(pid)
        if len(token_list) == 1:
            pattern = token_list[0]
            minAddr = None
            maxAddr = None
        else:
            pattern = token_list[-1]
            addr, size = self.input_address_range(token_list[:-1], pid, tid)
            minAddr = addr
            maxAddr = addr + size
        iter = process.search_hexa(pattern)
        if process.get_bits() == 32:
            addr_width = 8
        else:
            addr_width = 16
        for addr, bytes in iter:
            print(HexDump.hexblock(bytes, addr, addr_width),)

    do_sh = do_searchhex

# #    def do_strings(self, arg):
# #        """
# #        [~process] strings - extract ASCII strings from memory
# #        """
# #        if arg:
# #            raise CmdError("too many arguments")
# #        pid, tid   = self.get_process_and_thread_ids_from_prefix()
# #        process    = self.get_process(pid)
# #        for addr, size, data in process.strings():
# #            print("%s: %r" % (HexDump.address(addr), data)

    def do_d(self, arg):
        """
        [~thread] d <register> - show memory contents
        [~thread] d <register-register> - show memory contents
        [~thread] d <register> <size> - show memory contents
        [~process] d <address> - show memory contents
        [~process] d <address-address> - show memory contents
        [~process] d <address> <size> - show memory contents
        """
        return self.last_display_command(arg)

    def do_db(self, arg):
        """
        [~thread] db <register> - show memory contents as bytes
        [~thread] db <register-register> - show memory contents as bytes
        [~thread] db <register> <size> - show memory contents as bytes
        [~process] db <address> - show memory contents as bytes
        [~process] db <address-address> - show memory contents as bytes
        [~process] db <address> <size> - show memory contents as bytes
        """
        self.print_memory_display(arg, HexDump.hexblock)
        self.last_display_command = self.do_db

    def do_dw(self, arg):
        """
        [~thread] dw <register> - show memory contents as words
        [~thread] dw <register-register> - show memory contents as words
        [~thread] dw <register> <size> - show memory contents as words
        [~process] dw <address> - show memory contents as words
        [~process] dw <address-address> - show memory contents as words
        [~process] dw <address> <size> - show memory contents as words
        """
        self.print_memory_display(arg, HexDump.hexblock_word)
        self.last_display_command = self.do_dw

    def do_dd(self, arg):
        """
        [~thread] dd <register> - show memory contents as dwords
        [~thread] dd <register-register> - show memory contents as dwords
        [~thread] dd <register> <size> - show memory contents as dwords
        [~process] dd <address> - show memory contents as dwords
        [~process] dd <address-address> - show memory contents as dwords
        [~process] dd <address> <size> - show memory contents as dwords
        """
        self.print_memory_display(arg, HexDump.hexblock_dword)
        self.last_display_command = self.do_dd

    def do_dq(self, arg):
        """
        [~thread] dq <register> - show memory contents as qwords
        [~thread] dq <register-register> - show memory contents as qwords
        [~thread] dq <register> <size> - show memory contents as qwords
        [~process] dq <address> - show memory contents as qwords
        [~process] dq <address-address> - show memory contents as qwords
        [~process] dq <address> <size> - show memory contents as qwords
        """
        self.print_memory_display(arg, HexDump.hexblock_qword)
        self.last_display_command = self.do_dq

    # XXX TODO
    # Change the way the default is used with ds and du

    def do_ds(self, arg):
        """
        [~thread] ds <register> - show memory contents as ANSI string
        [~process] ds <address> - show memory contents as ANSI string
        """
        if not arg:
            arg = self.default_display_target
        token_list = self.split_tokens(arg, 1, 1)
        pid, tid, address, size = self.input_display(token_list, 256)
        process = self.get_process(pid)
        data = process.peek_string(address, False, size)
        if data:
            print(repr(data))
        self.last_display_command = self.do_ds

    def do_du(self, arg):
        """
        [~thread] du <register> - show memory contents as Unicode string
        [~process] du <address> - show memory contents as Unicode string
        """
        if not arg:
            arg = self.default_display_target
        token_list = self.split_tokens(arg, 1, 2)
        pid, tid, address, size = self.input_display(token_list, 256)
        process = self.get_process(pid)
        data = process.peek_string(address, True, size)
        if data:
            print(repr(data))
        self.last_display_command = self.do_du

    def do_register(self, arg):
        """
        [~thread] r - print(the value of all registers
        [~thread] r <register> - print(the value of a register
        [~thread] r <register>=<value> - change the value of a register
        [~thread] register - print(the value of all registers
        [~thread] register <register> - print(the value of a register
        [~thread] register <register>=<value> - change the value of a register
        """
        arg = arg.strip()
        if not arg:
            self.print_current_location()
        else:
            equ = arg.find('=')
            if equ >= 0:
                register = arg[:equ].strip()
                value = arg[equ + 1:].strip()
                if not value:
                    value = '0'
                self.change_register(register, value)
            else:
                value = self.input_register(arg)
                if value is None:
                    raise CmdError("unknown register: %s" % arg)
                try:
                    label = None
                    thread = self.get_thread_from_prefix()
                    process = thread.get_process()
                    module = process.get_module_at_address(value)
                    if module:
                        label = module.get_label_at_address(value)
                except RuntimeError:
                    label = None
                reg = arg.upper()
                val = HexDump.address(value)
                if label:
                    print("%s: %s (%s)" % (reg, val, label))
                else:
                    print("%s: %s" % (reg, val))

    do_r = do_register

    def do_eb(self, arg):
        """
        [~process] eb <address> <data> - write the data to the specified address
        """
        # TODO
        # data parameter should be optional, use a child Cmd here
        pid = self.get_process_id_from_prefix()
        token_list = self.split_tokens(arg, 2)
        address = self.input_address(token_list[0], pid)
        data = HexInput.hexadecimal(' '.join(token_list[1:]))
        self.write_memory(address, data, pid)

    # XXX TODO
    # add ew, ed and eq here

    def do_find(self, arg):
        """
        [~process] f <string> - find the string in the process memory
        [~process] find <string> - find the string in the process memory
        """
        if not arg:
            raise CmdError("missing parameter: string")
        process = self.get_process_from_prefix()
        self.find_in_memory(arg, process)

    do_f = do_find

    def do_memory(self, arg):
        """
        [~process] m - show the process memory map
        [~process] memory - show the process memory map
        """
        if arg:  # TODO: take min and max addresses
            raise CmdError("too many arguments")
        process = self.get_process_from_prefix()
        try:
            memoryMap = process.get_memory_map()
            mappedFilenames = process.get_mapped_filenames()
            print('')
            print(CrashDump.dump_memory_map(memoryMap, mappedFilenames))
        except WindowsError:
            msg = "can't get memory information for process (%d)"
            raise CmdError(msg % process.get_pid())

    do_m = do_memory

#------------------------------------------------------------------------------
# Event handling

# TODO
# * add configurable stop/don't stop behavior on events and exceptions

    # Stop for all events, unless stated otherwise.
    def event(self, event):
        self.print_event(event)
        self.prompt_user()

    # Stop for all exceptions, unless stated otherwise.
    def exception(self, event):
        self.print_exception(event)
        self.prompt_user()

    # Stop for breakpoint exceptions.
    def breakpoint(self, event):
        if hasattr(event, 'breakpoint') and event.breakpoint:
            self.print_breakpoint_location(event)
        else:
            self.print_exception(event)
        self.prompt_user()

    # Stop for WOW64 breakpoint exceptions.
    def wow64_breakpoint(self, event):
        self.print_exception(event)
        self.prompt_user()

    # Stop for single step exceptions.
    def single_step(self, event):
        if event.debug.is_tracing(event.get_tid()):
            self.print_breakpoint_location(event)
        else:
            self.print_exception(event)
        self.prompt_user()

    # Don't stop for C++ exceptions.
    def ms_vc_exception(self, event):
        self.print_exception(event)
        event.continueStatus = win32.DBG_CONTINUE

    # Don't stop for process start.
    def create_process(self, event):
        self.print_process_start(event)
        self.print_thread_start(event)
        self.print_module_load(event)

    # Don't stop for process exit.
    def exit_process(self, event):
        self.print_process_end(event)

    # Don't stop for thread creation.
    def create_thread(self, event):
        self.print_thread_start(event)

    # Don't stop for thread exit.
    def exit_thread(self, event):
        self.print_thread_end(event)

    # Don't stop for DLL load.
    def load_dll(self, event):
        self.print_module_load(event)

    # Don't stop for DLL unload.
    def unload_dll(self, event):
        self.print_module_unload(event)

    # Don't stop for debug strings.
    def output_string(self, event):
        self.print_debug_string(event)

#------------------------------------------------------------------------------
# History file

    def load_history(self):
        global readline
        if readline is None:
            try:
                import readline
            except ImportError:
                return
        if self.history_file_full_path is None:
            folder = os.environ.get('USERPROFILE', '')
            if not folder:
                folder = os.environ.get('HOME', '')
            if not folder:
                folder = os.path.split(sys.argv[0])[1]
            if not folder:
                folder = os.path.curdir
            self.history_file_full_path = os.path.join(folder,
                                                       self.history_file)
        try:
            if os.path.exists(self.history_file_full_path):
                readline.read_history_file(self.history_file_full_path)
        except IOError:
            e = sys.exc_info()[1]
            warnings.warn("Cannot load history file, reason: %s" % str(e))

    def save_history(self):
        if self.history_file_full_path is not None:
            global readline
            if readline is None:
                try:
                    import readline
                except ImportError:
                    return
            try:
                readline.write_history_file(self.history_file_full_path)
            except IOError:
                e = sys.exc_info()[1]
                warnings.warn("Cannot save history file, reason: %s" % str(e))

#------------------------------------------------------------------------------
# Main loop

    # Debugging loop.
    def loop(self):
        self.debuggerExit = False
        debug = self.debug

        # Stop on the initial event, if any.
        if self.lastEvent is not None:
            self.cmdqueue.append('r')
            self.prompt_user()

        # Loop until the debugger is told to quit.
        while not self.debuggerExit:

            try:

                # If for some reason the last event wasn't continued,
                # continue it here. This won't be done more than once
                # for a given Event instance, though.
                try:
                    debug.cont()
                # On error, show the command prompt.
                except Exception:
                    traceback.print_exc()
                    self.prompt_user()

                # While debugees are attached, handle debug events.
                # Some debug events may cause the command prompt to be shown.
                if self.debug.get_debugee_count() > 0:
                    try:

                        # Get the next debug event.
                        debug.wait()

                        # Dispatch the debug event.
                        try:
                            debug.dispatch()

                        # Continue the debug event.
                        finally:
                            debug.cont()

                    # On error, show the command prompt.
                    except Exception:
                        traceback.print_exc()
                        self.prompt_user()

                # While no debugees are attached, show the command prompt.
                else:
                    self.prompt_user()

            # When the user presses Ctrl-C send a debug break to all debugees.
            except KeyboardInterrupt:
                success = False
                try:
                    print("*** User requested debug break")
                    system = debug.system
                    for pid in debug.get_debugee_pids():
                        try:
                            system.get_process(pid).debug_break()
                            success = True
                        except:
                            traceback.print_exc()
                except:
                    traceback.print_exc()
                if not success:
                    raise  # This should never happen!
