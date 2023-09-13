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
Crash dump support.

@group Crash reporting:
    Crash, CrashDictionary

@group Warnings:
    CrashWarning

@group Deprecated classes:
    CrashContainer, CrashTable, CrashTableMSSQL,
    VolatileCrashContainer, DummyCrashContainer
"""

__revision__ = "$Id$"

__all__ = [

    # Object that represents a crash in the debugee.
    'Crash',

    # Crash storage.
    'CrashDictionary',

    # Warnings.
    'CrashWarning',

    # Backwards compatibility with WinAppDbg 1.4 and before.
    'CrashContainer',
    'CrashTable',
    'CrashTableMSSQL',
    'VolatileCrashContainer',
    'DummyCrashContainer',
]

from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations

import sys
import os
import time
import zlib
import warnings

# lazy imports
sql = None
anydbm = None

#==============================================================================

# Secure alternative to pickle, use it if present.
try:
    import cerealizer
    pickle = cerealizer

    # There is no optimization function for cerealized objects.
    def optimize(picklestring):
        return picklestring

    # There is no HIGHEST_PROTOCOL in cerealizer.
    HIGHEST_PROTOCOL = 0

    # Note: it's important NOT to provide backwards compatibility, otherwise
    # it'd be just the same as not having this!
    #
    # To disable this security upgrade simply uncomment the following line:
    #
    # raise ImportError("Fallback to pickle for backwards compatibility")

# If cerealizer is not present fallback to the insecure pickle module.
except ImportError:

    # Faster implementation of the pickle module as a C extension.
    try:
        import cPickle as pickle

    # If all fails fallback to the classic pickle module.
    except ImportError:
        import pickle

    # Fetch the highest protocol version.
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

    # Try to use the pickle optimizer if found.
    try:
        from pickletools import optimize
    except ImportError:
        def optimize(picklestring):
            return picklestring

class Marshaller (StaticClass):
    """
    Custom pickler for L{Crash} objects. Optimizes the pickled data when using
    the standard C{pickle} (or C{cPickle}) module. The pickled data is then
    compressed using zlib.
    """

    @staticmethod
    def dumps(obj, protocol=HIGHEST_PROTOCOL):
        return zlib.compress(optimize(pickle.dumps(obj)), 9)

    @staticmethod
    def loads(data):
        return pickle.loads(zlib.decompress(data))

#==============================================================================

class CrashWarning (Warning):
    """
    An error occurred while gathering crash data.
    Some data may be incomplete or missing.
    """

#==============================================================================

# Crash object. Must be serializable.
class Crash (object):
    """
    Represents a crash, bug, or another interesting event in the debugee.

    @group Basic information:
        timeStamp, signature, eventCode, eventName, pid, tid, arch, os, bits,
        registers, labelPC, pc, sp, fp

    @group Optional information:
        debugString,
        modFileName,
        lpBaseOfDll,
        exceptionCode,
        exceptionName,
        exceptionDescription,
        exceptionAddress,
        exceptionLabel,
        firstChance,
        faultType,
        faultAddress,
        faultLabel,
        isOurBreakpoint,
        isSystemBreakpoint,
        stackTrace,
        stackTracePC,
        stackTraceLabels,
        stackTracePretty

    @group Extra information:
        commandLine,
        environment,
        environmentData,
        registersPeek,
        stackRange,
        stackFrame,
        stackPeek,
        faultCode,
        faultMem,
        faultPeek,
        faultDisasm,
        memoryMap

    @group Report:
        briefReport, fullReport, notesReport, environmentReport, isExploitable

    @group Notes:
        addNote, getNotes, iterNotes, hasNotes, clearNotes, notes

    @group Miscellaneous:
        fetch_extra_data

    @type timeStamp: float
    @ivar timeStamp: Timestamp as returned by time.time().

    @type signature: object
    @ivar signature: Approximately unique signature for the Crash object.

        This signature can be used as an heuristic to determine if two crashes
        were caused by the same software error. Ideally it should be treated as
        as opaque serializable object that can be tested for equality.

    @type notes: list( str )
    @ivar notes: List of strings, each string is a note.

    @type eventCode: int
    @ivar eventCode: Event code as defined by the Win32 API.

    @type eventName: str
    @ivar eventName: Event code user-friendly name.

    @type pid: int
    @ivar pid: Process global ID.

    @type tid: int
    @ivar tid: Thread global ID.

    @type arch: str
    @ivar arch: Processor architecture.

    @type os: str
    @ivar os: Operating system version.

        May indicate a 64 bit version even if L{arch} and L{bits} indicate 32
        bits. This means the crash occurred inside a WOW64 process.

    @type bits: int
    @ivar bits: C{32} or C{64} bits.

    @type commandLine: None or str
    @ivar commandLine: Command line for the target process.

        C{None} if unapplicable or unable to retrieve.

    @type environmentData: None or list of str
    @ivar environmentData: Environment data for the target process.

        C{None} if unapplicable or unable to retrieve.

    @type environment: None or dict( str S{->} str )
    @ivar environment: Environment variables for the target process.

        C{None} if unapplicable or unable to retrieve.

    @type registers: dict( str S{->} int )
    @ivar registers: Dictionary mapping register names to their values.

    @type registersPeek: None or dict( str S{->} str )
    @ivar registersPeek: Dictionary mapping register names to the data they point to.

        C{None} if unapplicable or unable to retrieve.

    @type labelPC: None or str
    @ivar labelPC: Label pointing to the program counter.

        C{None} or invalid if unapplicable or unable to retrieve.

    @type debugString: None or str
    @ivar debugString: Debug string sent by the debugee.

        C{None} if unapplicable or unable to retrieve.

    @type exceptionCode: None or int
    @ivar exceptionCode: Exception code as defined by the Win32 API.

        C{None} if unapplicable or unable to retrieve.

    @type exceptionName: None or str
    @ivar exceptionName: Exception code user-friendly name.

        C{None} if unapplicable or unable to retrieve.

    @type exceptionDescription: None or str
    @ivar exceptionDescription: Exception description.

        C{None} if unapplicable or unable to retrieve.

    @type exceptionAddress: None or int
    @ivar exceptionAddress: Memory address where the exception occured.

        C{None} if unapplicable or unable to retrieve.

    @type exceptionLabel: None or str
    @ivar exceptionLabel: Label pointing to the exception address.

        C{None} or invalid if unapplicable or unable to retrieve.

    @type faultType: None or int
    @ivar faultType: Access violation type.
        Only applicable to memory faults.
        Should be one of the following constants:

         - L{win32.ACCESS_VIOLATION_TYPE_READ}
         - L{win32.ACCESS_VIOLATION_TYPE_WRITE}
         - L{win32.ACCESS_VIOLATION_TYPE_DEP}

        C{None} if unapplicable or unable to retrieve.

    @type faultAddress: None or int
    @ivar faultAddress: Access violation memory address.
        Only applicable to memory faults.

        C{None} if unapplicable or unable to retrieve.

    @type faultLabel: None or str
    @ivar faultLabel: Label pointing to the access violation memory address.
        Only applicable to memory faults.

        C{None} if unapplicable or unable to retrieve.

    @type firstChance: None or bool
    @ivar firstChance:
        C{True} for first chance exceptions, C{False} for second chance.

        C{None} if unapplicable or unable to retrieve.

    @type isOurBreakpoint: bool
    @ivar isOurBreakpoint:
        C{True} for breakpoints defined by the L{Debug} class,
        C{False} otherwise.

        C{None} if unapplicable.

    @type isSystemBreakpoint: bool
    @ivar isSystemBreakpoint:
        C{True} for known system-defined breakpoints,
        C{False} otherwise.

        C{None} if unapplicable.

    @type modFileName: None or str
    @ivar modFileName: File name of module where the program counter points to.

        C{None} or invalid if unapplicable or unable to retrieve.

    @type lpBaseOfDll: None or int
    @ivar lpBaseOfDll: Base of module where the program counter points to.

        C{None} if unapplicable or unable to retrieve.

    @type stackTrace: None or tuple of tuple( int, int, str )
    @ivar stackTrace:
        Stack trace of the current thread as a tuple of
        ( frame pointer, return address, module filename ).

        C{None} or empty if unapplicable or unable to retrieve.

    @type stackTracePretty: None or tuple of tuple( int, str )
    @ivar stackTracePretty:
        Stack trace of the current thread as a tuple of
        ( frame pointer, return location ).

        C{None} or empty if unapplicable or unable to retrieve.

    @type stackTracePC: None or tuple( int... )
    @ivar stackTracePC: Tuple of return addresses in the stack trace.

        C{None} or empty if unapplicable or unable to retrieve.

    @type stackTraceLabels: None or tuple( str... )
    @ivar stackTraceLabels:
        Tuple of labels pointing to the return addresses in the stack trace.

        C{None} or empty if unapplicable or unable to retrieve.

    @type stackRange: tuple( int, int )
    @ivar stackRange:
        Stack beginning and end pointers, in memory addresses order.

        C{None} if unapplicable or unable to retrieve.

    @type stackFrame: None or str
    @ivar stackFrame: Data pointed to by the stack pointer.

        C{None} or empty if unapplicable or unable to retrieve.

    @type stackPeek: None or dict( int S{->} str )
    @ivar stackPeek: Dictionary mapping stack offsets to the data they point to.

        C{None} or empty if unapplicable or unable to retrieve.

    @type faultCode: None or str
    @ivar faultCode: Data pointed to by the program counter.

        C{None} or empty if unapplicable or unable to retrieve.

    @type faultMem: None or str
    @ivar faultMem: Data pointed to by the exception address.

        C{None} or empty if unapplicable or unable to retrieve.

    @type faultPeek: None or dict( intS{->} str )
    @ivar faultPeek: Dictionary mapping guessed pointers at L{faultMem} to the data they point to.

        C{None} or empty if unapplicable or unable to retrieve.

    @type faultDisasm: None or tuple of tuple( long, int, str, str )
    @ivar faultDisasm: Dissassembly around the program counter.

        C{None} or empty if unapplicable or unable to retrieve.

    @type memoryMap: None or list of L{win32.MemoryBasicInformation} objects.
    @ivar memoryMap: Memory snapshot of the program. May contain the actual
        data from the entire process memory if requested.
        See L{fetch_extra_data} for more details.

        C{None} or empty if unapplicable or unable to retrieve.

    @type _rowid: int
    @ivar _rowid: Row ID in the database. Internally used by the DAO layer.
        Only present in crash dumps retrieved from the database. Do not rely
        on this property to be present in future versions of WinAppDbg.
    """

    def __init__(self, event):
        """
        @type  event: L{Event}
        @param event: Event object for crash.
        """

        # First of all, take the timestamp.
        self.timeStamp          = time.time()

        # Notes are initially empty.
        self.notes              = list()

        # Get the process and thread, but dont't store them in the DB.
        process = event.get_process()
        thread  = event.get_thread()

        # Determine the architecture.
        self.os                 = System.os
        self.arch               = process.get_arch()
        self.bits               = process.get_bits()

        # The following properties are always retrieved for all events.
        self.eventCode          = event.get_event_code()
        self.eventName          = event.get_event_name()
        self.pid                = event.get_pid()
        self.tid                = event.get_tid()
        self.registers          = dict(thread.get_context())
        self.labelPC            = process.get_label_at_address(self.pc)

        # The following properties are only retrieved for some events.
        self.commandLine        = None
        self.environment        = None
        self.environmentData    = None
        self.registersPeek      = None
        self.debugString        = None
        self.modFileName        = None
        self.lpBaseOfDll        = None
        self.exceptionCode      = None
        self.exceptionName      = None
        self.exceptionDescription = None
        self.exceptionAddress   = None
        self.exceptionLabel     = None
        self.firstChance        = None
        self.faultType          = None
        self.faultAddress       = None
        self.faultLabel         = None
        self.isOurBreakpoint    = None
        self.isSystemBreakpoint = None
        self.stackTrace         = None
        self.stackTracePC       = None
        self.stackTraceLabels   = None
        self.stackTracePretty   = None
        self.stackRange         = None
        self.stackFrame         = None
        self.stackPeek          = None
        self.faultCode          = None
        self.faultMem           = None
        self.faultPeek          = None
        self.faultDisasm        = None
        self.memoryMap          = None

        # Get information for debug string events.
        if self.eventCode == win32.OUTPUT_DEBUG_STRING_EVENT:
            self.debugString = event.get_debug_string()

        # Get information for module load and unload events.
        # For create and exit process events, get the information
        # for the main module.
        elif self.eventCode in (win32.CREATE_PROCESS_DEBUG_EVENT,
                                win32.EXIT_PROCESS_DEBUG_EVENT,
                                win32.LOAD_DLL_DEBUG_EVENT,
                                win32.UNLOAD_DLL_DEBUG_EVENT):
            aModule = event.get_module()
            self.modFileName = event.get_filename()
            if not self.modFileName:
                self.modFileName = aModule.get_filename()
            self.lpBaseOfDll = event.get_module_base()
            if not self.lpBaseOfDll:
                self.lpBaseOfDll = aModule.get_base()

        # Get some information for exception events.
        # To get the remaining information call fetch_extra_data().
        elif self.eventCode == win32.EXCEPTION_DEBUG_EVENT:

            # Exception information.
            self.exceptionCode          = event.get_exception_code()
            self.exceptionName          = event.get_exception_name()
            self.exceptionDescription   = event.get_exception_description()
            self.exceptionAddress       = event.get_exception_address()
            self.firstChance            = event.is_first_chance()
            self.exceptionLabel         = process.get_label_at_address(
                                                         self.exceptionAddress)
            if self.exceptionCode in (win32.EXCEPTION_ACCESS_VIOLATION,
                                      win32.EXCEPTION_GUARD_PAGE,
                                      win32.EXCEPTION_IN_PAGE_ERROR):
                self.faultType    = event.get_fault_type()
                self.faultAddress = event.get_fault_address()
                self.faultLabel   = process.get_label_at_address(
                                                            self.faultAddress)
            elif self.exceptionCode in (win32.EXCEPTION_BREAKPOINT,
                                        win32.EXCEPTION_SINGLE_STEP):
                self.isOurBreakpoint = hasattr(event, 'breakpoint') \
                                       and event.breakpoint
                self.isSystemBreakpoint = \
                    process.is_system_defined_breakpoint(self.exceptionAddress)

            # Stack trace.
            try:
                self.stackTracePretty = thread.get_stack_trace_with_labels()
            except Exception:
                e = sys.exc_info()[1]
                warnings.warn(
                    "Cannot get stack trace with labels, reason: %s" % str(e),
                    CrashWarning)
            try:
                self.stackTrace     = thread.get_stack_trace()
                stackTracePC        = [ ra for (_,ra,_) in self.stackTrace ]
                self.stackTracePC   = tuple(stackTracePC)
                stackTraceLabels    = [ process.get_label_at_address(ra) \
                                             for ra in self.stackTracePC ]
                self.stackTraceLabels = tuple(stackTraceLabels)
            except Exception:
                e = sys.exc_info()[1]
                warnings.warn("Cannot get stack trace, reason: %s" % str(e),
                              CrashWarning)

    def fetch_extra_data(self, event, takeMemorySnapshot = 0):
        """
        Fetch extra data from the L{Event} object.

        @note: Since this method may take a little longer to run, it's best to
            call it only after you've determined the crash is interesting and
            you want to save it.

        @type  event: L{Event}
        @param event: Event object for crash.

        @type  takeMemorySnapshot: int
        @param takeMemorySnapshot:
            Memory snapshot behavior:
             - C{0} to take no memory information (default).
             - C{1} to take only the memory map.
               See L{Process.get_memory_map}.
             - C{2} to take a full memory snapshot.
               See L{Process.take_memory_snapshot}.
             - C{3} to take a live memory snapshot.
               See L{Process.generate_memory_snapshot}.
        """

        # Get the process and thread, we'll use them below.
        process = event.get_process()
        thread  = event.get_thread()

        # Get the command line for the target process.
        try:
            self.commandLine = process.get_command_line()
        except Exception:
            e = sys.exc_info()[1]
            warnings.warn("Cannot get command line, reason: %s" % str(e),
                          CrashWarning)

        # Get the environment variables for the target process.
        try:
            self.environmentData = process.get_environment_data()
            self.environment     = process.parse_environment_data(
                                                        self.environmentData)
        except Exception:
            e = sys.exc_info()[1]
            warnings.warn("Cannot get environment, reason: %s" % str(e),
                          CrashWarning)

        # Data pointed to by registers.
        self.registersPeek = thread.peek_pointers_in_registers()

        # Module where execution is taking place.
        aModule = process.get_module_at_address(self.pc)
        if aModule is not None:
            self.modFileName = aModule.get_filename()
            self.lpBaseOfDll = aModule.get_base()

        # Contents of the stack frame.
        try:
            self.stackRange = thread.get_stack_range()
        except Exception:
            e = sys.exc_info()[1]
            warnings.warn("Cannot get stack range, reason: %s" % str(e),
                          CrashWarning)
        try:
            self.stackFrame = thread.get_stack_frame()
            stackFrame = self.stackFrame
        except Exception:
            self.stackFrame = thread.peek_stack_data()
            stackFrame = self.stackFrame[:64]
        if stackFrame:
            self.stackPeek = process.peek_pointers_in_data(stackFrame)

        # Code being executed.
        self.faultCode   = thread.peek_code_bytes()
        try:
            self.faultDisasm = thread.disassemble_around_pc(32)
        except Exception:
            e = sys.exc_info()[1]
            warnings.warn("Cannot disassemble, reason: %s" % str(e),
                          CrashWarning)

        # For memory related exceptions, get the memory contents
        # of the location that caused the exception to be raised.
        if self.eventCode == win32.EXCEPTION_DEBUG_EVENT:
            if self.pc != self.exceptionAddress and self.exceptionCode in (
                        win32.EXCEPTION_ACCESS_VIOLATION,
                        win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED,
                        win32.EXCEPTION_DATATYPE_MISALIGNMENT,
                        win32.EXCEPTION_IN_PAGE_ERROR,
                        win32.EXCEPTION_STACK_OVERFLOW,
                        win32.EXCEPTION_GUARD_PAGE,
                        ):
                self.faultMem = process.peek(self.exceptionAddress, 64)
                if self.faultMem:
                    self.faultPeek = process.peek_pointers_in_data(
                                                                 self.faultMem)

        # TODO: maybe add names and versions of DLLs and EXE?

        # Take a snapshot of the process memory. Additionally get the
        # memory contents if requested.
        if takeMemorySnapshot == 1:
            self.memoryMap = process.get_memory_map()
            mappedFilenames = process.get_mapped_filenames(self.memoryMap)
            for mbi in self.memoryMap:
                mbi.filename = mappedFilenames.get(mbi.BaseAddress, None)
                mbi.content  = None
        elif takeMemorySnapshot == 2:
            self.memoryMap = process.take_memory_snapshot()
        elif takeMemorySnapshot == 3:
            self.memoryMap = process.generate_memory_snapshot()

    @property
    def pc(self):
        """
        Value of the program counter register.

        @rtype:  int
        """
        try:
            return self.registers['Eip']        # i386
        except KeyError:
            return self.registers['Rip']        # amd64

    @property
    def sp(self):
        """
        Value of the stack pointer register.

        @rtype:  int
        """
        try:
            return self.registers['Esp']        # i386
        except KeyError:
            return self.registers['Rsp']        # amd64

    @property
    def fp(self):
        """
        Value of the frame pointer register.

        @rtype:  int
        """
        try:
            return self.registers['Ebp']        # i386
        except KeyError:
            return self.registers['Rbp']        # amd64

    def __str__(self):
        return self.fullReport()

    def key(self):
        """
        Alias of L{signature}. Deprecated since WinAppDbg 1.5.
        """
        warnings.warn("Crash.key() method was deprecated in WinAppDbg 1.5",
                      DeprecationWarning)
        return self.signature

    @property
    def signature(self):
        if self.labelPC:
            pc = self.labelPC
        else:
            pc = self.pc
        if self.stackTraceLabels:
            trace = self.stackTraceLabels
        else:
            trace = self.stackTracePC
        return  (
                self.arch,
                self.eventCode,
                self.exceptionCode,
                pc,
                trace,
                self.debugString,
                )
        # TODO
        # add the name and version of the binary where the crash happened?

    def isExploitable(self):
        """
        Guess how likely is it that the bug causing the crash can be leveraged
        into an exploitable vulnerability.

        @note: Don't take this as an equivalent of a real exploitability
            analysis, that can only be done by a human being! This is only
            a guideline, useful for example to sort crashes - placing the most
            interesting ones at the top.

        @see: The heuristics are similar to those of the B{!exploitable}
            extension for I{WinDBG}, which can be downloaded from here:

            U{http://www.codeplex.com/msecdbg}

        @rtype: tuple( str, str, str )
        @return: The first element of the tuple is the result of the analysis,
            being one of the following:

             - Not an exception
             - Not exploitable
             - Not likely exploitable
             - Unknown
             - Probably exploitable
             - Exploitable

            The second element of the tuple is a code to identify the matched
            heuristic rule.

            The third element of the tuple is a description string of the
            reason behind the result.
        """

        # Terminal rules

        if self.eventCode != win32.EXCEPTION_DEBUG_EVENT:
            return ("Not an exception", "NotAnException", "The event is not an exception.")

        if self.stackRange and self.pc is not None and self.stackRange[0] <= self.pc < self.stackRange[1]:
            return ("Exploitable", "StackCodeExecution", "Code execution from the stack is considered exploitable.")

        # This rule is NOT from !exploitable
        if self.stackRange and self.sp is not None and not (self.stackRange[0] <= self.sp < self.stackRange[1]):
            return ("Exploitable", "StackPointerCorruption", "Stack pointer corruption is considered exploitable.")

        if self.exceptionCode == win32.EXCEPTION_ILLEGAL_INSTRUCTION:
            return ("Exploitable", "IllegalInstruction", "An illegal instruction exception indicates that the attacker controls execution flow.")

        if self.exceptionCode == win32.EXCEPTION_PRIV_INSTRUCTION:
            return ("Exploitable", "PrivilegedInstruction", "A privileged instruction exception indicates that the attacker controls execution flow.")

        if self.exceptionCode == win32.EXCEPTION_GUARD_PAGE:
            return ("Exploitable", "GuardPage", "A guard page violation indicates a stack overflow has occured, and the stack of another thread was reached (possibly the overflow length is not controlled by the attacker).")

        if self.exceptionCode == win32.STATUS_STACK_BUFFER_OVERRUN:
            return ("Exploitable", "GSViolation", "An overrun of a protected stack buffer has been detected. This is considered exploitable, and must be fixed.")

        if self.exceptionCode == win32.STATUS_HEAP_CORRUPTION:
            return ("Exploitable", "HeapCorruption", "Heap Corruption has been detected. This is considered exploitable, and must be fixed.")

        if self.exceptionCode == win32.EXCEPTION_ACCESS_VIOLATION:
            nearNull      = self.faultAddress is None or MemoryAddresses.align_address_to_page_start(self.faultAddress) == 0
            controlFlow   = self.__is_control_flow()
            blockDataMove = self.__is_block_data_move()
            if self.faultType == win32.EXCEPTION_EXECUTE_FAULT:
                if nearNull:
                    return ("Probably exploitable", "DEPViolation", "User mode DEP access violations are probably exploitable if near NULL.")
                else:
                    return ("Exploitable", "DEPViolation", "User mode DEP access violations are exploitable.")
            elif self.faultType == win32.EXCEPTION_WRITE_FAULT:
                if nearNull:
                    return ("Probably exploitable", "WriteAV", "User mode write access violations that are near NULL are probably exploitable.")
                else:
                    return ("Exploitable", "WriteAV", "User mode write access violations that are not near NULL are exploitable.")
            elif self.faultType == win32.EXCEPTION_READ_FAULT:
                if self.faultAddress == self.pc:
                    if nearNull:
                        return ("Probably exploitable", "ReadAVonIP", "Access violations at the instruction pointer are probably exploitable if near NULL.")
                    else:
                        return ("Exploitable", "ReadAVonIP", "Access violations at the instruction pointer are exploitable if not near NULL.")
                if controlFlow:
                    if nearNull:
                        return ("Probably exploitable", "ReadAVonControlFlow", "Access violations near null in control flow instructions are considered probably exploitable.")
                    else:
                        return ("Exploitable", "ReadAVonControlFlow", "Access violations not near null in control flow instructions are considered exploitable.")
                if blockDataMove:
                    return ("Probably exploitable", "ReadAVonBlockMove", "This is a read access violation in a block data move, and is therefore classified as probably exploitable.")

                # Rule: Tainted information used to control branch addresses is considered probably exploitable
                # Rule: Tainted information used to control the target of a later write is probably exploitable

        # Non terminal rules

        # XXX TODO add rule to check if code is in writeable memory (probably exploitable)

        # XXX TODO maybe we should be returning a list of tuples instead?

        result = ("Unknown", "Unknown", "Exploitability unknown.")

        if self.exceptionCode == win32.EXCEPTION_ACCESS_VIOLATION:
            if self.faultType == win32.EXCEPTION_READ_FAULT:
                if nearNull:
                    result = ("Not likely exploitable", "ReadAVNearNull", "This is a user mode read access violation near null, and is probably not exploitable.")

        elif self.exceptionCode == win32.EXCEPTION_INT_DIVIDE_BY_ZERO:
            result = ("Not likely exploitable", "DivideByZero", "This is an integer divide by zero, and is probably not exploitable.")

        elif self.exceptionCode == win32.EXCEPTION_FLT_DIVIDE_BY_ZERO:
            result = ("Not likely exploitable", "DivideByZero", "This is a floating point divide by zero, and is probably not exploitable.")

        elif self.exceptionCode in (win32.EXCEPTION_BREAKPOINT, win32.STATUS_WX86_BREAKPOINT):
            result = ("Unknown", "Breakpoint", "While a breakpoint itself is probably not exploitable, it may also be an indication that an attacker is testing a target. In either case breakpoints should not exist in production code.")

        # Rule: If the stack contains unknown symbols in user mode, call that out
        # Rule: Tainted information used to control the source of a later block move unknown, but called out explicitly
        # Rule: Tainted information used as an argument to a function is an unknown risk, but called out explicitly
        # Rule: Tainted information used to control branch selection is an unknown risk, but called out explicitly

        return result

    def __is_control_flow(self):
        """
        Private method to tell if the instruction pointed to by the program
        counter is a control flow instruction.

        Currently only works for x86 and amd64 architectures.
        """
        jump_instructions = (
            'jmp', 'jecxz', 'jcxz',
            'ja', 'jnbe', 'jae', 'jnb', 'jb', 'jnae', 'jbe', 'jna', 'jc', 'je',
            'jz', 'jnc', 'jne', 'jnz', 'jnp', 'jpo', 'jp', 'jpe', 'jg', 'jnle',
            'jge', 'jnl', 'jl', 'jnge', 'jle', 'jng', 'jno', 'jns', 'jo', 'js'
        )
        call_instructions = ( 'call', 'ret', 'retn' )
        loop_instructions = ( 'loop', 'loopz', 'loopnz', 'loope', 'loopne' )
        control_flow_instructions = call_instructions + loop_instructions + \
                                    jump_instructions
        isControlFlow = False
        instruction = None
        if self.pc is not None and self.faultDisasm:
            for disasm in self.faultDisasm:
                if disasm[0] == self.pc:
                    instruction = disasm[2].lower().strip()
                    break
        if instruction:
            for x in control_flow_instructions:
                if x in instruction:
                    isControlFlow = True
                    break
        return isControlFlow

    def __is_block_data_move(self):
        """
        Private method to tell if the instruction pointed to by the program
        counter is a block data move instruction.

        Currently only works for x86 and amd64 architectures.
        """
        block_data_move_instructions = ('movs', 'stos', 'lods')
        isBlockDataMove = False
        instruction = None
        if self.pc is not None and self.faultDisasm:
            for disasm in self.faultDisasm:
                if disasm[0] == self.pc:
                    instruction = disasm[2].lower().strip()
                    break
        if instruction:
            for x in block_data_move_instructions:
                if x in instruction:
                    isBlockDataMove = True
                    break
        return isBlockDataMove

    def briefReport(self):
        """
        @rtype:  str
        @return: Short description of the event.
        """
        if self.exceptionCode is not None:
            if self.exceptionCode == win32.EXCEPTION_BREAKPOINT:
                if self.isOurBreakpoint:
                    what = "Breakpoint hit"
                elif self.isSystemBreakpoint:
                    what = "System breakpoint hit"
                else:
                    what = "Assertion failed"
            elif self.exceptionDescription:
                what = self.exceptionDescription
            elif self.exceptionName:
                what = self.exceptionName
            else:
                what = "Exception %s" % \
                            HexDump.integer(self.exceptionCode, self.bits)
            if self.firstChance:
                chance = 'first'
            else:
                chance = 'second'
            if self.exceptionLabel:
                where = self.exceptionLabel
            elif self.exceptionAddress:
                where = HexDump.address(self.exceptionAddress, self.bits)
            elif self.labelPC:
                where = self.labelPC
            else:
                where = HexDump.address(self.pc, self.bits)
            msg = "%s (%s chance) at %s" % (what, chance, where)
        elif self.debugString is not None:
            if self.labelPC:
                where = self.labelPC
            else:
                where = HexDump.address(self.pc, self.bits)
            msg = "Debug string from %s: %r" % (where, self.debugString)
        else:
            if self.labelPC:
                where = self.labelPC
            else:
                where = HexDump.address(self.pc, self.bits)
            msg = "%s (%s) at %s" % (
                        self.eventName,
                        HexDump.integer(self.eventCode, self.bits),
                        where
                       )
        return msg

    def fullReport(self, bShowNotes = True):
        """
        @type  bShowNotes: bool
        @param bShowNotes: C{True} to show the user notes, C{False} otherwise.

        @rtype:  str
        @return: Long description of the event.
        """
        msg  = self.briefReport()
        msg += '\n'

        if self.bits == 32:
            width = 16
        else:
            width = 8

        if self.eventCode == win32.EXCEPTION_DEBUG_EVENT:
            (exploitability, expcode, expdescription) = self.isExploitable()
            msg += '\nSecurity risk level: %s\n' % exploitability
            msg += '  %s\n' % expdescription

        if bShowNotes and self.notes:
            msg += '\nNotes:\n'
            msg += self.notesReport()

        if self.commandLine:
            msg += '\nCommand line: %s\n' % self.commandLine

        if self.environment:
            msg += '\nEnvironment:\n'
            msg += self.environmentReport()

        if not self.labelPC:
            base = HexDump.address(self.lpBaseOfDll, self.bits)
            if self.modFileName:
                fn   = PathOperations.pathname_to_filename(self.modFileName)
                msg += '\nRunning in %s (%s)\n' % (fn, base)
            else:
                msg += '\nRunning in module at %s\n' % base

        if self.registers:
            msg += '\nRegisters:\n'
            msg += CrashDump.dump_registers(self.registers)
            if self.registersPeek:
                msg += '\n'
                msg += CrashDump.dump_registers_peek(self.registers,
                                                     self.registersPeek,
                                                     width = width)

        if self.faultDisasm:
            msg += '\nCode disassembly:\n'
            msg += CrashDump.dump_code(self.faultDisasm, self.pc,
                                       bits = self.bits)

        if self.stackTrace:
            msg += '\nStack trace:\n'
            if self.stackTracePretty:
                msg += CrashDump.dump_stack_trace_with_labels(
                                        self.stackTracePretty,
                                        bits = self.bits)
            else:
                msg += CrashDump.dump_stack_trace(self.stackTrace,
                                                  bits = self.bits)

        if self.stackFrame:
            if self.stackPeek:
                msg += '\nStack pointers:\n'
                msg += CrashDump.dump_stack_peek(self.stackPeek, width = width)
            msg += '\nStack dump:\n'
            msg += HexDump.hexblock(self.stackFrame, self.sp,
                                    bits = self.bits, width = width)

        if self.faultCode and not self.modFileName:
            msg += '\nCode dump:\n'
            msg += HexDump.hexblock(self.faultCode, self.pc,
                                    bits = self.bits, width = width)

        if self.faultMem:
            if self.faultPeek:
                msg += '\nException address pointers:\n'
                msg += CrashDump.dump_data_peek(self.faultPeek,
                                                self.exceptionAddress,
                                                bits  = self.bits,
                                                width = width)
            msg += '\nException address dump:\n'
            msg += HexDump.hexblock(self.faultMem, self.exceptionAddress,
                                    bits = self.bits, width = width)

        if self.memoryMap:
            msg += '\nMemory map:\n'
            mappedFileNames = dict()
            for mbi in self.memoryMap:
                if hasattr(mbi, 'filename') and mbi.filename:
                    mappedFileNames[mbi.BaseAddress] = mbi.filename
            msg += CrashDump.dump_memory_map(self.memoryMap, mappedFileNames,
                                             bits = self.bits)

        if not msg.endswith('\n\n'):
            if not msg.endswith('\n'):
                msg += '\n'
            msg += '\n'
        return msg

    def environmentReport(self):
        """
        @rtype: str
        @return: The process environment variables,
            merged and formatted for a report.
        """
        msg = ''
        if self.environment:
            for key, value in compat.iteritems(self.environment):
                msg += '  %s=%s\n' % (key, value)
        return msg

    def notesReport(self):
        """
        @rtype:  str
        @return: All notes, merged and formatted for a report.
        """
        msg = ''
        if self.notes:
            for n in self.notes:
                n = n.strip('\n')
                if '\n' in n:
                    n = n.strip('\n')
                    msg += ' * %s\n' % n.pop(0)
                    for x in n:
                        msg += '   %s\n' % x
                else:
                    msg += ' * %s\n' % n
        return msg

    def addNote(self, msg):
        """
        Add a note to the crash event.

        @type msg:  str
        @param msg: Note text.
        """
        self.notes.append(msg)

    def clearNotes(self):
        """
        Clear the notes of this crash event.
        """
        self.notes = list()

    def getNotes(self):
        """
        Get the list of notes of this crash event.

        @rtype:  list( str )
        @return: List of notes.
        """
        return self.notes

    def iterNotes(self):
        """
        Iterate the notes of this crash event.

        @rtype:  listiterator
        @return: Iterator of the list of notes.
        """
        return self.notes.__iter__()

    def hasNotes(self):
        """
        @rtype:  bool
        @return: C{True} if there are notes for this crash event.
        """
        return bool( self.notes )

#==============================================================================

class CrashContainer (object):
    """
    Old crash dump persistencer using a DBM database.
    Doesn't support duplicate crashes.

    @warning:
        DBM database support is provided for backwards compatibility with older
        versions of WinAppDbg. New applications should not use this class.
        Also, DBM databases in Python suffer from multiple problems that can
        easily be avoided by switching to a SQL database.

    @see: If you really must use a DBM database, try the standard C{shelve}
        module instead: U{http://docs.python.org/library/shelve.html}

    @group Marshalling configuration:
        optimizeKeys, optimizeValues, compressKeys, compressValues, escapeKeys,
        escapeValues, binaryKeys, binaryValues

    @type optimizeKeys: bool
    @cvar optimizeKeys: Ignored by the current implementation.

        Up to WinAppDbg 1.4 this setting caused the database keys to be
        optimized when pickled with the standard C{pickle} module.

        But with a DBM database backend that causes inconsistencies, since the
        same key can be serialized into multiple optimized pickles, thus losing
        uniqueness.

    @type optimizeValues: bool
    @cvar optimizeValues: C{True} to optimize the marshalling of keys, C{False}
        otherwise. Only used with the C{pickle} module, ignored when using the
        more secure C{cerealizer} module.

    @type compressKeys: bool
    @cvar compressKeys: C{True} to compress keys when marshalling, C{False}
        to leave them uncompressed.

    @type compressValues: bool
    @cvar compressValues: C{True} to compress values when marshalling, C{False}
        to leave them uncompressed.

    @type escapeKeys: bool
    @cvar escapeKeys: C{True} to escape keys when marshalling, C{False}
        to leave them uncompressed.

    @type escapeValues: bool
    @cvar escapeValues: C{True} to escape values when marshalling, C{False}
        to leave them uncompressed.

    @type binaryKeys: bool
    @cvar binaryKeys: C{True} to marshall keys to binary format (the Python
        C{buffer} type), C{False} to use text marshalled keys (C{str} type).

    @type binaryValues: bool
    @cvar binaryValues: C{True} to marshall values to binary format (the Python
        C{buffer} type), C{False} to use text marshalled values (C{str} type).
    """

    optimizeKeys    = False
    optimizeValues  = True
    compressKeys    = False
    compressValues  = True
    escapeKeys      = False
    escapeValues    = False
    binaryKeys      = False
    binaryValues    = False

    def __init__(self, filename = None, allowRepeatedKeys = False):
        """
        @type  filename: str
        @param filename: (Optional) File name for crash database.
            If no filename is specified, the container is volatile.

            Volatile containers are stored only in memory and
            destroyed when they go out of scope.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            Currently not supported, always use C{False}.
        """
        if allowRepeatedKeys:
            raise NotImplementedError()
        self.__filename = filename
        if filename:
            global anydbm
            if not anydbm:
                import anydbm
            self.__db = anydbm.open(filename, 'c')
            self.__keys = dict([ (self.unmarshall_key(mk), mk)
                                                  for mk in self.__db.keys() ])
        else:
            self.__db = dict()
            self.__keys = dict()

    def remove_key(self, key):
        """
        Removes the given key from the set of known keys.

        @type  key: L{Crash} key.
        @param key: Key to remove.
        """
        del self.__keys[key]

    def marshall_key(self, key):
        """
        Marshalls a Crash key to be used in the database.

        @see: L{__init__}

        @type  key: L{Crash} key.
        @param key: Key to convert.

        @rtype:  str or buffer
        @return: Converted key.
        """
        if key in self.__keys:
            return self.__keys[key]
        skey = pickle.dumps(key, protocol = 0)
        if self.compressKeys:
            skey = zlib.compress(skey, zlib.Z_BEST_COMPRESSION)
        if self.escapeKeys:
            skey = skey.encode('hex')
        if self.binaryKeys:
            skey = buffer(skey)
        self.__keys[key] = skey
        return skey

    def unmarshall_key(self, key):
        """
        Unmarshalls a Crash key read from the database.

        @type  key: str or buffer
        @param key: Key to convert.

        @rtype:  L{Crash} key.
        @return: Converted key.
        """
        key = str(key)
        if self.escapeKeys:
            key = key.decode('hex')
        if self.compressKeys:
            key = zlib.decompress(key)
        key = pickle.loads(key)
        return key

    def marshall_value(self, value, storeMemoryMap = False):
        """
        Marshalls a Crash object to be used in the database.
        By default the C{memoryMap} member is B{NOT} stored here.

        @warning: Setting the C{storeMemoryMap} argument to C{True} can lead to
            a severe performance penalty!

        @type  value: L{Crash}
        @param value: Object to convert.

        @type  storeMemoryMap: bool
        @param storeMemoryMap: C{True} to store the memory map, C{False}
            otherwise.

        @rtype:  str
        @return: Converted object.
        """
        if hasattr(value, 'memoryMap'):
            crash = value
            memoryMap = crash.memoryMap
            try:
                crash.memoryMap = None
                if storeMemoryMap and memoryMap is not None:
                    # convert the generator to a list
                    crash.memoryMap = list(memoryMap)
                if self.optimizeValues:
                    value = pickle.dumps(crash, protocol = HIGHEST_PROTOCOL)
                    value = optimize(value)
                else:
                    value = pickle.dumps(crash, protocol = 0)
            finally:
                crash.memoryMap = memoryMap
                del memoryMap
                del crash
        if self.compressValues:
            value = zlib.compress(value, zlib.Z_BEST_COMPRESSION)
        if self.escapeValues:
            value = value.encode('hex')
        if self.binaryValues:
            value = buffer(value)
        return value

    def unmarshall_value(self, value):
        """
        Unmarshalls a Crash object read from the database.

        @type  value: str
        @param value: Object to convert.

        @rtype:  L{Crash}
        @return: Converted object.
        """
        value = str(value)
        if self.escapeValues:
            value = value.decode('hex')
        if self.compressValues:
            value = zlib.decompress(value)
        value = pickle.loads(value)
        return value

    # The interface is meant to be similar to a Python set.
    # However it may not be necessary to implement all of the set methods.
    # Other methods like get, has_key, iterkeys and itervalues
    # are dictionary-like.

    def __len__(self):
        """
        @rtype:  int
        @return: Count of known keys.
        """
        return len(self.__keys)

    def __bool__(self):
        """
        @rtype:  bool
        @return: C{False} if there are no known keys.
        """
        return bool(self.__keys)

    def __contains__(self, crash):
        """
        @type  crash: L{Crash}
        @param crash: Crash object.

        @rtype:  bool
        @return:
            C{True} if a Crash object with the same key is in the container.
        """
        return self.has_key( crash.key() )

    def has_key(self, key):
        """
        @type  key: L{Crash} key.
        @param key: Key to find.

        @rtype:  bool
        @return: C{True} if the key is present in the set of known keys.
        """
        return key in self.__keys

    def iterkeys(self):
        """
        @rtype:  iterator
        @return: Iterator of known L{Crash} keys.
        """
        return compat.iterkeys(self.__keys)

    class __CrashContainerIterator (object):
        """
        Iterator of Crash objects. Returned by L{CrashContainer.__iter__}.
        """

        def __init__(self, container):
            """
            @type  container: L{CrashContainer}
            @param container: Crash set to iterate.
            """
            # It's important to keep a reference to the CrashContainer,
            # rather than it's underlying database.
            # Otherwise the destructor of CrashContainer may close the
            # database while we're still iterating it.
            #
            # TODO: lock the database when iterating it.
            #
            self.__container = container
            self.__keys_iter = compat.iterkeys(container)

        def next(self):
            """
            @rtype:  L{Crash}
            @return: A B{copy} of a Crash object in the L{CrashContainer}.
            @raise StopIteration: No more items left.
            """
            key  = self.__keys_iter.next()
            return self.__container.get(key)

    def __del__(self):
        "Class destructor. Closes the database when this object is destroyed."
        try:
            if self.__filename:
                self.__db.close()
        except:
            pass

    def __iter__(self):
        """
        @see:    L{itervalues}
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.
        """
        return self.itervalues()

    def itervalues(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.

        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        return self.__CrashContainerIterator(self)

    def add(self, crash):
        """
        Adds a new crash to the container.
        If the crash appears to be already known, it's ignored.

        @see: L{Crash.key}

        @type  crash: L{Crash}
        @param crash: Crash object to add.
        """
        if crash not in self:
            key  = crash.key()
            skey = self.marshall_key(key)
            data = self.marshall_value(crash, storeMemoryMap = True)
            self.__db[skey] = data

    def __delitem__(self, key):
        """
        Removes a crash from the container.

        @type  key: L{Crash} unique key.
        @param key: Key of the crash to get.
        """
        skey = self.marshall_key(key)
        del self.__db[skey]
        self.remove_key(key)

    def remove(self, crash):
        """
        Removes a crash from the container.

        @type  crash: L{Crash}
        @param crash: Crash object to remove.
        """
        del self[ crash.key() ]

    def get(self, key):
        """
        Retrieves a crash from the container.

        @type  key: L{Crash} unique key.
        @param key: Key of the crash to get.

        @rtype:  L{Crash} object.
        @return: Crash matching the given key.

        @see:     L{iterkeys}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        skey  = self.marshall_key(key)
        data  = self.__db[skey]
        crash = self.unmarshall_value(data)
        return crash

    def __getitem__(self, key):
        """
        Retrieves a crash from the container.

        @type  key: L{Crash} unique key.
        @param key: Key of the crash to get.

        @rtype:  L{Crash} object.
        @return: Crash matching the given key.

        @see:     L{iterkeys}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        return self.get(key)

#==============================================================================

class CrashDictionary(object):
    """
    Dictionary-like persistence interface for L{Crash} objects.

    Currently the only implementation is through L{sql.CrashDAO}.
    """

    def __init__(self, url, creator = None, allowRepeatedKeys = True):
        """
        @type  url: str
        @param url: Connection URL of the crash database.
            See L{sql.CrashDAO.__init__} for more details.

        @type  creator: callable
        @param creator: (Optional) Callback function that creates the SQL
            database connection.

            Normally it's not necessary to use this argument. However in some
            odd cases you may need to customize the database connection, for
            example when using the integrated authentication in MSSQL.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same signature as a
            previously existing object will be ignored.
        """
        global sql
        if sql is None:
            from winappdbg import sql
        self._allowRepeatedKeys = allowRepeatedKeys
        self._dao = sql.CrashDAO(url, creator)

    def add(self, crash):
        """
        Adds a new crash to the container.

        @note:
            When the C{allowRepeatedKeys} parameter of the constructor
            is set to C{False}, duplicated crashes are ignored.

        @see: L{Crash.key}

        @type  crash: L{Crash}
        @param crash: Crash object to add.
        """
        self._dao.add(crash, self._allowRepeatedKeys)

    def get(self, key):
        """
        Retrieves a crash from the container.

        @type  key: L{Crash} signature.
        @param key: Heuristic signature of the crash to get.

        @rtype:  L{Crash} object.
        @return: Crash matching the given signature. If more than one is found,
            retrieve the newest one.

        @see:     L{iterkeys}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        found = self._dao.find(signature=key, limit=1, order=-1)
        if not found:
            raise KeyError(key)
        return found[0]

    def __iter__(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.
        """
        offset = 0
        limit  = 10
        while 1:
            found = self._dao.find(offset=offset, limit=limit)
            if not found:
                break
            offset += len(found)
            for crash in found:
                yield crash

    def itervalues(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.
        """
        return self.__iter__()

    def iterkeys(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} heuristic signatures.
        """
        for crash in self:
            yield crash.signature       # FIXME this gives repeated results!

    def __contains__(self, crash):
        """
        @type  crash: L{Crash}
        @param crash: Crash object.

        @rtype:  bool
        @return: C{True} if the Crash object is in the container.
        """
        return self._dao.count(signature=crash.signature) > 0

    def has_key(self, key):
        """
        @type  key: L{Crash} signature.
        @param key: Heuristic signature of the crash to get.

        @rtype:  bool
        @return: C{True} if a matching L{Crash} object is in the container.
        """
        return self._dao.count(signature=key) > 0

    def __len__(self):
        """
        @rtype:  int
        @return: Count of L{Crash} elements in the container.
        """
        return self._dao.count()

    def __bool__(self):
        """
        @rtype:  bool
        @return: C{False} if the container is empty.
        """
        return bool( len(self) )

class CrashTable(CrashDictionary):
    """
    Old crash dump persistencer using a SQLite database.

    @warning:
        Superceded by L{CrashDictionary} since WinAppDbg 1.5.
        New applications should not use this class.
    """

    def __init__(self, location = None, allowRepeatedKeys = True):
        """
        @type  location: str
        @param location: (Optional) Location of the crash database.
            If the location is a filename, it's an SQLite database file.

            If no location is specified, the container is volatile.
            Volatile containers are stored only in memory and
            destroyed when they go out of scope.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same signature as a
            previously existing object will be ignored.
        """
        warnings.warn(
            "The %s class is deprecated since WinAppDbg 1.5." % self.__class__,
            DeprecationWarning)
        if location:
            url = "sqlite:///%s" % location
        else:
            url = "sqlite://"
        super(CrashTable, self).__init__(url, allowRepeatedKeys)

class CrashTableMSSQL (CrashDictionary):
    """
    Old crash dump persistencer using a Microsoft SQL Server database.

    @warning:
        Superceded by L{CrashDictionary} since WinAppDbg 1.5.
        New applications should not use this class.
    """

    def __init__(self, location = None, allowRepeatedKeys = True):
        """
        @type  location: str
        @param location: Location of the crash database.
            It must be an ODBC connection string.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same signature as a
            previously existing object will be ignored.
        """
        warnings.warn(
            "The %s class is deprecated since WinAppDbg 1.5." % self.__class__,
            DeprecationWarning)
        import urllib
        url = "mssql+pyodbc:///?odbc_connect=" + urllib.quote_plus(location)
        super(CrashTableMSSQL, self).__init__(url, allowRepeatedKeys)

class VolatileCrashContainer (CrashTable):
    """
    Old in-memory crash dump storage.

    @warning:
        Superceded by L{CrashDictionary} since WinAppDbg 1.5.
        New applications should not use this class.
    """

    def __init__(self, allowRepeatedKeys = True):
        """
        Volatile containers are stored only in memory and
        destroyed when they go out of scope.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same key as a
            previously existing object will be ignored.
        """
        super(VolatileCrashContainer, self).__init__(
            allowRepeatedKeys=allowRepeatedKeys)

class DummyCrashContainer(object):
    """
    Fakes a database of volatile Crash objects,
    trying to mimic part of it's interface, but
    doesn't actually store anything.

    Normally applications don't need to use this.

    @see: L{CrashDictionary}
    """

    def __init__(self, allowRepeatedKeys = True):
        """
        Fake containers don't store L{Crash} objects, but they implement the
        interface properly.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            Mimics the duplicate filter behavior found in real containers.
        """
        self.__keys  = set()
        self.__count = 0
        self.__allowRepeatedKeys = allowRepeatedKeys

    def __contains__(self, crash):
        """
        @type  crash: L{Crash}
        @param crash: Crash object.

        @rtype:  bool
        @return: C{True} if the Crash object is in the container.
        """
        return crash.signature in self.__keys

    def __len__(self):
        """
        @rtype:  int
        @return: Count of L{Crash} elements in the container.
        """
        if self.__allowRepeatedKeys:
            return self.__count
        return len( self.__keys )

    def __bool__(self):
        """
        @rtype:  bool
        @return: C{False} if the container is empty.
        """
        return bool( len(self) )

    def add(self, crash):
        """
        Adds a new crash to the container.

        @note:
            When the C{allowRepeatedKeys} parameter of the constructor
            is set to C{False}, duplicated crashes are ignored.

        @see: L{Crash.key}

        @type  crash: L{Crash}
        @param crash: Crash object to add.
        """
        self.__keys.add( crash.signature )
        self.__count += 1

    def get(self, key):
        """
        This method is not supported.
        """
        raise NotImplementedError()

    def has_key(self, key):
        """
        @type  key: L{Crash} signature.
        @param key: Heuristic signature of the crash to get.

        @rtype:  bool
        @return: C{True} if a matching L{Crash} object is in the container.
        """
        return self.__keys.has_key( key )

    def iterkeys(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} object keys.

        @see:     L{get}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        return iter(self.__keys)

#==============================================================================
# Register the Crash class with the secure serializer.

try:
    cerealizer.register(Crash)
    cerealizer.register(win32.MemoryBasicInformation)
except NameError:
    pass
