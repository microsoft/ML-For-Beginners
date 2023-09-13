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
Miscellaneous utility classes and functions.

@group Helpers:
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
    Regenerator,
    BannerHelpFormatter,
    StaticClass,
    classproperty
"""

__revision__ = "$Id$"

__all__ = [

    # Filename and pathname manipulation
    'PathOperations',

    # Memory address operations
    'MemoryAddresses',
    'CustomAddressIterator',
    'DataAddressIterator',
    'ImageAddressIterator',
    'MappedAddressIterator',
    'ExecutableAddressIterator',
    'ReadableAddressIterator',
    'WriteableAddressIterator',
    'ExecutableAndWriteableAddressIterator',

    # Debug registers manipulation
    'DebugRegister',

    # Miscellaneous
    'Regenerator',
    ]

import sys
import os
import ctypes
import optparse

from winappdbg import win32
from winappdbg import compat

#==============================================================================

class classproperty(property):
    """
    Class property method.

    Only works for getting properties, if you set them
    the symbol gets overwritten in the class namespace.

    Inspired on: U{http://stackoverflow.com/a/7864317/426293}
    """
    def __init__(self, fget=None, fset=None, fdel=None, doc=""):
        if fset is not None or fdel is not None:
            raise NotImplementedError()
        super(classproperty, self).__init__(fget=classmethod(fget), doc=doc)
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class BannerHelpFormatter(optparse.IndentedHelpFormatter):
    "Just a small tweak to optparse to be able to print a banner."
    def __init__(self, banner, *argv, **argd):
        self.banner = banner
        optparse.IndentedHelpFormatter.__init__(self, *argv, **argd)
    def format_usage(self, usage):
        msg = optparse.IndentedHelpFormatter.format_usage(self, usage)
        return '%s\n%s' % (self.banner, msg)

# See Process.generate_memory_snapshot()
class Regenerator(object):
    """
    Calls a generator and iterates it. When it's finished iterating, the
    generator is called again. This allows you to iterate a generator more
    than once (well, sort of).
    """

    def __init__(self, g_function, *v_args, **d_args):
        """
        @type  g_function: function
        @param g_function: Function that when called returns a generator.

        @type  v_args: tuple
        @param v_args: Variable arguments to pass to the generator function.

        @type  d_args: dict
        @param d_args: Variable arguments to pass to the generator function.
        """
        self.__g_function = g_function
        self.__v_args     = v_args
        self.__d_args     = d_args
        self.__g_object   = None

    def __iter__(self):
        'x.__iter__() <==> iter(x)'
        return self

    def next(self):
        'x.next() -> the next value, or raise StopIteration'
        if self.__g_object is None:
            self.__g_object = self.__g_function( *self.__v_args, **self.__d_args )
        try:
            return self.__g_object.next()
        except StopIteration:
            self.__g_object = None
            raise

class StaticClass (object):
    def __new__(cls, *argv, **argd):
        "Don't try to instance this class, just use the static methods."
        raise NotImplementedError(
                "Cannot instance static class %s" % cls.__name__)

#==============================================================================

class PathOperations (StaticClass):
    """
    Static methods for filename and pathname manipulation.
    """

    @staticmethod
    def path_is_relative(path):
        """
        @see: L{path_is_absolute}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  bool
        @return: C{True} if the path is relative, C{False} if it's absolute.
        """
        return win32.PathIsRelative(path)

    @staticmethod
    def path_is_absolute(path):
        """
        @see: L{path_is_relative}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  bool
        @return: C{True} if the path is absolute, C{False} if it's relative.
        """
        return not win32.PathIsRelative(path)

    @staticmethod
    def make_relative(path, current = None):
        """
        @type  path: str
        @param path: Absolute path.

        @type  current: str
        @param current: (Optional) Path to the current directory.

        @rtype:  str
        @return: Relative path.

        @raise WindowsError: It's impossible to make the path relative.
            This happens when the path and the current path are not on the
            same disk drive or network share.
        """
        return win32.PathRelativePathTo(pszFrom = current, pszTo = path)

    @staticmethod
    def make_absolute(path):
        """
        @type  path: str
        @param path: Relative path.

        @rtype:  str
        @return: Absolute path.
        """
        return win32.GetFullPathName(path)[0]

    @staticmethod
    def split_extension(pathname):
        """
        @type  pathname: str
        @param pathname: Absolute path.

        @rtype:  tuple( str, str )
        @return:
            Tuple containing the file and extension components of the filename.
        """
        filepart = win32.PathRemoveExtension(pathname)
        extpart  = win32.PathFindExtension(pathname)
        return (filepart, extpart)

    @staticmethod
    def split_filename(pathname):
        """
        @type  pathname: str
        @param pathname: Absolute path.

        @rtype:  tuple( str, str )
        @return: Tuple containing the path to the file and the base filename.
        """
        filepart = win32.PathFindFileName(pathname)
        pathpart = win32.PathRemoveFileSpec(pathname)
        return (pathpart, filepart)

    @staticmethod
    def split_path(path):
        """
        @see: L{join_path}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  list( str... )
        @return: List of path components.
        """
        components = list()
        while path:
            next = win32.PathFindNextComponent(path)
            if next:
                prev = path[ : -len(next) ]
                components.append(prev)
            path = next
        return components

    @staticmethod
    def join_path(*components):
        """
        @see: L{split_path}

        @type  components: tuple( str... )
        @param components: Path components.

        @rtype:  str
        @return: Absolute or relative path.
        """
        if components:
            path = components[0]
            for next in components[1:]:
                path = win32.PathAppend(path, next)
        else:
            path = ""
        return path

    @staticmethod
    def native_to_win32_pathname(name):
        """
        @type  name: str
        @param name: Native (NT) absolute pathname.

        @rtype:  str
        @return: Win32 absolute pathname.
        """
        # XXX TODO
        # There are probably some native paths that
        # won't be converted by this naive approach.
        if name.startswith(compat.b("\\")):
            if name.startswith(compat.b("\\??\\")):
                name = name[4:]
            elif name.startswith(compat.b("\\SystemRoot\\")):
                system_root_path = os.environ['SYSTEMROOT']
                if system_root_path.endswith('\\'):
                    system_root_path = system_root_path[:-1]
                name = system_root_path + name[11:]
            else:
                for drive_number in compat.xrange(ord('A'), ord('Z') + 1):
                    drive_letter = '%c:' % drive_number
                    try:
                        device_native_path = win32.QueryDosDevice(drive_letter)
                    except WindowsError:
                        e = sys.exc_info()[1]
                        if e.winerror in (win32.ERROR_FILE_NOT_FOUND, \
                                                 win32.ERROR_PATH_NOT_FOUND):
                            continue
                        raise
                    if not device_native_path.endswith(compat.b('\\')):
                        device_native_path += compat.b('\\')
                    if name.startswith(device_native_path):
                        name = drive_letter + compat.b('\\') + \
                                              name[ len(device_native_path) : ]
                        break
        return name

    @staticmethod
    def pathname_to_filename(pathname):
        """
        Equivalent to: C{PathOperations.split_filename(pathname)[0]}

        @note: This function is preserved for backwards compatibility with
            WinAppDbg 1.4 and earlier. It may be removed in future versions.

        @type  pathname: str
        @param pathname: Absolute path to a file.

        @rtype:  str
        @return: Filename component of the path.
        """
        return win32.PathFindFileName(pathname)

#==============================================================================

class MemoryAddresses (StaticClass):
    """
    Class to manipulate memory addresses.

    @type pageSize: int
    @cvar pageSize: Page size in bytes. Defaults to 0x1000 but it's
        automatically updated on runtime when importing the module.
    """

    @classproperty
    def pageSize(cls):
        """
        Try to get the pageSize value on runtime.
        """
        try:
            try:
                pageSize = win32.GetSystemInfo().dwPageSize
            except WindowsError:
                pageSize = 0x1000
        except NameError:
            pageSize = 0x1000
        cls.pageSize = pageSize     # now this function won't be called again
        return pageSize

    @classmethod
    def align_address_to_page_start(cls, address):
        """
        Align the given address to the start of the page it occupies.

        @type  address: int
        @param address: Memory address.

        @rtype:  int
        @return: Aligned memory address.
        """
        return address - ( address % cls.pageSize )

    @classmethod
    def align_address_to_page_end(cls, address):
        """
        Align the given address to the end of the page it occupies.
        That is, to point to the start of the next page.

        @type  address: int
        @param address: Memory address.

        @rtype:  int
        @return: Aligned memory address.
        """
        return address + cls.pageSize - ( address % cls.pageSize )

    @classmethod
    def align_address_range(cls, begin, end):
        """
        Align the given address range to the start and end of the page(s) it occupies.

        @type  begin: int
        @param begin: Memory address of the beginning of the buffer.
            Use C{None} for the first legal address in the address space.

        @type  end: int
        @param end: Memory address of the end of the buffer.
            Use C{None} for the last legal address in the address space.

        @rtype:  tuple( int, int )
        @return: Aligned memory addresses.
        """
        if begin is None:
            begin = 0
        if end is None:
            end = win32.LPVOID(-1).value  # XXX HACK
        if end < begin:
            begin, end = end, begin
        begin = cls.align_address_to_page_start(begin)
        if end != cls.align_address_to_page_start(end):
            end = cls.align_address_to_page_end(end)
        return (begin, end)

    @classmethod
    def get_buffer_size_in_pages(cls, address, size):
        """
        Get the number of pages in use by the given buffer.

        @type  address: int
        @param address: Aligned memory address.

        @type  size: int
        @param size: Buffer size.

        @rtype:  int
        @return: Buffer size in number of pages.
        """
        if size < 0:
            size    = -size
            address = address - size
        begin, end = cls.align_address_range(address, address + size)
        # XXX FIXME
        # I think this rounding fails at least for address 0xFFFFFFFF size 1
        return int(float(end - begin) / float(cls.pageSize))

    @staticmethod
    def do_ranges_intersect(begin, end, old_begin, old_end):
        """
        Determine if the two given memory address ranges intersect.

        @type  begin: int
        @param begin: Start address of the first range.

        @type  end: int
        @param end: End address of the first range.

        @type  old_begin: int
        @param old_begin: Start address of the second range.

        @type  old_end: int
        @param old_end: End address of the second range.

        @rtype:  bool
        @return: C{True} if the two ranges intersect, C{False} otherwise.
        """
        return  (old_begin <= begin < old_end) or \
                (old_begin < end <= old_end)   or \
                (begin <= old_begin < end)     or \
                (begin < old_end <= end)

#==============================================================================

def CustomAddressIterator(memory_map, condition):
    """
    Generator function that iterates through a memory map, filtering memory
    region blocks by any given condition.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @type  condition: function
    @param condition: Callback function that returns C{True} if the memory
        block should be returned, or C{False} if it should be filtered.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    for mbi in memory_map:
        if condition(mbi):
            address  = mbi.BaseAddress
            max_addr = address + mbi.RegionSize
            while address < max_addr:
                yield address
                address = address + 1

def DataAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that contain data.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map,
                                      win32.MemoryBasicInformation.has_content)

def ImageAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that belong to executable images.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map,
                                         win32.MemoryBasicInformation.is_image)

def MappedAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that belong to memory mapped files.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map,
                                        win32.MemoryBasicInformation.is_mapped)

def ReadableAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that are readable.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map,
                                      win32.MemoryBasicInformation.is_readable)

def WriteableAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that are writeable.

    @note: Writeable memory is always readable too.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map,
                                     win32.MemoryBasicInformation.is_writeable)

def ExecutableAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that are executable.

    @note: Executable memory is always readable too.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map,
                                    win32.MemoryBasicInformation.is_executable)

def ExecutableAndWriteableAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that are executable and writeable.

    @note: The presence of such pages make memory corruption vulnerabilities
        much easier to exploit.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map,
                      win32.MemoryBasicInformation.is_executable_and_writeable)

#==============================================================================
try:
    _registerMask = win32.SIZE_T(-1).value
except TypeError:
    if win32.SIZEOF(win32.SIZE_T) == 4:
        _registerMask = 0xFFFFFFFF
    elif win32.SIZEOF(win32.SIZE_T) == 8:
        _registerMask = 0xFFFFFFFFFFFFFFFF
    else:
        raise

class DebugRegister (StaticClass):
    """
    Class to manipulate debug registers.
    Used by L{HardwareBreakpoint}.

    @group Trigger flags used by HardwareBreakpoint:
        BREAK_ON_EXECUTION, BREAK_ON_WRITE, BREAK_ON_ACCESS, BREAK_ON_IO_ACCESS
    @group Size flags used by HardwareBreakpoint:
        WATCH_BYTE, WATCH_WORD, WATCH_DWORD, WATCH_QWORD
    @group Bitwise masks for Dr7:
        enableMask, disableMask, triggerMask, watchMask, clearMask,
        generalDetectMask
    @group Bitwise masks for Dr6:
        hitMask, hitMaskAll, debugAccessMask, singleStepMask, taskSwitchMask,
        clearDr6Mask, clearHitMask
    @group Debug control MSR definitions:
        DebugCtlMSR, LastBranchRecord, BranchTrapFlag, PinControl,
        LastBranchToIP, LastBranchFromIP,
        LastExceptionToIP, LastExceptionFromIP

    @type BREAK_ON_EXECUTION: int
    @cvar BREAK_ON_EXECUTION: Break on execution.

    @type BREAK_ON_WRITE: int
    @cvar BREAK_ON_WRITE: Break on write.

    @type BREAK_ON_ACCESS: int
    @cvar BREAK_ON_ACCESS: Break on read or write.

    @type BREAK_ON_IO_ACCESS: int
    @cvar BREAK_ON_IO_ACCESS: Break on I/O port access.
        Not supported by any hardware.

    @type WATCH_BYTE: int
    @cvar WATCH_BYTE: Watch a byte.

    @type WATCH_WORD: int
    @cvar WATCH_WORD: Watch a word.

    @type WATCH_DWORD: int
    @cvar WATCH_DWORD: Watch a double word.

    @type WATCH_QWORD: int
    @cvar WATCH_QWORD: Watch one quad word.

    @type enableMask: 4-tuple of integers
    @cvar enableMask:
        Enable bit on C{Dr7} for each slot.
        Works as a bitwise-OR mask.

    @type disableMask: 4-tuple of integers
    @cvar disableMask:
        Mask of the enable bit on C{Dr7} for each slot.
        Works as a bitwise-AND mask.

    @type triggerMask: 4-tuple of 2-tuples of integers
    @cvar triggerMask:
        Trigger bits on C{Dr7} for each trigger flag value.
        Each 2-tuple has the bitwise-OR mask and the bitwise-AND mask.

    @type watchMask: 4-tuple of 2-tuples of integers
    @cvar watchMask:
        Watch bits on C{Dr7} for each watch flag value.
        Each 2-tuple has the bitwise-OR mask and the bitwise-AND mask.

    @type clearMask: 4-tuple of integers
    @cvar clearMask:
        Mask of all important bits on C{Dr7} for each slot.
        Works as a bitwise-AND mask.

    @type generalDetectMask: integer
    @cvar generalDetectMask:
        General detect mode bit. It enables the processor to notify the
        debugger when the debugee is trying to access one of the debug
        registers.

    @type hitMask: 4-tuple of integers
    @cvar hitMask:
        Hit bit on C{Dr6} for each slot.
        Works as a bitwise-AND mask.

    @type hitMaskAll: integer
    @cvar hitMaskAll:
        Bitmask for all hit bits in C{Dr6}. Useful to know if at least one
        hardware breakpoint was hit, or to clear the hit bits only.

    @type clearHitMask: integer
    @cvar clearHitMask:
        Bitmask to clear all the hit bits in C{Dr6}.

    @type debugAccessMask: integer
    @cvar debugAccessMask:
        The debugee tried to access a debug register. Needs bit
        L{generalDetectMask} enabled in C{Dr7}.

    @type singleStepMask: integer
    @cvar singleStepMask:
        A single step exception was raised. Needs the trap flag enabled.

    @type taskSwitchMask: integer
    @cvar taskSwitchMask:
        A task switch has occurred. Needs the TSS T-bit set to 1.

    @type clearDr6Mask: integer
    @cvar clearDr6Mask:
        Bitmask to clear all meaningful bits in C{Dr6}.
    """

    BREAK_ON_EXECUTION  = 0
    BREAK_ON_WRITE      = 1
    BREAK_ON_ACCESS     = 3
    BREAK_ON_IO_ACCESS  = 2

    WATCH_BYTE  = 0
    WATCH_WORD  = 1
    WATCH_DWORD = 3
    WATCH_QWORD = 2

    registerMask = _registerMask

#------------------------------------------------------------------------------

    ###########################################################################
    # http://en.wikipedia.org/wiki/Debug_register
    #
    # DR7 - Debug control
    #
    # The low-order eight bits of DR7 (0,2,4,6 and 1,3,5,7) selectively enable
    # the four address breakpoint conditions. There are two levels of enabling:
    # the local (0,2,4,6) and global (1,3,5,7) levels. The local enable bits
    # are automatically reset by the processor at every task switch to avoid
    # unwanted breakpoint conditions in the new task. The global enable bits
    # are not reset by a task switch; therefore, they can be used for
    # conditions that are global to all tasks.
    #
    # Bits 16-17 (DR0), 20-21 (DR1), 24-25 (DR2), 28-29 (DR3), define when
    # breakpoints trigger. Each breakpoint has a two-bit entry that specifies
    # whether they break on execution (00b), data write (01b), data read or
    # write (11b). 10b is defined to mean break on IO read or write but no
    # hardware supports it. Bits 18-19 (DR0), 22-23 (DR1), 26-27 (DR2), 30-31
    # (DR3), define how large area of memory is watched by breakpoints. Again
    # each breakpoint has a two-bit entry that specifies whether they watch
    # one (00b), two (01b), eight (10b) or four (11b) bytes.
    ###########################################################################

    # Dr7 |= enableMask[register]
    enableMask = (
        1 << 0,     # Dr0 (bit 0)
        1 << 2,     # Dr1 (bit 2)
        1 << 4,     # Dr2 (bit 4)
        1 << 6,     # Dr3 (bit 6)
    )

    # Dr7 &= disableMask[register]
    disableMask = tuple( [_registerMask ^ x for x in enableMask] ) # The registerMask from the class is not there in py3
    try:
        del x # It's not there in py3
    except:
        pass

    # orMask, andMask = triggerMask[register][trigger]
    # Dr7 = (Dr7 & andMask) | orMask    # to set
    # Dr7 = Dr7 & andMask               # to remove
    triggerMask = (
        # Dr0 (bits 16-17)
        (
            ((0 << 16), (3 << 16) ^ registerMask),  # execute
            ((1 << 16), (3 << 16) ^ registerMask),  # write
            ((2 << 16), (3 << 16) ^ registerMask),  # io read
            ((3 << 16), (3 << 16) ^ registerMask),  # access
        ),
        # Dr1 (bits 20-21)
        (
            ((0 << 20), (3 << 20) ^ registerMask),  # execute
            ((1 << 20), (3 << 20) ^ registerMask),  # write
            ((2 << 20), (3 << 20) ^ registerMask),  # io read
            ((3 << 20), (3 << 20) ^ registerMask),  # access
        ),
        # Dr2 (bits 24-25)
        (
            ((0 << 24), (3 << 24) ^ registerMask),  # execute
            ((1 << 24), (3 << 24) ^ registerMask),  # write
            ((2 << 24), (3 << 24) ^ registerMask),  # io read
            ((3 << 24), (3 << 24) ^ registerMask),  # access
        ),
        # Dr3 (bits 28-29)
        (
            ((0 << 28), (3 << 28) ^ registerMask),  # execute
            ((1 << 28), (3 << 28) ^ registerMask),  # write
            ((2 << 28), (3 << 28) ^ registerMask),  # io read
            ((3 << 28), (3 << 28) ^ registerMask),  # access
        ),
    )

    # orMask, andMask = watchMask[register][watch]
    # Dr7 = (Dr7 & andMask) | orMask    # to set
    # Dr7 = Dr7 & andMask               # to remove
    watchMask = (
        # Dr0 (bits 18-19)
        (
            ((0 << 18), (3 << 18) ^ registerMask),  # byte
            ((1 << 18), (3 << 18) ^ registerMask),  # word
            ((2 << 18), (3 << 18) ^ registerMask),  # qword
            ((3 << 18), (3 << 18) ^ registerMask),  # dword
        ),
        # Dr1 (bits 22-23)
        (
            ((0 << 23), (3 << 23) ^ registerMask),  # byte
            ((1 << 23), (3 << 23) ^ registerMask),  # word
            ((2 << 23), (3 << 23) ^ registerMask),  # qword
            ((3 << 23), (3 << 23) ^ registerMask),  # dword
        ),
        # Dr2 (bits 26-27)
        (
            ((0 << 26), (3 << 26) ^ registerMask),  # byte
            ((1 << 26), (3 << 26) ^ registerMask),  # word
            ((2 << 26), (3 << 26) ^ registerMask),  # qword
            ((3 << 26), (3 << 26) ^ registerMask),  # dword
        ),
        # Dr3 (bits 30-31)
        (
            ((0 << 30), (3 << 31) ^ registerMask),  # byte
            ((1 << 30), (3 << 31) ^ registerMask),  # word
            ((2 << 30), (3 << 31) ^ registerMask),  # qword
            ((3 << 30), (3 << 31) ^ registerMask),  # dword
        ),
    )

    # Dr7 = Dr7 & clearMask[register]
    clearMask = (
        registerMask ^ ( (1 << 0) + (3 << 16) + (3 << 18) ),    # Dr0
        registerMask ^ ( (1 << 2) + (3 << 20) + (3 << 22) ),    # Dr1
        registerMask ^ ( (1 << 4) + (3 << 24) + (3 << 26) ),    # Dr2
        registerMask ^ ( (1 << 6) + (3 << 28) + (3 << 30) ),    # Dr3
    )

    # Dr7 = Dr7 | generalDetectMask
    generalDetectMask = (1 << 13)

    ###########################################################################
    # http://en.wikipedia.org/wiki/Debug_register
    #
    # DR6 - Debug status
    #
    # The debug status register permits the debugger to determine which debug
    # conditions have occurred. When the processor detects an enabled debug
    # exception, it sets the low-order bits of this register (0,1,2,3) before
    # entering the debug exception handler.
    #
    # Note that the bits of DR6 are never cleared by the processor. To avoid
    # any confusion in identifying the next debug exception, the debug handler
    # should move zeros to DR6 immediately before returning.
    ###########################################################################

    # bool(Dr6 & hitMask[register])
    hitMask = (
        (1 << 0),   # Dr0
        (1 << 1),   # Dr1
        (1 << 2),   # Dr2
        (1 << 3),   # Dr3
    )

    # bool(Dr6 & anyHitMask)
    hitMaskAll = hitMask[0] | hitMask[1] | hitMask[2] | hitMask[3]

    # Dr6 = Dr6 & clearHitMask
    clearHitMask = registerMask ^ hitMaskAll

    # bool(Dr6 & debugAccessMask)
    debugAccessMask = (1 << 13)

    # bool(Dr6 & singleStepMask)
    singleStepMask  = (1 << 14)

    # bool(Dr6 & taskSwitchMask)
    taskSwitchMask  = (1 << 15)

    # Dr6 = Dr6 & clearDr6Mask
    clearDr6Mask = registerMask ^ (hitMaskAll | \
                            debugAccessMask | singleStepMask | taskSwitchMask)

#------------------------------------------------------------------------------

###############################################################################
#
#    (from the AMD64 manuals)
#
#    The fields within the DebugCtlMSR register are:
#
#    Last-Branch Record (LBR) - Bit 0, read/write. Software sets this bit to 1
#    to cause the processor to record the source and target addresses of the
#    last control transfer taken before a debug exception occurs. The recorded
#    control transfers include branch instructions, interrupts, and exceptions.
#
#    Branch Single Step (BTF) - Bit 1, read/write. Software uses this bit to
#    change the behavior of the rFLAGS.TF bit. When this bit is cleared to 0,
#    the rFLAGS.TF bit controls instruction single stepping, (normal behavior).
#    When this bit is set to 1, the rFLAGS.TF bit controls single stepping on
#    control transfers. The single-stepped control transfers include branch
#    instructions, interrupts, and exceptions. Control-transfer single stepping
#    requires both BTF=1 and rFLAGS.TF=1.
#
#    Performance-Monitoring/Breakpoint Pin-Control (PBi) - Bits 5-2, read/write.
#    Software uses these bits to control the type of information reported by
#    the four external performance-monitoring/breakpoint pins on the processor.
#    When a PBi bit is cleared to 0, the corresponding external pin (BPi)
#    reports performance-monitor information. When a PBi bit is set to 1, the
#    corresponding external pin (BPi) reports breakpoint information.
#
#    All remaining bits in the DebugCtlMSR register are reserved.
#
#    Software can enable control-transfer single stepping by setting
#    DebugCtlMSR.BTF to 1 and rFLAGS.TF to 1. The processor automatically
#    disables control-transfer single stepping when a debug exception (#DB)
#    occurs by clearing DebugCtlMSR.BTF to 0. rFLAGS.TF is also cleared when a
#    #DB exception occurs. Before exiting the debug-exception handler, software
#    must set both DebugCtlMSR.BTF and rFLAGS.TF to 1 to restart single
#    stepping.
#
###############################################################################

    DebugCtlMSR      = 0x1D9
    LastBranchRecord = (1 << 0)
    BranchTrapFlag   = (1 << 1)
    PinControl       = (
                        (1 << 2),   # PB1
                        (1 << 3),   # PB2
                        (1 << 4),   # PB3
                        (1 << 5),   # PB4
                       )

###############################################################################
#
#    (from the AMD64 manuals)
#
#    Control-transfer recording MSRs: LastBranchToIP, LastBranchFromIP,
#    LastExceptionToIP, and LastExceptionFromIP. These registers are loaded
#    automatically by the processor when the DebugCtlMSR.LBR bit is set to 1.
#    These MSRs are read-only.
#
#    The processor automatically disables control-transfer recording when a
#    debug exception (#DB) occurs by clearing DebugCtlMSR.LBR to 0. The
#    contents of the control-transfer recording MSRs are not altered by the
#    processor when the #DB occurs. Before exiting the debug-exception handler,
#    software can set DebugCtlMSR.LBR to 1 to re-enable the recording mechanism.
#
###############################################################################

    LastBranchToIP      = 0x1DC
    LastBranchFromIP    = 0x1DB
    LastExceptionToIP   = 0x1DE
    LastExceptionFromIP = 0x1DD

#------------------------------------------------------------------------------

    @classmethod
    def clear_bp(cls, ctx, register):
        """
        Clears a hardware breakpoint.

        @see: find_slot, set_bp

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @type  register: int
        @param register: Slot (debug register) for hardware breakpoint.
        """
        ctx['Dr7'] &= cls.clearMask[register]
        ctx['Dr%d' % register] = 0

    @classmethod
    def set_bp(cls, ctx, register, address, trigger, watch):
        """
        Sets a hardware breakpoint.

        @see: clear_bp, find_slot

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @type  register: int
        @param register: Slot (debug register).

        @type  address: int
        @param address: Memory address.

        @type  trigger: int
        @param trigger: Trigger flag. See L{HardwareBreakpoint.validTriggers}.

        @type  watch: int
        @param watch: Watch flag. See L{HardwareBreakpoint.validWatchSizes}.
        """
        Dr7 = ctx['Dr7']
        Dr7 |= cls.enableMask[register]
        orMask, andMask = cls.triggerMask[register][trigger]
        Dr7 &= andMask
        Dr7 |= orMask
        orMask, andMask = cls.watchMask[register][watch]
        Dr7 &= andMask
        Dr7 |= orMask
        ctx['Dr7'] = Dr7
        ctx['Dr%d' % register] = address

    @classmethod
    def find_slot(cls, ctx):
        """
        Finds an empty slot to set a hardware breakpoint.

        @see: clear_bp, set_bp

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @rtype:  int
        @return: Slot (debug register) for hardware breakpoint.
        """
        Dr7  = ctx['Dr7']
        slot = 0
        for m in cls.enableMask:
            if (Dr7 & m) == 0:
                return slot
            slot += 1
        return None
