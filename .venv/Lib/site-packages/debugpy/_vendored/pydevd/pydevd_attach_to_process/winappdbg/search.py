#!~/.wine/drive_c/Python25/python.exe
# -*- coding: utf-8 -*-

# Process memory finder
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
Process memory search.

@group Memory search:
    Search,
    Pattern,
    BytePattern,
    TextPattern,
    RegExpPattern,
    HexPattern
"""

__revision__ = "$Id$"

__all__ =   [
                'Search',
                'Pattern',
                'BytePattern',
                'TextPattern',
                'RegExpPattern',
                'HexPattern',
            ]

from winappdbg.textio import HexInput
from winappdbg.util import StaticClass, MemoryAddresses
from winappdbg import win32

import warnings

try:
    # http://pypi.python.org/pypi/regex
    import regex as re
except ImportError:
    import re

#==============================================================================

class Pattern (object):
    """
    Base class for search patterns.

    The following L{Pattern} subclasses are provided by WinAppDbg:
     - L{BytePattern}
     - L{TextPattern}
     - L{RegExpPattern}
     - L{HexPattern}

    @see: L{Search.search_process}
    """

    def __init__(self, pattern):
        """
        Class constructor.

        The only mandatory argument should be the pattern string.

        This method B{MUST} be reimplemented by subclasses of L{Pattern}.
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Returns the maximum expected length of the strings matched by this
        pattern. Exact behavior is implementation dependent.

        Ideally it should be an exact value, but in some cases it's not
        possible to calculate so an upper limit should be returned instead.

        If that's not possible either an exception must be raised.

        This value will be used to calculate the required buffer size when
        doing buffered searches.

        This method B{MUST} be reimplemented by subclasses of L{Pattern}.
        """
        raise NotImplementedError()

    def read(self, process, address, size):
        """
        Reads the requested number of bytes from the process memory at the
        given address.

        Subclasses of L{Pattern} tipically don't need to reimplement this
        method.
        """
        return process.read(address, size)

    def find(self, buffer, pos = None):
        """
        Searches for the pattern in the given buffer, optionally starting at
        the given position within the buffer.

        This method B{MUST} be reimplemented by subclasses of L{Pattern}.

        @type  buffer: str
        @param buffer: Buffer to search on.

        @type  pos: int
        @param pos:
            (Optional) Position within the buffer to start searching from.

        @rtype:  tuple( int, int )
        @return: Tuple containing the following:
             - Position within the buffer where a match is found, or C{-1} if
               no match was found.
             - Length of the matched data if a match is found, or undefined if
               no match was found.
        """
        raise NotImplementedError()

    def found(self, address, size, data):
        """
        This method gets called when a match is found.

        This allows subclasses of L{Pattern} to filter out unwanted results,
        or modify the results before giving them to the caller of
        L{Search.search_process}.

        If the return value is C{None} the result is skipped.

        Subclasses of L{Pattern} don't need to reimplement this method unless
        filtering is needed.

        @type  address: int
        @param address: The memory address where the pattern was found.

        @type  size: int
        @param size: The size of the data that matches the pattern.

        @type  data: str
        @param data: The data that matches the pattern.

        @rtype:  tuple( int, int, str )
        @return: Tuple containing the following:
             * The memory address where the pattern was found.
             * The size of the data that matches the pattern.
             * The data that matches the pattern.
        """
        return (address, size, data)

#------------------------------------------------------------------------------

class BytePattern (Pattern):
    """
    Fixed byte pattern.

    @type pattern: str
    @ivar pattern: Byte string to search for.

    @type length: int
    @ivar length: Length of the byte pattern.
    """

    def __init__(self, pattern):
        """
        @type  pattern: str
        @param pattern: Byte string to search for.
        """
        self.pattern = str(pattern)
        self.length  = len(pattern)

    def __len__(self):
        """
        Returns the exact length of the pattern.

        @see: L{Pattern.__len__}
        """
        return self.length

    def find(self, buffer, pos = None):
        return buffer.find(self.pattern, pos), self.length

#------------------------------------------------------------------------------

# FIXME: case insensitive compat.unicode searches are probably buggy!

class TextPattern (BytePattern):
    """
    Text pattern.

    @type isUnicode: bool
    @ivar isUnicode: C{True} if the text to search for is a compat.unicode string,
        C{False} otherwise.

    @type encoding: str
    @ivar encoding: Encoding for the text parameter.
        Only used when the text to search for is a Unicode string.
        Don't change unless you know what you're doing!

    @type caseSensitive: bool
    @ivar caseSensitive: C{True} of the search is case sensitive,
        C{False} otherwise.
    """

    def __init__(self, text, encoding = "utf-16le", caseSensitive = False):
        """
        @type  text: str or compat.unicode
        @param text: Text to search for.

        @type  encoding: str
        @param encoding: (Optional) Encoding for the text parameter.
            Only used when the text to search for is a Unicode string.
            Don't change unless you know what you're doing!

        @type  caseSensitive: bool
        @param caseSensitive: C{True} of the search is case sensitive,
            C{False} otherwise.
        """
        self.isUnicode = isinstance(text, compat.unicode)
        self.encoding = encoding
        self.caseSensitive = caseSensitive
        if not self.caseSensitive:
            pattern = text.lower()
        if self.isUnicode:
            pattern = text.encode(encoding)
        super(TextPattern, self).__init__(pattern)

    def read(self, process, address, size):
        data = super(TextPattern, self).read(address, size)
        if not self.caseSensitive:
            if self.isUnicode:
                try:
                    encoding = self.encoding
                    text = data.decode(encoding, "replace")
                    text = text.lower()
                    new_data = text.encode(encoding, "replace")
                    if len(data) == len(new_data):
                        data = new_data
                    else:
                        data = data.lower()
                except Exception:
                    data = data.lower()
            else:
                data = data.lower()
        return data

    def found(self, address, size, data):
        if self.isUnicode:
            try:
                data = compat.unicode(data, self.encoding)
            except Exception:
##                traceback.print_exc()    # XXX DEBUG
                return None
        return (address, size, data)

#------------------------------------------------------------------------------

class RegExpPattern (Pattern):
    """
    Regular expression pattern.

    @type pattern: str
    @ivar pattern: Regular expression in text form.

    @type flags: int
    @ivar flags: Regular expression flags.

    @type regexp: re.compile
    @ivar regexp: Regular expression in compiled form.

    @type maxLength: int
    @ivar maxLength:
        Maximum expected length of the strings matched by this regular
        expression.

        This value will be used to calculate the required buffer size when
        doing buffered searches.

        Ideally it should be an exact value, but in some cases it's not
        possible to calculate so an upper limit should be given instead.

        If that's not possible either, C{None} should be used. That will
        cause an exception to be raised if this pattern is used in a
        buffered search.
    """

    def __init__(self, regexp, flags = 0, maxLength = None):
        """
        @type  regexp: str
        @param regexp: Regular expression string.

        @type  flags: int
        @param flags: Regular expression flags.

        @type  maxLength: int
        @param maxLength: Maximum expected length of the strings matched by
            this regular expression.

            This value will be used to calculate the required buffer size when
            doing buffered searches.

            Ideally it should be an exact value, but in some cases it's not
            possible to calculate so an upper limit should be given instead.

            If that's not possible either, C{None} should be used. That will
            cause an exception to be raised if this pattern is used in a
            buffered search.
        """
        self.pattern   = regexp
        self.flags     = flags
        self.regexp    = re.compile(regexp, flags)
        self.maxLength = maxLength

    def __len__(self):
        """
        Returns the maximum expected length of the strings matched by this
        pattern. This value is taken from the C{maxLength} argument of the
        constructor if this class.

        Ideally it should be an exact value, but in some cases it's not
        possible to calculate so an upper limit should be returned instead.

        If that's not possible either an exception must be raised.

        This value will be used to calculate the required buffer size when
        doing buffered searches.
        """
        if self.maxLength is None:
            raise NotImplementedError()
        return self.maxLength

    def find(self, buffer, pos = None):
        if not pos:   # make sure pos is an int
            pos = 0
        match = self.regexp.search(buffer, pos)
        if match:
            start, end = match.span()
            return start, end - start
        return -1, 0

#------------------------------------------------------------------------------

class HexPattern (RegExpPattern):
    """
    Hexadecimal pattern.

    Hex patterns must be in this form::
        "68 65 6c 6c 6f 20 77 6f 72 6c 64"  # "hello world"

    Spaces are optional. Capitalization of hex digits doesn't matter.
    This is exactly equivalent to the previous example::
        "68656C6C6F20776F726C64"            # "hello world"

    Wildcards are allowed, in the form of a C{?} sign in any hex digit::
        "5? 5? c3"          # pop register / pop register / ret
        "b8 ?? ?? ?? ??"    # mov eax, immediate value

    @type pattern: str
    @ivar pattern: Hexadecimal pattern.
    """

    def __new__(cls, pattern):
        """
        If the pattern is completely static (no wildcards are present) a
        L{BytePattern} is created instead. That's because searching for a
        fixed byte pattern is faster than searching for a regular expression.
        """
        if '?' not in pattern:
            return BytePattern( HexInput.hexadecimal(pattern) )
        return object.__new__(cls, pattern)

    def __init__(self, hexa):
        """
        Hex patterns must be in this form::
            "68 65 6c 6c 6f 20 77 6f 72 6c 64"  # "hello world"

        Spaces are optional. Capitalization of hex digits doesn't matter.
        This is exactly equivalent to the previous example::
            "68656C6C6F20776F726C64"            # "hello world"

        Wildcards are allowed, in the form of a C{?} sign in any hex digit::
            "5? 5? c3"          # pop register / pop register / ret
            "b8 ?? ?? ?? ??"    # mov eax, immediate value

        @type  hexa: str
        @param hexa: Pattern to search for.
        """
        maxLength = len([x for x in hexa
                            if x in "?0123456789ABCDEFabcdef"]) / 2
        super(HexPattern, self).__init__(HexInput.pattern(hexa),
                                         maxLength = maxLength)

#==============================================================================

class Search (StaticClass):
    """
    Static class to group the search functionality.

    Do not instance this class! Use its static methods instead.
    """

    # TODO: aligned searches
    # TODO: method to coalesce search results
    # TODO: search memory dumps
    # TODO: search non-ascii C strings

    @staticmethod
    def search_process(process, pattern, minAddr = None,
                                         maxAddr = None,
                                         bufferPages = None,
                                         overlapping = False):
        """
        Search for the given pattern within the process memory.

        @type  process: L{Process}
        @param process: Process to search.

        @type  pattern: L{Pattern}
        @param pattern: Pattern to search for.
            It must be an instance of a subclass of L{Pattern}.

            The following L{Pattern} subclasses are provided by WinAppDbg:
             - L{BytePattern}
             - L{TextPattern}
             - L{RegExpPattern}
             - L{HexPattern}

            You can also write your own subclass of L{Pattern} for customized
            searches.

        @type  minAddr: int
        @param minAddr: (Optional) Start the search at this memory address.

        @type  maxAddr: int
        @param maxAddr: (Optional) Stop the search at this memory address.

        @type  bufferPages: int
        @param bufferPages: (Optional) Number of memory pages to buffer when
            performing the search. Valid values are:
             - C{0} or C{None}:
               Automatically determine the required buffer size. May not give
               complete results for regular expressions that match variable
               sized strings.
             - C{> 0}: Set the buffer size, in memory pages.
             - C{< 0}: Disable buffering entirely. This may give you a little
               speed gain at the cost of an increased memory usage. If the
               target process has very large contiguous memory regions it may
               actually be slower or even fail. It's also the only way to
               guarantee complete results for regular expressions that match
               variable sized strings.

        @type  overlapping: bool
        @param overlapping: C{True} to allow overlapping results, C{False}
            otherwise.

            Overlapping results yield the maximum possible number of results.

            For example, if searching for "AAAA" within "AAAAAAAA" at address
            C{0x10000}, when overlapping is turned off the following matches
            are yielded::
                (0x10000, 4, "AAAA")
                (0x10004, 4, "AAAA")

            If overlapping is turned on, the following matches are yielded::
                (0x10000, 4, "AAAA")
                (0x10001, 4, "AAAA")
                (0x10002, 4, "AAAA")
                (0x10003, 4, "AAAA")
                (0x10004, 4, "AAAA")

            As you can see, the middle results are overlapping the last two.

        @rtype:  iterator of tuple( int, int, str )
        @return: An iterator of tuples. Each tuple contains the following:
             - The memory address where the pattern was found.
             - The size of the data that matches the pattern.
             - The data that matches the pattern.

        @raise WindowsError: An error occurred when querying or reading the
            process memory.
        """

        # Do some namespace lookups of symbols we'll be using frequently.
        MEM_COMMIT = win32.MEM_COMMIT
        PAGE_GUARD = win32.PAGE_GUARD
        page = MemoryAddresses.pageSize
        read = pattern.read
        find = pattern.find

        # Calculate the address range.
        if minAddr is None:
            minAddr = 0
        if maxAddr is None:
            maxAddr = win32.LPVOID(-1).value  # XXX HACK

        # Calculate the buffer size from the number of pages.
        if bufferPages is None:
            try:
                size = MemoryAddresses.\
                            align_address_to_page_end(len(pattern)) + page
            except NotImplementedError:
                size = None
        elif bufferPages > 0:
                size = page * (bufferPages + 1)
        else:
                size = None

        # Get the memory map of the process.
        memory_map = process.iter_memory_map(minAddr, maxAddr)

        # Perform search with buffering enabled.
        if size:

            # Loop through all memory blocks containing data.
            buffer     = "" # buffer to hold the memory data
            prev_addr  = 0  # previous memory block address
            last       = 0  # position of the last match
            delta      = 0  # delta of last read address and start of buffer
            for mbi in memory_map:

                # Skip blocks with no data to search on.
                if not mbi.has_content():
                    continue

                # Get the address and size of this block.
                address    = mbi.BaseAddress    # current address to search on
                block_size = mbi.RegionSize     # total size of the block
                if address >= maxAddr:
                    break
                end = address + block_size      # end address of the block

                # If the block is contiguous to the previous block,
                # coalesce the new data in the buffer.
                if delta and address == prev_addr:
                    buffer += read(process, address, page)

                # If not, clear the buffer and read new data.
                else:
                    buffer = read(process, address, min(size, block_size))
                    last   = 0
                    delta  = 0

                # Search for the pattern in this block.
                while 1:

                    # Yield each match of the pattern in the buffer.
                    pos, length = find(buffer, last)
                    while pos >= last:
                        match_addr = address + pos - delta
                        if minAddr <= match_addr < maxAddr:
                            result = pattern.found(
                                            match_addr, length,
                                            buffer [ pos : pos + length ] )
                            if result is not None:
                                yield result
                        if overlapping:
                            last = pos + 1
                        else:
                            last = pos + length
                        pos, length = find(buffer, last)

                    # Advance to the next page.
                    address    = address + page
                    block_size = block_size - page
                    prev_addr  = address

                    # Fix the position of the last match.
                    last = last - page
                    if last < 0:
                        last = 0

                    # Remove the first page in the buffer.
                    buffer = buffer[ page : ]
                    delta  = page

                    # If we haven't reached the end of the block yet,
                    # read the next page in the block and keep seaching.
                    if address < end:
                        buffer = buffer + read(process, address, page)

                    # Otherwise, we're done searching this block.
                    else:
                        break

        # Perform search with buffering disabled.
        else:

            # Loop through all memory blocks containing data.
            for mbi in memory_map:

                # Skip blocks with no data to search on.
                if not mbi.has_content():
                    continue

                # Get the address and size of this block.
                address    = mbi.BaseAddress
                block_size = mbi.RegionSize
                if address >= maxAddr:
                    break;

                # Read the whole memory region.
                buffer = process.read(address, block_size)

                # Search for the pattern in this region.
                pos, length = find(buffer)
                last = 0
                while pos >= last:
                    match_addr = address + pos
                    if minAddr <= match_addr < maxAddr:
                        result = pattern.found(
                                        match_addr, length,
                                        buffer [ pos : pos + length ] )
                        if result is not None:
                            yield result
                    if overlapping:
                        last = pos + 1
                    else:
                        last = pos + length
                    pos, length = find(buffer, last)

    @classmethod
    def extract_ascii_strings(cls, process, minSize = 4, maxSize = 1024):
        """
        Extract ASCII strings from the process memory.

        @type  process: L{Process}
        @param process: Process to search.

        @type  minSize: int
        @param minSize: (Optional) Minimum size of the strings to search for.

        @type  maxSize: int
        @param maxSize: (Optional) Maximum size of the strings to search for.

        @rtype:  iterator of tuple(int, int, str)
        @return: Iterator of strings extracted from the process memory.
            Each tuple contains the following:
             - The memory address where the string was found.
             - The size of the string.
             - The string.
        """
        regexp = r"[\s\w\!\@\#\$\%%\^\&\*\(\)\{\}\[\]\~\`\'\"\:\;\.\,\\\/\-\+\=\_\<\>]{%d,%d}\0" % (minSize, maxSize)
        pattern = RegExpPattern(regexp, 0, maxSize)
        return cls.search_process(process, pattern, overlapping = False)
