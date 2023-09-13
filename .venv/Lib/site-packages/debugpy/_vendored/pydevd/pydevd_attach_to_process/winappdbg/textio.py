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
Functions for text input, logging or text output.

@group Helpers:
    HexDump,
    HexInput,
    HexOutput,
    Color,
    Table,
    Logger
    DebugLog
    CrashDump
"""

__revision__ = "$Id$"

__all__ =   [
                'HexDump',
                'HexInput',
                'HexOutput',
                'Color',
                'Table',
                'CrashDump',
                'DebugLog',
                'Logger',
            ]

import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass

import re
import time
import struct
import traceback

#------------------------------------------------------------------------------

class HexInput (StaticClass):
    """
    Static functions for user input parsing.
    The counterparts for each method are in the L{HexOutput} class.
    """

    @staticmethod
    def integer(token):
        """
        Convert numeric strings into integers.

        @type  token: str
        @param token: String to parse.

        @rtype:  int
        @return: Parsed integer value.
        """
        token = token.strip()
        neg = False
        if token.startswith(compat.b('-')):
            token = token[1:]
            neg = True
        if token.startswith(compat.b('0x')):
            result = int(token, 16)     # hexadecimal
        elif token.startswith(compat.b('0b')):
            result = int(token[2:], 2)  # binary
        elif token.startswith(compat.b('0o')):
            result = int(token, 8)      # octal
        else:
            try:
                result = int(token)     # decimal
            except ValueError:
                result = int(token, 16) # hexadecimal (no "0x" prefix)
        if neg:
            result = -result
        return result

    @staticmethod
    def address(token):
        """
        Convert numeric strings into memory addresses.

        @type  token: str
        @param token: String to parse.

        @rtype:  int
        @return: Parsed integer value.
        """
        return int(token, 16)

    @staticmethod
    def hexadecimal(token):
        """
        Convert a strip of hexadecimal numbers into binary data.

        @type  token: str
        @param token: String to parse.

        @rtype:  str
        @return: Parsed string value.
        """
        token = ''.join([ c for c in token if c.isalnum() ])
        if len(token) % 2 != 0:
            raise ValueError("Missing characters in hex data")
        data = ''
        for i in compat.xrange(0, len(token), 2):
            x = token[i:i+2]
            d = int(x, 16)
            s = struct.pack('<B', d)
            data += s
        return data

    @staticmethod
    def pattern(token):
        """
        Convert an hexadecimal search pattern into a POSIX regular expression.

        For example, the following pattern::

            "B8 0? ?0 ?? ??"

        Would match the following data::

            "B8 0D F0 AD BA"    # mov eax, 0xBAADF00D

        @type  token: str
        @param token: String to parse.

        @rtype:  str
        @return: Parsed string value.
        """
        token = ''.join([ c for c in token if c == '?' or c.isalnum() ])
        if len(token) % 2 != 0:
            raise ValueError("Missing characters in hex data")
        regexp = ''
        for i in compat.xrange(0, len(token), 2):
            x = token[i:i+2]
            if x == '??':
                regexp += '.'
            elif x[0] == '?':
                f = '\\x%%.1x%s' % x[1]
                x = ''.join([ f % c for c in compat.xrange(0, 0x10) ])
                regexp = '%s[%s]' % (regexp, x)
            elif x[1] == '?':
                f = '\\x%s%%.1x' % x[0]
                x = ''.join([ f % c for c in compat.xrange(0, 0x10) ])
                regexp = '%s[%s]' % (regexp, x)
            else:
                regexp = '%s\\x%s' % (regexp, x)
        return regexp

    @staticmethod
    def is_pattern(token):
        """
        Determine if the given argument is a valid hexadecimal pattern to be
        used with L{pattern}.

        @type  token: str
        @param token: String to parse.

        @rtype:  bool
        @return:
            C{True} if it's a valid hexadecimal pattern, C{False} otherwise.
        """
        return re.match(r"^(?:[\?A-Fa-f0-9][\?A-Fa-f0-9]\s*)+$", token)

    @classmethod
    def integer_list_file(cls, filename):
        """
        Read a list of integers from a file.

        The file format is:

         - # anywhere in the line begins a comment
         - leading and trailing spaces are ignored
         - empty lines are ignored
         - integers can be specified as:
            - decimal numbers ("100" is 100)
            - hexadecimal numbers ("0x100" is 256)
            - binary numbers ("0b100" is 4)
            - octal numbers ("0100" is 64)

        @type  filename: str
        @param filename: Name of the file to read.

        @rtype:  list( int )
        @return: List of integers read from the file.
        """
        count  = 0
        result = list()
        fd     = open(filename, 'r')
        for line in fd:
            count = count + 1
            if '#' in line:
                line = line[ : line.find('#') ]
            line = line.strip()
            if line:
                try:
                    value = cls.integer(line)
                except ValueError:
                    e = sys.exc_info()[1]
                    msg = "Error in line %d of %s: %s"
                    msg = msg % (count, filename, str(e))
                    raise ValueError(msg)
                result.append(value)
        return result

    @classmethod
    def string_list_file(cls, filename):
        """
        Read a list of string values from a file.

        The file format is:

         - # anywhere in the line begins a comment
         - leading and trailing spaces are ignored
         - empty lines are ignored
         - strings cannot span over a single line

        @type  filename: str
        @param filename: Name of the file to read.

        @rtype:  list
        @return: List of integers and strings read from the file.
        """
        count  = 0
        result = list()
        fd     = open(filename, 'r')
        for line in fd:
            count = count + 1
            if '#' in line:
                line = line[ : line.find('#') ]
            line = line.strip()
            if line:
                result.append(line)
        return result

    @classmethod
    def mixed_list_file(cls, filename):
        """
        Read a list of mixed values from a file.

        The file format is:

         - # anywhere in the line begins a comment
         - leading and trailing spaces are ignored
         - empty lines are ignored
         - strings cannot span over a single line
         - integers can be specified as:
            - decimal numbers ("100" is 100)
            - hexadecimal numbers ("0x100" is 256)
            - binary numbers ("0b100" is 4)
            - octal numbers ("0100" is 64)

        @type  filename: str
        @param filename: Name of the file to read.

        @rtype:  list
        @return: List of integers and strings read from the file.
        """
        count  = 0
        result = list()
        fd     = open(filename, 'r')
        for line in fd:
            count = count + 1
            if '#' in line:
                line = line[ : line.find('#') ]
            line = line.strip()
            if line:
                try:
                    value = cls.integer(line)
                except ValueError:
                    value = line
                result.append(value)
        return result

#------------------------------------------------------------------------------

class HexOutput (StaticClass):
    """
    Static functions for user output parsing.
    The counterparts for each method are in the L{HexInput} class.

    @type integer_size: int
    @cvar integer_size: Default size in characters of an outputted integer.
        This value is platform dependent.

    @type address_size: int
    @cvar address_size: Default Number of bits of the target architecture.
        This value is platform dependent.
    """

    integer_size = (win32.SIZEOF(win32.DWORD)  * 2) + 2
    address_size = (win32.SIZEOF(win32.SIZE_T) * 2) + 2

    @classmethod
    def integer(cls, integer, bits = None):
        """
        @type  integer: int
        @param integer: Integer.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.integer_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            integer_size = cls.integer_size
        else:
            integer_size = (bits / 4) + 2
        if integer >= 0:
            return ('0x%%.%dx' % (integer_size - 2)) % integer
        return ('-0x%%.%dx' % (integer_size - 2)) % -integer

    @classmethod
    def address(cls, address, bits = None):
        """
        @type  address: int
        @param address: Memory address.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.address_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            address_size = cls.address_size
            bits = win32.bits
        else:
            address_size = (bits / 4) + 2
        if address < 0:
            address = ((2 ** bits) - 1) ^ ~address
        return ('0x%%.%dx' % (address_size - 2)) % address

    @staticmethod
    def hexadecimal(data):
        """
        Convert binary data to a string of hexadecimal numbers.

        @type  data: str
        @param data: Binary data.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        return HexDump.hexadecimal(data, separator = '')

    @classmethod
    def integer_list_file(cls, filename, values, bits = None):
        """
        Write a list of integers to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.integer_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of integers to write to the file.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.integer_size}
        """
        fd = open(filename, 'w')
        for integer in values:
            print >> fd, cls.integer(integer, bits)
        fd.close()

    @classmethod
    def string_list_file(cls, filename, values):
        """
        Write a list of strings to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.string_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of strings to write to the file.
        """
        fd = open(filename, 'w')
        for string in values:
            print >> fd, string
        fd.close()

    @classmethod
    def mixed_list_file(cls, filename, values, bits):
        """
        Write a list of mixed values to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.mixed_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of mixed values to write to the file.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.integer_size}
        """
        fd = open(filename, 'w')
        for original in values:
            try:
                parsed = cls.integer(original, bits)
            except TypeError:
                parsed = repr(original)
            print >> fd, parsed
        fd.close()

#------------------------------------------------------------------------------

class HexDump (StaticClass):
    """
    Static functions for hexadecimal dumps.

    @type integer_size: int
    @cvar integer_size: Size in characters of an outputted integer.
        This value is platform dependent.

    @type address_size: int
    @cvar address_size: Size in characters of an outputted address.
        This value is platform dependent.
    """

    integer_size = (win32.SIZEOF(win32.DWORD)  * 2)
    address_size = (win32.SIZEOF(win32.SIZE_T) * 2)

    @classmethod
    def integer(cls, integer, bits = None):
        """
        @type  integer: int
        @param integer: Integer.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.integer_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            integer_size = cls.integer_size
        else:
            integer_size = bits / 4
        return ('%%.%dX' % integer_size) % integer

    @classmethod
    def address(cls, address, bits = None):
        """
        @type  address: int
        @param address: Memory address.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            address_size = cls.address_size
            bits = win32.bits
        else:
            address_size = bits / 4
        if address < 0:
            address = ((2 ** bits) - 1) ^ ~address
        return ('%%.%dX' % address_size) % address

    @staticmethod
    def printable(data):
        """
        Replace unprintable characters with dots.

        @type  data: str
        @param data: Binary data.

        @rtype:  str
        @return: Printable text.
        """
        result = ''
        for c in data:
            if 32 < ord(c) < 128:
                result += c
            else:
                result += '.'
        return result

    @staticmethod
    def hexadecimal(data, separator = ''):
        """
        Convert binary data to a string of hexadecimal numbers.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        return separator.join( [ '%.2x' % ord(c) for c in data ] )

    @staticmethod
    def hexa_word(data, separator = ' '):
        """
        Convert binary data to a string of hexadecimal WORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each WORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        if len(data) & 1 != 0:
            data += '\0'
        return separator.join( [ '%.4x' % struct.unpack('<H', data[i:i+2])[0] \
                                           for i in compat.xrange(0, len(data), 2) ] )

    @staticmethod
    def hexa_dword(data, separator = ' '):
        """
        Convert binary data to a string of hexadecimal DWORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each DWORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        if len(data) & 3 != 0:
            data += '\0' * (4 - (len(data) & 3))
        return separator.join( [ '%.8x' % struct.unpack('<L', data[i:i+4])[0] \
                                           for i in compat.xrange(0, len(data), 4) ] )

    @staticmethod
    def hexa_qword(data, separator = ' '):
        """
        Convert binary data to a string of hexadecimal QWORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each QWORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        if len(data) & 7 != 0:
            data += '\0' * (8 - (len(data) & 7))
        return separator.join( [ '%.16x' % struct.unpack('<Q', data[i:i+8])[0]\
                                           for i in compat.xrange(0, len(data), 8) ] )

    @classmethod
    def hexline(cls, data, separator = ' ', width = None):
        """
        Dump a line of hexadecimal numbers from binary data.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.
            This value is also used for padding.

        @rtype:  str
        @return: Multiline output text.
        """
        if width is None:
            fmt = '%s  %s'
        else:
            fmt = '%%-%ds  %%-%ds' % ((len(separator)+2)*width-1, width)
        return fmt % (cls.hexadecimal(data, separator), cls.printable(data))

    @classmethod
    def hexblock(cls, data,                                    address = None,
                                                                  bits = None,
                                                             separator = ' ',
                                                                 width = 8):
        """
        Dump a block of hexadecimal numbers from binary data.
        Also show a printable text version of the data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexline, data, address, bits, width,
                 cb_kwargs = {'width' : width, 'separator' : separator})

    @classmethod
    def hexblock_cb(cls, callback, data,                        address = None,
                                                                   bits = None,
                                                                  width = 16,
                                                                cb_args = (),
                                                              cb_kwargs = {}):
        """
        Dump a block of binary data using a callback function to convert each
        line of text.

        @type  callback: function
        @param callback: Callback function to convert each line of data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address:
            (Optional) Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  cb_args: str
        @param cb_args:
            (Optional) Arguments to pass to the callback function.

        @type  cb_kwargs: str
        @param cb_kwargs:
            (Optional) Keyword arguments to pass to the callback function.

        @type  width: int
        @param width:
            (Optional) Maximum number of bytes to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        result = ''
        if address is None:
            for i in compat.xrange(0, len(data), width):
                result = '%s%s\n' % ( result, \
                             callback(data[i:i+width], *cb_args, **cb_kwargs) )
        else:
            for i in compat.xrange(0, len(data), width):
                result = '%s%s: %s\n' % (
                             result,
                             cls.address(address, bits),
                             callback(data[i:i+width], *cb_args, **cb_kwargs)
                             )
                address += width
        return result

    @classmethod
    def hexblock_byte(cls, data,                                address = None,
                                                                   bits = None,
                                                              separator = ' ',
                                                                  width = 16):
        """
        Dump a block of hexadecimal BYTEs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each BYTE.

        @type  width: int
        @param width:
            (Optional) Maximum number of BYTEs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexadecimal, data,
                               address, bits, width,
                               cb_kwargs = {'separator': separator})

    @classmethod
    def hexblock_word(cls, data,                                address = None,
                                                                   bits = None,
                                                              separator = ' ',
                                                                  width = 8):
        """
        Dump a block of hexadecimal WORDs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each WORD.

        @type  width: int
        @param width:
            (Optional) Maximum number of WORDs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexa_word, data,
                               address, bits, width * 2,
                               cb_kwargs = {'separator': separator})

    @classmethod
    def hexblock_dword(cls, data,                               address = None,
                                                                   bits = None,
                                                              separator = ' ',
                                                                  width = 4):
        """
        Dump a block of hexadecimal DWORDs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each DWORD.

        @type  width: int
        @param width:
            (Optional) Maximum number of DWORDs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexa_dword, data,
                               address, bits, width * 4,
                               cb_kwargs = {'separator': separator})

    @classmethod
    def hexblock_qword(cls, data,                               address = None,
                                                                   bits = None,
                                                              separator = ' ',
                                                                  width = 2):
        """
        Dump a block of hexadecimal QWORDs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each QWORD.

        @type  width: int
        @param width:
            (Optional) Maximum number of QWORDs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexa_qword, data,
                               address, bits, width * 8,
                               cb_kwargs = {'separator': separator})

#------------------------------------------------------------------------------

# TODO: implement an ANSI parser to simplify using colors

class Color (object):
    """
    Colored console output.
    """

    @staticmethod
    def _get_text_attributes():
        return win32.GetConsoleScreenBufferInfo().wAttributes

    @staticmethod
    def _set_text_attributes(wAttributes):
        win32.SetConsoleTextAttribute(wAttributes = wAttributes)

    #--------------------------------------------------------------------------

    @classmethod
    def can_use_colors(cls):
        """
        Determine if we can use colors.

        Colored output only works when the output is a real console, and fails
        when redirected to a file or pipe. Call this method before issuing a
        call to any other method of this class to make sure it's actually
        possible to use colors.

        @rtype:  bool
        @return: C{True} if it's possible to output text with color,
            C{False} otherwise.
        """
        try:
            cls._get_text_attributes()
            return True
        except Exception:
            return False

    @classmethod
    def reset(cls):
        "Reset the colors to the default values."
        cls._set_text_attributes(win32.FOREGROUND_GREY)

    #--------------------------------------------------------------------------

    #@classmethod
    #def underscore(cls, on = True):
    #    wAttributes = cls._get_text_attributes()
    #    if on:
    #        wAttributes |=  win32.COMMON_LVB_UNDERSCORE
    #    else:
    #        wAttributes &= ~win32.COMMON_LVB_UNDERSCORE
    #    cls._set_text_attributes(wAttributes)

    #--------------------------------------------------------------------------

    @classmethod
    def default(cls):
        "Make the current foreground color the default."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_GREY
        wAttributes &= ~win32.FOREGROUND_INTENSITY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def light(cls):
        "Make the current foreground color light."
        wAttributes = cls._get_text_attributes()
        wAttributes |= win32.FOREGROUND_INTENSITY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def dark(cls):
        "Make the current foreground color dark."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_INTENSITY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def black(cls):
        "Make the text foreground color black."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        #wAttributes |=  win32.FOREGROUND_BLACK
        cls._set_text_attributes(wAttributes)

    @classmethod
    def white(cls):
        "Make the text foreground color white."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_GREY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def red(cls):
        "Make the text foreground color red."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_RED
        cls._set_text_attributes(wAttributes)

    @classmethod
    def green(cls):
        "Make the text foreground color green."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_GREEN
        cls._set_text_attributes(wAttributes)

    @classmethod
    def blue(cls):
        "Make the text foreground color blue."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_BLUE
        cls._set_text_attributes(wAttributes)

    @classmethod
    def cyan(cls):
        "Make the text foreground color cyan."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_CYAN
        cls._set_text_attributes(wAttributes)

    @classmethod
    def magenta(cls):
        "Make the text foreground color magenta."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_MAGENTA
        cls._set_text_attributes(wAttributes)

    @classmethod
    def yellow(cls):
        "Make the text foreground color yellow."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.FOREGROUND_MASK
        wAttributes |=  win32.FOREGROUND_YELLOW
        cls._set_text_attributes(wAttributes)

    #--------------------------------------------------------------------------

    @classmethod
    def bk_default(cls):
        "Make the current background color the default."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        #wAttributes |= win32.BACKGROUND_BLACK
        wAttributes &= ~win32.BACKGROUND_INTENSITY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_light(cls):
        "Make the current background color light."
        wAttributes = cls._get_text_attributes()
        wAttributes |= win32.BACKGROUND_INTENSITY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_dark(cls):
        "Make the current background color dark."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_INTENSITY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_black(cls):
        "Make the text background color black."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        #wAttributes |= win32.BACKGROUND_BLACK
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_white(cls):
        "Make the text background color white."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        wAttributes |=  win32.BACKGROUND_GREY
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_red(cls):
        "Make the text background color red."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        wAttributes |=  win32.BACKGROUND_RED
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_green(cls):
        "Make the text background color green."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        wAttributes |=  win32.BACKGROUND_GREEN
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_blue(cls):
        "Make the text background color blue."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        wAttributes |=  win32.BACKGROUND_BLUE
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_cyan(cls):
        "Make the text background color cyan."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        wAttributes |=  win32.BACKGROUND_CYAN
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_magenta(cls):
        "Make the text background color magenta."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        wAttributes |=  win32.BACKGROUND_MAGENTA
        cls._set_text_attributes(wAttributes)

    @classmethod
    def bk_yellow(cls):
        "Make the text background color yellow."
        wAttributes = cls._get_text_attributes()
        wAttributes &= ~win32.BACKGROUND_MASK
        wAttributes |=  win32.BACKGROUND_YELLOW
        cls._set_text_attributes(wAttributes)

#------------------------------------------------------------------------------

# TODO: another class for ASCII boxes

class Table (object):
    """
    Text based table. The number of columns and the width of each column
    is automatically calculated.
    """

    def __init__(self, sep = ' '):
        """
        @type  sep: str
        @param sep: Separator between cells in each row.
        """
        self.__cols  = list()
        self.__width = list()
        self.__sep   = sep

    def addRow(self, *row):
        """
        Add a row to the table. All items are converted to strings.

        @type    row: tuple
        @keyword row: Each argument is a cell in the table.
        """
        row     = [ str(item) for item in row ]
        len_row = [ len(item) for item in row ]
        width   = self.__width
        len_old = len(width)
        len_new = len(row)
        known   = min(len_old, len_new)
        missing = len_new - len_old
        if missing > 0:
            width.extend( len_row[ -missing : ] )
        elif missing < 0:
            len_row.extend( [0] * (-missing) )
        self.__width = [ max( width[i], len_row[i] ) for i in compat.xrange(len(len_row)) ]
        self.__cols.append(row)

    def justify(self, column, direction):
        """
        Make the text in a column left or right justified.

        @type  column: int
        @param column: Index of the column.

        @type  direction: int
        @param direction:
            C{-1} to justify left,
            C{1} to justify right.

        @raise IndexError: Bad column index.
        @raise ValueError: Bad direction value.
        """
        if direction == -1:
            self.__width[column] =   abs(self.__width[column])
        elif direction == 1:
            self.__width[column] = - abs(self.__width[column])
        else:
            raise ValueError("Bad direction value.")

    def getWidth(self):
        """
        Get the width of the text output for the table.

        @rtype:  int
        @return: Width in characters for the text output,
            including the newline character.
        """
        width = 0
        if self.__width:
            width = sum( abs(x) for x in self.__width )
            width = width + len(self.__width) * len(self.__sep) + 1
        return width

    def getOutput(self):
        """
        Get the text output for the table.

        @rtype:  str
        @return: Text output.
        """
        return '%s\n' % '\n'.join( self.yieldOutput() )

    def yieldOutput(self):
        """
        Generate the text output for the table.

        @rtype:  generator of str
        @return: Text output.
        """
        width = self.__width
        if width:
            num_cols = len(width)
            fmt = ['%%%ds' % -w for w in width]
            if width[-1] > 0:
                fmt[-1] = '%s'
            fmt = self.__sep.join(fmt)
            for row in self.__cols:
                row.extend( [''] * (num_cols - len(row)) )
                yield fmt % tuple(row)

    def show(self):
        """
        Print the text output for the table.
        """
        print(self.getOutput())

#------------------------------------------------------------------------------

class CrashDump (StaticClass):
    """
    Static functions for crash dumps.

    @type reg_template: str
    @cvar reg_template: Template for the L{dump_registers} method.
    """

    # Templates for the dump_registers method.
    reg_template = {
        win32.ARCH_I386 : (
            'eax=%(Eax).8x ebx=%(Ebx).8x ecx=%(Ecx).8x edx=%(Edx).8x esi=%(Esi).8x edi=%(Edi).8x\n'
            'eip=%(Eip).8x esp=%(Esp).8x ebp=%(Ebp).8x %(efl_dump)s\n'
            'cs=%(SegCs).4x  ss=%(SegSs).4x  ds=%(SegDs).4x  es=%(SegEs).4x  fs=%(SegFs).4x  gs=%(SegGs).4x             efl=%(EFlags).8x\n'
            ),
        win32.ARCH_AMD64 : (
            'rax=%(Rax).16x rbx=%(Rbx).16x rcx=%(Rcx).16x\n'
            'rdx=%(Rdx).16x rsi=%(Rsi).16x rdi=%(Rdi).16x\n'
            'rip=%(Rip).16x rsp=%(Rsp).16x rbp=%(Rbp).16x\n'
            ' r8=%(R8).16x  r9=%(R9).16x r10=%(R10).16x\n'
            'r11=%(R11).16x r12=%(R12).16x r13=%(R13).16x\n'
            'r14=%(R14).16x r15=%(R15).16x\n'
            '%(efl_dump)s\n'
            'cs=%(SegCs).4x  ss=%(SegSs).4x  ds=%(SegDs).4x  es=%(SegEs).4x  fs=%(SegFs).4x  gs=%(SegGs).4x             efl=%(EFlags).8x\n'
            ),
    }

    @staticmethod
    def dump_flags(efl):
        """
        Dump the x86 processor flags.
        The output mimics that of the WinDBG debugger.
        Used by L{dump_registers}.

        @type  efl: int
        @param efl: Value of the eFlags register.

        @rtype:  str
        @return: Text suitable for logging.
        """
        if efl is None:
            return ''
        efl_dump = 'iopl=%1d' % ((efl & 0x3000) >> 12)
        if efl & 0x100000:
            efl_dump += ' vip'
        else:
            efl_dump += '    '
        if efl & 0x80000:
            efl_dump += ' vif'
        else:
            efl_dump += '    '
        # 0x20000 ???
        if efl & 0x800:
            efl_dump += ' ov'       # Overflow
        else:
            efl_dump += ' no'       # No overflow
        if efl & 0x400:
            efl_dump += ' dn'       # Downwards
        else:
            efl_dump += ' up'       # Upwards
        if efl & 0x200:
            efl_dump += ' ei'       # Enable interrupts
        else:
            efl_dump += ' di'       # Disable interrupts
        # 0x100 trap flag
        if efl & 0x80:
            efl_dump += ' ng'       # Negative
        else:
            efl_dump += ' pl'       # Positive
        if efl & 0x40:
            efl_dump += ' zr'       # Zero
        else:
            efl_dump += ' nz'       # Nonzero
        if efl & 0x10:
            efl_dump += ' ac'       # Auxiliary carry
        else:
            efl_dump += ' na'       # No auxiliary carry
        # 0x8 ???
        if efl & 0x4:
            efl_dump += ' pe'       # Parity odd
        else:
            efl_dump += ' po'       # Parity even
        # 0x2 ???
        if efl & 0x1:
            efl_dump += ' cy'       # Carry
        else:
            efl_dump += ' nc'       # No carry
        return efl_dump

    @classmethod
    def dump_registers(cls, registers, arch = None):
        """
        Dump the x86/x64 processor register values.
        The output mimics that of the WinDBG debugger.

        @type  registers: dict( str S{->} int )
        @param registers: Dictionary mapping register names to their values.

        @type  arch: str
        @param arch: Architecture of the machine whose registers were dumped.
            Defaults to the current architecture.
            Currently only the following architectures are supported:
             - L{win32.ARCH_I386}
             - L{win32.ARCH_AMD64}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if registers is None:
            return ''
        if arch is None:
            if 'Eax' in registers:
                arch = win32.ARCH_I386
            elif 'Rax' in registers:
                arch = win32.ARCH_AMD64
            else:
                arch = 'Unknown'
        if arch not in cls.reg_template:
            msg = "Don't know how to dump the registers for architecture: %s"
            raise NotImplementedError(msg % arch)
        registers = registers.copy()
        registers['efl_dump'] = cls.dump_flags( registers['EFlags'] )
        return cls.reg_template[arch] % registers

    @staticmethod
    def dump_registers_peek(registers, data, separator = ' ', width = 16):
        """
        Dump data pointed to by the given registers, if any.

        @type  registers: dict( str S{->} int )
        @param registers: Dictionary mapping register names to their values.
            This value is returned by L{Thread.get_context}.

        @type  data: dict( str S{->} str )
        @param data: Dictionary mapping register names to the data they point to.
            This value is returned by L{Thread.peek_pointers_in_registers}.

        @rtype:  str
        @return: Text suitable for logging.
        """
        if None in (registers, data):
            return ''
        names = compat.keys(data)
        names.sort()
        result = ''
        for reg_name in names:
            tag     = reg_name.lower()
            dumped  = HexDump.hexline(data[reg_name], separator, width)
            result += '%s -> %s\n' % (tag, dumped)
        return result

    @staticmethod
    def dump_data_peek(data,                                      base = 0,
                                                             separator = ' ',
                                                                 width = 16,
                                                                  bits = None):
        """
        Dump data from pointers guessed within the given binary data.

        @type  data: str
        @param data: Dictionary mapping offsets to the data they point to.

        @type  base: int
        @param base: Base offset.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if data is None:
            return ''
        pointers = compat.keys(data)
        pointers.sort()
        result = ''
        for offset in pointers:
            dumped  = HexDump.hexline(data[offset], separator, width)
            address = HexDump.address(base + offset, bits)
            result += '%s -> %s\n' % (address, dumped)
        return result

    @staticmethod
    def dump_stack_peek(data, separator = ' ', width = 16, arch = None):
        """
        Dump data from pointers guessed within the given stack dump.

        @type  data: str
        @param data: Dictionary mapping stack offsets to the data they point to.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.
            This value is also used for padding.

        @type  arch: str
        @param arch: Architecture of the machine whose registers were dumped.
            Defaults to the current architecture.

        @rtype:  str
        @return: Text suitable for logging.
        """
        if data is None:
            return ''
        if arch is None:
            arch = win32.arch
        pointers = compat.keys(data)
        pointers.sort()
        result = ''
        if pointers:
            if arch == win32.ARCH_I386:
                spreg = 'esp'
            elif arch == win32.ARCH_AMD64:
                spreg = 'rsp'
            else:
                spreg = 'STACK' # just a generic tag
            tag_fmt = '[%s+0x%%.%dx]' % (spreg, len( '%x' % pointers[-1] ) )
            for offset in pointers:
                dumped  = HexDump.hexline(data[offset], separator, width)
                tag     = tag_fmt % offset
                result += '%s -> %s\n' % (tag, dumped)
        return result

    @staticmethod
    def dump_stack_trace(stack_trace, bits = None):
        """
        Dump a stack trace, as returned by L{Thread.get_stack_trace} with the
        C{bUseLabels} parameter set to C{False}.

        @type  stack_trace: list( int, int, str )
        @param stack_trace: Stack trace as a list of tuples of
            ( return address, frame pointer, module filename )

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not stack_trace:
            return ''
        table = Table()
        table.addRow('Frame', 'Origin', 'Module')
        for (fp, ra, mod) in stack_trace:
            fp_d = HexDump.address(fp, bits)
            ra_d = HexDump.address(ra, bits)
            table.addRow(fp_d, ra_d, mod)
        return table.getOutput()

    @staticmethod
    def dump_stack_trace_with_labels(stack_trace, bits = None):
        """
        Dump a stack trace,
        as returned by L{Thread.get_stack_trace_with_labels}.

        @type  stack_trace: list( int, int, str )
        @param stack_trace: Stack trace as a list of tuples of
            ( return address, frame pointer, module filename )

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not stack_trace:
            return ''
        table = Table()
        table.addRow('Frame', 'Origin')
        for (fp, label) in stack_trace:
            table.addRow( HexDump.address(fp, bits), label )
        return table.getOutput()

    # TODO
    # + Instead of a star when EIP points to, it would be better to show
    # any register value (or other values like the exception address) that
    # points to a location in the dissassembled code.
    # + It'd be very useful to show some labels here.
    # + It'd be very useful to show register contents for code at EIP
    @staticmethod
    def dump_code(disassembly,                                      pc = None,
                                                            bLowercase = True,
                                                                  bits = None):
        """
        Dump a disassembly. Optionally mark where the program counter is.

        @type  disassembly: list of tuple( int, int, str, str )
        @param disassembly: Disassembly dump as returned by
            L{Process.disassemble} or L{Thread.disassemble_around_pc}.

        @type  pc: int
        @param pc: (Optional) Program counter.

        @type  bLowercase: bool
        @param bLowercase: (Optional) If C{True} convert the code to lowercase.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not disassembly:
            return ''
        table = Table(sep = ' | ')
        for (addr, size, code, dump) in disassembly:
            if bLowercase:
                code = code.lower()
            if addr == pc:
                addr = ' * %s' % HexDump.address(addr, bits)
            else:
                addr = '   %s' % HexDump.address(addr, bits)
            table.addRow(addr, dump, code)
        table.justify(1, 1)
        return table.getOutput()

    @staticmethod
    def dump_code_line(disassembly_line,                  bShowAddress = True,
                                                             bShowDump = True,
                                                            bLowercase = True,
                                                           dwDumpWidth = None,
                                                           dwCodeWidth = None,
                                                                  bits = None):
        """
        Dump a single line of code. To dump a block of code use L{dump_code}.

        @type  disassembly_line: tuple( int, int, str, str )
        @param disassembly_line: Single item of the list returned by
            L{Process.disassemble} or L{Thread.disassemble_around_pc}.

        @type  bShowAddress: bool
        @param bShowAddress: (Optional) If C{True} show the memory address.

        @type  bShowDump: bool
        @param bShowDump: (Optional) If C{True} show the hexadecimal dump.

        @type  bLowercase: bool
        @param bLowercase: (Optional) If C{True} convert the code to lowercase.

        @type  dwDumpWidth: int or None
        @param dwDumpWidth: (Optional) Width in characters of the hex dump.

        @type  dwCodeWidth: int or None
        @param dwCodeWidth: (Optional) Width in characters of the code.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if bits is None:
            address_size = HexDump.address_size
        else:
            address_size = bits / 4
        (addr, size, code, dump) = disassembly_line
        dump = dump.replace(' ', '')
        result = list()
        fmt = ''
        if bShowAddress:
            result.append( HexDump.address(addr, bits) )
            fmt += '%%%ds:' % address_size
        if bShowDump:
            result.append(dump)
            if dwDumpWidth:
                fmt += ' %%-%ds' % dwDumpWidth
            else:
                fmt += ' %s'
        if bLowercase:
            code = code.lower()
        result.append(code)
        if dwCodeWidth:
            fmt += ' %%-%ds' % dwCodeWidth
        else:
            fmt += ' %s'
        return fmt % tuple(result)

    @staticmethod
    def dump_memory_map(memoryMap, mappedFilenames = None, bits = None):
        """
        Dump the memory map of a process. Optionally show the filenames for
        memory mapped files as well.

        @type  memoryMap: list( L{win32.MemoryBasicInformation} )
        @param memoryMap: Memory map returned by L{Process.get_memory_map}.

        @type  mappedFilenames: dict( int S{->} str )
        @param mappedFilenames: (Optional) Memory mapped filenames
            returned by L{Process.get_mapped_filenames}.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not memoryMap:
            return ''

        table = Table()
        if mappedFilenames:
            table.addRow("Address", "Size", "State", "Access", "Type", "File")
        else:
            table.addRow("Address", "Size", "State", "Access", "Type")

        # For each memory block in the map...
        for mbi in memoryMap:

            # Address and size of memory block.
            BaseAddress = HexDump.address(mbi.BaseAddress, bits)
            RegionSize  = HexDump.address(mbi.RegionSize,  bits)

            # State (free or allocated).
            mbiState = mbi.State
            if   mbiState == win32.MEM_RESERVE:
                State   = "Reserved"
            elif mbiState == win32.MEM_COMMIT:
                State   = "Commited"
            elif mbiState == win32.MEM_FREE:
                State   = "Free"
            else:
                State   = "Unknown"

            # Page protection bits (R/W/X/G).
            if mbiState != win32.MEM_COMMIT:
                Protect = ""
            else:
                mbiProtect = mbi.Protect
                if   mbiProtect & win32.PAGE_NOACCESS:
                    Protect = "--- "
                elif mbiProtect & win32.PAGE_READONLY:
                    Protect = "R-- "
                elif mbiProtect & win32.PAGE_READWRITE:
                    Protect = "RW- "
                elif mbiProtect & win32.PAGE_WRITECOPY:
                    Protect = "RC- "
                elif mbiProtect & win32.PAGE_EXECUTE:
                    Protect = "--X "
                elif mbiProtect & win32.PAGE_EXECUTE_READ:
                    Protect = "R-X "
                elif mbiProtect & win32.PAGE_EXECUTE_READWRITE:
                    Protect = "RWX "
                elif mbiProtect & win32.PAGE_EXECUTE_WRITECOPY:
                    Protect = "RCX "
                else:
                    Protect = "??? "
                if   mbiProtect & win32.PAGE_GUARD:
                    Protect += "G"
                else:
                    Protect += "-"
                if   mbiProtect & win32.PAGE_NOCACHE:
                    Protect += "N"
                else:
                    Protect += "-"
                if   mbiProtect & win32.PAGE_WRITECOMBINE:
                    Protect += "W"
                else:
                    Protect += "-"

            # Type (file mapping, executable image, or private memory).
            mbiType = mbi.Type
            if   mbiType == win32.MEM_IMAGE:
                Type    = "Image"
            elif mbiType == win32.MEM_MAPPED:
                Type    = "Mapped"
            elif mbiType == win32.MEM_PRIVATE:
                Type    = "Private"
            elif mbiType == 0:
                Type    = ""
            else:
                Type    = "Unknown"

            # Output a row in the table.
            if mappedFilenames:
                FileName = mappedFilenames.get(mbi.BaseAddress, '')
                table.addRow( BaseAddress, RegionSize, State, Protect, Type, FileName )
            else:
                table.addRow( BaseAddress, RegionSize, State, Protect, Type )

        # Return the table output.
        return table.getOutput()

#------------------------------------------------------------------------------

class DebugLog (StaticClass):
    'Static functions for debug logging.'

    @staticmethod
    def log_text(text):
        """
        Log lines of text, inserting a timestamp.

        @type  text: str
        @param text: Text to log.

        @rtype:  str
        @return: Log line.
        """
        if text.endswith('\n'):
            text = text[:-len('\n')]
        #text  = text.replace('\n', '\n\t\t')           # text CSV
        ltime = time.strftime("%X")
        msecs = (time.time() % 1) * 1000
        return '[%s.%04d] %s' % (ltime, msecs, text)
        #return '[%s.%04d]\t%s' % (ltime, msecs, text)  # text CSV

    @classmethod
    def log_event(cls, event, text = None):
        """
        Log lines of text associated with a debug event.

        @type  event: L{Event}
        @param event: Event object.

        @type  text: str
        @param text: (Optional) Text to log. If no text is provided the default
            is to show a description of the event itself.

        @rtype:  str
        @return: Log line.
        """
        if not text:
            if event.get_event_code() == win32.EXCEPTION_DEBUG_EVENT:
                what = event.get_exception_description()
                if event.is_first_chance():
                    what = '%s (first chance)' % what
                else:
                    what = '%s (second chance)' % what
                try:
                    address = event.get_fault_address()
                except NotImplementedError:
                    address = event.get_exception_address()
            else:
                what    = event.get_event_name()
                address = event.get_thread().get_pc()
            process = event.get_process()
            label = process.get_label_at_address(address)
            address = HexDump.address(address, process.get_bits())
            if label:
                where = '%s (%s)' % (address, label)
            else:
                where = address
            text = '%s at %s' % (what, where)
        text = 'pid %d tid %d: %s' % (event.get_pid(), event.get_tid(), text)
        #text = 'pid %d tid %d:\t%s' % (event.get_pid(), event.get_tid(), text)     # text CSV
        return cls.log_text(text)

#------------------------------------------------------------------------------

class Logger(object):
    """
    Logs text to standard output and/or a text file.

    @type logfile: str or None
    @ivar logfile: Append messages to this text file.

    @type verbose: bool
    @ivar verbose: C{True} to print messages to standard output.

    @type fd: file
    @ivar fd: File object where log messages are printed to.
        C{None} if no log file is used.
    """

    def __init__(self, logfile = None, verbose = True):
        """
        @type  logfile: str or None
        @param logfile: Append messages to this text file.

        @type  verbose: bool
        @param verbose: C{True} to print messages to standard output.
        """
        self.verbose = verbose
        self.logfile = logfile
        if self.logfile:
            self.fd = open(self.logfile, 'a+')

    def __logfile_error(self, e):
        """
        Shows an error message to standard error
        if the log file can't be written to.

        Used internally.

        @type  e: Exception
        @param e: Exception raised when trying to write to the log file.
        """
        from sys import stderr
        msg = "Warning, error writing log file %s: %s\n"
        msg = msg % (self.logfile, str(e))
        stderr.write(DebugLog.log_text(msg))
        self.logfile = None
        self.fd      = None

    def __do_log(self, text):
        """
        Writes the given text verbatim into the log file (if any)
        and/or standard input (if the verbose flag is turned on).

        Used internally.

        @type  text: str
        @param text: Text to print.
        """
        if isinstance(text, compat.unicode):
            text = text.encode('cp1252')
        if self.verbose:
            print(text)
        if self.logfile:
            try:
                self.fd.writelines('%s\n' % text)
            except IOError:
                e = sys.exc_info()[1]
                self.__logfile_error(e)

    def log_text(self, text):
        """
        Log lines of text, inserting a timestamp.

        @type  text: str
        @param text: Text to log.
        """
        self.__do_log( DebugLog.log_text(text) )

    def log_event(self, event, text = None):
        """
        Log lines of text associated with a debug event.

        @type  event: L{Event}
        @param event: Event object.

        @type  text: str
        @param text: (Optional) Text to log. If no text is provided the default
            is to show a description of the event itself.
        """
        self.__do_log( DebugLog.log_event(event, text) )

    def log_exc(self):
        """
        Log lines of text associated with the last Python exception.
        """
        self.__do_log( 'Exception raised: %s' % traceback.format_exc() )

    def is_enabled(self):
        """
        Determines if the logger will actually print anything when the log_*
        methods are called.

        This may save some processing if the log text requires a lengthy
        calculation to prepare. If no log file is set and stdout logging
        is disabled, there's no point in preparing a log text that won't
        be shown to anyone.

        @rtype:  bool
        @return: C{True} if a log file was set and/or standard output logging
            is enabled, or C{False} otherwise.
        """
        return self.verbose or self.logfile
