#!/usr/bin/env python
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
Common definitions.
"""

# TODO
# + add TCHAR and related types?

__revision__ = "$Id$"

import ctypes
import functools
from winappdbg import compat

#------------------------------------------------------------------------------

# Some stuff from ctypes we'll be using very frequently.
addressof   = ctypes.addressof
sizeof      = ctypes.sizeof
SIZEOF      = ctypes.sizeof
POINTER     = ctypes.POINTER
Structure   = ctypes.Structure
Union       = ctypes.Union
WINFUNCTYPE = ctypes.WINFUNCTYPE
windll      = ctypes.windll

# The IronPython implementation of byref() was giving me problems,
# so I'm replacing it with the slower pointer() function.
try:
    ctypes.c_void_p(ctypes.byref(ctypes.c_char()))  # this fails in IronPython
    byref = ctypes.byref
except TypeError:
    byref = ctypes.pointer

# XXX DEBUG
# The following code can be enabled to make the Win32 API wrappers log to
# standard output the dll and function names, the parameter values and the
# return value for each call.

##WIN32_VERBOSE_MODE = True
WIN32_VERBOSE_MODE = False

if WIN32_VERBOSE_MODE:

    class WinDllHook(object):
        def __getattr__(self, name):
            if name.startswith('_'):
                return object.__getattr__(self, name)
            return WinFuncHook(name)

    class WinFuncHook(object):
        def __init__(self, name):
            self.__name = name

        def __getattr__(self, name):
            if name.startswith('_'):
                return object.__getattr__(self, name)
            return WinCallHook(self.__name, name)

    class WinCallHook(object):
        def __init__(self, dllname, funcname):
            self.__dllname = dllname
            self.__funcname = funcname
            self.__func = getattr(getattr(ctypes.windll, dllname), funcname)

        def __copy_attribute(self, attribute):
            try:
                value = getattr(self, attribute)
                setattr(self.__func, attribute, value)
            except AttributeError:
                try:
                    delattr(self.__func, attribute)
                except AttributeError:
                    pass

        def __call__(self, *argv):
            self.__copy_attribute('argtypes')
            self.__copy_attribute('restype')
            self.__copy_attribute('errcheck')
            print("-"*10)
            print("%s ! %s %r" % (self.__dllname, self.__funcname, argv))
            retval = self.__func(*argv)
            print("== %r" % (retval,))
            return retval

    windll = WinDllHook()

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

def RaiseIfZero(result, func = None, arguments = ()):
    """
    Error checking for most Win32 API calls.

    The function is assumed to return an integer, which is C{0} on error.
    In that case the C{WindowsError} exception is raised.
    """
    if not result:
        raise ctypes.WinError()
    return result

def RaiseIfNotZero(result, func = None, arguments = ()):
    """
    Error checking for some odd Win32 API calls.

    The function is assumed to return an integer, which is zero on success.
    If the return value is nonzero the C{WindowsError} exception is raised.

    This is mostly useful for free() like functions, where the return value is
    the pointer to the memory block on failure or a C{NULL} pointer on success.
    """
    if result:
        raise ctypes.WinError()
    return result

def RaiseIfNotErrorSuccess(result, func = None, arguments = ()):
    """
    Error checking for Win32 Registry API calls.

    The function is assumed to return a Win32 error code. If the code is not
    C{ERROR_SUCCESS} then a C{WindowsError} exception is raised.
    """
    if result != ERROR_SUCCESS:
        raise ctypes.WinError(result)
    return result

class GuessStringType(object):
    """
    Decorator that guesses the correct version (A or W) to call
    based on the types of the strings passed as parameters.

    Calls the B{ANSI} version if the only string types are ANSI.

    Calls the B{Unicode} version if Unicode or mixed string types are passed.

    The default if no string arguments are passed depends on the value of the
    L{t_default} class variable.

    @type fn_ansi: function
    @ivar fn_ansi: ANSI version of the API function to call.
    @type fn_unicode: function
    @ivar fn_unicode: Unicode (wide) version of the API function to call.

    @type t_default: type
    @cvar t_default: Default string type to use.
        Possible values are:
         - type('') for ANSI
         - type(u'') for Unicode
    """

    # ANSI and Unicode types
    t_ansi    = type('')
    t_unicode = type(u'')

    # Default is ANSI for Python 2.x
    t_default = t_ansi

    def __init__(self, fn_ansi, fn_unicode):
        """
        @type  fn_ansi: function
        @param fn_ansi: ANSI version of the API function to call.
        @type  fn_unicode: function
        @param fn_unicode: Unicode (wide) version of the API function to call.
        """
        self.fn_ansi    = fn_ansi
        self.fn_unicode = fn_unicode

        # Copy the wrapped function attributes.
        try:
            self.__name__ = self.fn_ansi.__name__[:-1]  # remove the A or W
        except AttributeError:
            pass
        try:
            self.__module__ = self.fn_ansi.__module__
        except AttributeError:
            pass
        try:
            self.__doc__ = self.fn_ansi.__doc__
        except AttributeError:
            pass

    def __call__(self, *argv, **argd):

        # Shortcut to self.t_ansi
        t_ansi    = self.t_ansi

        # Get the types of all arguments for the function
        v_types   = [ type(item) for item in argv ]
        v_types.extend( [ type(value) for (key, value) in compat.iteritems(argd) ] )

        # Get the appropriate function for the default type
        if self.t_default == t_ansi:
            fn = self.fn_ansi
        else:
            fn = self.fn_unicode

        # If at least one argument is a Unicode string...
        if self.t_unicode in v_types:

            # If al least one argument is an ANSI string,
            # convert all ANSI strings to Unicode
            if t_ansi in v_types:
                argv = list(argv)
                for index in compat.xrange(len(argv)):
                    if v_types[index] == t_ansi:
                        argv[index] = compat.unicode(argv[index])
                for (key, value) in argd.items():
                    if type(value) == t_ansi:
                        argd[key] = compat.unicode(value)

            # Use the W version
            fn = self.fn_unicode

        # If at least one argument is an ANSI string,
        # but there are no Unicode strings...
        elif t_ansi in v_types:

            # Use the A version
            fn = self.fn_ansi

        # Call the function and return the result
        return fn(*argv, **argd)

class DefaultStringType(object):
    """
    Decorator that uses the default version (A or W) to call
    based on the configuration of the L{GuessStringType} decorator.

    @see: L{GuessStringType.t_default}

    @type fn_ansi: function
    @ivar fn_ansi: ANSI version of the API function to call.
    @type fn_unicode: function
    @ivar fn_unicode: Unicode (wide) version of the API function to call.
    """

    def __init__(self, fn_ansi, fn_unicode):
        """
        @type  fn_ansi: function
        @param fn_ansi: ANSI version of the API function to call.
        @type  fn_unicode: function
        @param fn_unicode: Unicode (wide) version of the API function to call.
        """
        self.fn_ansi    = fn_ansi
        self.fn_unicode = fn_unicode

        # Copy the wrapped function attributes.
        try:
            self.__name__ = self.fn_ansi.__name__[:-1]  # remove the A or W
        except AttributeError:
            pass
        try:
            self.__module__ = self.fn_ansi.__module__
        except AttributeError:
            pass
        try:
            self.__doc__ = self.fn_ansi.__doc__
        except AttributeError:
            pass

    def __call__(self, *argv, **argd):

        # Get the appropriate function based on the default.
        if GuessStringType.t_default == GuessStringType.t_ansi:
            fn = self.fn_ansi
        else:
            fn = self.fn_unicode

        # Call the function and return the result
        return fn(*argv, **argd)

def MakeANSIVersion(fn):
    """
    Decorator that generates an ANSI version of a Unicode (wide) only API call.

    @type  fn: callable
    @param fn: Unicode (wide) version of the API function to call.
    """
    @functools.wraps(fn)
    def wrapper(*argv, **argd):
        t_ansi    = GuessStringType.t_ansi
        t_unicode = GuessStringType.t_unicode
        v_types   = [ type(item) for item in argv ]
        v_types.extend( [ type(value) for (key, value) in compat.iteritems(argd) ] )
        if t_ansi in v_types:
            argv = list(argv)
            for index in compat.xrange(len(argv)):
                if v_types[index] == t_ansi:
                    argv[index] = t_unicode(argv[index])
            for key, value in argd.items():
                if type(value) == t_ansi:
                    argd[key] = t_unicode(value)
        return fn(*argv, **argd)
    return wrapper

def MakeWideVersion(fn):
    """
    Decorator that generates a Unicode (wide) version of an ANSI only API call.

    @type  fn: callable
    @param fn: ANSI version of the API function to call.
    """
    @functools.wraps(fn)
    def wrapper(*argv, **argd):
        t_ansi    = GuessStringType.t_ansi
        t_unicode = GuessStringType.t_unicode
        v_types   = [ type(item) for item in argv ]
        v_types.extend( [ type(value) for (key, value) in compat.iteritems(argd) ] )
        if t_unicode in v_types:
            argv = list(argv)
            for index in compat.xrange(len(argv)):
                if v_types[index] == t_unicode:
                    argv[index] = t_ansi(argv[index])
            for key, value in argd.items():
                if type(value) == t_unicode:
                    argd[key] = t_ansi(value)
        return fn(*argv, **argd)
    return wrapper

#--- Types --------------------------------------------------------------------
# http://msdn.microsoft.com/en-us/library/aa383751(v=vs.85).aspx

# Map of basic C types to Win32 types
LPVOID      = ctypes.c_void_p
CHAR        = ctypes.c_char
WCHAR       = ctypes.c_wchar
BYTE        = ctypes.c_ubyte
SBYTE       = ctypes.c_byte
WORD        = ctypes.c_uint16
SWORD       = ctypes.c_int16
DWORD       = ctypes.c_uint32
SDWORD      = ctypes.c_int32
QWORD       = ctypes.c_uint64
SQWORD      = ctypes.c_int64
SHORT       = ctypes.c_short
USHORT      = ctypes.c_ushort
INT         = ctypes.c_int
UINT        = ctypes.c_uint
LONG        = ctypes.c_long
ULONG       = ctypes.c_ulong
LONGLONG    = ctypes.c_int64        # c_longlong
ULONGLONG   = ctypes.c_uint64       # c_ulonglong
LPSTR       = ctypes.c_char_p
LPWSTR      = ctypes.c_wchar_p
INT8        = ctypes.c_int8
INT16       = ctypes.c_int16
INT32       = ctypes.c_int32
INT64       = ctypes.c_int64
UINT8       = ctypes.c_uint8
UINT16      = ctypes.c_uint16
UINT32      = ctypes.c_uint32
UINT64      = ctypes.c_uint64
LONG32      = ctypes.c_int32
LONG64      = ctypes.c_int64
ULONG32     = ctypes.c_uint32
ULONG64     = ctypes.c_uint64
DWORD32     = ctypes.c_uint32
DWORD64     = ctypes.c_uint64
BOOL        = ctypes.c_int
FLOAT       = ctypes.c_float

# Map size_t to SIZE_T
try:
    SIZE_T  = ctypes.c_size_t
    SSIZE_T = ctypes.c_ssize_t
except AttributeError:
    # Size of a pointer
    SIZE_T  = {1:BYTE, 2:WORD, 4:DWORD, 8:QWORD}[sizeof(LPVOID)]
    SSIZE_T = {1:SBYTE, 2:SWORD, 4:SDWORD, 8:SQWORD}[sizeof(LPVOID)]
PSIZE_T     = POINTER(SIZE_T)

# Not really pointers but pointer-sized integers
DWORD_PTR   = SIZE_T
ULONG_PTR   = SIZE_T
LONG_PTR    = SIZE_T

# Other Win32 types, more may be added as needed
PVOID       = LPVOID
PPVOID      = POINTER(PVOID)
PSTR        = LPSTR
PWSTR       = LPWSTR
PCHAR       = LPSTR
PWCHAR      = LPWSTR
LPBYTE      = POINTER(BYTE)
LPSBYTE     = POINTER(SBYTE)
LPWORD      = POINTER(WORD)
LPSWORD     = POINTER(SWORD)
LPDWORD     = POINTER(DWORD)
LPSDWORD    = POINTER(SDWORD)
LPULONG     = POINTER(ULONG)
LPLONG      = POINTER(LONG)
PDWORD      = LPDWORD
PDWORD_PTR  = POINTER(DWORD_PTR)
PULONG      = LPULONG
PLONG       = LPLONG
CCHAR       = CHAR
BOOLEAN     = BYTE
PBOOL       = POINTER(BOOL)
LPBOOL      = PBOOL
TCHAR       = CHAR      # XXX ANSI by default?
UCHAR       = BYTE
DWORDLONG   = ULONGLONG
LPDWORD32   = POINTER(DWORD32)
LPULONG32   = POINTER(ULONG32)
LPDWORD64   = POINTER(DWORD64)
LPULONG64   = POINTER(ULONG64)
PDWORD32    = LPDWORD32
PULONG32    = LPULONG32
PDWORD64    = LPDWORD64
PULONG64    = LPULONG64
ATOM        = WORD
HANDLE      = LPVOID
PHANDLE     = POINTER(HANDLE)
LPHANDLE    = PHANDLE
HMODULE     = HANDLE
HINSTANCE   = HANDLE
HTASK       = HANDLE
HKEY        = HANDLE
PHKEY       = POINTER(HKEY)
HDESK       = HANDLE
HRSRC       = HANDLE
HSTR        = HANDLE
HWINSTA     = HANDLE
HKL         = HANDLE
HDWP        = HANDLE
HFILE       = HANDLE
HRESULT     = LONG
HGLOBAL     = HANDLE
HLOCAL      = HANDLE
HGDIOBJ     = HANDLE
HDC         = HGDIOBJ
HRGN        = HGDIOBJ
HBITMAP     = HGDIOBJ
HPALETTE    = HGDIOBJ
HPEN        = HGDIOBJ
HBRUSH      = HGDIOBJ
HMF         = HGDIOBJ
HEMF        = HGDIOBJ
HENHMETAFILE = HGDIOBJ
HMETAFILE   = HGDIOBJ
HMETAFILEPICT = HGDIOBJ
HWND        = HANDLE
NTSTATUS    = LONG
PNTSTATUS   = POINTER(NTSTATUS)
KAFFINITY   = ULONG_PTR
RVA         = DWORD
RVA64       = QWORD
WPARAM      = DWORD
LPARAM      = LPVOID
LRESULT     = LPVOID
ACCESS_MASK = DWORD
REGSAM      = ACCESS_MASK
PACCESS_MASK = POINTER(ACCESS_MASK)
PREGSAM     = POINTER(REGSAM)

# Since the SID is an opaque structure, let's treat its pointers as void*
PSID = PVOID

# typedef union _LARGE_INTEGER {
#   struct {
#     DWORD LowPart;
#     LONG HighPart;
#   } ;
#   struct {
#     DWORD LowPart;
#     LONG HighPart;
#   } u;
#   LONGLONG QuadPart;
# } LARGE_INTEGER,
#  *PLARGE_INTEGER;

# XXX TODO

# typedef struct _FLOAT128 {
#     __int64 LowPart;
#     __int64 HighPart;
# } FLOAT128;
class FLOAT128 (Structure):
    _fields_ = [
        ("LowPart",     QWORD),
        ("HighPart",    QWORD),
    ]
PFLOAT128 = POINTER(FLOAT128)

# typedef struct DECLSPEC_ALIGN(16) _M128A {
#     ULONGLONG Low;
#     LONGLONG High;
# } M128A, *PM128A;
class M128A(Structure):
    _fields_ = [
        ("Low",     ULONGLONG),
        ("High",    LONGLONG),
    ]
PM128A = POINTER(M128A)

#--- Constants ----------------------------------------------------------------

NULL        = None
INFINITE    = -1
TRUE        = 1
FALSE       = 0

# http://blogs.msdn.com/oldnewthing/archive/2004/08/26/220873.aspx
ANYSIZE_ARRAY = 1

# Invalid handle value is -1 casted to void pointer.
try:
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value #-1 #0xFFFFFFFF
except TypeError:
    if sizeof(ctypes.c_void_p) == 4:
        INVALID_HANDLE_VALUE = 0xFFFFFFFF
    elif sizeof(ctypes.c_void_p) == 8:
        INVALID_HANDLE_VALUE = 0xFFFFFFFFFFFFFFFF
    else:
        raise

MAX_MODULE_NAME32   = 255
MAX_PATH            = 260

# Error codes
# TODO maybe add more error codes?
# if they're too many they could be pickled instead,
# or at the very least put in a new file
ERROR_SUCCESS                       = 0
ERROR_INVALID_FUNCTION              = 1
ERROR_FILE_NOT_FOUND                = 2
ERROR_PATH_NOT_FOUND                = 3
ERROR_ACCESS_DENIED                 = 5
ERROR_INVALID_HANDLE                = 6
ERROR_NOT_ENOUGH_MEMORY             = 8
ERROR_INVALID_DRIVE                 = 15
ERROR_NO_MORE_FILES                 = 18
ERROR_BAD_LENGTH                    = 24
ERROR_HANDLE_EOF                    = 38
ERROR_HANDLE_DISK_FULL              = 39
ERROR_NOT_SUPPORTED                 = 50
ERROR_FILE_EXISTS                   = 80
ERROR_INVALID_PARAMETER             = 87
ERROR_BUFFER_OVERFLOW               = 111
ERROR_DISK_FULL                     = 112
ERROR_CALL_NOT_IMPLEMENTED          = 120
ERROR_SEM_TIMEOUT                   = 121
ERROR_INSUFFICIENT_BUFFER           = 122
ERROR_INVALID_NAME                  = 123
ERROR_MOD_NOT_FOUND                 = 126
ERROR_PROC_NOT_FOUND                = 127
ERROR_DIR_NOT_EMPTY                 = 145
ERROR_BAD_THREADID_ADDR             = 159
ERROR_BAD_ARGUMENTS                 = 160
ERROR_BAD_PATHNAME                  = 161
ERROR_ALREADY_EXISTS                = 183
ERROR_INVALID_FLAG_NUMBER           = 186
ERROR_ENVVAR_NOT_FOUND              = 203
ERROR_FILENAME_EXCED_RANGE          = 206
ERROR_MORE_DATA                     = 234

WAIT_TIMEOUT                        = 258

ERROR_NO_MORE_ITEMS                 = 259
ERROR_PARTIAL_COPY                  = 299
ERROR_INVALID_ADDRESS               = 487
ERROR_THREAD_NOT_IN_PROCESS         = 566
ERROR_CONTROL_C_EXIT                = 572
ERROR_UNHANDLED_EXCEPTION           = 574
ERROR_ASSERTION_FAILURE             = 668
ERROR_WOW_ASSERTION                 = 670

ERROR_DBG_EXCEPTION_NOT_HANDLED     = 688
ERROR_DBG_REPLY_LATER               = 689
ERROR_DBG_UNABLE_TO_PROVIDE_HANDLE  = 690
ERROR_DBG_TERMINATE_THREAD          = 691
ERROR_DBG_TERMINATE_PROCESS         = 692
ERROR_DBG_CONTROL_C                 = 693
ERROR_DBG_PRINTEXCEPTION_C          = 694
ERROR_DBG_RIPEXCEPTION              = 695
ERROR_DBG_CONTROL_BREAK             = 696
ERROR_DBG_COMMAND_EXCEPTION         = 697
ERROR_DBG_EXCEPTION_HANDLED         = 766
ERROR_DBG_CONTINUE                  = 767

ERROR_ELEVATION_REQUIRED            = 740
ERROR_NOACCESS                      = 998

ERROR_CIRCULAR_DEPENDENCY           = 1059
ERROR_SERVICE_DOES_NOT_EXIST        = 1060
ERROR_SERVICE_CANNOT_ACCEPT_CTRL    = 1061
ERROR_SERVICE_NOT_ACTIVE            = 1062
ERROR_FAILED_SERVICE_CONTROLLER_CONNECT = 1063
ERROR_EXCEPTION_IN_SERVICE          = 1064
ERROR_DATABASE_DOES_NOT_EXIST       = 1065
ERROR_SERVICE_SPECIFIC_ERROR        = 1066
ERROR_PROCESS_ABORTED               = 1067
ERROR_SERVICE_DEPENDENCY_FAIL       = 1068
ERROR_SERVICE_LOGON_FAILED          = 1069
ERROR_SERVICE_START_HANG            = 1070
ERROR_INVALID_SERVICE_LOCK          = 1071
ERROR_SERVICE_MARKED_FOR_DELETE     = 1072
ERROR_SERVICE_EXISTS                = 1073
ERROR_ALREADY_RUNNING_LKG           = 1074
ERROR_SERVICE_DEPENDENCY_DELETED    = 1075
ERROR_BOOT_ALREADY_ACCEPTED         = 1076
ERROR_SERVICE_NEVER_STARTED         = 1077
ERROR_DUPLICATE_SERVICE_NAME        = 1078
ERROR_DIFFERENT_SERVICE_ACCOUNT     = 1079
ERROR_CANNOT_DETECT_DRIVER_FAILURE  = 1080
ERROR_CANNOT_DETECT_PROCESS_ABORT   = 1081
ERROR_NO_RECOVERY_PROGRAM           = 1082
ERROR_SERVICE_NOT_IN_EXE            = 1083
ERROR_NOT_SAFEBOOT_SERVICE          = 1084

ERROR_DEBUGGER_INACTIVE             = 1284

ERROR_PRIVILEGE_NOT_HELD            = 1314

ERROR_NONE_MAPPED                   = 1332

RPC_S_SERVER_UNAVAILABLE            = 1722

# Standard access rights
import sys
if sys.version_info[0] >= 3:
    long = int

DELETE                           = long(0x00010000)
READ_CONTROL                     = long(0x00020000)
WRITE_DAC                        = long(0x00040000)
WRITE_OWNER                      = long(0x00080000)
SYNCHRONIZE                      = long(0x00100000)
STANDARD_RIGHTS_REQUIRED         = long(0x000F0000)
STANDARD_RIGHTS_READ             = READ_CONTROL
STANDARD_RIGHTS_WRITE            = READ_CONTROL
STANDARD_RIGHTS_EXECUTE          = READ_CONTROL
STANDARD_RIGHTS_ALL              = long(0x001F0000)
SPECIFIC_RIGHTS_ALL              = long(0x0000FFFF)

#--- Structures ---------------------------------------------------------------

# typedef struct _LSA_UNICODE_STRING {
#   USHORT Length;
#   USHORT MaximumLength;
#   PWSTR Buffer;
# } LSA_UNICODE_STRING,
#  *PLSA_UNICODE_STRING,
#  UNICODE_STRING,
#  *PUNICODE_STRING;
class UNICODE_STRING(Structure):
    _fields_ = [
        ("Length",          USHORT),
        ("MaximumLength",   USHORT),
        ("Buffer",          PVOID),
    ]

# From MSDN:
#
# typedef struct _GUID {
#   DWORD Data1;
#   WORD Data2;
#   WORD Data3;
#   BYTE Data4[8];
# } GUID;
class GUID(Structure):
    _fields_ = [
        ("Data1",   DWORD),
        ("Data2",   WORD),
        ("Data3",   WORD),
        ("Data4",   BYTE * 8),
]

# From MSDN:
#
# typedef struct _LIST_ENTRY {
#     struct _LIST_ENTRY *Flink;
#     struct _LIST_ENTRY *Blink;
# } LIST_ENTRY, *PLIST_ENTRY, *RESTRICTED_POINTER PRLIST_ENTRY;
class LIST_ENTRY(Structure):
    _fields_ = [
        ("Flink",   PVOID),     # POINTER(LIST_ENTRY)
        ("Blink",   PVOID),     # POINTER(LIST_ENTRY)
]

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
##__all__ = [_x for _x in _all if not _x.startswith('_')]
##__all__.sort()
#==============================================================================
