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
Wrapper for kernel32.dll in ctypes.
"""

__revision__ = "$Id$"

import warnings

from winappdbg.win32.defines import *

from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
_all.add('version')
#==============================================================================

from winappdbg.win32.version import *

#------------------------------------------------------------------------------

# This can't be defined in defines.py because it calls GetLastError().
def RaiseIfLastError(result, func = None, arguments = ()):
    """
    Error checking for Win32 API calls with no error-specific return value.

    Regardless of the return value, the function calls GetLastError(). If the
    code is not C{ERROR_SUCCESS} then a C{WindowsError} exception is raised.

    For this to work, the user MUST call SetLastError(ERROR_SUCCESS) prior to
    calling the API. Otherwise an exception may be raised even on success,
    since most API calls don't clear the error status code.
    """
    code = GetLastError()
    if code != ERROR_SUCCESS:
        raise ctypes.WinError(code)
    return result

#--- CONTEXT structure and constants ------------------------------------------

ContextArchMask = 0x0FFF0000    # just guessing here! seems to work, though

if   arch == ARCH_I386:
    from winappdbg.win32.context_i386 import *
elif arch == ARCH_AMD64:
    if bits == 64:
        from winappdbg.win32.context_amd64 import *
    else:
        from winappdbg.win32.context_i386 import *
else:
    warnings.warn("Unknown or unsupported architecture: %s" % arch)

#--- Constants ----------------------------------------------------------------

STILL_ACTIVE = 259

WAIT_TIMEOUT        = 0x102
WAIT_FAILED         = -1
WAIT_OBJECT_0       = 0

EXCEPTION_NONCONTINUABLE        = 0x1       # Noncontinuable exception
EXCEPTION_MAXIMUM_PARAMETERS    = 15        # maximum number of exception parameters
MAXIMUM_WAIT_OBJECTS            = 64        # Maximum number of wait objects
MAXIMUM_SUSPEND_COUNT           = 0x7f      # Maximum times thread can be suspended

FORMAT_MESSAGE_ALLOCATE_BUFFER  = 0x00000100
FORMAT_MESSAGE_FROM_SYSTEM      = 0x00001000

GR_GDIOBJECTS  = 0
GR_USEROBJECTS = 1

PROCESS_NAME_NATIVE = 1

MAXINTATOM = 0xC000

STD_INPUT_HANDLE  = 0xFFFFFFF6      # (DWORD)-10
STD_OUTPUT_HANDLE = 0xFFFFFFF5      # (DWORD)-11
STD_ERROR_HANDLE  = 0xFFFFFFF4      # (DWORD)-12

ATTACH_PARENT_PROCESS = 0xFFFFFFFF  # (DWORD)-1

# LoadLibraryEx constants
DONT_RESOLVE_DLL_REFERENCES         = 0x00000001
LOAD_LIBRARY_AS_DATAFILE            = 0x00000002
LOAD_WITH_ALTERED_SEARCH_PATH       = 0x00000008
LOAD_IGNORE_CODE_AUTHZ_LEVEL        = 0x00000010
LOAD_LIBRARY_AS_IMAGE_RESOURCE      = 0x00000020
LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE  = 0x00000040

# SetSearchPathMode flags
# TODO I couldn't find these constants :(
##BASE_SEARCH_PATH_ENABLE_SAFE_SEARCHMODE     = ???
##BASE_SEARCH_PATH_DISABLE_SAFE_SEARCHMODE    = ???
##BASE_SEARCH_PATH_PERMANENT                  = ???

# Console control events
CTRL_C_EVENT        = 0
CTRL_BREAK_EVENT    = 1
CTRL_CLOSE_EVENT    = 2
CTRL_LOGOFF_EVENT   = 5
CTRL_SHUTDOWN_EVENT = 6

# Heap flags
HEAP_NO_SERIALIZE           = 0x00000001
HEAP_GENERATE_EXCEPTIONS    = 0x00000004
HEAP_ZERO_MEMORY            = 0x00000008
HEAP_CREATE_ENABLE_EXECUTE  = 0x00040000

# Standard access rights
DELETE                      = long(0x00010000)
READ_CONTROL                = long(0x00020000)
WRITE_DAC                   = long(0x00040000)
WRITE_OWNER                 = long(0x00080000)
SYNCHRONIZE                 = long(0x00100000)
STANDARD_RIGHTS_REQUIRED    = long(0x000F0000)
STANDARD_RIGHTS_READ        = (READ_CONTROL)
STANDARD_RIGHTS_WRITE       = (READ_CONTROL)
STANDARD_RIGHTS_EXECUTE     = (READ_CONTROL)
STANDARD_RIGHTS_ALL         = long(0x001F0000)
SPECIFIC_RIGHTS_ALL         = long(0x0000FFFF)

# Mutex access rights
MUTEX_ALL_ACCESS   = 0x1F0001
MUTEX_MODIFY_STATE = 1

# Event access rights
EVENT_ALL_ACCESS   = 0x1F0003
EVENT_MODIFY_STATE = 2

# Semaphore access rights
SEMAPHORE_ALL_ACCESS   = 0x1F0003
SEMAPHORE_MODIFY_STATE = 2

# Timer access rights
TIMER_ALL_ACCESS   = 0x1F0003
TIMER_MODIFY_STATE = 2
TIMER_QUERY_STATE  = 1

# Process access rights for OpenProcess
PROCESS_TERMINATE                 = 0x0001
PROCESS_CREATE_THREAD             = 0x0002
PROCESS_SET_SESSIONID             = 0x0004
PROCESS_VM_OPERATION              = 0x0008
PROCESS_VM_READ                   = 0x0010
PROCESS_VM_WRITE                  = 0x0020
PROCESS_DUP_HANDLE                = 0x0040
PROCESS_CREATE_PROCESS            = 0x0080
PROCESS_SET_QUOTA                 = 0x0100
PROCESS_SET_INFORMATION           = 0x0200
PROCESS_QUERY_INFORMATION         = 0x0400
PROCESS_SUSPEND_RESUME            = 0x0800
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Thread access rights for OpenThread
THREAD_TERMINATE                 = 0x0001
THREAD_SUSPEND_RESUME            = 0x0002
THREAD_ALERT                     = 0x0004
THREAD_GET_CONTEXT               = 0x0008
THREAD_SET_CONTEXT               = 0x0010
THREAD_SET_INFORMATION           = 0x0020
THREAD_QUERY_INFORMATION         = 0x0040
THREAD_SET_THREAD_TOKEN          = 0x0080
THREAD_IMPERSONATE               = 0x0100
THREAD_DIRECT_IMPERSONATION      = 0x0200
THREAD_SET_LIMITED_INFORMATION   = 0x0400
THREAD_QUERY_LIMITED_INFORMATION = 0x0800

# The values of PROCESS_ALL_ACCESS and THREAD_ALL_ACCESS were changed in Vista/2008
PROCESS_ALL_ACCESS_NT = (STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0xFFF)
PROCESS_ALL_ACCESS_VISTA = (STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0xFFFF)
THREAD_ALL_ACCESS_NT = (STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x3FF)
THREAD_ALL_ACCESS_VISTA = (STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0xFFFF)
if NTDDI_VERSION < NTDDI_VISTA:
    PROCESS_ALL_ACCESS = PROCESS_ALL_ACCESS_NT
    THREAD_ALL_ACCESS = THREAD_ALL_ACCESS_NT
else:
    PROCESS_ALL_ACCESS = PROCESS_ALL_ACCESS_VISTA
    THREAD_ALL_ACCESS = THREAD_ALL_ACCESS_VISTA

# Process priority classes

IDLE_PRIORITY_CLASS         = 0x00000040
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
NORMAL_PRIORITY_CLASS       = 0x00000020
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
HIGH_PRIORITY_CLASS         = 0x00000080
REALTIME_PRIORITY_CLASS     = 0x00000100

PROCESS_MODE_BACKGROUND_BEGIN   = 0x00100000
PROCESS_MODE_BACKGROUND_END     = 0x00200000

# dwCreationFlag values

DEBUG_PROCESS                     = 0x00000001
DEBUG_ONLY_THIS_PROCESS           = 0x00000002
CREATE_SUSPENDED                  = 0x00000004    # Threads and processes
DETACHED_PROCESS                  = 0x00000008
CREATE_NEW_CONSOLE                = 0x00000010
NORMAL_PRIORITY_CLASS             = 0x00000020
IDLE_PRIORITY_CLASS               = 0x00000040
HIGH_PRIORITY_CLASS               = 0x00000080
REALTIME_PRIORITY_CLASS           = 0x00000100
CREATE_NEW_PROCESS_GROUP          = 0x00000200
CREATE_UNICODE_ENVIRONMENT        = 0x00000400
CREATE_SEPARATE_WOW_VDM           = 0x00000800
CREATE_SHARED_WOW_VDM             = 0x00001000
CREATE_FORCEDOS                   = 0x00002000
BELOW_NORMAL_PRIORITY_CLASS       = 0x00004000
ABOVE_NORMAL_PRIORITY_CLASS       = 0x00008000
INHERIT_PARENT_AFFINITY           = 0x00010000
STACK_SIZE_PARAM_IS_A_RESERVATION = 0x00010000    # Threads only
INHERIT_CALLER_PRIORITY           = 0x00020000    # Deprecated
CREATE_PROTECTED_PROCESS          = 0x00040000
EXTENDED_STARTUPINFO_PRESENT      = 0x00080000
PROCESS_MODE_BACKGROUND_BEGIN     = 0x00100000
PROCESS_MODE_BACKGROUND_END       = 0x00200000
CREATE_BREAKAWAY_FROM_JOB         = 0x01000000
CREATE_PRESERVE_CODE_AUTHZ_LEVEL  = 0x02000000
CREATE_DEFAULT_ERROR_MODE         = 0x04000000
CREATE_NO_WINDOW                  = 0x08000000
PROFILE_USER                      = 0x10000000
PROFILE_KERNEL                    = 0x20000000
PROFILE_SERVER                    = 0x40000000
CREATE_IGNORE_SYSTEM_DEFAULT      = 0x80000000

# Thread priority values

THREAD_BASE_PRIORITY_LOWRT  = 15    # value that gets a thread to LowRealtime-1
THREAD_BASE_PRIORITY_MAX    = 2     # maximum thread base priority boost
THREAD_BASE_PRIORITY_MIN    = (-2)  # minimum thread base priority boost
THREAD_BASE_PRIORITY_IDLE   = (-15) # value that gets a thread to idle

THREAD_PRIORITY_LOWEST          = THREAD_BASE_PRIORITY_MIN
THREAD_PRIORITY_BELOW_NORMAL    = (THREAD_PRIORITY_LOWEST+1)
THREAD_PRIORITY_NORMAL          = 0
THREAD_PRIORITY_HIGHEST         = THREAD_BASE_PRIORITY_MAX
THREAD_PRIORITY_ABOVE_NORMAL    = (THREAD_PRIORITY_HIGHEST-1)
THREAD_PRIORITY_ERROR_RETURN    = long(0xFFFFFFFF)

THREAD_PRIORITY_TIME_CRITICAL   = THREAD_BASE_PRIORITY_LOWRT
THREAD_PRIORITY_IDLE            = THREAD_BASE_PRIORITY_IDLE

# Memory access
SECTION_QUERY                = 0x0001
SECTION_MAP_WRITE            = 0x0002
SECTION_MAP_READ             = 0x0004
SECTION_MAP_EXECUTE          = 0x0008
SECTION_EXTEND_SIZE          = 0x0010
SECTION_MAP_EXECUTE_EXPLICIT = 0x0020 # not included in SECTION_ALL_ACCESS

SECTION_ALL_ACCESS = (STANDARD_RIGHTS_REQUIRED|SECTION_QUERY|\
                             SECTION_MAP_WRITE |      \
                             SECTION_MAP_READ |       \
                             SECTION_MAP_EXECUTE |    \
                             SECTION_EXTEND_SIZE)
PAGE_NOACCESS          = 0x01
PAGE_READONLY          = 0x02
PAGE_READWRITE         = 0x04
PAGE_WRITECOPY         = 0x08
PAGE_EXECUTE           = 0x10
PAGE_EXECUTE_READ      = 0x20
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80
PAGE_GUARD            = 0x100
PAGE_NOCACHE          = 0x200
PAGE_WRITECOMBINE     = 0x400
MEM_COMMIT           = 0x1000
MEM_RESERVE          = 0x2000
MEM_DECOMMIT         = 0x4000
MEM_RELEASE          = 0x8000
MEM_FREE            = 0x10000
MEM_PRIVATE         = 0x20000
MEM_MAPPED          = 0x40000
MEM_RESET           = 0x80000
MEM_TOP_DOWN       = 0x100000
MEM_WRITE_WATCH    = 0x200000
MEM_PHYSICAL       = 0x400000
MEM_LARGE_PAGES  = 0x20000000
MEM_4MB_PAGES    = 0x80000000
SEC_FILE           = 0x800000
SEC_IMAGE         = 0x1000000
SEC_RESERVE       = 0x4000000
SEC_COMMIT        = 0x8000000
SEC_NOCACHE      = 0x10000000
SEC_LARGE_PAGES  = 0x80000000
MEM_IMAGE         = SEC_IMAGE
WRITE_WATCH_FLAG_RESET = 0x01
FILE_MAP_ALL_ACCESS = 0xF001F

SECTION_QUERY                   = 0x0001
SECTION_MAP_WRITE               = 0x0002
SECTION_MAP_READ                = 0x0004
SECTION_MAP_EXECUTE             = 0x0008
SECTION_EXTEND_SIZE             = 0x0010
SECTION_MAP_EXECUTE_EXPLICIT    = 0x0020 # not included in SECTION_ALL_ACCESS

SECTION_ALL_ACCESS = (STANDARD_RIGHTS_REQUIRED|SECTION_QUERY|\
                 SECTION_MAP_WRITE |      \
                 SECTION_MAP_READ |       \
                 SECTION_MAP_EXECUTE |    \
                 SECTION_EXTEND_SIZE)

FILE_MAP_COPY       = SECTION_QUERY
FILE_MAP_WRITE      = SECTION_MAP_WRITE
FILE_MAP_READ       = SECTION_MAP_READ
FILE_MAP_ALL_ACCESS = SECTION_ALL_ACCESS
FILE_MAP_EXECUTE    = SECTION_MAP_EXECUTE_EXPLICIT  # not included in FILE_MAP_ALL_ACCESS

GENERIC_READ                     = 0x80000000
GENERIC_WRITE                    = 0x40000000
GENERIC_EXECUTE                  = 0x20000000
GENERIC_ALL                      = 0x10000000

FILE_SHARE_READ                  = 0x00000001
FILE_SHARE_WRITE                 = 0x00000002
FILE_SHARE_DELETE                = 0x00000004

CREATE_NEW                       = 1
CREATE_ALWAYS                    = 2
OPEN_EXISTING                    = 3
OPEN_ALWAYS                      = 4
TRUNCATE_EXISTING                = 5

FILE_ATTRIBUTE_READONLY          = 0x00000001
FILE_ATTRIBUTE_NORMAL            = 0x00000080
FILE_ATTRIBUTE_TEMPORARY         = 0x00000100

FILE_FLAG_WRITE_THROUGH          = 0x80000000
FILE_FLAG_NO_BUFFERING           = 0x20000000
FILE_FLAG_RANDOM_ACCESS          = 0x10000000
FILE_FLAG_SEQUENTIAL_SCAN        = 0x08000000
FILE_FLAG_DELETE_ON_CLOSE        = 0x04000000
FILE_FLAG_OVERLAPPED             = 0x40000000

FILE_ATTRIBUTE_READONLY          = 0x00000001
FILE_ATTRIBUTE_HIDDEN            = 0x00000002
FILE_ATTRIBUTE_SYSTEM            = 0x00000004
FILE_ATTRIBUTE_DIRECTORY         = 0x00000010
FILE_ATTRIBUTE_ARCHIVE           = 0x00000020
FILE_ATTRIBUTE_DEVICE            = 0x00000040
FILE_ATTRIBUTE_NORMAL            = 0x00000080
FILE_ATTRIBUTE_TEMPORARY         = 0x00000100

# Debug events
EXCEPTION_DEBUG_EVENT       = 1
CREATE_THREAD_DEBUG_EVENT   = 2
CREATE_PROCESS_DEBUG_EVENT  = 3
EXIT_THREAD_DEBUG_EVENT     = 4
EXIT_PROCESS_DEBUG_EVENT    = 5
LOAD_DLL_DEBUG_EVENT        = 6
UNLOAD_DLL_DEBUG_EVENT      = 7
OUTPUT_DEBUG_STRING_EVENT   = 8
RIP_EVENT                   = 9

# Debug status codes (ContinueDebugEvent)
DBG_EXCEPTION_HANDLED           = long(0x00010001)
DBG_CONTINUE                    = long(0x00010002)
DBG_REPLY_LATER                 = long(0x40010001)
DBG_UNABLE_TO_PROVIDE_HANDLE    = long(0x40010002)
DBG_TERMINATE_THREAD            = long(0x40010003)
DBG_TERMINATE_PROCESS           = long(0x40010004)
DBG_CONTROL_C                   = long(0x40010005)
DBG_PRINTEXCEPTION_C            = long(0x40010006)
DBG_RIPEXCEPTION                = long(0x40010007)
DBG_CONTROL_BREAK               = long(0x40010008)
DBG_COMMAND_EXCEPTION           = long(0x40010009)
DBG_EXCEPTION_NOT_HANDLED       = long(0x80010001)
DBG_NO_STATE_CHANGE             = long(0xC0010001)
DBG_APP_NOT_IDLE                = long(0xC0010002)

# Status codes
STATUS_WAIT_0                   = long(0x00000000)
STATUS_ABANDONED_WAIT_0         = long(0x00000080)
STATUS_USER_APC                 = long(0x000000C0)
STATUS_TIMEOUT                  = long(0x00000102)
STATUS_PENDING                  = long(0x00000103)
STATUS_SEGMENT_NOTIFICATION     = long(0x40000005)
STATUS_GUARD_PAGE_VIOLATION     = long(0x80000001)
STATUS_DATATYPE_MISALIGNMENT    = long(0x80000002)
STATUS_BREAKPOINT               = long(0x80000003)
STATUS_SINGLE_STEP              = long(0x80000004)
STATUS_INVALID_INFO_CLASS       = long(0xC0000003)
STATUS_ACCESS_VIOLATION         = long(0xC0000005)
STATUS_IN_PAGE_ERROR            = long(0xC0000006)
STATUS_INVALID_HANDLE           = long(0xC0000008)
STATUS_NO_MEMORY                = long(0xC0000017)
STATUS_ILLEGAL_INSTRUCTION      = long(0xC000001D)
STATUS_NONCONTINUABLE_EXCEPTION = long(0xC0000025)
STATUS_INVALID_DISPOSITION      = long(0xC0000026)
STATUS_ARRAY_BOUNDS_EXCEEDED    = long(0xC000008C)
STATUS_FLOAT_DENORMAL_OPERAND   = long(0xC000008D)
STATUS_FLOAT_DIVIDE_BY_ZERO     = long(0xC000008E)
STATUS_FLOAT_INEXACT_RESULT     = long(0xC000008F)
STATUS_FLOAT_INVALID_OPERATION  = long(0xC0000090)
STATUS_FLOAT_OVERFLOW           = long(0xC0000091)
STATUS_FLOAT_STACK_CHECK        = long(0xC0000092)
STATUS_FLOAT_UNDERFLOW          = long(0xC0000093)
STATUS_INTEGER_DIVIDE_BY_ZERO   = long(0xC0000094)
STATUS_INTEGER_OVERFLOW         = long(0xC0000095)
STATUS_PRIVILEGED_INSTRUCTION   = long(0xC0000096)
STATUS_STACK_OVERFLOW           = long(0xC00000FD)
STATUS_CONTROL_C_EXIT           = long(0xC000013A)
STATUS_FLOAT_MULTIPLE_FAULTS    = long(0xC00002B4)
STATUS_FLOAT_MULTIPLE_TRAPS     = long(0xC00002B5)
STATUS_REG_NAT_CONSUMPTION      = long(0xC00002C9)
STATUS_SXS_EARLY_DEACTIVATION   = long(0xC015000F)
STATUS_SXS_INVALID_DEACTIVATION = long(0xC0150010)

STATUS_STACK_BUFFER_OVERRUN     = long(0xC0000409)
STATUS_WX86_BREAKPOINT          = long(0x4000001F)
STATUS_HEAP_CORRUPTION          = long(0xC0000374)

STATUS_POSSIBLE_DEADLOCK        = long(0xC0000194)

STATUS_UNWIND_CONSOLIDATE       = long(0x80000029)

# Exception codes

EXCEPTION_ACCESS_VIOLATION          = STATUS_ACCESS_VIOLATION
EXCEPTION_ARRAY_BOUNDS_EXCEEDED     = STATUS_ARRAY_BOUNDS_EXCEEDED
EXCEPTION_BREAKPOINT                = STATUS_BREAKPOINT
EXCEPTION_DATATYPE_MISALIGNMENT     = STATUS_DATATYPE_MISALIGNMENT
EXCEPTION_FLT_DENORMAL_OPERAND      = STATUS_FLOAT_DENORMAL_OPERAND
EXCEPTION_FLT_DIVIDE_BY_ZERO        = STATUS_FLOAT_DIVIDE_BY_ZERO
EXCEPTION_FLT_INEXACT_RESULT        = STATUS_FLOAT_INEXACT_RESULT
EXCEPTION_FLT_INVALID_OPERATION     = STATUS_FLOAT_INVALID_OPERATION
EXCEPTION_FLT_OVERFLOW              = STATUS_FLOAT_OVERFLOW
EXCEPTION_FLT_STACK_CHECK           = STATUS_FLOAT_STACK_CHECK
EXCEPTION_FLT_UNDERFLOW             = STATUS_FLOAT_UNDERFLOW
EXCEPTION_ILLEGAL_INSTRUCTION       = STATUS_ILLEGAL_INSTRUCTION
EXCEPTION_IN_PAGE_ERROR             = STATUS_IN_PAGE_ERROR
EXCEPTION_INT_DIVIDE_BY_ZERO        = STATUS_INTEGER_DIVIDE_BY_ZERO
EXCEPTION_INT_OVERFLOW              = STATUS_INTEGER_OVERFLOW
EXCEPTION_INVALID_DISPOSITION       = STATUS_INVALID_DISPOSITION
EXCEPTION_NONCONTINUABLE_EXCEPTION  = STATUS_NONCONTINUABLE_EXCEPTION
EXCEPTION_PRIV_INSTRUCTION          = STATUS_PRIVILEGED_INSTRUCTION
EXCEPTION_SINGLE_STEP               = STATUS_SINGLE_STEP
EXCEPTION_STACK_OVERFLOW            = STATUS_STACK_OVERFLOW

EXCEPTION_GUARD_PAGE                = STATUS_GUARD_PAGE_VIOLATION
EXCEPTION_INVALID_HANDLE            = STATUS_INVALID_HANDLE
EXCEPTION_POSSIBLE_DEADLOCK         = STATUS_POSSIBLE_DEADLOCK
EXCEPTION_WX86_BREAKPOINT           = STATUS_WX86_BREAKPOINT

CONTROL_C_EXIT                      = STATUS_CONTROL_C_EXIT

DBG_CONTROL_C                       = long(0x40010005)
MS_VC_EXCEPTION                     = long(0x406D1388)

# Access violation types
ACCESS_VIOLATION_TYPE_READ      = EXCEPTION_READ_FAULT
ACCESS_VIOLATION_TYPE_WRITE     = EXCEPTION_WRITE_FAULT
ACCESS_VIOLATION_TYPE_DEP       = EXCEPTION_EXECUTE_FAULT

# RIP event types
SLE_ERROR      = 1
SLE_MINORERROR = 2
SLE_WARNING    = 3

# DuplicateHandle constants
DUPLICATE_CLOSE_SOURCE      = 0x00000001
DUPLICATE_SAME_ACCESS       = 0x00000002

# GetFinalPathNameByHandle constants
FILE_NAME_NORMALIZED        = 0x0
FILE_NAME_OPENED            = 0x8
VOLUME_NAME_DOS             = 0x0
VOLUME_NAME_GUID            = 0x1
VOLUME_NAME_NONE            = 0x4
VOLUME_NAME_NT              = 0x2

# GetProductInfo constants
PRODUCT_BUSINESS = 0x00000006
PRODUCT_BUSINESS_N = 0x00000010
PRODUCT_CLUSTER_SERVER = 0x00000012
PRODUCT_DATACENTER_SERVER = 0x00000008
PRODUCT_DATACENTER_SERVER_CORE = 0x0000000C
PRODUCT_DATACENTER_SERVER_CORE_V = 0x00000027
PRODUCT_DATACENTER_SERVER_V = 0x00000025
PRODUCT_ENTERPRISE = 0x00000004
PRODUCT_ENTERPRISE_E = 0x00000046
PRODUCT_ENTERPRISE_N = 0x0000001B
PRODUCT_ENTERPRISE_SERVER = 0x0000000A
PRODUCT_ENTERPRISE_SERVER_CORE = 0x0000000E
PRODUCT_ENTERPRISE_SERVER_CORE_V = 0x00000029
PRODUCT_ENTERPRISE_SERVER_IA64 = 0x0000000F
PRODUCT_ENTERPRISE_SERVER_V = 0x00000026
PRODUCT_HOME_BASIC = 0x00000002
PRODUCT_HOME_BASIC_E = 0x00000043
PRODUCT_HOME_BASIC_N = 0x00000005
PRODUCT_HOME_PREMIUM = 0x00000003
PRODUCT_HOME_PREMIUM_E = 0x00000044
PRODUCT_HOME_PREMIUM_N = 0x0000001A
PRODUCT_HYPERV = 0x0000002A
PRODUCT_MEDIUMBUSINESS_SERVER_MANAGEMENT = 0x0000001E
PRODUCT_MEDIUMBUSINESS_SERVER_MESSAGING = 0x00000020
PRODUCT_MEDIUMBUSINESS_SERVER_SECURITY = 0x0000001F
PRODUCT_PROFESSIONAL = 0x00000030
PRODUCT_PROFESSIONAL_E = 0x00000045
PRODUCT_PROFESSIONAL_N = 0x00000031
PRODUCT_SERVER_FOR_SMALLBUSINESS = 0x00000018
PRODUCT_SERVER_FOR_SMALLBUSINESS_V = 0x00000023
PRODUCT_SERVER_FOUNDATION = 0x00000021
PRODUCT_SMALLBUSINESS_SERVER = 0x00000009
PRODUCT_STANDARD_SERVER = 0x00000007
PRODUCT_STANDARD_SERVER_CORE = 0x0000000D
PRODUCT_STANDARD_SERVER_CORE_V = 0x00000028
PRODUCT_STANDARD_SERVER_V = 0x00000024
PRODUCT_STARTER = 0x0000000B
PRODUCT_STARTER_E = 0x00000042
PRODUCT_STARTER_N = 0x0000002F
PRODUCT_STORAGE_ENTERPRISE_SERVER = 0x00000017
PRODUCT_STORAGE_EXPRESS_SERVER = 0x00000014
PRODUCT_STORAGE_STANDARD_SERVER = 0x00000015
PRODUCT_STORAGE_WORKGROUP_SERVER = 0x00000016
PRODUCT_UNDEFINED = 0x00000000
PRODUCT_UNLICENSED = 0xABCDABCD
PRODUCT_ULTIMATE = 0x00000001
PRODUCT_ULTIMATE_E = 0x00000047
PRODUCT_ULTIMATE_N = 0x0000001C
PRODUCT_WEB_SERVER = 0x00000011
PRODUCT_WEB_SERVER_CORE = 0x0000001D

# DEP policy flags
PROCESS_DEP_ENABLE = 1
PROCESS_DEP_DISABLE_ATL_THUNK_EMULATION = 2

# Error modes
SEM_FAILCRITICALERRORS      = 0x001
SEM_NOGPFAULTERRORBOX       = 0x002
SEM_NOALIGNMENTFAULTEXCEPT  = 0x004
SEM_NOOPENFILEERRORBOX      = 0x800

# GetHandleInformation / SetHandleInformation
HANDLE_FLAG_INHERIT             = 0x00000001
HANDLE_FLAG_PROTECT_FROM_CLOSE  = 0x00000002

#--- Handle wrappers ----------------------------------------------------------

class Handle (object):
    """
    Encapsulates Win32 handles to avoid leaking them.

    @type inherit: bool
    @ivar inherit: C{True} if the handle is to be inherited by child processes,
        C{False} otherwise.

    @type protectFromClose: bool
    @ivar protectFromClose: Set to C{True} to prevent the handle from being
        closed. Must be set to C{False} before you're done using the handle,
        or it will be left open until the debugger exits. Use with care!

    @see:
        L{ProcessHandle}, L{ThreadHandle}, L{FileHandle}, L{SnapshotHandle}
    """

    # XXX DEBUG
    # When this private flag is True each Handle will print a message to
    # standard output when it's created and destroyed. This is useful for
    # detecting handle leaks within WinAppDbg itself.
    __bLeakDetection = False

    def __init__(self, aHandle = None, bOwnership = True):
        """
        @type  aHandle: int
        @param aHandle: Win32 handle value.

        @type  bOwnership: bool
        @param bOwnership:
           C{True} if we own the handle and we need to close it.
           C{False} if someone else will be calling L{CloseHandle}.
        """
        super(Handle, self).__init__()
        self._value     = self._normalize(aHandle)
        self.bOwnership = bOwnership
        if Handle.__bLeakDetection:     # XXX DEBUG
            print("INIT HANDLE (%r) %r" % (self.value, self))

    @property
    def value(self):
        return self._value

    def __del__(self):
        """
        Closes the Win32 handle when the Python object is destroyed.
        """
        try:
            if Handle.__bLeakDetection:     # XXX DEBUG
                print("DEL HANDLE %r" % self)
            self.close()
        except Exception:
            pass

    def __enter__(self):
        """
        Compatibility with the "C{with}" Python statement.
        """
        if Handle.__bLeakDetection:     # XXX DEBUG
            print("ENTER HANDLE %r" % self)
        return self

    def __exit__(self, type, value, traceback):
        """
        Compatibility with the "C{with}" Python statement.
        """
        if Handle.__bLeakDetection:     # XXX DEBUG
            print("EXIT HANDLE %r" % self)
        try:
            self.close()
        except Exception:
            pass

    def __copy__(self):
        """
        Duplicates the Win32 handle when copying the Python object.

        @rtype:  L{Handle}
        @return: A new handle to the same Win32 object.
        """
        return self.dup()

    def __deepcopy__(self):
        """
        Duplicates the Win32 handle when copying the Python object.

        @rtype:  L{Handle}
        @return: A new handle to the same win32 object.
        """
        return self.dup()

    @property
    def _as_parameter_(self):
        """
        Compatibility with ctypes.
        Allows passing transparently a Handle object to an API call.
        """
        return HANDLE(self.value)

    @staticmethod
    def from_param(value):
        """
        Compatibility with ctypes.
        Allows passing transparently a Handle object to an API call.

        @type  value: int
        @param value: Numeric handle value.
        """
        return HANDLE(value)

    def close(self):
        """
        Closes the Win32 handle.
        """
        if self.bOwnership and self.value not in (None, INVALID_HANDLE_VALUE):
            if Handle.__bLeakDetection:     # XXX DEBUG
                print("CLOSE HANDLE (%d) %r" % (self.value, self))
            try:
                self._close()
            finally:
                self._value = None

    def _close(self):
        """
        Low-level close method.
        This is a private method, do not call it.
        """
        CloseHandle(self.value)

    def dup(self):
        """
        @rtype:  L{Handle}
        @return: A new handle to the same Win32 object.
        """
        if self.value is None:
            raise ValueError("Closed handles can't be duplicated!")
        new_handle = DuplicateHandle(self.value)
        if Handle.__bLeakDetection:     # XXX DEBUG
            print("DUP HANDLE (%d -> %d) %r %r" % \
                            (self.value, new_handle.value, self, new_handle))
        return new_handle

    @staticmethod
    def _normalize(value):
        """
        Normalize handle values.
        """
        if hasattr(value, 'value'):
            value = value.value
        if value is not None:
            value = long(value)
        return value

    def wait(self, dwMilliseconds = None):
        """
        Wait for the Win32 object to be signaled.

        @type  dwMilliseconds: int
        @param dwMilliseconds: (Optional) Timeout value in milliseconds.
            Use C{INFINITE} or C{None} for no timeout.
        """
        if self.value is None:
            raise ValueError("Handle is already closed!")
        if dwMilliseconds is None:
            dwMilliseconds = INFINITE
        r = WaitForSingleObject(self.value, dwMilliseconds)
        if r != WAIT_OBJECT_0:
            raise ctypes.WinError(r)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.value)

    def __get_inherit(self):
        if self.value is None:
            raise ValueError("Handle is already closed!")
        return bool( GetHandleInformation(self.value) & HANDLE_FLAG_INHERIT )

    def __set_inherit(self, value):
        if self.value is None:
            raise ValueError("Handle is already closed!")
        flag = (0, HANDLE_FLAG_INHERIT)[ bool(value) ]
        SetHandleInformation(self.value, flag, flag)

    inherit = property(__get_inherit, __set_inherit)

    def __get_protectFromClose(self):
        if self.value is None:
            raise ValueError("Handle is already closed!")
        return bool( GetHandleInformation(self.value) & HANDLE_FLAG_PROTECT_FROM_CLOSE )

    def __set_protectFromClose(self, value):
        if self.value is None:
            raise ValueError("Handle is already closed!")
        flag = (0, HANDLE_FLAG_PROTECT_FROM_CLOSE)[ bool(value) ]
        SetHandleInformation(self.value, flag, flag)

    protectFromClose = property(__get_protectFromClose, __set_protectFromClose)

class UserModeHandle (Handle):
    """
    Base class for non-kernel handles. Generally this means they are closed
    by special Win32 API functions instead of CloseHandle() and some standard
    operations (synchronizing, duplicating, inheritance) are not supported.

    @type _TYPE: C type
    @cvar _TYPE: C type to translate this handle to.
        Subclasses should override this.
        Defaults to L{HANDLE}.
    """

    # Subclasses should override this.
    _TYPE = HANDLE

    # This method must be implemented by subclasses.
    def _close(self):
        raise NotImplementedError()

    # Translation to C type.
    @property
    def _as_parameter_(self):
        return self._TYPE(self.value)

    # Translation to C type.
    @staticmethod
    def from_param(value):
        return self._TYPE(self.value)

    # Operation not supported.
    @property
    def inherit(self):
        return False

    # Operation not supported.
    @property
    def protectFromClose(self):
        return False

    # Operation not supported.
    def dup(self):
        raise NotImplementedError()

    # Operation not supported.
    def wait(self, dwMilliseconds = None):
        raise NotImplementedError()

class ProcessHandle (Handle):
    """
    Win32 process handle.

    @type dwAccess: int
    @ivar dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenProcess}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{PROCESS_ALL_ACCESS}.

    @see: L{Handle}
    """

    def __init__(self, aHandle = None, bOwnership = True,
                       dwAccess = PROCESS_ALL_ACCESS):
        """
        @type  aHandle: int
        @param aHandle: Win32 handle value.

        @type  bOwnership: bool
        @param bOwnership:
           C{True} if we own the handle and we need to close it.
           C{False} if someone else will be calling L{CloseHandle}.

        @type  dwAccess: int
        @param dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenProcess}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{PROCESS_ALL_ACCESS}.
        """
        super(ProcessHandle, self).__init__(aHandle, bOwnership)
        self.dwAccess = dwAccess
        if aHandle is not None and dwAccess is None:
            msg = "Missing access flags for process handle: %x" % aHandle
            raise TypeError(msg)

    def get_pid(self):
        """
        @rtype:  int
        @return: Process global ID.
        """
        return GetProcessId(self.value)

class ThreadHandle (Handle):
    """
    Win32 thread handle.

    @type dwAccess: int
    @ivar dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenThread}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{THREAD_ALL_ACCESS}.

    @see: L{Handle}
    """

    def __init__(self, aHandle = None, bOwnership = True,
                       dwAccess = THREAD_ALL_ACCESS):
        """
        @type  aHandle: int
        @param aHandle: Win32 handle value.

        @type  bOwnership: bool
        @param bOwnership:
           C{True} if we own the handle and we need to close it.
           C{False} if someone else will be calling L{CloseHandle}.

        @type  dwAccess: int
        @param dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenThread}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{THREAD_ALL_ACCESS}.
        """
        super(ThreadHandle, self).__init__(aHandle, bOwnership)
        self.dwAccess = dwAccess
        if aHandle is not None and dwAccess is None:
            msg = "Missing access flags for thread handle: %x" % aHandle
            raise TypeError(msg)

    def get_tid(self):
        """
        @rtype:  int
        @return: Thread global ID.
        """
        return GetThreadId(self.value)

class FileHandle (Handle):
    """
    Win32 file handle.

    @see: L{Handle}
    """

    def get_filename(self):
        """
        @rtype:  None or str
        @return: Name of the open file, or C{None} if unavailable.
        """
        #
        # XXX BUG
        #
        # This code truncates the first two bytes of the path.
        # It seems to be the expected behavior of NtQueryInformationFile.
        #
        # My guess is it only returns the NT pathname, without the device name.
        # It's like dropping the drive letter in a Win32 pathname.
        #
        # Note that using the "official" GetFileInformationByHandleEx
        # API introduced in Vista doesn't change the results!
        #
        dwBufferSize      = 0x1004
        lpFileInformation = ctypes.create_string_buffer(dwBufferSize)
        try:
            GetFileInformationByHandleEx(self.value,
                                        FILE_INFO_BY_HANDLE_CLASS.FileNameInfo,
                                        lpFileInformation, dwBufferSize)
        except AttributeError:
            from winappdbg.win32.ntdll import NtQueryInformationFile, \
                              FileNameInformation, \
                              FILE_NAME_INFORMATION
            NtQueryInformationFile(self.value,
                                   FileNameInformation,
                                   lpFileInformation,
                                   dwBufferSize)
        FileName = compat.unicode(lpFileInformation.raw[sizeof(DWORD):], 'U16')
        FileName = ctypes.create_unicode_buffer(FileName).value
        if not FileName:
            FileName = None
        elif FileName[1:2] != ':':
            # When the drive letter is missing, we'll assume SYSTEMROOT.
            # Not a good solution but it could be worse.
            import os
            FileName = os.environ['SYSTEMROOT'][:2] + FileName
        return FileName

class FileMappingHandle (Handle):
    """
    File mapping handle.

    @see: L{Handle}
    """
    pass

# XXX maybe add functions related to the toolhelp snapshots here?
class SnapshotHandle (Handle):
    """
    Toolhelp32 snapshot handle.

    @see: L{Handle}
    """
    pass

#--- Structure wrappers -------------------------------------------------------

class ProcessInformation (object):
    """
    Process information object returned by L{CreateProcess}.
    """

    def __init__(self, pi):
        self.hProcess    = ProcessHandle(pi.hProcess)
        self.hThread     = ThreadHandle(pi.hThread)
        self.dwProcessId = pi.dwProcessId
        self.dwThreadId  = pi.dwThreadId

# Don't psyco-optimize this class because it needs to be serialized.
class MemoryBasicInformation (object):
    """
    Memory information object returned by L{VirtualQueryEx}.
    """

    READABLE = (
                PAGE_EXECUTE_READ       |
                PAGE_EXECUTE_READWRITE  |
                PAGE_EXECUTE_WRITECOPY  |
                PAGE_READONLY           |
                PAGE_READWRITE          |
                PAGE_WRITECOPY
    )

    WRITEABLE = (
                PAGE_EXECUTE_READWRITE  |
                PAGE_EXECUTE_WRITECOPY  |
                PAGE_READWRITE          |
                PAGE_WRITECOPY
    )

    COPY_ON_WRITE = (
                PAGE_EXECUTE_WRITECOPY  |
                PAGE_WRITECOPY
    )

    EXECUTABLE = (
                PAGE_EXECUTE            |
                PAGE_EXECUTE_READ       |
                PAGE_EXECUTE_READWRITE  |
                PAGE_EXECUTE_WRITECOPY
    )

    EXECUTABLE_AND_WRITEABLE = (
                PAGE_EXECUTE_READWRITE  |
                PAGE_EXECUTE_WRITECOPY
    )

    def __init__(self, mbi=None):
        """
        @type  mbi: L{MEMORY_BASIC_INFORMATION} or L{MemoryBasicInformation}
        @param mbi: Either a L{MEMORY_BASIC_INFORMATION} structure or another
            L{MemoryBasicInformation} instance.
        """
        if mbi is None:
            self.BaseAddress        = None
            self.AllocationBase     = None
            self.AllocationProtect  = None
            self.RegionSize         = None
            self.State              = None
            self.Protect            = None
            self.Type               = None
        else:
            self.BaseAddress        = mbi.BaseAddress
            self.AllocationBase     = mbi.AllocationBase
            self.AllocationProtect  = mbi.AllocationProtect
            self.RegionSize         = mbi.RegionSize
            self.State              = mbi.State
            self.Protect            = mbi.Protect
            self.Type               = mbi.Type

            # Only used when copying MemoryBasicInformation objects, instead of
            # instancing them from a MEMORY_BASIC_INFORMATION structure.
            if hasattr(mbi, 'content'):
                self.content = mbi.content
            if hasattr(mbi, 'filename'):
                self.content = mbi.filename

    def __contains__(self, address):
        """
        Test if the given memory address falls within this memory region.

        @type  address: int
        @param address: Memory address to test.

        @rtype:  bool
        @return: C{True} if the given memory address falls within this memory
            region, C{False} otherwise.
        """
        return self.BaseAddress <= address < (self.BaseAddress + self.RegionSize)

    def is_free(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is free.
        """
        return self.State == MEM_FREE

    def is_reserved(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is reserved.
        """
        return self.State == MEM_RESERVE

    def is_commited(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is commited.
        """
        return self.State == MEM_COMMIT

    def is_image(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region belongs to an executable
            image.
        """
        return self.Type == MEM_IMAGE

    def is_mapped(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region belongs to a mapped file.
        """
        return self.Type == MEM_MAPPED

    def is_private(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is private.
        """
        return self.Type == MEM_PRIVATE

    def is_guard(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are guard pages.
        """
        return self.is_commited() and bool(self.Protect & PAGE_GUARD)

    def has_content(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region has any data in it.
        """
        return self.is_commited() and not bool(self.Protect & (PAGE_GUARD | PAGE_NOACCESS))

    def is_readable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are readable.
        """
        return self.has_content() and bool(self.Protect & self.READABLE)

    def is_writeable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are writeable.
        """
        return self.has_content() and bool(self.Protect & self.WRITEABLE)

    def is_copy_on_write(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are marked as
            copy-on-write. This means the pages are writeable, but changes
            are not propagated to disk.
        @note:
            Tipically data sections in executable images are marked like this.
        """
        return self.has_content() and bool(self.Protect & self.COPY_ON_WRITE)

    def is_executable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are executable.
        @note: Executable pages are always readable.
        """
        return self.has_content() and bool(self.Protect & self.EXECUTABLE)

    def is_executable_and_writeable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are executable and
            writeable.
        @note: The presence of such pages make memory corruption
            vulnerabilities much easier to exploit.
        """
        return self.has_content() and bool(self.Protect & self.EXECUTABLE_AND_WRITEABLE)

class ProcThreadAttributeList (object):
    """
    Extended process and thread attribute support.

    To be used with L{STARTUPINFOEX}.
    Only available for Windows Vista and above.

    @type AttributeList: list of tuple( int, ctypes-compatible object )
    @ivar AttributeList: List of (Attribute, Value) pairs.

    @type AttributeListBuffer: L{LPPROC_THREAD_ATTRIBUTE_LIST}
    @ivar AttributeListBuffer: Memory buffer used to store the attribute list.
        L{InitializeProcThreadAttributeList},
        L{UpdateProcThreadAttribute},
        L{DeleteProcThreadAttributeList} and
        L{STARTUPINFOEX}.
    """

    def __init__(self, AttributeList):
        """
        @type  AttributeList: list of tuple( int, ctypes-compatible object )
        @param AttributeList: List of (Attribute, Value) pairs.
        """
        self.AttributeList = AttributeList
        self.AttributeListBuffer = InitializeProcThreadAttributeList(
                                                            len(AttributeList))
        try:
            for Attribute, Value in AttributeList:
                UpdateProcThreadAttribute(self.AttributeListBuffer,
                                          Attribute, Value)
        except:
            ProcThreadAttributeList.__del__(self)
            raise

    def __del__(self):
        try:
            DeleteProcThreadAttributeList(self.AttributeListBuffer)
            del self.AttributeListBuffer
        except Exception:
            pass

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self):
        return self.__class__(self.AttributeList)

    @property
    def value(self):
        return ctypes.cast(ctypes.pointer(self.AttributeListBuffer), LPVOID)

    @property
    def _as_parameter_(self):
        return self.value

    # XXX TODO
    @staticmethod
    def from_param(value):
        raise NotImplementedError()

#--- OVERLAPPED structure -----------------------------------------------------

# typedef struct _OVERLAPPED {
#   ULONG_PTR Internal;
#   ULONG_PTR InternalHigh;
#   union {
#     struct {
#       DWORD Offset;
#       DWORD OffsetHigh;
#     } ;
#     PVOID Pointer;
#   } ;
#   HANDLE    hEvent;
# }OVERLAPPED, *LPOVERLAPPED;
class _OVERLAPPED_STRUCT(Structure):
    _fields_ = [
        ('Offset',          DWORD),
        ('OffsetHigh',      DWORD),
    ]
class _OVERLAPPED_UNION(Union):
    _fields_ = [
        ('s',               _OVERLAPPED_STRUCT),
        ('Pointer',         PVOID),
    ]
class OVERLAPPED(Structure):
    _fields_ = [
        ('Internal',        ULONG_PTR),
        ('InternalHigh',    ULONG_PTR),
        ('u',               _OVERLAPPED_UNION),
        ('hEvent',          HANDLE),
    ]
LPOVERLAPPED = POINTER(OVERLAPPED)

#--- SECURITY_ATTRIBUTES structure --------------------------------------------

# typedef struct _SECURITY_ATTRIBUTES {
#     DWORD nLength;
#     LPVOID lpSecurityDescriptor;
#     BOOL bInheritHandle;
# } SECURITY_ATTRIBUTES, *PSECURITY_ATTRIBUTES, *LPSECURITY_ATTRIBUTES;
class SECURITY_ATTRIBUTES(Structure):
    _fields_ = [
        ('nLength',                 DWORD),
        ('lpSecurityDescriptor',    LPVOID),
        ('bInheritHandle',          BOOL),
    ]
LPSECURITY_ATTRIBUTES = POINTER(SECURITY_ATTRIBUTES)

# --- Extended process and thread attribute support ---------------------------

PPROC_THREAD_ATTRIBUTE_LIST  = LPVOID
LPPROC_THREAD_ATTRIBUTE_LIST = PPROC_THREAD_ATTRIBUTE_LIST

PROC_THREAD_ATTRIBUTE_NUMBER   = 0x0000FFFF
PROC_THREAD_ATTRIBUTE_THREAD   = 0x00010000  # Attribute may be used with thread creation
PROC_THREAD_ATTRIBUTE_INPUT    = 0x00020000  # Attribute is input only
PROC_THREAD_ATTRIBUTE_ADDITIVE = 0x00040000  # Attribute may be "accumulated," e.g. bitmasks, counters, etc.

# PROC_THREAD_ATTRIBUTE_NUM
ProcThreadAttributeParentProcess    = 0
ProcThreadAttributeExtendedFlags    = 1
ProcThreadAttributeHandleList       = 2
ProcThreadAttributeGroupAffinity    = 3
ProcThreadAttributePreferredNode    = 4
ProcThreadAttributeIdealProcessor   = 5
ProcThreadAttributeUmsThread        = 6
ProcThreadAttributeMitigationPolicy = 7
ProcThreadAttributeMax              = 8

PROC_THREAD_ATTRIBUTE_PARENT_PROCESS    = ProcThreadAttributeParentProcess      |                                PROC_THREAD_ATTRIBUTE_INPUT
PROC_THREAD_ATTRIBUTE_EXTENDED_FLAGS    = ProcThreadAttributeExtendedFlags      |                                PROC_THREAD_ATTRIBUTE_INPUT | PROC_THREAD_ATTRIBUTE_ADDITIVE
PROC_THREAD_ATTRIBUTE_HANDLE_LIST       = ProcThreadAttributeHandleList         |                                PROC_THREAD_ATTRIBUTE_INPUT
PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY    = ProcThreadAttributeGroupAffinity      | PROC_THREAD_ATTRIBUTE_THREAD | PROC_THREAD_ATTRIBUTE_INPUT
PROC_THREAD_ATTRIBUTE_PREFERRED_NODE    = ProcThreadAttributePreferredNode      |                                PROC_THREAD_ATTRIBUTE_INPUT
PROC_THREAD_ATTRIBUTE_IDEAL_PROCESSOR   = ProcThreadAttributeIdealProcessor     | PROC_THREAD_ATTRIBUTE_THREAD | PROC_THREAD_ATTRIBUTE_INPUT
PROC_THREAD_ATTRIBUTE_UMS_THREAD        = ProcThreadAttributeUmsThread          | PROC_THREAD_ATTRIBUTE_THREAD | PROC_THREAD_ATTRIBUTE_INPUT
PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY = ProcThreadAttributeMitigationPolicy   |                                PROC_THREAD_ATTRIBUTE_INPUT

PROCESS_CREATION_MITIGATION_POLICY_DEP_ENABLE           = 0x01
PROCESS_CREATION_MITIGATION_POLICY_DEP_ATL_THUNK_ENABLE = 0x02
PROCESS_CREATION_MITIGATION_POLICY_SEHOP_ENABLE         = 0x04

#--- VS_FIXEDFILEINFO structure -----------------------------------------------

# struct VS_FIXEDFILEINFO {
#   DWORD dwSignature;
#   DWORD dwStrucVersion;
#   DWORD dwFileVersionMS;
#   DWORD dwFileVersionLS;
#   DWORD dwProductVersionMS;
#   DWORD dwProductVersionLS;
#   DWORD dwFileFlagsMask;
#   DWORD dwFileFlags;
#   DWORD dwFileOS;
#   DWORD dwFileType;
#   DWORD dwFileSubtype;
#   DWORD dwFileDateMS;
#   DWORD dwFileDateLS;
# };
class VS_FIXEDFILEINFO (Structure):
    _fields_ = [
        ("dwSignature",             DWORD),     # 0xFEEF04BD
        ("dwStrucVersion",          DWORD),
        ("dwFileVersionMS",         DWORD),
        ("dwFileVersionLS",         DWORD),
        ("dwProductVersionMS",      DWORD),
        ("dwProductVersionLS",      DWORD),
        ("dwFileFlagsMask",         DWORD),
        ("dwFileFlags",             DWORD),
        ("dwFileOS",                DWORD),
        ("dwFileType",              DWORD),
        ("dwFileSubtype",           DWORD),
        ("dwFileDateMS",            DWORD),
        ("dwFileDateLS",            DWORD),
    ]

#--- THREADNAME_INFO structure ------------------------------------------------

# typedef struct tagTHREADNAME_INFO
# {
#    DWORD dwType; // Must be 0x1000.
#    LPCSTR szName; // Pointer to name (in user addr space).
#    DWORD dwThreadID; // Thread ID (-1=caller thread).
#    DWORD dwFlags; // Reserved for future use, must be zero.
# } THREADNAME_INFO;
class THREADNAME_INFO(Structure):
    _fields_ = [
        ("dwType",      DWORD),     # 0x1000
        ("szName",      LPVOID),    # remote pointer
        ("dwThreadID",  DWORD),     # -1 usually
        ("dwFlags",     DWORD),     # 0
    ]

#--- MEMORY_BASIC_INFORMATION structure ---------------------------------------

# typedef struct _MEMORY_BASIC_INFORMATION32 {
#     DWORD BaseAddress;
#     DWORD AllocationBase;
#     DWORD AllocationProtect;
#     DWORD RegionSize;
#     DWORD State;
#     DWORD Protect;
#     DWORD Type;
# } MEMORY_BASIC_INFORMATION32, *PMEMORY_BASIC_INFORMATION32;
class MEMORY_BASIC_INFORMATION32(Structure):
    _fields_ = [
        ('BaseAddress',         DWORD),         # remote pointer
        ('AllocationBase',      DWORD),         # remote pointer
        ('AllocationProtect',   DWORD),
        ('RegionSize',          DWORD),
        ('State',               DWORD),
        ('Protect',             DWORD),
        ('Type',                DWORD),
    ]

# typedef struct DECLSPEC_ALIGN(16) _MEMORY_BASIC_INFORMATION64 {
#     ULONGLONG BaseAddress;
#     ULONGLONG AllocationBase;
#     DWORD     AllocationProtect;
#     DWORD     __alignment1;
#     ULONGLONG RegionSize;
#     DWORD     State;
#     DWORD     Protect;
#     DWORD     Type;
#     DWORD     __alignment2;
# } MEMORY_BASIC_INFORMATION64, *PMEMORY_BASIC_INFORMATION64;
class MEMORY_BASIC_INFORMATION64(Structure):
    _fields_ = [
        ('BaseAddress',         ULONGLONG),     # remote pointer
        ('AllocationBase',      ULONGLONG),     # remote pointer
        ('AllocationProtect',   DWORD),
        ('__alignment1',        DWORD),
        ('RegionSize',          ULONGLONG),
        ('State',               DWORD),
        ('Protect',             DWORD),
        ('Type',                DWORD),
        ('__alignment2',        DWORD),
    ]

# typedef struct _MEMORY_BASIC_INFORMATION {
#     PVOID BaseAddress;
#     PVOID AllocationBase;
#     DWORD AllocationProtect;
#     SIZE_T RegionSize;
#     DWORD State;
#     DWORD Protect;
#     DWORD Type;
# } MEMORY_BASIC_INFORMATION, *PMEMORY_BASIC_INFORMATION;
class MEMORY_BASIC_INFORMATION(Structure):
    _fields_ = [
        ('BaseAddress',         SIZE_T),    # remote pointer
        ('AllocationBase',      SIZE_T),    # remote pointer
        ('AllocationProtect',   DWORD),
        ('RegionSize',          SIZE_T),
        ('State',               DWORD),
        ('Protect',             DWORD),
        ('Type',                DWORD),
    ]
PMEMORY_BASIC_INFORMATION = POINTER(MEMORY_BASIC_INFORMATION)

#--- BY_HANDLE_FILE_INFORMATION structure -------------------------------------

# typedef struct _FILETIME {
#    DWORD dwLowDateTime;
#    DWORD dwHighDateTime;
# } FILETIME, *PFILETIME;
class FILETIME(Structure):
    _fields_ = [
        ('dwLowDateTime',       DWORD),
        ('dwHighDateTime',      DWORD),
    ]
LPFILETIME = POINTER(FILETIME)

# typedef struct _SYSTEMTIME {
#   WORD wYear;
#   WORD wMonth;
#   WORD wDayOfWeek;
#   WORD wDay;
#   WORD wHour;
#   WORD wMinute;
#   WORD wSecond;
#   WORD wMilliseconds;
# }SYSTEMTIME, *PSYSTEMTIME;
class SYSTEMTIME(Structure):
    _fields_ = [
        ('wYear',           WORD),
        ('wMonth',          WORD),
        ('wDayOfWeek',      WORD),
        ('wDay',            WORD),
        ('wHour',           WORD),
        ('wMinute',         WORD),
        ('wSecond',         WORD),
        ('wMilliseconds',   WORD),
    ]
LPSYSTEMTIME = POINTER(SYSTEMTIME)

# typedef struct _BY_HANDLE_FILE_INFORMATION {
#   DWORD dwFileAttributes;
#   FILETIME ftCreationTime;
#   FILETIME ftLastAccessTime;
#   FILETIME ftLastWriteTime;
#   DWORD dwVolumeSerialNumber;
#   DWORD nFileSizeHigh;
#   DWORD nFileSizeLow;
#   DWORD nNumberOfLinks;
#   DWORD nFileIndexHigh;
#   DWORD nFileIndexLow;
# } BY_HANDLE_FILE_INFORMATION, *PBY_HANDLE_FILE_INFORMATION;
class BY_HANDLE_FILE_INFORMATION(Structure):
    _fields_ = [
        ('dwFileAttributes',        DWORD),
        ('ftCreationTime',          FILETIME),
        ('ftLastAccessTime',        FILETIME),
        ('ftLastWriteTime',         FILETIME),
        ('dwVolumeSerialNumber',    DWORD),
        ('nFileSizeHigh',           DWORD),
        ('nFileSizeLow',            DWORD),
        ('nNumberOfLinks',          DWORD),
        ('nFileIndexHigh',          DWORD),
        ('nFileIndexLow',           DWORD),
    ]
LPBY_HANDLE_FILE_INFORMATION = POINTER(BY_HANDLE_FILE_INFORMATION)

# typedef enum _FILE_INFO_BY_HANDLE_CLASS {
#   FileBasicInfo = 0,
#   FileStandardInfo = 1,
#   FileNameInfo = 2,
#   FileRenameInfo = 3,
#   FileDispositionInfo = 4,
#   FileAllocationInfo = 5,
#   FileEndOfFileInfo = 6,
#   FileStreamInfo = 7,
#   FileCompressionInfo = 8,
#   FileAttributeTagInfo = 9,
#   FileIdBothDirectoryInfo = 10,
#   FileIdBothDirectoryRestartInfo = 11,
#   FileIoPriorityHintInfo = 12,
#   MaximumFileInfoByHandlesClass = 13
# } FILE_INFO_BY_HANDLE_CLASS, *PFILE_INFO_BY_HANDLE_CLASS;
class FILE_INFO_BY_HANDLE_CLASS(object):
    FileBasicInfo                   = 0
    FileStandardInfo                = 1
    FileNameInfo                    = 2
    FileRenameInfo                  = 3
    FileDispositionInfo             = 4
    FileAllocationInfo              = 5
    FileEndOfFileInfo               = 6
    FileStreamInfo                  = 7
    FileCompressionInfo             = 8
    FileAttributeTagInfo            = 9
    FileIdBothDirectoryInfo         = 10
    FileIdBothDirectoryRestartInfo  = 11
    FileIoPriorityHintInfo          = 12
    MaximumFileInfoByHandlesClass   = 13

# typedef struct _FILE_NAME_INFO {
#   DWORD  FileNameLength;
#   WCHAR FileName[1];
# } FILE_NAME_INFO, *PFILE_NAME_INFO;
##class FILE_NAME_INFO(Structure):
##    _fields_ = [
##        ('FileNameLength',  DWORD),
##        ('FileName',        WCHAR * 1),
##    ]

# TO DO: add more structures used by GetFileInformationByHandleEx()

#--- PROCESS_INFORMATION structure --------------------------------------------

# typedef struct _PROCESS_INFORMATION {
#     HANDLE hProcess;
#     HANDLE hThread;
#     DWORD dwProcessId;
#     DWORD dwThreadId;
# } PROCESS_INFORMATION, *PPROCESS_INFORMATION, *LPPROCESS_INFORMATION;
class PROCESS_INFORMATION(Structure):
    _fields_ = [
        ('hProcess',    HANDLE),
        ('hThread',     HANDLE),
        ('dwProcessId', DWORD),
        ('dwThreadId',  DWORD),
    ]
LPPROCESS_INFORMATION = POINTER(PROCESS_INFORMATION)

#--- STARTUPINFO and STARTUPINFOEX structures ---------------------------------

# typedef struct _STARTUPINFO {
#   DWORD  cb;
#   LPTSTR lpReserved;
#   LPTSTR lpDesktop;
#   LPTSTR lpTitle;
#   DWORD  dwX;
#   DWORD  dwY;
#   DWORD  dwXSize;
#   DWORD  dwYSize;
#   DWORD  dwXCountChars;
#   DWORD  dwYCountChars;
#   DWORD  dwFillAttribute;
#   DWORD  dwFlags;
#   WORD   wShowWindow;
#   WORD   cbReserved2;
#   LPBYTE lpReserved2;
#   HANDLE hStdInput;
#   HANDLE hStdOutput;
#   HANDLE hStdError;
# }STARTUPINFO, *LPSTARTUPINFO;
class STARTUPINFO(Structure):
    _fields_ = [
        ('cb',              DWORD),
        ('lpReserved',      LPSTR),
        ('lpDesktop',       LPSTR),
        ('lpTitle',         LPSTR),
        ('dwX',             DWORD),
        ('dwY',             DWORD),
        ('dwXSize',         DWORD),
        ('dwYSize',         DWORD),
        ('dwXCountChars',   DWORD),
        ('dwYCountChars',   DWORD),
        ('dwFillAttribute', DWORD),
        ('dwFlags',         DWORD),
        ('wShowWindow',     WORD),
        ('cbReserved2',     WORD),
        ('lpReserved2',     LPVOID),    # LPBYTE
        ('hStdInput',       HANDLE),
        ('hStdOutput',      HANDLE),
        ('hStdError',       HANDLE),
    ]
LPSTARTUPINFO = POINTER(STARTUPINFO)

# typedef struct _STARTUPINFOEX {
#   STARTUPINFO StartupInfo;
#   PPROC_THREAD_ATTRIBUTE_LIST lpAttributeList;
# } STARTUPINFOEX,  *LPSTARTUPINFOEX;
class STARTUPINFOEX(Structure):
    _fields_ = [
        ('StartupInfo',     STARTUPINFO),
        ('lpAttributeList', PPROC_THREAD_ATTRIBUTE_LIST),
    ]
LPSTARTUPINFOEX = POINTER(STARTUPINFOEX)

class STARTUPINFOW(Structure):
    _fields_ = [
        ('cb',              DWORD),
        ('lpReserved',      LPWSTR),
        ('lpDesktop',       LPWSTR),
        ('lpTitle',         LPWSTR),
        ('dwX',             DWORD),
        ('dwY',             DWORD),
        ('dwXSize',         DWORD),
        ('dwYSize',         DWORD),
        ('dwXCountChars',   DWORD),
        ('dwYCountChars',   DWORD),
        ('dwFillAttribute', DWORD),
        ('dwFlags',         DWORD),
        ('wShowWindow',     WORD),
        ('cbReserved2',     WORD),
        ('lpReserved2',     LPVOID),    # LPBYTE
        ('hStdInput',       HANDLE),
        ('hStdOutput',      HANDLE),
        ('hStdError',       HANDLE),
    ]
LPSTARTUPINFOW = POINTER(STARTUPINFOW)

class STARTUPINFOEXW(Structure):
    _fields_ = [
        ('StartupInfo',     STARTUPINFOW),
        ('lpAttributeList', PPROC_THREAD_ATTRIBUTE_LIST),
    ]
LPSTARTUPINFOEXW = POINTER(STARTUPINFOEXW)

#--- JIT_DEBUG_INFO structure -------------------------------------------------

# typedef struct _JIT_DEBUG_INFO {
#     DWORD dwSize;
#     DWORD dwProcessorArchitecture;
#     DWORD dwThreadID;
#     DWORD dwReserved0;
#     ULONG64 lpExceptionAddress;
#     ULONG64 lpExceptionRecord;
#     ULONG64 lpContextRecord;
# } JIT_DEBUG_INFO, *LPJIT_DEBUG_INFO;
class JIT_DEBUG_INFO(Structure):
    _fields_ = [
        ('dwSize',                  DWORD),
        ('dwProcessorArchitecture', DWORD),
        ('dwThreadID',              DWORD),
        ('dwReserved0',             DWORD),
        ('lpExceptionAddress',      ULONG64),
        ('lpExceptionRecord',       ULONG64),
        ('lpContextRecord',         ULONG64),
    ]
JIT_DEBUG_INFO32 = JIT_DEBUG_INFO
JIT_DEBUG_INFO64 = JIT_DEBUG_INFO

LPJIT_DEBUG_INFO   = POINTER(JIT_DEBUG_INFO)
LPJIT_DEBUG_INFO32 = POINTER(JIT_DEBUG_INFO32)
LPJIT_DEBUG_INFO64 = POINTER(JIT_DEBUG_INFO64)

#--- DEBUG_EVENT structure ----------------------------------------------------

# typedef struct _EXCEPTION_RECORD32 {
#     DWORD ExceptionCode;
#     DWORD ExceptionFlags;
#     DWORD ExceptionRecord;
#     DWORD ExceptionAddress;
#     DWORD NumberParameters;
#     DWORD ExceptionInformation[EXCEPTION_MAXIMUM_PARAMETERS];
# } EXCEPTION_RECORD32, *PEXCEPTION_RECORD32;
class EXCEPTION_RECORD32(Structure):
    _fields_ = [
        ('ExceptionCode',           DWORD),
        ('ExceptionFlags',          DWORD),
        ('ExceptionRecord',         DWORD),
        ('ExceptionAddress',        DWORD),
        ('NumberParameters',        DWORD),
        ('ExceptionInformation',    DWORD * EXCEPTION_MAXIMUM_PARAMETERS),
    ]

PEXCEPTION_RECORD32 = POINTER(EXCEPTION_RECORD32)

# typedef struct _EXCEPTION_RECORD64 {
#     DWORD    ExceptionCode;
#     DWORD ExceptionFlags;
#     DWORD64 ExceptionRecord;
#     DWORD64 ExceptionAddress;
#     DWORD NumberParameters;
#     DWORD __unusedAlignment;
#     DWORD64 ExceptionInformation[EXCEPTION_MAXIMUM_PARAMETERS];
# } EXCEPTION_RECORD64, *PEXCEPTION_RECORD64;
class EXCEPTION_RECORD64(Structure):
    _fields_ = [
        ('ExceptionCode',           DWORD),
        ('ExceptionFlags',          DWORD),
        ('ExceptionRecord',         DWORD64),
        ('ExceptionAddress',        DWORD64),
        ('NumberParameters',        DWORD),
        ('__unusedAlignment',       DWORD),
        ('ExceptionInformation',    DWORD64 * EXCEPTION_MAXIMUM_PARAMETERS),
    ]

PEXCEPTION_RECORD64 = POINTER(EXCEPTION_RECORD64)

# typedef struct _EXCEPTION_RECORD {
#     DWORD ExceptionCode;
#     DWORD ExceptionFlags;
#     LPVOID ExceptionRecord;
#     LPVOID ExceptionAddress;
#     DWORD NumberParameters;
#     LPVOID ExceptionInformation[EXCEPTION_MAXIMUM_PARAMETERS];
# } EXCEPTION_RECORD, *PEXCEPTION_RECORD;
class EXCEPTION_RECORD(Structure):
    pass
PEXCEPTION_RECORD = POINTER(EXCEPTION_RECORD)
EXCEPTION_RECORD._fields_ = [
        ('ExceptionCode',           DWORD),
        ('ExceptionFlags',          DWORD),
        ('ExceptionRecord',         PEXCEPTION_RECORD),
        ('ExceptionAddress',        LPVOID),
        ('NumberParameters',        DWORD),
        ('ExceptionInformation',    LPVOID * EXCEPTION_MAXIMUM_PARAMETERS),
    ]

# typedef struct _EXCEPTION_DEBUG_INFO {
#   EXCEPTION_RECORD ExceptionRecord;
#   DWORD dwFirstChance;
# } EXCEPTION_DEBUG_INFO;
class EXCEPTION_DEBUG_INFO(Structure):
    _fields_ = [
        ('ExceptionRecord',     EXCEPTION_RECORD),
        ('dwFirstChance',       DWORD),
    ]

# typedef struct _CREATE_THREAD_DEBUG_INFO {
#   HANDLE hThread;
#   LPVOID lpThreadLocalBase;
#   LPTHREAD_START_ROUTINE lpStartAddress;
# } CREATE_THREAD_DEBUG_INFO;
class CREATE_THREAD_DEBUG_INFO(Structure):
    _fields_ = [
        ('hThread',             HANDLE),
        ('lpThreadLocalBase',   LPVOID),
        ('lpStartAddress',      LPVOID),
    ]

# typedef struct _CREATE_PROCESS_DEBUG_INFO {
#   HANDLE hFile;
#   HANDLE hProcess;
#   HANDLE hThread;
#   LPVOID lpBaseOfImage;
#   DWORD dwDebugInfoFileOffset;
#   DWORD nDebugInfoSize;
#   LPVOID lpThreadLocalBase;
#   LPTHREAD_START_ROUTINE lpStartAddress;
#   LPVOID lpImageName;
#   WORD fUnicode;
# } CREATE_PROCESS_DEBUG_INFO;
class CREATE_PROCESS_DEBUG_INFO(Structure):
    _fields_ = [
        ('hFile',                   HANDLE),
        ('hProcess',                HANDLE),
        ('hThread',                 HANDLE),
        ('lpBaseOfImage',           LPVOID),
        ('dwDebugInfoFileOffset',   DWORD),
        ('nDebugInfoSize',          DWORD),
        ('lpThreadLocalBase',       LPVOID),
        ('lpStartAddress',          LPVOID),
        ('lpImageName',             LPVOID),
        ('fUnicode',                WORD),
    ]

# typedef struct _EXIT_THREAD_DEBUG_INFO {
#   DWORD dwExitCode;
# } EXIT_THREAD_DEBUG_INFO;
class EXIT_THREAD_DEBUG_INFO(Structure):
    _fields_ = [
        ('dwExitCode',          DWORD),
    ]

# typedef struct _EXIT_PROCESS_DEBUG_INFO {
#   DWORD dwExitCode;
# } EXIT_PROCESS_DEBUG_INFO;
class EXIT_PROCESS_DEBUG_INFO(Structure):
    _fields_ = [
        ('dwExitCode',          DWORD),
    ]

# typedef struct _LOAD_DLL_DEBUG_INFO {
#   HANDLE hFile;
#   LPVOID lpBaseOfDll;
#   DWORD dwDebugInfoFileOffset;
#   DWORD nDebugInfoSize;
#   LPVOID lpImageName;
#   WORD fUnicode;
# } LOAD_DLL_DEBUG_INFO;
class LOAD_DLL_DEBUG_INFO(Structure):
    _fields_ = [
        ('hFile',                   HANDLE),
        ('lpBaseOfDll',             LPVOID),
        ('dwDebugInfoFileOffset',   DWORD),
        ('nDebugInfoSize',          DWORD),
        ('lpImageName',             LPVOID),
        ('fUnicode',                WORD),
    ]

# typedef struct _UNLOAD_DLL_DEBUG_INFO {
#   LPVOID lpBaseOfDll;
# } UNLOAD_DLL_DEBUG_INFO;
class UNLOAD_DLL_DEBUG_INFO(Structure):
    _fields_ = [
        ('lpBaseOfDll',         LPVOID),
    ]

# typedef struct _OUTPUT_DEBUG_STRING_INFO {
#   LPSTR lpDebugStringData;
#   WORD fUnicode;
#   WORD nDebugStringLength;
# } OUTPUT_DEBUG_STRING_INFO;
class OUTPUT_DEBUG_STRING_INFO(Structure):
    _fields_ = [
        ('lpDebugStringData',   LPVOID),    # don't use LPSTR
        ('fUnicode',            WORD),
        ('nDebugStringLength',  WORD),
    ]

# typedef struct _RIP_INFO {
#     DWORD dwError;
#     DWORD dwType;
# } RIP_INFO, *LPRIP_INFO;
class RIP_INFO(Structure):
    _fields_ = [
        ('dwError',             DWORD),
        ('dwType',              DWORD),
    ]

# typedef struct _DEBUG_EVENT {
#   DWORD dwDebugEventCode;
#   DWORD dwProcessId;
#   DWORD dwThreadId;
#   union {
#     EXCEPTION_DEBUG_INFO Exception;
#     CREATE_THREAD_DEBUG_INFO CreateThread;
#     CREATE_PROCESS_DEBUG_INFO CreateProcessInfo;
#     EXIT_THREAD_DEBUG_INFO ExitThread;
#     EXIT_PROCESS_DEBUG_INFO ExitProcess;
#     LOAD_DLL_DEBUG_INFO LoadDll;
#     UNLOAD_DLL_DEBUG_INFO UnloadDll;
#     OUTPUT_DEBUG_STRING_INFO DebugString;
#     RIP_INFO RipInfo;
#   } u;
# } DEBUG_EVENT;.
class _DEBUG_EVENT_UNION_(Union):
    _fields_ = [
        ('Exception',           EXCEPTION_DEBUG_INFO),
        ('CreateThread',        CREATE_THREAD_DEBUG_INFO),
        ('CreateProcessInfo',   CREATE_PROCESS_DEBUG_INFO),
        ('ExitThread',          EXIT_THREAD_DEBUG_INFO),
        ('ExitProcess',         EXIT_PROCESS_DEBUG_INFO),
        ('LoadDll',             LOAD_DLL_DEBUG_INFO),
        ('UnloadDll',           UNLOAD_DLL_DEBUG_INFO),
        ('DebugString',         OUTPUT_DEBUG_STRING_INFO),
        ('RipInfo',             RIP_INFO),
    ]
class DEBUG_EVENT(Structure):
    _fields_ = [
        ('dwDebugEventCode',    DWORD),
        ('dwProcessId',         DWORD),
        ('dwThreadId',          DWORD),
        ('u',                   _DEBUG_EVENT_UNION_),
    ]
LPDEBUG_EVENT = POINTER(DEBUG_EVENT)

#--- Console API defines and structures ---------------------------------------

FOREGROUND_MASK = 0x000F
BACKGROUND_MASK = 0x00F0
COMMON_LVB_MASK = 0xFF00

FOREGROUND_BLACK     = 0x0000
FOREGROUND_BLUE      = 0x0001
FOREGROUND_GREEN     = 0x0002
FOREGROUND_CYAN      = 0x0003
FOREGROUND_RED       = 0x0004
FOREGROUND_MAGENTA   = 0x0005
FOREGROUND_YELLOW    = 0x0006
FOREGROUND_GREY      = 0x0007
FOREGROUND_INTENSITY = 0x0008

BACKGROUND_BLACK     = 0x0000
BACKGROUND_BLUE      = 0x0010
BACKGROUND_GREEN     = 0x0020
BACKGROUND_CYAN      = 0x0030
BACKGROUND_RED       = 0x0040
BACKGROUND_MAGENTA   = 0x0050
BACKGROUND_YELLOW    = 0x0060
BACKGROUND_GREY      = 0x0070
BACKGROUND_INTENSITY = 0x0080

COMMON_LVB_LEADING_BYTE    = 0x0100
COMMON_LVB_TRAILING_BYTE   = 0x0200
COMMON_LVB_GRID_HORIZONTAL = 0x0400
COMMON_LVB_GRID_LVERTICAL  = 0x0800
COMMON_LVB_GRID_RVERTICAL  = 0x1000
COMMON_LVB_REVERSE_VIDEO   = 0x4000
COMMON_LVB_UNDERSCORE      = 0x8000

# typedef struct _CHAR_INFO {
#   union {
#     WCHAR UnicodeChar;
#     CHAR  AsciiChar;
#   } Char;
#   WORD  Attributes;
# } CHAR_INFO, *PCHAR_INFO;
class _CHAR_INFO_CHAR(Union):
    _fields_ = [
        ('UnicodeChar', WCHAR),
        ('AsciiChar',   CHAR),
    ]
class CHAR_INFO(Structure):
    _fields_ = [
        ('Char',       _CHAR_INFO_CHAR),
        ('Attributes', WORD),
   ]
PCHAR_INFO = POINTER(CHAR_INFO)

# typedef struct _COORD {
#   SHORT X;
#   SHORT Y;
# } COORD, *PCOORD;
class COORD(Structure):
    _fields_ = [
        ('X', SHORT),
        ('Y', SHORT),
    ]
PCOORD = POINTER(COORD)

# typedef struct _SMALL_RECT {
#   SHORT Left;
#   SHORT Top;
#   SHORT Right;
#   SHORT Bottom;
# } SMALL_RECT;
class SMALL_RECT(Structure):
    _fields_ = [
        ('Left',   SHORT),
        ('Top',    SHORT),
        ('Right',  SHORT),
        ('Bottom', SHORT),
    ]
PSMALL_RECT = POINTER(SMALL_RECT)

# typedef struct _CONSOLE_SCREEN_BUFFER_INFO {
#   COORD      dwSize;
#   COORD      dwCursorPosition;
#   WORD       wAttributes;
#   SMALL_RECT srWindow;
#   COORD      dwMaximumWindowSize;
# } CONSOLE_SCREEN_BUFFER_INFO;
class CONSOLE_SCREEN_BUFFER_INFO(Structure):
    _fields_ = [
        ('dwSize',              COORD),
        ('dwCursorPosition',    COORD),
        ('wAttributes',         WORD),
        ('srWindow',            SMALL_RECT),
        ('dwMaximumWindowSize', COORD),
    ]
PCONSOLE_SCREEN_BUFFER_INFO = POINTER(CONSOLE_SCREEN_BUFFER_INFO)

#--- Toolhelp library defines and structures ----------------------------------

TH32CS_SNAPHEAPLIST = 0x00000001
TH32CS_SNAPPROCESS  = 0x00000002
TH32CS_SNAPTHREAD   = 0x00000004
TH32CS_SNAPMODULE   = 0x00000008
TH32CS_INHERIT      = 0x80000000
TH32CS_SNAPALL      = (TH32CS_SNAPHEAPLIST | TH32CS_SNAPPROCESS | TH32CS_SNAPTHREAD | TH32CS_SNAPMODULE)

# typedef struct tagTHREADENTRY32 {
#   DWORD dwSize;
#   DWORD cntUsage;
#   DWORD th32ThreadID;
#   DWORD th32OwnerProcessID;
#   LONG tpBasePri;
#   LONG tpDeltaPri;
#   DWORD dwFlags;
# } THREADENTRY32,  *PTHREADENTRY32;
class THREADENTRY32(Structure):
    _fields_ = [
        ('dwSize',             DWORD),
        ('cntUsage',           DWORD),
        ('th32ThreadID',       DWORD),
        ('th32OwnerProcessID', DWORD),
        ('tpBasePri',          LONG),
        ('tpDeltaPri',         LONG),
        ('dwFlags',            DWORD),
    ]
LPTHREADENTRY32 = POINTER(THREADENTRY32)

# typedef struct tagPROCESSENTRY32 {
#    DWORD dwSize;
#    DWORD cntUsage;
#    DWORD th32ProcessID;
#    ULONG_PTR th32DefaultHeapID;
#    DWORD th32ModuleID;
#    DWORD cntThreads;
#    DWORD th32ParentProcessID;
#    LONG pcPriClassBase;
#    DWORD dwFlags;
#    TCHAR szExeFile[MAX_PATH];
# } PROCESSENTRY32,  *PPROCESSENTRY32;
class PROCESSENTRY32(Structure):
    _fields_ = [
        ('dwSize',              DWORD),
        ('cntUsage',            DWORD),
        ('th32ProcessID',       DWORD),
        ('th32DefaultHeapID',   ULONG_PTR),
        ('th32ModuleID',        DWORD),
        ('cntThreads',          DWORD),
        ('th32ParentProcessID', DWORD),
        ('pcPriClassBase',      LONG),
        ('dwFlags',             DWORD),
        ('szExeFile',           TCHAR * 260),
    ]
LPPROCESSENTRY32 = POINTER(PROCESSENTRY32)

# typedef struct tagMODULEENTRY32 {
#   DWORD dwSize;
#   DWORD th32ModuleID;
#   DWORD th32ProcessID;
#   DWORD GlblcntUsage;
#   DWORD ProccntUsage;
#   BYTE* modBaseAddr;
#   DWORD modBaseSize;
#   HMODULE hModule;
#   TCHAR szModule[MAX_MODULE_NAME32 + 1];
#   TCHAR szExePath[MAX_PATH];
# } MODULEENTRY32,  *PMODULEENTRY32;
class MODULEENTRY32(Structure):
    _fields_ = [
        ("dwSize",        DWORD),
        ("th32ModuleID",  DWORD),
        ("th32ProcessID", DWORD),
        ("GlblcntUsage",  DWORD),
        ("ProccntUsage",  DWORD),
        ("modBaseAddr",   LPVOID),  # BYTE*
        ("modBaseSize",   DWORD),
        ("hModule",       HMODULE),
        ("szModule",      TCHAR * (MAX_MODULE_NAME32 + 1)),
        ("szExePath",     TCHAR * MAX_PATH),
    ]
LPMODULEENTRY32 = POINTER(MODULEENTRY32)

# typedef struct tagHEAPENTRY32 {
#   SIZE_T    dwSize;
#   HANDLE    hHandle;
#   ULONG_PTR dwAddress;
#   SIZE_T    dwBlockSize;
#   DWORD     dwFlags;
#   DWORD     dwLockCount;
#   DWORD     dwResvd;
#   DWORD     th32ProcessID;
#   ULONG_PTR th32HeapID;
# } HEAPENTRY32,
# *PHEAPENTRY32;
class HEAPENTRY32(Structure):
    _fields_ = [
        ("dwSize",          SIZE_T),
        ("hHandle",         HANDLE),
        ("dwAddress",       ULONG_PTR),
        ("dwBlockSize",     SIZE_T),
        ("dwFlags",         DWORD),
        ("dwLockCount",     DWORD),
        ("dwResvd",         DWORD),
        ("th32ProcessID",   DWORD),
        ("th32HeapID",      ULONG_PTR),
]
LPHEAPENTRY32 = POINTER(HEAPENTRY32)

# typedef struct tagHEAPLIST32 {
#   SIZE_T    dwSize;
#   DWORD     th32ProcessID;
#   ULONG_PTR th32HeapID;
#   DWORD     dwFlags;
# } HEAPLIST32,
#  *PHEAPLIST32;
class HEAPLIST32(Structure):
    _fields_ = [
        ("dwSize",          SIZE_T),
        ("th32ProcessID",   DWORD),
        ("th32HeapID",      ULONG_PTR),
        ("dwFlags",         DWORD),
]
LPHEAPLIST32 = POINTER(HEAPLIST32)

#--- kernel32.dll -------------------------------------------------------------

# DWORD WINAPI GetLastError(void);
def GetLastError():
    _GetLastError = windll.kernel32.GetLastError
    _GetLastError.argtypes = []
    _GetLastError.restype  = DWORD
    return _GetLastError()

# void WINAPI SetLastError(
#   __in  DWORD dwErrCode
# );
def SetLastError(dwErrCode):
    _SetLastError = windll.kernel32.SetLastError
    _SetLastError.argtypes = [DWORD]
    _SetLastError.restype  = None
    _SetLastError(dwErrCode)

# UINT WINAPI GetErrorMode(void);
def GetErrorMode():
    _GetErrorMode = windll.kernel32.GetErrorMode
    _GetErrorMode.argtypes = []
    _GetErrorMode.restype  = UINT
    return _GetErrorMode()

# UINT WINAPI SetErrorMode(
#   __in  UINT uMode
# );
def SetErrorMode(uMode):
    _SetErrorMode = windll.kernel32.SetErrorMode
    _SetErrorMode.argtypes = [UINT]
    _SetErrorMode.restype  = UINT
    return _SetErrorMode(dwErrCode)

# DWORD GetThreadErrorMode(void);
def GetThreadErrorMode():
    _GetThreadErrorMode = windll.kernel32.GetThreadErrorMode
    _GetThreadErrorMode.argtypes = []
    _GetThreadErrorMode.restype  = DWORD
    return _GetThreadErrorMode()

# BOOL SetThreadErrorMode(
#   __in   DWORD dwNewMode,
#   __out  LPDWORD lpOldMode
# );
def SetThreadErrorMode(dwNewMode):
    _SetThreadErrorMode = windll.kernel32.SetThreadErrorMode
    _SetThreadErrorMode.argtypes = [DWORD, LPDWORD]
    _SetThreadErrorMode.restype  = BOOL
    _SetThreadErrorMode.errcheck = RaiseIfZero

    old = DWORD(0)
    _SetThreadErrorMode(dwErrCode, byref(old))
    return old.value

# BOOL WINAPI CloseHandle(
#   __in  HANDLE hObject
# );
def CloseHandle(hHandle):
    if isinstance(hHandle, Handle):
        # Prevents the handle from being closed without notifying the Handle object.
        hHandle.close()
    else:
        _CloseHandle = windll.kernel32.CloseHandle
        _CloseHandle.argtypes = [HANDLE]
        _CloseHandle.restype  = bool
        _CloseHandle.errcheck = RaiseIfZero
        _CloseHandle(hHandle)

# BOOL WINAPI DuplicateHandle(
#   __in   HANDLE hSourceProcessHandle,
#   __in   HANDLE hSourceHandle,
#   __in   HANDLE hTargetProcessHandle,
#   __out  LPHANDLE lpTargetHandle,
#   __in   DWORD dwDesiredAccess,
#   __in   BOOL bInheritHandle,
#   __in   DWORD dwOptions
# );
def DuplicateHandle(hSourceHandle, hSourceProcessHandle = None, hTargetProcessHandle = None, dwDesiredAccess = STANDARD_RIGHTS_ALL, bInheritHandle = False, dwOptions = DUPLICATE_SAME_ACCESS):
    _DuplicateHandle = windll.kernel32.DuplicateHandle
    _DuplicateHandle.argtypes = [HANDLE, HANDLE, HANDLE, LPHANDLE, DWORD, BOOL, DWORD]
    _DuplicateHandle.restype  = bool
    _DuplicateHandle.errcheck = RaiseIfZero

    # NOTE: the arguments to this function are in a different order,
    # so we can set default values for all of them but one (hSourceHandle).

    if hSourceProcessHandle is None:
        hSourceProcessHandle = GetCurrentProcess()
    if hTargetProcessHandle is None:
        hTargetProcessHandle = hSourceProcessHandle
    lpTargetHandle = HANDLE(INVALID_HANDLE_VALUE)
    _DuplicateHandle(hSourceProcessHandle, hSourceHandle, hTargetProcessHandle, byref(lpTargetHandle), dwDesiredAccess, bool(bInheritHandle), dwOptions)
    if isinstance(hSourceHandle, Handle):
        HandleClass = hSourceHandle.__class__
    else:
        HandleClass = Handle
    if hasattr(hSourceHandle, 'dwAccess'):
        return HandleClass(lpTargetHandle.value, dwAccess = hSourceHandle.dwAccess)
    else:
        return HandleClass(lpTargetHandle.value)

# HLOCAL WINAPI LocalFree(
#   __in  HLOCAL hMem
# );
def LocalFree(hMem):
    _LocalFree = windll.kernel32.LocalFree
    _LocalFree.argtypes = [HLOCAL]
    _LocalFree.restype  = HLOCAL

    result = _LocalFree(hMem)
    if result != NULL:
        ctypes.WinError()

#------------------------------------------------------------------------------
# Console API

# HANDLE WINAPI GetStdHandle(
#   _In_  DWORD nStdHandle
# );
def GetStdHandle(nStdHandle):
    _GetStdHandle = windll.kernel32.GetStdHandle
    _GetStdHandle.argytpes = [DWORD]
    _GetStdHandle.restype  = HANDLE
    _GetStdHandle.errcheck = RaiseIfZero
    return Handle( _GetStdHandle(nStdHandle), bOwnership = False )

# BOOL WINAPI SetStdHandle(
#   _In_  DWORD nStdHandle,
#   _In_  HANDLE hHandle
# );

# TODO

# UINT WINAPI GetConsoleCP(void);
def GetConsoleCP():
    _GetConsoleCP = windll.kernel32.GetConsoleCP
    _GetConsoleCP.argytpes = []
    _GetConsoleCP.restype  = UINT
    return _GetConsoleCP()

# UINT WINAPI GetConsoleOutputCP(void);
def GetConsoleOutputCP():
    _GetConsoleOutputCP = windll.kernel32.GetConsoleOutputCP
    _GetConsoleOutputCP.argytpes = []
    _GetConsoleOutputCP.restype  = UINT
    return _GetConsoleOutputCP()

#BOOL WINAPI SetConsoleCP(
#  _In_  UINT wCodePageID
#);
def SetConsoleCP(wCodePageID):
    _SetConsoleCP = windll.kernel32.SetConsoleCP
    _SetConsoleCP.argytpes = [UINT]
    _SetConsoleCP.restype  = bool
    _SetConsoleCP.errcheck = RaiseIfZero
    _SetConsoleCP(wCodePageID)

#BOOL WINAPI SetConsoleOutputCP(
#  _In_  UINT wCodePageID
#);
def SetConsoleOutputCP(wCodePageID):
    _SetConsoleOutputCP = windll.kernel32.SetConsoleOutputCP
    _SetConsoleOutputCP.argytpes = [UINT]
    _SetConsoleOutputCP.restype  = bool
    _SetConsoleOutputCP.errcheck = RaiseIfZero
    _SetConsoleOutputCP(wCodePageID)

# HANDLE WINAPI CreateConsoleScreenBuffer(
#   _In_        DWORD dwDesiredAccess,
#   _In_        DWORD dwShareMode,
#   _In_opt_    const SECURITY_ATTRIBUTES *lpSecurityAttributes,
#   _In_        DWORD dwFlags,
#   _Reserved_  LPVOID lpScreenBufferData
# );

# TODO

# BOOL WINAPI SetConsoleActiveScreenBuffer(
#   _In_  HANDLE hConsoleOutput
# );
def SetConsoleActiveScreenBuffer(hConsoleOutput = None):
    _SetConsoleActiveScreenBuffer = windll.kernel32.SetConsoleActiveScreenBuffer
    _SetConsoleActiveScreenBuffer.argytpes = [HANDLE]
    _SetConsoleActiveScreenBuffer.restype  = bool
    _SetConsoleActiveScreenBuffer.errcheck = RaiseIfZero

    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    _SetConsoleActiveScreenBuffer(hConsoleOutput)

# BOOL WINAPI GetConsoleScreenBufferInfo(
#   _In_   HANDLE hConsoleOutput,
#   _Out_  PCONSOLE_SCREEN_BUFFER_INFO lpConsoleScreenBufferInfo
# );
def GetConsoleScreenBufferInfo(hConsoleOutput = None):
    _GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
    _GetConsoleScreenBufferInfo.argytpes = [HANDLE, PCONSOLE_SCREEN_BUFFER_INFO]
    _GetConsoleScreenBufferInfo.restype  = bool
    _GetConsoleScreenBufferInfo.errcheck = RaiseIfZero

    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    ConsoleScreenBufferInfo = CONSOLE_SCREEN_BUFFER_INFO()
    _GetConsoleScreenBufferInfo(hConsoleOutput, byref(ConsoleScreenBufferInfo))
    return ConsoleScreenBufferInfo

# BOOL WINAPI GetConsoleScreenBufferInfoEx(
#   _In_   HANDLE hConsoleOutput,
#   _Out_  PCONSOLE_SCREEN_BUFFER_INFOEX lpConsoleScreenBufferInfoEx
# );

# TODO

# BOOL WINAPI SetConsoleWindowInfo(
#   _In_  HANDLE hConsoleOutput,
#   _In_  BOOL bAbsolute,
#   _In_  const SMALL_RECT *lpConsoleWindow
# );
def SetConsoleWindowInfo(hConsoleOutput, bAbsolute, lpConsoleWindow):
    _SetConsoleWindowInfo = windll.kernel32.SetConsoleWindowInfo
    _SetConsoleWindowInfo.argytpes = [HANDLE, BOOL, PSMALL_RECT]
    _SetConsoleWindowInfo.restype  = bool
    _SetConsoleWindowInfo.errcheck = RaiseIfZero

    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    if isinstance(lpConsoleWindow, SMALL_RECT):
        ConsoleWindow = lpConsoleWindow
    else:
        ConsoleWindow = SMALL_RECT(*lpConsoleWindow)
    _SetConsoleWindowInfo(hConsoleOutput, bAbsolute, byref(ConsoleWindow))

# BOOL WINAPI SetConsoleTextAttribute(
#   _In_  HANDLE hConsoleOutput,
#   _In_  WORD wAttributes
# );
def SetConsoleTextAttribute(hConsoleOutput = None, wAttributes = 0):
    _SetConsoleTextAttribute = windll.kernel32.SetConsoleTextAttribute
    _SetConsoleTextAttribute.argytpes = [HANDLE, WORD]
    _SetConsoleTextAttribute.restype  = bool
    _SetConsoleTextAttribute.errcheck = RaiseIfZero

    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    _SetConsoleTextAttribute(hConsoleOutput, wAttributes)

# HANDLE WINAPI CreateConsoleScreenBuffer(
#   _In_        DWORD dwDesiredAccess,
#   _In_        DWORD dwShareMode,
#   _In_opt_    const SECURITY_ATTRIBUTES *lpSecurityAttributes,
#   _In_        DWORD dwFlags,
#   _Reserved_  LPVOID lpScreenBufferData
# );

# TODO

# BOOL WINAPI AllocConsole(void);
def AllocConsole():
    _AllocConsole = windll.kernel32.AllocConsole
    _AllocConsole.argytpes = []
    _AllocConsole.restype  = bool
    _AllocConsole.errcheck = RaiseIfZero
    _AllocConsole()

# BOOL WINAPI AttachConsole(
#   _In_  DWORD dwProcessId
# );
def AttachConsole(dwProcessId = ATTACH_PARENT_PROCESS):
    _AttachConsole = windll.kernel32.AttachConsole
    _AttachConsole.argytpes = [DWORD]
    _AttachConsole.restype  = bool
    _AttachConsole.errcheck = RaiseIfZero
    _AttachConsole(dwProcessId)

# BOOL WINAPI FreeConsole(void);
def FreeConsole():
    _FreeConsole = windll.kernel32.FreeConsole
    _FreeConsole.argytpes = []
    _FreeConsole.restype  = bool
    _FreeConsole.errcheck = RaiseIfZero
    _FreeConsole()

# DWORD WINAPI GetConsoleProcessList(
#   _Out_  LPDWORD lpdwProcessList,
#   _In_   DWORD dwProcessCount
# );

# TODO

# DWORD WINAPI GetConsoleTitle(
#   _Out_  LPTSTR lpConsoleTitle,
#   _In_   DWORD nSize
# );

# TODO

#BOOL WINAPI SetConsoleTitle(
#  _In_  LPCTSTR lpConsoleTitle
#);

# TODO

# COORD WINAPI GetLargestConsoleWindowSize(
#   _In_  HANDLE hConsoleOutput
# );

# TODO

# BOOL WINAPI GetConsoleHistoryInfo(
#   _Out_  PCONSOLE_HISTORY_INFO lpConsoleHistoryInfo
# );

# TODO

#------------------------------------------------------------------------------
# DLL API

# DWORD WINAPI GetDllDirectory(
#   __in   DWORD nBufferLength,
#   __out  LPTSTR lpBuffer
# );
def GetDllDirectoryA():
    _GetDllDirectoryA = windll.kernel32.GetDllDirectoryA
    _GetDllDirectoryA.argytpes = [DWORD, LPSTR]
    _GetDllDirectoryA.restype  = DWORD

    nBufferLength = _GetDllDirectoryA(0, None)
    if nBufferLength == 0:
        return None
    lpBuffer = ctypes.create_string_buffer("", nBufferLength)
    _GetDllDirectoryA(nBufferLength, byref(lpBuffer))
    return lpBuffer.value

def GetDllDirectoryW():
    _GetDllDirectoryW = windll.kernel32.GetDllDirectoryW
    _GetDllDirectoryW.argytpes = [DWORD, LPWSTR]
    _GetDllDirectoryW.restype  = DWORD

    nBufferLength = _GetDllDirectoryW(0, None)
    if nBufferLength == 0:
        return None
    lpBuffer = ctypes.create_unicode_buffer(u"", nBufferLength)
    _GetDllDirectoryW(nBufferLength, byref(lpBuffer))
    return lpBuffer.value

GetDllDirectory = GuessStringType(GetDllDirectoryA, GetDllDirectoryW)

# BOOL WINAPI SetDllDirectory(
#   __in_opt  LPCTSTR lpPathName
# );
def SetDllDirectoryA(lpPathName = None):
    _SetDllDirectoryA = windll.kernel32.SetDllDirectoryA
    _SetDllDirectoryA.argytpes = [LPSTR]
    _SetDllDirectoryA.restype  = bool
    _SetDllDirectoryA.errcheck = RaiseIfZero
    _SetDllDirectoryA(lpPathName)

def SetDllDirectoryW(lpPathName):
    _SetDllDirectoryW = windll.kernel32.SetDllDirectoryW
    _SetDllDirectoryW.argytpes = [LPWSTR]
    _SetDllDirectoryW.restype  = bool
    _SetDllDirectoryW.errcheck = RaiseIfZero
    _SetDllDirectoryW(lpPathName)

SetDllDirectory = GuessStringType(SetDllDirectoryA, SetDllDirectoryW)

# HMODULE WINAPI LoadLibrary(
#   __in  LPCTSTR lpFileName
# );
def LoadLibraryA(pszLibrary):
    _LoadLibraryA = windll.kernel32.LoadLibraryA
    _LoadLibraryA.argtypes = [LPSTR]
    _LoadLibraryA.restype  = HMODULE
    hModule = _LoadLibraryA(pszLibrary)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule

def LoadLibraryW(pszLibrary):
    _LoadLibraryW = windll.kernel32.LoadLibraryW
    _LoadLibraryW.argtypes = [LPWSTR]
    _LoadLibraryW.restype  = HMODULE
    hModule = _LoadLibraryW(pszLibrary)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule

LoadLibrary = GuessStringType(LoadLibraryA, LoadLibraryW)

# HMODULE WINAPI LoadLibraryEx(
#   __in        LPCTSTR lpFileName,
#   __reserved  HANDLE hFile,
#   __in        DWORD dwFlags
# );
def LoadLibraryExA(pszLibrary, dwFlags = 0):
    _LoadLibraryExA = windll.kernel32.LoadLibraryExA
    _LoadLibraryExA.argtypes = [LPSTR, HANDLE, DWORD]
    _LoadLibraryExA.restype  = HMODULE
    hModule = _LoadLibraryExA(pszLibrary, NULL, dwFlags)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule

def LoadLibraryExW(pszLibrary, dwFlags = 0):
    _LoadLibraryExW = windll.kernel32.LoadLibraryExW
    _LoadLibraryExW.argtypes = [LPWSTR, HANDLE, DWORD]
    _LoadLibraryExW.restype  = HMODULE
    hModule = _LoadLibraryExW(pszLibrary, NULL, dwFlags)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule

LoadLibraryEx = GuessStringType(LoadLibraryExA, LoadLibraryExW)

# HMODULE WINAPI GetModuleHandle(
#   __in_opt  LPCTSTR lpModuleName
# );
def GetModuleHandleA(lpModuleName):
    _GetModuleHandleA = windll.kernel32.GetModuleHandleA
    _GetModuleHandleA.argtypes = [LPSTR]
    _GetModuleHandleA.restype  = HMODULE
    hModule = _GetModuleHandleA(lpModuleName)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule

def GetModuleHandleW(lpModuleName):
    _GetModuleHandleW = windll.kernel32.GetModuleHandleW
    _GetModuleHandleW.argtypes = [LPWSTR]
    _GetModuleHandleW.restype  = HMODULE
    hModule = _GetModuleHandleW(lpModuleName)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule

GetModuleHandle = GuessStringType(GetModuleHandleA, GetModuleHandleW)

# FARPROC WINAPI GetProcAddress(
#   __in  HMODULE hModule,
#   __in  LPCSTR lpProcName
# );
def GetProcAddressA(hModule, lpProcName):
    _GetProcAddress = windll.kernel32.GetProcAddress
    _GetProcAddress.argtypes = [HMODULE, LPVOID]
    _GetProcAddress.restype  = LPVOID

    if type(lpProcName) in (type(0), type(long(0))):
        lpProcName = LPVOID(lpProcName)
        if lpProcName.value & (~0xFFFF):
            raise ValueError('Ordinal number too large: %d' % lpProcName.value)
    elif type(lpProcName) == type(compat.b("")):
        lpProcName = ctypes.c_char_p(lpProcName)
    else:
        raise TypeError(str(type(lpProcName)))
    return _GetProcAddress(hModule, lpProcName)

GetProcAddressW = MakeWideVersion(GetProcAddressA)
GetProcAddress = GuessStringType(GetProcAddressA, GetProcAddressW)

# BOOL WINAPI FreeLibrary(
#   __in  HMODULE hModule
# );
def FreeLibrary(hModule):
    _FreeLibrary = windll.kernel32.FreeLibrary
    _FreeLibrary.argtypes = [HMODULE]
    _FreeLibrary.restype  = bool
    _FreeLibrary.errcheck = RaiseIfZero
    _FreeLibrary(hModule)

# PVOID WINAPI RtlPcToFileHeader(
#   __in   PVOID PcValue,
#   __out  PVOID *BaseOfImage
# );
def RtlPcToFileHeader(PcValue):
    _RtlPcToFileHeader = windll.kernel32.RtlPcToFileHeader
    _RtlPcToFileHeader.argtypes = [PVOID, POINTER(PVOID)]
    _RtlPcToFileHeader.restype  = PRUNTIME_FUNCTION

    BaseOfImage = PVOID(0)
    _RtlPcToFileHeader(PcValue, byref(BaseOfImage))
    return BaseOfImage.value

#------------------------------------------------------------------------------
# File API and related

# BOOL WINAPI GetHandleInformation(
#   __in   HANDLE hObject,
#   __out  LPDWORD lpdwFlags
# );
def GetHandleInformation(hObject):
    _GetHandleInformation = windll.kernel32.GetHandleInformation
    _GetHandleInformation.argtypes = [HANDLE, PDWORD]
    _GetHandleInformation.restype  = bool
    _GetHandleInformation.errcheck = RaiseIfZero

    dwFlags = DWORD(0)
    _GetHandleInformation(hObject, byref(dwFlags))
    return dwFlags.value

# BOOL WINAPI SetHandleInformation(
#   __in  HANDLE hObject,
#   __in  DWORD dwMask,
#   __in  DWORD dwFlags
# );
def SetHandleInformation(hObject, dwMask, dwFlags):
    _SetHandleInformation = windll.kernel32.SetHandleInformation
    _SetHandleInformation.argtypes = [HANDLE, DWORD, DWORD]
    _SetHandleInformation.restype  = bool
    _SetHandleInformation.errcheck = RaiseIfZero
    _SetHandleInformation(hObject, dwMask, dwFlags)

# UINT WINAPI GetWindowModuleFileName(
#   __in   HWND hwnd,
#   __out  LPTSTR lpszFileName,
#   __in   UINT cchFileNameMax
# );
# Not included because it doesn't work in other processes.
# See: http://support.microsoft.com/?id=228469

# BOOL WINAPI QueryFullProcessImageName(
#   __in     HANDLE hProcess,
#   __in     DWORD dwFlags,
#   __out    LPTSTR lpExeName,
#   __inout  PDWORD lpdwSize
# );
def QueryFullProcessImageNameA(hProcess, dwFlags = 0):
    _QueryFullProcessImageNameA = windll.kernel32.QueryFullProcessImageNameA
    _QueryFullProcessImageNameA.argtypes = [HANDLE, DWORD, LPSTR, PDWORD]
    _QueryFullProcessImageNameA.restype  = bool

    dwSize = MAX_PATH
    while 1:
        lpdwSize = DWORD(dwSize)
        lpExeName = ctypes.create_string_buffer('', lpdwSize.value + 1)
        success = _QueryFullProcessImageNameA(hProcess, dwFlags, lpExeName, byref(lpdwSize))
        if success and 0 < lpdwSize.value < dwSize:
            break
        error = GetLastError()
        if error != ERROR_INSUFFICIENT_BUFFER:
            raise ctypes.WinError(error)
        dwSize = dwSize + 256
        if dwSize > 0x1000:
            # this prevents an infinite loop in Windows 2008 when the path has spaces,
            # see http://msdn.microsoft.com/en-us/library/ms684919(VS.85).aspx#4
            raise ctypes.WinError(error)
    return lpExeName.value

def QueryFullProcessImageNameW(hProcess, dwFlags = 0):
    _QueryFullProcessImageNameW = windll.kernel32.QueryFullProcessImageNameW
    _QueryFullProcessImageNameW.argtypes = [HANDLE, DWORD, LPWSTR, PDWORD]
    _QueryFullProcessImageNameW.restype  = bool

    dwSize = MAX_PATH
    while 1:
        lpdwSize = DWORD(dwSize)
        lpExeName = ctypes.create_unicode_buffer('', lpdwSize.value + 1)
        success = _QueryFullProcessImageNameW(hProcess, dwFlags, lpExeName, byref(lpdwSize))
        if success and 0 < lpdwSize.value < dwSize:
            break
        error = GetLastError()
        if error != ERROR_INSUFFICIENT_BUFFER:
            raise ctypes.WinError(error)
        dwSize = dwSize + 256
        if dwSize > 0x1000:
            # this prevents an infinite loop in Windows 2008 when the path has spaces,
            # see http://msdn.microsoft.com/en-us/library/ms684919(VS.85).aspx#4
            raise ctypes.WinError(error)
    return lpExeName.value

QueryFullProcessImageName = GuessStringType(QueryFullProcessImageNameA, QueryFullProcessImageNameW)

# DWORD WINAPI GetLogicalDriveStrings(
#   __in   DWORD nBufferLength,
#   __out  LPTSTR lpBuffer
# );
def GetLogicalDriveStringsA():
    _GetLogicalDriveStringsA = ctypes.windll.kernel32.GetLogicalDriveStringsA
    _GetLogicalDriveStringsA.argtypes = [DWORD, LPSTR]
    _GetLogicalDriveStringsA.restype  = DWORD
    _GetLogicalDriveStringsA.errcheck = RaiseIfZero

    nBufferLength = (4 * 26) + 1    # "X:\\\0" from A to Z plus empty string
    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
    _GetLogicalDriveStringsA(nBufferLength, lpBuffer)
    drive_strings = list()
    string_p = addressof(lpBuffer)
    sizeof_char = sizeof(ctypes.c_char)
    while True:
        string_v = ctypes.string_at(string_p)
        if string_v == '':
            break
        drive_strings.append(string_v)
        string_p += len(string_v) + sizeof_char
    return drive_strings

def GetLogicalDriveStringsW():
    _GetLogicalDriveStringsW = ctypes.windll.kernel32.GetLogicalDriveStringsW
    _GetLogicalDriveStringsW.argtypes = [DWORD, LPWSTR]
    _GetLogicalDriveStringsW.restype  = DWORD
    _GetLogicalDriveStringsW.errcheck = RaiseIfZero

    nBufferLength = (4 * 26) + 1    # "X:\\\0" from A to Z plus empty string
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
    _GetLogicalDriveStringsW(nBufferLength, lpBuffer)
    drive_strings = list()
    string_p = addressof(lpBuffer)
    sizeof_wchar = sizeof(ctypes.c_wchar)
    while True:
        string_v = ctypes.wstring_at(string_p)
        if string_v == u'':
            break
        drive_strings.append(string_v)
        string_p += (len(string_v) * sizeof_wchar) + sizeof_wchar
    return drive_strings

##def GetLogicalDriveStringsA():
##    _GetLogicalDriveStringsA = windll.kernel32.GetLogicalDriveStringsA
##    _GetLogicalDriveStringsA.argtypes = [DWORD, LPSTR]
##    _GetLogicalDriveStringsA.restype  = DWORD
##    _GetLogicalDriveStringsA.errcheck = RaiseIfZero
##
##    nBufferLength = (4 * 26) + 1    # "X:\\\0" from A to Z plus empty string
##    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
##    _GetLogicalDriveStringsA(nBufferLength, lpBuffer)
##    result = list()
##    index = 0
##    while 1:
##        string = list()
##        while 1:
##            character = lpBuffer[index]
##            index = index + 1
##            if character == '\0':
##                break
##            string.append(character)
##        if not string:
##            break
##        result.append(''.join(string))
##    return result
##
##def GetLogicalDriveStringsW():
##    _GetLogicalDriveStringsW = windll.kernel32.GetLogicalDriveStringsW
##    _GetLogicalDriveStringsW.argtypes = [DWORD, LPWSTR]
##    _GetLogicalDriveStringsW.restype  = DWORD
##    _GetLogicalDriveStringsW.errcheck = RaiseIfZero
##
##    nBufferLength = (4 * 26) + 1    # "X:\\\0" from A to Z plus empty string
##    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
##    _GetLogicalDriveStringsW(nBufferLength, lpBuffer)
##    result = list()
##    index = 0
##    while 1:
##        string = list()
##        while 1:
##            character = lpBuffer[index]
##            index = index + 1
##            if character == u'\0':
##                break
##            string.append(character)
##        if not string:
##            break
##        result.append(u''.join(string))
##    return result

GetLogicalDriveStrings = GuessStringType(GetLogicalDriveStringsA, GetLogicalDriveStringsW)

# DWORD WINAPI QueryDosDevice(
#   __in_opt  LPCTSTR lpDeviceName,
#   __out     LPTSTR lpTargetPath,
#   __in      DWORD ucchMax
# );
def QueryDosDeviceA(lpDeviceName = None):
    _QueryDosDeviceA = windll.kernel32.QueryDosDeviceA
    _QueryDosDeviceA.argtypes = [LPSTR, LPSTR, DWORD]
    _QueryDosDeviceA.restype  = DWORD
    _QueryDosDeviceA.errcheck = RaiseIfZero

    if not lpDeviceName:
        lpDeviceName = None
    ucchMax = 0x1000
    lpTargetPath = ctypes.create_string_buffer('', ucchMax)
    _QueryDosDeviceA(lpDeviceName, lpTargetPath, ucchMax)
    return lpTargetPath.value

def QueryDosDeviceW(lpDeviceName):
    _QueryDosDeviceW = windll.kernel32.QueryDosDeviceW
    _QueryDosDeviceW.argtypes = [LPWSTR, LPWSTR, DWORD]
    _QueryDosDeviceW.restype  = DWORD
    _QueryDosDeviceW.errcheck = RaiseIfZero

    if not lpDeviceName:
        lpDeviceName = None
    ucchMax = 0x1000
    lpTargetPath = ctypes.create_unicode_buffer(u'', ucchMax)
    _QueryDosDeviceW(lpDeviceName, lpTargetPath, ucchMax)
    return lpTargetPath.value

QueryDosDevice = GuessStringType(QueryDosDeviceA, QueryDosDeviceW)

# LPVOID WINAPI MapViewOfFile(
#   __in  HANDLE hFileMappingObject,
#   __in  DWORD dwDesiredAccess,
#   __in  DWORD dwFileOffsetHigh,
#   __in  DWORD dwFileOffsetLow,
#   __in  SIZE_T dwNumberOfBytesToMap
# );
def MapViewOfFile(hFileMappingObject, dwDesiredAccess = FILE_MAP_ALL_ACCESS | FILE_MAP_EXECUTE, dwFileOffsetHigh = 0, dwFileOffsetLow = 0, dwNumberOfBytesToMap = 0):
    _MapViewOfFile = windll.kernel32.MapViewOfFile
    _MapViewOfFile.argtypes = [HANDLE, DWORD, DWORD, DWORD, SIZE_T]
    _MapViewOfFile.restype  = LPVOID
    lpBaseAddress = _MapViewOfFile(hFileMappingObject, dwDesiredAccess, dwFileOffsetHigh, dwFileOffsetLow, dwNumberOfBytesToMap)
    if lpBaseAddress == NULL:
        raise ctypes.WinError()
    return lpBaseAddress

# BOOL WINAPI UnmapViewOfFile(
#   __in  LPCVOID lpBaseAddress
# );
def UnmapViewOfFile(lpBaseAddress):
    _UnmapViewOfFile = windll.kernel32.UnmapViewOfFile
    _UnmapViewOfFile.argtypes = [LPVOID]
    _UnmapViewOfFile.restype  = bool
    _UnmapViewOfFile.errcheck = RaiseIfZero
    _UnmapViewOfFile(lpBaseAddress)

# HANDLE WINAPI OpenFileMapping(
#   __in  DWORD dwDesiredAccess,
#   __in  BOOL bInheritHandle,
#   __in  LPCTSTR lpName
# );
def OpenFileMappingA(dwDesiredAccess, bInheritHandle, lpName):
    _OpenFileMappingA = windll.kernel32.OpenFileMappingA
    _OpenFileMappingA.argtypes = [DWORD, BOOL, LPSTR]
    _OpenFileMappingA.restype  = HANDLE
    _OpenFileMappingA.errcheck = RaiseIfZero
    hFileMappingObject = _OpenFileMappingA(dwDesiredAccess, bool(bInheritHandle), lpName)
    return FileMappingHandle(hFileMappingObject)

def OpenFileMappingW(dwDesiredAccess, bInheritHandle, lpName):
    _OpenFileMappingW = windll.kernel32.OpenFileMappingW
    _OpenFileMappingW.argtypes = [DWORD, BOOL, LPWSTR]
    _OpenFileMappingW.restype  = HANDLE
    _OpenFileMappingW.errcheck = RaiseIfZero
    hFileMappingObject = _OpenFileMappingW(dwDesiredAccess, bool(bInheritHandle), lpName)
    return FileMappingHandle(hFileMappingObject)

OpenFileMapping = GuessStringType(OpenFileMappingA, OpenFileMappingW)

# HANDLE WINAPI CreateFileMapping(
#   __in      HANDLE hFile,
#   __in_opt  LPSECURITY_ATTRIBUTES lpAttributes,
#   __in      DWORD flProtect,
#   __in      DWORD dwMaximumSizeHigh,
#   __in      DWORD dwMaximumSizeLow,
#   __in_opt  LPCTSTR lpName
# );
def CreateFileMappingA(hFile, lpAttributes = None, flProtect = PAGE_EXECUTE_READWRITE, dwMaximumSizeHigh = 0, dwMaximumSizeLow = 0, lpName = None):
    _CreateFileMappingA = windll.kernel32.CreateFileMappingA
    _CreateFileMappingA.argtypes = [HANDLE, LPVOID, DWORD, DWORD, DWORD, LPSTR]
    _CreateFileMappingA.restype  = HANDLE
    _CreateFileMappingA.errcheck = RaiseIfZero

    if lpAttributes:
        lpAttributes = ctypes.pointer(lpAttributes)
    if not lpName:
        lpName = None
    hFileMappingObject = _CreateFileMappingA(hFile, lpAttributes, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, lpName)
    return FileMappingHandle(hFileMappingObject)

def CreateFileMappingW(hFile, lpAttributes = None, flProtect = PAGE_EXECUTE_READWRITE, dwMaximumSizeHigh = 0, dwMaximumSizeLow = 0, lpName = None):
    _CreateFileMappingW = windll.kernel32.CreateFileMappingW
    _CreateFileMappingW.argtypes = [HANDLE, LPVOID, DWORD, DWORD, DWORD, LPWSTR]
    _CreateFileMappingW.restype  = HANDLE
    _CreateFileMappingW.errcheck = RaiseIfZero

    if lpAttributes:
        lpAttributes = ctypes.pointer(lpAttributes)
    if not lpName:
        lpName = None
    hFileMappingObject = _CreateFileMappingW(hFile, lpAttributes, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, lpName)
    return FileMappingHandle(hFileMappingObject)

CreateFileMapping = GuessStringType(CreateFileMappingA, CreateFileMappingW)

# HANDLE WINAPI CreateFile(
#   __in      LPCTSTR lpFileName,
#   __in      DWORD dwDesiredAccess,
#   __in      DWORD dwShareMode,
#   __in_opt  LPSECURITY_ATTRIBUTES lpSecurityAttributes,
#   __in      DWORD dwCreationDisposition,
#   __in      DWORD dwFlagsAndAttributes,
#   __in_opt  HANDLE hTemplateFile
# );
def CreateFileA(lpFileName, dwDesiredAccess = GENERIC_ALL, dwShareMode = 0, lpSecurityAttributes = None, dwCreationDisposition = OPEN_ALWAYS, dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL, hTemplateFile = None):
    _CreateFileA = windll.kernel32.CreateFileA
    _CreateFileA.argtypes = [LPSTR, DWORD, DWORD, LPVOID, DWORD, DWORD, HANDLE]
    _CreateFileA.restype  = HANDLE

    if not lpFileName:
        lpFileName = None
    if lpSecurityAttributes:
        lpSecurityAttributes = ctypes.pointer(lpSecurityAttributes)
    hFile = _CreateFileA(lpFileName, dwDesiredAccess, dwShareMode, lpSecurityAttributes, dwCreationDisposition, dwFlagsAndAttributes, hTemplateFile)
    if hFile == INVALID_HANDLE_VALUE:
        raise ctypes.WinError()
    return FileHandle(hFile)

def CreateFileW(lpFileName, dwDesiredAccess = GENERIC_ALL, dwShareMode = 0, lpSecurityAttributes = None, dwCreationDisposition = OPEN_ALWAYS, dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL, hTemplateFile = None):
    _CreateFileW = windll.kernel32.CreateFileW
    _CreateFileW.argtypes = [LPWSTR, DWORD, DWORD, LPVOID, DWORD, DWORD, HANDLE]
    _CreateFileW.restype  = HANDLE

    if not lpFileName:
        lpFileName = None
    if lpSecurityAttributes:
        lpSecurityAttributes = ctypes.pointer(lpSecurityAttributes)
    hFile = _CreateFileW(lpFileName, dwDesiredAccess, dwShareMode, lpSecurityAttributes, dwCreationDisposition, dwFlagsAndAttributes, hTemplateFile)
    if hFile == INVALID_HANDLE_VALUE:
        raise ctypes.WinError()
    return FileHandle(hFile)

CreateFile = GuessStringType(CreateFileA, CreateFileW)

# BOOL WINAPI FlushFileBuffers(
#   __in  HANDLE hFile
# );
def FlushFileBuffers(hFile):
    _FlushFileBuffers = windll.kernel32.FlushFileBuffers
    _FlushFileBuffers.argtypes = [HANDLE]
    _FlushFileBuffers.restype  = bool
    _FlushFileBuffers.errcheck = RaiseIfZero
    _FlushFileBuffers(hFile)

# BOOL WINAPI FlushViewOfFile(
#   __in  LPCVOID lpBaseAddress,
#   __in  SIZE_T dwNumberOfBytesToFlush
# );
def FlushViewOfFile(lpBaseAddress, dwNumberOfBytesToFlush = 0):
    _FlushViewOfFile = windll.kernel32.FlushViewOfFile
    _FlushViewOfFile.argtypes = [LPVOID, SIZE_T]
    _FlushViewOfFile.restype  = bool
    _FlushViewOfFile.errcheck = RaiseIfZero
    _FlushViewOfFile(lpBaseAddress, dwNumberOfBytesToFlush)

# DWORD WINAPI SearchPath(
#   __in_opt   LPCTSTR lpPath,
#   __in       LPCTSTR lpFileName,
#   __in_opt   LPCTSTR lpExtension,
#   __in       DWORD nBufferLength,
#   __out      LPTSTR lpBuffer,
#   __out_opt  LPTSTR *lpFilePart
# );
def SearchPathA(lpPath, lpFileName, lpExtension):
    _SearchPathA = windll.kernel32.SearchPathA
    _SearchPathA.argtypes = [LPSTR, LPSTR, LPSTR, DWORD, LPSTR, POINTER(LPSTR)]
    _SearchPathA.restype  = DWORD
    _SearchPathA.errcheck = RaiseIfZero

    if not lpPath:
        lpPath = None
    if not lpExtension:
        lpExtension = None
    nBufferLength = _SearchPathA(lpPath, lpFileName, lpExtension, 0, None, None)
    lpBuffer = ctypes.create_string_buffer('', nBufferLength + 1)
    lpFilePart = LPSTR()
    _SearchPathA(lpPath, lpFileName, lpExtension, nBufferLength, lpBuffer, byref(lpFilePart))
    lpFilePart = lpFilePart.value
    lpBuffer = lpBuffer.value
    if lpBuffer == '':
        if GetLastError() == ERROR_SUCCESS:
            raise ctypes.WinError(ERROR_FILE_NOT_FOUND)
        raise ctypes.WinError()
    return (lpBuffer, lpFilePart)

def SearchPathW(lpPath, lpFileName, lpExtension):
    _SearchPathW = windll.kernel32.SearchPathW
    _SearchPathW.argtypes = [LPWSTR, LPWSTR, LPWSTR, DWORD, LPWSTR, POINTER(LPWSTR)]
    _SearchPathW.restype  = DWORD
    _SearchPathW.errcheck = RaiseIfZero

    if not lpPath:
        lpPath = None
    if not lpExtension:
        lpExtension = None
    nBufferLength = _SearchPathW(lpPath, lpFileName, lpExtension, 0, None, None)
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength + 1)
    lpFilePart = LPWSTR()
    _SearchPathW(lpPath, lpFileName, lpExtension, nBufferLength, lpBuffer, byref(lpFilePart))
    lpFilePart = lpFilePart.value
    lpBuffer = lpBuffer.value
    if lpBuffer == u'':
        if GetLastError() == ERROR_SUCCESS:
            raise ctypes.WinError(ERROR_FILE_NOT_FOUND)
        raise ctypes.WinError()
    return (lpBuffer, lpFilePart)

SearchPath = GuessStringType(SearchPathA, SearchPathW)

# BOOL SetSearchPathMode(
#   __in  DWORD Flags
# );
def SetSearchPathMode(Flags):
    _SetSearchPathMode = windll.kernel32.SetSearchPathMode
    _SetSearchPathMode.argtypes = [DWORD]
    _SetSearchPathMode.restype  = bool
    _SetSearchPathMode.errcheck = RaiseIfZero
    _SetSearchPathMode(Flags)

# BOOL WINAPI DeviceIoControl(
#   __in         HANDLE hDevice,
#   __in         DWORD dwIoControlCode,
#   __in_opt     LPVOID lpInBuffer,
#   __in         DWORD nInBufferSize,
#   __out_opt    LPVOID lpOutBuffer,
#   __in         DWORD nOutBufferSize,
#   __out_opt    LPDWORD lpBytesReturned,
#   __inout_opt  LPOVERLAPPED lpOverlapped
# );
def DeviceIoControl(hDevice, dwIoControlCode, lpInBuffer, nInBufferSize, lpOutBuffer, nOutBufferSize, lpOverlapped):
    _DeviceIoControl = windll.kernel32.DeviceIoControl
    _DeviceIoControl.argtypes = [HANDLE, DWORD, LPVOID, DWORD, LPVOID, DWORD, LPDWORD, LPOVERLAPPED]
    _DeviceIoControl.restype  = bool
    _DeviceIoControl.errcheck = RaiseIfZero

    if not lpInBuffer:
        lpInBuffer = None
    if not lpOutBuffer:
        lpOutBuffer = None
    if lpOverlapped:
        lpOverlapped = ctypes.pointer(lpOverlapped)
    lpBytesReturned = DWORD(0)
    _DeviceIoControl(hDevice, dwIoControlCode, lpInBuffer, nInBufferSize, lpOutBuffer, nOutBufferSize, byref(lpBytesReturned), lpOverlapped)
    return lpBytesReturned.value

# BOOL GetFileInformationByHandle(
#   HANDLE hFile,
#   LPBY_HANDLE_FILE_INFORMATION lpFileInformation
# );
def GetFileInformationByHandle(hFile):
    _GetFileInformationByHandle = windll.kernel32.GetFileInformationByHandle
    _GetFileInformationByHandle.argtypes = [HANDLE, LPBY_HANDLE_FILE_INFORMATION]
    _GetFileInformationByHandle.restype  = bool
    _GetFileInformationByHandle.errcheck = RaiseIfZero

    lpFileInformation = BY_HANDLE_FILE_INFORMATION()
    _GetFileInformationByHandle(hFile, byref(lpFileInformation))
    return lpFileInformation

# BOOL WINAPI GetFileInformationByHandleEx(
#   __in   HANDLE hFile,
#   __in   FILE_INFO_BY_HANDLE_CLASS FileInformationClass,
#   __out  LPVOID lpFileInformation,
#   __in   DWORD dwBufferSize
# );
def GetFileInformationByHandleEx(hFile, FileInformationClass, lpFileInformation, dwBufferSize):
    _GetFileInformationByHandleEx = windll.kernel32.GetFileInformationByHandleEx
    _GetFileInformationByHandleEx.argtypes = [HANDLE, DWORD, LPVOID, DWORD]
    _GetFileInformationByHandleEx.restype  = bool
    _GetFileInformationByHandleEx.errcheck = RaiseIfZero
    # XXX TODO
    # support each FileInformationClass so the function can allocate the
    # corresponding structure for the lpFileInformation parameter
    _GetFileInformationByHandleEx(hFile, FileInformationClass, byref(lpFileInformation), dwBufferSize)

# DWORD WINAPI GetFinalPathNameByHandle(
#   __in   HANDLE hFile,
#   __out  LPTSTR lpszFilePath,
#   __in   DWORD cchFilePath,
#   __in   DWORD dwFlags
# );
def GetFinalPathNameByHandleA(hFile, dwFlags = FILE_NAME_NORMALIZED | VOLUME_NAME_DOS):
    _GetFinalPathNameByHandleA = windll.kernel32.GetFinalPathNameByHandleA
    _GetFinalPathNameByHandleA.argtypes = [HANDLE, LPSTR, DWORD, DWORD]
    _GetFinalPathNameByHandleA.restype  = DWORD

    cchFilePath = _GetFinalPathNameByHandleA(hFile, None, 0, dwFlags)
    if cchFilePath == 0:
        raise ctypes.WinError()
    lpszFilePath = ctypes.create_string_buffer('', cchFilePath + 1)
    nCopied = _GetFinalPathNameByHandleA(hFile, lpszFilePath, cchFilePath, dwFlags)
    if nCopied <= 0 or nCopied > cchFilePath:
        raise ctypes.WinError()
    return lpszFilePath.value

def GetFinalPathNameByHandleW(hFile, dwFlags = FILE_NAME_NORMALIZED | VOLUME_NAME_DOS):
    _GetFinalPathNameByHandleW = windll.kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = [HANDLE, LPWSTR, DWORD, DWORD]
    _GetFinalPathNameByHandleW.restype  = DWORD

    cchFilePath = _GetFinalPathNameByHandleW(hFile, None, 0, dwFlags)
    if cchFilePath == 0:
        raise ctypes.WinError()
    lpszFilePath = ctypes.create_unicode_buffer(u'', cchFilePath + 1)
    nCopied = _GetFinalPathNameByHandleW(hFile, lpszFilePath, cchFilePath, dwFlags)
    if nCopied <= 0 or nCopied > cchFilePath:
        raise ctypes.WinError()
    return lpszFilePath.value

GetFinalPathNameByHandle = GuessStringType(GetFinalPathNameByHandleA, GetFinalPathNameByHandleW)

# DWORD GetFullPathName(
#   LPCTSTR lpFileName,
#   DWORD nBufferLength,
#   LPTSTR lpBuffer,
#   LPTSTR* lpFilePart
# );
def GetFullPathNameA(lpFileName):
    _GetFullPathNameA = windll.kernel32.GetFullPathNameA
    _GetFullPathNameA.argtypes = [LPSTR, DWORD, LPSTR, POINTER(LPSTR)]
    _GetFullPathNameA.restype  = DWORD

    nBufferLength = _GetFullPathNameA(lpFileName, 0, None, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer   = ctypes.create_string_buffer('', nBufferLength + 1)
    lpFilePart = LPSTR()
    nCopied = _GetFullPathNameA(lpFileName, nBufferLength, lpBuffer, byref(lpFilePart))
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value, lpFilePart.value

def GetFullPathNameW(lpFileName):
    _GetFullPathNameW = windll.kernel32.GetFullPathNameW
    _GetFullPathNameW.argtypes = [LPWSTR, DWORD, LPWSTR, POINTER(LPWSTR)]
    _GetFullPathNameW.restype  = DWORD

    nBufferLength = _GetFullPathNameW(lpFileName, 0, None, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer   = ctypes.create_unicode_buffer(u'', nBufferLength + 1)
    lpFilePart = LPWSTR()
    nCopied = _GetFullPathNameW(lpFileName, nBufferLength, lpBuffer, byref(lpFilePart))
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value, lpFilePart.value

GetFullPathName = GuessStringType(GetFullPathNameA, GetFullPathNameW)

# DWORD WINAPI GetTempPath(
#   __in   DWORD nBufferLength,
#   __out  LPTSTR lpBuffer
# );
def GetTempPathA():
    _GetTempPathA = windll.kernel32.GetTempPathA
    _GetTempPathA.argtypes = [DWORD, LPSTR]
    _GetTempPathA.restype  = DWORD

    nBufferLength = _GetTempPathA(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
    nCopied = _GetTempPathA(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value

def GetTempPathW():
    _GetTempPathW = windll.kernel32.GetTempPathW
    _GetTempPathW.argtypes = [DWORD, LPWSTR]
    _GetTempPathW.restype  = DWORD

    nBufferLength = _GetTempPathW(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
    nCopied = _GetTempPathW(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value

GetTempPath = GuessStringType(GetTempPathA, GetTempPathW)

# UINT WINAPI GetTempFileName(
#   __in   LPCTSTR lpPathName,
#   __in   LPCTSTR lpPrefixString,
#   __in   UINT uUnique,
#   __out  LPTSTR lpTempFileName
# );
def GetTempFileNameA(lpPathName = None, lpPrefixString = "TMP", uUnique = 0):
    _GetTempFileNameA = windll.kernel32.GetTempFileNameA
    _GetTempFileNameA.argtypes = [LPSTR, LPSTR, UINT, LPSTR]
    _GetTempFileNameA.restype  = UINT

    if lpPathName is None:
        lpPathName = GetTempPathA()
    lpTempFileName = ctypes.create_string_buffer('', MAX_PATH)
    uUnique = _GetTempFileNameA(lpPathName, lpPrefixString, uUnique, lpTempFileName)
    if uUnique == 0:
        raise ctypes.WinError()
    return lpTempFileName.value, uUnique

def GetTempFileNameW(lpPathName = None, lpPrefixString = u"TMP", uUnique = 0):
    _GetTempFileNameW = windll.kernel32.GetTempFileNameW
    _GetTempFileNameW.argtypes = [LPWSTR, LPWSTR, UINT, LPWSTR]
    _GetTempFileNameW.restype  = UINT

    if lpPathName is None:
        lpPathName = GetTempPathW()
    lpTempFileName = ctypes.create_unicode_buffer(u'', MAX_PATH)
    uUnique = _GetTempFileNameW(lpPathName, lpPrefixString, uUnique, lpTempFileName)
    if uUnique == 0:
        raise ctypes.WinError()
    return lpTempFileName.value, uUnique

GetTempFileName = GuessStringType(GetTempFileNameA, GetTempFileNameW)

# DWORD WINAPI GetCurrentDirectory(
#   __in   DWORD nBufferLength,
#   __out  LPTSTR lpBuffer
# );
def GetCurrentDirectoryA():
    _GetCurrentDirectoryA = windll.kernel32.GetCurrentDirectoryA
    _GetCurrentDirectoryA.argtypes = [DWORD, LPSTR]
    _GetCurrentDirectoryA.restype  = DWORD

    nBufferLength = _GetCurrentDirectoryA(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
    nCopied = _GetCurrentDirectoryA(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value

def GetCurrentDirectoryW():
    _GetCurrentDirectoryW = windll.kernel32.GetCurrentDirectoryW
    _GetCurrentDirectoryW.argtypes = [DWORD, LPWSTR]
    _GetCurrentDirectoryW.restype  = DWORD

    nBufferLength = _GetCurrentDirectoryW(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
    nCopied = _GetCurrentDirectoryW(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value

GetCurrentDirectory = GuessStringType(GetCurrentDirectoryA, GetCurrentDirectoryW)

#------------------------------------------------------------------------------
# Contrl-C handler

# BOOL WINAPI HandlerRoutine(
#   __in  DWORD dwCtrlType
# );
PHANDLER_ROUTINE = ctypes.WINFUNCTYPE(BOOL, DWORD)

# BOOL WINAPI SetConsoleCtrlHandler(
#   __in_opt  PHANDLER_ROUTINE HandlerRoutine,
#   __in      BOOL Add
# );
def SetConsoleCtrlHandler(HandlerRoutine = None, Add = True):
    _SetConsoleCtrlHandler = windll.kernel32.SetConsoleCtrlHandler
    _SetConsoleCtrlHandler.argtypes = [PHANDLER_ROUTINE, BOOL]
    _SetConsoleCtrlHandler.restype  = bool
    _SetConsoleCtrlHandler.errcheck = RaiseIfZero
    _SetConsoleCtrlHandler(HandlerRoutine, bool(Add))
    # we can't automagically transform Python functions to PHANDLER_ROUTINE
    # because a) the actual pointer value is meaningful to the API
    # and b) if it gets garbage collected bad things would happen

# BOOL WINAPI GenerateConsoleCtrlEvent(
#   __in  DWORD dwCtrlEvent,
#   __in  DWORD dwProcessGroupId
# );
def GenerateConsoleCtrlEvent(dwCtrlEvent, dwProcessGroupId):
    _GenerateConsoleCtrlEvent = windll.kernel32.GenerateConsoleCtrlEvent
    _GenerateConsoleCtrlEvent.argtypes = [DWORD, DWORD]
    _GenerateConsoleCtrlEvent.restype  = bool
    _GenerateConsoleCtrlEvent.errcheck = RaiseIfZero
    _GenerateConsoleCtrlEvent(dwCtrlEvent, dwProcessGroupId)

#------------------------------------------------------------------------------
# Synchronization API

# XXX NOTE
#
# Instead of waiting forever, we wait for a small period of time and loop.
# This is a workaround for an unwanted behavior of psyco-accelerated code:
# you can't interrupt a blocking call using Ctrl+C, because signal processing
# is only done between C calls.
#
# Also see: bug #2793618 in Psyco project
# http://sourceforge.net/tracker/?func=detail&aid=2793618&group_id=41036&atid=429622

# DWORD WINAPI WaitForSingleObject(
#   HANDLE hHandle,
#   DWORD dwMilliseconds
# );
def WaitForSingleObject(hHandle, dwMilliseconds = INFINITE):
    _WaitForSingleObject = windll.kernel32.WaitForSingleObject
    _WaitForSingleObject.argtypes = [HANDLE, DWORD]
    _WaitForSingleObject.restype  = DWORD

    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    if dwMilliseconds != INFINITE:
        r = _WaitForSingleObject(hHandle, dwMilliseconds)
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForSingleObject(hHandle, 100)
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r

# DWORD WINAPI WaitForSingleObjectEx(
#   HANDLE hHandle,
#   DWORD dwMilliseconds,
#   BOOL bAlertable
# );
def WaitForSingleObjectEx(hHandle, dwMilliseconds = INFINITE, bAlertable = True):
    _WaitForSingleObjectEx = windll.kernel32.WaitForSingleObjectEx
    _WaitForSingleObjectEx.argtypes = [HANDLE, DWORD, BOOL]
    _WaitForSingleObjectEx.restype  = DWORD

    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    if dwMilliseconds != INFINITE:
        r = _WaitForSingleObjectEx(hHandle, dwMilliseconds, bool(bAlertable))
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForSingleObjectEx(hHandle, 100, bool(bAlertable))
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r

# DWORD WINAPI WaitForMultipleObjects(
#   DWORD nCount,
#   const HANDLE *lpHandles,
#   BOOL bWaitAll,
#   DWORD dwMilliseconds
# );
def WaitForMultipleObjects(handles, bWaitAll = False, dwMilliseconds = INFINITE):
    _WaitForMultipleObjects = windll.kernel32.WaitForMultipleObjects
    _WaitForMultipleObjects.argtypes = [DWORD, POINTER(HANDLE), BOOL, DWORD]
    _WaitForMultipleObjects.restype  = DWORD

    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    nCount          = len(handles)
    lpHandlesType   = HANDLE * nCount
    lpHandles       = lpHandlesType(*handles)
    if dwMilliseconds != INFINITE:
        r = _WaitForMultipleObjects(byref(lpHandles), bool(bWaitAll), dwMilliseconds)
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForMultipleObjects(byref(lpHandles), bool(bWaitAll), 100)
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r

# DWORD WINAPI WaitForMultipleObjectsEx(
#   DWORD nCount,
#   const HANDLE *lpHandles,
#   BOOL bWaitAll,
#   DWORD dwMilliseconds,
#   BOOL bAlertable
# );
def WaitForMultipleObjectsEx(handles, bWaitAll = False, dwMilliseconds = INFINITE, bAlertable = True):
    _WaitForMultipleObjectsEx = windll.kernel32.WaitForMultipleObjectsEx
    _WaitForMultipleObjectsEx.argtypes = [DWORD, POINTER(HANDLE), BOOL, DWORD]
    _WaitForMultipleObjectsEx.restype  = DWORD

    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    nCount          = len(handles)
    lpHandlesType   = HANDLE * nCount
    lpHandles       = lpHandlesType(*handles)
    if dwMilliseconds != INFINITE:
        r = _WaitForMultipleObjectsEx(byref(lpHandles), bool(bWaitAll), dwMilliseconds, bool(bAlertable))
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForMultipleObjectsEx(byref(lpHandles), bool(bWaitAll), 100, bool(bAlertable))
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r

# HANDLE WINAPI CreateMutex(
#   _In_opt_  LPSECURITY_ATTRIBUTES lpMutexAttributes,
#   _In_      BOOL bInitialOwner,
#   _In_opt_  LPCTSTR lpName
# );
def CreateMutexA(lpMutexAttributes = None, bInitialOwner = True, lpName = None):
    _CreateMutexA = windll.kernel32.CreateMutexA
    _CreateMutexA.argtypes = [LPVOID, BOOL, LPSTR]
    _CreateMutexA.restype  = HANDLE
    _CreateMutexA.errcheck = RaiseIfZero
    return Handle( _CreateMutexA(lpMutexAttributes, bInitialOwner, lpName) )

def CreateMutexW(lpMutexAttributes = None, bInitialOwner = True, lpName = None):
    _CreateMutexW = windll.kernel32.CreateMutexW
    _CreateMutexW.argtypes = [LPVOID, BOOL, LPWSTR]
    _CreateMutexW.restype  = HANDLE
    _CreateMutexW.errcheck = RaiseIfZero
    return Handle( _CreateMutexW(lpMutexAttributes, bInitialOwner, lpName) )

CreateMutex = GuessStringType(CreateMutexA, CreateMutexW)

# HANDLE WINAPI OpenMutex(
#   _In_  DWORD dwDesiredAccess,
#   _In_  BOOL bInheritHandle,
#   _In_  LPCTSTR lpName
# );
def OpenMutexA(dwDesiredAccess = MUTEX_ALL_ACCESS, bInitialOwner = True, lpName = None):
    _OpenMutexA = windll.kernel32.OpenMutexA
    _OpenMutexA.argtypes = [DWORD, BOOL, LPSTR]
    _OpenMutexA.restype  = HANDLE
    _OpenMutexA.errcheck = RaiseIfZero
    return Handle( _OpenMutexA(lpMutexAttributes, bInitialOwner, lpName) )

def OpenMutexW(dwDesiredAccess = MUTEX_ALL_ACCESS, bInitialOwner = True, lpName = None):
    _OpenMutexW = windll.kernel32.OpenMutexW
    _OpenMutexW.argtypes = [DWORD, BOOL, LPWSTR]
    _OpenMutexW.restype  = HANDLE
    _OpenMutexW.errcheck = RaiseIfZero
    return Handle( _OpenMutexW(lpMutexAttributes, bInitialOwner, lpName) )

OpenMutex = GuessStringType(OpenMutexA, OpenMutexW)

# HANDLE WINAPI CreateEvent(
#   _In_opt_  LPSECURITY_ATTRIBUTES lpEventAttributes,
#   _In_      BOOL bManualReset,
#   _In_      BOOL bInitialState,
#   _In_opt_  LPCTSTR lpName
# );
def CreateEventA(lpMutexAttributes = None, bManualReset = False, bInitialState = False, lpName = None):
    _CreateEventA = windll.kernel32.CreateEventA
    _CreateEventA.argtypes = [LPVOID, BOOL, BOOL, LPSTR]
    _CreateEventA.restype  = HANDLE
    _CreateEventA.errcheck = RaiseIfZero
    return Handle( _CreateEventA(lpMutexAttributes, bManualReset, bInitialState, lpName) )

def CreateEventW(lpMutexAttributes = None, bManualReset = False, bInitialState = False, lpName = None):
    _CreateEventW = windll.kernel32.CreateEventW
    _CreateEventW.argtypes = [LPVOID, BOOL, BOOL, LPWSTR]
    _CreateEventW.restype  = HANDLE
    _CreateEventW.errcheck = RaiseIfZero
    return Handle( _CreateEventW(lpMutexAttributes, bManualReset, bInitialState, lpName) )

CreateEvent = GuessStringType(CreateEventA, CreateEventW)

# HANDLE WINAPI OpenEvent(
#   _In_  DWORD dwDesiredAccess,
#   _In_  BOOL bInheritHandle,
#   _In_  LPCTSTR lpName
# );
def OpenEventA(dwDesiredAccess = EVENT_ALL_ACCESS, bInheritHandle = False, lpName = None):
    _OpenEventA = windll.kernel32.OpenEventA
    _OpenEventA.argtypes = [DWORD, BOOL, LPSTR]
    _OpenEventA.restype  = HANDLE
    _OpenEventA.errcheck = RaiseIfZero
    return Handle( _OpenEventA(dwDesiredAccess, bInheritHandle, lpName) )

def OpenEventW(dwDesiredAccess = EVENT_ALL_ACCESS, bInheritHandle = False, lpName = None):
    _OpenEventW = windll.kernel32.OpenEventW
    _OpenEventW.argtypes = [DWORD, BOOL, LPWSTR]
    _OpenEventW.restype  = HANDLE
    _OpenEventW.errcheck = RaiseIfZero
    return Handle( _OpenEventW(dwDesiredAccess, bInheritHandle, lpName) )

OpenEvent = GuessStringType(OpenEventA, OpenEventW)

# HANDLE WINAPI CreateSemaphore(
#   _In_opt_  LPSECURITY_ATTRIBUTES lpSemaphoreAttributes,
#   _In_      LONG lInitialCount,
#   _In_      LONG lMaximumCount,
#   _In_opt_  LPCTSTR lpName
# );

# TODO

# HANDLE WINAPI OpenSemaphore(
#   _In_  DWORD dwDesiredAccess,
#   _In_  BOOL bInheritHandle,
#   _In_  LPCTSTR lpName
# );

# TODO

# BOOL WINAPI ReleaseMutex(
#   _In_  HANDLE hMutex
# );
def ReleaseMutex(hMutex):
    _ReleaseMutex = windll.kernel32.ReleaseMutex
    _ReleaseMutex.argtypes = [HANDLE]
    _ReleaseMutex.restype  = bool
    _ReleaseMutex.errcheck = RaiseIfZero
    _ReleaseMutex(hMutex)

# BOOL WINAPI SetEvent(
#   _In_  HANDLE hEvent
# );
def SetEvent(hEvent):
    _SetEvent = windll.kernel32.SetEvent
    _SetEvent.argtypes = [HANDLE]
    _SetEvent.restype  = bool
    _SetEvent.errcheck = RaiseIfZero
    _SetEvent(hEvent)

# BOOL WINAPI ResetEvent(
#   _In_  HANDLE hEvent
# );
def ResetEvent(hEvent):
    _ResetEvent = windll.kernel32.ResetEvent
    _ResetEvent.argtypes = [HANDLE]
    _ResetEvent.restype  = bool
    _ResetEvent.errcheck = RaiseIfZero
    _ResetEvent(hEvent)

# BOOL WINAPI PulseEvent(
#   _In_  HANDLE hEvent
# );
def PulseEvent(hEvent):
    _PulseEvent = windll.kernel32.PulseEvent
    _PulseEvent.argtypes = [HANDLE]
    _PulseEvent.restype  = bool
    _PulseEvent.errcheck = RaiseIfZero
    _PulseEvent(hEvent)

# BOOL WINAPI ReleaseSemaphore(
#   _In_       HANDLE hSemaphore,
#   _In_       LONG lReleaseCount,
#   _Out_opt_  LPLONG lpPreviousCount
# );

# TODO

#------------------------------------------------------------------------------
# Debug API

# BOOL WaitForDebugEvent(
#   LPDEBUG_EVENT lpDebugEvent,
#   DWORD dwMilliseconds
# );
def WaitForDebugEvent(dwMilliseconds = INFINITE):
    _WaitForDebugEvent = windll.kernel32.WaitForDebugEvent
    _WaitForDebugEvent.argtypes = [LPDEBUG_EVENT, DWORD]
    _WaitForDebugEvent.restype  = DWORD

    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    lpDebugEvent                  = DEBUG_EVENT()
    lpDebugEvent.dwDebugEventCode = 0
    lpDebugEvent.dwProcessId      = 0
    lpDebugEvent.dwThreadId       = 0
    if dwMilliseconds != INFINITE:
        success = _WaitForDebugEvent(byref(lpDebugEvent), dwMilliseconds)
        if success == 0:
            raise ctypes.WinError()
    else:
        # this avoids locking the Python GIL for too long
        while 1:
            success = _WaitForDebugEvent(byref(lpDebugEvent), 100)
            if success != 0:
                break
            code = GetLastError()
            if code not in (ERROR_SEM_TIMEOUT, WAIT_TIMEOUT):
                raise ctypes.WinError(code)
    return lpDebugEvent

# BOOL ContinueDebugEvent(
#   DWORD dwProcessId,
#   DWORD dwThreadId,
#   DWORD dwContinueStatus
# );
def ContinueDebugEvent(dwProcessId, dwThreadId, dwContinueStatus = DBG_EXCEPTION_NOT_HANDLED):
    _ContinueDebugEvent = windll.kernel32.ContinueDebugEvent
    _ContinueDebugEvent.argtypes = [DWORD, DWORD, DWORD]
    _ContinueDebugEvent.restype  = bool
    _ContinueDebugEvent.errcheck = RaiseIfZero
    _ContinueDebugEvent(dwProcessId, dwThreadId, dwContinueStatus)

# BOOL WINAPI FlushInstructionCache(
#   __in  HANDLE hProcess,
#   __in  LPCVOID lpBaseAddress,
#   __in  SIZE_T dwSize
# );
def FlushInstructionCache(hProcess, lpBaseAddress = None, dwSize = 0):
    # http://blogs.msdn.com/oldnewthing/archive/2003/12/08/55954.aspx#55958
    _FlushInstructionCache = windll.kernel32.FlushInstructionCache
    _FlushInstructionCache.argtypes = [HANDLE, LPVOID, SIZE_T]
    _FlushInstructionCache.restype  = bool
    _FlushInstructionCache.errcheck = RaiseIfZero
    _FlushInstructionCache(hProcess, lpBaseAddress, dwSize)

# BOOL DebugActiveProcess(
#   DWORD dwProcessId
# );
def DebugActiveProcess(dwProcessId):
    _DebugActiveProcess = windll.kernel32.DebugActiveProcess
    _DebugActiveProcess.argtypes = [DWORD]
    _DebugActiveProcess.restype  = bool
    _DebugActiveProcess.errcheck = RaiseIfZero
    _DebugActiveProcess(dwProcessId)

# BOOL DebugActiveProcessStop(
#   DWORD dwProcessId
# );
def DebugActiveProcessStop(dwProcessId):
    _DebugActiveProcessStop = windll.kernel32.DebugActiveProcessStop
    _DebugActiveProcessStop.argtypes = [DWORD]
    _DebugActiveProcessStop.restype  = bool
    _DebugActiveProcessStop.errcheck = RaiseIfZero
    _DebugActiveProcessStop(dwProcessId)

# BOOL CheckRemoteDebuggerPresent(
#   HANDLE hProcess,
#   PBOOL pbDebuggerPresent
# );
def CheckRemoteDebuggerPresent(hProcess):
    _CheckRemoteDebuggerPresent = windll.kernel32.CheckRemoteDebuggerPresent
    _CheckRemoteDebuggerPresent.argtypes = [HANDLE, PBOOL]
    _CheckRemoteDebuggerPresent.restype  = bool
    _CheckRemoteDebuggerPresent.errcheck = RaiseIfZero

    pbDebuggerPresent = BOOL(0)
    _CheckRemoteDebuggerPresent(hProcess, byref(pbDebuggerPresent))
    return bool(pbDebuggerPresent.value)

# BOOL DebugSetProcessKillOnExit(
#   BOOL KillOnExit
# );
def DebugSetProcessKillOnExit(KillOnExit):
    _DebugSetProcessKillOnExit = windll.kernel32.DebugSetProcessKillOnExit
    _DebugSetProcessKillOnExit.argtypes = [BOOL]
    _DebugSetProcessKillOnExit.restype  = bool
    _DebugSetProcessKillOnExit.errcheck = RaiseIfZero
    _DebugSetProcessKillOnExit(bool(KillOnExit))

# BOOL DebugBreakProcess(
#   HANDLE Process
# );
def DebugBreakProcess(hProcess):
    _DebugBreakProcess = windll.kernel32.DebugBreakProcess
    _DebugBreakProcess.argtypes = [HANDLE]
    _DebugBreakProcess.restype  = bool
    _DebugBreakProcess.errcheck = RaiseIfZero
    _DebugBreakProcess(hProcess)

# void WINAPI OutputDebugString(
#   __in_opt  LPCTSTR lpOutputString
# );
def OutputDebugStringA(lpOutputString):
    _OutputDebugStringA = windll.kernel32.OutputDebugStringA
    _OutputDebugStringA.argtypes = [LPSTR]
    _OutputDebugStringA.restype  = None
    _OutputDebugStringA(lpOutputString)

def OutputDebugStringW(lpOutputString):
    _OutputDebugStringW = windll.kernel32.OutputDebugStringW
    _OutputDebugStringW.argtypes = [LPWSTR]
    _OutputDebugStringW.restype  = None
    _OutputDebugStringW(lpOutputString)

OutputDebugString = GuessStringType(OutputDebugStringA, OutputDebugStringW)

# BOOL WINAPI ReadProcessMemory(
#   __in   HANDLE hProcess,
#   __in   LPCVOID lpBaseAddress,
#   __out  LPVOID lpBuffer,
#   __in   SIZE_T nSize,
#   __out  SIZE_T* lpNumberOfBytesRead
# );
def ReadProcessMemory(hProcess, lpBaseAddress, nSize):
    _ReadProcessMemory = windll.kernel32.ReadProcessMemory
    _ReadProcessMemory.argtypes = [HANDLE, LPVOID, LPVOID, SIZE_T, POINTER(SIZE_T)]
    _ReadProcessMemory.restype  = bool

    lpBuffer            = ctypes.create_string_buffer(compat.b(''), nSize)
    lpNumberOfBytesRead = SIZE_T(0)
    success = _ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, byref(lpNumberOfBytesRead))
    if not success and GetLastError() != ERROR_PARTIAL_COPY:
        raise ctypes.WinError()
    return compat.b(lpBuffer.raw)[:lpNumberOfBytesRead.value]

# BOOL WINAPI WriteProcessMemory(
#   __in   HANDLE hProcess,
#   __in   LPCVOID lpBaseAddress,
#   __in   LPVOID lpBuffer,
#   __in   SIZE_T nSize,
#   __out  SIZE_T* lpNumberOfBytesWritten
# );
def WriteProcessMemory(hProcess, lpBaseAddress, lpBuffer):
    _WriteProcessMemory = windll.kernel32.WriteProcessMemory
    _WriteProcessMemory.argtypes = [HANDLE, LPVOID, LPVOID, SIZE_T, POINTER(SIZE_T)]
    _WriteProcessMemory.restype  = bool

    nSize                   = len(lpBuffer)
    lpBuffer                = ctypes.create_string_buffer(lpBuffer)
    lpNumberOfBytesWritten  = SIZE_T(0)
    success = _WriteProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, byref(lpNumberOfBytesWritten))
    if not success and GetLastError() != ERROR_PARTIAL_COPY:
        raise ctypes.WinError()
    return lpNumberOfBytesWritten.value

# LPVOID WINAPI VirtualAllocEx(
#   __in      HANDLE hProcess,
#   __in_opt  LPVOID lpAddress,
#   __in      SIZE_T dwSize,
#   __in      DWORD flAllocationType,
#   __in      DWORD flProtect
# );
def VirtualAllocEx(hProcess, lpAddress = 0, dwSize = 0x1000, flAllocationType = MEM_COMMIT | MEM_RESERVE, flProtect = PAGE_EXECUTE_READWRITE):
    _VirtualAllocEx = windll.kernel32.VirtualAllocEx
    _VirtualAllocEx.argtypes = [HANDLE, LPVOID, SIZE_T, DWORD, DWORD]
    _VirtualAllocEx.restype  = LPVOID

    lpAddress = _VirtualAllocEx(hProcess, lpAddress, dwSize, flAllocationType, flProtect)
    if lpAddress == NULL:
        raise ctypes.WinError()
    return lpAddress

# SIZE_T WINAPI VirtualQueryEx(
#   __in      HANDLE hProcess,
#   __in_opt  LPCVOID lpAddress,
#   __out     PMEMORY_BASIC_INFORMATION lpBuffer,
#   __in      SIZE_T dwLength
# );
def VirtualQueryEx(hProcess, lpAddress):
    _VirtualQueryEx = windll.kernel32.VirtualQueryEx
    _VirtualQueryEx.argtypes = [HANDLE, LPVOID, PMEMORY_BASIC_INFORMATION, SIZE_T]
    _VirtualQueryEx.restype  = SIZE_T

    lpBuffer  = MEMORY_BASIC_INFORMATION()
    dwLength  = sizeof(MEMORY_BASIC_INFORMATION)
    success   = _VirtualQueryEx(hProcess, lpAddress, byref(lpBuffer), dwLength)
    if success == 0:
        raise ctypes.WinError()
    return MemoryBasicInformation(lpBuffer)

# BOOL WINAPI VirtualProtectEx(
#   __in   HANDLE hProcess,
#   __in   LPVOID lpAddress,
#   __in   SIZE_T dwSize,
#   __in   DWORD flNewProtect,
#   __out  PDWORD lpflOldProtect
# );
def VirtualProtectEx(hProcess, lpAddress, dwSize, flNewProtect = PAGE_EXECUTE_READWRITE):
    _VirtualProtectEx = windll.kernel32.VirtualProtectEx
    _VirtualProtectEx.argtypes = [HANDLE, LPVOID, SIZE_T, DWORD, PDWORD]
    _VirtualProtectEx.restype  = bool
    _VirtualProtectEx.errcheck = RaiseIfZero

    flOldProtect = DWORD(0)
    _VirtualProtectEx(hProcess, lpAddress, dwSize, flNewProtect, byref(flOldProtect))
    return flOldProtect.value

# BOOL WINAPI VirtualFreeEx(
#   __in  HANDLE hProcess,
#   __in  LPVOID lpAddress,
#   __in  SIZE_T dwSize,
#   __in  DWORD dwFreeType
# );
def VirtualFreeEx(hProcess, lpAddress, dwSize = 0, dwFreeType = MEM_RELEASE):
    _VirtualFreeEx = windll.kernel32.VirtualFreeEx
    _VirtualFreeEx.argtypes = [HANDLE, LPVOID, SIZE_T, DWORD]
    _VirtualFreeEx.restype  = bool
    _VirtualFreeEx.errcheck = RaiseIfZero
    _VirtualFreeEx(hProcess, lpAddress, dwSize, dwFreeType)

# HANDLE WINAPI CreateRemoteThread(
#   __in   HANDLE hProcess,
#   __in   LPSECURITY_ATTRIBUTES lpThreadAttributes,
#   __in   SIZE_T dwStackSize,
#   __in   LPTHREAD_START_ROUTINE lpStartAddress,
#   __in   LPVOID lpParameter,
#   __in   DWORD dwCreationFlags,
#   __out  LPDWORD lpThreadId
# );
def CreateRemoteThread(hProcess, lpThreadAttributes, dwStackSize, lpStartAddress, lpParameter, dwCreationFlags):
    _CreateRemoteThread = windll.kernel32.CreateRemoteThread
    _CreateRemoteThread.argtypes = [HANDLE, LPSECURITY_ATTRIBUTES, SIZE_T, LPVOID, LPVOID, DWORD, LPDWORD]
    _CreateRemoteThread.restype  = HANDLE

    if not lpThreadAttributes:
        lpThreadAttributes = None
    else:
        lpThreadAttributes = byref(lpThreadAttributes)
    dwThreadId = DWORD(0)
    hThread = _CreateRemoteThread(hProcess, lpThreadAttributes, dwStackSize, lpStartAddress, lpParameter, dwCreationFlags, byref(dwThreadId))
    if not hThread:
        raise ctypes.WinError()
    return ThreadHandle(hThread), dwThreadId.value

#------------------------------------------------------------------------------
# Process API

# BOOL WINAPI CreateProcess(
#   __in_opt     LPCTSTR lpApplicationName,
#   __inout_opt  LPTSTR lpCommandLine,
#   __in_opt     LPSECURITY_ATTRIBUTES lpProcessAttributes,
#   __in_opt     LPSECURITY_ATTRIBUTES lpThreadAttributes,
#   __in         BOOL bInheritHandles,
#   __in         DWORD dwCreationFlags,
#   __in_opt     LPVOID lpEnvironment,
#   __in_opt     LPCTSTR lpCurrentDirectory,
#   __in         LPSTARTUPINFO lpStartupInfo,
#   __out        LPPROCESS_INFORMATION lpProcessInformation
# );
def CreateProcessA(lpApplicationName, lpCommandLine=None, lpProcessAttributes=None, lpThreadAttributes=None, bInheritHandles=False, dwCreationFlags=0, lpEnvironment=None, lpCurrentDirectory=None, lpStartupInfo=None):
    _CreateProcessA = windll.kernel32.CreateProcessA
    _CreateProcessA.argtypes = [LPSTR, LPSTR, LPSECURITY_ATTRIBUTES, LPSECURITY_ATTRIBUTES, BOOL, DWORD, LPVOID, LPSTR, LPVOID, LPPROCESS_INFORMATION]
    _CreateProcessA.restype  = bool
    _CreateProcessA.errcheck = RaiseIfZero

    if not lpApplicationName:
        lpApplicationName   = None
    if not lpCommandLine:
        lpCommandLine       = None
    else:
        lpCommandLine       = ctypes.create_string_buffer(lpCommandLine, max(MAX_PATH, len(lpCommandLine)))
    if not lpEnvironment:
        lpEnvironment       = None
    else:
        lpEnvironment       = ctypes.create_string_buffer(lpEnvironment)
    if not lpCurrentDirectory:
        lpCurrentDirectory  = None
    if not lpProcessAttributes:
        lpProcessAttributes = None
    else:
        lpProcessAttributes = byref(lpProcessAttributes)
    if not lpThreadAttributes:
        lpThreadAttributes = None
    else:
        lpThreadAttributes = byref(lpThreadAttributes)
    if not lpStartupInfo:
        lpStartupInfo              = STARTUPINFO()
        lpStartupInfo.cb           = sizeof(STARTUPINFO)
        lpStartupInfo.lpReserved   = 0
        lpStartupInfo.lpDesktop    = 0
        lpStartupInfo.lpTitle      = 0
        lpStartupInfo.dwFlags      = 0
        lpStartupInfo.cbReserved2  = 0
        lpStartupInfo.lpReserved2  = 0
    lpProcessInformation              = PROCESS_INFORMATION()
    lpProcessInformation.hProcess     = INVALID_HANDLE_VALUE
    lpProcessInformation.hThread      = INVALID_HANDLE_VALUE
    lpProcessInformation.dwProcessId  = 0
    lpProcessInformation.dwThreadId   = 0
    _CreateProcessA(lpApplicationName, lpCommandLine, lpProcessAttributes, lpThreadAttributes, bool(bInheritHandles), dwCreationFlags, lpEnvironment, lpCurrentDirectory, byref(lpStartupInfo), byref(lpProcessInformation))
    return ProcessInformation(lpProcessInformation)

def CreateProcessW(lpApplicationName, lpCommandLine=None, lpProcessAttributes=None, lpThreadAttributes=None, bInheritHandles=False, dwCreationFlags=0, lpEnvironment=None, lpCurrentDirectory=None, lpStartupInfo=None):
    _CreateProcessW = windll.kernel32.CreateProcessW
    _CreateProcessW.argtypes = [LPWSTR, LPWSTR, LPSECURITY_ATTRIBUTES, LPSECURITY_ATTRIBUTES, BOOL, DWORD, LPVOID, LPWSTR, LPVOID, LPPROCESS_INFORMATION]
    _CreateProcessW.restype  = bool
    _CreateProcessW.errcheck = RaiseIfZero

    if not lpApplicationName:
        lpApplicationName   = None
    if not lpCommandLine:
        lpCommandLine       = None
    else:
        lpCommandLine       = ctypes.create_unicode_buffer(lpCommandLine, max(MAX_PATH, len(lpCommandLine)))
    if not lpEnvironment:
        lpEnvironment       = None
    else:
        lpEnvironment       = ctypes.create_unicode_buffer(lpEnvironment)
    if not lpCurrentDirectory:
        lpCurrentDirectory  = None
    if not lpProcessAttributes:
        lpProcessAttributes = None
    else:
        lpProcessAttributes = byref(lpProcessAttributes)
    if not lpThreadAttributes:
        lpThreadAttributes = None
    else:
        lpThreadAttributes = byref(lpThreadAttributes)
    if not lpStartupInfo:
        lpStartupInfo              = STARTUPINFO()
        lpStartupInfo.cb           = sizeof(STARTUPINFO)
        lpStartupInfo.lpReserved   = 0
        lpStartupInfo.lpDesktop    = 0
        lpStartupInfo.lpTitle      = 0
        lpStartupInfo.dwFlags      = 0
        lpStartupInfo.cbReserved2  = 0
        lpStartupInfo.lpReserved2  = 0
    lpProcessInformation              = PROCESS_INFORMATION()
    lpProcessInformation.hProcess     = INVALID_HANDLE_VALUE
    lpProcessInformation.hThread      = INVALID_HANDLE_VALUE
    lpProcessInformation.dwProcessId  = 0
    lpProcessInformation.dwThreadId   = 0
    _CreateProcessW(lpApplicationName, lpCommandLine, lpProcessAttributes, lpThreadAttributes, bool(bInheritHandles), dwCreationFlags, lpEnvironment, lpCurrentDirectory, byref(lpStartupInfo), byref(lpProcessInformation))
    return ProcessInformation(lpProcessInformation)

CreateProcess = GuessStringType(CreateProcessA, CreateProcessW)

# BOOL WINAPI InitializeProcThreadAttributeList(
#   __out_opt   LPPROC_THREAD_ATTRIBUTE_LIST lpAttributeList,
#   __in        DWORD dwAttributeCount,
#   __reserved  DWORD dwFlags,
#   __inout     PSIZE_T lpSize
# );
def InitializeProcThreadAttributeList(dwAttributeCount):
    _InitializeProcThreadAttributeList = windll.kernel32.InitializeProcThreadAttributeList
    _InitializeProcThreadAttributeList.argtypes = [LPPROC_THREAD_ATTRIBUTE_LIST, DWORD, DWORD, PSIZE_T]
    _InitializeProcThreadAttributeList.restype  = bool

    Size = SIZE_T(0)
    _InitializeProcThreadAttributeList(None, dwAttributeCount, 0, byref(Size))
    RaiseIfZero(Size.value)
    AttributeList = (BYTE * Size.value)()
    success = _InitializeProcThreadAttributeList(byref(AttributeList), dwAttributeCount, 0, byref(Size))
    RaiseIfZero(success)
    return AttributeList

# BOOL WINAPI UpdateProcThreadAttribute(
#   __inout    LPPROC_THREAD_ATTRIBUTE_LIST lpAttributeList,
#   __in       DWORD dwFlags,
#   __in       DWORD_PTR Attribute,
#   __in       PVOID lpValue,
#   __in       SIZE_T cbSize,
#   __out_opt  PVOID lpPreviousValue,
#   __in_opt   PSIZE_T lpReturnSize
# );
def UpdateProcThreadAttribute(lpAttributeList, Attribute, Value, cbSize = None):
    _UpdateProcThreadAttribute = windll.kernel32.UpdateProcThreadAttribute
    _UpdateProcThreadAttribute.argtypes = [LPPROC_THREAD_ATTRIBUTE_LIST, DWORD, DWORD_PTR, PVOID, SIZE_T, PVOID, PSIZE_T]
    _UpdateProcThreadAttribute.restype  = bool
    _UpdateProcThreadAttribute.errcheck = RaiseIfZero

    if cbSize is None:
        cbSize = sizeof(Value)
    _UpdateProcThreadAttribute(byref(lpAttributeList), 0, Attribute, byref(Value), cbSize, None, None)

# VOID WINAPI DeleteProcThreadAttributeList(
#   __inout  LPPROC_THREAD_ATTRIBUTE_LIST lpAttributeList
# );
def DeleteProcThreadAttributeList(lpAttributeList):
    _DeleteProcThreadAttributeList = windll.kernel32.DeleteProcThreadAttributeList
    _DeleteProcThreadAttributeList.restype = None
    _DeleteProcThreadAttributeList(byref(lpAttributeList))

# HANDLE WINAPI OpenProcess(
#   __in  DWORD dwDesiredAccess,
#   __in  BOOL bInheritHandle,
#   __in  DWORD dwProcessId
# );
def OpenProcess(dwDesiredAccess, bInheritHandle, dwProcessId):
    _OpenProcess = windll.kernel32.OpenProcess
    _OpenProcess.argtypes = [DWORD, BOOL, DWORD]
    _OpenProcess.restype  = HANDLE

    hProcess = _OpenProcess(dwDesiredAccess, bool(bInheritHandle), dwProcessId)
    if hProcess == NULL:
        raise ctypes.WinError()
    return ProcessHandle(hProcess, dwAccess = dwDesiredAccess)

# HANDLE WINAPI OpenThread(
#   __in  DWORD dwDesiredAccess,
#   __in  BOOL bInheritHandle,
#   __in  DWORD dwThreadId
# );
def OpenThread(dwDesiredAccess, bInheritHandle, dwThreadId):
    _OpenThread = windll.kernel32.OpenThread
    _OpenThread.argtypes = [DWORD, BOOL, DWORD]
    _OpenThread.restype  = HANDLE

    hThread = _OpenThread(dwDesiredAccess, bool(bInheritHandle), dwThreadId)
    if hThread == NULL:
        raise ctypes.WinError()
    return ThreadHandle(hThread, dwAccess = dwDesiredAccess)

# DWORD WINAPI SuspendThread(
#   __in  HANDLE hThread
# );
def SuspendThread(hThread):
    _SuspendThread = windll.kernel32.SuspendThread
    _SuspendThread.argtypes = [HANDLE]
    _SuspendThread.restype  = DWORD

    previousCount = _SuspendThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount

# DWORD WINAPI ResumeThread(
#   __in  HANDLE hThread
# );
def ResumeThread(hThread):
    _ResumeThread = windll.kernel32.ResumeThread
    _ResumeThread.argtypes = [HANDLE]
    _ResumeThread.restype  = DWORD

    previousCount = _ResumeThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount

# BOOL WINAPI TerminateThread(
#   __inout  HANDLE hThread,
#   __in     DWORD dwExitCode
# );
def TerminateThread(hThread, dwExitCode = 0):
    _TerminateThread = windll.kernel32.TerminateThread
    _TerminateThread.argtypes = [HANDLE, DWORD]
    _TerminateThread.restype  = bool
    _TerminateThread.errcheck = RaiseIfZero
    _TerminateThread(hThread, dwExitCode)

# BOOL WINAPI TerminateProcess(
#   __inout  HANDLE hProcess,
#   __in     DWORD dwExitCode
# );
def TerminateProcess(hProcess, dwExitCode = 0):
    _TerminateProcess = windll.kernel32.TerminateProcess
    _TerminateProcess.argtypes = [HANDLE, DWORD]
    _TerminateProcess.restype  = bool
    _TerminateProcess.errcheck = RaiseIfZero
    _TerminateProcess(hProcess, dwExitCode)

# DWORD WINAPI GetCurrentProcessId(void);
def GetCurrentProcessId():
    _GetCurrentProcessId = windll.kernel32.GetCurrentProcessId
    _GetCurrentProcessId.argtypes = []
    _GetCurrentProcessId.restype  = DWORD
    return _GetCurrentProcessId()

# DWORD WINAPI GetCurrentThreadId(void);
def GetCurrentThreadId():
    _GetCurrentThreadId = windll.kernel32.GetCurrentThreadId
    _GetCurrentThreadId.argtypes = []
    _GetCurrentThreadId.restype  = DWORD
    return _GetCurrentThreadId()

# DWORD WINAPI GetProcessId(
#   __in  HANDLE hProcess
# );
def GetProcessId(hProcess):
    _GetProcessId = windll.kernel32.GetProcessId
    _GetProcessId.argtypes = [HANDLE]
    _GetProcessId.restype  = DWORD
    _GetProcessId.errcheck = RaiseIfZero
    return _GetProcessId(hProcess)

# DWORD WINAPI GetThreadId(
#   __in  HANDLE hThread
# );
def GetThreadId(hThread):
    _GetThreadId = windll.kernel32._GetThreadId
    _GetThreadId.argtypes = [HANDLE]
    _GetThreadId.restype  = DWORD

    dwThreadId = _GetThreadId(hThread)
    if dwThreadId == 0:
        raise ctypes.WinError()
    return dwThreadId

# DWORD WINAPI GetProcessIdOfThread(
#   __in  HANDLE hThread
# );
def GetProcessIdOfThread(hThread):
    _GetProcessIdOfThread = windll.kernel32.GetProcessIdOfThread
    _GetProcessIdOfThread.argtypes = [HANDLE]
    _GetProcessIdOfThread.restype  = DWORD

    dwProcessId = _GetProcessIdOfThread(hThread)
    if dwProcessId == 0:
        raise ctypes.WinError()
    return dwProcessId

# BOOL WINAPI GetExitCodeProcess(
#   __in   HANDLE hProcess,
#   __out  LPDWORD lpExitCode
# );
def GetExitCodeProcess(hProcess):
    _GetExitCodeProcess = windll.kernel32.GetExitCodeProcess
    _GetExitCodeProcess.argtypes = [HANDLE]
    _GetExitCodeProcess.restype  = bool
    _GetExitCodeProcess.errcheck = RaiseIfZero

    lpExitCode = DWORD(0)
    _GetExitCodeProcess(hProcess, byref(lpExitCode))
    return lpExitCode.value

# BOOL WINAPI GetExitCodeThread(
#   __in   HANDLE hThread,
#   __out  LPDWORD lpExitCode
# );
def GetExitCodeThread(hThread):
    _GetExitCodeThread = windll.kernel32.GetExitCodeThread
    _GetExitCodeThread.argtypes = [HANDLE]
    _GetExitCodeThread.restype  = bool
    _GetExitCodeThread.errcheck = RaiseIfZero

    lpExitCode = DWORD(0)
    _GetExitCodeThread(hThread, byref(lpExitCode))
    return lpExitCode.value

# DWORD WINAPI GetProcessVersion(
#   __in  DWORD ProcessId
# );
def GetProcessVersion(ProcessId):
    _GetProcessVersion = windll.kernel32.GetProcessVersion
    _GetProcessVersion.argtypes = [DWORD]
    _GetProcessVersion.restype  = DWORD

    retval = _GetProcessVersion(ProcessId)
    if retval == 0:
        raise ctypes.WinError()
    return retval

# DWORD WINAPI GetPriorityClass(
#   __in  HANDLE hProcess
# );
def GetPriorityClass(hProcess):
    _GetPriorityClass = windll.kernel32.GetPriorityClass
    _GetPriorityClass.argtypes = [HANDLE]
    _GetPriorityClass.restype  = DWORD

    retval = _GetPriorityClass(hProcess)
    if retval == 0:
        raise ctypes.WinError()
    return retval

# BOOL WINAPI SetPriorityClass(
#   __in  HANDLE hProcess,
#   __in  DWORD dwPriorityClass
# );
def SetPriorityClass(hProcess, dwPriorityClass = NORMAL_PRIORITY_CLASS):
    _SetPriorityClass = windll.kernel32.SetPriorityClass
    _SetPriorityClass.argtypes = [HANDLE, DWORD]
    _SetPriorityClass.restype  = bool
    _SetPriorityClass.errcheck = RaiseIfZero
    _SetPriorityClass(hProcess, dwPriorityClass)

# BOOL WINAPI GetProcessPriorityBoost(
#   __in   HANDLE hProcess,
#   __out  PBOOL pDisablePriorityBoost
# );
def GetProcessPriorityBoost(hProcess):
    _GetProcessPriorityBoost = windll.kernel32.GetProcessPriorityBoost
    _GetProcessPriorityBoost.argtypes = [HANDLE, PBOOL]
    _GetProcessPriorityBoost.restype  = bool
    _GetProcessPriorityBoost.errcheck = RaiseIfZero

    pDisablePriorityBoost = BOOL(False)
    _GetProcessPriorityBoost(hProcess, byref(pDisablePriorityBoost))
    return bool(pDisablePriorityBoost.value)

# BOOL WINAPI SetProcessPriorityBoost(
#   __in  HANDLE hProcess,
#   __in  BOOL DisablePriorityBoost
# );
def SetProcessPriorityBoost(hProcess, DisablePriorityBoost):
    _SetProcessPriorityBoost = windll.kernel32.SetProcessPriorityBoost
    _SetProcessPriorityBoost.argtypes = [HANDLE, BOOL]
    _SetProcessPriorityBoost.restype  = bool
    _SetProcessPriorityBoost.errcheck = RaiseIfZero
    _SetProcessPriorityBoost(hProcess, bool(DisablePriorityBoost))

# BOOL WINAPI GetProcessAffinityMask(
#   __in   HANDLE hProcess,
#   __out  PDWORD_PTR lpProcessAffinityMask,
#   __out  PDWORD_PTR lpSystemAffinityMask
# );
def GetProcessAffinityMask(hProcess):
    _GetProcessAffinityMask = windll.kernel32.GetProcessAffinityMask
    _GetProcessAffinityMask.argtypes = [HANDLE, PDWORD_PTR, PDWORD_PTR]
    _GetProcessAffinityMask.restype  = bool
    _GetProcessAffinityMask.errcheck = RaiseIfZero

    lpProcessAffinityMask = DWORD_PTR(0)
    lpSystemAffinityMask  = DWORD_PTR(0)
    _GetProcessAffinityMask(hProcess, byref(lpProcessAffinityMask), byref(lpSystemAffinityMask))
    return lpProcessAffinityMask.value, lpSystemAffinityMask.value

# BOOL WINAPI SetProcessAffinityMask(
#   __in  HANDLE hProcess,
#   __in  DWORD_PTR dwProcessAffinityMask
# );
def SetProcessAffinityMask(hProcess, dwProcessAffinityMask):
    _SetProcessAffinityMask = windll.kernel32.SetProcessAffinityMask
    _SetProcessAffinityMask.argtypes = [HANDLE, DWORD_PTR]
    _SetProcessAffinityMask.restype  = bool
    _SetProcessAffinityMask.errcheck = RaiseIfZero
    _SetProcessAffinityMask(hProcess, dwProcessAffinityMask)

#------------------------------------------------------------------------------
# Toolhelp32 API

# HANDLE WINAPI CreateToolhelp32Snapshot(
#   __in  DWORD dwFlags,
#   __in  DWORD th32ProcessID
# );
def CreateToolhelp32Snapshot(dwFlags = TH32CS_SNAPALL, th32ProcessID = 0):
    _CreateToolhelp32Snapshot = windll.kernel32.CreateToolhelp32Snapshot
    _CreateToolhelp32Snapshot.argtypes = [DWORD, DWORD]
    _CreateToolhelp32Snapshot.restype  = HANDLE

    hSnapshot = _CreateToolhelp32Snapshot(dwFlags, th32ProcessID)
    if hSnapshot == INVALID_HANDLE_VALUE:
        raise ctypes.WinError()
    return SnapshotHandle(hSnapshot)

# BOOL WINAPI Process32First(
#   __in     HANDLE hSnapshot,
#   __inout  LPPROCESSENTRY32 lppe
# );
def Process32First(hSnapshot):
    _Process32First = windll.kernel32.Process32First
    _Process32First.argtypes = [HANDLE, LPPROCESSENTRY32]
    _Process32First.restype  = bool

    pe        = PROCESSENTRY32()
    pe.dwSize = sizeof(PROCESSENTRY32)
    success = _Process32First(hSnapshot, byref(pe))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return pe

# BOOL WINAPI Process32Next(
#   __in     HANDLE hSnapshot,
#   __out  LPPROCESSENTRY32 lppe
# );
def Process32Next(hSnapshot, pe = None):
    _Process32Next = windll.kernel32.Process32Next
    _Process32Next.argtypes = [HANDLE, LPPROCESSENTRY32]
    _Process32Next.restype  = bool

    if pe is None:
        pe = PROCESSENTRY32()
    pe.dwSize = sizeof(PROCESSENTRY32)
    success = _Process32Next(hSnapshot, byref(pe))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return pe

# BOOL WINAPI Thread32First(
#   __in     HANDLE hSnapshot,
#   __inout  LPTHREADENTRY32 lpte
# );
def Thread32First(hSnapshot):
    _Thread32First = windll.kernel32.Thread32First
    _Thread32First.argtypes = [HANDLE, LPTHREADENTRY32]
    _Thread32First.restype  = bool

    te = THREADENTRY32()
    te.dwSize = sizeof(THREADENTRY32)
    success = _Thread32First(hSnapshot, byref(te))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return te

# BOOL WINAPI Thread32Next(
#   __in     HANDLE hSnapshot,
#   __out  LPTHREADENTRY32 lpte
# );
def Thread32Next(hSnapshot, te = None):
    _Thread32Next = windll.kernel32.Thread32Next
    _Thread32Next.argtypes = [HANDLE, LPTHREADENTRY32]
    _Thread32Next.restype  = bool

    if te is None:
        te = THREADENTRY32()
    te.dwSize = sizeof(THREADENTRY32)
    success = _Thread32Next(hSnapshot, byref(te))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return te

# BOOL WINAPI Module32First(
#   __in     HANDLE hSnapshot,
#   __inout  LPMODULEENTRY32 lpme
# );
def Module32First(hSnapshot):
    _Module32First = windll.kernel32.Module32First
    _Module32First.argtypes = [HANDLE, LPMODULEENTRY32]
    _Module32First.restype  = bool

    me = MODULEENTRY32()
    me.dwSize = sizeof(MODULEENTRY32)
    success = _Module32First(hSnapshot, byref(me))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return me

# BOOL WINAPI Module32Next(
#   __in     HANDLE hSnapshot,
#   __out  LPMODULEENTRY32 lpme
# );
def Module32Next(hSnapshot, me = None):
    _Module32Next = windll.kernel32.Module32Next
    _Module32Next.argtypes = [HANDLE, LPMODULEENTRY32]
    _Module32Next.restype  = bool

    if me is None:
        me = MODULEENTRY32()
    me.dwSize = sizeof(MODULEENTRY32)
    success = _Module32Next(hSnapshot, byref(me))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return me

# BOOL WINAPI Heap32First(
#   __inout  LPHEAPENTRY32 lphe,
#   __in     DWORD th32ProcessID,
#   __in     ULONG_PTR th32HeapID
# );
def Heap32First(th32ProcessID, th32HeapID):
    _Heap32First = windll.kernel32.Heap32First
    _Heap32First.argtypes = [LPHEAPENTRY32, DWORD, ULONG_PTR]
    _Heap32First.restype  = bool

    he = HEAPENTRY32()
    he.dwSize = sizeof(HEAPENTRY32)
    success = _Heap32First(byref(he), th32ProcessID, th32HeapID)
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return he

# BOOL WINAPI Heap32Next(
#   __out  LPHEAPENTRY32 lphe
# );
def Heap32Next(he):
    _Heap32Next = windll.kernel32.Heap32Next
    _Heap32Next.argtypes = [LPHEAPENTRY32]
    _Heap32Next.restype  = bool

    he.dwSize = sizeof(HEAPENTRY32)
    success = _Heap32Next(byref(he))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return he

# BOOL WINAPI Heap32ListFirst(
#   __in     HANDLE hSnapshot,
#   __inout  LPHEAPLIST32 lphl
# );
def Heap32ListFirst(hSnapshot):
    _Heap32ListFirst = windll.kernel32.Heap32ListFirst
    _Heap32ListFirst.argtypes = [HANDLE, LPHEAPLIST32]
    _Heap32ListFirst.restype  = bool

    hl = HEAPLIST32()
    hl.dwSize = sizeof(HEAPLIST32)
    success = _Heap32ListFirst(hSnapshot, byref(hl))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return hl

# BOOL WINAPI Heap32ListNext(
#   __in     HANDLE hSnapshot,
#   __out  LPHEAPLIST32 lphl
# );
def Heap32ListNext(hSnapshot, hl = None):
    _Heap32ListNext = windll.kernel32.Heap32ListNext
    _Heap32ListNext.argtypes = [HANDLE, LPHEAPLIST32]
    _Heap32ListNext.restype  = bool

    if hl is None:
        hl = HEAPLIST32()
    hl.dwSize = sizeof(HEAPLIST32)
    success = _Heap32ListNext(hSnapshot, byref(hl))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return hl

# BOOL WINAPI Toolhelp32ReadProcessMemory(
#   __in   DWORD th32ProcessID,
#   __in   LPCVOID lpBaseAddress,
#   __out  LPVOID lpBuffer,
#   __in   SIZE_T cbRead,
#   __out  SIZE_T lpNumberOfBytesRead
# );
def Toolhelp32ReadProcessMemory(th32ProcessID, lpBaseAddress, cbRead):
    _Toolhelp32ReadProcessMemory = windll.kernel32.Toolhelp32ReadProcessMemory
    _Toolhelp32ReadProcessMemory.argtypes = [DWORD, LPVOID, LPVOID, SIZE_T, POINTER(SIZE_T)]
    _Toolhelp32ReadProcessMemory.restype  = bool

    lpBuffer            = ctypes.create_string_buffer('', cbRead)
    lpNumberOfBytesRead = SIZE_T(0)
    success = _Toolhelp32ReadProcessMemory(th32ProcessID, lpBaseAddress, lpBuffer, cbRead, byref(lpNumberOfBytesRead))
    if not success and GetLastError() != ERROR_PARTIAL_COPY:
        raise ctypes.WinError()
    return str(lpBuffer.raw)[:lpNumberOfBytesRead.value]

#------------------------------------------------------------------------------
# Miscellaneous system information

# BOOL WINAPI GetProcessDEPPolicy(
#  __in   HANDLE hProcess,
#  __out  LPDWORD lpFlags,
#  __out  PBOOL lpPermanent
# );
# Contribution by ivanlef0u (http://ivanlef0u.fr/)
# XP SP3 and > only
def GetProcessDEPPolicy(hProcess):
    _GetProcessDEPPolicy = windll.kernel32.GetProcessDEPPolicy
    _GetProcessDEPPolicy.argtypes = [HANDLE, LPDWORD, PBOOL]
    _GetProcessDEPPolicy.restype  = bool
    _GetProcessDEPPolicy.errcheck = RaiseIfZero

    lpFlags = DWORD(0)
    lpPermanent = BOOL(0)
    _GetProcessDEPPolicy(hProcess, byref(lpFlags), byref(lpPermanent))
    return (lpFlags.value, lpPermanent.value)

# DWORD WINAPI GetCurrentProcessorNumber(void);
def GetCurrentProcessorNumber():
    _GetCurrentProcessorNumber = windll.kernel32.GetCurrentProcessorNumber
    _GetCurrentProcessorNumber.argtypes = []
    _GetCurrentProcessorNumber.restype  = DWORD
    _GetCurrentProcessorNumber.errcheck = RaiseIfZero
    return _GetCurrentProcessorNumber()

# VOID WINAPI FlushProcessWriteBuffers(void);
def FlushProcessWriteBuffers():
    _FlushProcessWriteBuffers = windll.kernel32.FlushProcessWriteBuffers
    _FlushProcessWriteBuffers.argtypes = []
    _FlushProcessWriteBuffers.restype  = None
    _FlushProcessWriteBuffers()

# BOOL WINAPI GetLogicalProcessorInformation(
#   __out    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION Buffer,
#   __inout  PDWORD ReturnLength
# );

# TO DO http://msdn.microsoft.com/en-us/library/ms683194(VS.85).aspx

# BOOL WINAPI GetProcessIoCounters(
#   __in   HANDLE hProcess,
#   __out  PIO_COUNTERS lpIoCounters
# );

# TO DO http://msdn.microsoft.com/en-us/library/ms683218(VS.85).aspx

# DWORD WINAPI GetGuiResources(
#   __in  HANDLE hProcess,
#   __in  DWORD uiFlags
# );
def GetGuiResources(hProcess, uiFlags = GR_GDIOBJECTS):
    _GetGuiResources = windll.kernel32.GetGuiResources
    _GetGuiResources.argtypes = [HANDLE, DWORD]
    _GetGuiResources.restype  = DWORD

    dwCount = _GetGuiResources(hProcess, uiFlags)
    if dwCount == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return dwCount

# BOOL WINAPI GetProcessHandleCount(
#   __in     HANDLE hProcess,
#   __inout  PDWORD pdwHandleCount
# );
def GetProcessHandleCount(hProcess):
    _GetProcessHandleCount = windll.kernel32.GetProcessHandleCount
    _GetProcessHandleCount.argtypes = [HANDLE, PDWORD]
    _GetProcessHandleCount.restype  = DWORD
    _GetProcessHandleCount.errcheck = RaiseIfZero

    pdwHandleCount = DWORD(0)
    _GetProcessHandleCount(hProcess, byref(pdwHandleCount))
    return pdwHandleCount.value

# BOOL WINAPI GetProcessTimes(
#   __in   HANDLE hProcess,
#   __out  LPFILETIME lpCreationTime,
#   __out  LPFILETIME lpExitTime,
#   __out  LPFILETIME lpKernelTime,
#   __out  LPFILETIME lpUserTime
# );
def GetProcessTimes(hProcess = None):
    _GetProcessTimes = windll.kernel32.GetProcessTimes
    _GetProcessTimes.argtypes = [HANDLE, LPFILETIME, LPFILETIME, LPFILETIME, LPFILETIME]
    _GetProcessTimes.restype  = bool
    _GetProcessTimes.errcheck = RaiseIfZero

    if hProcess is None:
        hProcess = GetCurrentProcess()

    CreationTime = FILETIME()
    ExitTime     = FILETIME()
    KernelTime   = FILETIME()
    UserTime     = FILETIME()

    _GetProcessTimes(hProcess, byref(CreationTime), byref(ExitTime), byref(KernelTime), byref(UserTime))

    return (CreationTime, ExitTime, KernelTime, UserTime)

# BOOL WINAPI FileTimeToSystemTime(
#   __in   const FILETIME *lpFileTime,
#   __out  LPSYSTEMTIME lpSystemTime
# );
def FileTimeToSystemTime(lpFileTime):
    _FileTimeToSystemTime = windll.kernel32.FileTimeToSystemTime
    _FileTimeToSystemTime.argtypes = [LPFILETIME, LPSYSTEMTIME]
    _FileTimeToSystemTime.restype  = bool
    _FileTimeToSystemTime.errcheck = RaiseIfZero

    if isinstance(lpFileTime, FILETIME):
        FileTime = lpFileTime
    else:
        FileTime = FILETIME()
        FileTime.dwLowDateTime  = lpFileTime & 0xFFFFFFFF
        FileTime.dwHighDateTime = lpFileTime >> 32
    SystemTime = SYSTEMTIME()
    _FileTimeToSystemTime(byref(FileTime), byref(SystemTime))
    return SystemTime

# void WINAPI GetSystemTimeAsFileTime(
#   __out  LPFILETIME lpSystemTimeAsFileTime
# );
def GetSystemTimeAsFileTime():
    _GetSystemTimeAsFileTime = windll.kernel32.GetSystemTimeAsFileTime
    _GetSystemTimeAsFileTime.argtypes = [LPFILETIME]
    _GetSystemTimeAsFileTime.restype  = None

    FileTime = FILETIME()
    _GetSystemTimeAsFileTime(byref(FileTime))
    return FileTime

#------------------------------------------------------------------------------
# Global ATOM API

# ATOM GlobalAddAtom(
#   __in  LPCTSTR lpString
# );
def GlobalAddAtomA(lpString):
    _GlobalAddAtomA = windll.kernel32.GlobalAddAtomA
    _GlobalAddAtomA.argtypes = [LPSTR]
    _GlobalAddAtomA.restype  = ATOM
    _GlobalAddAtomA.errcheck = RaiseIfZero
    return _GlobalAddAtomA(lpString)

def GlobalAddAtomW(lpString):
    _GlobalAddAtomW = windll.kernel32.GlobalAddAtomW
    _GlobalAddAtomW.argtypes = [LPWSTR]
    _GlobalAddAtomW.restype  = ATOM
    _GlobalAddAtomW.errcheck = RaiseIfZero
    return _GlobalAddAtomW(lpString)

GlobalAddAtom = GuessStringType(GlobalAddAtomA, GlobalAddAtomW)

# ATOM GlobalFindAtom(
#   __in  LPCTSTR lpString
# );
def GlobalFindAtomA(lpString):
    _GlobalFindAtomA = windll.kernel32.GlobalFindAtomA
    _GlobalFindAtomA.argtypes = [LPSTR]
    _GlobalFindAtomA.restype  = ATOM
    _GlobalFindAtomA.errcheck = RaiseIfZero
    return _GlobalFindAtomA(lpString)

def GlobalFindAtomW(lpString):
    _GlobalFindAtomW = windll.kernel32.GlobalFindAtomW
    _GlobalFindAtomW.argtypes = [LPWSTR]
    _GlobalFindAtomW.restype  = ATOM
    _GlobalFindAtomW.errcheck = RaiseIfZero
    return _GlobalFindAtomW(lpString)

GlobalFindAtom = GuessStringType(GlobalFindAtomA, GlobalFindAtomW)

# UINT GlobalGetAtomName(
#   __in   ATOM nAtom,
#   __out  LPTSTR lpBuffer,
#   __in   int nSize
# );
def GlobalGetAtomNameA(nAtom):
    _GlobalGetAtomNameA = windll.kernel32.GlobalGetAtomNameA
    _GlobalGetAtomNameA.argtypes = [ATOM, LPSTR, ctypes.c_int]
    _GlobalGetAtomNameA.restype  = UINT
    _GlobalGetAtomNameA.errcheck = RaiseIfZero

    nSize = 64
    while 1:
        lpBuffer = ctypes.create_string_buffer("", nSize)
        nCopied  = _GlobalGetAtomNameA(nAtom, lpBuffer, nSize)
        if nCopied < nSize - 1:
            break
        nSize = nSize + 64
    return lpBuffer.value

def GlobalGetAtomNameW(nAtom):
    _GlobalGetAtomNameW = windll.kernel32.GlobalGetAtomNameW
    _GlobalGetAtomNameW.argtypes = [ATOM, LPWSTR, ctypes.c_int]
    _GlobalGetAtomNameW.restype  = UINT
    _GlobalGetAtomNameW.errcheck = RaiseIfZero

    nSize = 64
    while 1:
        lpBuffer = ctypes.create_unicode_buffer(u"", nSize)
        nCopied  = _GlobalGetAtomNameW(nAtom, lpBuffer, nSize)
        if nCopied < nSize - 1:
            break
        nSize = nSize + 64
    return lpBuffer.value

GlobalGetAtomName = GuessStringType(GlobalGetAtomNameA, GlobalGetAtomNameW)

# ATOM GlobalDeleteAtom(
#   __in  ATOM nAtom
# );
def GlobalDeleteAtom(nAtom):
    _GlobalDeleteAtom = windll.kernel32.GlobalDeleteAtom
    _GlobalDeleteAtom.argtypes
    _GlobalDeleteAtom.restype
    SetLastError(ERROR_SUCCESS)
    _GlobalDeleteAtom(nAtom)
    error = GetLastError()
    if error != ERROR_SUCCESS:
        raise ctypes.WinError(error)

#------------------------------------------------------------------------------
# Wow64

# DWORD WINAPI Wow64SuspendThread(
#   _In_  HANDLE hThread
# );
def Wow64SuspendThread(hThread):
    _Wow64SuspendThread = windll.kernel32.Wow64SuspendThread
    _Wow64SuspendThread.argtypes = [HANDLE]
    _Wow64SuspendThread.restype  = DWORD

    previousCount = _Wow64SuspendThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount

# BOOLEAN WINAPI Wow64EnableWow64FsRedirection(
#   __in  BOOLEAN Wow64FsEnableRedirection
# );
def Wow64EnableWow64FsRedirection(Wow64FsEnableRedirection):
    """
    This function may not work reliably when there are nested calls. Therefore,
    this function has been replaced by the L{Wow64DisableWow64FsRedirection}
    and L{Wow64RevertWow64FsRedirection} functions.

    @see: U{http://msdn.microsoft.com/en-us/library/windows/desktop/aa365744(v=vs.85).aspx}
    """
    _Wow64EnableWow64FsRedirection = windll.kernel32.Wow64EnableWow64FsRedirection
    _Wow64EnableWow64FsRedirection.argtypes = [BOOLEAN]
    _Wow64EnableWow64FsRedirection.restype  = BOOLEAN
    _Wow64EnableWow64FsRedirection.errcheck = RaiseIfZero

# BOOL WINAPI Wow64DisableWow64FsRedirection(
#   __out  PVOID *OldValue
# );
def Wow64DisableWow64FsRedirection():
    _Wow64DisableWow64FsRedirection = windll.kernel32.Wow64DisableWow64FsRedirection
    _Wow64DisableWow64FsRedirection.argtypes = [PPVOID]
    _Wow64DisableWow64FsRedirection.restype  = BOOL
    _Wow64DisableWow64FsRedirection.errcheck = RaiseIfZero

    OldValue = PVOID(None)
    _Wow64DisableWow64FsRedirection(byref(OldValue))
    return OldValue

# BOOL WINAPI Wow64RevertWow64FsRedirection(
#   __in  PVOID OldValue
# );
def Wow64RevertWow64FsRedirection(OldValue):
    _Wow64RevertWow64FsRedirection = windll.kernel32.Wow64RevertWow64FsRedirection
    _Wow64RevertWow64FsRedirection.argtypes = [PVOID]
    _Wow64RevertWow64FsRedirection.restype  = BOOL
    _Wow64RevertWow64FsRedirection.errcheck = RaiseIfZero
    _Wow64RevertWow64FsRedirection(OldValue)

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================

#==============================================================================
# Mark functions that Psyco cannot compile.
# In your programs, don't use psyco.full().
# Call psyco.bind() on your main function instead.

try:
    import psyco
    psyco.cannotcompile(WaitForDebugEvent)
    psyco.cannotcompile(WaitForSingleObject)
    psyco.cannotcompile(WaitForSingleObjectEx)
    psyco.cannotcompile(WaitForMultipleObjects)
    psyco.cannotcompile(WaitForMultipleObjectsEx)
except ImportError:
    pass
#==============================================================================
