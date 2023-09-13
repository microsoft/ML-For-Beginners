# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import ctypes
from ctypes.wintypes import BOOL, DWORD, HANDLE, LARGE_INTEGER, LPCSTR, UINT

from debugpy.common import log


JOBOBJECTCLASS = ctypes.c_int
LPDWORD = ctypes.POINTER(DWORD)
LPVOID = ctypes.c_void_p
SIZE_T = ctypes.c_size_t
ULONGLONG = ctypes.c_ulonglong


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ULONGLONG),
        ("WriteOperationCount", ULONGLONG),
        ("OtherOperationCount", ULONGLONG),
        ("ReadTransferCount", ULONGLONG),
        ("WriteTransferCount", ULONGLONG),
        ("OtherTransferCount", ULONGLONG),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", LARGE_INTEGER),
        ("PerJobUserTimeLimit", LARGE_INTEGER),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", SIZE_T),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    ]


JobObjectExtendedLimitInformation = JOBOBJECTCLASS(9)

JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100


def _errcheck(is_error_result=(lambda result: not result)):
    def impl(result, func, args):
        if is_error_result(result):
            log.debug("{0} returned {1}", func.__name__, result)
            raise ctypes.WinError()
        else:
            return result

    return impl


kernel32 = ctypes.windll.kernel32

kernel32.AssignProcessToJobObject.errcheck = _errcheck()
kernel32.AssignProcessToJobObject.restype = BOOL
kernel32.AssignProcessToJobObject.argtypes = (HANDLE, HANDLE)

kernel32.CreateJobObjectA.errcheck = _errcheck(lambda result: result == 0)
kernel32.CreateJobObjectA.restype = HANDLE
kernel32.CreateJobObjectA.argtypes = (LPVOID, LPCSTR)

kernel32.OpenProcess.errcheck = _errcheck(lambda result: result == 0)
kernel32.OpenProcess.restype = HANDLE
kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)

kernel32.QueryInformationJobObject.errcheck = _errcheck()
kernel32.QueryInformationJobObject.restype = BOOL
kernel32.QueryInformationJobObject.argtypes = (
    HANDLE,
    JOBOBJECTCLASS,
    LPVOID,
    DWORD,
    LPDWORD,
)

kernel32.SetInformationJobObject.errcheck = _errcheck()
kernel32.SetInformationJobObject.restype = BOOL
kernel32.SetInformationJobObject.argtypes = (HANDLE, JOBOBJECTCLASS, LPVOID, DWORD)

kernel32.TerminateJobObject.errcheck = _errcheck()
kernel32.TerminateJobObject.restype = BOOL
kernel32.TerminateJobObject.argtypes = (HANDLE, UINT)
