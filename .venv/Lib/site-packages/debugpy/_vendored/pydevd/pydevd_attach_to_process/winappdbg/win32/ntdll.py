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
Wrapper for ntdll.dll in ctypes.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
_all.add('peb_teb')
#==============================================================================

from winappdbg.win32.peb_teb import *

#--- Types --------------------------------------------------------------------

SYSDBG_COMMAND          = DWORD
PROCESSINFOCLASS        = DWORD
THREADINFOCLASS         = DWORD
FILE_INFORMATION_CLASS  = DWORD

#--- Constants ----------------------------------------------------------------

# DEP flags for ProcessExecuteFlags
MEM_EXECUTE_OPTION_ENABLE               = 1
MEM_EXECUTE_OPTION_DISABLE              = 2
MEM_EXECUTE_OPTION_ATL7_THUNK_EMULATION = 4
MEM_EXECUTE_OPTION_PERMANENT            = 8

# SYSTEM_INFORMATION_CLASS
# http://www.informit.com/articles/article.aspx?p=22442&seqNum=4
SystemBasicInformation                  = 1     # 0x002C
SystemProcessorInformation              = 2     # 0x000C
SystemPerformanceInformation            = 3     # 0x0138
SystemTimeInformation                   = 4     # 0x0020
SystemPathInformation                   = 5     # not implemented
SystemProcessInformation                = 6     # 0x00F8 + per process
SystemCallInformation                   = 7     # 0x0018 + (n * 0x0004)
SystemConfigurationInformation          = 8     # 0x0018
SystemProcessorCounters                 = 9     # 0x0030 per cpu
SystemGlobalFlag                        = 10    # 0x0004
SystemInfo10                            = 11    # not implemented
SystemModuleInformation                 = 12    # 0x0004 + (n * 0x011C)
SystemLockInformation                   = 13    # 0x0004 + (n * 0x0024)
SystemInfo13                            = 14    # not implemented
SystemPagedPoolInformation              = 15    # checked build only
SystemNonPagedPoolInformation           = 16    # checked build only
SystemHandleInformation                 = 17    # 0x0004 + (n * 0x0010)
SystemObjectInformation                 = 18    # 0x0038+ + (n * 0x0030+)
SystemPagefileInformation               = 19    # 0x0018+ per page file
SystemInstemulInformation               = 20    # 0x0088
SystemInfo20                            = 21    # invalid info class
SystemCacheInformation                  = 22    # 0x0024
SystemPoolTagInformation                = 23    # 0x0004 + (n * 0x001C)
SystemProcessorStatistics               = 24    # 0x0000, or 0x0018 per cpu
SystemDpcInformation                    = 25    # 0x0014
SystemMemoryUsageInformation1           = 26    # checked build only
SystemLoadImage                         = 27    # 0x0018, set mode only
SystemUnloadImage                       = 28    # 0x0004, set mode only
SystemTimeAdjustmentInformation         = 29    # 0x000C, 0x0008 writeable
SystemMemoryUsageInformation2           = 30    # checked build only
SystemInfo30                            = 31    # checked build only
SystemInfo31                            = 32    # checked build only
SystemCrashDumpInformation              = 33    # 0x0004
SystemExceptionInformation              = 34    # 0x0010
SystemCrashDumpStateInformation         = 35    # 0x0008
SystemDebuggerInformation               = 36    # 0x0002
SystemThreadSwitchInformation           = 37    # 0x0030
SystemRegistryQuotaInformation          = 38    # 0x000C
SystemLoadDriver                        = 39    # 0x0008, set mode only
SystemPrioritySeparationInformation     = 40    # 0x0004, set mode only
SystemInfo40                            = 41    # not implemented
SystemInfo41                            = 42    # not implemented
SystemInfo42                            = 43    # invalid info class
SystemInfo43                            = 44    # invalid info class
SystemTimeZoneInformation               = 45    # 0x00AC
SystemLookasideInformation              = 46    # n * 0x0020
# info classes specific to Windows 2000
# WTS = Windows Terminal Server
SystemSetTimeSlipEvent                  = 47    # set mode only
SystemCreateSession                     = 48    # WTS, set mode only
SystemDeleteSession                     = 49    # WTS, set mode only
SystemInfo49                            = 50    # invalid info class
SystemRangeStartInformation             = 51    # 0x0004
SystemVerifierInformation               = 52    # 0x0068
SystemAddVerifier                       = 53    # set mode only
SystemSessionProcessesInformation       = 54    # WTS

# NtQueryInformationProcess constants (from MSDN)
##ProcessBasicInformation = 0
##ProcessDebugPort        = 7
##ProcessWow64Information = 26
##ProcessImageFileName    = 27

# PROCESS_INFORMATION_CLASS
# http://undocumented.ntinternals.net/UserMode/Undocumented%20Functions/NT%20Objects/Process/PROCESS_INFORMATION_CLASS.html
ProcessBasicInformation             = 0
ProcessQuotaLimits                  = 1
ProcessIoCounters                   = 2
ProcessVmCounters                   = 3
ProcessTimes                        = 4
ProcessBasePriority                 = 5
ProcessRaisePriority                = 6
ProcessDebugPort                    = 7
ProcessExceptionPort                = 8
ProcessAccessToken                  = 9
ProcessLdtInformation               = 10
ProcessLdtSize                      = 11
ProcessDefaultHardErrorMode         = 12
ProcessIoPortHandlers               = 13
ProcessPooledUsageAndLimits         = 14
ProcessWorkingSetWatch              = 15
ProcessUserModeIOPL                 = 16
ProcessEnableAlignmentFaultFixup    = 17
ProcessPriorityClass                = 18
ProcessWx86Information              = 19
ProcessHandleCount                  = 20
ProcessAffinityMask                 = 21
ProcessPriorityBoost                = 22

ProcessWow64Information             = 26
ProcessImageFileName                = 27

# http://www.codeproject.com/KB/security/AntiReverseEngineering.aspx
ProcessDebugObjectHandle            = 30

ProcessExecuteFlags                 = 34

# THREAD_INFORMATION_CLASS
ThreadBasicInformation              = 0
ThreadTimes                         = 1
ThreadPriority                      = 2
ThreadBasePriority                  = 3
ThreadAffinityMask                  = 4
ThreadImpersonationToken            = 5
ThreadDescriptorTableEntry          = 6
ThreadEnableAlignmentFaultFixup     = 7
ThreadEventPair                     = 8
ThreadQuerySetWin32StartAddress     = 9
ThreadZeroTlsCell                   = 10
ThreadPerformanceCount              = 11
ThreadAmILastThread                 = 12
ThreadIdealProcessor                = 13
ThreadPriorityBoost                 = 14
ThreadSetTlsArrayAddress            = 15
ThreadIsIoPending                   = 16
ThreadHideFromDebugger              = 17

# OBJECT_INFORMATION_CLASS
ObjectBasicInformation              = 0
ObjectNameInformation               = 1
ObjectTypeInformation               = 2
ObjectAllTypesInformation           = 3
ObjectHandleInformation             = 4

# FILE_INFORMATION_CLASS
FileDirectoryInformation            = 1
FileFullDirectoryInformation        = 2
FileBothDirectoryInformation        = 3
FileBasicInformation                = 4
FileStandardInformation             = 5
FileInternalInformation             = 6
FileEaInformation                   = 7
FileAccessInformation               = 8
FileNameInformation                 = 9
FileRenameInformation               = 10
FileLinkInformation                 = 11
FileNamesInformation                = 12
FileDispositionInformation          = 13
FilePositionInformation             = 14
FileFullEaInformation               = 15
FileModeInformation                 = 16
FileAlignmentInformation            = 17
FileAllInformation                  = 18
FileAllocationInformation           = 19
FileEndOfFileInformation            = 20
FileAlternateNameInformation        = 21
FileStreamInformation               = 22
FilePipeInformation                 = 23
FilePipeLocalInformation            = 24
FilePipeRemoteInformation           = 25
FileMailslotQueryInformation        = 26
FileMailslotSetInformation          = 27
FileCompressionInformation          = 28
FileCopyOnWriteInformation          = 29
FileCompletionInformation           = 30
FileMoveClusterInformation          = 31
FileQuotaInformation                = 32
FileReparsePointInformation         = 33
FileNetworkOpenInformation          = 34
FileObjectIdInformation             = 35
FileTrackingInformation             = 36
FileOleDirectoryInformation         = 37
FileContentIndexInformation         = 38
FileInheritContentIndexInformation  = 37
FileOleInformation                  = 39
FileMaximumInformation              = 40

# From http://www.nirsoft.net/kernel_struct/vista/EXCEPTION_DISPOSITION.html
# typedef enum _EXCEPTION_DISPOSITION
# {
#          ExceptionContinueExecution = 0,
#          ExceptionContinueSearch = 1,
#          ExceptionNestedException = 2,
#          ExceptionCollidedUnwind = 3
# } EXCEPTION_DISPOSITION;
ExceptionContinueExecution  = 0
ExceptionContinueSearch     = 1
ExceptionNestedException    = 2
ExceptionCollidedUnwind     = 3

#--- PROCESS_BASIC_INFORMATION structure --------------------------------------

# From MSDN:
#
# typedef struct _PROCESS_BASIC_INFORMATION {
#     PVOID Reserved1;
#     PPEB PebBaseAddress;
#     PVOID Reserved2[2];
#     ULONG_PTR UniqueProcessId;
#     PVOID Reserved3;
# } PROCESS_BASIC_INFORMATION;
##class PROCESS_BASIC_INFORMATION(Structure):
##    _fields_ = [
##        ("Reserved1",       PVOID),
##        ("PebBaseAddress",  PPEB),
##        ("Reserved2",       PVOID * 2),
##        ("UniqueProcessId", ULONG_PTR),
##        ("Reserved3",       PVOID),
##]

# From http://catch22.net/tuts/tips2
# (Only valid for 32 bits)
#
# typedef struct
# {
#     ULONG      ExitStatus;
#     PVOID      PebBaseAddress;
#     ULONG      AffinityMask;
#     ULONG      BasePriority;
#     ULONG_PTR  UniqueProcessId;
#     ULONG_PTR  InheritedFromUniqueProcessId;
# } PROCESS_BASIC_INFORMATION;

# My own definition follows:
class PROCESS_BASIC_INFORMATION(Structure):
    _fields_ = [
        ("ExitStatus",                      SIZE_T),
        ("PebBaseAddress",                  PVOID),     # PPEB
        ("AffinityMask",                    KAFFINITY),
        ("BasePriority",                    SDWORD),
        ("UniqueProcessId",                 ULONG_PTR),
        ("InheritedFromUniqueProcessId",    ULONG_PTR),
]

#--- THREAD_BASIC_INFORMATION structure ---------------------------------------

# From http://undocumented.ntinternals.net/UserMode/Structures/THREAD_BASIC_INFORMATION.html
#
# typedef struct _THREAD_BASIC_INFORMATION {
#   NTSTATUS ExitStatus;
#   PVOID TebBaseAddress;
#   CLIENT_ID ClientId;
#   KAFFINITY AffinityMask;
#   KPRIORITY Priority;
#   KPRIORITY BasePriority;
# } THREAD_BASIC_INFORMATION, *PTHREAD_BASIC_INFORMATION;
class THREAD_BASIC_INFORMATION(Structure):
    _fields_ = [
        ("ExitStatus",      NTSTATUS),
        ("TebBaseAddress",  PVOID),     # PTEB
        ("ClientId",        CLIENT_ID),
        ("AffinityMask",    KAFFINITY),
        ("Priority",        SDWORD),
        ("BasePriority",    SDWORD),
]

#--- FILE_NAME_INFORMATION structure ------------------------------------------

# typedef struct _FILE_NAME_INFORMATION {
#     ULONG FileNameLength;
#     WCHAR FileName[1];
# } FILE_NAME_INFORMATION, *PFILE_NAME_INFORMATION;
class FILE_NAME_INFORMATION(Structure):
    _fields_ = [
        ("FileNameLength",  ULONG),
        ("FileName",        WCHAR * 1),
    ]

#--- SYSDBG_MSR structure and constants ---------------------------------------

SysDbgReadMsr  = 16
SysDbgWriteMsr = 17

class SYSDBG_MSR(Structure):
    _fields_ = [
        ("Address", ULONG),
        ("Data",    ULONGLONG),
]

#--- IO_STATUS_BLOCK structure ------------------------------------------------

# typedef struct _IO_STATUS_BLOCK {
#     union {
#         NTSTATUS Status;
#         PVOID Pointer;
#     };
#     ULONG_PTR Information;
# } IO_STATUS_BLOCK, *PIO_STATUS_BLOCK;
class IO_STATUS_BLOCK(Structure):
    _fields_ = [
        ("Status",      NTSTATUS),
        ("Information", ULONG_PTR),
    ]
    def __get_Pointer(self):
        return PVOID(self.Status)
    def __set_Pointer(self, ptr):
        self.Status = ptr.value
    Pointer = property(__get_Pointer, __set_Pointer)

PIO_STATUS_BLOCK = POINTER(IO_STATUS_BLOCK)

#--- ntdll.dll ----------------------------------------------------------------

# ULONG WINAPI RtlNtStatusToDosError(
#   __in  NTSTATUS Status
# );
def RtlNtStatusToDosError(Status):
    _RtlNtStatusToDosError = windll.ntdll.RtlNtStatusToDosError
    _RtlNtStatusToDosError.argtypes = [NTSTATUS]
    _RtlNtStatusToDosError.restype = ULONG
    return _RtlNtStatusToDosError(Status)

# NTSYSAPI NTSTATUS NTAPI NtSystemDebugControl(
#   IN SYSDBG_COMMAND Command,
#   IN PVOID InputBuffer OPTIONAL,
#   IN ULONG InputBufferLength,
#   OUT PVOID OutputBuffer OPTIONAL,
#   IN ULONG OutputBufferLength,
#   OUT PULONG ReturnLength OPTIONAL
# );
def NtSystemDebugControl(Command, InputBuffer = None, InputBufferLength = None, OutputBuffer = None, OutputBufferLength = None):
    _NtSystemDebugControl = windll.ntdll.NtSystemDebugControl
    _NtSystemDebugControl.argtypes = [SYSDBG_COMMAND, PVOID, ULONG, PVOID, ULONG, PULONG]
    _NtSystemDebugControl.restype = NTSTATUS

    # Validate the input buffer
    if InputBuffer is None:
        if InputBufferLength is None:
            InputBufferLength = 0
        else:
            raise ValueError(
                "Invalid call to NtSystemDebugControl: "
                "input buffer length given but no input buffer!")
    else:
        if InputBufferLength is None:
            InputBufferLength = sizeof(InputBuffer)
        InputBuffer = byref(InputBuffer)

    # Validate the output buffer
    if OutputBuffer is None:
        if OutputBufferLength is None:
            OutputBufferLength = 0
        else:
            OutputBuffer = ctypes.create_string_buffer("", OutputBufferLength)
    elif OutputBufferLength is None:
        OutputBufferLength = sizeof(OutputBuffer)

    # Make the call (with an output buffer)
    if OutputBuffer is not None:
        ReturnLength = ULONG(0)
        ntstatus = _NtSystemDebugControl(Command, InputBuffer, InputBufferLength, byref(OutputBuffer), OutputBufferLength, byref(ReturnLength))
        if ntstatus != 0:
            raise ctypes.WinError( RtlNtStatusToDosError(ntstatus) )
        ReturnLength = ReturnLength.value
        if ReturnLength != OutputBufferLength:
            raise ctypes.WinError(ERROR_BAD_LENGTH)
        return OutputBuffer, ReturnLength

    # Make the call (without an output buffer)
    ntstatus = _NtSystemDebugControl(Command, InputBuffer, InputBufferLength, OutputBuffer, OutputBufferLength, None)
    if ntstatus != 0:
        raise ctypes.WinError( RtlNtStatusToDosError(ntstatus) )

ZwSystemDebugControl = NtSystemDebugControl

# NTSTATUS WINAPI NtQueryInformationProcess(
#   __in       HANDLE ProcessHandle,
#   __in       PROCESSINFOCLASS ProcessInformationClass,
#   __out      PVOID ProcessInformation,
#   __in       ULONG ProcessInformationLength,
#   __out_opt  PULONG ReturnLength
# );
def NtQueryInformationProcess(ProcessHandle, ProcessInformationClass, ProcessInformationLength = None):
    _NtQueryInformationProcess = windll.ntdll.NtQueryInformationProcess
    _NtQueryInformationProcess.argtypes = [HANDLE, PROCESSINFOCLASS, PVOID, ULONG, PULONG]
    _NtQueryInformationProcess.restype = NTSTATUS
    if ProcessInformationLength is not None:
        ProcessInformation = ctypes.create_string_buffer("", ProcessInformationLength)
    else:
        if   ProcessInformationClass == ProcessBasicInformation:
            ProcessInformation = PROCESS_BASIC_INFORMATION()
            ProcessInformationLength = sizeof(PROCESS_BASIC_INFORMATION)
        elif ProcessInformationClass == ProcessImageFileName:
            unicode_buffer = ctypes.create_unicode_buffer(u"", 0x1000)
            ProcessInformation = UNICODE_STRING(0, 0x1000, addressof(unicode_buffer))
            ProcessInformationLength = sizeof(UNICODE_STRING)
        elif ProcessInformationClass in (ProcessDebugPort, ProcessWow64Information, ProcessWx86Information, ProcessHandleCount, ProcessPriorityBoost):
            ProcessInformation = DWORD()
            ProcessInformationLength = sizeof(DWORD)
        else:
            raise Exception("Unknown ProcessInformationClass, use an explicit ProcessInformationLength value instead")
    ReturnLength = ULONG(0)
    ntstatus = _NtQueryInformationProcess(ProcessHandle, ProcessInformationClass, byref(ProcessInformation), ProcessInformationLength, byref(ReturnLength))
    if ntstatus != 0:
        raise ctypes.WinError( RtlNtStatusToDosError(ntstatus) )
    if   ProcessInformationClass == ProcessBasicInformation:
        retval = ProcessInformation
    elif ProcessInformationClass in (ProcessDebugPort, ProcessWow64Information, ProcessWx86Information, ProcessHandleCount, ProcessPriorityBoost):
        retval = ProcessInformation.value
    elif ProcessInformationClass == ProcessImageFileName:
        vptr = ctypes.c_void_p(ProcessInformation.Buffer)
        cptr = ctypes.cast( vptr, ctypes.c_wchar * ProcessInformation.Length )
        retval = cptr.contents.raw
    else:
        retval = ProcessInformation.raw[:ReturnLength.value]
    return retval

ZwQueryInformationProcess = NtQueryInformationProcess

# NTSTATUS WINAPI NtQueryInformationThread(
#   __in       HANDLE ThreadHandle,
#   __in       THREADINFOCLASS ThreadInformationClass,
#   __out      PVOID ThreadInformation,
#   __in       ULONG ThreadInformationLength,
#   __out_opt  PULONG ReturnLength
# );
def NtQueryInformationThread(ThreadHandle, ThreadInformationClass, ThreadInformationLength = None):
    _NtQueryInformationThread = windll.ntdll.NtQueryInformationThread
    _NtQueryInformationThread.argtypes = [HANDLE, THREADINFOCLASS, PVOID, ULONG, PULONG]
    _NtQueryInformationThread.restype = NTSTATUS
    if ThreadInformationLength is not None:
        ThreadInformation = ctypes.create_string_buffer("", ThreadInformationLength)
    else:
        if   ThreadInformationClass == ThreadBasicInformation:
            ThreadInformation = THREAD_BASIC_INFORMATION()
        elif ThreadInformationClass == ThreadHideFromDebugger:
            ThreadInformation = BOOLEAN()
        elif ThreadInformationClass == ThreadQuerySetWin32StartAddress:
            ThreadInformation = PVOID()
        elif ThreadInformationClass in (ThreadAmILastThread, ThreadPriorityBoost):
            ThreadInformation = DWORD()
        elif ThreadInformationClass == ThreadPerformanceCount:
            ThreadInformation = LONGLONG()  # LARGE_INTEGER
        else:
            raise Exception("Unknown ThreadInformationClass, use an explicit ThreadInformationLength value instead")
        ThreadInformationLength = sizeof(ThreadInformation)
    ReturnLength = ULONG(0)
    ntstatus = _NtQueryInformationThread(ThreadHandle, ThreadInformationClass, byref(ThreadInformation), ThreadInformationLength, byref(ReturnLength))
    if ntstatus != 0:
        raise ctypes.WinError( RtlNtStatusToDosError(ntstatus) )
    if   ThreadInformationClass == ThreadBasicInformation:
        retval = ThreadInformation
    elif ThreadInformationClass == ThreadHideFromDebugger:
        retval = bool(ThreadInformation.value)
    elif ThreadInformationClass in (ThreadQuerySetWin32StartAddress, ThreadAmILastThread, ThreadPriorityBoost, ThreadPerformanceCount):
        retval = ThreadInformation.value
    else:
        retval = ThreadInformation.raw[:ReturnLength.value]
    return retval

ZwQueryInformationThread = NtQueryInformationThread

# NTSTATUS
#   NtQueryInformationFile(
#     IN HANDLE  FileHandle,
#     OUT PIO_STATUS_BLOCK  IoStatusBlock,
#     OUT PVOID  FileInformation,
#     IN ULONG  Length,
#     IN FILE_INFORMATION_CLASS  FileInformationClass
#     );
def NtQueryInformationFile(FileHandle, FileInformationClass, FileInformation, Length):
    _NtQueryInformationFile = windll.ntdll.NtQueryInformationFile
    _NtQueryInformationFile.argtypes = [HANDLE, PIO_STATUS_BLOCK, PVOID, ULONG, DWORD]
    _NtQueryInformationFile.restype = NTSTATUS
    IoStatusBlock = IO_STATUS_BLOCK()
    ntstatus = _NtQueryInformationFile(FileHandle, byref(IoStatusBlock), byref(FileInformation), Length, FileInformationClass)
    if ntstatus != 0:
        raise ctypes.WinError( RtlNtStatusToDosError(ntstatus) )
    return IoStatusBlock

ZwQueryInformationFile = NtQueryInformationFile

# DWORD STDCALL CsrGetProcessId (VOID);
def CsrGetProcessId():
    _CsrGetProcessId = windll.ntdll.CsrGetProcessId
    _CsrGetProcessId.argtypes = []
    _CsrGetProcessId.restype = DWORD
    return _CsrGetProcessId()

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
