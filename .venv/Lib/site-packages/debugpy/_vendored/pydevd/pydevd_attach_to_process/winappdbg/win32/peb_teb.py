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
PEB and TEB structures, constants and data types.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *
from winappdbg.win32.version import os

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

#--- PEB and TEB structures, constants and data types -------------------------

# From http://www.nirsoft.net/kernel_struct/vista/CLIENT_ID.html
#
# typedef struct _CLIENT_ID
# {
#     PVOID UniqueProcess;
#     PVOID UniqueThread;
# } CLIENT_ID, *PCLIENT_ID;
class CLIENT_ID(Structure):
    _fields_ = [
        ("UniqueProcess",   PVOID),
        ("UniqueThread",    PVOID),
]

# From MSDN:
#
# typedef struct _LDR_DATA_TABLE_ENTRY {
#     BYTE Reserved1[2];
#     LIST_ENTRY InMemoryOrderLinks;
#     PVOID Reserved2[2];
#     PVOID DllBase;
#     PVOID EntryPoint;
#     PVOID Reserved3;
#     UNICODE_STRING FullDllName;
#     BYTE Reserved4[8];
#     PVOID Reserved5[3];
#     union {
#         ULONG CheckSum;
#         PVOID Reserved6;
#     };
#     ULONG TimeDateStamp;
# } LDR_DATA_TABLE_ENTRY, *PLDR_DATA_TABLE_ENTRY;
##class LDR_DATA_TABLE_ENTRY(Structure):
##    _fields_ = [
##        ("Reserved1",           BYTE * 2),
##        ("InMemoryOrderLinks",  LIST_ENTRY),
##        ("Reserved2",           PVOID * 2),
##        ("DllBase",             PVOID),
##        ("EntryPoint",          PVOID),
##        ("Reserved3",           PVOID),
##        ("FullDllName",           UNICODE_STRING),
##        ("Reserved4",           BYTE * 8),
##        ("Reserved5",           PVOID * 3),
##        ("CheckSum",            ULONG),
##        ("TimeDateStamp",       ULONG),
##]

# From MSDN:
#
# typedef struct _PEB_LDR_DATA {
#   BYTE         Reserved1[8];
#   PVOID        Reserved2[3];
#   LIST_ENTRY   InMemoryOrderModuleList;
# } PEB_LDR_DATA,
#  *PPEB_LDR_DATA;
##class PEB_LDR_DATA(Structure):
##    _fields_ = [
##        ("Reserved1",               BYTE),
##        ("Reserved2",               PVOID),
##        ("InMemoryOrderModuleList", LIST_ENTRY),
##]

# From http://undocumented.ntinternals.net/UserMode/Structures/RTL_USER_PROCESS_PARAMETERS.html
# typedef struct _RTL_USER_PROCESS_PARAMETERS {
#   ULONG                   MaximumLength;
#   ULONG                   Length;
#   ULONG                   Flags;
#   ULONG                   DebugFlags;
#   PVOID                   ConsoleHandle;
#   ULONG                   ConsoleFlags;
#   HANDLE                  StdInputHandle;
#   HANDLE                  StdOutputHandle;
#   HANDLE                  StdErrorHandle;
#   UNICODE_STRING          CurrentDirectoryPath;
#   HANDLE                  CurrentDirectoryHandle;
#   UNICODE_STRING          DllPath;
#   UNICODE_STRING          ImagePathName;
#   UNICODE_STRING          CommandLine;
#   PVOID                   Environment;
#   ULONG                   StartingPositionLeft;
#   ULONG                   StartingPositionTop;
#   ULONG                   Width;
#   ULONG                   Height;
#   ULONG                   CharWidth;
#   ULONG                   CharHeight;
#   ULONG                   ConsoleTextAttributes;
#   ULONG                   WindowFlags;
#   ULONG                   ShowWindowFlags;
#   UNICODE_STRING          WindowTitle;
#   UNICODE_STRING          DesktopName;
#   UNICODE_STRING          ShellInfo;
#   UNICODE_STRING          RuntimeData;
#   RTL_DRIVE_LETTER_CURDIR DLCurrentDirectory[0x20];
# } RTL_USER_PROCESS_PARAMETERS, *PRTL_USER_PROCESS_PARAMETERS;

# kd> dt _RTL_USER_PROCESS_PARAMETERS
# ntdll!_RTL_USER_PROCESS_PARAMETERS
#    +0x000 MaximumLength    : Uint4B
#    +0x004 Length           : Uint4B
#    +0x008 Flags            : Uint4B
#    +0x00c DebugFlags       : Uint4B
#    +0x010 ConsoleHandle    : Ptr32 Void
#    +0x014 ConsoleFlags     : Uint4B
#    +0x018 StandardInput    : Ptr32 Void
#    +0x01c StandardOutput   : Ptr32 Void
#    +0x020 StandardError    : Ptr32 Void
#    +0x024 CurrentDirectory : _CURDIR
#    +0x030 DllPath          : _UNICODE_STRING
#    +0x038 ImagePathName    : _UNICODE_STRING
#    +0x040 CommandLine      : _UNICODE_STRING
#    +0x048 Environment      : Ptr32 Void
#    +0x04c StartingX        : Uint4B
#    +0x050 StartingY        : Uint4B
#    +0x054 CountX           : Uint4B
#    +0x058 CountY           : Uint4B
#    +0x05c CountCharsX      : Uint4B
#    +0x060 CountCharsY      : Uint4B
#    +0x064 FillAttribute    : Uint4B
#    +0x068 WindowFlags      : Uint4B
#    +0x06c ShowWindowFlags  : Uint4B
#    +0x070 WindowTitle      : _UNICODE_STRING
#    +0x078 DesktopInfo      : _UNICODE_STRING
#    +0x080 ShellInfo        : _UNICODE_STRING
#    +0x088 RuntimeData      : _UNICODE_STRING
#    +0x090 CurrentDirectores : [32] _RTL_DRIVE_LETTER_CURDIR
#    +0x290 EnvironmentSize  : Uint4B
##class RTL_USER_PROCESS_PARAMETERS(Structure):
##    _fields_ = [
##        ("MaximumLength",           ULONG),
##        ("Length",                  ULONG),
##        ("Flags",                   ULONG),
##        ("DebugFlags",              ULONG),
##        ("ConsoleHandle",           PVOID),
##        ("ConsoleFlags",            ULONG),
##        ("StandardInput",           HANDLE),
##        ("StandardOutput",          HANDLE),
##        ("StandardError",           HANDLE),
##        ("CurrentDirectory",        CURDIR),
##        ("DllPath",                 UNICODE_STRING),
##        ("ImagePathName",           UNICODE_STRING),
##        ("CommandLine",             UNICODE_STRING),
##        ("Environment",             PVOID),
##        ("StartingX",               ULONG),
##        ("StartingY",               ULONG),
##        ("CountX",                  ULONG),
##        ("CountY",                  ULONG),
##        ("CountCharsX",             ULONG),
##        ("CountCharsY",             ULONG),
##        ("FillAttribute",           ULONG),
##        ("WindowFlags",             ULONG),
##        ("ShowWindowFlags",         ULONG),
##        ("WindowTitle",             UNICODE_STRING),
##        ("DesktopInfo",             UNICODE_STRING),
##        ("ShellInfo",               UNICODE_STRING),
##        ("RuntimeData",             UNICODE_STRING),
##        ("CurrentDirectores",       RTL_DRIVE_LETTER_CURDIR * 32), # typo here?
##
##        # Windows 2008 and Vista
##        ("EnvironmentSize",         ULONG),
##]
##    @property
##    def CurrentDirectories(self):
##        return self.CurrentDirectores

# From MSDN:
#
# typedef struct _RTL_USER_PROCESS_PARAMETERS {
#   BYTE             Reserved1[16];
#   PVOID            Reserved2[10];
#   UNICODE_STRING   ImagePathName;
#   UNICODE_STRING   CommandLine;
# } RTL_USER_PROCESS_PARAMETERS,
#  *PRTL_USER_PROCESS_PARAMETERS;
class RTL_USER_PROCESS_PARAMETERS(Structure):
    _fields_ = [
        ("Reserved1",               BYTE * 16),
        ("Reserved2",               PVOID * 10),
        ("ImagePathName",           UNICODE_STRING),
        ("CommandLine",             UNICODE_STRING),
        ("Environment",             PVOID),             # undocumented!
        #
        # XXX TODO
        # This structure should be defined with all undocumented fields for
        # each version of Windows, just like it's being done for PEB and TEB.
        #
]

PPS_POST_PROCESS_INIT_ROUTINE = PVOID

#from MSDN:
#
# typedef struct _PEB {
#     BYTE Reserved1[2];
#     BYTE BeingDebugged;
#     BYTE Reserved2[21];
#     PPEB_LDR_DATA LoaderData;
#     PRTL_USER_PROCESS_PARAMETERS ProcessParameters;
#     BYTE Reserved3[520];
#     PPS_POST_PROCESS_INIT_ROUTINE PostProcessInitRoutine;
#     BYTE Reserved4[136];
#     ULONG SessionId;
# } PEB;
##class PEB(Structure):
##    _fields_ = [
##        ("Reserved1",               BYTE * 2),
##        ("BeingDebugged",           BYTE),
##        ("Reserved2",               BYTE * 21),
##        ("LoaderData",              PVOID,    # PPEB_LDR_DATA
##        ("ProcessParameters",       PVOID,    # PRTL_USER_PROCESS_PARAMETERS
##        ("Reserved3",               BYTE * 520),
##        ("PostProcessInitRoutine",  PPS_POST_PROCESS_INIT_ROUTINE),
##        ("Reserved4",               BYTE),
##        ("SessionId",               ULONG),
##]

# from MSDN:
#
# typedef struct _TEB {
#   BYTE    Reserved1[1952];
#   PVOID   Reserved2[412];
#   PVOID   TlsSlots[64];
#   BYTE    Reserved3[8];
#   PVOID   Reserved4[26];
#   PVOID   ReservedForOle;
#   PVOID   Reserved5[4];
#   PVOID   TlsExpansionSlots;
# } TEB,
#  *PTEB;
##class TEB(Structure):
##    _fields_ = [
##        ("Reserved1",           PVOID * 1952),
##        ("Reserved2",           PVOID * 412),
##        ("TlsSlots",            PVOID * 64),
##        ("Reserved3",           BYTE  * 8),
##        ("Reserved4",           PVOID * 26),
##        ("ReservedForOle",      PVOID),
##        ("Reserved5",           PVOID * 4),
##        ("TlsExpansionSlots",   PVOID),
##]

# from http://undocumented.ntinternals.net/UserMode/Structures/LDR_MODULE.html
#
# typedef struct _LDR_MODULE {
#   LIST_ENTRY InLoadOrderModuleList;
#   LIST_ENTRY InMemoryOrderModuleList;
#   LIST_ENTRY InInitializationOrderModuleList;
#   PVOID BaseAddress;
#   PVOID EntryPoint;
#   ULONG SizeOfImage;
#   UNICODE_STRING FullDllName;
#   UNICODE_STRING BaseDllName;
#   ULONG Flags;
#   SHORT LoadCount;
#   SHORT TlsIndex;
#   LIST_ENTRY HashTableEntry;
#   ULONG TimeDateStamp;
# } LDR_MODULE, *PLDR_MODULE;
class LDR_MODULE(Structure):
    _fields_ = [
        ("InLoadOrderModuleList",           LIST_ENTRY),
        ("InMemoryOrderModuleList",         LIST_ENTRY),
        ("InInitializationOrderModuleList", LIST_ENTRY),
        ("BaseAddress",                     PVOID),
        ("EntryPoint",                      PVOID),
        ("SizeOfImage",                     ULONG),
        ("FullDllName",                     UNICODE_STRING),
        ("BaseDllName",                     UNICODE_STRING),
        ("Flags",                           ULONG),
        ("LoadCount",                       SHORT),
        ("TlsIndex",                        SHORT),
        ("HashTableEntry",                  LIST_ENTRY),
        ("TimeDateStamp",                   ULONG),
]

# from http://undocumented.ntinternals.net/UserMode/Structures/PEB_LDR_DATA.html
#
# typedef struct _PEB_LDR_DATA {
#   ULONG Length;
#   BOOLEAN Initialized;
#   PVOID SsHandle;
#   LIST_ENTRY InLoadOrderModuleList;
#   LIST_ENTRY InMemoryOrderModuleList;
#   LIST_ENTRY InInitializationOrderModuleList;
# } PEB_LDR_DATA, *PPEB_LDR_DATA;
class PEB_LDR_DATA(Structure):
    _fields_ = [
        ("Length",                          ULONG),
        ("Initialized",                     BOOLEAN),
        ("SsHandle",                        PVOID),
        ("InLoadOrderModuleList",           LIST_ENTRY),
        ("InMemoryOrderModuleList",         LIST_ENTRY),
        ("InInitializationOrderModuleList", LIST_ENTRY),
]

# From http://undocumented.ntinternals.net/UserMode/Undocumented%20Functions/NT%20Objects/Process/PEB_FREE_BLOCK.html
#
# typedef struct _PEB_FREE_BLOCK {
#   PEB_FREE_BLOCK *Next;
#   ULONG Size;
# } PEB_FREE_BLOCK, *PPEB_FREE_BLOCK;
class PEB_FREE_BLOCK(Structure):
    pass

##PPEB_FREE_BLOCK = POINTER(PEB_FREE_BLOCK)
PPEB_FREE_BLOCK = PVOID

PEB_FREE_BLOCK._fields_ = [
        ("Next", PPEB_FREE_BLOCK),
        ("Size", ULONG),
]

# From http://undocumented.ntinternals.net/UserMode/Structures/RTL_DRIVE_LETTER_CURDIR.html
#
# typedef struct _RTL_DRIVE_LETTER_CURDIR {
#   USHORT Flags;
#   USHORT Length;
#   ULONG TimeStamp;
#   UNICODE_STRING DosPath;
# } RTL_DRIVE_LETTER_CURDIR, *PRTL_DRIVE_LETTER_CURDIR;
class RTL_DRIVE_LETTER_CURDIR(Structure):
    _fields_ = [
        ("Flags",       USHORT),
        ("Length",      USHORT),
        ("TimeStamp",   ULONG),
        ("DosPath",     UNICODE_STRING),
]

# From http://www.nirsoft.net/kernel_struct/vista/CURDIR.html
#
# typedef struct _CURDIR
# {
#      UNICODE_STRING DosPath;
#      PVOID Handle;
# } CURDIR, *PCURDIR;
class CURDIR(Structure):
    _fields_ = [
        ("DosPath", UNICODE_STRING),
        ("Handle",  PVOID),
]

# From http://www.nirsoft.net/kernel_struct/vista/RTL_CRITICAL_SECTION_DEBUG.html
#
# typedef struct _RTL_CRITICAL_SECTION_DEBUG
# {
#      WORD Type;
#      WORD CreatorBackTraceIndex;
#      PRTL_CRITICAL_SECTION CriticalSection;
#      LIST_ENTRY ProcessLocksList;
#      ULONG EntryCount;
#      ULONG ContentionCount;
#      ULONG Flags;
#      WORD CreatorBackTraceIndexHigh;
#      WORD SpareUSHORT;
# } RTL_CRITICAL_SECTION_DEBUG, *PRTL_CRITICAL_SECTION_DEBUG;
#
# From http://www.nirsoft.net/kernel_struct/vista/RTL_CRITICAL_SECTION.html
#
# typedef struct _RTL_CRITICAL_SECTION
# {
#      PRTL_CRITICAL_SECTION_DEBUG DebugInfo;
#      LONG LockCount;
#      LONG RecursionCount;
#      PVOID OwningThread;
#      PVOID LockSemaphore;
#      ULONG SpinCount;
# } RTL_CRITICAL_SECTION, *PRTL_CRITICAL_SECTION;
#
class RTL_CRITICAL_SECTION(Structure):
    _fields_ = [
        ("DebugInfo",       PVOID),     # PRTL_CRITICAL_SECTION_DEBUG
        ("LockCount",       LONG),
        ("RecursionCount",  LONG),
        ("OwningThread",    PVOID),
        ("LockSemaphore",   PVOID),
        ("SpinCount",       ULONG),
]
class RTL_CRITICAL_SECTION_DEBUG(Structure):
    _fields_ = [
        ("Type",                        WORD),
        ("CreatorBackTraceIndex",       WORD),
        ("CriticalSection",             PVOID),         # PRTL_CRITICAL_SECTION
        ("ProcessLocksList",            LIST_ENTRY),
        ("EntryCount",                  ULONG),
        ("ContentionCount",             ULONG),
        ("Flags",                       ULONG),
        ("CreatorBackTraceIndexHigh",   WORD),
        ("SpareUSHORT",                 WORD),
]
PRTL_CRITICAL_SECTION       = POINTER(RTL_CRITICAL_SECTION)
PRTL_CRITICAL_SECTION_DEBUG = POINTER(RTL_CRITICAL_SECTION_DEBUG)

PPEB_LDR_DATA                   = POINTER(PEB_LDR_DATA)
PRTL_USER_PROCESS_PARAMETERS    = POINTER(RTL_USER_PROCESS_PARAMETERS)

PPEBLOCKROUTINE                 = PVOID

# BitField
ImageUsesLargePages             = 1 << 0
IsProtectedProcess              = 1 << 1
IsLegacyProcess                 = 1 << 2
IsImageDynamicallyRelocated     = 1 << 3
SkipPatchingUser32Forwarders    = 1 << 4

# CrossProcessFlags
ProcessInJob                    = 1 << 0
ProcessInitializing             = 1 << 1
ProcessUsingVEH                 = 1 << 2
ProcessUsingVCH                 = 1 << 3
ProcessUsingFTH                 = 1 << 4

# TracingFlags
HeapTracingEnabled              = 1 << 0
CritSecTracingEnabled           = 1 << 1

# NtGlobalFlags
FLG_VALID_BITS                  = 0x003FFFFF    # not a flag
FLG_STOP_ON_EXCEPTION           = 0x00000001
FLG_SHOW_LDR_SNAPS              = 0x00000002
FLG_DEBUG_INITIAL_COMMAND       = 0x00000004
FLG_STOP_ON_HUNG_GUI            = 0x00000008
FLG_HEAP_ENABLE_TAIL_CHECK      = 0x00000010
FLG_HEAP_ENABLE_FREE_CHECK      = 0x00000020
FLG_HEAP_VALIDATE_PARAMETERS    = 0x00000040
FLG_HEAP_VALIDATE_ALL           = 0x00000080
FLG_POOL_ENABLE_TAIL_CHECK      = 0x00000100
FLG_POOL_ENABLE_FREE_CHECK      = 0x00000200
FLG_POOL_ENABLE_TAGGING         = 0x00000400
FLG_HEAP_ENABLE_TAGGING         = 0x00000800
FLG_USER_STACK_TRACE_DB         = 0x00001000
FLG_KERNEL_STACK_TRACE_DB       = 0x00002000
FLG_MAINTAIN_OBJECT_TYPELIST    = 0x00004000
FLG_HEAP_ENABLE_TAG_BY_DLL      = 0x00008000
FLG_IGNORE_DEBUG_PRIV           = 0x00010000
FLG_ENABLE_CSRDEBUG             = 0x00020000
FLG_ENABLE_KDEBUG_SYMBOL_LOAD   = 0x00040000
FLG_DISABLE_PAGE_KERNEL_STACKS  = 0x00080000
FLG_HEAP_ENABLE_CALL_TRACING    = 0x00100000
FLG_HEAP_DISABLE_COALESCING     = 0x00200000
FLG_ENABLE_CLOSE_EXCEPTION      = 0x00400000
FLG_ENABLE_EXCEPTION_LOGGING    = 0x00800000
FLG_ENABLE_HANDLE_TYPE_TAGGING  = 0x01000000
FLG_HEAP_PAGE_ALLOCS            = 0x02000000
FLG_DEBUG_WINLOGON              = 0x04000000
FLG_ENABLE_DBGPRINT_BUFFERING   = 0x08000000
FLG_EARLY_CRITICAL_SECTION_EVT  = 0x10000000
FLG_DISABLE_DLL_VERIFICATION    = 0x80000000

class _PEB_NT(Structure):
    _pack_   = 4
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID),
        ("FastPebLockRoutine",                  PVOID), # PPEBLOCKROUTINE
        ("FastPebUnlockRoutine",                PVOID), # PPEBLOCKROUTINE
        ("EnvironmentUpdateCount",              ULONG),
        ("KernelCallbackTable",                 PVOID), # Ptr32 Ptr32 Void
        ("EventLogSection",                     PVOID),
        ("EventLog",                            PVOID),
        ("FreeList",                            PVOID), # PPEB_FREE_BLOCK
        ("TlsExpansionCounter",                 ULONG),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       ULONG * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("ReadOnlySharedMemoryHeap",            PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr32 Ptr32 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  ULONG),
        ("NtGlobalFlag",                        ULONG),
        ("Spare2",                              BYTE * 4),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  ULONG),
        ("HeapSegmentCommit",                   ULONG),
        ("HeapDeCommitTotalFreeThreshold",      ULONG),
        ("HeapDeCommitFreeBlockThreshold",      ULONG),
        ("NumberOfHeaps",                       ULONG),
        ("MaximumNumberOfHeaps",                ULONG),
        ("ProcessHeaps",                        PVOID), # Ptr32 Ptr32 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  PVOID),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      ULONG),
        ("OSMinorVersion",                      ULONG),
        ("OSBuildNumber",                       ULONG),
        ("OSPlatformId",                        ULONG),
        ("ImageSubSystem",                      ULONG),
        ("ImageSubSystemMajorVersion",          ULONG),
        ("ImageSubSystemMinorVersion",          ULONG),
        ("ImageProcessAffinityMask",            ULONG),
        ("GdiHandleBuffer",                     ULONG * 34),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  ULONG),
        ("TlsExpansionBitmapBits",              BYTE * 128),
        ("SessionId",                           ULONG),
    ]

# not really, but "dt _PEB" in w2k isn't working for me :(
_PEB_2000 = _PEB_NT

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 SpareBool        : UChar
#    +0x004 Mutant           : Ptr32 Void
#    +0x008 ImageBaseAddress : Ptr32 Void
#    +0x00c Ldr              : Ptr32 _PEB_LDR_DATA
#    +0x010 ProcessParameters : Ptr32 _RTL_USER_PROCESS_PARAMETERS
#    +0x014 SubSystemData    : Ptr32 Void
#    +0x018 ProcessHeap      : Ptr32 Void
#    +0x01c FastPebLock      : Ptr32 _RTL_CRITICAL_SECTION
#    +0x020 FastPebLockRoutine : Ptr32 Void
#    +0x024 FastPebUnlockRoutine : Ptr32 Void
#    +0x028 EnvironmentUpdateCount : Uint4B
#    +0x02c KernelCallbackTable : Ptr32 Void
#    +0x030 SystemReserved   : [1] Uint4B
#    +0x034 AtlThunkSListPtr32 : Uint4B
#    +0x038 FreeList         : Ptr32 _PEB_FREE_BLOCK
#    +0x03c TlsExpansionCounter : Uint4B
#    +0x040 TlsBitmap        : Ptr32 Void
#    +0x044 TlsBitmapBits    : [2] Uint4B
#    +0x04c ReadOnlySharedMemoryBase : Ptr32 Void
#    +0x050 ReadOnlySharedMemoryHeap : Ptr32 Void
#    +0x054 ReadOnlyStaticServerData : Ptr32 Ptr32 Void
#    +0x058 AnsiCodePageData : Ptr32 Void
#    +0x05c OemCodePageData  : Ptr32 Void
#    +0x060 UnicodeCaseTableData : Ptr32 Void
#    +0x064 NumberOfProcessors : Uint4B
#    +0x068 NtGlobalFlag     : Uint4B
#    +0x070 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x078 HeapSegmentReserve : Uint4B
#    +0x07c HeapSegmentCommit : Uint4B
#    +0x080 HeapDeCommitTotalFreeThreshold : Uint4B
#    +0x084 HeapDeCommitFreeBlockThreshold : Uint4B
#    +0x088 NumberOfHeaps    : Uint4B
#    +0x08c MaximumNumberOfHeaps : Uint4B
#    +0x090 ProcessHeaps     : Ptr32 Ptr32 Void
#    +0x094 GdiSharedHandleTable : Ptr32 Void
#    +0x098 ProcessStarterHelper : Ptr32 Void
#    +0x09c GdiDCAttributeList : Uint4B
#    +0x0a0 LoaderLock       : Ptr32 Void
#    +0x0a4 OSMajorVersion   : Uint4B
#    +0x0a8 OSMinorVersion   : Uint4B
#    +0x0ac OSBuildNumber    : Uint2B
#    +0x0ae OSCSDVersion     : Uint2B
#    +0x0b0 OSPlatformId     : Uint4B
#    +0x0b4 ImageSubsystem   : Uint4B
#    +0x0b8 ImageSubsystemMajorVersion : Uint4B
#    +0x0bc ImageSubsystemMinorVersion : Uint4B
#    +0x0c0 ImageProcessAffinityMask : Uint4B
#    +0x0c4 GdiHandleBuffer  : [34] Uint4B
#    +0x14c PostProcessInitRoutine : Ptr32     void
#    +0x150 TlsExpansionBitmap : Ptr32 Void
#    +0x154 TlsExpansionBitmapBits : [32] Uint4B
#    +0x1d4 SessionId        : Uint4B
#    +0x1d8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x1e0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x1e8 pShimData        : Ptr32 Void
#    +0x1ec AppCompatInfo    : Ptr32 Void
#    +0x1f0 CSDVersion       : _UNICODE_STRING
#    +0x1f8 ActivationContextData : Ptr32 Void
#    +0x1fc ProcessAssemblyStorageMap : Ptr32 Void
#    +0x200 SystemDefaultActivationContextData : Ptr32 Void
#    +0x204 SystemAssemblyStorageMap : Ptr32 Void
#    +0x208 MinimumStackCommit : Uint4B
class _PEB_XP(Structure):
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("SpareBool",                           UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID),
        ("FastPebLockRoutine",                  PVOID),
        ("FastPebUnlockRoutine",                PVOID),
        ("EnvironmentUpdateCount",              DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("AtlThunkSListPtr32",                  DWORD),
        ("FreeList",                            PVOID), # PPEB_FREE_BLOCK
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("ReadOnlySharedMemoryHeap",            PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr32 Ptr32 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  DWORD),
        ("HeapSegmentCommit",                   DWORD),
        ("HeapDeCommitTotalFreeThreshold",      DWORD),
        ("HeapDeCommitFreeBlockThreshold",      DWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr32 Ptr32 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ImageProcessAffinityMask",            DWORD),
        ("GdiHandleBuffer",                     DWORD * 34),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  DWORD),
    ]

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 BitField         : UChar
#    +0x003 ImageUsesLargePages : Pos 0, 1 Bit
#    +0x003 SpareBits        : Pos 1, 7 Bits
#    +0x008 Mutant           : Ptr64 Void
#    +0x010 ImageBaseAddress : Ptr64 Void
#    +0x018 Ldr              : Ptr64 _PEB_LDR_DATA
#    +0x020 ProcessParameters : Ptr64 _RTL_USER_PROCESS_PARAMETERS
#    +0x028 SubSystemData    : Ptr64 Void
#    +0x030 ProcessHeap      : Ptr64 Void
#    +0x038 FastPebLock      : Ptr64 _RTL_CRITICAL_SECTION
#    +0x040 AtlThunkSListPtr : Ptr64 Void
#    +0x048 SparePtr2        : Ptr64 Void
#    +0x050 EnvironmentUpdateCount : Uint4B
#    +0x058 KernelCallbackTable : Ptr64 Void
#    +0x060 SystemReserved   : [1] Uint4B
#    +0x064 SpareUlong       : Uint4B
#    +0x068 FreeList         : Ptr64 _PEB_FREE_BLOCK
#    +0x070 TlsExpansionCounter : Uint4B
#    +0x078 TlsBitmap        : Ptr64 Void
#    +0x080 TlsBitmapBits    : [2] Uint4B
#    +0x088 ReadOnlySharedMemoryBase : Ptr64 Void
#    +0x090 ReadOnlySharedMemoryHeap : Ptr64 Void
#    +0x098 ReadOnlyStaticServerData : Ptr64 Ptr64 Void
#    +0x0a0 AnsiCodePageData : Ptr64 Void
#    +0x0a8 OemCodePageData  : Ptr64 Void
#    +0x0b0 UnicodeCaseTableData : Ptr64 Void
#    +0x0b8 NumberOfProcessors : Uint4B
#    +0x0bc NtGlobalFlag     : Uint4B
#    +0x0c0 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x0c8 HeapSegmentReserve : Uint8B
#    +0x0d0 HeapSegmentCommit : Uint8B
#    +0x0d8 HeapDeCommitTotalFreeThreshold : Uint8B
#    +0x0e0 HeapDeCommitFreeBlockThreshold : Uint8B
#    +0x0e8 NumberOfHeaps    : Uint4B
#    +0x0ec MaximumNumberOfHeaps : Uint4B
#    +0x0f0 ProcessHeaps     : Ptr64 Ptr64 Void
#    +0x0f8 GdiSharedHandleTable : Ptr64 Void
#    +0x100 ProcessStarterHelper : Ptr64 Void
#    +0x108 GdiDCAttributeList : Uint4B
#    +0x110 LoaderLock       : Ptr64 _RTL_CRITICAL_SECTION
#    +0x118 OSMajorVersion   : Uint4B
#    +0x11c OSMinorVersion   : Uint4B
#    +0x120 OSBuildNumber    : Uint2B
#    +0x122 OSCSDVersion     : Uint2B
#    +0x124 OSPlatformId     : Uint4B
#    +0x128 ImageSubsystem   : Uint4B
#    +0x12c ImageSubsystemMajorVersion : Uint4B
#    +0x130 ImageSubsystemMinorVersion : Uint4B
#    +0x138 ImageProcessAffinityMask : Uint8B
#    +0x140 GdiHandleBuffer  : [60] Uint4B
#    +0x230 PostProcessInitRoutine : Ptr64     void
#    +0x238 TlsExpansionBitmap : Ptr64 Void
#    +0x240 TlsExpansionBitmapBits : [32] Uint4B
#    +0x2c0 SessionId        : Uint4B
#    +0x2c8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x2d0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x2d8 pShimData        : Ptr64 Void
#    +0x2e0 AppCompatInfo    : Ptr64 Void
#    +0x2e8 CSDVersion       : _UNICODE_STRING
#    +0x2f8 ActivationContextData : Ptr64 _ACTIVATION_CONTEXT_DATA
#    +0x300 ProcessAssemblyStorageMap : Ptr64 _ASSEMBLY_STORAGE_MAP
#    +0x308 SystemDefaultActivationContextData : Ptr64 _ACTIVATION_CONTEXT_DATA
#    +0x310 SystemAssemblyStorageMap : Ptr64 _ASSEMBLY_STORAGE_MAP
#    +0x318 MinimumStackCommit : Uint8B
#    +0x320 FlsCallback      : Ptr64 Ptr64 Void
#    +0x328 FlsListHead      : _LIST_ENTRY
#    +0x338 FlsBitmap        : Ptr64 Void
#    +0x340 FlsBitmapBits    : [4] Uint4B
#    +0x350 FlsHighIndex     : Uint4B
class _PEB_XP_64(Structure):
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID), # PRTL_CRITICAL_SECTION
        ("AtlThunkSListPtr",                    PVOID),
        ("SparePtr2",                           PVOID),
        ("EnvironmentUpdateCount",              DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("SpareUlong",                          DWORD),
        ("FreeList",                            PVOID), # PPEB_FREE_BLOCK
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("ReadOnlySharedMemoryHeap",            PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr64 Ptr64 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  QWORD),
        ("HeapSegmentCommit",                   QWORD),
        ("HeapDeCommitTotalFreeThreshold",      QWORD),
        ("HeapDeCommitFreeBlockThreshold",      QWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr64 Ptr64 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ImageProcessAffinityMask",            QWORD),
        ("GdiHandleBuffer",                     DWORD * 60),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  QWORD),
        ("FlsCallback",                         PVOID), # Ptr64 Ptr64 Void
        ("FlsListHead",                         LIST_ENTRY),
        ("FlsBitmap",                           PVOID),
        ("FlsBitmapBits",                       DWORD * 4),
        ("FlsHighIndex",                        DWORD),
    ]

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 BitField         : UChar
#    +0x003 ImageUsesLargePages : Pos 0, 1 Bit
#    +0x003 SpareBits        : Pos 1, 7 Bits
#    +0x004 Mutant           : Ptr32 Void
#    +0x008 ImageBaseAddress : Ptr32 Void
#    +0x00c Ldr              : Ptr32 _PEB_LDR_DATA
#    +0x010 ProcessParameters : Ptr32 _RTL_USER_PROCESS_PARAMETERS
#    +0x014 SubSystemData    : Ptr32 Void
#    +0x018 ProcessHeap      : Ptr32 Void
#    +0x01c FastPebLock      : Ptr32 _RTL_CRITICAL_SECTION
#    +0x020 AtlThunkSListPtr : Ptr32 Void
#    +0x024 SparePtr2        : Ptr32 Void
#    +0x028 EnvironmentUpdateCount : Uint4B
#    +0x02c KernelCallbackTable : Ptr32 Void
#    +0x030 SystemReserved   : [1] Uint4B
#    +0x034 SpareUlong       : Uint4B
#    +0x038 FreeList         : Ptr32 _PEB_FREE_BLOCK
#    +0x03c TlsExpansionCounter : Uint4B
#    +0x040 TlsBitmap        : Ptr32 Void
#    +0x044 TlsBitmapBits    : [2] Uint4B
#    +0x04c ReadOnlySharedMemoryBase : Ptr32 Void
#    +0x050 ReadOnlySharedMemoryHeap : Ptr32 Void
#    +0x054 ReadOnlyStaticServerData : Ptr32 Ptr32 Void
#    +0x058 AnsiCodePageData : Ptr32 Void
#    +0x05c OemCodePageData  : Ptr32 Void
#    +0x060 UnicodeCaseTableData : Ptr32 Void
#    +0x064 NumberOfProcessors : Uint4B
#    +0x068 NtGlobalFlag     : Uint4B
#    +0x070 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x078 HeapSegmentReserve : Uint4B
#    +0x07c HeapSegmentCommit : Uint4B
#    +0x080 HeapDeCommitTotalFreeThreshold : Uint4B
#    +0x084 HeapDeCommitFreeBlockThreshold : Uint4B
#    +0x088 NumberOfHeaps    : Uint4B
#    +0x08c MaximumNumberOfHeaps : Uint4B
#    +0x090 ProcessHeaps     : Ptr32 Ptr32 Void
#    +0x094 GdiSharedHandleTable : Ptr32 Void
#    +0x098 ProcessStarterHelper : Ptr32 Void
#    +0x09c GdiDCAttributeList : Uint4B
#    +0x0a0 LoaderLock       : Ptr32 _RTL_CRITICAL_SECTION
#    +0x0a4 OSMajorVersion   : Uint4B
#    +0x0a8 OSMinorVersion   : Uint4B
#    +0x0ac OSBuildNumber    : Uint2B
#    +0x0ae OSCSDVersion     : Uint2B
#    +0x0b0 OSPlatformId     : Uint4B
#    +0x0b4 ImageSubsystem   : Uint4B
#    +0x0b8 ImageSubsystemMajorVersion : Uint4B
#    +0x0bc ImageSubsystemMinorVersion : Uint4B
#    +0x0c0 ImageProcessAffinityMask : Uint4B
#    +0x0c4 GdiHandleBuffer  : [34] Uint4B
#    +0x14c PostProcessInitRoutine : Ptr32     void
#    +0x150 TlsExpansionBitmap : Ptr32 Void
#    +0x154 TlsExpansionBitmapBits : [32] Uint4B
#    +0x1d4 SessionId        : Uint4B
#    +0x1d8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x1e0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x1e8 pShimData        : Ptr32 Void
#    +0x1ec AppCompatInfo    : Ptr32 Void
#    +0x1f0 CSDVersion       : _UNICODE_STRING
#    +0x1f8 ActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x1fc ProcessAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x200 SystemDefaultActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x204 SystemAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x208 MinimumStackCommit : Uint4B
#    +0x20c FlsCallback      : Ptr32 Ptr32 Void
#    +0x210 FlsListHead      : _LIST_ENTRY
#    +0x218 FlsBitmap        : Ptr32 Void
#    +0x21c FlsBitmapBits    : [4] Uint4B
#    +0x22c FlsHighIndex     : Uint4B
class _PEB_2003(Structure):
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID), # PRTL_CRITICAL_SECTION
        ("AtlThunkSListPtr",                    PVOID),
        ("SparePtr2",                           PVOID),
        ("EnvironmentUpdateCount",              DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("SpareUlong",                          DWORD),
        ("FreeList",                            PVOID), # PPEB_FREE_BLOCK
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("ReadOnlySharedMemoryHeap",            PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr32 Ptr32 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  DWORD),
        ("HeapSegmentCommit",                   DWORD),
        ("HeapDeCommitTotalFreeThreshold",      DWORD),
        ("HeapDeCommitFreeBlockThreshold",      DWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr32 Ptr32 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ImageProcessAffinityMask",            DWORD),
        ("GdiHandleBuffer",                     DWORD * 34),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  QWORD),
        ("FlsCallback",                         PVOID), # Ptr32 Ptr32 Void
        ("FlsListHead",                         LIST_ENTRY),
        ("FlsBitmap",                           PVOID),
        ("FlsBitmapBits",                       DWORD * 4),
        ("FlsHighIndex",                        DWORD),
    ]

_PEB_2003_64    = _PEB_XP_64
_PEB_2003_R2    = _PEB_2003
_PEB_2003_R2_64 = _PEB_2003_64

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 BitField         : UChar
#    +0x003 ImageUsesLargePages : Pos 0, 1 Bit
#    +0x003 IsProtectedProcess : Pos 1, 1 Bit
#    +0x003 IsLegacyProcess  : Pos 2, 1 Bit
#    +0x003 IsImageDynamicallyRelocated : Pos 3, 1 Bit
#    +0x003 SkipPatchingUser32Forwarders : Pos 4, 1 Bit
#    +0x003 SpareBits        : Pos 5, 3 Bits
#    +0x004 Mutant           : Ptr32 Void
#    +0x008 ImageBaseAddress : Ptr32 Void
#    +0x00c Ldr              : Ptr32 _PEB_LDR_DATA
#    +0x010 ProcessParameters : Ptr32 _RTL_USER_PROCESS_PARAMETERS
#    +0x014 SubSystemData    : Ptr32 Void
#    +0x018 ProcessHeap      : Ptr32 Void
#    +0x01c FastPebLock      : Ptr32 _RTL_CRITICAL_SECTION
#    +0x020 AtlThunkSListPtr : Ptr32 Void
#    +0x024 IFEOKey          : Ptr32 Void
#    +0x028 CrossProcessFlags : Uint4B
#    +0x028 ProcessInJob     : Pos 0, 1 Bit
#    +0x028 ProcessInitializing : Pos 1, 1 Bit
#    +0x028 ProcessUsingVEH  : Pos 2, 1 Bit
#    +0x028 ProcessUsingVCH  : Pos 3, 1 Bit
#    +0x028 ReservedBits0    : Pos 4, 28 Bits
#    +0x02c KernelCallbackTable : Ptr32 Void
#    +0x02c UserSharedInfoPtr : Ptr32 Void
#    +0x030 SystemReserved   : [1] Uint4B
#    +0x034 SpareUlong       : Uint4B
#    +0x038 SparePebPtr0     : Uint4B
#    +0x03c TlsExpansionCounter : Uint4B
#    +0x040 TlsBitmap        : Ptr32 Void
#    +0x044 TlsBitmapBits    : [2] Uint4B
#    +0x04c ReadOnlySharedMemoryBase : Ptr32 Void
#    +0x050 HotpatchInformation : Ptr32 Void
#    +0x054 ReadOnlyStaticServerData : Ptr32 Ptr32 Void
#    +0x058 AnsiCodePageData : Ptr32 Void
#    +0x05c OemCodePageData  : Ptr32 Void
#    +0x060 UnicodeCaseTableData : Ptr32 Void
#    +0x064 NumberOfProcessors : Uint4B
#    +0x068 NtGlobalFlag     : Uint4B
#    +0x070 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x078 HeapSegmentReserve : Uint4B
#    +0x07c HeapSegmentCommit : Uint4B
#    +0x080 HeapDeCommitTotalFreeThreshold : Uint4B
#    +0x084 HeapDeCommitFreeBlockThreshold : Uint4B
#    +0x088 NumberOfHeaps    : Uint4B
#    +0x08c MaximumNumberOfHeaps : Uint4B
#    +0x090 ProcessHeaps     : Ptr32 Ptr32 Void
#    +0x094 GdiSharedHandleTable : Ptr32 Void
#    +0x098 ProcessStarterHelper : Ptr32 Void
#    +0x09c GdiDCAttributeList : Uint4B
#    +0x0a0 LoaderLock       : Ptr32 _RTL_CRITICAL_SECTION
#    +0x0a4 OSMajorVersion   : Uint4B
#    +0x0a8 OSMinorVersion   : Uint4B
#    +0x0ac OSBuildNumber    : Uint2B
#    +0x0ae OSCSDVersion     : Uint2B
#    +0x0b0 OSPlatformId     : Uint4B
#    +0x0b4 ImageSubsystem   : Uint4B
#    +0x0b8 ImageSubsystemMajorVersion : Uint4B
#    +0x0bc ImageSubsystemMinorVersion : Uint4B
#    +0x0c0 ActiveProcessAffinityMask : Uint4B
#    +0x0c4 GdiHandleBuffer  : [34] Uint4B
#    +0x14c PostProcessInitRoutine : Ptr32     void
#    +0x150 TlsExpansionBitmap : Ptr32 Void
#    +0x154 TlsExpansionBitmapBits : [32] Uint4B
#    +0x1d4 SessionId        : Uint4B
#    +0x1d8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x1e0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x1e8 pShimData        : Ptr32 Void
#    +0x1ec AppCompatInfo    : Ptr32 Void
#    +0x1f0 CSDVersion       : _UNICODE_STRING
#    +0x1f8 ActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x1fc ProcessAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x200 SystemDefaultActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x204 SystemAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x208 MinimumStackCommit : Uint4B
#    +0x20c FlsCallback      : Ptr32 _FLS_CALLBACK_INFO
#    +0x210 FlsListHead      : _LIST_ENTRY
#    +0x218 FlsBitmap        : Ptr32 Void
#    +0x21c FlsBitmapBits    : [4] Uint4B
#    +0x22c FlsHighIndex     : Uint4B
#    +0x230 WerRegistrationData : Ptr32 Void
#    +0x234 WerShipAssertPtr : Ptr32 Void
class _PEB_2008(Structure):
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID), # PRTL_CRITICAL_SECTION
        ("AtlThunkSListPtr",                    PVOID),
        ("IFEOKey",                             PVOID),
        ("CrossProcessFlags",                   DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("SpareUlong",                          DWORD),
        ("SparePebPtr0",                        PVOID),
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("HotpatchInformation",                 PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr32 Ptr32 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  DWORD),
        ("HeapSegmentCommit",                   DWORD),
        ("HeapDeCommitTotalFreeThreshold",      DWORD),
        ("HeapDeCommitFreeBlockThreshold",      DWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr32 Ptr32 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ActiveProcessAffinityMask",           DWORD),
        ("GdiHandleBuffer",                     DWORD * 34),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  DWORD),
        ("FlsCallback",                         PVOID), # PFLS_CALLBACK_INFO
        ("FlsListHead",                         LIST_ENTRY),
        ("FlsBitmap",                           PVOID),
        ("FlsBitmapBits",                       DWORD * 4),
        ("FlsHighIndex",                        DWORD),
        ("WerRegistrationData",                 PVOID),
        ("WerShipAssertPtr",                    PVOID),
    ]
    def __get_UserSharedInfoPtr(self):
        return self.KernelCallbackTable
    def __set_UserSharedInfoPtr(self, value):
        self.KernelCallbackTable = value
    UserSharedInfoPtr = property(__get_UserSharedInfoPtr, __set_UserSharedInfoPtr)

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 BitField         : UChar
#    +0x003 ImageUsesLargePages : Pos 0, 1 Bit
#    +0x003 IsProtectedProcess : Pos 1, 1 Bit
#    +0x003 IsLegacyProcess  : Pos 2, 1 Bit
#    +0x003 IsImageDynamicallyRelocated : Pos 3, 1 Bit
#    +0x003 SkipPatchingUser32Forwarders : Pos 4, 1 Bit
#    +0x003 SpareBits        : Pos 5, 3 Bits
#    +0x008 Mutant           : Ptr64 Void
#    +0x010 ImageBaseAddress : Ptr64 Void
#    +0x018 Ldr              : Ptr64 _PEB_LDR_DATA
#    +0x020 ProcessParameters : Ptr64 _RTL_USER_PROCESS_PARAMETERS
#    +0x028 SubSystemData    : Ptr64 Void
#    +0x030 ProcessHeap      : Ptr64 Void
#    +0x038 FastPebLock      : Ptr64 _RTL_CRITICAL_SECTION
#    +0x040 AtlThunkSListPtr : Ptr64 Void
#    +0x048 IFEOKey          : Ptr64 Void
#    +0x050 CrossProcessFlags : Uint4B
#    +0x050 ProcessInJob     : Pos 0, 1 Bit
#    +0x050 ProcessInitializing : Pos 1, 1 Bit
#    +0x050 ProcessUsingVEH  : Pos 2, 1 Bit
#    +0x050 ProcessUsingVCH  : Pos 3, 1 Bit
#    +0x050 ReservedBits0    : Pos 4, 28 Bits
#    +0x058 KernelCallbackTable : Ptr64 Void
#    +0x058 UserSharedInfoPtr : Ptr64 Void
#    +0x060 SystemReserved   : [1] Uint4B
#    +0x064 SpareUlong       : Uint4B
#    +0x068 SparePebPtr0     : Uint8B
#    +0x070 TlsExpansionCounter : Uint4B
#    +0x078 TlsBitmap        : Ptr64 Void
#    +0x080 TlsBitmapBits    : [2] Uint4B
#    +0x088 ReadOnlySharedMemoryBase : Ptr64 Void
#    +0x090 HotpatchInformation : Ptr64 Void
#    +0x098 ReadOnlyStaticServerData : Ptr64 Ptr64 Void
#    +0x0a0 AnsiCodePageData : Ptr64 Void
#    +0x0a8 OemCodePageData  : Ptr64 Void
#    +0x0b0 UnicodeCaseTableData : Ptr64 Void
#    +0x0b8 NumberOfProcessors : Uint4B
#    +0x0bc NtGlobalFlag     : Uint4B
#    +0x0c0 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x0c8 HeapSegmentReserve : Uint8B
#    +0x0d0 HeapSegmentCommit : Uint8B
#    +0x0d8 HeapDeCommitTotalFreeThreshold : Uint8B
#    +0x0e0 HeapDeCommitFreeBlockThreshold : Uint8B
#    +0x0e8 NumberOfHeaps    : Uint4B
#    +0x0ec MaximumNumberOfHeaps : Uint4B
#    +0x0f0 ProcessHeaps     : Ptr64 Ptr64 Void
#    +0x0f8 GdiSharedHandleTable : Ptr64 Void
#    +0x100 ProcessStarterHelper : Ptr64 Void
#    +0x108 GdiDCAttributeList : Uint4B
#    +0x110 LoaderLock       : Ptr64 _RTL_CRITICAL_SECTION
#    +0x118 OSMajorVersion   : Uint4B
#    +0x11c OSMinorVersion   : Uint4B
#    +0x120 OSBuildNumber    : Uint2B
#    +0x122 OSCSDVersion     : Uint2B
#    +0x124 OSPlatformId     : Uint4B
#    +0x128 ImageSubsystem   : Uint4B
#    +0x12c ImageSubsystemMajorVersion : Uint4B
#    +0x130 ImageSubsystemMinorVersion : Uint4B
#    +0x138 ActiveProcessAffinityMask : Uint8B
#    +0x140 GdiHandleBuffer  : [60] Uint4B
#    +0x230 PostProcessInitRoutine : Ptr64     void
#    +0x238 TlsExpansionBitmap : Ptr64 Void
#    +0x240 TlsExpansionBitmapBits : [32] Uint4B
#    +0x2c0 SessionId        : Uint4B
#    +0x2c8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x2d0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x2d8 pShimData        : Ptr64 Void
#    +0x2e0 AppCompatInfo    : Ptr64 Void
#    +0x2e8 CSDVersion       : _UNICODE_STRING
#    +0x2f8 ActivationContextData : Ptr64 _ACTIVATION_CONTEXT_DATA
#    +0x300 ProcessAssemblyStorageMap : Ptr64 _ASSEMBLY_STORAGE_MAP
#    +0x308 SystemDefaultActivationContextData : Ptr64 _ACTIVATION_CONTEXT_DATA
#    +0x310 SystemAssemblyStorageMap : Ptr64 _ASSEMBLY_STORAGE_MAP
#    +0x318 MinimumStackCommit : Uint8B
#    +0x320 FlsCallback      : Ptr64 _FLS_CALLBACK_INFO
#    +0x328 FlsListHead      : _LIST_ENTRY
#    +0x338 FlsBitmap        : Ptr64 Void
#    +0x340 FlsBitmapBits    : [4] Uint4B
#    +0x350 FlsHighIndex     : Uint4B
#    +0x358 WerRegistrationData : Ptr64 Void
#    +0x360 WerShipAssertPtr : Ptr64 Void
class _PEB_2008_64(Structure):
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID), # PRTL_CRITICAL_SECTION
        ("AtlThunkSListPtr",                    PVOID),
        ("IFEOKey",                             PVOID),
        ("CrossProcessFlags",                   DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("SpareUlong",                          DWORD),
        ("SparePebPtr0",                        PVOID),
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("HotpatchInformation",                 PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr64 Ptr64 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  QWORD),
        ("HeapSegmentCommit",                   QWORD),
        ("HeapDeCommitTotalFreeThreshold",      QWORD),
        ("HeapDeCommitFreeBlockThreshold",      QWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr64 Ptr64 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ActiveProcessAffinityMask",           QWORD),
        ("GdiHandleBuffer",                     DWORD * 60),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  QWORD),
        ("FlsCallback",                         PVOID), # PFLS_CALLBACK_INFO
        ("FlsListHead",                         LIST_ENTRY),
        ("FlsBitmap",                           PVOID),
        ("FlsBitmapBits",                       DWORD * 4),
        ("FlsHighIndex",                        DWORD),
        ("WerRegistrationData",                 PVOID),
        ("WerShipAssertPtr",                    PVOID),
    ]
    def __get_UserSharedInfoPtr(self):
        return self.KernelCallbackTable
    def __set_UserSharedInfoPtr(self, value):
        self.KernelCallbackTable = value
    UserSharedInfoPtr = property(__get_UserSharedInfoPtr, __set_UserSharedInfoPtr)

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 BitField         : UChar
#    +0x003 ImageUsesLargePages : Pos 0, 1 Bit
#    +0x003 IsProtectedProcess : Pos 1, 1 Bit
#    +0x003 IsLegacyProcess  : Pos 2, 1 Bit
#    +0x003 IsImageDynamicallyRelocated : Pos 3, 1 Bit
#    +0x003 SkipPatchingUser32Forwarders : Pos 4, 1 Bit
#    +0x003 SpareBits        : Pos 5, 3 Bits
#    +0x004 Mutant           : Ptr32 Void
#    +0x008 ImageBaseAddress : Ptr32 Void
#    +0x00c Ldr              : Ptr32 _PEB_LDR_DATA
#    +0x010 ProcessParameters : Ptr32 _RTL_USER_PROCESS_PARAMETERS
#    +0x014 SubSystemData    : Ptr32 Void
#    +0x018 ProcessHeap      : Ptr32 Void
#    +0x01c FastPebLock      : Ptr32 _RTL_CRITICAL_SECTION
#    +0x020 AtlThunkSListPtr : Ptr32 Void
#    +0x024 IFEOKey          : Ptr32 Void
#    +0x028 CrossProcessFlags : Uint4B
#    +0x028 ProcessInJob     : Pos 0, 1 Bit
#    +0x028 ProcessInitializing : Pos 1, 1 Bit
#    +0x028 ProcessUsingVEH  : Pos 2, 1 Bit
#    +0x028 ProcessUsingVCH  : Pos 3, 1 Bit
#    +0x028 ProcessUsingFTH  : Pos 4, 1 Bit
#    +0x028 ReservedBits0    : Pos 5, 27 Bits
#    +0x02c KernelCallbackTable : Ptr32 Void
#    +0x02c UserSharedInfoPtr : Ptr32 Void
#    +0x030 SystemReserved   : [1] Uint4B
#    +0x034 AtlThunkSListPtr32 : Uint4B
#    +0x038 ApiSetMap        : Ptr32 Void
#    +0x03c TlsExpansionCounter : Uint4B
#    +0x040 TlsBitmap        : Ptr32 Void
#    +0x044 TlsBitmapBits    : [2] Uint4B
#    +0x04c ReadOnlySharedMemoryBase : Ptr32 Void
#    +0x050 HotpatchInformation : Ptr32 Void
#    +0x054 ReadOnlyStaticServerData : Ptr32 Ptr32 Void
#    +0x058 AnsiCodePageData : Ptr32 Void
#    +0x05c OemCodePageData  : Ptr32 Void
#    +0x060 UnicodeCaseTableData : Ptr32 Void
#    +0x064 NumberOfProcessors : Uint4B
#    +0x068 NtGlobalFlag     : Uint4B
#    +0x070 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x078 HeapSegmentReserve : Uint4B
#    +0x07c HeapSegmentCommit : Uint4B
#    +0x080 HeapDeCommitTotalFreeThreshold : Uint4B
#    +0x084 HeapDeCommitFreeBlockThreshold : Uint4B
#    +0x088 NumberOfHeaps    : Uint4B
#    +0x08c MaximumNumberOfHeaps : Uint4B
#    +0x090 ProcessHeaps     : Ptr32 Ptr32 Void
#    +0x094 GdiSharedHandleTable : Ptr32 Void
#    +0x098 ProcessStarterHelper : Ptr32 Void
#    +0x09c GdiDCAttributeList : Uint4B
#    +0x0a0 LoaderLock       : Ptr32 _RTL_CRITICAL_SECTION
#    +0x0a4 OSMajorVersion   : Uint4B
#    +0x0a8 OSMinorVersion   : Uint4B
#    +0x0ac OSBuildNumber    : Uint2B
#    +0x0ae OSCSDVersion     : Uint2B
#    +0x0b0 OSPlatformId     : Uint4B
#    +0x0b4 ImageSubsystem   : Uint4B
#    +0x0b8 ImageSubsystemMajorVersion : Uint4B
#    +0x0bc ImageSubsystemMinorVersion : Uint4B
#    +0x0c0 ActiveProcessAffinityMask : Uint4B
#    +0x0c4 GdiHandleBuffer  : [34] Uint4B
#    +0x14c PostProcessInitRoutine : Ptr32     void
#    +0x150 TlsExpansionBitmap : Ptr32 Void
#    +0x154 TlsExpansionBitmapBits : [32] Uint4B
#    +0x1d4 SessionId        : Uint4B
#    +0x1d8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x1e0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x1e8 pShimData        : Ptr32 Void
#    +0x1ec AppCompatInfo    : Ptr32 Void
#    +0x1f0 CSDVersion       : _UNICODE_STRING
#    +0x1f8 ActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x1fc ProcessAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x200 SystemDefaultActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x204 SystemAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x208 MinimumStackCommit : Uint4B
#    +0x20c FlsCallback      : Ptr32 _FLS_CALLBACK_INFO
#    +0x210 FlsListHead      : _LIST_ENTRY
#    +0x218 FlsBitmap        : Ptr32 Void
#    +0x21c FlsBitmapBits    : [4] Uint4B
#    +0x22c FlsHighIndex     : Uint4B
#    +0x230 WerRegistrationData : Ptr32 Void
#    +0x234 WerShipAssertPtr : Ptr32 Void
#    +0x238 pContextData     : Ptr32 Void
#    +0x23c pImageHeaderHash : Ptr32 Void
#    +0x240 TracingFlags     : Uint4B
#    +0x240 HeapTracingEnabled : Pos 0, 1 Bit
#    +0x240 CritSecTracingEnabled : Pos 1, 1 Bit
#    +0x240 SpareTracingBits : Pos 2, 30 Bits
class _PEB_2008_R2(Structure):
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID), # PRTL_CRITICAL_SECTION
        ("AtlThunkSListPtr",                    PVOID),
        ("IFEOKey",                             PVOID),
        ("CrossProcessFlags",                   DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("AtlThunkSListPtr32",                  PVOID),
        ("ApiSetMap",                           PVOID),
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("HotpatchInformation",                 PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr32 Ptr32 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  DWORD),
        ("HeapSegmentCommit",                   DWORD),
        ("HeapDeCommitTotalFreeThreshold",      DWORD),
        ("HeapDeCommitFreeBlockThreshold",      DWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr32 Ptr32 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ActiveProcessAffinityMask",           DWORD),
        ("GdiHandleBuffer",                     DWORD * 34),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  DWORD),
        ("FlsCallback",                         PVOID), # PFLS_CALLBACK_INFO
        ("FlsListHead",                         LIST_ENTRY),
        ("FlsBitmap",                           PVOID),
        ("FlsBitmapBits",                       DWORD * 4),
        ("FlsHighIndex",                        DWORD),
        ("WerRegistrationData",                 PVOID),
        ("WerShipAssertPtr",                    PVOID),
        ("pContextData",                        PVOID),
        ("pImageHeaderHash",                    PVOID),
        ("TracingFlags",                        DWORD),
    ]
    def __get_UserSharedInfoPtr(self):
        return self.KernelCallbackTable
    def __set_UserSharedInfoPtr(self, value):
        self.KernelCallbackTable = value
    UserSharedInfoPtr = property(__get_UserSharedInfoPtr, __set_UserSharedInfoPtr)

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 BitField         : UChar
#    +0x003 ImageUsesLargePages : Pos 0, 1 Bit
#    +0x003 IsProtectedProcess : Pos 1, 1 Bit
#    +0x003 IsLegacyProcess  : Pos 2, 1 Bit
#    +0x003 IsImageDynamicallyRelocated : Pos 3, 1 Bit
#    +0x003 SkipPatchingUser32Forwarders : Pos 4, 1 Bit
#    +0x003 SpareBits        : Pos 5, 3 Bits
#    +0x008 Mutant           : Ptr64 Void
#    +0x010 ImageBaseAddress : Ptr64 Void
#    +0x018 Ldr              : Ptr64 _PEB_LDR_DATA
#    +0x020 ProcessParameters : Ptr64 _RTL_USER_PROCESS_PARAMETERS
#    +0x028 SubSystemData    : Ptr64 Void
#    +0x030 ProcessHeap      : Ptr64 Void
#    +0x038 FastPebLock      : Ptr64 _RTL_CRITICAL_SECTION
#    +0x040 AtlThunkSListPtr : Ptr64 Void
#    +0x048 IFEOKey          : Ptr64 Void
#    +0x050 CrossProcessFlags : Uint4B
#    +0x050 ProcessInJob     : Pos 0, 1 Bit
#    +0x050 ProcessInitializing : Pos 1, 1 Bit
#    +0x050 ProcessUsingVEH  : Pos 2, 1 Bit
#    +0x050 ProcessUsingVCH  : Pos 3, 1 Bit
#    +0x050 ProcessUsingFTH  : Pos 4, 1 Bit
#    +0x050 ReservedBits0    : Pos 5, 27 Bits
#    +0x058 KernelCallbackTable : Ptr64 Void
#    +0x058 UserSharedInfoPtr : Ptr64 Void
#    +0x060 SystemReserved   : [1] Uint4B
#    +0x064 AtlThunkSListPtr32 : Uint4B
#    +0x068 ApiSetMap        : Ptr64 Void
#    +0x070 TlsExpansionCounter : Uint4B
#    +0x078 TlsBitmap        : Ptr64 Void
#    +0x080 TlsBitmapBits    : [2] Uint4B
#    +0x088 ReadOnlySharedMemoryBase : Ptr64 Void
#    +0x090 HotpatchInformation : Ptr64 Void
#    +0x098 ReadOnlyStaticServerData : Ptr64 Ptr64 Void
#    +0x0a0 AnsiCodePageData : Ptr64 Void
#    +0x0a8 OemCodePageData  : Ptr64 Void
#    +0x0b0 UnicodeCaseTableData : Ptr64 Void
#    +0x0b8 NumberOfProcessors : Uint4B
#    +0x0bc NtGlobalFlag     : Uint4B
#    +0x0c0 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x0c8 HeapSegmentReserve : Uint8B
#    +0x0d0 HeapSegmentCommit : Uint8B
#    +0x0d8 HeapDeCommitTotalFreeThreshold : Uint8B
#    +0x0e0 HeapDeCommitFreeBlockThreshold : Uint8B
#    +0x0e8 NumberOfHeaps    : Uint4B
#    +0x0ec MaximumNumberOfHeaps : Uint4B
#    +0x0f0 ProcessHeaps     : Ptr64 Ptr64 Void
#    +0x0f8 GdiSharedHandleTable : Ptr64 Void
#    +0x100 ProcessStarterHelper : Ptr64 Void
#    +0x108 GdiDCAttributeList : Uint4B
#    +0x110 LoaderLock       : Ptr64 _RTL_CRITICAL_SECTION
#    +0x118 OSMajorVersion   : Uint4B
#    +0x11c OSMinorVersion   : Uint4B
#    +0x120 OSBuildNumber    : Uint2B
#    +0x122 OSCSDVersion     : Uint2B
#    +0x124 OSPlatformId     : Uint4B
#    +0x128 ImageSubsystem   : Uint4B
#    +0x12c ImageSubsystemMajorVersion : Uint4B
#    +0x130 ImageSubsystemMinorVersion : Uint4B
#    +0x138 ActiveProcessAffinityMask : Uint8B
#    +0x140 GdiHandleBuffer  : [60] Uint4B
#    +0x230 PostProcessInitRoutine : Ptr64     void
#    +0x238 TlsExpansionBitmap : Ptr64 Void
#    +0x240 TlsExpansionBitmapBits : [32] Uint4B
#    +0x2c0 SessionId        : Uint4B
#    +0x2c8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x2d0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x2d8 pShimData        : Ptr64 Void
#    +0x2e0 AppCompatInfo    : Ptr64 Void
#    +0x2e8 CSDVersion       : _UNICODE_STRING
#    +0x2f8 ActivationContextData : Ptr64 _ACTIVATION_CONTEXT_DATA
#    +0x300 ProcessAssemblyStorageMap : Ptr64 _ASSEMBLY_STORAGE_MAP
#    +0x308 SystemDefaultActivationContextData : Ptr64 _ACTIVATION_CONTEXT_DATA
#    +0x310 SystemAssemblyStorageMap : Ptr64 _ASSEMBLY_STORAGE_MAP
#    +0x318 MinimumStackCommit : Uint8B
#    +0x320 FlsCallback      : Ptr64 _FLS_CALLBACK_INFO
#    +0x328 FlsListHead      : _LIST_ENTRY
#    +0x338 FlsBitmap        : Ptr64 Void
#    +0x340 FlsBitmapBits    : [4] Uint4B
#    +0x350 FlsHighIndex     : Uint4B
#    +0x358 WerRegistrationData : Ptr64 Void
#    +0x360 WerShipAssertPtr : Ptr64 Void
#    +0x368 pContextData     : Ptr64 Void
#    +0x370 pImageHeaderHash : Ptr64 Void
#    +0x378 TracingFlags     : Uint4B
#    +0x378 HeapTracingEnabled : Pos 0, 1 Bit
#    +0x378 CritSecTracingEnabled : Pos 1, 1 Bit
#    +0x378 SpareTracingBits : Pos 2, 30 Bits
class _PEB_2008_R2_64(Structure):
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID), # PRTL_CRITICAL_SECTION
        ("AtlThunkSListPtr",                    PVOID),
        ("IFEOKey",                             PVOID),
        ("CrossProcessFlags",                   DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("AtlThunkSListPtr32",                  DWORD),
        ("ApiSetMap",                           PVOID),
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("HotpatchInformation",                 PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr32 Ptr32 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  QWORD),
        ("HeapSegmentCommit",                   QWORD),
        ("HeapDeCommitTotalFreeThreshold",      QWORD),
        ("HeapDeCommitFreeBlockThreshold",      QWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr64 Ptr64 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ActiveProcessAffinityMask",           QWORD),
        ("GdiHandleBuffer",                     DWORD * 60),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  QWORD),
        ("FlsCallback",                         PVOID), # PFLS_CALLBACK_INFO
        ("FlsListHead",                         LIST_ENTRY),
        ("FlsBitmap",                           PVOID),
        ("FlsBitmapBits",                       DWORD * 4),
        ("FlsHighIndex",                        DWORD),
        ("WerRegistrationData",                 PVOID),
        ("WerShipAssertPtr",                    PVOID),
        ("pContextData",                        PVOID),
        ("pImageHeaderHash",                    PVOID),
        ("TracingFlags",                        DWORD),
    ]
    def __get_UserSharedInfoPtr(self):
        return self.KernelCallbackTable
    def __set_UserSharedInfoPtr(self, value):
        self.KernelCallbackTable = value
    UserSharedInfoPtr = property(__get_UserSharedInfoPtr, __set_UserSharedInfoPtr)

_PEB_Vista      = _PEB_2008
_PEB_Vista_64   = _PEB_2008_64
_PEB_W7         = _PEB_2008_R2
_PEB_W7_64      = _PEB_2008_R2_64

#    +0x000 InheritedAddressSpace : UChar
#    +0x001 ReadImageFileExecOptions : UChar
#    +0x002 BeingDebugged    : UChar
#    +0x003 BitField         : UChar
#    +0x003 ImageUsesLargePages : Pos 0, 1 Bit
#    +0x003 IsProtectedProcess : Pos 1, 1 Bit
#    +0x003 IsLegacyProcess  : Pos 2, 1 Bit
#    +0x003 IsImageDynamicallyRelocated : Pos 3, 1 Bit
#    +0x003 SkipPatchingUser32Forwarders : Pos 4, 1 Bit
#    +0x003 SpareBits        : Pos 5, 3 Bits
#    +0x004 Mutant           : Ptr32 Void
#    +0x008 ImageBaseAddress : Ptr32 Void
#    +0x00c Ldr              : Ptr32 _PEB_LDR_DATA
#    +0x010 ProcessParameters : Ptr32 _RTL_USER_PROCESS_PARAMETERS
#    +0x014 SubSystemData    : Ptr32 Void
#    +0x018 ProcessHeap      : Ptr32 Void
#    +0x01c FastPebLock      : Ptr32 _RTL_CRITICAL_SECTION
#    +0x020 AtlThunkSListPtr : Ptr32 Void
#    +0x024 IFEOKey          : Ptr32 Void
#    +0x028 CrossProcessFlags : Uint4B
#    +0x028 ProcessInJob     : Pos 0, 1 Bit
#    +0x028 ProcessInitializing : Pos 1, 1 Bit
#    +0x028 ProcessUsingVEH  : Pos 2, 1 Bit
#    +0x028 ProcessUsingVCH  : Pos 3, 1 Bit
#    +0x028 ProcessUsingFTH  : Pos 4, 1 Bit
#    +0x028 ReservedBits0    : Pos 5, 27 Bits
#    +0x02c KernelCallbackTable : Ptr32 Void
#    +0x02c UserSharedInfoPtr : Ptr32 Void
#    +0x030 SystemReserved   : [1] Uint4B
#    +0x034 TracingFlags     : Uint4B
#    +0x034 HeapTracingEnabled : Pos 0, 1 Bit
#    +0x034 CritSecTracingEnabled : Pos 1, 1 Bit
#    +0x034 SpareTracingBits : Pos 2, 30 Bits
#    +0x038 ApiSetMap        : Ptr32 Void
#    +0x03c TlsExpansionCounter : Uint4B
#    +0x040 TlsBitmap        : Ptr32 Void
#    +0x044 TlsBitmapBits    : [2] Uint4B
#    +0x04c ReadOnlySharedMemoryBase : Ptr32 Void
#    +0x050 HotpatchInformation : Ptr32 Void
#    +0x054 ReadOnlyStaticServerData : Ptr32 Ptr32 Void
#    +0x058 AnsiCodePageData : Ptr32 Void
#    +0x05c OemCodePageData  : Ptr32 Void
#    +0x060 UnicodeCaseTableData : Ptr32 Void
#    +0x064 NumberOfProcessors : Uint4B
#    +0x068 NtGlobalFlag     : Uint4B
#    +0x070 CriticalSectionTimeout : _LARGE_INTEGER
#    +0x078 HeapSegmentReserve : Uint4B
#    +0x07c HeapSegmentCommit : Uint4B
#    +0x080 HeapDeCommitTotalFreeThreshold : Uint4B
#    +0x084 HeapDeCommitFreeBlockThreshold : Uint4B
#    +0x088 NumberOfHeaps    : Uint4B
#    +0x08c MaximumNumberOfHeaps : Uint4B
#    +0x090 ProcessHeaps     : Ptr32 Ptr32 Void
#    +0x094 GdiSharedHandleTable : Ptr32 Void
#    +0x098 ProcessStarterHelper : Ptr32 Void
#    +0x09c GdiDCAttributeList : Uint4B
#    +0x0a0 LoaderLock       : Ptr32 _RTL_CRITICAL_SECTION
#    +0x0a4 OSMajorVersion   : Uint4B
#    +0x0a8 OSMinorVersion   : Uint4B
#    +0x0ac OSBuildNumber    : Uint2B
#    +0x0ae OSCSDVersion     : Uint2B
#    +0x0b0 OSPlatformId     : Uint4B
#    +0x0b4 ImageSubsystem   : Uint4B
#    +0x0b8 ImageSubsystemMajorVersion : Uint4B
#    +0x0bc ImageSubsystemMinorVersion : Uint4B
#    +0x0c0 ActiveProcessAffinityMask : Uint4B
#    +0x0c4 GdiHandleBuffer  : [34] Uint4B
#    +0x14c PostProcessInitRoutine : Ptr32     void
#    +0x150 TlsExpansionBitmap : Ptr32 Void
#    +0x154 TlsExpansionBitmapBits : [32] Uint4B
#    +0x1d4 SessionId        : Uint4B
#    +0x1d8 AppCompatFlags   : _ULARGE_INTEGER
#    +0x1e0 AppCompatFlagsUser : _ULARGE_INTEGER
#    +0x1e8 pShimData        : Ptr32 Void
#    +0x1ec AppCompatInfo    : Ptr32 Void
#    +0x1f0 CSDVersion       : _UNICODE_STRING
#    +0x1f8 ActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x1fc ProcessAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x200 SystemDefaultActivationContextData : Ptr32 _ACTIVATION_CONTEXT_DATA
#    +0x204 SystemAssemblyStorageMap : Ptr32 _ASSEMBLY_STORAGE_MAP
#    +0x208 MinimumStackCommit : Uint4B
#    +0x20c FlsCallback      : Ptr32 _FLS_CALLBACK_INFO
#    +0x210 FlsListHead      : _LIST_ENTRY
#    +0x218 FlsBitmap        : Ptr32 Void
#    +0x21c FlsBitmapBits    : [4] Uint4B
#    +0x22c FlsHighIndex     : Uint4B
#    +0x230 WerRegistrationData : Ptr32 Void
#    +0x234 WerShipAssertPtr : Ptr32 Void
#    +0x238 pContextData     : Ptr32 Void
#    +0x23c pImageHeaderHash : Ptr32 Void
class _PEB_W7_Beta(Structure):
    """
    This definition of the PEB structure is only valid for the beta versions
    of Windows 7. For the final version of Windows 7 use L{_PEB_W7} instead.
    This structure is not chosen automatically.
    """
    _pack_   = 8
    _fields_ = [
        ("InheritedAddressSpace",               BOOLEAN),
        ("ReadImageFileExecOptions",            UCHAR),
        ("BeingDebugged",                       BOOLEAN),
        ("BitField",                            UCHAR),
        ("Mutant",                              HANDLE),
        ("ImageBaseAddress",                    PVOID),
        ("Ldr",                                 PVOID), # PPEB_LDR_DATA
        ("ProcessParameters",                   PVOID), # PRTL_USER_PROCESS_PARAMETERS
        ("SubSystemData",                       PVOID),
        ("ProcessHeap",                         PVOID),
        ("FastPebLock",                         PVOID), # PRTL_CRITICAL_SECTION
        ("AtlThunkSListPtr",                    PVOID),
        ("IFEOKey",                             PVOID),
        ("CrossProcessFlags",                   DWORD),
        ("KernelCallbackTable",                 PVOID),
        ("SystemReserved",                      DWORD),
        ("TracingFlags",                        DWORD),
        ("ApiSetMap",                           PVOID),
        ("TlsExpansionCounter",                 DWORD),
        ("TlsBitmap",                           PVOID),
        ("TlsBitmapBits",                       DWORD * 2),
        ("ReadOnlySharedMemoryBase",            PVOID),
        ("HotpatchInformation",                 PVOID),
        ("ReadOnlyStaticServerData",            PVOID), # Ptr32 Ptr32 Void
        ("AnsiCodePageData",                    PVOID),
        ("OemCodePageData",                     PVOID),
        ("UnicodeCaseTableData",                PVOID),
        ("NumberOfProcessors",                  DWORD),
        ("NtGlobalFlag",                        DWORD),
        ("CriticalSectionTimeout",              LONGLONG),  # LARGE_INTEGER
        ("HeapSegmentReserve",                  DWORD),
        ("HeapSegmentCommit",                   DWORD),
        ("HeapDeCommitTotalFreeThreshold",      DWORD),
        ("HeapDeCommitFreeBlockThreshold",      DWORD),
        ("NumberOfHeaps",                       DWORD),
        ("MaximumNumberOfHeaps",                DWORD),
        ("ProcessHeaps",                        PVOID), # Ptr32 Ptr32 Void
        ("GdiSharedHandleTable",                PVOID),
        ("ProcessStarterHelper",                PVOID),
        ("GdiDCAttributeList",                  DWORD),
        ("LoaderLock",                          PVOID), # PRTL_CRITICAL_SECTION
        ("OSMajorVersion",                      DWORD),
        ("OSMinorVersion",                      DWORD),
        ("OSBuildNumber",                       WORD),
        ("OSCSDVersion",                        WORD),
        ("OSPlatformId",                        DWORD),
        ("ImageSubsystem",                      DWORD),
        ("ImageSubsystemMajorVersion",          DWORD),
        ("ImageSubsystemMinorVersion",          DWORD),
        ("ActiveProcessAffinityMask",           DWORD),
        ("GdiHandleBuffer",                     DWORD * 34),
        ("PostProcessInitRoutine",              PPS_POST_PROCESS_INIT_ROUTINE),
        ("TlsExpansionBitmap",                  PVOID),
        ("TlsExpansionBitmapBits",              DWORD * 32),
        ("SessionId",                           DWORD),
        ("AppCompatFlags",                      ULONGLONG), # ULARGE_INTEGER
        ("AppCompatFlagsUser",                  ULONGLONG), # ULARGE_INTEGER
        ("pShimData",                           PVOID),
        ("AppCompatInfo",                       PVOID),
        ("CSDVersion",                          UNICODE_STRING),
        ("ActivationContextData",               PVOID), # ACTIVATION_CONTEXT_DATA
        ("ProcessAssemblyStorageMap",           PVOID), # ASSEMBLY_STORAGE_MAP
        ("SystemDefaultActivationContextData",  PVOID), # ACTIVATION_CONTEXT_DATA
        ("SystemAssemblyStorageMap",            PVOID), # ASSEMBLY_STORAGE_MAP
        ("MinimumStackCommit",                  DWORD),
        ("FlsCallback",                         PVOID), # PFLS_CALLBACK_INFO
        ("FlsListHead",                         LIST_ENTRY),
        ("FlsBitmap",                           PVOID),
        ("FlsBitmapBits",                       DWORD * 4),
        ("FlsHighIndex",                        DWORD),
        ("WerRegistrationData",                 PVOID),
        ("WerShipAssertPtr",                    PVOID),
        ("pContextData",                        PVOID),
        ("pImageHeaderHash",                    PVOID),
    ]
    def __get_UserSharedInfoPtr(self):
        return self.KernelCallbackTable
    def __set_UserSharedInfoPtr(self, value):
        self.KernelCallbackTable = value
    UserSharedInfoPtr = property(__get_UserSharedInfoPtr, __set_UserSharedInfoPtr)

# Use the correct PEB structure definition.
# Defaults to the latest Windows version.
class PEB(Structure):
    _pack_ = 8
    if os == 'Windows NT':
        _pack_   = _PEB_NT._pack_
        _fields_ = _PEB_NT._fields_
    elif os == 'Windows 2000':
        _pack_   = _PEB_2000._pack_
        _fields_ = _PEB_2000._fields_
    elif os == 'Windows XP':
        _fields_ = _PEB_XP._fields_
    elif os == 'Windows XP (64 bits)':
        _fields_ = _PEB_XP_64._fields_
    elif os == 'Windows 2003':
        _fields_ = _PEB_2003._fields_
    elif os == 'Windows 2003 (64 bits)':
        _fields_ = _PEB_2003_64._fields_
    elif os == 'Windows 2003 R2':
        _fields_ = _PEB_2003_R2._fields_
    elif os == 'Windows 2003 R2 (64 bits)':
        _fields_ = _PEB_2003_R2_64._fields_
    elif os == 'Windows 2008':
        _fields_ = _PEB_2008._fields_
    elif os == 'Windows 2008 (64 bits)':
        _fields_ = _PEB_2008_64._fields_
    elif os == 'Windows 2008 R2':
        _fields_ = _PEB_2008_R2._fields_
    elif os == 'Windows 2008 R2 (64 bits)':
        _fields_ = _PEB_2008_R2_64._fields_
    elif os == 'Windows Vista':
        _fields_ = _PEB_Vista._fields_
    elif os == 'Windows Vista (64 bits)':
        _fields_ = _PEB_Vista_64._fields_
    elif os == 'Windows 7':
        _fields_ = _PEB_W7._fields_
    elif os == 'Windows 7 (64 bits)':
        _fields_ = _PEB_W7_64._fields_
    elif sizeof(SIZE_T) == sizeof(DWORD):
        _fields_ = _PEB_W7._fields_
    else:
        _fields_ = _PEB_W7_64._fields_
PPEB = POINTER(PEB)

# PEB structure for WOW64 processes.
class PEB_32(Structure):
    _pack_ = 8
    if os == 'Windows NT':
        _pack_   = _PEB_NT._pack_
        _fields_ = _PEB_NT._fields_
    elif os == 'Windows 2000':
        _pack_   = _PEB_2000._pack_
        _fields_ = _PEB_2000._fields_
    elif os.startswith('Windows XP'):
        _fields_ = _PEB_XP._fields_
    elif os.startswith('Windows 2003 R2'):
        _fields_ = _PEB_2003_R2._fields_
    elif os.startswith('Windows 2003'):
        _fields_ = _PEB_2003._fields_
    elif os.startswith('Windows 2008 R2'):
        _fields_ = _PEB_2008_R2._fields_
    elif os.startswith('Windows 2008'):
        _fields_ = _PEB_2008._fields_
    elif os.startswith('Windows Vista'):
        _fields_ = _PEB_Vista._fields_
    else: #if os.startswith('Windows 7'):
        _fields_ = _PEB_W7._fields_

# from https://vmexplorer.svn.codeplex.com/svn/VMExplorer/src/Win32/Threads.cs
#
# [StructLayout (LayoutKind.Sequential, Size = 0x0C)]
# public struct Wx86ThreadState
# {
# 	public IntPtr  CallBx86Eip; // Ptr32 to Uint4B
# 	public IntPtr  DeallocationCpu; // Ptr32 to Void
# 	public Byte  UseKnownWx86Dll; // UChar
# 	public Byte  OleStubInvoked; // Char
# };
class Wx86ThreadState(Structure):
    _fields_ = [
        ("CallBx86Eip",             PVOID),
        ("DeallocationCpu",         PVOID),
        ("UseKnownWx86Dll",         UCHAR),
        ("OleStubInvoked",          CHAR),
]

# ntdll!_RTL_ACTIVATION_CONTEXT_STACK_FRAME
#    +0x000 Previous         : Ptr64 _RTL_ACTIVATION_CONTEXT_STACK_FRAME
#    +0x008 ActivationContext : Ptr64 _ACTIVATION_CONTEXT
#    +0x010 Flags            : Uint4B
class RTL_ACTIVATION_CONTEXT_STACK_FRAME(Structure):
    _fields_ = [
        ("Previous",                    PVOID),
        ("ActivationContext",           PVOID),
        ("Flags",                       DWORD),
]

# ntdll!_ACTIVATION_CONTEXT_STACK
#    +0x000 ActiveFrame      : Ptr64 _RTL_ACTIVATION_CONTEXT_STACK_FRAME
#    +0x008 FrameListCache   : _LIST_ENTRY
#    +0x018 Flags            : Uint4B
#    +0x01c NextCookieSequenceNumber : Uint4B
#    +0x020 StackId          : Uint4B
class ACTIVATION_CONTEXT_STACK(Structure):
    _fields_ = [
        ("ActiveFrame",                 PVOID),
        ("FrameListCache",              LIST_ENTRY),
        ("Flags",                       DWORD),
        ("NextCookieSequenceNumber",    DWORD),
        ("StackId",                     DWORD),
]

# typedef struct _PROCESSOR_NUMBER {
#   WORD Group;
#   BYTE Number;
#   BYTE Reserved;
# }PROCESSOR_NUMBER, *PPROCESSOR_NUMBER;
class PROCESSOR_NUMBER(Structure):
    _fields_ = [
        ("Group",       WORD),
        ("Number",      BYTE),
        ("Reserved",    BYTE),
]

# from http://www.nirsoft.net/kernel_struct/vista/NT_TIB.html
#
# typedef struct _NT_TIB
# {
#      PEXCEPTION_REGISTRATION_RECORD ExceptionList;
#      PVOID StackBase;
#      PVOID StackLimit;
#      PVOID SubSystemTib;
#      union
#      {
#           PVOID FiberData;
#           ULONG Version;
#      };
#      PVOID ArbitraryUserPointer;
#      PNT_TIB Self;
# } NT_TIB, *PNT_TIB;
class _NT_TIB_UNION(Union):
    _fields_ = [
        ("FiberData",   PVOID),
        ("Version",     ULONG),
    ]
class NT_TIB(Structure):
    _fields_ = [
        ("ExceptionList",           PVOID), # PEXCEPTION_REGISTRATION_RECORD
        ("StackBase",               PVOID),
        ("StackLimit",              PVOID),
        ("SubSystemTib",            PVOID),
        ("u",                       _NT_TIB_UNION),
        ("ArbitraryUserPointer",    PVOID),
        ("Self",                    PVOID), # PNTTIB
    ]

    def __get_FiberData(self):
        return self.u.FiberData
    def __set_FiberData(self, value):
        self.u.FiberData = value
    FiberData = property(__get_FiberData, __set_FiberData)

    def __get_Version(self):
        return self.u.Version
    def __set_Version(self, value):
        self.u.Version = value
    Version = property(__get_Version, __set_Version)

PNTTIB = POINTER(NT_TIB)

# From http://www.nirsoft.net/kernel_struct/vista/EXCEPTION_REGISTRATION_RECORD.html
#
# typedef struct _EXCEPTION_REGISTRATION_RECORD
# {
#      PEXCEPTION_REGISTRATION_RECORD Next;
#      PEXCEPTION_DISPOSITION Handler;
# } EXCEPTION_REGISTRATION_RECORD, *PEXCEPTION_REGISTRATION_RECORD;
class EXCEPTION_REGISTRATION_RECORD(Structure):
    pass

EXCEPTION_DISPOSITION           = DWORD
##PEXCEPTION_DISPOSITION          = POINTER(EXCEPTION_DISPOSITION)
##PEXCEPTION_REGISTRATION_RECORD  = POINTER(EXCEPTION_REGISTRATION_RECORD)
PEXCEPTION_DISPOSITION          = PVOID
PEXCEPTION_REGISTRATION_RECORD  = PVOID

EXCEPTION_REGISTRATION_RECORD._fields_ = [
        ("Next",    PEXCEPTION_REGISTRATION_RECORD),
        ("Handler", PEXCEPTION_DISPOSITION),
]

##PPEB = POINTER(PEB)
PPEB = PVOID

# From http://www.nirsoft.net/kernel_struct/vista/GDI_TEB_BATCH.html
#
# typedef struct _GDI_TEB_BATCH
# {
#      ULONG Offset;
#      ULONG HDC;
#      ULONG Buffer[310];
# } GDI_TEB_BATCH, *PGDI_TEB_BATCH;
class GDI_TEB_BATCH(Structure):
    _fields_ = [
        ("Offset",  ULONG),
        ("HDC",     ULONG),
        ("Buffer",  ULONG * 310),
]

# ntdll!_TEB_ACTIVE_FRAME_CONTEXT
#    +0x000 Flags            : Uint4B
#    +0x008 FrameName        : Ptr64 Char
class TEB_ACTIVE_FRAME_CONTEXT(Structure):
    _fields_ = [
        ("Flags",       DWORD),
        ("FrameName",   LPVOID),    # LPCHAR
]
PTEB_ACTIVE_FRAME_CONTEXT = POINTER(TEB_ACTIVE_FRAME_CONTEXT)

# ntdll!_TEB_ACTIVE_FRAME
#    +0x000 Flags            : Uint4B
#    +0x008 Previous         : Ptr64 _TEB_ACTIVE_FRAME
#    +0x010 Context          : Ptr64 _TEB_ACTIVE_FRAME_CONTEXT
class TEB_ACTIVE_FRAME(Structure):
    _fields_ = [
        ("Flags",       DWORD),
        ("Previous",    LPVOID),    # PTEB_ACTIVE_FRAME
        ("Context",     LPVOID),    # PTEB_ACTIVE_FRAME_CONTEXT
]
PTEB_ACTIVE_FRAME = POINTER(TEB_ACTIVE_FRAME)

# SameTebFlags
DbgSafeThunkCall        = 1 << 0
DbgInDebugPrint         = 1 << 1
DbgHasFiberData         = 1 << 2
DbgSkipThreadAttach     = 1 << 3
DbgWerInShipAssertCode  = 1 << 4
DbgRanProcessInit       = 1 << 5
DbgClonedThread         = 1 << 6
DbgSuppressDebugMsg     = 1 << 7
RtlDisableUserStackWalk = 1 << 8
RtlExceptionAttached    = 1 << 9
RtlInitialThread        = 1 << 10

# XXX This is quite wrong :P
class _TEB_NT(Structure):
    _pack_ = 4
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 HANDLE),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PPEB),
        ("LastErrorValue",                  ULONG),
        ("CountOfOwnedCriticalSections",    ULONG),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  ULONG * 26),
        ("UserReserved",                    ULONG * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   ULONG),
        ("FpSoftwareStatusRegister",        ULONG),
        ("SystemReserved1",                 PVOID * 54),
        ("Spare1",                          PVOID),
        ("ExceptionCode",                   ULONG),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes1",                     ULONG * 36),
        ("TxFsContext",                     ULONG),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          PVOID),
        ("GdiClientPID",                    ULONG),
        ("GdiClientTID",                    ULONG),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 PVOID * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     ULONG * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorDisabled",               ULONG),
        ("Instrumentation",                 PVOID * 9),
        ("ActivityId",                      GUID),
        ("SubProcessTag",                   PVOID),
        ("EtwLocalData",                    PVOID),
        ("EtwTraceData",                    PVOID),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   ULONG),
        ("SpareBool0",                      BOOLEAN),
        ("SpareBool1",                      BOOLEAN),
        ("SpareBool2",                      BOOLEAN),
        ("IdealProcessor",                  UCHAR),
        ("GuaranteedStackBytes",            ULONG),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             ULONG),
        ("StackCommit",                     PVOID),
        ("StackCommitMax",                  PVOID),
        ("StackReserved",                   PVOID),
]

# not really, but "dt _TEB" in w2k isn't working for me :(
_TEB_2000 = _TEB_NT

#    +0x000 NtTib            : _NT_TIB
#    +0x01c EnvironmentPointer : Ptr32 Void
#    +0x020 ClientId         : _CLIENT_ID
#    +0x028 ActiveRpcHandle  : Ptr32 Void
#    +0x02c ThreadLocalStoragePointer : Ptr32 Void
#    +0x030 ProcessEnvironmentBlock : Ptr32 _PEB
#    +0x034 LastErrorValue   : Uint4B
#    +0x038 CountOfOwnedCriticalSections : Uint4B
#    +0x03c CsrClientThread  : Ptr32 Void
#    +0x040 Win32ThreadInfo  : Ptr32 Void
#    +0x044 User32Reserved   : [26] Uint4B
#    +0x0ac UserReserved     : [5] Uint4B
#    +0x0c0 WOW32Reserved    : Ptr32 Void
#    +0x0c4 CurrentLocale    : Uint4B
#    +0x0c8 FpSoftwareStatusRegister : Uint4B
#    +0x0cc SystemReserved1  : [54] Ptr32 Void
#    +0x1a4 ExceptionCode    : Int4B
#    +0x1a8 ActivationContextStack : _ACTIVATION_CONTEXT_STACK
#    +0x1bc SpareBytes1      : [24] UChar
#    +0x1d4 GdiTebBatch      : _GDI_TEB_BATCH
#    +0x6b4 RealClientId     : _CLIENT_ID
#    +0x6bc GdiCachedProcessHandle : Ptr32 Void
#    +0x6c0 GdiClientPID     : Uint4B
#    +0x6c4 GdiClientTID     : Uint4B
#    +0x6c8 GdiThreadLocalInfo : Ptr32 Void
#    +0x6cc Win32ClientInfo  : [62] Uint4B
#    +0x7c4 glDispatchTable  : [233] Ptr32 Void
#    +0xb68 glReserved1      : [29] Uint4B
#    +0xbdc glReserved2      : Ptr32 Void
#    +0xbe0 glSectionInfo    : Ptr32 Void
#    +0xbe4 glSection        : Ptr32 Void
#    +0xbe8 glTable          : Ptr32 Void
#    +0xbec glCurrentRC      : Ptr32 Void
#    +0xbf0 glContext        : Ptr32 Void
#    +0xbf4 LastStatusValue  : Uint4B
#    +0xbf8 StaticUnicodeString : _UNICODE_STRING
#    +0xc00 StaticUnicodeBuffer : [261] Uint2B
#    +0xe0c DeallocationStack : Ptr32 Void
#    +0xe10 TlsSlots         : [64] Ptr32 Void
#    +0xf10 TlsLinks         : _LIST_ENTRY
#    +0xf18 Vdm              : Ptr32 Void
#    +0xf1c ReservedForNtRpc : Ptr32 Void
#    +0xf20 DbgSsReserved    : [2] Ptr32 Void
#    +0xf28 HardErrorsAreDisabled : Uint4B
#    +0xf2c Instrumentation  : [16] Ptr32 Void
#    +0xf6c WinSockData      : Ptr32 Void
#    +0xf70 GdiBatchCount    : Uint4B
#    +0xf74 InDbgPrint       : UChar
#    +0xf75 FreeStackOnTermination : UChar
#    +0xf76 HasFiberData     : UChar
#    +0xf77 IdealProcessor   : UChar
#    +0xf78 Spare3           : Uint4B
#    +0xf7c ReservedForPerf  : Ptr32 Void
#    +0xf80 ReservedForOle   : Ptr32 Void
#    +0xf84 WaitingOnLoaderLock : Uint4B
#    +0xf88 Wx86Thread       : _Wx86ThreadState
#    +0xf94 TlsExpansionSlots : Ptr32 Ptr32 Void
#    +0xf98 ImpersonationLocale : Uint4B
#    +0xf9c IsImpersonating  : Uint4B
#    +0xfa0 NlsCache         : Ptr32 Void
#    +0xfa4 pShimData        : Ptr32 Void
#    +0xfa8 HeapVirtualAffinity : Uint4B
#    +0xfac CurrentTransactionHandle : Ptr32 Void
#    +0xfb0 ActiveFrame      : Ptr32 _TEB_ACTIVE_FRAME
#    +0xfb4 SafeThunkCall    : UChar
#    +0xfb5 BooleanSpare     : [3] UChar
class _TEB_XP(Structure):
    _pack_ = 8
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 HANDLE),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PVOID), # PPEB
        ("LastErrorValue",                  DWORD),
        ("CountOfOwnedCriticalSections",    DWORD),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  DWORD * 26),
        ("UserReserved",                    DWORD * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   DWORD),
        ("FpSoftwareStatusRegister",        DWORD),
        ("SystemReserved1",                 PVOID * 54),
        ("ExceptionCode",                   SDWORD),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes1",                     UCHAR * 24),
        ("TxFsContext",                     DWORD),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          HANDLE),
        ("GdiClientPID",                    DWORD),
        ("GdiClientTID",                    DWORD),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 DWORD * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     DWORD * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorsAreDisabled",           DWORD),
        ("Instrumentation",                 PVOID * 16),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   DWORD),
        ("InDbgPrint",                      BOOLEAN),
        ("FreeStackOnTermination",          BOOLEAN),
        ("HasFiberData",                    BOOLEAN),
        ("IdealProcessor",                  UCHAR),
        ("Spare3",                          DWORD),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             DWORD),
        ("Wx86Thread",                      Wx86ThreadState),
        ("TlsExpansionSlots",               PVOID), # Ptr32 Ptr32 Void
        ("ImpersonationLocale",             DWORD),
        ("IsImpersonating",                 BOOL),
        ("NlsCache",                        PVOID),
        ("pShimData",                       PVOID),
        ("HeapVirtualAffinity",             DWORD),
        ("CurrentTransactionHandle",        HANDLE),
        ("ActiveFrame",                     PVOID), # PTEB_ACTIVE_FRAME
        ("SafeThunkCall",                   BOOLEAN),
        ("BooleanSpare",                    BOOLEAN * 3),
]

#    +0x000 NtTib            : _NT_TIB
#    +0x038 EnvironmentPointer : Ptr64 Void
#    +0x040 ClientId         : _CLIENT_ID
#    +0x050 ActiveRpcHandle  : Ptr64 Void
#    +0x058 ThreadLocalStoragePointer : Ptr64 Void
#    +0x060 ProcessEnvironmentBlock : Ptr64 _PEB
#    +0x068 LastErrorValue   : Uint4B
#    +0x06c CountOfOwnedCriticalSections : Uint4B
#    +0x070 CsrClientThread  : Ptr64 Void
#    +0x078 Win32ThreadInfo  : Ptr64 Void
#    +0x080 User32Reserved   : [26] Uint4B
#    +0x0e8 UserReserved     : [5] Uint4B
#    +0x100 WOW32Reserved    : Ptr64 Void
#    +0x108 CurrentLocale    : Uint4B
#    +0x10c FpSoftwareStatusRegister : Uint4B
#    +0x110 SystemReserved1  : [54] Ptr64 Void
#    +0x2c0 ExceptionCode    : Int4B
#    +0x2c8 ActivationContextStackPointer : Ptr64 _ACTIVATION_CONTEXT_STACK
#    +0x2d0 SpareBytes1      : [28] UChar
#    +0x2f0 GdiTebBatch      : _GDI_TEB_BATCH
#    +0x7d8 RealClientId     : _CLIENT_ID
#    +0x7e8 GdiCachedProcessHandle : Ptr64 Void
#    +0x7f0 GdiClientPID     : Uint4B
#    +0x7f4 GdiClientTID     : Uint4B
#    +0x7f8 GdiThreadLocalInfo : Ptr64 Void
#    +0x800 Win32ClientInfo  : [62] Uint8B
#    +0x9f0 glDispatchTable  : [233] Ptr64 Void
#    +0x1138 glReserved1      : [29] Uint8B
#    +0x1220 glReserved2      : Ptr64 Void
#    +0x1228 glSectionInfo    : Ptr64 Void
#    +0x1230 glSection        : Ptr64 Void
#    +0x1238 glTable          : Ptr64 Void
#    +0x1240 glCurrentRC      : Ptr64 Void
#    +0x1248 glContext        : Ptr64 Void
#    +0x1250 LastStatusValue  : Uint4B
#    +0x1258 StaticUnicodeString : _UNICODE_STRING
#    +0x1268 StaticUnicodeBuffer : [261] Uint2B
#    +0x1478 DeallocationStack : Ptr64 Void
#    +0x1480 TlsSlots         : [64] Ptr64 Void
#    +0x1680 TlsLinks         : _LIST_ENTRY
#    +0x1690 Vdm              : Ptr64 Void
#    +0x1698 ReservedForNtRpc : Ptr64 Void
#    +0x16a0 DbgSsReserved    : [2] Ptr64 Void
#    +0x16b0 HardErrorMode    : Uint4B
#    +0x16b8 Instrumentation  : [14] Ptr64 Void
#    +0x1728 SubProcessTag    : Ptr64 Void
#    +0x1730 EtwTraceData     : Ptr64 Void
#    +0x1738 WinSockData      : Ptr64 Void
#    +0x1740 GdiBatchCount    : Uint4B
#    +0x1744 InDbgPrint       : UChar
#    +0x1745 FreeStackOnTermination : UChar
#    +0x1746 HasFiberData     : UChar
#    +0x1747 IdealProcessor   : UChar
#    +0x1748 GuaranteedStackBytes : Uint4B
#    +0x1750 ReservedForPerf  : Ptr64 Void
#    +0x1758 ReservedForOle   : Ptr64 Void
#    +0x1760 WaitingOnLoaderLock : Uint4B
#    +0x1768 SparePointer1    : Uint8B
#    +0x1770 SoftPatchPtr1    : Uint8B
#    +0x1778 SoftPatchPtr2    : Uint8B
#    +0x1780 TlsExpansionSlots : Ptr64 Ptr64 Void
#    +0x1788 DeallocationBStore : Ptr64 Void
#    +0x1790 BStoreLimit      : Ptr64 Void
#    +0x1798 ImpersonationLocale : Uint4B
#    +0x179c IsImpersonating  : Uint4B
#    +0x17a0 NlsCache         : Ptr64 Void
#    +0x17a8 pShimData        : Ptr64 Void
#    +0x17b0 HeapVirtualAffinity : Uint4B
#    +0x17b8 CurrentTransactionHandle : Ptr64 Void
#    +0x17c0 ActiveFrame      : Ptr64 _TEB_ACTIVE_FRAME
#    +0x17c8 FlsData          : Ptr64 Void
#    +0x17d0 SafeThunkCall    : UChar
#    +0x17d1 BooleanSpare     : [3] UChar
class _TEB_XP_64(Structure):
    _pack_ = 8
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 PVOID),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PVOID), # PPEB
        ("LastErrorValue",                  DWORD),
        ("CountOfOwnedCriticalSections",    DWORD),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  DWORD * 26),
        ("UserReserved",                    DWORD * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   DWORD),
        ("FpSoftwareStatusRegister",        DWORD),
        ("SystemReserved1",                 PVOID * 54),
        ("ExceptionCode",                   SDWORD),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes1",                     UCHAR * 28),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          HANDLE),
        ("GdiClientPID",                    DWORD),
        ("GdiClientTID",                    DWORD),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 QWORD * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     QWORD * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorMode",                   DWORD),
        ("Instrumentation",                 PVOID * 14),
        ("SubProcessTag",                   PVOID),
        ("EtwTraceData",                    PVOID),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   DWORD),
        ("InDbgPrint",                      BOOLEAN),
        ("FreeStackOnTermination",          BOOLEAN),
        ("HasFiberData",                    BOOLEAN),
        ("IdealProcessor",                  UCHAR),
        ("GuaranteedStackBytes",            DWORD),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             DWORD),
        ("SparePointer1",                   PVOID),
        ("SoftPatchPtr1",                   PVOID),
        ("SoftPatchPtr2",                   PVOID),
        ("TlsExpansionSlots",               PVOID), # Ptr64 Ptr64 Void
        ("DeallocationBStore",              PVOID),
        ("BStoreLimit",                     PVOID),
        ("ImpersonationLocale",             DWORD),
        ("IsImpersonating",                 BOOL),
        ("NlsCache",                        PVOID),
        ("pShimData",                       PVOID),
        ("HeapVirtualAffinity",             DWORD),
        ("CurrentTransactionHandle",        HANDLE),
        ("ActiveFrame",                     PVOID), # PTEB_ACTIVE_FRAME
        ("FlsData",                         PVOID),
        ("SafeThunkCall",                   BOOLEAN),
        ("BooleanSpare",                    BOOLEAN * 3),
]

#    +0x000 NtTib            : _NT_TIB
#    +0x01c EnvironmentPointer : Ptr32 Void
#    +0x020 ClientId         : _CLIENT_ID
#    +0x028 ActiveRpcHandle  : Ptr32 Void
#    +0x02c ThreadLocalStoragePointer : Ptr32 Void
#    +0x030 ProcessEnvironmentBlock : Ptr32 _PEB
#    +0x034 LastErrorValue   : Uint4B
#    +0x038 CountOfOwnedCriticalSections : Uint4B
#    +0x03c CsrClientThread  : Ptr32 Void
#    +0x040 Win32ThreadInfo  : Ptr32 Void
#    +0x044 User32Reserved   : [26] Uint4B
#    +0x0ac UserReserved     : [5] Uint4B
#    +0x0c0 WOW32Reserved    : Ptr32 Void
#    +0x0c4 CurrentLocale    : Uint4B
#    +0x0c8 FpSoftwareStatusRegister : Uint4B
#    +0x0cc SystemReserved1  : [54] Ptr32 Void
#    +0x1a4 ExceptionCode    : Int4B
#    +0x1a8 ActivationContextStackPointer : Ptr32 _ACTIVATION_CONTEXT_STACK
#    +0x1ac SpareBytes1      : [40] UChar
#    +0x1d4 GdiTebBatch      : _GDI_TEB_BATCH
#    +0x6b4 RealClientId     : _CLIENT_ID
#    +0x6bc GdiCachedProcessHandle : Ptr32 Void
#    +0x6c0 GdiClientPID     : Uint4B
#    +0x6c4 GdiClientTID     : Uint4B
#    +0x6c8 GdiThreadLocalInfo : Ptr32 Void
#    +0x6cc Win32ClientInfo  : [62] Uint4B
#    +0x7c4 glDispatchTable  : [233] Ptr32 Void
#    +0xb68 glReserved1      : [29] Uint4B
#    +0xbdc glReserved2      : Ptr32 Void
#    +0xbe0 glSectionInfo    : Ptr32 Void
#    +0xbe4 glSection        : Ptr32 Void
#    +0xbe8 glTable          : Ptr32 Void
#    +0xbec glCurrentRC      : Ptr32 Void
#    +0xbf0 glContext        : Ptr32 Void
#    +0xbf4 LastStatusValue  : Uint4B
#    +0xbf8 StaticUnicodeString : _UNICODE_STRING
#    +0xc00 StaticUnicodeBuffer : [261] Uint2B
#    +0xe0c DeallocationStack : Ptr32 Void
#    +0xe10 TlsSlots         : [64] Ptr32 Void
#    +0xf10 TlsLinks         : _LIST_ENTRY
#    +0xf18 Vdm              : Ptr32 Void
#    +0xf1c ReservedForNtRpc : Ptr32 Void
#    +0xf20 DbgSsReserved    : [2] Ptr32 Void
#    +0xf28 HardErrorMode    : Uint4B
#    +0xf2c Instrumentation  : [14] Ptr32 Void
#    +0xf64 SubProcessTag    : Ptr32 Void
#    +0xf68 EtwTraceData     : Ptr32 Void
#    +0xf6c WinSockData      : Ptr32 Void
#    +0xf70 GdiBatchCount    : Uint4B
#    +0xf74 InDbgPrint       : UChar
#    +0xf75 FreeStackOnTermination : UChar
#    +0xf76 HasFiberData     : UChar
#    +0xf77 IdealProcessor   : UChar
#    +0xf78 GuaranteedStackBytes : Uint4B
#    +0xf7c ReservedForPerf  : Ptr32 Void
#    +0xf80 ReservedForOle   : Ptr32 Void
#    +0xf84 WaitingOnLoaderLock : Uint4B
#    +0xf88 SparePointer1    : Uint4B
#    +0xf8c SoftPatchPtr1    : Uint4B
#    +0xf90 SoftPatchPtr2    : Uint4B
#    +0xf94 TlsExpansionSlots : Ptr32 Ptr32 Void
#    +0xf98 ImpersonationLocale : Uint4B
#    +0xf9c IsImpersonating  : Uint4B
#    +0xfa0 NlsCache         : Ptr32 Void
#    +0xfa4 pShimData        : Ptr32 Void
#    +0xfa8 HeapVirtualAffinity : Uint4B
#    +0xfac CurrentTransactionHandle : Ptr32 Void
#    +0xfb0 ActiveFrame      : Ptr32 _TEB_ACTIVE_FRAME
#    +0xfb4 FlsData          : Ptr32 Void
#    +0xfb8 SafeThunkCall    : UChar
#    +0xfb9 BooleanSpare     : [3] UChar
class _TEB_2003(Structure):
    _pack_ = 8
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 HANDLE),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PVOID), # PPEB
        ("LastErrorValue",                  DWORD),
        ("CountOfOwnedCriticalSections",    DWORD),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  DWORD * 26),
        ("UserReserved",                    DWORD * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   DWORD),
        ("FpSoftwareStatusRegister",        DWORD),
        ("SystemReserved1",                 PVOID * 54),
        ("ExceptionCode",                   SDWORD),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes1",                     UCHAR * 40),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          HANDLE),
        ("GdiClientPID",                    DWORD),
        ("GdiClientTID",                    DWORD),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 DWORD * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     DWORD * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorMode",                   DWORD),
        ("Instrumentation",                 PVOID * 14),
        ("SubProcessTag",                   PVOID),
        ("EtwTraceData",                    PVOID),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   DWORD),
        ("InDbgPrint",                      BOOLEAN),
        ("FreeStackOnTermination",          BOOLEAN),
        ("HasFiberData",                    BOOLEAN),
        ("IdealProcessor",                  UCHAR),
        ("GuaranteedStackBytes",            DWORD),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             DWORD),
        ("SparePointer1",                   PVOID),
        ("SoftPatchPtr1",                   PVOID),
        ("SoftPatchPtr2",                   PVOID),
        ("TlsExpansionSlots",               PVOID), # Ptr32 Ptr32 Void
        ("ImpersonationLocale",             DWORD),
        ("IsImpersonating",                 BOOL),
        ("NlsCache",                        PVOID),
        ("pShimData",                       PVOID),
        ("HeapVirtualAffinity",             DWORD),
        ("CurrentTransactionHandle",        HANDLE),
        ("ActiveFrame",                     PVOID), # PTEB_ACTIVE_FRAME
        ("FlsData",                         PVOID),
        ("SafeThunkCall",                   BOOLEAN),
        ("BooleanSpare",                    BOOLEAN * 3),
]

_TEB_2003_64    = _TEB_XP_64
_TEB_2003_R2    = _TEB_2003
_TEB_2003_R2_64 = _TEB_2003_64

#    +0x000 NtTib            : _NT_TIB
#    +0x01c EnvironmentPointer : Ptr32 Void
#    +0x020 ClientId         : _CLIENT_ID
#    +0x028 ActiveRpcHandle  : Ptr32 Void
#    +0x02c ThreadLocalStoragePointer : Ptr32 Void
#    +0x030 ProcessEnvironmentBlock : Ptr32 _PEB
#    +0x034 LastErrorValue   : Uint4B
#    +0x038 CountOfOwnedCriticalSections : Uint4B
#    +0x03c CsrClientThread  : Ptr32 Void
#    +0x040 Win32ThreadInfo  : Ptr32 Void
#    +0x044 User32Reserved   : [26] Uint4B
#    +0x0ac UserReserved     : [5] Uint4B
#    +0x0c0 WOW32Reserved    : Ptr32 Void
#    +0x0c4 CurrentLocale    : Uint4B
#    +0x0c8 FpSoftwareStatusRegister : Uint4B
#    +0x0cc SystemReserved1  : [54] Ptr32 Void
#    +0x1a4 ExceptionCode    : Int4B
#    +0x1a8 ActivationContextStackPointer : Ptr32 _ACTIVATION_CONTEXT_STACK
#    +0x1ac SpareBytes1      : [36] UChar
#    +0x1d0 TxFsContext      : Uint4B
#    +0x1d4 GdiTebBatch      : _GDI_TEB_BATCH
#    +0x6b4 RealClientId     : _CLIENT_ID
#    +0x6bc GdiCachedProcessHandle : Ptr32 Void
#    +0x6c0 GdiClientPID     : Uint4B
#    +0x6c4 GdiClientTID     : Uint4B
#    +0x6c8 GdiThreadLocalInfo : Ptr32 Void
#    +0x6cc Win32ClientInfo  : [62] Uint4B
#    +0x7c4 glDispatchTable  : [233] Ptr32 Void
#    +0xb68 glReserved1      : [29] Uint4B
#    +0xbdc glReserved2      : Ptr32 Void
#    +0xbe0 glSectionInfo    : Ptr32 Void
#    +0xbe4 glSection        : Ptr32 Void
#    +0xbe8 glTable          : Ptr32 Void
#    +0xbec glCurrentRC      : Ptr32 Void
#    +0xbf0 glContext        : Ptr32 Void
#    +0xbf4 LastStatusValue  : Uint4B
#    +0xbf8 StaticUnicodeString : _UNICODE_STRING
#    +0xc00 StaticUnicodeBuffer : [261] Wchar
#    +0xe0c DeallocationStack : Ptr32 Void
#    +0xe10 TlsSlots         : [64] Ptr32 Void
#    +0xf10 TlsLinks         : _LIST_ENTRY
#    +0xf18 Vdm              : Ptr32 Void
#    +0xf1c ReservedForNtRpc : Ptr32 Void
#    +0xf20 DbgSsReserved    : [2] Ptr32 Void
#    +0xf28 HardErrorMode    : Uint4B
#    +0xf2c Instrumentation  : [9] Ptr32 Void
#    +0xf50 ActivityId       : _GUID
#    +0xf60 SubProcessTag    : Ptr32 Void
#    +0xf64 EtwLocalData     : Ptr32 Void
#    +0xf68 EtwTraceData     : Ptr32 Void
#    +0xf6c WinSockData      : Ptr32 Void
#    +0xf70 GdiBatchCount    : Uint4B
#    +0xf74 SpareBool0       : UChar
#    +0xf75 SpareBool1       : UChar
#    +0xf76 SpareBool2       : UChar
#    +0xf77 IdealProcessor   : UChar
#    +0xf78 GuaranteedStackBytes : Uint4B
#    +0xf7c ReservedForPerf  : Ptr32 Void
#    +0xf80 ReservedForOle   : Ptr32 Void
#    +0xf84 WaitingOnLoaderLock : Uint4B
#    +0xf88 SavedPriorityState : Ptr32 Void
#    +0xf8c SoftPatchPtr1    : Uint4B
#    +0xf90 ThreadPoolData   : Ptr32 Void
#    +0xf94 TlsExpansionSlots : Ptr32 Ptr32 Void
#    +0xf98 ImpersonationLocale : Uint4B
#    +0xf9c IsImpersonating  : Uint4B
#    +0xfa0 NlsCache         : Ptr32 Void
#    +0xfa4 pShimData        : Ptr32 Void
#    +0xfa8 HeapVirtualAffinity : Uint4B
#    +0xfac CurrentTransactionHandle : Ptr32 Void
#    +0xfb0 ActiveFrame      : Ptr32 _TEB_ACTIVE_FRAME
#    +0xfb4 FlsData          : Ptr32 Void
#    +0xfb8 PreferredLanguages : Ptr32 Void
#    +0xfbc UserPrefLanguages : Ptr32 Void
#    +0xfc0 MergedPrefLanguages : Ptr32 Void
#    +0xfc4 MuiImpersonation : Uint4B
#    +0xfc8 CrossTebFlags    : Uint2B
#    +0xfc8 SpareCrossTebBits : Pos 0, 16 Bits
#    +0xfca SameTebFlags     : Uint2B
#    +0xfca DbgSafeThunkCall : Pos 0, 1 Bit
#    +0xfca DbgInDebugPrint  : Pos 1, 1 Bit
#    +0xfca DbgHasFiberData  : Pos 2, 1 Bit
#    +0xfca DbgSkipThreadAttach : Pos 3, 1 Bit
#    +0xfca DbgWerInShipAssertCode : Pos 4, 1 Bit
#    +0xfca DbgRanProcessInit : Pos 5, 1 Bit
#    +0xfca DbgClonedThread  : Pos 6, 1 Bit
#    +0xfca DbgSuppressDebugMsg : Pos 7, 1 Bit
#    +0xfca RtlDisableUserStackWalk : Pos 8, 1 Bit
#    +0xfca RtlExceptionAttached : Pos 9, 1 Bit
#    +0xfca SpareSameTebBits : Pos 10, 6 Bits
#    +0xfcc TxnScopeEnterCallback : Ptr32 Void
#    +0xfd0 TxnScopeExitCallback : Ptr32 Void
#    +0xfd4 TxnScopeContext  : Ptr32 Void
#    +0xfd8 LockCount        : Uint4B
#    +0xfdc ProcessRundown   : Uint4B
#    +0xfe0 LastSwitchTime   : Uint8B
#    +0xfe8 TotalSwitchOutTime : Uint8B
#    +0xff0 WaitReasonBitMap : _LARGE_INTEGER
class _TEB_2008(Structure):
    _pack_ = 8
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 HANDLE),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PVOID), # PPEB
        ("LastErrorValue",                  DWORD),
        ("CountOfOwnedCriticalSections",    DWORD),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  DWORD * 26),
        ("UserReserved",                    DWORD * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   DWORD),
        ("FpSoftwareStatusRegister",        DWORD),
        ("SystemReserved1",                 PVOID * 54),
        ("ExceptionCode",                   SDWORD),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes1",                     UCHAR * 36),
        ("TxFsContext",                     DWORD),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          HANDLE),
        ("GdiClientPID",                    DWORD),
        ("GdiClientTID",                    DWORD),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 DWORD * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     DWORD * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorMode",                   DWORD),
        ("Instrumentation",                 PVOID * 9),
        ("ActivityId",                      GUID),
        ("SubProcessTag",                   PVOID),
        ("EtwLocalData",                    PVOID),
        ("EtwTraceData",                    PVOID),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   DWORD),
        ("SpareBool0",                      BOOLEAN),
        ("SpareBool1",                      BOOLEAN),
        ("SpareBool2",                      BOOLEAN),
        ("IdealProcessor",                  UCHAR),
        ("GuaranteedStackBytes",            DWORD),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             DWORD),
        ("SavedPriorityState",              PVOID),
        ("SoftPatchPtr1",                   PVOID),
        ("ThreadPoolData",                  PVOID),
        ("TlsExpansionSlots",               PVOID), # Ptr32 Ptr32 Void
        ("ImpersonationLocale",             DWORD),
        ("IsImpersonating",                 BOOL),
        ("NlsCache",                        PVOID),
        ("pShimData",                       PVOID),
        ("HeapVirtualAffinity",             DWORD),
        ("CurrentTransactionHandle",        HANDLE),
        ("ActiveFrame",                     PVOID), # PTEB_ACTIVE_FRAME
        ("FlsData",                         PVOID),
        ("PreferredLanguages",              PVOID),
        ("UserPrefLanguages",               PVOID),
        ("MergedPrefLanguages",             PVOID),
        ("MuiImpersonation",                BOOL),
        ("CrossTebFlags",                   WORD),
        ("SameTebFlags",                    WORD),
        ("TxnScopeEnterCallback",           PVOID),
        ("TxnScopeExitCallback",            PVOID),
        ("TxnScopeContext",                 PVOID),
        ("LockCount",                       DWORD),
        ("ProcessRundown",                  DWORD),
        ("LastSwitchTime",                  QWORD),
        ("TotalSwitchOutTime",              QWORD),
        ("WaitReasonBitMap",                LONGLONG),  # LARGE_INTEGER
]

#    +0x000 NtTib            : _NT_TIB
#    +0x038 EnvironmentPointer : Ptr64 Void
#    +0x040 ClientId         : _CLIENT_ID
#    +0x050 ActiveRpcHandle  : Ptr64 Void
#    +0x058 ThreadLocalStoragePointer : Ptr64 Void
#    +0x060 ProcessEnvironmentBlock : Ptr64 _PEB
#    +0x068 LastErrorValue   : Uint4B
#    +0x06c CountOfOwnedCriticalSections : Uint4B
#    +0x070 CsrClientThread  : Ptr64 Void
#    +0x078 Win32ThreadInfo  : Ptr64 Void
#    +0x080 User32Reserved   : [26] Uint4B
#    +0x0e8 UserReserved     : [5] Uint4B
#    +0x100 WOW32Reserved    : Ptr64 Void
#    +0x108 CurrentLocale    : Uint4B
#    +0x10c FpSoftwareStatusRegister : Uint4B
#    +0x110 SystemReserved1  : [54] Ptr64 Void
#    +0x2c0 ExceptionCode    : Int4B
#    +0x2c8 ActivationContextStackPointer : Ptr64 _ACTIVATION_CONTEXT_STACK
#    +0x2d0 SpareBytes1      : [24] UChar
#    +0x2e8 TxFsContext      : Uint4B
#    +0x2f0 GdiTebBatch      : _GDI_TEB_BATCH
#    +0x7d8 RealClientId     : _CLIENT_ID
#    +0x7e8 GdiCachedProcessHandle : Ptr64 Void
#    +0x7f0 GdiClientPID     : Uint4B
#    +0x7f4 GdiClientTID     : Uint4B
#    +0x7f8 GdiThreadLocalInfo : Ptr64 Void
#    +0x800 Win32ClientInfo  : [62] Uint8B
#    +0x9f0 glDispatchTable  : [233] Ptr64 Void
#    +0x1138 glReserved1      : [29] Uint8B
#    +0x1220 glReserved2      : Ptr64 Void
#    +0x1228 glSectionInfo    : Ptr64 Void
#    +0x1230 glSection        : Ptr64 Void
#    +0x1238 glTable          : Ptr64 Void
#    +0x1240 glCurrentRC      : Ptr64 Void
#    +0x1248 glContext        : Ptr64 Void
#    +0x1250 LastStatusValue  : Uint4B
#    +0x1258 StaticUnicodeString : _UNICODE_STRING
#    +0x1268 StaticUnicodeBuffer : [261] Wchar
#    +0x1478 DeallocationStack : Ptr64 Void
#    +0x1480 TlsSlots         : [64] Ptr64 Void
#    +0x1680 TlsLinks         : _LIST_ENTRY
#    +0x1690 Vdm              : Ptr64 Void
#    +0x1698 ReservedForNtRpc : Ptr64 Void
#    +0x16a0 DbgSsReserved    : [2] Ptr64 Void
#    +0x16b0 HardErrorMode    : Uint4B
#    +0x16b8 Instrumentation  : [11] Ptr64 Void
#    +0x1710 ActivityId       : _GUID
#    +0x1720 SubProcessTag    : Ptr64 Void
#    +0x1728 EtwLocalData     : Ptr64 Void
#    +0x1730 EtwTraceData     : Ptr64 Void
#    +0x1738 WinSockData      : Ptr64 Void
#    +0x1740 GdiBatchCount    : Uint4B
#    +0x1744 SpareBool0       : UChar
#    +0x1745 SpareBool1       : UChar
#    +0x1746 SpareBool2       : UChar
#    +0x1747 IdealProcessor   : UChar
#    +0x1748 GuaranteedStackBytes : Uint4B
#    +0x1750 ReservedForPerf  : Ptr64 Void
#    +0x1758 ReservedForOle   : Ptr64 Void
#    +0x1760 WaitingOnLoaderLock : Uint4B
#    +0x1768 SavedPriorityState : Ptr64 Void
#    +0x1770 SoftPatchPtr1    : Uint8B
#    +0x1778 ThreadPoolData   : Ptr64 Void
#    +0x1780 TlsExpansionSlots : Ptr64 Ptr64 Void
#    +0x1788 DeallocationBStore : Ptr64 Void
#    +0x1790 BStoreLimit      : Ptr64 Void
#    +0x1798 ImpersonationLocale : Uint4B
#    +0x179c IsImpersonating  : Uint4B
#    +0x17a0 NlsCache         : Ptr64 Void
#    +0x17a8 pShimData        : Ptr64 Void
#    +0x17b0 HeapVirtualAffinity : Uint4B
#    +0x17b8 CurrentTransactionHandle : Ptr64 Void
#    +0x17c0 ActiveFrame      : Ptr64 _TEB_ACTIVE_FRAME
#    +0x17c8 FlsData          : Ptr64 Void
#    +0x17d0 PreferredLanguages : Ptr64 Void
#    +0x17d8 UserPrefLanguages : Ptr64 Void
#    +0x17e0 MergedPrefLanguages : Ptr64 Void
#    +0x17e8 MuiImpersonation : Uint4B
#    +0x17ec CrossTebFlags    : Uint2B
#    +0x17ec SpareCrossTebBits : Pos 0, 16 Bits
#    +0x17ee SameTebFlags     : Uint2B
#    +0x17ee DbgSafeThunkCall : Pos 0, 1 Bit
#    +0x17ee DbgInDebugPrint  : Pos 1, 1 Bit
#    +0x17ee DbgHasFiberData  : Pos 2, 1 Bit
#    +0x17ee DbgSkipThreadAttach : Pos 3, 1 Bit
#    +0x17ee DbgWerInShipAssertCode : Pos 4, 1 Bit
#    +0x17ee DbgRanProcessInit : Pos 5, 1 Bit
#    +0x17ee DbgClonedThread  : Pos 6, 1 Bit
#    +0x17ee DbgSuppressDebugMsg : Pos 7, 1 Bit
#    +0x17ee RtlDisableUserStackWalk : Pos 8, 1 Bit
#    +0x17ee RtlExceptionAttached : Pos 9, 1 Bit
#    +0x17ee SpareSameTebBits : Pos 10, 6 Bits
#    +0x17f0 TxnScopeEnterCallback : Ptr64 Void
#    +0x17f8 TxnScopeExitCallback : Ptr64 Void
#    +0x1800 TxnScopeContext  : Ptr64 Void
#    +0x1808 LockCount        : Uint4B
#    +0x180c ProcessRundown   : Uint4B
#    +0x1810 LastSwitchTime   : Uint8B
#    +0x1818 TotalSwitchOutTime : Uint8B
#    +0x1820 WaitReasonBitMap : _LARGE_INTEGER
class _TEB_2008_64(Structure):
    _pack_ = 8
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 HANDLE),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PVOID), # PPEB
        ("LastErrorValue",                  DWORD),
        ("CountOfOwnedCriticalSections",    DWORD),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  DWORD * 26),
        ("UserReserved",                    DWORD * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   DWORD),
        ("FpSoftwareStatusRegister",        DWORD),
        ("SystemReserved1",                 PVOID * 54),
        ("ExceptionCode",                   SDWORD),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes1",                     UCHAR * 24),
        ("TxFsContext",                     DWORD),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          HANDLE),
        ("GdiClientPID",                    DWORD),
        ("GdiClientTID",                    DWORD),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 QWORD * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     QWORD * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorMode",                   DWORD),
        ("Instrumentation",                 PVOID * 11),
        ("ActivityId",                      GUID),
        ("SubProcessTag",                   PVOID),
        ("EtwLocalData",                    PVOID),
        ("EtwTraceData",                    PVOID),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   DWORD),
        ("SpareBool0",                      BOOLEAN),
        ("SpareBool1",                      BOOLEAN),
        ("SpareBool2",                      BOOLEAN),
        ("IdealProcessor",                  UCHAR),
        ("GuaranteedStackBytes",            DWORD),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             DWORD),
        ("SavedPriorityState",              PVOID),
        ("SoftPatchPtr1",                   PVOID),
        ("ThreadPoolData",                  PVOID),
        ("TlsExpansionSlots",               PVOID), # Ptr64 Ptr64 Void
        ("DeallocationBStore",              PVOID),
        ("BStoreLimit",                     PVOID),
        ("ImpersonationLocale",             DWORD),
        ("IsImpersonating",                 BOOL),
        ("NlsCache",                        PVOID),
        ("pShimData",                       PVOID),
        ("HeapVirtualAffinity",             DWORD),
        ("CurrentTransactionHandle",        HANDLE),
        ("ActiveFrame",                     PVOID), # PTEB_ACTIVE_FRAME
        ("FlsData",                         PVOID),
        ("PreferredLanguages",              PVOID),
        ("UserPrefLanguages",               PVOID),
        ("MergedPrefLanguages",             PVOID),
        ("MuiImpersonation",                BOOL),
        ("CrossTebFlags",                   WORD),
        ("SameTebFlags",                    WORD),
        ("TxnScopeEnterCallback",           PVOID),
        ("TxnScopeExitCallback",            PVOID),
        ("TxnScopeContext",                 PVOID),
        ("LockCount",                       DWORD),
        ("ProcessRundown",                  DWORD),
        ("LastSwitchTime",                  QWORD),
        ("TotalSwitchOutTime",              QWORD),
        ("WaitReasonBitMap",                LONGLONG),  # LARGE_INTEGER
]

#    +0x000 NtTib            : _NT_TIB
#    +0x01c EnvironmentPointer : Ptr32 Void
#    +0x020 ClientId         : _CLIENT_ID
#    +0x028 ActiveRpcHandle  : Ptr32 Void
#    +0x02c ThreadLocalStoragePointer : Ptr32 Void
#    +0x030 ProcessEnvironmentBlock : Ptr32 _PEB
#    +0x034 LastErrorValue   : Uint4B
#    +0x038 CountOfOwnedCriticalSections : Uint4B
#    +0x03c CsrClientThread  : Ptr32 Void
#    +0x040 Win32ThreadInfo  : Ptr32 Void
#    +0x044 User32Reserved   : [26] Uint4B
#    +0x0ac UserReserved     : [5] Uint4B
#    +0x0c0 WOW32Reserved    : Ptr32 Void
#    +0x0c4 CurrentLocale    : Uint4B
#    +0x0c8 FpSoftwareStatusRegister : Uint4B
#    +0x0cc SystemReserved1  : [54] Ptr32 Void
#    +0x1a4 ExceptionCode    : Int4B
#    +0x1a8 ActivationContextStackPointer : Ptr32 _ACTIVATION_CONTEXT_STACK
#    +0x1ac SpareBytes       : [36] UChar
#    +0x1d0 TxFsContext      : Uint4B
#    +0x1d4 GdiTebBatch      : _GDI_TEB_BATCH
#    +0x6b4 RealClientId     : _CLIENT_ID
#    +0x6bc GdiCachedProcessHandle : Ptr32 Void
#    +0x6c0 GdiClientPID     : Uint4B
#    +0x6c4 GdiClientTID     : Uint4B
#    +0x6c8 GdiThreadLocalInfo : Ptr32 Void
#    +0x6cc Win32ClientInfo  : [62] Uint4B
#    +0x7c4 glDispatchTable  : [233] Ptr32 Void
#    +0xb68 glReserved1      : [29] Uint4B
#    +0xbdc glReserved2      : Ptr32 Void
#    +0xbe0 glSectionInfo    : Ptr32 Void
#    +0xbe4 glSection        : Ptr32 Void
#    +0xbe8 glTable          : Ptr32 Void
#    +0xbec glCurrentRC      : Ptr32 Void
#    +0xbf0 glContext        : Ptr32 Void
#    +0xbf4 LastStatusValue  : Uint4B
#    +0xbf8 StaticUnicodeString : _UNICODE_STRING
#    +0xc00 StaticUnicodeBuffer : [261] Wchar
#    +0xe0c DeallocationStack : Ptr32 Void
#    +0xe10 TlsSlots         : [64] Ptr32 Void
#    +0xf10 TlsLinks         : _LIST_ENTRY
#    +0xf18 Vdm              : Ptr32 Void
#    +0xf1c ReservedForNtRpc : Ptr32 Void
#    +0xf20 DbgSsReserved    : [2] Ptr32 Void
#    +0xf28 HardErrorMode    : Uint4B
#    +0xf2c Instrumentation  : [9] Ptr32 Void
#    +0xf50 ActivityId       : _GUID
#    +0xf60 SubProcessTag    : Ptr32 Void
#    +0xf64 EtwLocalData     : Ptr32 Void
#    +0xf68 EtwTraceData     : Ptr32 Void
#    +0xf6c WinSockData      : Ptr32 Void
#    +0xf70 GdiBatchCount    : Uint4B
#    +0xf74 CurrentIdealProcessor : _PROCESSOR_NUMBER
#    +0xf74 IdealProcessorValue : Uint4B
#    +0xf74 ReservedPad0     : UChar
#    +0xf75 ReservedPad1     : UChar
#    +0xf76 ReservedPad2     : UChar
#    +0xf77 IdealProcessor   : UChar
#    +0xf78 GuaranteedStackBytes : Uint4B
#    +0xf7c ReservedForPerf  : Ptr32 Void
#    +0xf80 ReservedForOle   : Ptr32 Void
#    +0xf84 WaitingOnLoaderLock : Uint4B
#    +0xf88 SavedPriorityState : Ptr32 Void
#    +0xf8c SoftPatchPtr1    : Uint4B
#    +0xf90 ThreadPoolData   : Ptr32 Void
#    +0xf94 TlsExpansionSlots : Ptr32 Ptr32 Void
#    +0xf98 MuiGeneration    : Uint4B
#    +0xf9c IsImpersonating  : Uint4B
#    +0xfa0 NlsCache         : Ptr32 Void
#    +0xfa4 pShimData        : Ptr32 Void
#    +0xfa8 HeapVirtualAffinity : Uint4B
#    +0xfac CurrentTransactionHandle : Ptr32 Void
#    +0xfb0 ActiveFrame      : Ptr32 _TEB_ACTIVE_FRAME
#    +0xfb4 FlsData          : Ptr32 Void
#    +0xfb8 PreferredLanguages : Ptr32 Void
#    +0xfbc UserPrefLanguages : Ptr32 Void
#    +0xfc0 MergedPrefLanguages : Ptr32 Void
#    +0xfc4 MuiImpersonation : Uint4B
#    +0xfc8 CrossTebFlags    : Uint2B
#    +0xfc8 SpareCrossTebBits : Pos 0, 16 Bits
#    +0xfca SameTebFlags     : Uint2B
#    +0xfca SafeThunkCall    : Pos 0, 1 Bit
#    +0xfca InDebugPrint     : Pos 1, 1 Bit
#    +0xfca HasFiberData     : Pos 2, 1 Bit
#    +0xfca SkipThreadAttach : Pos 3, 1 Bit
#    +0xfca WerInShipAssertCode : Pos 4, 1 Bit
#    +0xfca RanProcessInit   : Pos 5, 1 Bit
#    +0xfca ClonedThread     : Pos 6, 1 Bit
#    +0xfca SuppressDebugMsg : Pos 7, 1 Bit
#    +0xfca DisableUserStackWalk : Pos 8, 1 Bit
#    +0xfca RtlExceptionAttached : Pos 9, 1 Bit
#    +0xfca InitialThread    : Pos 10, 1 Bit
#    +0xfca SpareSameTebBits : Pos 11, 5 Bits
#    +0xfcc TxnScopeEnterCallback : Ptr32 Void
#    +0xfd0 TxnScopeExitCallback : Ptr32 Void
#    +0xfd4 TxnScopeContext  : Ptr32 Void
#    +0xfd8 LockCount        : Uint4B
#    +0xfdc SpareUlong0      : Uint4B
#    +0xfe0 ResourceRetValue : Ptr32 Void
class _TEB_2008_R2(Structure):
    _pack_ = 8
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 HANDLE),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PVOID), # PPEB
        ("LastErrorValue",                  DWORD),
        ("CountOfOwnedCriticalSections",    DWORD),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  DWORD * 26),
        ("UserReserved",                    DWORD * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   DWORD),
        ("FpSoftwareStatusRegister",        DWORD),
        ("SystemReserved1",                 PVOID * 54),
        ("ExceptionCode",                   SDWORD),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes",                      UCHAR * 36),
        ("TxFsContext",                     DWORD),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          HANDLE),
        ("GdiClientPID",                    DWORD),
        ("GdiClientTID",                    DWORD),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 DWORD * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     DWORD * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorMode",                   DWORD),
        ("Instrumentation",                 PVOID * 9),
        ("ActivityId",                      GUID),
        ("SubProcessTag",                   PVOID),
        ("EtwLocalData",                    PVOID),
        ("EtwTraceData",                    PVOID),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   DWORD),
        ("CurrentIdealProcessor",           PROCESSOR_NUMBER),
        ("IdealProcessorValue",             DWORD),
        ("ReservedPad0",                    UCHAR),
        ("ReservedPad1",                    UCHAR),
        ("ReservedPad2",                    UCHAR),
        ("IdealProcessor",                  UCHAR),
        ("GuaranteedStackBytes",            DWORD),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             DWORD),
        ("SavedPriorityState",              PVOID),
        ("SoftPatchPtr1",                   PVOID),
        ("ThreadPoolData",                  PVOID),
        ("TlsExpansionSlots",               PVOID), # Ptr32 Ptr32 Void
        ("MuiGeneration",                   DWORD),
        ("IsImpersonating",                 BOOL),
        ("NlsCache",                        PVOID),
        ("pShimData",                       PVOID),
        ("HeapVirtualAffinity",             DWORD),
        ("CurrentTransactionHandle",        HANDLE),
        ("ActiveFrame",                     PVOID), # PTEB_ACTIVE_FRAME
        ("FlsData",                         PVOID),
        ("PreferredLanguages",              PVOID),
        ("UserPrefLanguages",               PVOID),
        ("MergedPrefLanguages",             PVOID),
        ("MuiImpersonation",                BOOL),
        ("CrossTebFlags",                   WORD),
        ("SameTebFlags",                    WORD),
        ("TxnScopeEnterCallback",           PVOID),
        ("TxnScopeExitCallback",            PVOID),
        ("TxnScopeContext",                 PVOID),
        ("LockCount",                       DWORD),
        ("SpareUlong0",                     ULONG),
        ("ResourceRetValue",                PVOID),
]

#    +0x000 NtTib            : _NT_TIB
#    +0x038 EnvironmentPointer : Ptr64 Void
#    +0x040 ClientId         : _CLIENT_ID
#    +0x050 ActiveRpcHandle  : Ptr64 Void
#    +0x058 ThreadLocalStoragePointer : Ptr64 Void
#    +0x060 ProcessEnvironmentBlock : Ptr64 _PEB
#    +0x068 LastErrorValue   : Uint4B
#    +0x06c CountOfOwnedCriticalSections : Uint4B
#    +0x070 CsrClientThread  : Ptr64 Void
#    +0x078 Win32ThreadInfo  : Ptr64 Void
#    +0x080 User32Reserved   : [26] Uint4B
#    +0x0e8 UserReserved     : [5] Uint4B
#    +0x100 WOW32Reserved    : Ptr64 Void
#    +0x108 CurrentLocale    : Uint4B
#    +0x10c FpSoftwareStatusRegister : Uint4B
#    +0x110 SystemReserved1  : [54] Ptr64 Void
#    +0x2c0 ExceptionCode    : Int4B
#    +0x2c8 ActivationContextStackPointer : Ptr64 _ACTIVATION_CONTEXT_STACK
#    +0x2d0 SpareBytes       : [24] UChar
#    +0x2e8 TxFsContext      : Uint4B
#    +0x2f0 GdiTebBatch      : _GDI_TEB_BATCH
#    +0x7d8 RealClientId     : _CLIENT_ID
#    +0x7e8 GdiCachedProcessHandle : Ptr64 Void
#    +0x7f0 GdiClientPID     : Uint4B
#    +0x7f4 GdiClientTID     : Uint4B
#    +0x7f8 GdiThreadLocalInfo : Ptr64 Void
#    +0x800 Win32ClientInfo  : [62] Uint8B
#    +0x9f0 glDispatchTable  : [233] Ptr64 Void
#    +0x1138 glReserved1      : [29] Uint8B
#    +0x1220 glReserved2      : Ptr64 Void
#    +0x1228 glSectionInfo    : Ptr64 Void
#    +0x1230 glSection        : Ptr64 Void
#    +0x1238 glTable          : Ptr64 Void
#    +0x1240 glCurrentRC      : Ptr64 Void
#    +0x1248 glContext        : Ptr64 Void
#    +0x1250 LastStatusValue  : Uint4B
#    +0x1258 StaticUnicodeString : _UNICODE_STRING
#    +0x1268 StaticUnicodeBuffer : [261] Wchar
#    +0x1478 DeallocationStack : Ptr64 Void
#    +0x1480 TlsSlots         : [64] Ptr64 Void
#    +0x1680 TlsLinks         : _LIST_ENTRY
#    +0x1690 Vdm              : Ptr64 Void
#    +0x1698 ReservedForNtRpc : Ptr64 Void
#    +0x16a0 DbgSsReserved    : [2] Ptr64 Void
#    +0x16b0 HardErrorMode    : Uint4B
#    +0x16b8 Instrumentation  : [11] Ptr64 Void
#    +0x1710 ActivityId       : _GUID
#    +0x1720 SubProcessTag    : Ptr64 Void
#    +0x1728 EtwLocalData     : Ptr64 Void
#    +0x1730 EtwTraceData     : Ptr64 Void
#    +0x1738 WinSockData      : Ptr64 Void
#    +0x1740 GdiBatchCount    : Uint4B
#    +0x1744 CurrentIdealProcessor : _PROCESSOR_NUMBER
#    +0x1744 IdealProcessorValue : Uint4B
#    +0x1744 ReservedPad0     : UChar
#    +0x1745 ReservedPad1     : UChar
#    +0x1746 ReservedPad2     : UChar
#    +0x1747 IdealProcessor   : UChar
#    +0x1748 GuaranteedStackBytes : Uint4B
#    +0x1750 ReservedForPerf  : Ptr64 Void
#    +0x1758 ReservedForOle   : Ptr64 Void
#    +0x1760 WaitingOnLoaderLock : Uint4B
#    +0x1768 SavedPriorityState : Ptr64 Void
#    +0x1770 SoftPatchPtr1    : Uint8B
#    +0x1778 ThreadPoolData   : Ptr64 Void
#    +0x1780 TlsExpansionSlots : Ptr64 Ptr64 Void
#    +0x1788 DeallocationBStore : Ptr64 Void
#    +0x1790 BStoreLimit      : Ptr64 Void
#    +0x1798 MuiGeneration    : Uint4B
#    +0x179c IsImpersonating  : Uint4B
#    +0x17a0 NlsCache         : Ptr64 Void
#    +0x17a8 pShimData        : Ptr64 Void
#    +0x17b0 HeapVirtualAffinity : Uint4B
#    +0x17b8 CurrentTransactionHandle : Ptr64 Void
#    +0x17c0 ActiveFrame      : Ptr64 _TEB_ACTIVE_FRAME
#    +0x17c8 FlsData          : Ptr64 Void
#    +0x17d0 PreferredLanguages : Ptr64 Void
#    +0x17d8 UserPrefLanguages : Ptr64 Void
#    +0x17e0 MergedPrefLanguages : Ptr64 Void
#    +0x17e8 MuiImpersonation : Uint4B
#    +0x17ec CrossTebFlags    : Uint2B
#    +0x17ec SpareCrossTebBits : Pos 0, 16 Bits
#    +0x17ee SameTebFlags     : Uint2B
#    +0x17ee SafeThunkCall    : Pos 0, 1 Bit
#    +0x17ee InDebugPrint     : Pos 1, 1 Bit
#    +0x17ee HasFiberData     : Pos 2, 1 Bit
#    +0x17ee SkipThreadAttach : Pos 3, 1 Bit
#    +0x17ee WerInShipAssertCode : Pos 4, 1 Bit
#    +0x17ee RanProcessInit   : Pos 5, 1 Bit
#    +0x17ee ClonedThread     : Pos 6, 1 Bit
#    +0x17ee SuppressDebugMsg : Pos 7, 1 Bit
#    +0x17ee DisableUserStackWalk : Pos 8, 1 Bit
#    +0x17ee RtlExceptionAttached : Pos 9, 1 Bit
#    +0x17ee InitialThread    : Pos 10, 1 Bit
#    +0x17ee SpareSameTebBits : Pos 11, 5 Bits
#    +0x17f0 TxnScopeEnterCallback : Ptr64 Void
#    +0x17f8 TxnScopeExitCallback : Ptr64 Void
#    +0x1800 TxnScopeContext  : Ptr64 Void
#    +0x1808 LockCount        : Uint4B
#    +0x180c SpareUlong0      : Uint4B
#    +0x1810 ResourceRetValue : Ptr64 Void
class _TEB_2008_R2_64(Structure):
    _pack_ = 8
    _fields_ = [
        ("NtTib",                           NT_TIB),
        ("EnvironmentPointer",              PVOID),
        ("ClientId",                        CLIENT_ID),
        ("ActiveRpcHandle",                 HANDLE),
        ("ThreadLocalStoragePointer",       PVOID),
        ("ProcessEnvironmentBlock",         PVOID), # PPEB
        ("LastErrorValue",                  DWORD),
        ("CountOfOwnedCriticalSections",    DWORD),
        ("CsrClientThread",                 PVOID),
        ("Win32ThreadInfo",                 PVOID),
        ("User32Reserved",                  DWORD * 26),
        ("UserReserved",                    DWORD * 5),
        ("WOW32Reserved",                   PVOID), # ptr to wow64cpu!X86SwitchTo64BitMode
        ("CurrentLocale",                   DWORD),
        ("FpSoftwareStatusRegister",        DWORD),
        ("SystemReserved1",                 PVOID * 54),
        ("ExceptionCode",                   SDWORD),
        ("ActivationContextStackPointer",   PVOID), # PACTIVATION_CONTEXT_STACK
        ("SpareBytes",                      UCHAR * 24),
        ("TxFsContext",                     DWORD),
        ("GdiTebBatch",                     GDI_TEB_BATCH),
        ("RealClientId",                    CLIENT_ID),
        ("GdiCachedProcessHandle",          HANDLE),
        ("GdiClientPID",                    DWORD),
        ("GdiClientTID",                    DWORD),
        ("GdiThreadLocalInfo",              PVOID),
        ("Win32ClientInfo",                 DWORD * 62),
        ("glDispatchTable",                 PVOID * 233),
        ("glReserved1",                     QWORD * 29),
        ("glReserved2",                     PVOID),
        ("glSectionInfo",                   PVOID),
        ("glSection",                       PVOID),
        ("glTable",                         PVOID),
        ("glCurrentRC",                     PVOID),
        ("glContext",                       PVOID),
        ("LastStatusValue",                 NTSTATUS),
        ("StaticUnicodeString",             UNICODE_STRING),
        ("StaticUnicodeBuffer",             WCHAR * 261),
        ("DeallocationStack",               PVOID),
        ("TlsSlots",                        PVOID * 64),
        ("TlsLinks",                        LIST_ENTRY),
        ("Vdm",                             PVOID),
        ("ReservedForNtRpc",                PVOID),
        ("DbgSsReserved",                   PVOID * 2),
        ("HardErrorMode",                   DWORD),
        ("Instrumentation",                 PVOID * 11),
        ("ActivityId",                      GUID),
        ("SubProcessTag",                   PVOID),
        ("EtwLocalData",                    PVOID),
        ("EtwTraceData",                    PVOID),
        ("WinSockData",                     PVOID),
        ("GdiBatchCount",                   DWORD),
        ("CurrentIdealProcessor",           PROCESSOR_NUMBER),
        ("IdealProcessorValue",             DWORD),
        ("ReservedPad0",                    UCHAR),
        ("ReservedPad1",                    UCHAR),
        ("ReservedPad2",                    UCHAR),
        ("IdealProcessor",                  UCHAR),
        ("GuaranteedStackBytes",            DWORD),
        ("ReservedForPerf",                 PVOID),
        ("ReservedForOle",                  PVOID),
        ("WaitingOnLoaderLock",             DWORD),
        ("SavedPriorityState",              PVOID),
        ("SoftPatchPtr1",                   PVOID),
        ("ThreadPoolData",                  PVOID),
        ("TlsExpansionSlots",               PVOID), # Ptr64 Ptr64 Void
        ("DeallocationBStore",              PVOID),
        ("BStoreLimit",                     PVOID),
        ("MuiGeneration",                   DWORD),
        ("IsImpersonating",                 BOOL),
        ("NlsCache",                        PVOID),
        ("pShimData",                       PVOID),
        ("HeapVirtualAffinity",             DWORD),
        ("CurrentTransactionHandle",        HANDLE),
        ("ActiveFrame",                     PVOID), # PTEB_ACTIVE_FRAME
        ("FlsData",                         PVOID),
        ("PreferredLanguages",              PVOID),
        ("UserPrefLanguages",               PVOID),
        ("MergedPrefLanguages",             PVOID),
        ("MuiImpersonation",                BOOL),
        ("CrossTebFlags",                   WORD),
        ("SameTebFlags",                    WORD),
        ("TxnScopeEnterCallback",           PVOID),
        ("TxnScopeExitCallback",            PVOID),
        ("TxnScopeContext",                 PVOID),
        ("LockCount",                       DWORD),
        ("SpareUlong0",                     ULONG),
        ("ResourceRetValue",                PVOID),
]

_TEB_Vista      = _TEB_2008
_TEB_Vista_64   = _TEB_2008_64
_TEB_W7         = _TEB_2008_R2
_TEB_W7_64      = _TEB_2008_R2_64

# Use the correct TEB structure definition.
# Defaults to the latest Windows version.
class TEB(Structure):
    _pack_ = 8
    if os == 'Windows NT':
        _pack_   = _TEB_NT._pack_
        _fields_ = _TEB_NT._fields_
    elif os == 'Windows 2000':
        _pack_   = _TEB_2000._pack_
        _fields_ = _TEB_2000._fields_
    elif os == 'Windows XP':
        _fields_ = _TEB_XP._fields_
    elif os == 'Windows XP (64 bits)':
        _fields_ = _TEB_XP_64._fields_
    elif os == 'Windows 2003':
        _fields_ = _TEB_2003._fields_
    elif os == 'Windows 2003 (64 bits)':
        _fields_ = _TEB_2003_64._fields_
    elif os == 'Windows 2008':
        _fields_ = _TEB_2008._fields_
    elif os == 'Windows 2008 (64 bits)':
        _fields_ = _TEB_2008_64._fields_
    elif os == 'Windows 2003 R2':
        _fields_ = _TEB_2003_R2._fields_
    elif os == 'Windows 2003 R2 (64 bits)':
        _fields_ = _TEB_2003_R2_64._fields_
    elif os == 'Windows 2008 R2':
        _fields_ = _TEB_2008_R2._fields_
    elif os == 'Windows 2008 R2 (64 bits)':
        _fields_ = _TEB_2008_R2_64._fields_
    elif os == 'Windows Vista':
        _fields_ = _TEB_Vista._fields_
    elif os == 'Windows Vista (64 bits)':
        _fields_ = _TEB_Vista_64._fields_
    elif os == 'Windows 7':
        _fields_ = _TEB_W7._fields_
    elif os == 'Windows 7 (64 bits)':
        _fields_ = _TEB_W7_64._fields_
    elif sizeof(SIZE_T) == sizeof(DWORD):
        _fields_ = _TEB_W7._fields_
    else:
        _fields_ = _TEB_W7_64._fields_
PTEB = POINTER(TEB)

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
