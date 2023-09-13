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
Detect the current architecture and operating system.

Some functions here are really from kernel32.dll, others from version.dll.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

#--- NTDDI version ------------------------------------------------------------

NTDDI_WIN8      = 0x06020000
NTDDI_WIN7SP1   = 0x06010100
NTDDI_WIN7      = 0x06010000
NTDDI_WS08      = 0x06000100
NTDDI_VISTASP1  = 0x06000100
NTDDI_VISTA     = 0x06000000
NTDDI_LONGHORN  = NTDDI_VISTA
NTDDI_WS03SP2   = 0x05020200
NTDDI_WS03SP1   = 0x05020100
NTDDI_WS03      = 0x05020000
NTDDI_WINXPSP3  = 0x05010300
NTDDI_WINXPSP2  = 0x05010200
NTDDI_WINXPSP1  = 0x05010100
NTDDI_WINXP     = 0x05010000
NTDDI_WIN2KSP4  = 0x05000400
NTDDI_WIN2KSP3  = 0x05000300
NTDDI_WIN2KSP2  = 0x05000200
NTDDI_WIN2KSP1  = 0x05000100
NTDDI_WIN2K     = 0x05000000
NTDDI_WINNT4    = 0x04000000

OSVERSION_MASK  = 0xFFFF0000
SPVERSION_MASK  = 0x0000FF00
SUBVERSION_MASK = 0x000000FF

#--- OSVERSIONINFO and OSVERSIONINFOEX structures and constants ---------------

VER_PLATFORM_WIN32s                 = 0
VER_PLATFORM_WIN32_WINDOWS          = 1
VER_PLATFORM_WIN32_NT               = 2

VER_SUITE_BACKOFFICE                = 0x00000004
VER_SUITE_BLADE                     = 0x00000400
VER_SUITE_COMPUTE_SERVER            = 0x00004000
VER_SUITE_DATACENTER                = 0x00000080
VER_SUITE_ENTERPRISE                = 0x00000002
VER_SUITE_EMBEDDEDNT                = 0x00000040
VER_SUITE_PERSONAL                  = 0x00000200
VER_SUITE_SINGLEUSERTS              = 0x00000100
VER_SUITE_SMALLBUSINESS             = 0x00000001
VER_SUITE_SMALLBUSINESS_RESTRICTED  = 0x00000020
VER_SUITE_STORAGE_SERVER            = 0x00002000
VER_SUITE_TERMINAL                  = 0x00000010
VER_SUITE_WH_SERVER                 = 0x00008000

VER_NT_DOMAIN_CONTROLLER            = 0x0000002
VER_NT_SERVER                       = 0x0000003
VER_NT_WORKSTATION                  = 0x0000001

VER_BUILDNUMBER                     = 0x0000004
VER_MAJORVERSION                    = 0x0000002
VER_MINORVERSION                    = 0x0000001
VER_PLATFORMID                      = 0x0000008
VER_PRODUCT_TYPE                    = 0x0000080
VER_SERVICEPACKMAJOR                = 0x0000020
VER_SERVICEPACKMINOR                = 0x0000010
VER_SUITENAME                       = 0x0000040

VER_EQUAL                           = 1
VER_GREATER                         = 2
VER_GREATER_EQUAL                   = 3
VER_LESS                            = 4
VER_LESS_EQUAL                      = 5
VER_AND                             = 6
VER_OR                              = 7

# typedef struct _OSVERSIONINFO {
#   DWORD dwOSVersionInfoSize;
#   DWORD dwMajorVersion;
#   DWORD dwMinorVersion;
#   DWORD dwBuildNumber;
#   DWORD dwPlatformId;
#   TCHAR szCSDVersion[128];
# }OSVERSIONINFO;
class OSVERSIONINFOA(Structure):
    _fields_ = [
        ("dwOSVersionInfoSize", DWORD),
        ("dwMajorVersion",      DWORD),
        ("dwMinorVersion",      DWORD),
        ("dwBuildNumber",       DWORD),
        ("dwPlatformId",        DWORD),
        ("szCSDVersion",        CHAR * 128),
    ]
class OSVERSIONINFOW(Structure):
    _fields_ = [
        ("dwOSVersionInfoSize", DWORD),
        ("dwMajorVersion",      DWORD),
        ("dwMinorVersion",      DWORD),
        ("dwBuildNumber",       DWORD),
        ("dwPlatformId",        DWORD),
        ("szCSDVersion",        WCHAR * 128),
    ]

# typedef struct _OSVERSIONINFOEX {
#   DWORD dwOSVersionInfoSize;
#   DWORD dwMajorVersion;
#   DWORD dwMinorVersion;
#   DWORD dwBuildNumber;
#   DWORD dwPlatformId;
#   TCHAR szCSDVersion[128];
#   WORD  wServicePackMajor;
#   WORD  wServicePackMinor;
#   WORD  wSuiteMask;
#   BYTE  wProductType;
#   BYTE  wReserved;
# }OSVERSIONINFOEX, *POSVERSIONINFOEX, *LPOSVERSIONINFOEX;
class OSVERSIONINFOEXA(Structure):
    _fields_ = [
        ("dwOSVersionInfoSize", DWORD),
        ("dwMajorVersion",      DWORD),
        ("dwMinorVersion",      DWORD),
        ("dwBuildNumber",       DWORD),
        ("dwPlatformId",        DWORD),
        ("szCSDVersion",        CHAR * 128),
        ("wServicePackMajor",   WORD),
        ("wServicePackMinor",   WORD),
        ("wSuiteMask",          WORD),
        ("wProductType",        BYTE),
        ("wReserved",           BYTE),
    ]
class OSVERSIONINFOEXW(Structure):
    _fields_ = [
        ("dwOSVersionInfoSize", DWORD),
        ("dwMajorVersion",      DWORD),
        ("dwMinorVersion",      DWORD),
        ("dwBuildNumber",       DWORD),
        ("dwPlatformId",        DWORD),
        ("szCSDVersion",        WCHAR * 128),
        ("wServicePackMajor",   WORD),
        ("wServicePackMinor",   WORD),
        ("wSuiteMask",          WORD),
        ("wProductType",        BYTE),
        ("wReserved",           BYTE),
    ]

LPOSVERSIONINFOA    = POINTER(OSVERSIONINFOA)
LPOSVERSIONINFOW    = POINTER(OSVERSIONINFOW)
LPOSVERSIONINFOEXA  = POINTER(OSVERSIONINFOEXA)
LPOSVERSIONINFOEXW  = POINTER(OSVERSIONINFOEXW)
POSVERSIONINFOA     = LPOSVERSIONINFOA
POSVERSIONINFOW     = LPOSVERSIONINFOW
POSVERSIONINFOEXA   = LPOSVERSIONINFOEXA
POSVERSIONINFOEXW   = LPOSVERSIONINFOA

#--- GetSystemMetrics constants -----------------------------------------------

SM_CXSCREEN             = 0
SM_CYSCREEN             = 1
SM_CXVSCROLL            = 2
SM_CYHSCROLL            = 3
SM_CYCAPTION            = 4
SM_CXBORDER             = 5
SM_CYBORDER             = 6
SM_CXDLGFRAME           = 7
SM_CYDLGFRAME           = 8
SM_CYVTHUMB             = 9
SM_CXHTHUMB             = 10
SM_CXICON               = 11
SM_CYICON               = 12
SM_CXCURSOR             = 13
SM_CYCURSOR             = 14
SM_CYMENU               = 15
SM_CXFULLSCREEN         = 16
SM_CYFULLSCREEN         = 17
SM_CYKANJIWINDOW        = 18
SM_MOUSEPRESENT         = 19
SM_CYVSCROLL            = 20
SM_CXHSCROLL            = 21
SM_DEBUG                = 22
SM_SWAPBUTTON           = 23
SM_RESERVED1            = 24
SM_RESERVED2            = 25
SM_RESERVED3            = 26
SM_RESERVED4            = 27
SM_CXMIN                = 28
SM_CYMIN                = 29
SM_CXSIZE               = 30
SM_CYSIZE               = 31
SM_CXFRAME              = 32
SM_CYFRAME              = 33
SM_CXMINTRACK           = 34
SM_CYMINTRACK           = 35
SM_CXDOUBLECLK          = 36
SM_CYDOUBLECLK          = 37
SM_CXICONSPACING        = 38
SM_CYICONSPACING        = 39
SM_MENUDROPALIGNMENT    = 40
SM_PENWINDOWS           = 41
SM_DBCSENABLED          = 42
SM_CMOUSEBUTTONS        = 43

SM_CXFIXEDFRAME         = SM_CXDLGFRAME     # ;win40 name change
SM_CYFIXEDFRAME         = SM_CYDLGFRAME     # ;win40 name change
SM_CXSIZEFRAME          = SM_CXFRAME        # ;win40 name change
SM_CYSIZEFRAME          = SM_CYFRAME        # ;win40 name change

SM_SECURE               = 44
SM_CXEDGE               = 45
SM_CYEDGE               = 46
SM_CXMINSPACING         = 47
SM_CYMINSPACING         = 48
SM_CXSMICON             = 49
SM_CYSMICON             = 50
SM_CYSMCAPTION          = 51
SM_CXSMSIZE             = 52
SM_CYSMSIZE             = 53
SM_CXMENUSIZE           = 54
SM_CYMENUSIZE           = 55
SM_ARRANGE              = 56
SM_CXMINIMIZED          = 57
SM_CYMINIMIZED          = 58
SM_CXMAXTRACK           = 59
SM_CYMAXTRACK           = 60
SM_CXMAXIMIZED          = 61
SM_CYMAXIMIZED          = 62
SM_NETWORK              = 63
SM_CLEANBOOT            = 67
SM_CXDRAG               = 68
SM_CYDRAG               = 69
SM_SHOWSOUNDS           = 70
SM_CXMENUCHECK          = 71  # Use instead of GetMenuCheckMarkDimensions()!
SM_CYMENUCHECK          = 72
SM_SLOWMACHINE          = 73
SM_MIDEASTENABLED       = 74
SM_MOUSEWHEELPRESENT    = 75
SM_XVIRTUALSCREEN       = 76
SM_YVIRTUALSCREEN       = 77
SM_CXVIRTUALSCREEN      = 78
SM_CYVIRTUALSCREEN      = 79
SM_CMONITORS            = 80
SM_SAMEDISPLAYFORMAT    = 81
SM_IMMENABLED           = 82
SM_CXFOCUSBORDER        = 83
SM_CYFOCUSBORDER        = 84
SM_TABLETPC             = 86
SM_MEDIACENTER          = 87
SM_STARTER              = 88
SM_SERVERR2             = 89
SM_MOUSEHORIZONTALWHEELPRESENT = 91
SM_CXPADDEDBORDER       = 92

SM_CMETRICS             = 93

SM_REMOTESESSION        = 0x1000
SM_SHUTTINGDOWN         = 0x2000
SM_REMOTECONTROL        = 0x2001
SM_CARETBLINKINGENABLED = 0x2002

#--- SYSTEM_INFO structure, GetSystemInfo() and GetNativeSystemInfo() ---------

# Values used by Wine
# Documented values at MSDN are marked with an asterisk
PROCESSOR_ARCHITECTURE_UNKNOWN        = 0xFFFF; # Unknown architecture.
PROCESSOR_ARCHITECTURE_INTEL          = 0       # x86 (AMD or Intel) *
PROCESSOR_ARCHITECTURE_MIPS           = 1       # MIPS
PROCESSOR_ARCHITECTURE_ALPHA          = 2       # Alpha
PROCESSOR_ARCHITECTURE_PPC            = 3       # Power PC
PROCESSOR_ARCHITECTURE_SHX            = 4       # SHX
PROCESSOR_ARCHITECTURE_ARM            = 5       # ARM
PROCESSOR_ARCHITECTURE_IA64           = 6       # Intel Itanium *
PROCESSOR_ARCHITECTURE_ALPHA64        = 7       # Alpha64
PROCESSOR_ARCHITECTURE_MSIL           = 8       # MSIL
PROCESSOR_ARCHITECTURE_AMD64          = 9       # x64 (AMD or Intel) *
PROCESSOR_ARCHITECTURE_IA32_ON_WIN64  = 10      # IA32 on Win64
PROCESSOR_ARCHITECTURE_SPARC          = 20      # Sparc (Wine)

# Values used by Wine
# PROCESSOR_OPTIL value found at http://code.google.com/p/ddab-lib/
# Documented values at MSDN are marked with an asterisk
PROCESSOR_INTEL_386     = 386    # Intel i386 *
PROCESSOR_INTEL_486     = 486    # Intel i486 *
PROCESSOR_INTEL_PENTIUM = 586    # Intel Pentium *
PROCESSOR_INTEL_IA64    = 2200   # Intel IA64 (Itanium) *
PROCESSOR_AMD_X8664     = 8664   # AMD X86 64 *
PROCESSOR_MIPS_R4000    = 4000   # MIPS R4000, R4101, R3910
PROCESSOR_ALPHA_21064   = 21064  # Alpha 210 64
PROCESSOR_PPC_601       = 601    # PPC 601
PROCESSOR_PPC_603       = 603    # PPC 603
PROCESSOR_PPC_604       = 604    # PPC 604
PROCESSOR_PPC_620       = 620    # PPC 620
PROCESSOR_HITACHI_SH3   = 10003  # Hitachi SH3 (Windows CE)
PROCESSOR_HITACHI_SH3E  = 10004  # Hitachi SH3E (Windows CE)
PROCESSOR_HITACHI_SH4   = 10005  # Hitachi SH4 (Windows CE)
PROCESSOR_MOTOROLA_821  = 821    # Motorola 821 (Windows CE)
PROCESSOR_SHx_SH3       = 103    # SHx SH3 (Windows CE)
PROCESSOR_SHx_SH4       = 104    # SHx SH4 (Windows CE)
PROCESSOR_STRONGARM     = 2577   # StrongARM (Windows CE)
PROCESSOR_ARM720        = 1824   # ARM 720 (Windows CE)
PROCESSOR_ARM820        = 2080   # ARM 820 (Windows CE)
PROCESSOR_ARM920        = 2336   # ARM 920 (Windows CE)
PROCESSOR_ARM_7TDMI     = 70001  # ARM 7TDMI (Windows CE)
PROCESSOR_OPTIL         = 0x494F # MSIL

# typedef struct _SYSTEM_INFO {
#   union {
#     DWORD dwOemId;
#     struct {
#       WORD wProcessorArchitecture;
#       WORD wReserved;
#     } ;
#   }     ;
#   DWORD     dwPageSize;
#   LPVOID    lpMinimumApplicationAddress;
#   LPVOID    lpMaximumApplicationAddress;
#   DWORD_PTR dwActiveProcessorMask;
#   DWORD     dwNumberOfProcessors;
#   DWORD     dwProcessorType;
#   DWORD     dwAllocationGranularity;
#   WORD      wProcessorLevel;
#   WORD      wProcessorRevision;
# } SYSTEM_INFO;

class _SYSTEM_INFO_OEM_ID_STRUCT(Structure):
    _fields_ = [
        ("wProcessorArchitecture",  WORD),
        ("wReserved",               WORD),
]

class _SYSTEM_INFO_OEM_ID(Union):
    _fields_ = [
        ("dwOemId",  DWORD),
        ("w",        _SYSTEM_INFO_OEM_ID_STRUCT),
]

class SYSTEM_INFO(Structure):
    _fields_ = [
        ("id",                              _SYSTEM_INFO_OEM_ID),
        ("dwPageSize",                      DWORD),
        ("lpMinimumApplicationAddress",     LPVOID),
        ("lpMaximumApplicationAddress",     LPVOID),
        ("dwActiveProcessorMask",           DWORD_PTR),
        ("dwNumberOfProcessors",            DWORD),
        ("dwProcessorType",                 DWORD),
        ("dwAllocationGranularity",         DWORD),
        ("wProcessorLevel",                 WORD),
        ("wProcessorRevision",              WORD),
    ]

    def __get_dwOemId(self):
        return self.id.dwOemId
    def __set_dwOemId(self, value):
        self.id.dwOemId = value
    dwOemId = property(__get_dwOemId, __set_dwOemId)

    def __get_wProcessorArchitecture(self):
        return self.id.w.wProcessorArchitecture
    def __set_wProcessorArchitecture(self, value):
        self.id.w.wProcessorArchitecture = value
    wProcessorArchitecture = property(__get_wProcessorArchitecture, __set_wProcessorArchitecture)

LPSYSTEM_INFO = ctypes.POINTER(SYSTEM_INFO)

# void WINAPI GetSystemInfo(
#   __out  LPSYSTEM_INFO lpSystemInfo
# );
def GetSystemInfo():
    _GetSystemInfo = windll.kernel32.GetSystemInfo
    _GetSystemInfo.argtypes = [LPSYSTEM_INFO]
    _GetSystemInfo.restype  = None

    sysinfo = SYSTEM_INFO()
    _GetSystemInfo(byref(sysinfo))
    return sysinfo

# void WINAPI GetNativeSystemInfo(
#   __out  LPSYSTEM_INFO lpSystemInfo
# );
def GetNativeSystemInfo():
    _GetNativeSystemInfo = windll.kernel32.GetNativeSystemInfo
    _GetNativeSystemInfo.argtypes = [LPSYSTEM_INFO]
    _GetNativeSystemInfo.restype  = None

    sysinfo = SYSTEM_INFO()
    _GetNativeSystemInfo(byref(sysinfo))
    return sysinfo

# int WINAPI GetSystemMetrics(
#   __in  int nIndex
# );
def GetSystemMetrics(nIndex):
    _GetSystemMetrics = windll.user32.GetSystemMetrics
    _GetSystemMetrics.argtypes = [ctypes.c_int]
    _GetSystemMetrics.restype  = ctypes.c_int
    return _GetSystemMetrics(nIndex)

# SIZE_T WINAPI GetLargePageMinimum(void);
def GetLargePageMinimum():
    _GetLargePageMinimum = windll.user32.GetLargePageMinimum
    _GetLargePageMinimum.argtypes = []
    _GetLargePageMinimum.restype  = SIZE_T
    return _GetLargePageMinimum()

# HANDLE WINAPI GetCurrentProcess(void);
def GetCurrentProcess():
##    return 0xFFFFFFFFFFFFFFFFL
    _GetCurrentProcess = windll.kernel32.GetCurrentProcess
    _GetCurrentProcess.argtypes = []
    _GetCurrentProcess.restype  = HANDLE
    return _GetCurrentProcess()

# HANDLE WINAPI GetCurrentThread(void);
def GetCurrentThread():
##    return 0xFFFFFFFFFFFFFFFEL
    _GetCurrentThread = windll.kernel32.GetCurrentThread
    _GetCurrentThread.argtypes = []
    _GetCurrentThread.restype  = HANDLE
    return _GetCurrentThread()

# BOOL WINAPI IsWow64Process(
#   __in   HANDLE hProcess,
#   __out  PBOOL Wow64Process
# );
def IsWow64Process(hProcess):
    _IsWow64Process = windll.kernel32.IsWow64Process
    _IsWow64Process.argtypes = [HANDLE, PBOOL]
    _IsWow64Process.restype  = bool
    _IsWow64Process.errcheck = RaiseIfZero

    Wow64Process = BOOL(FALSE)
    _IsWow64Process(hProcess, byref(Wow64Process))
    return bool(Wow64Process)

# DWORD WINAPI GetVersion(void);
def GetVersion():
    _GetVersion = windll.kernel32.GetVersion
    _GetVersion.argtypes = []
    _GetVersion.restype  = DWORD
    _GetVersion.errcheck = RaiseIfZero

    # See the example code here:
    # http://msdn.microsoft.com/en-us/library/ms724439(VS.85).aspx

    dwVersion       = _GetVersion()
    dwMajorVersion  = dwVersion & 0x000000FF
    dwMinorVersion  = (dwVersion & 0x0000FF00) >> 8
    if (dwVersion & 0x80000000) == 0:
        dwBuild     = (dwVersion & 0x7FFF0000) >> 16
    else:
        dwBuild     = None
    return int(dwMajorVersion), int(dwMinorVersion), int(dwBuild)

# BOOL WINAPI GetVersionEx(
#   __inout  LPOSVERSIONINFO lpVersionInfo
# );
def GetVersionExA():
    _GetVersionExA = windll.kernel32.GetVersionExA
    _GetVersionExA.argtypes = [POINTER(OSVERSIONINFOEXA)]
    _GetVersionExA.restype  = bool
    _GetVersionExA.errcheck = RaiseIfZero

    osi = OSVERSIONINFOEXA()
    osi.dwOSVersionInfoSize = sizeof(osi)
    try:
        _GetVersionExA(byref(osi))
    except WindowsError:
        osi = OSVERSIONINFOA()
        osi.dwOSVersionInfoSize = sizeof(osi)
        _GetVersionExA.argtypes = [POINTER(OSVERSIONINFOA)]
        _GetVersionExA(byref(osi))
    return osi

def GetVersionExW():
    _GetVersionExW = windll.kernel32.GetVersionExW
    _GetVersionExW.argtypes = [POINTER(OSVERSIONINFOEXW)]
    _GetVersionExW.restype  = bool
    _GetVersionExW.errcheck = RaiseIfZero

    osi = OSVERSIONINFOEXW()
    osi.dwOSVersionInfoSize = sizeof(osi)
    try:
        _GetVersionExW(byref(osi))
    except WindowsError:
        osi = OSVERSIONINFOW()
        osi.dwOSVersionInfoSize = sizeof(osi)
        _GetVersionExW.argtypes = [POINTER(OSVERSIONINFOW)]
        _GetVersionExW(byref(osi))
    return osi

GetVersionEx = GuessStringType(GetVersionExA, GetVersionExW)

# BOOL WINAPI GetProductInfo(
#   __in   DWORD dwOSMajorVersion,
#   __in   DWORD dwOSMinorVersion,
#   __in   DWORD dwSpMajorVersion,
#   __in   DWORD dwSpMinorVersion,
#   __out  PDWORD pdwReturnedProductType
# );
def GetProductInfo(dwOSMajorVersion, dwOSMinorVersion, dwSpMajorVersion, dwSpMinorVersion):
    _GetProductInfo = windll.kernel32.GetProductInfo
    _GetProductInfo.argtypes = [DWORD, DWORD, DWORD, DWORD, PDWORD]
    _GetProductInfo.restype  = BOOL
    _GetProductInfo.errcheck = RaiseIfZero

    dwReturnedProductType = DWORD(0)
    _GetProductInfo(dwOSMajorVersion, dwOSMinorVersion, dwSpMajorVersion, dwSpMinorVersion, byref(dwReturnedProductType))
    return dwReturnedProductType.value

# BOOL WINAPI VerifyVersionInfo(
#   __in  LPOSVERSIONINFOEX lpVersionInfo,
#   __in  DWORD dwTypeMask,
#   __in  DWORDLONG dwlConditionMask
# );
def VerifyVersionInfo(lpVersionInfo, dwTypeMask, dwlConditionMask):
    if isinstance(lpVersionInfo, OSVERSIONINFOEXA):
        return VerifyVersionInfoA(lpVersionInfo, dwTypeMask, dwlConditionMask)
    if isinstance(lpVersionInfo, OSVERSIONINFOEXW):
        return VerifyVersionInfoW(lpVersionInfo, dwTypeMask, dwlConditionMask)
    raise TypeError("Bad OSVERSIONINFOEX structure")

def VerifyVersionInfoA(lpVersionInfo, dwTypeMask, dwlConditionMask):
    _VerifyVersionInfoA = windll.kernel32.VerifyVersionInfoA
    _VerifyVersionInfoA.argtypes = [LPOSVERSIONINFOEXA, DWORD, DWORDLONG]
    _VerifyVersionInfoA.restype  = bool
    return _VerifyVersionInfoA(byref(lpVersionInfo), dwTypeMask, dwlConditionMask)

def VerifyVersionInfoW(lpVersionInfo, dwTypeMask, dwlConditionMask):
    _VerifyVersionInfoW = windll.kernel32.VerifyVersionInfoW
    _VerifyVersionInfoW.argtypes = [LPOSVERSIONINFOEXW, DWORD, DWORDLONG]
    _VerifyVersionInfoW.restype  = bool
    return _VerifyVersionInfoW(byref(lpVersionInfo), dwTypeMask, dwlConditionMask)

# ULONGLONG WINAPI VerSetConditionMask(
#   __in  ULONGLONG dwlConditionMask,
#   __in  DWORD dwTypeBitMask,
#   __in  BYTE dwConditionMask
# );
def VerSetConditionMask(dwlConditionMask, dwTypeBitMask, dwConditionMask):
    _VerSetConditionMask = windll.kernel32.VerSetConditionMask
    _VerSetConditionMask.argtypes = [ULONGLONG, DWORD, BYTE]
    _VerSetConditionMask.restype  = ULONGLONG
    return _VerSetConditionMask(dwlConditionMask, dwTypeBitMask, dwConditionMask)

#--- get_bits, get_arch and get_os --------------------------------------------

ARCH_UNKNOWN     = "unknown"
ARCH_I386        = "i386"
ARCH_MIPS        = "mips"
ARCH_ALPHA       = "alpha"
ARCH_PPC         = "ppc"
ARCH_SHX         = "shx"
ARCH_ARM         = "arm"
ARCH_ARM64       = "arm64"
ARCH_THUMB       = "thumb"
ARCH_IA64        = "ia64"
ARCH_ALPHA64     = "alpha64"
ARCH_MSIL        = "msil"
ARCH_AMD64       = "amd64"
ARCH_SPARC       = "sparc"

# aliases
ARCH_IA32    = ARCH_I386
ARCH_X86     = ARCH_I386
ARCH_X64     = ARCH_AMD64
ARCH_ARM7    = ARCH_ARM
ARCH_ARM8    = ARCH_ARM64
ARCH_T32     = ARCH_THUMB
ARCH_AARCH32 = ARCH_ARM7
ARCH_AARCH64 = ARCH_ARM8
ARCH_POWERPC = ARCH_PPC
ARCH_HITACHI = ARCH_SHX
ARCH_ITANIUM = ARCH_IA64

# win32 constants -> our constants
_arch_map = {
    PROCESSOR_ARCHITECTURE_INTEL          : ARCH_I386,
    PROCESSOR_ARCHITECTURE_MIPS           : ARCH_MIPS,
    PROCESSOR_ARCHITECTURE_ALPHA          : ARCH_ALPHA,
    PROCESSOR_ARCHITECTURE_PPC            : ARCH_PPC,
    PROCESSOR_ARCHITECTURE_SHX            : ARCH_SHX,
    PROCESSOR_ARCHITECTURE_ARM            : ARCH_ARM,
    PROCESSOR_ARCHITECTURE_IA64           : ARCH_IA64,
    PROCESSOR_ARCHITECTURE_ALPHA64        : ARCH_ALPHA64,
    PROCESSOR_ARCHITECTURE_MSIL           : ARCH_MSIL,
    PROCESSOR_ARCHITECTURE_AMD64          : ARCH_AMD64,
    PROCESSOR_ARCHITECTURE_SPARC          : ARCH_SPARC,
}

OS_UNKNOWN   = "Unknown"
OS_NT        = "Windows NT"
OS_W2K       = "Windows 2000"
OS_XP        = "Windows XP"
OS_XP_64     = "Windows XP (64 bits)"
OS_W2K3      = "Windows 2003"
OS_W2K3_64   = "Windows 2003 (64 bits)"
OS_W2K3R2    = "Windows 2003 R2"
OS_W2K3R2_64 = "Windows 2003 R2 (64 bits)"
OS_W2K8      = "Windows 2008"
OS_W2K8_64   = "Windows 2008 (64 bits)"
OS_W2K8R2    = "Windows 2008 R2"
OS_W2K8R2_64 = "Windows 2008 R2 (64 bits)"
OS_VISTA     = "Windows Vista"
OS_VISTA_64  = "Windows Vista (64 bits)"
OS_W7        = "Windows 7"
OS_W7_64     = "Windows 7 (64 bits)"

OS_SEVEN    = OS_W7
OS_SEVEN_64 = OS_W7_64

OS_WINDOWS_NT         = OS_NT
OS_WINDOWS_2000       = OS_W2K
OS_WINDOWS_XP         = OS_XP
OS_WINDOWS_XP_64      = OS_XP_64
OS_WINDOWS_2003       = OS_W2K3
OS_WINDOWS_2003_64    = OS_W2K3_64
OS_WINDOWS_2003_R2    = OS_W2K3R2
OS_WINDOWS_2003_R2_64 = OS_W2K3R2_64
OS_WINDOWS_2008       = OS_W2K8
OS_WINDOWS_2008_64    = OS_W2K8_64
OS_WINDOWS_2008_R2    = OS_W2K8R2
OS_WINDOWS_2008_R2_64 = OS_W2K8R2_64
OS_WINDOWS_VISTA      = OS_VISTA
OS_WINDOWS_VISTA_64   = OS_VISTA_64
OS_WINDOWS_SEVEN      = OS_W7
OS_WINDOWS_SEVEN_64   = OS_W7_64

def _get_bits():
    """
    Determines the current integer size in bits.

    This is useful to know if we're running in a 32 bits or a 64 bits machine.

    @rtype: int
    @return: Returns the size of L{SIZE_T} in bits.
    """
    return sizeof(SIZE_T) * 8

def _get_arch():
    """
    Determines the current processor architecture.

    @rtype: str
    @return:
        On error, returns:

         - L{ARCH_UNKNOWN} (C{"unknown"}) meaning the architecture could not be detected or is not known to WinAppDbg.

        On success, returns one of the following values:

         - L{ARCH_I386} (C{"i386"}) for Intel 32-bit x86 processor or compatible.
         - L{ARCH_AMD64} (C{"amd64"}) for Intel 64-bit x86_64 processor or compatible.

        May also return one of the following values if you get both Python and
        WinAppDbg to work in such machines... let me know if you do! :)

         - L{ARCH_MIPS} (C{"mips"}) for MIPS compatible processors.
         - L{ARCH_ALPHA} (C{"alpha"}) for Alpha processors.
         - L{ARCH_PPC} (C{"ppc"}) for PowerPC compatible processors.
         - L{ARCH_SHX} (C{"shx"}) for Hitachi SH processors.
         - L{ARCH_ARM} (C{"arm"}) for ARM compatible processors.
         - L{ARCH_IA64} (C{"ia64"}) for Intel Itanium processor or compatible.
         - L{ARCH_ALPHA64} (C{"alpha64"}) for Alpha64 processors.
         - L{ARCH_MSIL} (C{"msil"}) for the .NET virtual machine.
         - L{ARCH_SPARC} (C{"sparc"}) for Sun Sparc processors.

        Probably IronPython returns C{ARCH_MSIL} but I haven't tried it. Python
        on Windows CE and Windows Mobile should return C{ARCH_ARM}. Python on
        Solaris using Wine would return C{ARCH_SPARC}. Python in an Itanium
        machine should return C{ARCH_IA64} both on Wine and proper Windows.
        All other values should only be returned on Linux using Wine.
    """
    try:
        si = GetNativeSystemInfo()
    except Exception:
        si = GetSystemInfo()
    try:
        return _arch_map[si.id.w.wProcessorArchitecture]
    except KeyError:
        return ARCH_UNKNOWN

def _get_wow64():
    """
    Determines if the current process is running in Windows-On-Windows 64 bits.

    @rtype:  bool
    @return: C{True} of the current process is a 32 bit program running in a
        64 bit version of Windows, C{False} if it's either a 32 bit program
        in a 32 bit Windows or a 64 bit program in a 64 bit Windows.
    """
    # Try to determine if the debugger itself is running on WOW64.
    # On error assume False.
    if bits == 64:
        wow64 = False
    else:
        try:
            wow64 = IsWow64Process( GetCurrentProcess() )
        except Exception:
            wow64 = False
    return wow64

def _get_os(osvi = None):
    """
    Determines the current operating system.

    This function allows you to quickly tell apart major OS differences.
    For more detailed information call L{GetVersionEx} instead.

    @note:
        Wine reports itself as Windows XP 32 bits
        (even if the Linux host is 64 bits).
        ReactOS may report itself as Windows 2000 or Windows XP,
        depending on the version of ReactOS.

    @type  osvi: L{OSVERSIONINFOEXA}
    @param osvi: Optional. The return value from L{GetVersionEx}.

    @rtype: str
    @return:
        One of the following values:
         - L{OS_UNKNOWN} (C{"Unknown"})
         - L{OS_NT} (C{"Windows NT"})
         - L{OS_W2K} (C{"Windows 2000"})
         - L{OS_XP} (C{"Windows XP"})
         - L{OS_XP_64} (C{"Windows XP (64 bits)"})
         - L{OS_W2K3} (C{"Windows 2003"})
         - L{OS_W2K3_64} (C{"Windows 2003 (64 bits)"})
         - L{OS_W2K3R2} (C{"Windows 2003 R2"})
         - L{OS_W2K3R2_64} (C{"Windows 2003 R2 (64 bits)"})
         - L{OS_W2K8} (C{"Windows 2008"})
         - L{OS_W2K8_64} (C{"Windows 2008 (64 bits)"})
         - L{OS_W2K8R2} (C{"Windows 2008 R2"})
         - L{OS_W2K8R2_64} (C{"Windows 2008 R2 (64 bits)"})
         - L{OS_VISTA} (C{"Windows Vista"})
         - L{OS_VISTA_64} (C{"Windows Vista (64 bits)"})
         - L{OS_W7} (C{"Windows 7"})
         - L{OS_W7_64} (C{"Windows 7 (64 bits)"})
    """
    # rough port of http://msdn.microsoft.com/en-us/library/ms724429%28VS.85%29.aspx
    if not osvi:
        osvi = GetVersionEx()
    if osvi.dwPlatformId == VER_PLATFORM_WIN32_NT and osvi.dwMajorVersion > 4:
        if osvi.dwMajorVersion == 6:
            if osvi.dwMinorVersion == 0:
                if osvi.wProductType == VER_NT_WORKSTATION:
                    if bits == 64 or wow64:
                        return 'Windows Vista (64 bits)'
                    return 'Windows Vista'
                else:
                    if bits == 64 or wow64:
                        return 'Windows 2008 (64 bits)'
                    return 'Windows 2008'
            if osvi.dwMinorVersion == 1:
                if osvi.wProductType == VER_NT_WORKSTATION:
                    if bits == 64 or wow64:
                        return 'Windows 7 (64 bits)'
                    return 'Windows 7'
                else:
                    if bits == 64 or wow64:
                        return 'Windows 2008 R2 (64 bits)'
                    return 'Windows 2008 R2'
        if osvi.dwMajorVersion == 5:
            if osvi.dwMinorVersion == 2:
                if GetSystemMetrics(SM_SERVERR2):
                    if bits == 64 or wow64:
                        return 'Windows 2003 R2 (64 bits)'
                    return 'Windows 2003 R2'
                if osvi.wSuiteMask in (VER_SUITE_STORAGE_SERVER, VER_SUITE_WH_SERVER):
                    if bits == 64 or wow64:
                        return 'Windows 2003 (64 bits)'
                    return 'Windows 2003'
                if osvi.wProductType == VER_NT_WORKSTATION and arch == ARCH_AMD64:
                    return 'Windows XP (64 bits)'
                else:
                    if bits == 64 or wow64:
                        return 'Windows 2003 (64 bits)'
                    return 'Windows 2003'
            if osvi.dwMinorVersion == 1:
                return 'Windows XP'
            if osvi.dwMinorVersion == 0:
                return 'Windows 2000'
        if osvi.dwMajorVersion == 4:
            return 'Windows NT'
    return 'Unknown'

def _get_ntddi(osvi):
    """
    Determines the current operating system.

    This function allows you to quickly tell apart major OS differences.
    For more detailed information call L{kernel32.GetVersionEx} instead.

    @note:
        Wine reports itself as Windows XP 32 bits
        (even if the Linux host is 64 bits).
        ReactOS may report itself as Windows 2000 or Windows XP,
        depending on the version of ReactOS.

    @type  osvi: L{OSVERSIONINFOEXA}
    @param osvi: Optional. The return value from L{kernel32.GetVersionEx}.

    @rtype:  int
    @return: NTDDI version number.
    """
    if not osvi:
        osvi = GetVersionEx()
    ntddi = 0
    ntddi += (osvi.dwMajorVersion & 0xFF)    << 24
    ntddi += (osvi.dwMinorVersion & 0xFF)    << 16
    ntddi += (osvi.wServicePackMajor & 0xFF) << 8
    ntddi += (osvi.wServicePackMinor & 0xFF)
    return ntddi

# The order of the following definitions DOES matter!

# Current integer size in bits. See L{_get_bits} for more details.
bits = _get_bits()

# Current processor architecture. See L{_get_arch} for more details.
arch = _get_arch()

# Set to C{True} if the current process is running in WOW64. See L{_get_wow64} for more details.
wow64 = _get_wow64()

_osvi = GetVersionEx()

# Current operating system. See L{_get_os} for more details.
os = _get_os(_osvi)

# Current operating system as an NTDDI constant. See L{_get_ntddi} for more details.
NTDDI_VERSION = _get_ntddi(_osvi)

# Upper word of L{NTDDI_VERSION}, contains the OS major and minor version number.
WINVER = NTDDI_VERSION >> 16

#--- version.dll --------------------------------------------------------------

VS_FF_DEBUG         = 0x00000001
VS_FF_PRERELEASE    = 0x00000002
VS_FF_PATCHED       = 0x00000004
VS_FF_PRIVATEBUILD  = 0x00000008
VS_FF_INFOINFERRED  = 0x00000010
VS_FF_SPECIALBUILD  = 0x00000020

VOS_UNKNOWN     = 0x00000000
VOS__WINDOWS16  = 0x00000001
VOS__PM16       = 0x00000002
VOS__PM32       = 0x00000003
VOS__WINDOWS32  = 0x00000004
VOS_DOS         = 0x00010000
VOS_OS216       = 0x00020000
VOS_OS232       = 0x00030000
VOS_NT          = 0x00040000

VOS_DOS_WINDOWS16   = 0x00010001
VOS_DOS_WINDOWS32   = 0x00010004
VOS_NT_WINDOWS32    = 0x00040004
VOS_OS216_PM16      = 0x00020002
VOS_OS232_PM32      = 0x00030003

VFT_UNKNOWN     = 0x00000000
VFT_APP         = 0x00000001
VFT_DLL         = 0x00000002
VFT_DRV         = 0x00000003
VFT_FONT        = 0x00000004
VFT_VXD         = 0x00000005
VFT_RESERVED    = 0x00000006   # undocumented
VFT_STATIC_LIB  = 0x00000007

VFT2_UNKNOWN                = 0x00000000

VFT2_DRV_PRINTER            = 0x00000001
VFT2_DRV_KEYBOARD           = 0x00000002
VFT2_DRV_LANGUAGE           = 0x00000003
VFT2_DRV_DISPLAY            = 0x00000004
VFT2_DRV_MOUSE              = 0x00000005
VFT2_DRV_NETWORK            = 0x00000006
VFT2_DRV_SYSTEM             = 0x00000007
VFT2_DRV_INSTALLABLE        = 0x00000008
VFT2_DRV_SOUND              = 0x00000009
VFT2_DRV_COMM               = 0x0000000A
VFT2_DRV_RESERVED           = 0x0000000B    # undocumented
VFT2_DRV_VERSIONED_PRINTER  = 0x0000000C

VFT2_FONT_RASTER            = 0x00000001
VFT2_FONT_VECTOR            = 0x00000002
VFT2_FONT_TRUETYPE          = 0x00000003

# typedef struct tagVS_FIXEDFILEINFO {
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
# } VS_FIXEDFILEINFO;
class VS_FIXEDFILEINFO(Structure):
    _fields_ = [
        ("dwSignature",         DWORD),
        ("dwStrucVersion",      DWORD),
        ("dwFileVersionMS",     DWORD),
        ("dwFileVersionLS",     DWORD),
        ("dwProductVersionMS",  DWORD),
        ("dwProductVersionLS",  DWORD),
        ("dwFileFlagsMask",     DWORD),
        ("dwFileFlags",         DWORD),
        ("dwFileOS",            DWORD),
        ("dwFileType",          DWORD),
        ("dwFileSubtype",       DWORD),
        ("dwFileDateMS",        DWORD),
        ("dwFileDateLS",        DWORD),
]
PVS_FIXEDFILEINFO = POINTER(VS_FIXEDFILEINFO)
LPVS_FIXEDFILEINFO = PVS_FIXEDFILEINFO

# BOOL WINAPI GetFileVersionInfo(
#   _In_        LPCTSTR lptstrFilename,
#   _Reserved_  DWORD dwHandle,
#   _In_        DWORD dwLen,
#   _Out_       LPVOID lpData
# );
# DWORD WINAPI GetFileVersionInfoSize(
#   _In_       LPCTSTR lptstrFilename,
#   _Out_opt_  LPDWORD lpdwHandle
# );
def GetFileVersionInfoA(lptstrFilename):
    _GetFileVersionInfoA = windll.version.GetFileVersionInfoA
    _GetFileVersionInfoA.argtypes = [LPSTR, DWORD, DWORD, LPVOID]
    _GetFileVersionInfoA.restype  = bool
    _GetFileVersionInfoA.errcheck = RaiseIfZero

    _GetFileVersionInfoSizeA = windll.version.GetFileVersionInfoSizeA
    _GetFileVersionInfoSizeA.argtypes = [LPSTR, LPVOID]
    _GetFileVersionInfoSizeA.restype  = DWORD
    _GetFileVersionInfoSizeA.errcheck = RaiseIfZero

    dwLen = _GetFileVersionInfoSizeA(lptstrFilename, None)
    lpData = ctypes.create_string_buffer(dwLen)
    _GetFileVersionInfoA(lptstrFilename, 0, dwLen, byref(lpData))
    return lpData

def GetFileVersionInfoW(lptstrFilename):
    _GetFileVersionInfoW = windll.version.GetFileVersionInfoW
    _GetFileVersionInfoW.argtypes = [LPWSTR, DWORD, DWORD, LPVOID]
    _GetFileVersionInfoW.restype  = bool
    _GetFileVersionInfoW.errcheck = RaiseIfZero

    _GetFileVersionInfoSizeW = windll.version.GetFileVersionInfoSizeW
    _GetFileVersionInfoSizeW.argtypes = [LPWSTR, LPVOID]
    _GetFileVersionInfoSizeW.restype  = DWORD
    _GetFileVersionInfoSizeW.errcheck = RaiseIfZero

    dwLen = _GetFileVersionInfoSizeW(lptstrFilename, None)
    lpData = ctypes.create_string_buffer(dwLen)  # not a string!
    _GetFileVersionInfoW(lptstrFilename, 0, dwLen, byref(lpData))
    return lpData

GetFileVersionInfo = GuessStringType(GetFileVersionInfoA, GetFileVersionInfoW)

# BOOL WINAPI VerQueryValue(
#   _In_   LPCVOID pBlock,
#   _In_   LPCTSTR lpSubBlock,
#   _Out_  LPVOID *lplpBuffer,
#   _Out_  PUINT puLen
# );
def VerQueryValueA(pBlock, lpSubBlock):
    _VerQueryValueA = windll.version.VerQueryValueA
    _VerQueryValueA.argtypes = [LPVOID, LPSTR, LPVOID, POINTER(UINT)]
    _VerQueryValueA.restype  = bool
    _VerQueryValueA.errcheck = RaiseIfZero

    lpBuffer = LPVOID(0)
    uLen = UINT(0)
    _VerQueryValueA(pBlock, lpSubBlock, byref(lpBuffer), byref(uLen))
    return lpBuffer, uLen.value

def VerQueryValueW(pBlock, lpSubBlock):
    _VerQueryValueW = windll.version.VerQueryValueW
    _VerQueryValueW.argtypes = [LPVOID, LPWSTR, LPVOID, POINTER(UINT)]
    _VerQueryValueW.restype  = bool
    _VerQueryValueW.errcheck = RaiseIfZero

    lpBuffer = LPVOID(0)
    uLen = UINT(0)
    _VerQueryValueW(pBlock, lpSubBlock, byref(lpBuffer), byref(uLen))
    return lpBuffer, uLen.value

VerQueryValue = GuessStringType(VerQueryValueA, VerQueryValueW)

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
