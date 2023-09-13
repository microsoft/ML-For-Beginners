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
Wrapper for psapi.dll in ctypes.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

#--- PSAPI structures and constants -------------------------------------------

LIST_MODULES_DEFAULT    = 0x00
LIST_MODULES_32BIT      = 0x01
LIST_MODULES_64BIT      = 0x02
LIST_MODULES_ALL        = 0x03

# typedef struct _MODULEINFO {
#   LPVOID lpBaseOfDll;
#   DWORD  SizeOfImage;
#   LPVOID EntryPoint;
# } MODULEINFO, *LPMODULEINFO;
class MODULEINFO(Structure):
    _fields_ = [
        ("lpBaseOfDll",     LPVOID),    # remote pointer
        ("SizeOfImage",     DWORD),
        ("EntryPoint",      LPVOID),    # remote pointer
]
LPMODULEINFO = POINTER(MODULEINFO)

#--- psapi.dll ----------------------------------------------------------------

# BOOL WINAPI EnumDeviceDrivers(
#   __out  LPVOID *lpImageBase,
#   __in   DWORD cb,
#   __out  LPDWORD lpcbNeeded
# );
def EnumDeviceDrivers():
    _EnumDeviceDrivers = windll.psapi.EnumDeviceDrivers
    _EnumDeviceDrivers.argtypes = [LPVOID, DWORD, LPDWORD]
    _EnumDeviceDrivers.restype = bool
    _EnumDeviceDrivers.errcheck = RaiseIfZero

    size       = 0x1000
    lpcbNeeded = DWORD(size)
    unit       = sizeof(LPVOID)
    while 1:
        lpImageBase = (LPVOID * (size // unit))()
        _EnumDeviceDrivers(byref(lpImageBase), lpcbNeeded, byref(lpcbNeeded))
        needed = lpcbNeeded.value
        if needed <= size:
            break
        size = needed
    return [ lpImageBase[index] for index in compat.xrange(0, (needed // unit)) ]

# BOOL WINAPI EnumProcesses(
#   __out  DWORD *pProcessIds,
#   __in   DWORD cb,
#   __out  DWORD *pBytesReturned
# );
def EnumProcesses():
    _EnumProcesses = windll.psapi.EnumProcesses
    _EnumProcesses.argtypes = [LPVOID, DWORD, LPDWORD]
    _EnumProcesses.restype = bool
    _EnumProcesses.errcheck = RaiseIfZero

    size            = 0x1000
    cbBytesReturned = DWORD()
    unit            = sizeof(DWORD)
    while 1:
        ProcessIds = (DWORD * (size // unit))()
        cbBytesReturned.value = size
        _EnumProcesses(byref(ProcessIds), cbBytesReturned, byref(cbBytesReturned))
        returned = cbBytesReturned.value
        if returned < size:
            break
        size = size + 0x1000
    ProcessIdList = list()
    for ProcessId in ProcessIds:
        if ProcessId is None:
            break
        ProcessIdList.append(ProcessId)
    return ProcessIdList

# BOOL WINAPI EnumProcessModules(
#   __in   HANDLE hProcess,
#   __out  HMODULE *lphModule,
#   __in   DWORD cb,
#   __out  LPDWORD lpcbNeeded
# );
def EnumProcessModules(hProcess):
    _EnumProcessModules = windll.psapi.EnumProcessModules
    _EnumProcessModules.argtypes = [HANDLE, LPVOID, DWORD, LPDWORD]
    _EnumProcessModules.restype = bool
    _EnumProcessModules.errcheck = RaiseIfZero

    size = 0x1000
    lpcbNeeded = DWORD(size)
    unit = sizeof(HMODULE)
    while 1:
        lphModule = (HMODULE * (size // unit))()
        _EnumProcessModules(hProcess, byref(lphModule), lpcbNeeded, byref(lpcbNeeded))
        needed = lpcbNeeded.value
        if needed <= size:
            break
        size = needed
    return [ lphModule[index] for index in compat.xrange(0, int(needed // unit)) ]

# BOOL WINAPI EnumProcessModulesEx(
#   __in   HANDLE hProcess,
#   __out  HMODULE *lphModule,
#   __in   DWORD cb,
#   __out  LPDWORD lpcbNeeded,
#   __in   DWORD dwFilterFlag
# );
def EnumProcessModulesEx(hProcess, dwFilterFlag = LIST_MODULES_DEFAULT):
    _EnumProcessModulesEx = windll.psapi.EnumProcessModulesEx
    _EnumProcessModulesEx.argtypes = [HANDLE, LPVOID, DWORD, LPDWORD, DWORD]
    _EnumProcessModulesEx.restype = bool
    _EnumProcessModulesEx.errcheck = RaiseIfZero

    size = 0x1000
    lpcbNeeded = DWORD(size)
    unit = sizeof(HMODULE)
    while 1:
        lphModule = (HMODULE * (size // unit))()
        _EnumProcessModulesEx(hProcess, byref(lphModule), lpcbNeeded, byref(lpcbNeeded), dwFilterFlag)
        needed = lpcbNeeded.value
        if needed <= size:
            break
        size = needed
    return [ lphModule[index] for index in compat.xrange(0, (needed // unit)) ]

# DWORD WINAPI GetDeviceDriverBaseName(
#   __in   LPVOID ImageBase,
#   __out  LPTSTR lpBaseName,
#   __in   DWORD nSize
# );
def GetDeviceDriverBaseNameA(ImageBase):
    _GetDeviceDriverBaseNameA = windll.psapi.GetDeviceDriverBaseNameA
    _GetDeviceDriverBaseNameA.argtypes = [LPVOID, LPSTR, DWORD]
    _GetDeviceDriverBaseNameA.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpBaseName = ctypes.create_string_buffer("", nSize)
        nCopied = _GetDeviceDriverBaseNameA(ImageBase, lpBaseName, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpBaseName.value

def GetDeviceDriverBaseNameW(ImageBase):
    _GetDeviceDriverBaseNameW = windll.psapi.GetDeviceDriverBaseNameW
    _GetDeviceDriverBaseNameW.argtypes = [LPVOID, LPWSTR, DWORD]
    _GetDeviceDriverBaseNameW.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpBaseName = ctypes.create_unicode_buffer(u"", nSize)
        nCopied = _GetDeviceDriverBaseNameW(ImageBase, lpBaseName, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpBaseName.value

GetDeviceDriverBaseName = GuessStringType(GetDeviceDriverBaseNameA, GetDeviceDriverBaseNameW)

# DWORD WINAPI GetDeviceDriverFileName(
#   __in   LPVOID ImageBase,
#   __out  LPTSTR lpFilename,
#   __in   DWORD nSize
# );
def GetDeviceDriverFileNameA(ImageBase):
    _GetDeviceDriverFileNameA = windll.psapi.GetDeviceDriverFileNameA
    _GetDeviceDriverFileNameA.argtypes = [LPVOID, LPSTR, DWORD]
    _GetDeviceDriverFileNameA.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_string_buffer("", nSize)
        nCopied = ctypes.windll.psapi.GetDeviceDriverFileNameA(ImageBase, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

def GetDeviceDriverFileNameW(ImageBase):
    _GetDeviceDriverFileNameW = windll.psapi.GetDeviceDriverFileNameW
    _GetDeviceDriverFileNameW.argtypes = [LPVOID, LPWSTR, DWORD]
    _GetDeviceDriverFileNameW.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u"", nSize)
        nCopied = ctypes.windll.psapi.GetDeviceDriverFileNameW(ImageBase, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

GetDeviceDriverFileName = GuessStringType(GetDeviceDriverFileNameA, GetDeviceDriverFileNameW)

# DWORD WINAPI GetMappedFileName(
#   __in   HANDLE hProcess,
#   __in   LPVOID lpv,
#   __out  LPTSTR lpFilename,
#   __in   DWORD nSize
# );
def GetMappedFileNameA(hProcess, lpv):
    _GetMappedFileNameA = ctypes.windll.psapi.GetMappedFileNameA
    _GetMappedFileNameA.argtypes = [HANDLE, LPVOID, LPSTR, DWORD]
    _GetMappedFileNameA.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_string_buffer("", nSize)
        nCopied = _GetMappedFileNameA(hProcess, lpv, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

def GetMappedFileNameW(hProcess, lpv):
    _GetMappedFileNameW = ctypes.windll.psapi.GetMappedFileNameW
    _GetMappedFileNameW.argtypes = [HANDLE, LPVOID, LPWSTR, DWORD]
    _GetMappedFileNameW.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u"", nSize)
        nCopied = _GetMappedFileNameW(hProcess, lpv, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

GetMappedFileName = GuessStringType(GetMappedFileNameA, GetMappedFileNameW)

# DWORD WINAPI GetModuleFileNameEx(
#   __in      HANDLE hProcess,
#   __in_opt  HMODULE hModule,
#   __out     LPTSTR lpFilename,
#   __in      DWORD nSize
# );
def GetModuleFileNameExA(hProcess, hModule = None):
    _GetModuleFileNameExA = ctypes.windll.psapi.GetModuleFileNameExA
    _GetModuleFileNameExA.argtypes = [HANDLE, HMODULE, LPSTR, DWORD]
    _GetModuleFileNameExA.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_string_buffer("", nSize)
        nCopied = _GetModuleFileNameExA(hProcess, hModule, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

def GetModuleFileNameExW(hProcess, hModule = None):
    _GetModuleFileNameExW = ctypes.windll.psapi.GetModuleFileNameExW
    _GetModuleFileNameExW.argtypes = [HANDLE, HMODULE, LPWSTR, DWORD]
    _GetModuleFileNameExW.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u"", nSize)
        nCopied = _GetModuleFileNameExW(hProcess, hModule, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

GetModuleFileNameEx = GuessStringType(GetModuleFileNameExA, GetModuleFileNameExW)

# BOOL WINAPI GetModuleInformation(
#   __in   HANDLE hProcess,
#   __in   HMODULE hModule,
#   __out  LPMODULEINFO lpmodinfo,
#   __in   DWORD cb
# );
def GetModuleInformation(hProcess, hModule, lpmodinfo = None):
    _GetModuleInformation = windll.psapi.GetModuleInformation
    _GetModuleInformation.argtypes = [HANDLE, HMODULE, LPMODULEINFO, DWORD]
    _GetModuleInformation.restype = bool
    _GetModuleInformation.errcheck = RaiseIfZero

    if lpmodinfo is None:
        lpmodinfo = MODULEINFO()
    _GetModuleInformation(hProcess, hModule, byref(lpmodinfo), sizeof(lpmodinfo))
    return lpmodinfo

# DWORD WINAPI GetProcessImageFileName(
#   __in   HANDLE hProcess,
#   __out  LPTSTR lpImageFileName,
#   __in   DWORD nSize
# );
def GetProcessImageFileNameA(hProcess):
    _GetProcessImageFileNameA = windll.psapi.GetProcessImageFileNameA
    _GetProcessImageFileNameA.argtypes = [HANDLE, LPSTR, DWORD]
    _GetProcessImageFileNameA.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_string_buffer("", nSize)
        nCopied = _GetProcessImageFileNameA(hProcess, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

def GetProcessImageFileNameW(hProcess):
    _GetProcessImageFileNameW = windll.psapi.GetProcessImageFileNameW
    _GetProcessImageFileNameW.argtypes = [HANDLE, LPWSTR, DWORD]
    _GetProcessImageFileNameW.restype = DWORD

    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u"", nSize)
        nCopied = _GetProcessImageFileNameW(hProcess, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < (nSize - 1):
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value

GetProcessImageFileName = GuessStringType(GetProcessImageFileNameA, GetProcessImageFileNameW)

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
