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
Wrapper for wtsapi32.dll in ctypes.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

#--- Constants ----------------------------------------------------------------

WTS_CURRENT_SERVER_HANDLE = 0
WTS_CURRENT_SESSION       = 1

#--- WTS_PROCESS_INFO structure -----------------------------------------------

# typedef struct _WTS_PROCESS_INFO {
#   DWORD  SessionId;
#   DWORD  ProcessId;
#   LPTSTR pProcessName;
#   PSID   pUserSid;
# } WTS_PROCESS_INFO, *PWTS_PROCESS_INFO;

class WTS_PROCESS_INFOA(Structure):
    _fields_ = [
        ("SessionId",    DWORD),
        ("ProcessId",    DWORD),
        ("pProcessName", LPSTR),
        ("pUserSid",     PSID),
    ]
PWTS_PROCESS_INFOA = POINTER(WTS_PROCESS_INFOA)

class WTS_PROCESS_INFOW(Structure):
    _fields_ = [
        ("SessionId",    DWORD),
        ("ProcessId",    DWORD),
        ("pProcessName", LPWSTR),
        ("pUserSid",     PSID),
    ]
PWTS_PROCESS_INFOW = POINTER(WTS_PROCESS_INFOW)

#--- WTSQuerySessionInformation enums and structures --------------------------

# typedef enum _WTS_INFO_CLASS {
#   WTSInitialProgram          = 0,
#   WTSApplicationName         = 1,
#   WTSWorkingDirectory        = 2,
#   WTSOEMId                   = 3,
#   WTSSessionId               = 4,
#   WTSUserName                = 5,
#   WTSWinStationName          = 6,
#   WTSDomainName              = 7,
#   WTSConnectState            = 8,
#   WTSClientBuildNumber       = 9,
#   WTSClientName              = 10,
#   WTSClientDirectory         = 11,
#   WTSClientProductId         = 12,
#   WTSClientHardwareId        = 13,
#   WTSClientAddress           = 14,
#   WTSClientDisplay           = 15,
#   WTSClientProtocolType      = 16,
#   WTSIdleTime                = 17,
#   WTSLogonTime               = 18,
#   WTSIncomingBytes           = 19,
#   WTSOutgoingBytes           = 20,
#   WTSIncomingFrames          = 21,
#   WTSOutgoingFrames          = 22,
#   WTSClientInfo              = 23,
#   WTSSessionInfo             = 24,
#   WTSSessionInfoEx           = 25,
#   WTSConfigInfo              = 26,
#   WTSValidationInfo          = 27,
#   WTSSessionAddressV4        = 28,
#   WTSIsRemoteSession         = 29
# } WTS_INFO_CLASS;

WTSInitialProgram          = 0
WTSApplicationName         = 1
WTSWorkingDirectory        = 2
WTSOEMId                   = 3
WTSSessionId               = 4
WTSUserName                = 5
WTSWinStationName          = 6
WTSDomainName              = 7
WTSConnectState            = 8
WTSClientBuildNumber       = 9
WTSClientName              = 10
WTSClientDirectory         = 11
WTSClientProductId         = 12
WTSClientHardwareId        = 13
WTSClientAddress           = 14
WTSClientDisplay           = 15
WTSClientProtocolType      = 16
WTSIdleTime                = 17
WTSLogonTime               = 18
WTSIncomingBytes           = 19
WTSOutgoingBytes           = 20
WTSIncomingFrames          = 21
WTSOutgoingFrames          = 22
WTSClientInfo              = 23
WTSSessionInfo             = 24
WTSSessionInfoEx           = 25
WTSConfigInfo              = 26
WTSValidationInfo          = 27
WTSSessionAddressV4        = 28
WTSIsRemoteSession         = 29

WTS_INFO_CLASS = ctypes.c_int

# typedef enum _WTS_CONNECTSTATE_CLASS {
#   WTSActive,
#   WTSConnected,
#   WTSConnectQuery,
#   WTSShadow,
#   WTSDisconnected,
#   WTSIdle,
#   WTSListen,
#   WTSReset,
#   WTSDown,
#   WTSInit
# } WTS_CONNECTSTATE_CLASS;

WTSActive       = 0
WTSConnected    = 1
WTSConnectQuery = 2
WTSShadow       = 3
WTSDisconnected = 4
WTSIdle         = 5
WTSListen       = 6
WTSReset        = 7
WTSDown         = 8
WTSInit         = 9

WTS_CONNECTSTATE_CLASS = ctypes.c_int

# typedef struct _WTS_CLIENT_DISPLAY {
#   DWORD HorizontalResolution;
#   DWORD VerticalResolution;
#   DWORD ColorDepth;
# } WTS_CLIENT_DISPLAY, *PWTS_CLIENT_DISPLAY;
class WTS_CLIENT_DISPLAY(Structure):
    _fields_ = [
        ("HorizontalResolution", DWORD),
        ("VerticalResolution",   DWORD),
        ("ColorDepth",           DWORD),
    ]
PWTS_CLIENT_DISPLAY = POINTER(WTS_CLIENT_DISPLAY)

# typedef struct _WTS_CLIENT_ADDRESS {
#   DWORD AddressFamily;
#   BYTE  Address[20];
# } WTS_CLIENT_ADDRESS, *PWTS_CLIENT_ADDRESS;

# XXX TODO

# typedef struct _WTSCLIENT {
#   WCHAR   ClientName[CLIENTNAME_LENGTH + 1];
#   WCHAR   Domain[DOMAIN_LENGTH + 1 ];
#   WCHAR   UserName[USERNAME_LENGTH + 1];
#   WCHAR   WorkDirectory[MAX_PATH + 1];
#   WCHAR   InitialProgram[MAX_PATH + 1];
#   BYTE    EncryptionLevel;
#   ULONG   ClientAddressFamily;
#   USHORT  ClientAddress[CLIENTADDRESS_LENGTH + 1];
#   USHORT  HRes;
#   USHORT  VRes;
#   USHORT  ColorDepth;
#   WCHAR   ClientDirectory[MAX_PATH + 1];
#   ULONG   ClientBuildNumber;
#   ULONG   ClientHardwareId;
#   USHORT  ClientProductId;
#   USHORT  OutBufCountHost;
#   USHORT  OutBufCountClient;
#   USHORT  OutBufLength;
#   WCHAR     DeviceId[MAX_PATH + 1];
# } WTSCLIENT, *PWTSCLIENT;

# XXX TODO

# typedef struct _WTSINFO {
#   WTS_CONNECTSTATE_CLASS State;
#   DWORD                  SessionId;
#   DWORD                  IncomingBytes;
#   DWORD                  OutgoingBytes;
#   DWORD                  IncomingCompressedBytes;
#   DWORD                  OutgoingCompressedBytes;
#   WCHAR                  WinStationName;
#   WCHAR                  Domain;
#   WCHAR                  UserName;
#   LARGE_INTEGER          ConnectTime;
#   LARGE_INTEGER          DisconnectTime;
#   LARGE_INTEGER          LastInputTime;
#   LARGE_INTEGER          LogonTime;
#   LARGE_INTEGER          CurrentTime;
# } WTSINFO, *PWTSINFO;

# XXX TODO

# typedef struct _WTSINFOEX {
#   DWORD           Level;
#   WTSINFOEX_LEVEL Data;
# } WTSINFOEX, *PWTSINFOEX;

# XXX TODO

#--- wtsapi32.dll -------------------------------------------------------------

# void WTSFreeMemory(
#   __in  PVOID pMemory
# );
def WTSFreeMemory(pMemory):
    _WTSFreeMemory = windll.wtsapi32.WTSFreeMemory
    _WTSFreeMemory.argtypes = [PVOID]
    _WTSFreeMemory.restype  = None
    _WTSFreeMemory(pMemory)

# BOOL WTSEnumerateProcesses(
#   __in   HANDLE hServer,
#   __in   DWORD Reserved,
#   __in   DWORD Version,
#   __out  PWTS_PROCESS_INFO *ppProcessInfo,
#   __out  DWORD *pCount
# );
def WTSEnumerateProcessesA(hServer = WTS_CURRENT_SERVER_HANDLE):
    _WTSEnumerateProcessesA = windll.wtsapi32.WTSEnumerateProcessesA
    _WTSEnumerateProcessesA.argtypes = [HANDLE, DWORD, DWORD, POINTER(PWTS_PROCESS_INFOA), PDWORD]
    _WTSEnumerateProcessesA.restype  = bool
    _WTSEnumerateProcessesA.errcheck = RaiseIfZero

    pProcessInfo = PWTS_PROCESS_INFOA()
    Count = DWORD(0)
    _WTSEnumerateProcessesA(hServer, 0, 1, byref(pProcessInfo), byref(Count))
    return pProcessInfo, Count.value

def WTSEnumerateProcessesW(hServer = WTS_CURRENT_SERVER_HANDLE):
    _WTSEnumerateProcessesW = windll.wtsapi32.WTSEnumerateProcessesW
    _WTSEnumerateProcessesW.argtypes = [HANDLE, DWORD, DWORD, POINTER(PWTS_PROCESS_INFOW), PDWORD]
    _WTSEnumerateProcessesW.restype  = bool
    _WTSEnumerateProcessesW.errcheck = RaiseIfZero

    pProcessInfo = PWTS_PROCESS_INFOW()
    Count = DWORD(0)
    _WTSEnumerateProcessesW(hServer, 0, 1, byref(pProcessInfo), byref(Count))
    return pProcessInfo, Count.value

WTSEnumerateProcesses = DefaultStringType(WTSEnumerateProcessesA, WTSEnumerateProcessesW)

# BOOL WTSTerminateProcess(
#   __in  HANDLE hServer,
#   __in  DWORD ProcessId,
#   __in  DWORD ExitCode
# );
def WTSTerminateProcess(hServer, ProcessId, ExitCode):
    _WTSTerminateProcess = windll.wtsapi32.WTSTerminateProcess
    _WTSTerminateProcess.argtypes = [HANDLE, DWORD, DWORD]
    _WTSTerminateProcess.restype  = bool
    _WTSTerminateProcess.errcheck = RaiseIfZero
    _WTSTerminateProcess(hServer, ProcessId, ExitCode)

# BOOL WTSQuerySessionInformation(
#   __in   HANDLE hServer,
#   __in   DWORD SessionId,
#   __in   WTS_INFO_CLASS WTSInfoClass,
#   __out  LPTSTR *ppBuffer,
#   __out  DWORD *pBytesReturned
# );

# XXX TODO

#--- kernel32.dll -------------------------------------------------------------

# I've no idea why these functions are in kernel32.dll instead of wtsapi32.dll

# BOOL ProcessIdToSessionId(
#   __in   DWORD dwProcessId,
#   __out  DWORD *pSessionId
# );
def ProcessIdToSessionId(dwProcessId):
    _ProcessIdToSessionId = windll.kernel32.ProcessIdToSessionId
    _ProcessIdToSessionId.argtypes = [DWORD, PDWORD]
    _ProcessIdToSessionId.restype  = bool
    _ProcessIdToSessionId.errcheck = RaiseIfZero

    dwSessionId = DWORD(0)
    _ProcessIdToSessionId(dwProcessId, byref(dwSessionId))
    return dwSessionId.value

# DWORD WTSGetActiveConsoleSessionId(void);
def WTSGetActiveConsoleSessionId():
    _WTSGetActiveConsoleSessionId = windll.kernel32.WTSGetActiveConsoleSessionId
    _WTSGetActiveConsoleSessionId.argtypes = []
    _WTSGetActiveConsoleSessionId.restype  = DWORD
    _WTSGetActiveConsoleSessionId.errcheck = RaiseIfZero
    return _WTSGetActiveConsoleSessionId()

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
