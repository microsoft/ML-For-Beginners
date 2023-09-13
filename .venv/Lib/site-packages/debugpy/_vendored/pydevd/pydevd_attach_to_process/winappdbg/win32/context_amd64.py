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
CONTEXT structure for amd64.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

#--- CONTEXT structures and constants -----------------------------------------

# The following values specify the type of access in the first parameter
# of the exception record when the exception code specifies an access
# violation.
EXCEPTION_READ_FAULT        = 0     # exception caused by a read
EXCEPTION_WRITE_FAULT       = 1     # exception caused by a write
EXCEPTION_EXECUTE_FAULT     = 8     # exception caused by an instruction fetch

CONTEXT_AMD64           = 0x00100000

CONTEXT_CONTROL         = (CONTEXT_AMD64 | long(0x1))
CONTEXT_INTEGER         = (CONTEXT_AMD64 | long(0x2))
CONTEXT_SEGMENTS        = (CONTEXT_AMD64 | long(0x4))
CONTEXT_FLOATING_POINT  = (CONTEXT_AMD64 | long(0x8))
CONTEXT_DEBUG_REGISTERS = (CONTEXT_AMD64 | long(0x10))

CONTEXT_MMX_REGISTERS   = CONTEXT_FLOATING_POINT

CONTEXT_FULL = (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT)

CONTEXT_ALL = (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS | \
               CONTEXT_FLOATING_POINT | CONTEXT_DEBUG_REGISTERS)

CONTEXT_EXCEPTION_ACTIVE    = 0x8000000
CONTEXT_SERVICE_ACTIVE      = 0x10000000
CONTEXT_EXCEPTION_REQUEST   = 0x40000000
CONTEXT_EXCEPTION_REPORTING = 0x80000000

INITIAL_MXCSR = 0x1f80            # initial MXCSR value
INITIAL_FPCSR = 0x027f            # initial FPCSR value

# typedef struct _XMM_SAVE_AREA32 {
#     WORD   ControlWord;
#     WORD   StatusWord;
#     BYTE  TagWord;
#     BYTE  Reserved1;
#     WORD   ErrorOpcode;
#     DWORD ErrorOffset;
#     WORD   ErrorSelector;
#     WORD   Reserved2;
#     DWORD DataOffset;
#     WORD   DataSelector;
#     WORD   Reserved3;
#     DWORD MxCsr;
#     DWORD MxCsr_Mask;
#     M128A FloatRegisters[8];
#     M128A XmmRegisters[16];
#     BYTE  Reserved4[96];
# } XMM_SAVE_AREA32, *PXMM_SAVE_AREA32;
class XMM_SAVE_AREA32(Structure):
    _pack_ = 1
    _fields_ = [
        ('ControlWord',     WORD),
        ('StatusWord',      WORD),
        ('TagWord',         BYTE),
        ('Reserved1',       BYTE),
        ('ErrorOpcode',     WORD),
        ('ErrorOffset',     DWORD),
        ('ErrorSelector',   WORD),
        ('Reserved2',       WORD),
        ('DataOffset',      DWORD),
        ('DataSelector',    WORD),
        ('Reserved3',       WORD),
        ('MxCsr',           DWORD),
        ('MxCsr_Mask',      DWORD),
        ('FloatRegisters',  M128A * 8),
        ('XmmRegisters',    M128A * 16),
        ('Reserved4',       BYTE * 96),
    ]

    def from_dict(self):
        raise NotImplementedError()

    def to_dict(self):
        d = dict()
        for name, type in self._fields_:
            if name in ('FloatRegisters', 'XmmRegisters'):
                d[name] = tuple([ (x.LowPart + (x.HighPart << 64)) for x in getattr(self, name) ])
            elif name == 'Reserved4':
                d[name] = tuple([ chr(x) for x in getattr(self, name) ])
            else:
                d[name] = getattr(self, name)
        return d

LEGACY_SAVE_AREA_LENGTH = sizeof(XMM_SAVE_AREA32)

PXMM_SAVE_AREA32 = ctypes.POINTER(XMM_SAVE_AREA32)
LPXMM_SAVE_AREA32 = PXMM_SAVE_AREA32

# //
# // Context Frame
# //
# //  This frame has a several purposes: 1) it is used as an argument to
# //  NtContinue, 2) is is used to constuct a call frame for APC delivery,
# //  and 3) it is used in the user level thread creation routines.
# //
# //
# // The flags field within this record controls the contents of a CONTEXT
# // record.
# //
# // If the context record is used as an input parameter, then for each
# // portion of the context record controlled by a flag whose value is
# // set, it is assumed that that portion of the context record contains
# // valid context. If the context record is being used to modify a threads
# // context, then only that portion of the threads context is modified.
# //
# // If the context record is used as an output parameter to capture the
# // context of a thread, then only those portions of the thread's context
# // corresponding to set flags will be returned.
# //
# // CONTEXT_CONTROL specifies SegSs, Rsp, SegCs, Rip, and EFlags.
# //
# // CONTEXT_INTEGER specifies Rax, Rcx, Rdx, Rbx, Rbp, Rsi, Rdi, and R8-R15.
# //
# // CONTEXT_SEGMENTS specifies SegDs, SegEs, SegFs, and SegGs.
# //
# // CONTEXT_DEBUG_REGISTERS specifies Dr0-Dr3 and Dr6-Dr7.
# //
# // CONTEXT_MMX_REGISTERS specifies the floating point and extended registers
# //     Mm0/St0-Mm7/St7 and Xmm0-Xmm15).
# //
#
# typedef struct DECLSPEC_ALIGN(16) _CONTEXT {
#
#     //
#     // Register parameter home addresses.
#     //
#     // N.B. These fields are for convience - they could be used to extend the
#     //      context record in the future.
#     //
#
#     DWORD64 P1Home;
#     DWORD64 P2Home;
#     DWORD64 P3Home;
#     DWORD64 P4Home;
#     DWORD64 P5Home;
#     DWORD64 P6Home;
#
#     //
#     // Control flags.
#     //
#
#     DWORD ContextFlags;
#     DWORD MxCsr;
#
#     //
#     // Segment Registers and processor flags.
#     //
#
#     WORD   SegCs;
#     WORD   SegDs;
#     WORD   SegEs;
#     WORD   SegFs;
#     WORD   SegGs;
#     WORD   SegSs;
#     DWORD EFlags;
#
#     //
#     // Debug registers
#     //
#
#     DWORD64 Dr0;
#     DWORD64 Dr1;
#     DWORD64 Dr2;
#     DWORD64 Dr3;
#     DWORD64 Dr6;
#     DWORD64 Dr7;
#
#     //
#     // Integer registers.
#     //
#
#     DWORD64 Rax;
#     DWORD64 Rcx;
#     DWORD64 Rdx;
#     DWORD64 Rbx;
#     DWORD64 Rsp;
#     DWORD64 Rbp;
#     DWORD64 Rsi;
#     DWORD64 Rdi;
#     DWORD64 R8;
#     DWORD64 R9;
#     DWORD64 R10;
#     DWORD64 R11;
#     DWORD64 R12;
#     DWORD64 R13;
#     DWORD64 R14;
#     DWORD64 R15;
#
#     //
#     // Program counter.
#     //
#
#     DWORD64 Rip;
#
#     //
#     // Floating point state.
#     //
#
#     union {
#         XMM_SAVE_AREA32 FltSave;
#         struct {
#             M128A Header[2];
#             M128A Legacy[8];
#             M128A Xmm0;
#             M128A Xmm1;
#             M128A Xmm2;
#             M128A Xmm3;
#             M128A Xmm4;
#             M128A Xmm5;
#             M128A Xmm6;
#             M128A Xmm7;
#             M128A Xmm8;
#             M128A Xmm9;
#             M128A Xmm10;
#             M128A Xmm11;
#             M128A Xmm12;
#             M128A Xmm13;
#             M128A Xmm14;
#             M128A Xmm15;
#         };
#     };
#
#     //
#     // Vector registers.
#     //
#
#     M128A VectorRegister[26];
#     DWORD64 VectorControl;
#
#     //
#     // Special debug control registers.
#     //
#
#     DWORD64 DebugControl;
#     DWORD64 LastBranchToRip;
#     DWORD64 LastBranchFromRip;
#     DWORD64 LastExceptionToRip;
#     DWORD64 LastExceptionFromRip;
# } CONTEXT, *PCONTEXT;

class _CONTEXT_FLTSAVE_STRUCT(Structure):
    _fields_ = [
        ('Header',                  M128A * 2),
        ('Legacy',                  M128A * 8),
        ('Xmm0',                    M128A),
        ('Xmm1',                    M128A),
        ('Xmm2',                    M128A),
        ('Xmm3',                    M128A),
        ('Xmm4',                    M128A),
        ('Xmm5',                    M128A),
        ('Xmm6',                    M128A),
        ('Xmm7',                    M128A),
        ('Xmm8',                    M128A),
        ('Xmm9',                    M128A),
        ('Xmm10',                   M128A),
        ('Xmm11',                   M128A),
        ('Xmm12',                   M128A),
        ('Xmm13',                   M128A),
        ('Xmm14',                   M128A),
        ('Xmm15',                   M128A),
    ]

    def from_dict(self):
        raise NotImplementedError()

    def to_dict(self):
        d = dict()
        for name, type in self._fields_:
            if name in ('Header', 'Legacy'):
                d[name] = tuple([ (x.Low + (x.High << 64)) for x in getattr(self, name) ])
            else:
                x = getattr(self, name)
                d[name] = x.Low + (x.High << 64)
        return d

class _CONTEXT_FLTSAVE_UNION(Union):
    _fields_ = [
        ('flt',                     XMM_SAVE_AREA32),
        ('xmm',                     _CONTEXT_FLTSAVE_STRUCT),
    ]

    def from_dict(self):
        raise NotImplementedError()

    def to_dict(self):
        d = dict()
        d['flt'] = self.flt.to_dict()
        d['xmm'] = self.xmm.to_dict()
        return d

class CONTEXT(Structure):
    arch = ARCH_AMD64

    _pack_ = 16
    _fields_ = [

        # Register parameter home addresses.
        ('P1Home',                  DWORD64),
        ('P2Home',                  DWORD64),
        ('P3Home',                  DWORD64),
        ('P4Home',                  DWORD64),
        ('P5Home',                  DWORD64),
        ('P6Home',                  DWORD64),

        # Control flags.
        ('ContextFlags',            DWORD),
        ('MxCsr',                   DWORD),

        # Segment Registers and processor flags.
        ('SegCs',                   WORD),
        ('SegDs',                   WORD),
        ('SegEs',                   WORD),
        ('SegFs',                   WORD),
        ('SegGs',                   WORD),
        ('SegSs',                   WORD),
        ('EFlags',                  DWORD),

        # Debug registers.
        ('Dr0',                     DWORD64),
        ('Dr1',                     DWORD64),
        ('Dr2',                     DWORD64),
        ('Dr3',                     DWORD64),
        ('Dr6',                     DWORD64),
        ('Dr7',                     DWORD64),

        # Integer registers.
        ('Rax',                     DWORD64),
        ('Rcx',                     DWORD64),
        ('Rdx',                     DWORD64),
        ('Rbx',                     DWORD64),
        ('Rsp',                     DWORD64),
        ('Rbp',                     DWORD64),
        ('Rsi',                     DWORD64),
        ('Rdi',                     DWORD64),
        ('R8',                      DWORD64),
        ('R9',                      DWORD64),
        ('R10',                     DWORD64),
        ('R11',                     DWORD64),
        ('R12',                     DWORD64),
        ('R13',                     DWORD64),
        ('R14',                     DWORD64),
        ('R15',                     DWORD64),

        # Program counter.
        ('Rip',                     DWORD64),

        # Floating point state.
        ('FltSave',                 _CONTEXT_FLTSAVE_UNION),

        # Vector registers.
        ('VectorRegister',          M128A * 26),
        ('VectorControl',           DWORD64),

        # Special debug control registers.
        ('DebugControl',            DWORD64),
        ('LastBranchToRip',         DWORD64),
        ('LastBranchFromRip',       DWORD64),
        ('LastExceptionToRip',      DWORD64),
        ('LastExceptionFromRip',    DWORD64),
    ]

    _others = ('P1Home', 'P2Home', 'P3Home', 'P4Home', 'P5Home', 'P6Home', \
               'MxCsr', 'VectorRegister', 'VectorControl')
    _control = ('SegSs', 'Rsp', 'SegCs', 'Rip', 'EFlags')
    _integer = ('Rax', 'Rcx', 'Rdx', 'Rbx', 'Rsp', 'Rbp', 'Rsi', 'Rdi', \
                'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15')
    _segments = ('SegDs', 'SegEs', 'SegFs', 'SegGs')
    _debug = ('Dr0', 'Dr1', 'Dr2', 'Dr3', 'Dr6', 'Dr7', \
              'DebugControl', 'LastBranchToRip', 'LastBranchFromRip', \
              'LastExceptionToRip', 'LastExceptionFromRip')
    _mmx = ('Xmm0', 'Xmm1', 'Xmm2', 'Xmm3', 'Xmm4', 'Xmm5', 'Xmm6', 'Xmm7', \
          'Xmm8', 'Xmm9', 'Xmm10', 'Xmm11', 'Xmm12', 'Xmm13', 'Xmm14', 'Xmm15')

    # XXX TODO
    # Convert VectorRegister and Xmm0-Xmm15 to pure Python types!

    @classmethod
    def from_dict(cls, ctx):
        'Instance a new structure from a Python native type.'
        ctx = Context(ctx)
        s = cls()
        ContextFlags = ctx['ContextFlags']
        s.ContextFlags = ContextFlags
        for key in cls._others:
            if key != 'VectorRegister':
                setattr(s, key, ctx[key])
            else:
                w = ctx[key]
                v = (M128A * len(w))()
                i = 0
                for x in w:
                    y = M128A()
                    y.High = x >> 64
                    y.Low = x - (x >> 64)
                    v[i] = y
                    i += 1
                setattr(s, key, v)
        if (ContextFlags & CONTEXT_CONTROL) == CONTEXT_CONTROL:
            for key in cls._control:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_INTEGER) == CONTEXT_INTEGER:
            for key in cls._integer:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_SEGMENTS) == CONTEXT_SEGMENTS:
            for key in cls._segments:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_DEBUG_REGISTERS) == CONTEXT_DEBUG_REGISTERS:
            for key in cls._debug:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_MMX_REGISTERS) == CONTEXT_MMX_REGISTERS:
            xmm = s.FltSave.xmm
            for key in cls._mmx:
                y = M128A()
                y.High = x >> 64
                y.Low = x - (x >> 64)
                setattr(xmm, key, y)
        return s

    def to_dict(self):
        'Convert a structure into a Python dictionary.'
        ctx = Context()
        ContextFlags = self.ContextFlags
        ctx['ContextFlags'] = ContextFlags
        for key in self._others:
            if key != 'VectorRegister':
                ctx[key] = getattr(self, key)
            else:
                ctx[key] = tuple([ (x.Low + (x.High << 64)) for x in getattr(self, key) ])
        if (ContextFlags & CONTEXT_CONTROL) == CONTEXT_CONTROL:
            for key in self._control:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_INTEGER) == CONTEXT_INTEGER:
            for key in self._integer:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_SEGMENTS) == CONTEXT_SEGMENTS:
            for key in self._segments:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_DEBUG_REGISTERS) == CONTEXT_DEBUG_REGISTERS:
            for key in self._debug:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_MMX_REGISTERS) == CONTEXT_MMX_REGISTERS:
            xmm = self.FltSave.xmm.to_dict()
            for key in self._mmx:
                ctx[key] = xmm.get(key)
        return ctx

PCONTEXT = ctypes.POINTER(CONTEXT)
LPCONTEXT = PCONTEXT

class Context(dict):
    """
    Register context dictionary for the amd64 architecture.
    """

    arch = CONTEXT.arch

    def __get_pc(self):
        return self['Rip']
    def __set_pc(self, value):
        self['Rip'] = value
    pc = property(__get_pc, __set_pc)

    def __get_sp(self):
        return self['Rsp']
    def __set_sp(self, value):
        self['Rsp'] = value
    sp = property(__get_sp, __set_sp)

    def __get_fp(self):
        return self['Rbp']
    def __set_fp(self, value):
        self['Rbp'] = value
    fp = property(__get_fp, __set_fp)

#--- LDT_ENTRY structure ------------------------------------------------------

# typedef struct _LDT_ENTRY {
#   WORD LimitLow;
#   WORD BaseLow;
#   union {
#     struct {
#       BYTE BaseMid;
#       BYTE Flags1;
#       BYTE Flags2;
#       BYTE BaseHi;
#     } Bytes;
#     struct {
#       DWORD BaseMid  :8;
#       DWORD Type  :5;
#       DWORD Dpl  :2;
#       DWORD Pres  :1;
#       DWORD LimitHi  :4;
#       DWORD Sys  :1;
#       DWORD Reserved_0  :1;
#       DWORD Default_Big  :1;
#       DWORD Granularity  :1;
#       DWORD BaseHi  :8;
#     } Bits;
#   } HighWord;
# } LDT_ENTRY,
#  *PLDT_ENTRY;

class _LDT_ENTRY_BYTES_(Structure):
    _pack_ = 1
    _fields_ = [
        ('BaseMid',         BYTE),
        ('Flags1',          BYTE),
        ('Flags2',          BYTE),
        ('BaseHi',          BYTE),
    ]

class _LDT_ENTRY_BITS_(Structure):
    _pack_ = 1
    _fields_ = [
        ('BaseMid',         DWORD,  8),
        ('Type',            DWORD,  5),
        ('Dpl',             DWORD,  2),
        ('Pres',            DWORD,  1),
        ('LimitHi',         DWORD,  4),
        ('Sys',             DWORD,  1),
        ('Reserved_0',      DWORD,  1),
        ('Default_Big',     DWORD,  1),
        ('Granularity',     DWORD,  1),
        ('BaseHi',          DWORD,  8),
    ]

class _LDT_ENTRY_HIGHWORD_(Union):
    _pack_ = 1
    _fields_ = [
        ('Bytes',           _LDT_ENTRY_BYTES_),
        ('Bits',            _LDT_ENTRY_BITS_),
    ]

class LDT_ENTRY(Structure):
    _pack_ = 1
    _fields_ = [
        ('LimitLow',        WORD),
        ('BaseLow',         WORD),
        ('HighWord',        _LDT_ENTRY_HIGHWORD_),
    ]

PLDT_ENTRY = POINTER(LDT_ENTRY)
LPLDT_ENTRY = PLDT_ENTRY

#--- WOW64 CONTEXT structure and constants ------------------------------------

# Value of SegCs in a Wow64 thread when running in 32 bits mode
WOW64_CS32 = 0x23

WOW64_CONTEXT_i386 = long(0x00010000)
WOW64_CONTEXT_i486 = long(0x00010000)

WOW64_CONTEXT_CONTROL               = (WOW64_CONTEXT_i386 | long(0x00000001))
WOW64_CONTEXT_INTEGER               = (WOW64_CONTEXT_i386 | long(0x00000002))
WOW64_CONTEXT_SEGMENTS              = (WOW64_CONTEXT_i386 | long(0x00000004))
WOW64_CONTEXT_FLOATING_POINT        = (WOW64_CONTEXT_i386 | long(0x00000008))
WOW64_CONTEXT_DEBUG_REGISTERS       = (WOW64_CONTEXT_i386 | long(0x00000010))
WOW64_CONTEXT_EXTENDED_REGISTERS    = (WOW64_CONTEXT_i386 | long(0x00000020))

WOW64_CONTEXT_FULL                  = (WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER | WOW64_CONTEXT_SEGMENTS)
WOW64_CONTEXT_ALL                   = (WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER | WOW64_CONTEXT_SEGMENTS | WOW64_CONTEXT_FLOATING_POINT | WOW64_CONTEXT_DEBUG_REGISTERS | WOW64_CONTEXT_EXTENDED_REGISTERS)

WOW64_SIZE_OF_80387_REGISTERS       = 80
WOW64_MAXIMUM_SUPPORTED_EXTENSION   = 512

class WOW64_FLOATING_SAVE_AREA (context_i386.FLOATING_SAVE_AREA):
    pass

class WOW64_CONTEXT (context_i386.CONTEXT):
    pass

class WOW64_LDT_ENTRY (context_i386.LDT_ENTRY):
    pass

PWOW64_FLOATING_SAVE_AREA   = POINTER(WOW64_FLOATING_SAVE_AREA)
PWOW64_CONTEXT              = POINTER(WOW64_CONTEXT)
PWOW64_LDT_ENTRY            = POINTER(WOW64_LDT_ENTRY)

###############################################################################

# BOOL WINAPI GetThreadSelectorEntry(
#   __in   HANDLE hThread,
#   __in   DWORD dwSelector,
#   __out  LPLDT_ENTRY lpSelectorEntry
# );
def GetThreadSelectorEntry(hThread, dwSelector):
    _GetThreadSelectorEntry = windll.kernel32.GetThreadSelectorEntry
    _GetThreadSelectorEntry.argtypes = [HANDLE, DWORD, LPLDT_ENTRY]
    _GetThreadSelectorEntry.restype  = bool
    _GetThreadSelectorEntry.errcheck = RaiseIfZero

    ldt = LDT_ENTRY()
    _GetThreadSelectorEntry(hThread, dwSelector, byref(ldt))
    return ldt

# BOOL WINAPI GetThreadContext(
#   __in     HANDLE hThread,
#   __inout  LPCONTEXT lpContext
# );
def GetThreadContext(hThread, ContextFlags = None, raw = False):
    _GetThreadContext = windll.kernel32.GetThreadContext
    _GetThreadContext.argtypes = [HANDLE, LPCONTEXT]
    _GetThreadContext.restype  = bool
    _GetThreadContext.errcheck = RaiseIfZero

    if ContextFlags is None:
        ContextFlags = CONTEXT_ALL | CONTEXT_AMD64
    Context = CONTEXT()
    Context.ContextFlags = ContextFlags
    _GetThreadContext(hThread, byref(Context))
    if raw:
        return Context
    return Context.to_dict()

# BOOL WINAPI SetThreadContext(
#   __in  HANDLE hThread,
#   __in  const CONTEXT* lpContext
# );
def SetThreadContext(hThread, lpContext):
    _SetThreadContext = windll.kernel32.SetThreadContext
    _SetThreadContext.argtypes = [HANDLE, LPCONTEXT]
    _SetThreadContext.restype  = bool
    _SetThreadContext.errcheck = RaiseIfZero

    if isinstance(lpContext, dict):
        lpContext = CONTEXT.from_dict(lpContext)
    _SetThreadContext(hThread, byref(lpContext))

# BOOL Wow64GetThreadSelectorEntry(
#   __in   HANDLE hThread,
#   __in   DWORD dwSelector,
#   __out  PWOW64_LDT_ENTRY lpSelectorEntry
# );
def Wow64GetThreadSelectorEntry(hThread, dwSelector):
    _Wow64GetThreadSelectorEntry = windll.kernel32.Wow64GetThreadSelectorEntry
    _Wow64GetThreadSelectorEntry.argtypes = [HANDLE, DWORD, PWOW64_LDT_ENTRY]
    _Wow64GetThreadSelectorEntry.restype  = bool
    _Wow64GetThreadSelectorEntry.errcheck = RaiseIfZero

    lpSelectorEntry = WOW64_LDT_ENTRY()
    _Wow64GetThreadSelectorEntry(hThread, dwSelector, byref(lpSelectorEntry))
    return lpSelectorEntry

# DWORD WINAPI Wow64ResumeThread(
#   __in  HANDLE hThread
# );
def Wow64ResumeThread(hThread):
    _Wow64ResumeThread = windll.kernel32.Wow64ResumeThread
    _Wow64ResumeThread.argtypes = [HANDLE]
    _Wow64ResumeThread.restype  = DWORD

    previousCount = _Wow64ResumeThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount

# DWORD WINAPI Wow64SuspendThread(
#   __in  HANDLE hThread
# );
def Wow64SuspendThread(hThread):
    _Wow64SuspendThread = windll.kernel32.Wow64SuspendThread
    _Wow64SuspendThread.argtypes = [HANDLE]
    _Wow64SuspendThread.restype  = DWORD

    previousCount = _Wow64SuspendThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount

# XXX TODO Use this http://www.nynaeve.net/Code/GetThreadWow64Context.cpp
# Also see http://www.woodmann.com/forum/archive/index.php/t-11162.html

# BOOL WINAPI Wow64GetThreadContext(
#   __in     HANDLE hThread,
#   __inout  PWOW64_CONTEXT lpContext
# );
def Wow64GetThreadContext(hThread, ContextFlags = None):
    _Wow64GetThreadContext = windll.kernel32.Wow64GetThreadContext
    _Wow64GetThreadContext.argtypes = [HANDLE, PWOW64_CONTEXT]
    _Wow64GetThreadContext.restype  = bool
    _Wow64GetThreadContext.errcheck = RaiseIfZero

    # XXX doesn't exist in XP 64 bits

    Context = WOW64_CONTEXT()
    if ContextFlags is None:
        Context.ContextFlags = WOW64_CONTEXT_ALL | WOW64_CONTEXT_i386
    else:
        Context.ContextFlags = ContextFlags
    _Wow64GetThreadContext(hThread, byref(Context))
    return Context.to_dict()

# BOOL WINAPI Wow64SetThreadContext(
#   __in  HANDLE hThread,
#   __in  const WOW64_CONTEXT *lpContext
# );
def Wow64SetThreadContext(hThread, lpContext):
    _Wow64SetThreadContext = windll.kernel32.Wow64SetThreadContext
    _Wow64SetThreadContext.argtypes = [HANDLE, PWOW64_CONTEXT]
    _Wow64SetThreadContext.restype  = bool
    _Wow64SetThreadContext.errcheck = RaiseIfZero

    # XXX doesn't exist in XP 64 bits

    if isinstance(lpContext, dict):
        lpContext = WOW64_CONTEXT.from_dict(lpContext)
    _Wow64SetThreadContext(hThread, byref(lpContext))

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
