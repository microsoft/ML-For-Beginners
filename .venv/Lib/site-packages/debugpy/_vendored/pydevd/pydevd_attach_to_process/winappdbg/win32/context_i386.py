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
CONTEXT structure for i386.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386

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

CONTEXT_i386                = 0x00010000    # this assumes that i386 and
CONTEXT_i486                = 0x00010000    # i486 have identical context records

CONTEXT_CONTROL             = (CONTEXT_i386 | long(0x00000001)) # SS:SP, CS:IP, FLAGS, BP
CONTEXT_INTEGER             = (CONTEXT_i386 | long(0x00000002)) # AX, BX, CX, DX, SI, DI
CONTEXT_SEGMENTS            = (CONTEXT_i386 | long(0x00000004)) # DS, ES, FS, GS
CONTEXT_FLOATING_POINT      = (CONTEXT_i386 | long(0x00000008)) # 387 state
CONTEXT_DEBUG_REGISTERS     = (CONTEXT_i386 | long(0x00000010)) # DB 0-3,6,7
CONTEXT_EXTENDED_REGISTERS  = (CONTEXT_i386 | long(0x00000020)) # cpu specific extensions

CONTEXT_FULL = (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS)

CONTEXT_ALL = (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS | \
                CONTEXT_FLOATING_POINT | CONTEXT_DEBUG_REGISTERS | \
                CONTEXT_EXTENDED_REGISTERS)

SIZE_OF_80387_REGISTERS     = 80
MAXIMUM_SUPPORTED_EXTENSION = 512

# typedef struct _FLOATING_SAVE_AREA {
#     DWORD   ControlWord;
#     DWORD   StatusWord;
#     DWORD   TagWord;
#     DWORD   ErrorOffset;
#     DWORD   ErrorSelector;
#     DWORD   DataOffset;
#     DWORD   DataSelector;
#     BYTE    RegisterArea[SIZE_OF_80387_REGISTERS];
#     DWORD   Cr0NpxState;
# } FLOATING_SAVE_AREA;
class FLOATING_SAVE_AREA(Structure):
    _pack_ = 1
    _fields_ = [
        ('ControlWord',     DWORD),
        ('StatusWord',      DWORD),
        ('TagWord',         DWORD),
        ('ErrorOffset',     DWORD),
        ('ErrorSelector',   DWORD),
        ('DataOffset',      DWORD),
        ('DataSelector',    DWORD),
        ('RegisterArea',    BYTE * SIZE_OF_80387_REGISTERS),
        ('Cr0NpxState',     DWORD),
    ]

    _integer_members = ('ControlWord', 'StatusWord', 'TagWord', 'ErrorOffset', 'ErrorSelector', 'DataOffset', 'DataSelector', 'Cr0NpxState')

    @classmethod
    def from_dict(cls, fsa):
        'Instance a new structure from a Python dictionary.'
        fsa = dict(fsa)
        s = cls()
        for key in cls._integer_members:
            setattr(s, key, fsa.get(key))
        ra = fsa.get('RegisterArea', None)
        if ra is not None:
            for index in compat.xrange(0, SIZE_OF_80387_REGISTERS):
                s.RegisterArea[index] = ra[index]
        return s

    def to_dict(self):
        'Convert a structure into a Python dictionary.'
        fsa = dict()
        for key in self._integer_members:
            fsa[key] = getattr(self, key)
        ra = [ self.RegisterArea[index] for index in compat.xrange(0, SIZE_OF_80387_REGISTERS) ]
        ra = tuple(ra)
        fsa['RegisterArea'] = ra
        return fsa

PFLOATING_SAVE_AREA = POINTER(FLOATING_SAVE_AREA)
LPFLOATING_SAVE_AREA = PFLOATING_SAVE_AREA

# typedef struct _CONTEXT {
#     DWORD ContextFlags;
#     DWORD   Dr0;
#     DWORD   Dr1;
#     DWORD   Dr2;
#     DWORD   Dr3;
#     DWORD   Dr6;
#     DWORD   Dr7;
#     FLOATING_SAVE_AREA FloatSave;
#     DWORD   SegGs;
#     DWORD   SegFs;
#     DWORD   SegEs;
#     DWORD   SegDs;
#     DWORD   Edi;
#     DWORD   Esi;
#     DWORD   Ebx;
#     DWORD   Edx;
#     DWORD   Ecx;
#     DWORD   Eax;
#     DWORD   Ebp;
#     DWORD   Eip;
#     DWORD   SegCs;
#     DWORD   EFlags;
#     DWORD   Esp;
#     DWORD   SegSs;
#     BYTE    ExtendedRegisters[MAXIMUM_SUPPORTED_EXTENSION];
# } CONTEXT;
class CONTEXT(Structure):
    arch = ARCH_I386

    _pack_ = 1

    # Context Frame
    #
    #  This frame has a several purposes: 1) it is used as an argument to
    #  NtContinue, 2) is is used to constuct a call frame for APC delivery,
    #  and 3) it is used in the user level thread creation routines.
    #
    #  The layout of the record conforms to a standard call frame.

    _fields_ = [

        # The flags values within this flag control the contents of
        # a CONTEXT record.
        #
        # If the context record is used as an input parameter, then
        # for each portion of the context record controlled by a flag
        # whose value is set, it is assumed that that portion of the
        # context record contains valid context. If the context record
        # is being used to modify a threads context, then only that
        # portion of the threads context will be modified.
        #
        # If the context record is used as an IN OUT parameter to capture
        # the context of a thread, then only those portions of the thread's
        # context corresponding to set flags will be returned.
        #
        # The context record is never used as an OUT only parameter.

        ('ContextFlags',        DWORD),

        # This section is specified/returned if CONTEXT_DEBUG_REGISTERS is
        # set in ContextFlags.  Note that CONTEXT_DEBUG_REGISTERS is NOT
        # included in CONTEXT_FULL.

        ('Dr0',                 DWORD),
        ('Dr1',                 DWORD),
        ('Dr2',                 DWORD),
        ('Dr3',                 DWORD),
        ('Dr6',                 DWORD),
        ('Dr7',                 DWORD),

        # This section is specified/returned if the
        # ContextFlags word contains the flag CONTEXT_FLOATING_POINT.

        ('FloatSave',           FLOATING_SAVE_AREA),

        # This section is specified/returned if the
        # ContextFlags word contains the flag CONTEXT_SEGMENTS.

        ('SegGs',               DWORD),
        ('SegFs',               DWORD),
        ('SegEs',               DWORD),
        ('SegDs',               DWORD),

        # This section is specified/returned if the
        # ContextFlags word contains the flag CONTEXT_INTEGER.

        ('Edi',                 DWORD),
        ('Esi',                 DWORD),
        ('Ebx',                 DWORD),
        ('Edx',                 DWORD),
        ('Ecx',                 DWORD),
        ('Eax',                 DWORD),

        # This section is specified/returned if the
        # ContextFlags word contains the flag CONTEXT_CONTROL.

        ('Ebp',                 DWORD),
        ('Eip',                 DWORD),
        ('SegCs',               DWORD),         # MUST BE SANITIZED
        ('EFlags',              DWORD),         # MUST BE SANITIZED
        ('Esp',                 DWORD),
        ('SegSs',               DWORD),

        # This section is specified/returned if the ContextFlags word
        # contains the flag CONTEXT_EXTENDED_REGISTERS.
        # The format and contexts are processor specific.

        ('ExtendedRegisters',   BYTE * MAXIMUM_SUPPORTED_EXTENSION),
    ]

    _ctx_debug   = ('Dr0', 'Dr1', 'Dr2', 'Dr3', 'Dr6', 'Dr7')
    _ctx_segs    = ('SegGs', 'SegFs', 'SegEs', 'SegDs', )
    _ctx_int     = ('Edi', 'Esi', 'Ebx', 'Edx', 'Ecx', 'Eax')
    _ctx_ctrl    = ('Ebp', 'Eip', 'SegCs', 'EFlags', 'Esp', 'SegSs')

    @classmethod
    def from_dict(cls, ctx):
        'Instance a new structure from a Python dictionary.'
        ctx = Context(ctx)
        s = cls()
        ContextFlags = ctx['ContextFlags']
        setattr(s, 'ContextFlags', ContextFlags)
        if (ContextFlags & CONTEXT_DEBUG_REGISTERS) == CONTEXT_DEBUG_REGISTERS:
            for key in s._ctx_debug:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_FLOATING_POINT) == CONTEXT_FLOATING_POINT:
            fsa = ctx['FloatSave']
            s.FloatSave = FLOATING_SAVE_AREA.from_dict(fsa)
        if (ContextFlags & CONTEXT_SEGMENTS) == CONTEXT_SEGMENTS:
            for key in s._ctx_segs:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_INTEGER) == CONTEXT_INTEGER:
            for key in s._ctx_int:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_CONTROL) == CONTEXT_CONTROL:
            for key in s._ctx_ctrl:
                setattr(s, key, ctx[key])
        if (ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS:
            er = ctx['ExtendedRegisters']
            for index in compat.xrange(0, MAXIMUM_SUPPORTED_EXTENSION):
                s.ExtendedRegisters[index] = er[index]
        return s

    def to_dict(self):
        'Convert a structure into a Python native type.'
        ctx = Context()
        ContextFlags = self.ContextFlags
        ctx['ContextFlags'] = ContextFlags
        if (ContextFlags & CONTEXT_DEBUG_REGISTERS) == CONTEXT_DEBUG_REGISTERS:
            for key in self._ctx_debug:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_FLOATING_POINT) == CONTEXT_FLOATING_POINT:
            ctx['FloatSave'] = self.FloatSave.to_dict()
        if (ContextFlags & CONTEXT_SEGMENTS) == CONTEXT_SEGMENTS:
            for key in self._ctx_segs:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_INTEGER) == CONTEXT_INTEGER:
            for key in self._ctx_int:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_CONTROL) == CONTEXT_CONTROL:
            for key in self._ctx_ctrl:
                ctx[key] = getattr(self, key)
        if (ContextFlags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS:
            er = [ self.ExtendedRegisters[index] for index in compat.xrange(0, MAXIMUM_SUPPORTED_EXTENSION) ]
            er = tuple(er)
            ctx['ExtendedRegisters'] = er
        return ctx

PCONTEXT = POINTER(CONTEXT)
LPCONTEXT = PCONTEXT

class Context(dict):
    """
    Register context dictionary for the i386 architecture.
    """

    arch = CONTEXT.arch

    def __get_pc(self):
        return self['Eip']
    def __set_pc(self, value):
        self['Eip'] = value
    pc = property(__get_pc, __set_pc)

    def __get_sp(self):
        return self['Esp']
    def __set_sp(self, value):
        self['Esp'] = value
    sp = property(__get_sp, __set_sp)

    def __get_fp(self):
        return self['Ebp']
    def __set_fp(self, value):
        self['Ebp'] = value
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
        ContextFlags = CONTEXT_ALL | CONTEXT_i386
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

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
