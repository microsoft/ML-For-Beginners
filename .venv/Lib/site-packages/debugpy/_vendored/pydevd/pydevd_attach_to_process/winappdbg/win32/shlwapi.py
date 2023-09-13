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
Wrapper for shlwapi.dll in ctypes.
"""

__revision__ = "$Id$"

from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *

#==============================================================================
# This is used later on to calculate the list of exported symbols.
_all = None
_all = set(vars().keys())
#==============================================================================

OS_WINDOWS                  = 0
OS_NT                       = 1
OS_WIN95ORGREATER           = 2
OS_NT4ORGREATER             = 3
OS_WIN98ORGREATER           = 5
OS_WIN98_GOLD               = 6
OS_WIN2000ORGREATER         = 7
OS_WIN2000PRO               = 8
OS_WIN2000SERVER            = 9
OS_WIN2000ADVSERVER         = 10
OS_WIN2000DATACENTER        = 11
OS_WIN2000TERMINAL          = 12
OS_EMBEDDED                 = 13
OS_TERMINALCLIENT           = 14
OS_TERMINALREMOTEADMIN      = 15
OS_WIN95_GOLD               = 16
OS_MEORGREATER              = 17
OS_XPORGREATER              = 18
OS_HOME                     = 19
OS_PROFESSIONAL             = 20
OS_DATACENTER               = 21
OS_ADVSERVER                = 22
OS_SERVER                   = 23
OS_TERMINALSERVER           = 24
OS_PERSONALTERMINALSERVER   = 25
OS_FASTUSERSWITCHING        = 26
OS_WELCOMELOGONUI           = 27
OS_DOMAINMEMBER             = 28
OS_ANYSERVER                = 29
OS_WOW6432                  = 30
OS_WEBSERVER                = 31
OS_SMALLBUSINESSSERVER      = 32
OS_TABLETPC                 = 33
OS_SERVERADMINUI            = 34
OS_MEDIACENTER              = 35
OS_APPLIANCE                = 36

#--- shlwapi.dll --------------------------------------------------------------

# BOOL IsOS(
#     DWORD dwOS
# );
def IsOS(dwOS):
    try:
        _IsOS = windll.shlwapi.IsOS
        _IsOS.argtypes = [DWORD]
        _IsOS.restype  = bool
    except AttributeError:
        # According to MSDN, on Windows versions prior to Vista
        # this function is exported only by ordinal number 437.
        # http://msdn.microsoft.com/en-us/library/bb773795%28VS.85%29.aspx
        _GetProcAddress = windll.kernel32.GetProcAddress
        _GetProcAddress.argtypes = [HINSTANCE, DWORD]
        _GetProcAddress.restype  = LPVOID
        _IsOS = windll.kernel32.GetProcAddress(windll.shlwapi._handle, 437)
        _IsOS = WINFUNCTYPE(bool, DWORD)(_IsOS)
    return _IsOS(dwOS)

# LPTSTR PathAddBackslash(
#     LPTSTR lpszPath
# );
def PathAddBackslashA(lpszPath):
    _PathAddBackslashA = windll.shlwapi.PathAddBackslashA
    _PathAddBackslashA.argtypes = [LPSTR]
    _PathAddBackslashA.restype  = LPSTR

    lpszPath = ctypes.create_string_buffer(lpszPath, MAX_PATH)
    retval = _PathAddBackslashA(lpszPath)
    if retval == NULL:
        raise ctypes.WinError()
    return lpszPath.value

def PathAddBackslashW(lpszPath):
    _PathAddBackslashW = windll.shlwapi.PathAddBackslashW
    _PathAddBackslashW.argtypes = [LPWSTR]
    _PathAddBackslashW.restype  = LPWSTR

    lpszPath = ctypes.create_unicode_buffer(lpszPath, MAX_PATH)
    retval = _PathAddBackslashW(lpszPath)
    if retval == NULL:
        raise ctypes.WinError()
    return lpszPath.value

PathAddBackslash = GuessStringType(PathAddBackslashA, PathAddBackslashW)

# BOOL PathAddExtension(
#     LPTSTR pszPath,
#     LPCTSTR pszExtension
# );
def PathAddExtensionA(lpszPath, pszExtension = None):
    _PathAddExtensionA = windll.shlwapi.PathAddExtensionA
    _PathAddExtensionA.argtypes = [LPSTR, LPSTR]
    _PathAddExtensionA.restype  = bool
    _PathAddExtensionA.errcheck = RaiseIfZero

    if not pszExtension:
        pszExtension = None
    lpszPath = ctypes.create_string_buffer(lpszPath, MAX_PATH)
    _PathAddExtensionA(lpszPath, pszExtension)
    return lpszPath.value

def PathAddExtensionW(lpszPath, pszExtension = None):
    _PathAddExtensionW = windll.shlwapi.PathAddExtensionW
    _PathAddExtensionW.argtypes = [LPWSTR, LPWSTR]
    _PathAddExtensionW.restype  = bool
    _PathAddExtensionW.errcheck = RaiseIfZero

    if not pszExtension:
        pszExtension = None
    lpszPath = ctypes.create_unicode_buffer(lpszPath, MAX_PATH)
    _PathAddExtensionW(lpszPath, pszExtension)
    return lpszPath.value

PathAddExtension = GuessStringType(PathAddExtensionA, PathAddExtensionW)

# BOOL PathAppend(
#     LPTSTR pszPath,
#     LPCTSTR pszMore
# );
def PathAppendA(lpszPath, pszMore = None):
    _PathAppendA = windll.shlwapi.PathAppendA
    _PathAppendA.argtypes = [LPSTR, LPSTR]
    _PathAppendA.restype  = bool
    _PathAppendA.errcheck = RaiseIfZero

    if not pszMore:
        pszMore = None
    lpszPath = ctypes.create_string_buffer(lpszPath, MAX_PATH)
    _PathAppendA(lpszPath, pszMore)
    return lpszPath.value

def PathAppendW(lpszPath, pszMore = None):
    _PathAppendW = windll.shlwapi.PathAppendW
    _PathAppendW.argtypes = [LPWSTR, LPWSTR]
    _PathAppendW.restype  = bool
    _PathAppendW.errcheck = RaiseIfZero

    if not pszMore:
        pszMore = None
    lpszPath = ctypes.create_unicode_buffer(lpszPath, MAX_PATH)
    _PathAppendW(lpszPath, pszMore)
    return lpszPath.value

PathAppend = GuessStringType(PathAppendA, PathAppendW)

# LPTSTR PathCombine(
#     LPTSTR lpszDest,
#     LPCTSTR lpszDir,
#     LPCTSTR lpszFile
# );
def PathCombineA(lpszDir, lpszFile):
    _PathCombineA = windll.shlwapi.PathCombineA
    _PathCombineA.argtypes = [LPSTR, LPSTR, LPSTR]
    _PathCombineA.restype  = LPSTR

    lpszDest = ctypes.create_string_buffer("", max(MAX_PATH, len(lpszDir) + len(lpszFile) + 1))
    retval = _PathCombineA(lpszDest, lpszDir, lpszFile)
    if retval == NULL:
        return None
    return lpszDest.value

def PathCombineW(lpszDir, lpszFile):
    _PathCombineW = windll.shlwapi.PathCombineW
    _PathCombineW.argtypes = [LPWSTR, LPWSTR, LPWSTR]
    _PathCombineW.restype  = LPWSTR

    lpszDest = ctypes.create_unicode_buffer(u"", max(MAX_PATH, len(lpszDir) + len(lpszFile) + 1))
    retval = _PathCombineW(lpszDest, lpszDir, lpszFile)
    if retval == NULL:
        return None
    return lpszDest.value

PathCombine = GuessStringType(PathCombineA, PathCombineW)

# BOOL PathCanonicalize(
#     LPTSTR lpszDst,
#     LPCTSTR lpszSrc
# );
def PathCanonicalizeA(lpszSrc):
    _PathCanonicalizeA = windll.shlwapi.PathCanonicalizeA
    _PathCanonicalizeA.argtypes = [LPSTR, LPSTR]
    _PathCanonicalizeA.restype  = bool
    _PathCanonicalizeA.errcheck = RaiseIfZero

    lpszDst = ctypes.create_string_buffer("", MAX_PATH)
    _PathCanonicalizeA(lpszDst, lpszSrc)
    return lpszDst.value

def PathCanonicalizeW(lpszSrc):
    _PathCanonicalizeW = windll.shlwapi.PathCanonicalizeW
    _PathCanonicalizeW.argtypes = [LPWSTR, LPWSTR]
    _PathCanonicalizeW.restype  = bool
    _PathCanonicalizeW.errcheck = RaiseIfZero

    lpszDst = ctypes.create_unicode_buffer(u"", MAX_PATH)
    _PathCanonicalizeW(lpszDst, lpszSrc)
    return lpszDst.value

PathCanonicalize = GuessStringType(PathCanonicalizeA, PathCanonicalizeW)

# BOOL PathRelativePathTo(
#   _Out_  LPTSTR pszPath,
#   _In_   LPCTSTR pszFrom,
#   _In_   DWORD dwAttrFrom,
#   _In_   LPCTSTR pszTo,
#   _In_   DWORD dwAttrTo
# );
def PathRelativePathToA(pszFrom = None, dwAttrFrom = FILE_ATTRIBUTE_DIRECTORY, pszTo = None, dwAttrTo = FILE_ATTRIBUTE_DIRECTORY):
    _PathRelativePathToA = windll.shlwapi.PathRelativePathToA
    _PathRelativePathToA.argtypes = [LPSTR, LPSTR, DWORD, LPSTR, DWORD]
    _PathRelativePathToA.restype  = bool
    _PathRelativePathToA.errcheck = RaiseIfZero

    # Make the paths absolute or the function fails.
    if pszFrom:
        pszFrom = GetFullPathNameA(pszFrom)[0]
    else:
        pszFrom = GetCurrentDirectoryA()
    if pszTo:
        pszTo = GetFullPathNameA(pszTo)[0]
    else:
        pszTo = GetCurrentDirectoryA()

    # Argh, this function doesn't receive an output buffer size!
    # We'll try to guess the maximum possible buffer size.
    dwPath = max((len(pszFrom) + len(pszTo)) * 2 + 1, MAX_PATH + 1)
    pszPath = ctypes.create_string_buffer('', dwPath)

    # Also, it doesn't set the last error value.
    # Whoever coded it must have been drunk or tripping on acid. Or both.
    # The only failure conditions I've seen were invalid paths, paths not
    # on the same drive, or the path is not absolute.
    SetLastError(ERROR_INVALID_PARAMETER)

    _PathRelativePathToA(pszPath, pszFrom, dwAttrFrom, pszTo, dwAttrTo)
    return pszPath.value

def PathRelativePathToW(pszFrom = None, dwAttrFrom = FILE_ATTRIBUTE_DIRECTORY, pszTo = None, dwAttrTo = FILE_ATTRIBUTE_DIRECTORY):
    _PathRelativePathToW = windll.shlwapi.PathRelativePathToW
    _PathRelativePathToW.argtypes = [LPWSTR, LPWSTR, DWORD, LPWSTR, DWORD]
    _PathRelativePathToW.restype  = bool
    _PathRelativePathToW.errcheck = RaiseIfZero

    # Refer to PathRelativePathToA to know why this code is so ugly.
    if pszFrom:
        pszFrom = GetFullPathNameW(pszFrom)[0]
    else:
        pszFrom = GetCurrentDirectoryW()
    if pszTo:
        pszTo = GetFullPathNameW(pszTo)[0]
    else:
        pszTo = GetCurrentDirectoryW()
    dwPath = max((len(pszFrom) + len(pszTo)) * 2 + 1, MAX_PATH + 1)
    pszPath = ctypes.create_unicode_buffer(u'', dwPath)
    SetLastError(ERROR_INVALID_PARAMETER)
    _PathRelativePathToW(pszPath, pszFrom, dwAttrFrom, pszTo, dwAttrTo)
    return pszPath.value

PathRelativePathTo = GuessStringType(PathRelativePathToA, PathRelativePathToW)

# BOOL PathFileExists(
#     LPCTSTR pszPath
# );
def PathFileExistsA(pszPath):
    _PathFileExistsA = windll.shlwapi.PathFileExistsA
    _PathFileExistsA.argtypes = [LPSTR]
    _PathFileExistsA.restype  = bool
    return _PathFileExistsA(pszPath)

def PathFileExistsW(pszPath):
    _PathFileExistsW = windll.shlwapi.PathFileExistsW
    _PathFileExistsW.argtypes = [LPWSTR]
    _PathFileExistsW.restype  = bool
    return _PathFileExistsW(pszPath)

PathFileExists = GuessStringType(PathFileExistsA, PathFileExistsW)

# LPTSTR PathFindExtension(
#     LPCTSTR pszPath
# );
def PathFindExtensionA(pszPath):
    _PathFindExtensionA = windll.shlwapi.PathFindExtensionA
    _PathFindExtensionA.argtypes = [LPSTR]
    _PathFindExtensionA.restype  = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathFindExtensionA(pszPath)

def PathFindExtensionW(pszPath):
    _PathFindExtensionW = windll.shlwapi.PathFindExtensionW
    _PathFindExtensionW.argtypes = [LPWSTR]
    _PathFindExtensionW.restype  = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathFindExtensionW(pszPath)

PathFindExtension = GuessStringType(PathFindExtensionA, PathFindExtensionW)

# LPTSTR PathFindFileName(
#     LPCTSTR pszPath
# );
def PathFindFileNameA(pszPath):
    _PathFindFileNameA = windll.shlwapi.PathFindFileNameA
    _PathFindFileNameA.argtypes = [LPSTR]
    _PathFindFileNameA.restype  = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathFindFileNameA(pszPath)

def PathFindFileNameW(pszPath):
    _PathFindFileNameW = windll.shlwapi.PathFindFileNameW
    _PathFindFileNameW.argtypes = [LPWSTR]
    _PathFindFileNameW.restype  = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathFindFileNameW(pszPath)

PathFindFileName = GuessStringType(PathFindFileNameA, PathFindFileNameW)

# LPTSTR PathFindNextComponent(
#     LPCTSTR pszPath
# );
def PathFindNextComponentA(pszPath):
    _PathFindNextComponentA = windll.shlwapi.PathFindNextComponentA
    _PathFindNextComponentA.argtypes = [LPSTR]
    _PathFindNextComponentA.restype  = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathFindNextComponentA(pszPath)

def PathFindNextComponentW(pszPath):
    _PathFindNextComponentW = windll.shlwapi.PathFindNextComponentW
    _PathFindNextComponentW.argtypes = [LPWSTR]
    _PathFindNextComponentW.restype  = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathFindNextComponentW(pszPath)

PathFindNextComponent = GuessStringType(PathFindNextComponentA, PathFindNextComponentW)

# BOOL PathFindOnPath(
#     LPTSTR pszFile,
#     LPCTSTR *ppszOtherDirs
# );
def PathFindOnPathA(pszFile, ppszOtherDirs = None):
    _PathFindOnPathA = windll.shlwapi.PathFindOnPathA
    _PathFindOnPathA.argtypes = [LPSTR, LPSTR]
    _PathFindOnPathA.restype  = bool

    pszFile = ctypes.create_string_buffer(pszFile, MAX_PATH)
    if not ppszOtherDirs:
        ppszOtherDirs = None
    else:
        szArray = ""
        for pszOtherDirs in ppszOtherDirs:
            if pszOtherDirs:
                szArray = "%s%s\0" % (szArray, pszOtherDirs)
        szArray = szArray + "\0"
        pszOtherDirs = ctypes.create_string_buffer(szArray)
        ppszOtherDirs = ctypes.pointer(pszOtherDirs)
    if _PathFindOnPathA(pszFile, ppszOtherDirs):
        return pszFile.value
    return None

def PathFindOnPathW(pszFile, ppszOtherDirs = None):
    _PathFindOnPathW = windll.shlwapi.PathFindOnPathA
    _PathFindOnPathW.argtypes = [LPWSTR, LPWSTR]
    _PathFindOnPathW.restype  = bool

    pszFile = ctypes.create_unicode_buffer(pszFile, MAX_PATH)
    if not ppszOtherDirs:
        ppszOtherDirs = None
    else:
        szArray = u""
        for pszOtherDirs in ppszOtherDirs:
            if pszOtherDirs:
                szArray = u"%s%s\0" % (szArray, pszOtherDirs)
        szArray = szArray + u"\0"
        pszOtherDirs = ctypes.create_unicode_buffer(szArray)
        ppszOtherDirs = ctypes.pointer(pszOtherDirs)
    if _PathFindOnPathW(pszFile, ppszOtherDirs):
        return pszFile.value
    return None

PathFindOnPath = GuessStringType(PathFindOnPathA, PathFindOnPathW)

# LPTSTR PathGetArgs(
#     LPCTSTR pszPath
# );
def PathGetArgsA(pszPath):
    _PathGetArgsA = windll.shlwapi.PathGetArgsA
    _PathGetArgsA.argtypes = [LPSTR]
    _PathGetArgsA.restype  = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathGetArgsA(pszPath)

def PathGetArgsW(pszPath):
    _PathGetArgsW = windll.shlwapi.PathGetArgsW
    _PathGetArgsW.argtypes = [LPWSTR]
    _PathGetArgsW.restype  = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathGetArgsW(pszPath)

PathGetArgs = GuessStringType(PathGetArgsA, PathGetArgsW)

# BOOL PathIsContentType(
#     LPCTSTR pszPath,
#     LPCTSTR pszContentType
# );
def PathIsContentTypeA(pszPath, pszContentType):
    _PathIsContentTypeA = windll.shlwapi.PathIsContentTypeA
    _PathIsContentTypeA.argtypes = [LPSTR, LPSTR]
    _PathIsContentTypeA.restype  = bool
    return _PathIsContentTypeA(pszPath, pszContentType)

def PathIsContentTypeW(pszPath, pszContentType):
    _PathIsContentTypeW = windll.shlwapi.PathIsContentTypeW
    _PathIsContentTypeW.argtypes = [LPWSTR, LPWSTR]
    _PathIsContentTypeW.restype  = bool
    return _PathIsContentTypeW(pszPath, pszContentType)

PathIsContentType = GuessStringType(PathIsContentTypeA, PathIsContentTypeW)

# BOOL PathIsDirectory(
#     LPCTSTR pszPath
# );
def PathIsDirectoryA(pszPath):
    _PathIsDirectoryA = windll.shlwapi.PathIsDirectoryA
    _PathIsDirectoryA.argtypes = [LPSTR]
    _PathIsDirectoryA.restype  = bool
    return _PathIsDirectoryA(pszPath)

def PathIsDirectoryW(pszPath):
    _PathIsDirectoryW = windll.shlwapi.PathIsDirectoryW
    _PathIsDirectoryW.argtypes = [LPWSTR]
    _PathIsDirectoryW.restype  = bool
    return _PathIsDirectoryW(pszPath)

PathIsDirectory = GuessStringType(PathIsDirectoryA, PathIsDirectoryW)

# BOOL PathIsDirectoryEmpty(
#     LPCTSTR pszPath
# );
def PathIsDirectoryEmptyA(pszPath):
    _PathIsDirectoryEmptyA = windll.shlwapi.PathIsDirectoryEmptyA
    _PathIsDirectoryEmptyA.argtypes = [LPSTR]
    _PathIsDirectoryEmptyA.restype  = bool
    return _PathIsDirectoryEmptyA(pszPath)

def PathIsDirectoryEmptyW(pszPath):
    _PathIsDirectoryEmptyW = windll.shlwapi.PathIsDirectoryEmptyW
    _PathIsDirectoryEmptyW.argtypes = [LPWSTR]
    _PathIsDirectoryEmptyW.restype  = bool
    return _PathIsDirectoryEmptyW(pszPath)

PathIsDirectoryEmpty = GuessStringType(PathIsDirectoryEmptyA, PathIsDirectoryEmptyW)

# BOOL PathIsNetworkPath(
#     LPCTSTR pszPath
# );
def PathIsNetworkPathA(pszPath):
    _PathIsNetworkPathA = windll.shlwapi.PathIsNetworkPathA
    _PathIsNetworkPathA.argtypes = [LPSTR]
    _PathIsNetworkPathA.restype  = bool
    return _PathIsNetworkPathA(pszPath)

def PathIsNetworkPathW(pszPath):
    _PathIsNetworkPathW = windll.shlwapi.PathIsNetworkPathW
    _PathIsNetworkPathW.argtypes = [LPWSTR]
    _PathIsNetworkPathW.restype  = bool
    return _PathIsNetworkPathW(pszPath)

PathIsNetworkPath = GuessStringType(PathIsNetworkPathA, PathIsNetworkPathW)

# BOOL PathIsRelative(
#     LPCTSTR lpszPath
# );
def PathIsRelativeA(pszPath):
    _PathIsRelativeA = windll.shlwapi.PathIsRelativeA
    _PathIsRelativeA.argtypes = [LPSTR]
    _PathIsRelativeA.restype  = bool
    return _PathIsRelativeA(pszPath)

def PathIsRelativeW(pszPath):
    _PathIsRelativeW = windll.shlwapi.PathIsRelativeW
    _PathIsRelativeW.argtypes = [LPWSTR]
    _PathIsRelativeW.restype  = bool
    return _PathIsRelativeW(pszPath)

PathIsRelative = GuessStringType(PathIsRelativeA, PathIsRelativeW)

# BOOL PathIsRoot(
#     LPCTSTR pPath
# );
def PathIsRootA(pszPath):
    _PathIsRootA = windll.shlwapi.PathIsRootA
    _PathIsRootA.argtypes = [LPSTR]
    _PathIsRootA.restype  = bool
    return _PathIsRootA(pszPath)

def PathIsRootW(pszPath):
    _PathIsRootW = windll.shlwapi.PathIsRootW
    _PathIsRootW.argtypes = [LPWSTR]
    _PathIsRootW.restype  = bool
    return _PathIsRootW(pszPath)

PathIsRoot = GuessStringType(PathIsRootA, PathIsRootW)

# BOOL PathIsSameRoot(
#     LPCTSTR pszPath1,
#     LPCTSTR pszPath2
# );
def PathIsSameRootA(pszPath1, pszPath2):
    _PathIsSameRootA = windll.shlwapi.PathIsSameRootA
    _PathIsSameRootA.argtypes = [LPSTR, LPSTR]
    _PathIsSameRootA.restype  = bool
    return _PathIsSameRootA(pszPath1, pszPath2)

def PathIsSameRootW(pszPath1, pszPath2):
    _PathIsSameRootW = windll.shlwapi.PathIsSameRootW
    _PathIsSameRootW.argtypes = [LPWSTR, LPWSTR]
    _PathIsSameRootW.restype  = bool
    return _PathIsSameRootW(pszPath1, pszPath2)

PathIsSameRoot = GuessStringType(PathIsSameRootA, PathIsSameRootW)

# BOOL PathIsUNC(
#     LPCTSTR pszPath
# );
def PathIsUNCA(pszPath):
    _PathIsUNCA = windll.shlwapi.PathIsUNCA
    _PathIsUNCA.argtypes = [LPSTR]
    _PathIsUNCA.restype  = bool
    return _PathIsUNCA(pszPath)

def PathIsUNCW(pszPath):
    _PathIsUNCW = windll.shlwapi.PathIsUNCW
    _PathIsUNCW.argtypes = [LPWSTR]
    _PathIsUNCW.restype  = bool
    return _PathIsUNCW(pszPath)

PathIsUNC = GuessStringType(PathIsUNCA, PathIsUNCW)

# XXX WARNING
# PathMakePretty turns filenames into all lowercase.
# I'm not sure how well that might work on Wine.

# BOOL PathMakePretty(
#     LPCTSTR pszPath
# );
def PathMakePrettyA(pszPath):
    _PathMakePrettyA = windll.shlwapi.PathMakePrettyA
    _PathMakePrettyA.argtypes = [LPSTR]
    _PathMakePrettyA.restype  = bool
    _PathMakePrettyA.errcheck = RaiseIfZero

    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathMakePrettyA(pszPath)
    return pszPath.value

def PathMakePrettyW(pszPath):
    _PathMakePrettyW = windll.shlwapi.PathMakePrettyW
    _PathMakePrettyW.argtypes = [LPWSTR]
    _PathMakePrettyW.restype  = bool
    _PathMakePrettyW.errcheck = RaiseIfZero

    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathMakePrettyW(pszPath)
    return pszPath.value

PathMakePretty = GuessStringType(PathMakePrettyA, PathMakePrettyW)

# void PathRemoveArgs(
#     LPTSTR pszPath
# );
def PathRemoveArgsA(pszPath):
    _PathRemoveArgsA = windll.shlwapi.PathRemoveArgsA
    _PathRemoveArgsA.argtypes = [LPSTR]

    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveArgsA(pszPath)
    return pszPath.value

def PathRemoveArgsW(pszPath):
    _PathRemoveArgsW = windll.shlwapi.PathRemoveArgsW
    _PathRemoveArgsW.argtypes = [LPWSTR]

    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveArgsW(pszPath)
    return pszPath.value

PathRemoveArgs = GuessStringType(PathRemoveArgsA, PathRemoveArgsW)

# void PathRemoveBackslash(
#     LPTSTR pszPath
# );
def PathRemoveBackslashA(pszPath):
    _PathRemoveBackslashA = windll.shlwapi.PathRemoveBackslashA
    _PathRemoveBackslashA.argtypes = [LPSTR]

    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveBackslashA(pszPath)
    return pszPath.value

def PathRemoveBackslashW(pszPath):
    _PathRemoveBackslashW = windll.shlwapi.PathRemoveBackslashW
    _PathRemoveBackslashW.argtypes = [LPWSTR]

    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveBackslashW(pszPath)
    return pszPath.value

PathRemoveBackslash = GuessStringType(PathRemoveBackslashA, PathRemoveBackslashW)

# void PathRemoveExtension(
#     LPTSTR pszPath
# );
def PathRemoveExtensionA(pszPath):
    _PathRemoveExtensionA = windll.shlwapi.PathRemoveExtensionA
    _PathRemoveExtensionA.argtypes = [LPSTR]

    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveExtensionA(pszPath)
    return pszPath.value

def PathRemoveExtensionW(pszPath):
    _PathRemoveExtensionW = windll.shlwapi.PathRemoveExtensionW
    _PathRemoveExtensionW.argtypes = [LPWSTR]

    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveExtensionW(pszPath)
    return pszPath.value

PathRemoveExtension = GuessStringType(PathRemoveExtensionA, PathRemoveExtensionW)

# void PathRemoveFileSpec(
#     LPTSTR pszPath
# );
def PathRemoveFileSpecA(pszPath):
    _PathRemoveFileSpecA = windll.shlwapi.PathRemoveFileSpecA
    _PathRemoveFileSpecA.argtypes = [LPSTR]

    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveFileSpecA(pszPath)
    return pszPath.value

def PathRemoveFileSpecW(pszPath):
    _PathRemoveFileSpecW = windll.shlwapi.PathRemoveFileSpecW
    _PathRemoveFileSpecW.argtypes = [LPWSTR]

    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveFileSpecW(pszPath)
    return pszPath.value

PathRemoveFileSpec = GuessStringType(PathRemoveFileSpecA, PathRemoveFileSpecW)

# BOOL PathRenameExtension(
#     LPTSTR pszPath,
#     LPCTSTR pszExt
# );
def PathRenameExtensionA(pszPath, pszExt):
    _PathRenameExtensionA = windll.shlwapi.PathRenameExtensionA
    _PathRenameExtensionA.argtypes = [LPSTR, LPSTR]
    _PathRenameExtensionA.restype  = bool

    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    if _PathRenameExtensionA(pszPath, pszExt):
        return pszPath.value
    return None

def PathRenameExtensionW(pszPath, pszExt):
    _PathRenameExtensionW = windll.shlwapi.PathRenameExtensionW
    _PathRenameExtensionW.argtypes = [LPWSTR, LPWSTR]
    _PathRenameExtensionW.restype  = bool

    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    if _PathRenameExtensionW(pszPath, pszExt):
        return pszPath.value
    return None

PathRenameExtension = GuessStringType(PathRenameExtensionA, PathRenameExtensionW)

# BOOL PathUnExpandEnvStrings(
#     LPCTSTR pszPath,
#     LPTSTR pszBuf,
#     UINT cchBuf
# );
def PathUnExpandEnvStringsA(pszPath):
    _PathUnExpandEnvStringsA = windll.shlwapi.PathUnExpandEnvStringsA
    _PathUnExpandEnvStringsA.argtypes = [LPSTR, LPSTR]
    _PathUnExpandEnvStringsA.restype  = bool
    _PathUnExpandEnvStringsA.errcheck = RaiseIfZero

    cchBuf = MAX_PATH
    pszBuf = ctypes.create_string_buffer("", cchBuf)
    _PathUnExpandEnvStringsA(pszPath, pszBuf, cchBuf)
    return pszBuf.value

def PathUnExpandEnvStringsW(pszPath):
    _PathUnExpandEnvStringsW = windll.shlwapi.PathUnExpandEnvStringsW
    _PathUnExpandEnvStringsW.argtypes = [LPWSTR, LPWSTR]
    _PathUnExpandEnvStringsW.restype  = bool
    _PathUnExpandEnvStringsW.errcheck = RaiseIfZero

    cchBuf = MAX_PATH
    pszBuf = ctypes.create_unicode_buffer(u"", cchBuf)
    _PathUnExpandEnvStringsW(pszPath, pszBuf, cchBuf)
    return pszBuf.value

PathUnExpandEnvStrings = GuessStringType(PathUnExpandEnvStringsA, PathUnExpandEnvStringsW)

#==============================================================================
# This calculates the list of exported symbols.
_all = set(vars().keys()).difference(_all)
__all__ = [_x for _x in _all if not _x.startswith('_')]
__all__.sort()
#==============================================================================
