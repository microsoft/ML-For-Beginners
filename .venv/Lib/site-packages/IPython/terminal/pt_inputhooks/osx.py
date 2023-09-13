"""Inputhook for OS X

Calls NSApp / CoreFoundation APIs via ctypes.
"""

# obj-c boilerplate from appnope, used under BSD 2-clause

import ctypes
import ctypes.util
from threading import Event

objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))  # type: ignore

void_p = ctypes.c_void_p

objc.objc_getClass.restype = void_p
objc.sel_registerName.restype = void_p
objc.objc_msgSend.restype = void_p
objc.objc_msgSend.argtypes = [void_p, void_p]

msg = objc.objc_msgSend

def _utf8(s):
    """ensure utf8 bytes"""
    if not isinstance(s, bytes):
        s = s.encode('utf8')
    return s

def n(name):
    """create a selector name (for ObjC methods)"""
    return objc.sel_registerName(_utf8(name))

def C(classname):
    """get an ObjC Class by name"""
    return objc.objc_getClass(_utf8(classname))

# end obj-c boilerplate from appnope

# CoreFoundation C-API calls we will use:
CoreFoundation = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))  # type: ignore

CFFileDescriptorCreate = CoreFoundation.CFFileDescriptorCreate
CFFileDescriptorCreate.restype = void_p
CFFileDescriptorCreate.argtypes = [void_p, ctypes.c_int, ctypes.c_bool, void_p, void_p]

CFFileDescriptorGetNativeDescriptor = CoreFoundation.CFFileDescriptorGetNativeDescriptor
CFFileDescriptorGetNativeDescriptor.restype = ctypes.c_int
CFFileDescriptorGetNativeDescriptor.argtypes = [void_p]

CFFileDescriptorEnableCallBacks = CoreFoundation.CFFileDescriptorEnableCallBacks
CFFileDescriptorEnableCallBacks.restype = None
CFFileDescriptorEnableCallBacks.argtypes = [void_p, ctypes.c_ulong]

CFFileDescriptorCreateRunLoopSource = CoreFoundation.CFFileDescriptorCreateRunLoopSource
CFFileDescriptorCreateRunLoopSource.restype = void_p
CFFileDescriptorCreateRunLoopSource.argtypes = [void_p, void_p, void_p]

CFRunLoopGetCurrent = CoreFoundation.CFRunLoopGetCurrent
CFRunLoopGetCurrent.restype = void_p

CFRunLoopAddSource = CoreFoundation.CFRunLoopAddSource
CFRunLoopAddSource.restype = None
CFRunLoopAddSource.argtypes = [void_p, void_p, void_p]

CFRelease = CoreFoundation.CFRelease
CFRelease.restype = None
CFRelease.argtypes = [void_p]

CFFileDescriptorInvalidate = CoreFoundation.CFFileDescriptorInvalidate
CFFileDescriptorInvalidate.restype = None
CFFileDescriptorInvalidate.argtypes = [void_p]

# From CFFileDescriptor.h
kCFFileDescriptorReadCallBack = 1
kCFRunLoopCommonModes = void_p.in_dll(CoreFoundation, 'kCFRunLoopCommonModes')


def _NSApp():
    """Return the global NSApplication instance (NSApp)"""
    objc.objc_msgSend.argtypes = [void_p, void_p]
    return msg(C('NSApplication'), n('sharedApplication'))


def _wake(NSApp):
    """Wake the Application"""
    objc.objc_msgSend.argtypes = [
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
    ]
    event = msg(
        C("NSEvent"),
        n(
            "otherEventWithType:location:modifierFlags:"
            "timestamp:windowNumber:context:subtype:data1:data2:"
        ),
        15,  # Type
        0,  # location
        0,  # flags
        0,  # timestamp
        0,  # window
        None,  # context
        0,  # subtype
        0,  # data1
        0,  # data2
    )
    objc.objc_msgSend.argtypes = [void_p, void_p, void_p, void_p]
    msg(NSApp, n('postEvent:atStart:'), void_p(event), True)


def _input_callback(fdref, flags, info):
    """Callback to fire when there's input to be read"""
    CFFileDescriptorInvalidate(fdref)
    CFRelease(fdref)
    NSApp = _NSApp()
    objc.objc_msgSend.argtypes = [void_p, void_p, void_p]
    msg(NSApp, n('stop:'), NSApp)
    _wake(NSApp)

_c_callback_func_type = ctypes.CFUNCTYPE(None, void_p, void_p, void_p)
_c_input_callback = _c_callback_func_type(_input_callback)


def _stop_on_read(fd):
    """Register callback to stop eventloop when there's data on fd"""
    fdref = CFFileDescriptorCreate(None, fd, False, _c_input_callback, None)
    CFFileDescriptorEnableCallBacks(fdref, kCFFileDescriptorReadCallBack)
    source = CFFileDescriptorCreateRunLoopSource(None, fdref, 0)
    loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(loop, source, kCFRunLoopCommonModes)
    CFRelease(source)


def inputhook(context):
    """Inputhook for Cocoa (NSApp)"""
    NSApp = _NSApp()
    _stop_on_read(context.fileno())
    objc.objc_msgSend.argtypes = [void_p, void_p]
    msg(NSApp, n('run'))
