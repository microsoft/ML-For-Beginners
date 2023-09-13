"""Python version-independent methods for C/Python buffers.

This file was copied and adapted from mpi4py.

Authors
-------
* MinRK
"""

#-----------------------------------------------------------------------------
#  Copyright (c) 2010 Lisandro Dalcin
#  All rights reserved.
#  Used under BSD License: http://www.opensource.org/licenses/bsd-license.php
#
#  Retrieval:
#  Jul 23, 2010 18:00 PST (r539)
#  http://code.google.com/p/mpi4py/source/browse/trunk/src/MPI/asbuffer.pxi
#
#  Modifications from original:
#  Copyright (c) 2010-2012 Brian Granger, Min Ragan-Kelley
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file LICENSE.BSD, distributed as part of this software.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Python includes.
#-----------------------------------------------------------------------------

# get version-independent aliases:
cdef extern from "pyversion_compat.h":
    pass

# Python 3 buffer interface (PEP 3118)
cdef extern from "Python.h":
    int PY_MAJOR_VERSION
    int PY_MINOR_VERSION
    ctypedef int Py_ssize_t
    ctypedef struct PyMemoryViewObject:
        pass
    ctypedef struct Py_buffer:
        void *buf
        Py_ssize_t len
        int readonly
        char *format
        int ndim
        Py_ssize_t *shape
        Py_ssize_t *strides
        Py_ssize_t *suboffsets
        Py_ssize_t itemsize
        void *internal
    cdef enum:
        PyBUF_SIMPLE
        PyBUF_WRITABLE
        PyBUF_FORMAT
        PyBUF_ANY_CONTIGUOUS
    int  PyObject_CheckBuffer(object)
    int  PyObject_GetBuffer(object, Py_buffer *, int) except -1
    void PyBuffer_Release(Py_buffer *)
    
    int PyBuffer_FillInfo(Py_buffer *view, object obj, void *buf,
                Py_ssize_t len, int readonly, int infoflags) except -1
    object PyMemoryView_FromBuffer(Py_buffer *info)
    
    object PyMemoryView_FromObject(object)


#-----------------------------------------------------------------------------
# asbuffer: C buffer from python object
#-----------------------------------------------------------------------------


cdef inline int check_buffer(object ob):
    """Version independent check for whether an object is a buffer.
    
    Parameters
    ----------
    object : object
        Any Python object

    Returns
    -------
    int : 0 if no buffer interface, 3 if newstyle buffer interface, 2 if oldstyle (removed).
    """
    if PyObject_CheckBuffer(ob):
        return 3
    return 0


cdef inline object asbuffer(object ob, int writable, int format,
                            void **base, Py_ssize_t *size,
                            Py_ssize_t *itemsize):
    """Turn an object into a C buffer in a Python version-independent way.
    
    Parameters
    ----------
    ob : object
        The object to be turned into a buffer.
        Must provide a Python Buffer interface
    writable : int
        Whether the resulting buffer should be allowed to write
        to the object.
    format : int
        The format of the buffer.  See Python buffer docs.
    base : void **
        The pointer that will be used to store the resulting C buffer.
    size : Py_ssize_t *
        The size of the buffer(s).
    itemsize : Py_ssize_t *
        The size of an item, if the buffer is non-contiguous.
    
    Returns
    -------
    An object describing the buffer format. Generally a str, such as 'B'.
    """

    cdef void *bptr = NULL
    cdef Py_ssize_t blen = 0, bitemlen = 0
    cdef Py_buffer view
    cdef int flags = PyBUF_SIMPLE
    cdef int mode = 0
    
    bfmt = None

    mode = check_buffer(ob)
    if mode == 0:
        raise TypeError("%r does not provide a buffer interface."%ob)

    if mode == 3:
        flags = PyBUF_ANY_CONTIGUOUS
        if writable:
            flags |= PyBUF_WRITABLE
        if format:
            flags |= PyBUF_FORMAT
        PyObject_GetBuffer(ob, &view, flags)
        bptr = view.buf
        blen = view.len
        if format:
            if view.format != NULL:
                bfmt = view.format
                bitemlen = view.itemsize
        PyBuffer_Release(&view)

    if base: base[0] = <void *>bptr
    if size: size[0] = <Py_ssize_t>blen
    if itemsize: itemsize[0] = <Py_ssize_t>bitemlen
    
    if bfmt is not None:
        return bfmt.decode('ascii')
    return bfmt


cdef inline object asbuffer_r(object ob, void **base, Py_ssize_t *size):
    """Wrapper for standard calls to asbuffer with a readonly buffer."""
    asbuffer(ob, 0, 0, base, size, NULL)
    return ob


cdef inline object asbuffer_w(object ob, void **base, Py_ssize_t *size):
    """Wrapper for standard calls to asbuffer with a writable buffer."""
    asbuffer(ob, 1, 0, base, size, NULL)
    return ob

#------------------------------------------------------------------------------
# frombuffer: python buffer/view from C buffer
#------------------------------------------------------------------------------


cdef inline object frombuffer(void *ptr, Py_ssize_t s, int readonly):
    """Create a Python Memory View of a C array.
    
    Parameters
    ----------
    ptr : void *
        Pointer to the array to be copied.
    s : size_t
        Length of the buffer.
    readonly : int
        whether the resulting object should be allowed to write to the buffer.
    
    Returns
    -------
    Python Memory View of the C buffer.
    """
    cdef Py_buffer pybuf
    cdef Py_ssize_t *shape = [s]
    cdef str astr=""
    PyBuffer_FillInfo(&pybuf, astr, ptr, s, readonly, PyBUF_SIMPLE)
    pybuf.format = "B"
    pybuf.shape = shape
    pybuf.ndim = 1
    return PyMemoryView_FromBuffer(&pybuf)


cdef inline object frombuffer_r(void *ptr, Py_ssize_t s):
    """Wrapper for readonly view frombuffer."""
    return frombuffer(ptr, s, 1)


cdef inline object frombuffer_w(void *ptr, Py_ssize_t s):
    """Wrapper for writable view frombuffer."""
    return frombuffer(ptr, s, 0)

#------------------------------------------------------------------------------
# viewfromobject: python buffer/view from python object, refcounts intact
# frombuffer(asbuffer(obj)) would lose track of refs
#------------------------------------------------------------------------------

cdef inline object viewfromobject(object obj, int readonly):
    """Construct a Python Memory View object from another Python object.

    Parameters
    ----------
    obj : object
        The input object to be cast as a buffer
    readonly : int
        Whether the result should be prevented from overwriting the original.

    Returns
    -------
    MemoryView of the original object.
    """
    return PyMemoryView_FromObject(obj)


cdef inline object viewfromobject_r(object obj):
    """Wrapper for readonly viewfromobject."""
    return viewfromobject(obj, 1)


cdef inline object viewfromobject_w(object obj):
    """Wrapper for writable viewfromobject."""
    return viewfromobject(obj, 0)
