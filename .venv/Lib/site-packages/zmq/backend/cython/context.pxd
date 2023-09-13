"""0MQ Context class declaration."""

#
#    Copyright (c) 2010-2011 Brian E. Granger & Min Ragan-Kelley
#
#    This file is part of pyzmq.
#
#    pyzmq is free software; you can redistribute it and/or modify it under
#    the terms of the Lesser GNU General Public License as published by
#    the Free Software Foundation; either version 3 of the License, or
#    (at your option) any later version.
#
#    pyzmq is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    Lesser GNU General Public License for more details.
#
#    You should have received a copy of the Lesser GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

cdef class Context:

    cdef object __weakref__     # enable weakref
    cdef void *handle           # The C handle for the underlying zmq object.
    cdef bint _shadow           # whether the Context is a shadow wrapper of another
    cdef int _pid               # the pid of the process which created me (for fork safety)

    cdef public bint closed   # bool property for a closed context.
    cdef inline int _term(self)
