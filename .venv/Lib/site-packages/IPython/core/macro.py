"""Support for interactive macros in IPython"""

#*****************************************************************************
#       Copyright (C) 2001-2005 Fernando Perez <fperez@colorado.edu>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#*****************************************************************************

import re

from IPython.utils.encoding import DEFAULT_ENCODING

coding_declaration = re.compile(r"#\s*coding[:=]\s*([-\w.]+)")

class Macro(object):
    """Simple class to store the value of macros as strings.

    Macro is just a callable that executes a string of IPython
    input when called.
    """

    def __init__(self,code):
        """store the macro value, as a single string which can be executed"""
        lines = []
        enc = None
        for line in code.splitlines():
            coding_match = coding_declaration.match(line)
            if coding_match:
                enc = coding_match.group(1)
            else:
                lines.append(line)
        code = "\n".join(lines)
        if isinstance(code, bytes):
            code = code.decode(enc or DEFAULT_ENCODING)
        self.value = code + '\n'
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return 'IPython.macro.Macro(%s)' % repr(self.value)
    
    def __getstate__(self):
        """ needed for safe pickling via %store """
        return {'value': self.value}
    
    def __add__(self, other):
        if isinstance(other, Macro):
            return Macro(self.value + other.value)
        elif isinstance(other, str):
            return Macro(self.value + other)
        raise TypeError
