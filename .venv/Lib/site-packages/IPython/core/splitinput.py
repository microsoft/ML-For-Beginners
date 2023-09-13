# encoding: utf-8
"""
Simple utility for splitting user input. This is used by both inputsplitter and
prefilter.

Authors:

* Brian Granger
* Fernando Perez
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import re
import sys

from IPython.utils import py3compat
from IPython.utils.encoding import get_stream_enc
from IPython.core.oinspect import OInfo

#-----------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------

# RegExp for splitting line contents into pre-char//first word-method//rest.
# For clarity, each group in on one line.

# WARNING: update the regexp if the escapes in interactiveshell are changed, as
# they are hardwired in.

# Although it's not solely driven by the regex, note that:
# ,;/% only trigger if they are the first character on the line
# ! and !! trigger if they are first char(s) *or* follow an indent
# ? triggers as first or last char.

line_split = re.compile(r"""
             ^(\s*)               # any leading space
             ([,;/%]|!!?|\?\??)?  # escape character or characters
             \s*(%{0,2}[\w\.\*]*)     # function/method, possibly with leading %
                                  # to correctly treat things like '?%magic'
             (.*?$|$)             # rest of line
             """, re.VERBOSE)


def split_user_input(line, pattern=None):
    """Split user input into initial whitespace, escape character, function part
    and the rest.
    """
    # We need to ensure that the rest of this routine deals only with unicode
    encoding = get_stream_enc(sys.stdin, 'utf-8')
    line = py3compat.cast_unicode(line, encoding)

    if pattern is None:
        pattern = line_split
    match = pattern.match(line)
    if not match:
        # print "match failed for line '%s'" % line
        try:
            ifun, the_rest = line.split(None,1)
        except ValueError:
            # print "split failed for line '%s'" % line
            ifun, the_rest = line, u''
        pre = re.match(r'^(\s*)(.*)',line).groups()[0]
        esc = ""
    else:
        pre, esc, ifun, the_rest = match.groups()

    #print 'line:<%s>' % line # dbg
    #print 'pre <%s> ifun <%s> rest <%s>' % (pre,ifun.strip(),the_rest) # dbg
    return pre, esc or '', ifun.strip(), the_rest.lstrip()


class LineInfo(object):
    """A single line of input and associated info.

    Includes the following as properties:

    line
      The original, raw line

    continue_prompt
      Is this line a continuation in a sequence of multiline input?

    pre
      Any leading whitespace.

    esc
      The escape character(s) in pre or the empty string if there isn't one.
      Note that '!!' and '??' are possible values for esc. Otherwise it will
      always be a single character.

    ifun
      The 'function part', which is basically the maximal initial sequence
      of valid python identifiers and the '.' character. This is what is
      checked for alias and magic transformations, used for auto-calling,
      etc. In contrast to Python identifiers, it may start with "%" and contain
      "*".

    the_rest
      Everything else on the line.
    """
    def __init__(self, line, continue_prompt=False):
        self.line            = line
        self.continue_prompt = continue_prompt
        self.pre, self.esc, self.ifun, self.the_rest = split_user_input(line)

        self.pre_char       = self.pre.strip()
        if self.pre_char:
            self.pre_whitespace = '' # No whitespace allowed before esc chars
        else:
            self.pre_whitespace = self.pre

    def ofind(self, ip) -> OInfo:
        """Do a full, attribute-walking lookup of the ifun in the various
        namespaces for the given IPython InteractiveShell instance.

        Return a dict with keys: {found, obj, ospace, ismagic}

        Note: can cause state changes because of calling getattr, but should
        only be run if autocall is on and if the line hasn't matched any
        other, less dangerous handlers.

        Does cache the results of the call, so can be called multiple times
        without worrying about *further* damaging state.
        """
        return ip._ofind(self.ifun)

    def __str__(self):
        return "LineInfo [%s|%s|%s|%s]" %(self.pre, self.esc, self.ifun, self.the_rest)
