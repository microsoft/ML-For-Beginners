"""String dispatch class to match regexps and dispatch commands.
"""

# Stdlib imports
import re

# Our own modules
from IPython.core.hooks import CommandChainDispatcher

# Code begins
class StrDispatch(object):
    """Dispatch (lookup) a set of strings / regexps for match.

    Example:

    >>> dis = StrDispatch()
    >>> dis.add_s('hei',34, priority = 4)
    >>> dis.add_s('hei',123, priority = 2)
    >>> dis.add_re('h.i', 686)
    >>> print(list(dis.flat_matches('hei')))
    [123, 34, 686]
    """

    def __init__(self):
        self.strs = {}
        self.regexs = {}

    def add_s(self, s, obj, priority= 0 ):
        """ Adds a target 'string' for dispatching """

        chain = self.strs.get(s, CommandChainDispatcher())
        chain.add(obj,priority)
        self.strs[s] = chain

    def add_re(self, regex, obj, priority= 0 ):
        """ Adds a target regexp for dispatching """

        chain = self.regexs.get(regex, CommandChainDispatcher())
        chain.add(obj,priority)
        self.regexs[regex] = chain

    def dispatch(self, key):
        """ Get a seq of Commandchain objects that match key """
        if key in self.strs:
            yield self.strs[key]

        for r, obj in self.regexs.items():
            if re.match(r, key):
                yield obj
            else:
                #print "nomatch",key  # dbg
                pass

    def __repr__(self):
        return "<Strdispatch %s, %s>" % (self.strs, self.regexs)

    def s_matches(self, key):
        if key not in self.strs:
             return
        for el in self.strs[key]:
            yield el[1]

    def flat_matches(self, key):
        """ Yield all 'value' targets, without priority """
        for val in self.dispatch(key):
            for el in val:
                yield el[1] # only value, no priority
        return
