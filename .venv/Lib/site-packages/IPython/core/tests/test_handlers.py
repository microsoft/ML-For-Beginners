"""Tests for input handlers.
"""
#-----------------------------------------------------------------------------
# Module imports
#-----------------------------------------------------------------------------

# our own packages
from IPython.core import autocall
from IPython.testing import tools as tt

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# Get the public instance of IPython

failures = []
num_tests = 0

#-----------------------------------------------------------------------------
# Test functions
#-----------------------------------------------------------------------------

class CallableIndexable(object):
    def __getitem__(self, idx): return True
    def __call__(self, *args, **kws): return True


class Autocallable(autocall.IPyAutocall):
    def __call__(self):
        return "called"


def run(tests):
    """Loop through a list of (pre, post) inputs, where pre is the string
    handed to ipython, and post is how that string looks after it's been
    transformed (i.e. ipython's notion of _i)"""
    tt.check_pairs(ip.prefilter_manager.prefilter_lines, tests)


def test_handlers():
    call_idx = CallableIndexable()
    ip.user_ns['call_idx'] = call_idx

    # For many of the below, we're also checking that leading whitespace
    # turns off the esc char, which it should unless there is a continuation
    # line.
    run(
        [('"no change"', '"no change"'),             # normal
         (u"lsmagic",     "get_ipython().run_line_magic('lsmagic', '')"),   # magic
         #("a = b # PYTHON-MODE", '_i'),          # emacs -- avoids _in cache
         ])

    # Objects which are instances of IPyAutocall are *always* autocalled
    autocallable = Autocallable()
    ip.user_ns['autocallable'] = autocallable

    # auto
    ip.run_line_magic("autocall", "0")
    # Only explicit escapes or instances of IPyAutocallable should get
    # expanded
    run(
        [
            ('len "abc"', 'len "abc"'),
            ("autocallable", "autocallable()"),
            # Don't add extra brackets (gh-1117)
            ("autocallable()", "autocallable()"),
        ]
    )
    ip.run_line_magic("autocall", "1")
    run(
        [
            ('len "abc"', 'len("abc")'),
            ('len "abc";', 'len("abc");'),  # ; is special -- moves out of parens
            # Autocall is turned off if first arg is [] and the object
            # is both callable and indexable.  Like so:
            ("len [1,2]", "len([1,2])"),  # len doesn't support __getitem__...
            ("call_idx [1]", "call_idx [1]"),  # call_idx *does*..
            ("call_idx 1", "call_idx(1)"),
            ("len", "len"),  # only at 2 does it auto-call on single args
        ]
    )
    ip.run_line_magic("autocall", "2")
    run(
        [
            ('len "abc"', 'len("abc")'),
            ('len "abc";', 'len("abc");'),
            ("len [1,2]", "len([1,2])"),
            ("call_idx [1]", "call_idx [1]"),
            ("call_idx 1", "call_idx(1)"),
            # This is what's different:
            ("len", "len()"),  # only at 2 does it auto-call on single args
        ]
    )
    ip.run_line_magic("autocall", "1")

    assert failures == []
