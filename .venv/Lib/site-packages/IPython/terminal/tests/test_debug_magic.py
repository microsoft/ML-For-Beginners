"""Test embedding of IPython"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2013 The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import os
import sys
from IPython.testing.decorators import skip_win32
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

@skip_win32
def test_debug_magic_passes_through_generators():
    """
    This test that we can correctly pass through frames of a generator post-mortem.
    """
    import pexpect
    import re
    in_prompt = re.compile(br'In ?\[\d+\]:')
    ipdb_prompt = 'ipdb>'
    env = os.environ.copy()
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor', '--simple-prompt'],
                          env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE

    child.expect(in_prompt)

    child.timeout = 2 * IPYTHON_TESTING_TIMEOUT_SCALE

    child.sendline("def f(x):")
    child.sendline("    raise Exception")
    child.sendline("")

    child.expect(in_prompt)
    child.sendline("gen = (f(x) for x in [0])")
    child.sendline("")

    child.expect(in_prompt)
    child.sendline("for x in gen:")
    child.sendline("    pass")
    child.sendline("")

    child.timeout = 10 * IPYTHON_TESTING_TIMEOUT_SCALE

    child.expect('Exception:')

    child.expect(in_prompt)
    child.sendline(r'%debug')
    child.expect('----> 2     raise Exception')

    child.expect(ipdb_prompt)
    child.sendline('u')
    child.expect_exact(r'----> 1 gen = (f(x) for x in [0])')

    child.expect(ipdb_prompt)
    child.sendline('u')
    child.expect_exact('----> 1 for x in gen:')

    child.expect(ipdb_prompt)
    child.sendline("u")
    child.expect_exact(
        "*** all frames above hidden, use `skip_hidden False` to get get into those."
    )

    child.expect(ipdb_prompt)
    child.sendline('exit')

    child.expect(in_prompt)
    child.sendline('exit')

    child.close()
