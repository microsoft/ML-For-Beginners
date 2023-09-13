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
import subprocess
import sys

from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.testing.decorators import skip_win32
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------


_sample_embed = b"""
import IPython

a = 3
b = 14
print(a, '.', b)

IPython.embed()

print('bye!')
"""

_exit = b"exit\r"

def test_ipython_embed():
    """test that `IPython.embed()` works"""
    with NamedFileInTemporaryDirectory('file_with_embed.py') as f:
        f.write(_sample_embed)
        f.flush()
        f.close() # otherwise msft won't be able to read the file

        # run `python file_with_embed.py`
        cmd = [sys.executable, f.name]
        env = os.environ.copy()
        env['IPY_TEST_SIMPLE_PROMPT'] = '1'

        p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(_exit)
        std = out.decode('UTF-8')

        assert p.returncode == 0
        assert "3 . 14" in std
        if os.name != "nt":
            # TODO: Fix up our different stdout references, see issue gh-14
            assert "IPython" in std
        assert "bye!" in std


@skip_win32
def test_nest_embed():
    """test that `IPython.embed()` is nestable"""
    import pexpect
    ipy_prompt = r']:' #ansi color codes give problems matching beyond this
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'


    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'],
                          env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect(ipy_prompt)
    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.sendline("import IPython")
    child.expect(ipy_prompt)
    child.sendline("ip0 = get_ipython()")
    #enter first nested embed
    child.sendline("IPython.embed()")
    #skip the banner until we get to a prompt
    try:
        prompted = -1
        while prompted != 0:
            prompted = child.expect([ipy_prompt, '\r\n'])
    except pexpect.TIMEOUT as e:
        print(e)
        #child.interact()
    child.sendline("embed1 = get_ipython()")
    child.expect(ipy_prompt)
    child.sendline("print('true' if embed1 is not ip0 else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    child.sendline("print('true' if IPython.get_ipython() is embed1 else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    #enter second nested embed
    child.sendline("IPython.embed()")
    #skip the banner until we get to a prompt
    try:
        prompted = -1
        while prompted != 0:
            prompted = child.expect([ipy_prompt, '\r\n'])
    except pexpect.TIMEOUT as e:
        print(e)
        #child.interact()
    child.sendline("embed2 = get_ipython()")
    child.expect(ipy_prompt)
    child.sendline("print('true' if embed2 is not embed1 else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    child.sendline("print('true' if embed2 is IPython.get_ipython() else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    child.sendline('exit')
    #back at first embed
    child.expect(ipy_prompt)
    child.sendline("print('true' if get_ipython() is embed1 else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    child.sendline("print('true' if IPython.get_ipython() is embed1 else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    child.sendline('exit')
    #back at launching scope
    child.expect(ipy_prompt)
    child.sendline("print('true' if get_ipython() is ip0 else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    child.sendline("print('true' if IPython.get_ipython() is ip0 else 'false')")
    assert(child.expect(['true\r\n', 'false\r\n']) == 0)
    child.expect(ipy_prompt)
    child.sendline('exit')
    child.close()
