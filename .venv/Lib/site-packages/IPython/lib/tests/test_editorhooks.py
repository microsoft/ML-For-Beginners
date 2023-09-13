"""Test installing editor hooks"""
import sys
from unittest import mock

from IPython import get_ipython
from IPython.lib import editorhooks

def test_install_editor():
    called = []
    def fake_popen(*args, **kwargs):
        called.append({
            'args': args,
            'kwargs': kwargs,
        })
        return mock.MagicMock(**{'wait.return_value': 0})
    editorhooks.install_editor('foo -l {line} -f {filename}', wait=False)
    
    with mock.patch('subprocess.Popen', fake_popen):
        get_ipython().hooks.editor('the file', 64)
    
    assert len(called) == 1
    args = called[0]["args"]
    kwargs = called[0]["kwargs"]

    assert kwargs == {"shell": True}

    if sys.platform.startswith("win"):
        expected = ["foo", "-l", "64", "-f", "the file"]
    else:
        expected = "foo -l 64 -f 'the file'"
    cmd = args[0]
    assert cmd == expected
