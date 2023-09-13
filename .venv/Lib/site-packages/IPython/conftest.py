import builtins
import inspect
import os
import pathlib
import shutil
import sys
import types

import pytest

# Must register before it gets imported
pytest.register_assert_rewrite("IPython.testing.tools")

from .testing import tools


def pytest_collection_modifyitems(items):
    """This function is automatically run by pytest passing all collected test
    functions.

    We use it to add asyncio marker to all async tests and assert we don't use
    test functions that are async generators which wouldn't make sense.
    """
    for item in items:
        if inspect.iscoroutinefunction(item.obj):
            item.add_marker("asyncio")
        assert not inspect.isasyncgenfunction(item.obj)


def get_ipython():
    from .terminal.interactiveshell import TerminalInteractiveShell
    if TerminalInteractiveShell._instance:
        return TerminalInteractiveShell.instance()

    config = tools.default_config()
    config.TerminalInteractiveShell.simple_prompt = True

    # Create and initialize our test-friendly IPython instance.
    shell = TerminalInteractiveShell.instance(config=config)
    return shell


@pytest.fixture(scope='session', autouse=True)
def work_path():
    path = pathlib.Path("./tmp-ipython-pytest-profiledir")
    os.environ["IPYTHONDIR"] = str(path.absolute())
    if path.exists():
        raise ValueError('IPython dir temporary path already exists ! Did previous test run exit successfully ?')
    path.mkdir()
    yield
    shutil.rmtree(str(path.resolve()))


def nopage(strng, start=0, screen_lines=0, pager_cmd=None):
    if isinstance(strng, dict):
        strng = strng.get("text/plain", "")
    print(strng)


def xsys(self, cmd):
    """Replace the default system call with a capturing one for doctest.
    """
    # We use getoutput, but we need to strip it because pexpect captures
    # the trailing newline differently from commands.getoutput
    print(self.getoutput(cmd, split=False, depth=1).rstrip(), end="", file=sys.stdout)
    sys.stdout.flush()


# for things to work correctly we would need this as a session fixture;
# unfortunately this will fail on some test that get executed as _collection_
# time (before the fixture run), in particular parametrized test that contain
# yields. so for now execute at import time.
#@pytest.fixture(autouse=True, scope='session')
def inject():

    builtins.get_ipython = get_ipython
    builtins._ip = get_ipython()
    builtins.ip = get_ipython()
    builtins.ip.system = types.MethodType(xsys, ip)
    builtins.ip.builtin_trap.activate()
    from .core import page

    page.pager_page = nopage
    # yield


inject()
