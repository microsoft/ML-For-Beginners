"""Test eventloop integration"""

import asyncio
import os
import sys
import threading
import time

import pytest
import tornado

from ipykernel.eventloops import (
    enable_gui,
    loop_asyncio,
    loop_cocoa,
    loop_tk,
)

from .utils import execute, flush_channels, start_new_kernel

KC = KM = None

qt_guis_avail = []

gui_to_module = {'qt6': 'PySide6', 'qt5': 'PyQt5'}


def _get_qt_vers():
    """If any version of Qt is available, this will populate `guis_avail` with 'qt' and 'qtx'. Due
    to the import mechanism, we can't import multiple versions of Qt in one session."""
    for gui in ['qt6', 'qt5']:
        print(f'Trying {gui}')
        try:
            __import__(gui_to_module[gui])
            qt_guis_avail.append(gui)
            if 'QT_API' in os.environ:
                del os.environ['QT_API']
        except ImportError:
            pass  # that version of Qt isn't available.


_get_qt_vers()


def setup():
    """start the global kernel (if it isn't running) and return its client"""
    global KM, KC
    KM, KC = start_new_kernel()
    flush_channels(KC)


def teardown():
    assert KM is not None
    assert KC is not None
    KC.stop_channels()
    KM.shutdown_kernel(now=True)


async_code = """
from ipykernel.tests._asyncio_utils import async_func
async_func()
"""


@pytest.mark.skipif(tornado.version_info < (5,), reason="only relevant on tornado 5")
def test_asyncio_interrupt():
    assert KM is not None
    assert KC is not None
    flush_channels(KC)
    msg_id, content = execute("%gui asyncio", KC)
    assert content["status"] == "ok", content

    flush_channels(KC)
    msg_id, content = execute(async_code, KC)
    assert content["status"] == "ok", content

    KM.interrupt_kernel()

    flush_channels(KC)
    msg_id, content = execute(async_code, KC)
    assert content["status"] == "ok"


windows_skip = pytest.mark.skipif(os.name == "nt", reason="causing failures on windows")


@windows_skip
@pytest.mark.skipif(sys.platform == "darwin", reason="hangs on macos")
def test_tk_loop(kernel):
    def do_thing():
        time.sleep(1)
        try:
            kernel.app_wrapper.app.quit()
        # guard for tk failing to start (if there is no display)
        except AttributeError:
            pass

    t = threading.Thread(target=do_thing)
    t.start()
    # guard for tk failing to start (if there is no display)
    try:
        loop_tk(kernel)
    except Exception:
        pass
    t.join()


@windows_skip
def test_asyncio_loop(kernel):
    def do_thing():
        loop.call_soon(loop.stop)

    loop = asyncio.get_event_loop()
    loop.call_soon(do_thing)
    loop_asyncio(kernel)


@windows_skip
def test_enable_gui(kernel):
    enable_gui("tk", kernel)


@pytest.mark.skipif(sys.platform != "darwin", reason="MacOS-only")
def test_cocoa_loop(kernel):
    loop_cocoa(kernel)


@pytest.mark.skipif(
    len(qt_guis_avail) == 0, reason='No viable version of PyQt or PySide installed.'
)
def test_qt_enable_gui(kernel, capsys):
    gui = qt_guis_avail[0]

    enable_gui(gui, kernel)

    # We store the `QApplication` instance in the kernel.
    assert hasattr(kernel, 'app')

    # And the `QEventLoop` is added to `app`:`
    assert hasattr(kernel.app, 'qt_event_loop')

    # Don't create another app even if `gui` is the same.
    app = kernel.app
    enable_gui(gui, kernel)
    assert app == kernel.app

    # Event loop intergration can be turned off.
    enable_gui(None, kernel)
    assert not hasattr(kernel, 'app')

    # But now we're stuck with this version of Qt for good; can't switch.
    for not_gui in ['qt6', 'qt5']:
        if not_gui not in qt_guis_avail:
            break

    enable_gui(not_gui, kernel)
    captured = capsys.readouterr()
    assert captured.out == f'Cannot switch Qt versions for this session; you must use {gui}.\n'

    # Check 'qt' gui, which means "the best available"
    enable_gui(None, kernel)
    enable_gui('qt', kernel)
    assert gui_to_module[gui] in str(kernel.app)
