import os
import sys
import warnings
from unittest import mock

import pytest

from ipykernel.parentpoller import ParentPollerUnix, ParentPollerWindows


@pytest.mark.skipif(os.name == "nt", reason="only works on posix")
def test_parent_poller_unix():
    poller = ParentPollerUnix()
    with mock.patch("os.getppid", lambda: 1):

        def exit_mock(*args):
            sys.exit(1)

        with mock.patch("os._exit", exit_mock), pytest.raises(SystemExit):
            poller.run()

    def mock_getppid():
        raise ValueError("hi")

    with mock.patch("os.getppid", mock_getppid), pytest.raises(ValueError):
        poller.run()


@pytest.mark.skipif(os.name != "nt", reason="only works on windows")
def test_parent_poller_windows():
    poller = ParentPollerWindows(interrupt_handle=1)

    def mock_wait(*args, **kwargs):
        return -1

    with mock.patch("ctypes.windll.kernel32.WaitForMultipleObjects", mock_wait):  # noqa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poller.run()
