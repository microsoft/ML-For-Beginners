""" Tests for zmq shell / display publisher. """

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import unittest
import warnings
from queue import Queue
from threading import Thread
from unittest.mock import MagicMock

import pytest
import zmq
from jupyter_client.session import Session
from traitlets import Int

from ipykernel.zmqshell import (  # type:ignore
    InteractiveShell,
    KernelMagics,
    ZMQDisplayPublisher,
    ZMQInteractiveShell,
)


class NoReturnDisplayHook:
    """
    A dummy DisplayHook which allows us to monitor
    the number of times an object is called, but which
    does *not* return a message when it is called.
    """

    call_count = 0

    def __call__(self, obj):
        self.call_count += 1


class ReturnDisplayHook(NoReturnDisplayHook):
    """
    A dummy DisplayHook with the same counting ability
    as its base class, but which also returns the same
    message when it is called.
    """

    def __call__(self, obj):
        super().__call__(obj)
        return obj


class CounterSession(Session):
    """
    This is a simple subclass to allow us to count
    the calls made to the session object by the display
    publisher.
    """

    send_count = Int(0)

    def send(self, *args, **kwargs):
        """
        A trivial override to just augment the existing call
        with an increment to the send counter.
        """
        self.send_count += 1
        super().send(*args, **kwargs)


class ZMQDisplayPublisherTests(unittest.TestCase):
    """
    Tests the ZMQDisplayPublisher in zmqshell.py
    """

    def setUp(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.session = CounterSession()

        self.disp_pub = ZMQDisplayPublisher(session=self.session, pub_socket=self.socket)

    def tearDown(self):
        """
        We need to close the socket in order to proceed with the
        tests.
        TODO - There is still an open file handler to '/dev/null',
        presumably created by zmq.
        """
        self.disp_pub.clear_output()
        self.socket.close()
        self.context.term()

    def test_display_publisher_creation(self):
        """
        Since there's no explicit constructor, here we confirm
        that keyword args get assigned correctly, and override
        the defaults.
        """
        assert self.disp_pub.session == self.session
        assert self.disp_pub.pub_socket == self.socket

    def test_thread_local_hooks(self):
        """
        Confirms that the thread_local attribute is correctly
        initialised with an empty list for the display hooks
        """
        assert self.disp_pub._hooks == []

        def hook(msg):
            return msg

        self.disp_pub.register_hook(hook)
        assert self.disp_pub._hooks == [hook]

        q: Queue = Queue()

        def set_thread_hooks():
            q.put(self.disp_pub._hooks)

        t = Thread(target=set_thread_hooks)
        t.start()
        thread_hooks = q.get(timeout=10)
        assert thread_hooks == []

    def test_publish(self):
        """
        Publish should prepare the message and eventually call
        `send` by default.
        """
        data = dict(a=1)
        assert self.session.send_count == 0
        self.disp_pub.publish(data)
        assert self.session.send_count == 1

    def test_display_hook_halts_send(self):
        """
        If a hook is installed, and on calling the object
        it does *not* return a message, then we assume that
        the message has been consumed, and should not be
        processed (`sent`) in the normal manner.
        """
        data = dict(a=1)
        hook = NoReturnDisplayHook()

        self.disp_pub.register_hook(hook)
        assert hook.call_count == 0
        assert self.session.send_count == 0

        self.disp_pub.publish(data)

        assert hook.call_count == 1
        assert self.session.send_count == 0

    def test_display_hook_return_calls_send(self):
        """
        If a hook is installed and on calling the object
        it returns a new message, then we assume that this
        is just a message transformation, and the message
        should be sent in the usual manner.
        """
        data = dict(a=1)
        hook = ReturnDisplayHook()

        self.disp_pub.register_hook(hook)
        assert hook.call_count == 0
        assert self.session.send_count == 0

        self.disp_pub.publish(data)

        assert hook.call_count == 1
        assert self.session.send_count == 1

    def test_unregister_hook(self):
        """
        Once a hook is unregistered, it should not be called
        during `publish`.
        """
        data = dict(a=1)
        hook = NoReturnDisplayHook()

        self.disp_pub.register_hook(hook)
        assert hook.call_count == 0
        assert self.session.send_count == 0

        self.disp_pub.publish(data)

        assert hook.call_count == 1
        assert self.session.send_count == 0

        #
        # After unregistering the `NoReturn` hook, any calls
        # to publish should *not* got through the DisplayHook,
        # but should instead hit the usual `session.send` call
        # at the end.
        #
        # As a result, the hook call count should *not* increase,
        # but the session send count *should* increase.
        #
        first = self.disp_pub.unregister_hook(hook)
        self.disp_pub.publish(data)

        self.assertTrue(first)
        assert hook.call_count == 1
        assert self.session.send_count == 1

        #
        # If a hook is not installed, `unregister_hook`
        # should return false.
        #
        second = self.disp_pub.unregister_hook(hook)
        self.assertFalse(second)


def test_magics(tmp_path):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    shell = InteractiveShell()
    shell.user_ns["hi"] = 1
    magics = KernelMagics(shell)

    tmp_file = tmp_path / "test.txt"
    tmp_file.write_text("hi", "utf8")
    magics.edit(str(tmp_file))
    magics.clear([])
    magics.less(str(tmp_file))
    if os.name == "posix":
        magics.man("ls")
    magics.autosave("10")

    socket.close()
    context.destroy()


def test_zmq_interactive_shell(kernel):
    shell = ZMQInteractiveShell()

    with pytest.raises(RuntimeError):
        shell.enable_gui("tk")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shell.data_pub_class = MagicMock()  # type:ignore
        shell.data_pub
    shell.kernel = kernel
    shell.set_next_input("hi")
    assert shell.get_parent() is None
    if os.name == "posix":
        shell.system_piped("ls")
    else:
        shell.system_piped("dir")
    shell.ask_exit()


if __name__ == "__main__":
    unittest.main()
