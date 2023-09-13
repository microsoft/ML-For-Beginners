# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import unittest

from flaky import flaky

from ipykernel.inprocess.manager import InProcessKernelManager

# -----------------------------------------------------------------------------
# Test case
# -----------------------------------------------------------------------------


class InProcessKernelManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.km = InProcessKernelManager()

    def tearDown(self):
        if self.km.has_kernel:
            self.km.shutdown_kernel()

    @flaky
    def test_interface(self):
        """Does the in-process kernel manager implement the basic KM interface?"""
        km = self.km
        assert not km.has_kernel

        km.start_kernel()
        assert km.has_kernel
        assert km.kernel is not None

        kc = km.client()
        assert not kc.channels_running

        kc.start_channels()
        assert kc.channels_running

        old_kernel = km.kernel
        km.restart_kernel()
        self.assertIsNotNone(km.kernel)
        assert km.kernel != old_kernel

        km.shutdown_kernel()
        assert not km.has_kernel

        self.assertRaises(NotImplementedError, km.interrupt_kernel)
        self.assertRaises(NotImplementedError, km.signal_kernel, 9)

        kc.stop_channels()
        assert not kc.channels_running

    def test_execute(self):
        """Does executing code in an in-process kernel work?"""
        km = self.km
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready()
        kc.execute("foo = 1")
        assert km.kernel.shell.user_ns["foo"] == 1

    def test_complete(self):
        """Does requesting completion from an in-process kernel work?"""
        km = self.km
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready()
        km.kernel.shell.push({"my_bar": 0, "my_baz": 1})
        kc.complete("my_ba", 5)
        msg = kc.get_shell_msg()
        assert msg["header"]["msg_type"] == "complete_reply"
        self.assertEqual(sorted(msg["content"]["matches"]), ["my_bar", "my_baz"])

    def test_inspect(self):
        """Does requesting object information from an in-process kernel work?"""
        km = self.km
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready()
        km.kernel.shell.user_ns["foo"] = 1
        kc.inspect("foo")
        msg = kc.get_shell_msg()
        assert msg["header"]["msg_type"] == "inspect_reply"
        content = msg["content"]
        assert content["found"]
        text = content["data"]["text/plain"]
        self.assertIn("int", text)

    def test_history(self):
        """Does requesting history from an in-process kernel work?"""
        km = self.km
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready()
        kc.execute("1")
        kc.history(hist_access_type="tail", n=1)
        msg = kc.shell_channel.get_msgs()[-1]
        assert msg["header"]["msg_type"] == "history_reply"
        history = msg["content"]["history"]
        assert len(history) == 1
        assert history[0][2] == "1"


if __name__ == "__main__":
    unittest.main()
