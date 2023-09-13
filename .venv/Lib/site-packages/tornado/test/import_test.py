# flake8: noqa
import subprocess
import sys
import unittest

_import_everything = b"""
# The event loop is not fork-safe, and it's easy to initialize an asyncio.Future
# at startup, which in turn creates the default event loop and prevents forking.
# Explicitly disallow the default event loop so that an error will be raised
# if something tries to touch it.
import asyncio
asyncio.set_event_loop(None)

import importlib
import tornado

for mod in tornado.__all__:
    if mod == "curl_httpclient":
        # This module has extra dependencies; skip it if they're not installed.
        try:
            import pycurl
        except ImportError:
            continue
    importlib.import_module(f"tornado.{mod}")
"""

_import_lazy = b"""
import sys
import tornado

if "tornado.web" in sys.modules:
    raise Exception("unexpected eager import")

# Trigger a lazy import by referring to something in a submodule.
tornado.web.RequestHandler

if "tornado.web" not in sys.modules:
    raise Exception("lazy import did not update sys.modules")
"""


class ImportTest(unittest.TestCase):
    def test_import_everything(self):
        # Test that all Tornado modules can be imported without side effects,
        # specifically without initializing the default asyncio event loop.
        # Since we can't tell which modules may have already beein imported
        # in our process, do it in a subprocess for a clean slate.
        proc = subprocess.Popen([sys.executable], stdin=subprocess.PIPE)
        proc.communicate(_import_everything)
        self.assertEqual(proc.returncode, 0)

    def test_lazy_import(self):
        # Test that submodules can be referenced lazily after "import tornado"
        proc = subprocess.Popen([sys.executable], stdin=subprocess.PIPE)
        proc.communicate(_import_lazy)
        self.assertEqual(proc.returncode, 0)

    def test_import_aliases(self):
        # Ensure we don't delete formerly-documented aliases accidentally.
        import tornado
        import asyncio

        self.assertIs(tornado.ioloop.TimeoutError, tornado.util.TimeoutError)
        self.assertIs(tornado.gen.TimeoutError, tornado.util.TimeoutError)
        self.assertIs(tornado.util.TimeoutError, asyncio.TimeoutError)
