from _pydevd_bundle.pydevd_extension_api import DebuggerEventHandler
import os
import sys


class VerifyEvent(object):
    def on_debugger_modules_loaded(self, **kwargs):
        print ("INITIALIZE EVENT RECEIVED")
        # check that some core modules are loaded before this callback is invoked
        modules_loaded = all(mod in sys.modules for mod in ('pydevd_file_utils', '_pydevd_bundle.pydevd_constants'))
        if modules_loaded:
            print ("TEST SUCEEDED")  # incorrect spelling on purpose
        else:
            print ("TEST FAILED")


if os.environ.get("VERIFY_EVENT_TEST"):
    DebuggerEventHandler.register(VerifyEvent)
