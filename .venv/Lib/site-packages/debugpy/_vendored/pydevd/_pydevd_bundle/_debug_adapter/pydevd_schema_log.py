import os
import traceback
from _pydevd_bundle.pydevd_constants import ForkSafeLock

_pid = os.getpid()
_pid_msg = '%s: ' % (_pid,)

_debug_lock = ForkSafeLock()

DEBUG = False
DEBUG_FILE = os.path.join(os.path.dirname(__file__), '__debug_output__.txt')


def debug(msg):
    if DEBUG:
        with _debug_lock:
            _pid_prefix = _pid_msg
            if isinstance(msg, bytes):
                _pid_prefix = _pid_prefix.encode('utf-8')

                if not msg.endswith(b'\r') and not msg.endswith(b'\n'):
                    msg += b'\n'
                mode = 'a+b'
            else:
                if not msg.endswith('\r') and not msg.endswith('\n'):
                    msg += '\n'
                mode = 'a+'
            with open(DEBUG_FILE, mode) as stream:
                stream.write(_pid_prefix)
                stream.write(msg)


def debug_exception(msg=None):
    if DEBUG:
        if msg:
            debug(msg)

        with _debug_lock:

            with open(DEBUG_FILE, 'a+') as stream:
                _pid_prefix = _pid_msg
                if isinstance(msg, bytes):
                    _pid_prefix = _pid_prefix.encode('utf-8')
                stream.write(_pid_prefix)

                traceback.print_exc(file=stream)
