import os
import sys
import time
port = int(sys.argv[1])
root_dirname = os.path.dirname(os.path.dirname(__file__))

if root_dirname not in sys.path:
    sys.path.append(root_dirname)

import pydevd
try:
    pydevd._wait_for_attach()  # Cannot be called before _enable_attach.
except AssertionError:
    pass
else:
    raise AssertionError('Expected _wait_for_attach to raise exception.')

if '--use-dap-mode' in sys.argv:
    pydevd.config('http_json', 'debugpy-dap')

assert sys.gettrace() is None
print('enable attach to port: %s' % (port,))
pydevd._enable_attach(('127.0.0.1', port))
pydevd._enable_attach(('127.0.0.1', port))  # no-op in practice

try:
    pydevd._enable_attach(('127.0.0.1', port + 15))  # different port: raise error.
except AssertionError:
    pass
else:
    raise AssertionError('Expected _enable_attach to raise exception (because it is already hearing in another port).')

assert pydevd.get_global_debugger() is not None
assert sys.gettrace() is not None

a = 10  # Break 1
print('wait for attach')
pydevd._wait_for_attach()
print('finished wait for attach')
pydevd._wait_for_attach()  # Should promptly return (already connected).

a = 20  # Break 2

pydevd._wait_for_attach()  # As we disconnected on the 2nd break, this one should wait until a new configurationDone.

a = 20  # Break 3

while a == 20:  # Pause 1
    # The debugger should disconnect/reconnect, pause and then change 'a' to another value.
    time.sleep(1 / 20.)  # Pause 2

print('TEST SUCEEDED!')
