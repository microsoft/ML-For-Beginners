import os
import sys
port = int(sys.argv[1])
root_dirname = os.path.dirname(os.path.dirname(__file__))

if root_dirname not in sys.path:
    sys.path.append(root_dirname)

import pydevd

# Ensure that pydevd uses JSON protocol
from _pydevd_bundle import pydevd_constants
from _pydevd_bundle import pydevd_defaults
pydevd_defaults.PydevdCustomization.DEFAULT_PROTOCOL = pydevd_constants.HTTP_JSON_PROTOCOL

# Enable some defaults related to debugpy such as sending a single notification when
# threads pause and stopping on any exception.
pydevd_defaults.PydevdCustomization.DEBUG_MODE = 'debugpy-dap'

import tempfile
with tempfile.TemporaryDirectory('w') as tempdir:
    with open(os.path.join(tempdir, 'my_custom_module.py'), 'w') as stream:
        stream.write("print('Loaded my_custom_module')")

    pydevd_defaults.PydevdCustomization.PREIMPORT = '%s;my_custom_module' % (tempdir,)
    assert 'my_custom_module' not in sys.modules

    assert sys.gettrace() is None
    print('enable attach to port: %s' % (port,))
    pydevd._enable_attach(('127.0.0.1', port))

    assert pydevd.get_global_debugger() is not None
    # Set as a part of debugpy-dap
    assert pydevd.get_global_debugger().multi_threads_single_notification
    assert sys.gettrace() is not None

    assert 'my_custom_module' in sys.modules

    a = 10  # Break 1
    print('wait for attach')
    pydevd._wait_for_attach()

    a = 20  # Break 2

    print('TEST SUCEEDED!')
